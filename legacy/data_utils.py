import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import h5py
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import map_coordinates
from concurrent.futures import ThreadPoolExecutor
from filelock import FileLock
import matplotlib.pyplot as plt

def compute_normalization_params(dataset):
    """
    Iterates over the dataset and computes min/max normalization parameters for:
      - Element properties: ['cp_metal', 'h_vapor', 'kappa_metal', 'laser_reflectivity',
                              'mass_density_metal', 't_boil', 'tcrit']
      - Laser parameters: ['rspot', 'laser_power_wcm']
    
    Assumes each target in the dataset is a dictionary containing these keys.
    
    Returns:
        property_norm_params (dict): Keys are element property names and values are (min, max) tuples.
        laser_norm_params (dict): Keys are laser parameter names and values are (min, max) tuples.
    """
    property_keys = ['cp_metal', 'h_vapor', 'kappa_metal', 'laser_reflectivity',
                     'mass_density_metal', 't_boil', 'tcrit']
    laser_keys = ['rspot', 'laser_power_wcm']
    
    # Initialize dictionaries to hold all values.
    property_values = {key: [] for key in property_keys}
    laser_values = {key: [] for key in laser_keys}
    
    for _, target in dataset:
        # Accumulate element property values.
        for key in property_keys:
            if key in target:
                property_values[key].append(target[key])
            else:
                raise ValueError(f"Target missing key: {key}")
        # Accumulate laser parameter values.
        for key in laser_keys:
            if key in target:
                laser_values[key].append(target[key])
            else:
                raise ValueError(f"Target missing key: {key}")
    
    # Compute min and max for each element property.
    property_norm_params = {}
    for key, values in property_values.items():
        property_norm_params[key] = (min(values), max(values))
    
    # Compute min and max for each laser parameter.
    laser_norm_params = {}
    for key, values in laser_values.items():
        laser_norm_params[key] = (min(values), max(values))
    
    return property_norm_params, laser_norm_params

class LogTransform:
    def __init__(self, eps=1e-6):
        self.eps = eps
    def __call__(self, video):
        return torch.log(video + self.eps)
    
class StandardizeVideo:
    def __init__(self, global_mean, global_std, eps=1e-6, apply_log=False,target_size=(96, 96)):
        """
        Args:
            global_mean (float): Global mean pixel value.
            global_std (float): Global standard deviation of pixel values.
            eps (float): Small constant to prevent division by zero.
            apply_log (bool): If True, apply a logarithmic transform before standardization.
        """
        self.global_mean = global_mean
        self.global_std = global_std
        self.eps = eps
        self.apply_log = apply_log
        self.target_size = target_size

    def __call__(self, video):
        """
        Standardizes a video tensor using global statistics.
        Optionally applies a logarithmic transformation before standardization.
        
        Args:
            video (torch.Tensor): Input tensor of shape (C, T, H, W)
        
        Returns:
            torch.Tensor: Standardized video tensor.
        """
        video = video.float()
        if self.apply_log:
            video = torch.log(video + self.eps)
        # Standardize: (video - global_mean) / global_std
        video = (video - self.global_mean) / (self.global_std + self.eps)
        
        # Downsample spatially:
        # Current shape: (C, T, H, W). We need to operate on the H and W dims.
        # Permute to (T, C, H, W) so that interpolate (which expects (N, C, H, W)) can work.
        video = video.permute(1, 0, 2, 3)
        video = F.interpolate(video, size=self.target_size, mode='bilinear', align_corners=False)
        # Permute back to (C, T, H, W)
        video = video.permute(1, 0, 2, 3)
        
        return video
    
def compute_global_video_stats(dataset, transform=None):
    """
    Iterates over the dataset and computes the global mean and standard deviation 
    of video pixel values. Optionally applies a transform (e.g. log transform) 
    to each video before computing the statistics.
    
    Args:
        dataset: A dataset that returns video tensors of shape (C, T, H, W)
        transform: An optional transform to apply to each video before computing stats.
        
    Returns:
        global_mean, global_std: Global mean and standard deviation of pixel values.
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for video, _ in dataset:
        video = video.float()
        if transform is not None:
            video = transform(video)
        # Flatten the video tensor to compute overall statistics.
        video_flat = video.view(-1)
        total_sum += video_flat.sum().item()
        total_sq_sum += (video_flat ** 2).sum().item()
        total_count += video_flat.numel()

    global_mean = total_sum / total_count
    global_std = (total_sq_sum / total_count - global_mean**2)**0.5
    return global_mean, global_std

class ElementClassifyTransform:
    def __init__(self, classes=None, laser_norm_params=None):
        """
        Args:
            classes (list, optional): List of element symbols. Default is ['Co', 'Cu', 'W', 'Fe', 'Pt', 'Zn'].
            laser_norm_params (dict, optional): Dictionary with keys 'rspot' and 'laser_power_wcm', 
                where each maps to a (min, max) tuple for min–max normalization.
                For example: {'rspot': (0.00025, 0.002), 'laser_power_wcm': (80000000.0, 360000000.0)}
        """
        if classes is None:
            self.classes = ['Co', 'Cu', 'W', 'Fe', 'Pt', 'Zn']
        else:
            self.classes = classes
        self.class_to_idx = {elem: idx for idx, elem in enumerate(self.classes)}
        
        self.laser_norm_params = laser_norm_params
    
    def __call__(self, target):
        """
        Expects target to be a dictionary containing at least:
          - 'element': a string or list with the element name,
          - 'rspot': the raw value of rspot,
          - 'laser_power_wcm': the raw value of the laser power.
          
        Returns:
          A tuple (one_hot, laser_parameters) where:
            one_hot: one-hot encoded vector for the element.
            laser_parameters: a tensor with normalized [rspot, laser_power_wcm].
        """
        target = target.copy()
        
        # Process the element.
        element = target.get('element')
        if isinstance(element, list):
            element = element[0]
        
        num_classes = len(self.classes)
        one_hot = torch.zeros(num_classes, dtype=torch.float)
        idx = self.class_to_idx.get(element)
        if idx is None:
            raise ValueError(f"Element '{element}' not found in class list {self.classes}.")
        one_hot[idx] = 1.0
        
        # Get the raw laser parameter values.
        rspot = target.get('rspot')
        laser_power = target.get('laser_power_wcm')
        
        # If normalization parameters are provided, normalize each value using min-max normalization.
        if self.laser_norm_params is not None:
            # Normalize rspot.
            rspot_min, rspot_max = self.laser_norm_params.get('rspot', (None, None))
            if rspot_min is not None and rspot_max is not None:
                rspot_norm = (rspot - rspot_min) / (rspot_max - rspot_min)
            else:
                rspot_norm = rspot
            
            # Normalize laser_power.
            lp_min, lp_max = self.laser_norm_params.get('laser_power_wcm', (None, None))
            if lp_min is not None and lp_max is not None:
                laser_power_norm = (laser_power - lp_min) / (lp_max - lp_min)
            else:
                laser_power_norm = laser_power
        else:
            rspot_norm = rspot
            laser_power_norm = laser_power
        
        laser_parameters = torch.tensor([rspot_norm, laser_power_norm], dtype=torch.float)
        return one_hot, laser_parameters

class ElementPropertyTransform:
    def __init__(self, 
                 property_norm_params=None,  # dict mapping each property to (min, max)
                 laser_norm_params=None      # dict mapping 'rspot' and 'laser_power_wcm' to (min, max)
                ):
        """
        Args:
            property_norm_params (dict, optional): A dictionary where each key is one of
                ['cp_metal', 'h_vapor', 'kappa_metal', 'laser_reflectivity',
                 'mass_density_metal', 't_boil', 'tcrit']
                and each value is a tuple (min, max) for min–max normalization.
                If None, raw property values are returned.
            laser_norm_params (dict, optional): A dictionary for laser parameters, with keys
                'rspot' and 'laser_power_wcm' mapping to a (min, max) tuple.
                If None, raw values are returned.
        """
        self.properties = ['cp_metal', 'h_vapor', 'kappa_metal', 
                           'laser_reflectivity', 'mass_density_metal', 't_boil', 'tcrit']
        self.property_norm_params = property_norm_params

        self.laser_keys = ['rspot', 'laser_power_wcm']
        self.laser_norm_params = laser_norm_params

    def __call__(self, target):
        """
        Expects target to be a dictionary containing at least the following keys:
          - For regression: 'cp_metal', 'h_vapor', 'kappa_metal', 'laser_reflectivity',
            'mass_density_metal', 't_boil', 'tcrit'
          - For laser parameters: 'rspot', 'laser_power_wcm'
        
        Returns:
          A tuple (property_tensor, laser_tensor) where:
            property_tensor is a torch.Tensor containing the (optionally normalized)
              regression targets for the 7 properties.
            laser_tensor is a torch.Tensor containing the (optionally normalized)
              laser parameters (2 values).
        """
        target = target.copy()
        
        # Process the element properties:
        prop_values = []
        for prop in self.properties:
            if prop not in target:
                raise ValueError(f"Property '{prop}' not found in target.")
            value = target[prop]
            # Normalize if parameters provided.
            if self.property_norm_params is not None and prop in self.property_norm_params:
                pmin, pmax = self.property_norm_params[prop]
                if pmax != pmin:
                    value = (value - pmin) / (pmax - pmin)
                else:
                    value = value - pmin
            prop_values.append(value)
        property_tensor = torch.tensor(prop_values, dtype=torch.float)
        
        # Process the laser parameters:
        laser_values = []
        for key in self.laser_keys:
            if key not in target:
                raise ValueError(f"Laser parameter '{key}' not found in target.")
            value = target[key]
            if self.laser_norm_params is not None and key in self.laser_norm_params:
                lmin, lmax = self.laser_norm_params[key]
                if lmax != lmin:
                    value = (value - lmin) / (lmax - lmin)
                else:
                    value = value - lmin
            laser_values.append(value)
        laser_tensor = torch.tensor(laser_values, dtype=torch.float)
        
        return property_tensor, laser_tensor

class NormalizeVideo:
    def __init__(self, eps=1e-6):
        """
        Args:
            eps (float): Small constant to avoid log(0).
        """
        self.eps = eps

    def __call__(self, video):
        """
        Normalize a video tensor (C, T, H, W) using a logarithmic transformation
        to handle exponential intensity decay and then scaling the result to [0, 1].

        Args:
            video (torch.Tensor): Input tensor of shape (C, T, H, W).

        Returns:
            torch.Tensor: Normalized video tensor with values in [0, 1].
        """
        video = video.float()
        # Apply logarithmic transformation to compress dynamic range.
        video = torch.log(video + self.eps)
        v_min = video.min()
        v_max = video.max()
        # Normalize to [0, 1]
        if v_max > v_min:
            video = (video - v_min) / (v_max - v_min)
        else:
            video = video - v_min
            
        return video
    
def filter_dataset_by_frames(dataset, min_frames=10):
    valid_indices = []
    for i in range(len(dataset)):
        try:
            video, _ = dataset[i]
            # Check if the video tensor contains any inf values.
            if torch.isinf(video).any():
                print(f"Skipping index {i} because video contains inf values.")
                continue
            
            # Determine the frame count depending on tensor dimensions.
            if video.ndim == 4:  # (C, T, H, W)
                frame_count = video.shape[1]
            elif video.ndim == 3:  # (T, H, W)
                frame_count = video.shape[0]
            else:
                print(f"Skipping index {i} due to unexpected tensor shape: {video.shape}")
                continue
            
            # Skip if the tensor is empty or has fewer than min_frames.
            if frame_count >= min_frames:
                valid_indices.append(i)
            else:
                print(f"Skipping index {i} because frame count {frame_count} is less than {min_frames}.")
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")
    print(f"Found {len(valid_indices)} valid samples out of {len(dataset)}")
    return Subset(dataset, valid_indices)

class PlasmaSimDataset(Dataset):
    def __init__(self, hdf5_paths, return_raw_data=False, domain_bounds=0.1,
                 resolution=64, num_workers=4, cache_dir=None,
                 data_transform=None, target_transform=None):
        self.hdf5_paths = hdf5_paths  # List of file paths
        self.index = []               # List of (file_index, dataset_key)
        self.return_raw_data = return_raw_data
        self.domain_bounds = domain_bounds
        self.resolution = resolution
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.data_transform = data_transform
        self.target_transform = target_transform

        # Build the index mapping from each file and dataset key.
        for file_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    self.index.append((file_idx, key))

        # Precompute the resampling grid (common to all timesteps)
        self.Z, self.R, self.grid_coords = self._precompute_grid(domain_bounds, resolution)

        # Placeholder for file handles, one per file; these will be opened lazily.
        self.files = [None] * len(hdf5_paths)
        
    def _extract_numeric_part(self, name):
        try:
            return float(name.split('_')[1])
        except Exception:
            return 0.0
    
    def _precompute_grid(self, domain_bounds, resolution):
        z_resample = np.linspace(0, domain_bounds, resolution)
        r_resample = np.linspace(0, domain_bounds, resolution)
        Z, R = np.meshgrid(z_resample, r_resample)
        grid_coords = np.column_stack((Z.ravel(), R.ravel()))
        return Z, R, grid_coords
    
    def _resample_timestep(self, timestep_dataset, Z, R, grid_coords):
        z = timestep_dataset[:, 0]
        r = timestep_dataset[:, 1]
        rho = timestep_dataset[:, 2]
        #ux = timestep_dataset[:, 5]
        ne = timestep_dataset[:, 10]
        T  = timestep_dataset[:, 4]
        alpha_PI = timestep_dataset[:,19]
        alpha_Ben = timestep_dataset[:,17]
        alpha_Bei = timestep_dataset[:,18]
        params = [rho, T, ne, alpha_PI, alpha_Ben, alpha_Bei]
        mask = (z >= 0) & (z <= self.domain_bounds) & (r >= 0) & (r <= self.domain_bounds)
        if not mask.any():
            raise ValueError("No points found in the specified domain.")
        z_filtered = z[mask]
        r_filtered = r[mask]
        num_channels = len(params)
        resampled = np.empty((self.resolution, self.resolution, num_channels))
        
        def process_channel(param):
            param_filtered = param[mask]
            interp_vals = griddata((z_filtered, r_filtered), param_filtered, grid_coords, method='linear')
            interp_vals = interp_vals.reshape(self.resolution, self.resolution)
            if np.isnan(interp_vals).any():
                valid_mask = ~np.isnan(interp_vals)
                valid_coords = np.column_stack((Z[valid_mask], R[valid_mask]))
                valid_vals = interp_vals[valid_mask]
                nan_mask = np.isnan(interp_vals)
                nan_coords = np.column_stack((Z[nan_mask], R[nan_mask]))
                interp_vals[nan_mask] = griddata(valid_coords, valid_vals, nan_coords, method='nearest')
            return interp_vals

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_channel, params))
        for i, channel_interp in enumerate(results):
            resampled[..., i] = channel_interp
        return resampled
        
    def get_resampled_timeseries(self, sim_data):
        filtered_keys = [key for key in sim_data.keys() if key.startswith('res')]
        filtered_keys = sorted(filtered_keys, key=self._extract_numeric_part)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._resample_timestep, sim_data[i][:], self.Z, self.R, self.grid_coords)
                for i in filtered_keys
            ]
            resampled_list = [future.result() for future in futures]
        timestep_data = np.stack(resampled_list, axis=0)
        return torch.from_numpy(timestep_data[1:,:,:,:])
    
    def simulate_image(self, data):
        # Get dimensions from the data tensor.
        nt, nr, nz, nc = data.numpy().shape
        n_theta = 256  # Number of angular divisions

        # Convert data to float32 and get plasma data in cylindrical form.
        plasma_2D = data.numpy().astype(np.float32)  # shape: (nt, nr, nz, nc)
        t_vals = np.linspace(0, 50, nt, dtype=np.float32)
        r_vals = np.linspace(0, self.domain_bounds, nr, dtype=np.float32)
        z_vals = np.linspace(0, self.domain_bounds, nz, dtype=np.float32)
        theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False, dtype=np.float32)

        # Expand the axisymmetric plasma data into 3D cylindrical coordinates.
        # plasma_3D shape: (nt, nr, nz, n_theta, nc)
        plasma_3D = np.repeat(plasma_2D[:, :, :, np.newaxis, :], n_theta, axis=3)

        # Define Cartesian coordinate ranges.
        x_vals = np.linspace(-self.domain_bounds, self.domain_bounds, nr, dtype=np.float32)
        y_vals = np.linspace(-self.domain_bounds, self.domain_bounds, nr, dtype=np.float32)
        # Use same z_vals as above.
        X_grid, Y_grid, Z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        nx, ny, nz_new = X_grid.shape

        # Compute cylindrical coordinates from the Cartesian grid.
        R_grid = np.sqrt(X_grid**2 + Y_grid**2)
        Theta_grid = (np.arctan2(Y_grid, X_grid) + 2*np.pi) % (2*np.pi)

        # Preallocate array for integrated emission.
        # We'll integrate along Y (line-of-sight), resulting in an output of shape (nt, nc, nx, nz_new)
        integrated_emission = np.zeros((nt, nx, nz_new), dtype=np.float32)

        def process_timestep(i):
            # Preallocate an array for the Cartesian data for all channels at time step i.
            data_cart_channels = np.zeros((nc, nx, ny, nz_new), dtype=np.float32)
            for ch in range(nc):
                # Extract cylindrical data for time step i and channel ch.
                # data_cyl shape: (nr, nz, n_theta)
                data_cyl = plasma_3D[i, :, :, :, ch]
                # Convert Cartesian physical coordinates to index space of cylindrical grid.
                r_idx = (R_grid - r_vals[0]) / (r_vals[-1] - r_vals[0]) * (nr - 1)
                z_idx = (Z_grid - z_vals[0]) / (z_vals[-1] - z_vals[0]) * (nz - 1)
                theta_idx = Theta_grid / (2*np.pi) * n_theta
                coords = np.array([r_idx.ravel(), z_idx.ravel(), theta_idx.ravel()])
                interp_vals = map_coordinates(data_cyl, coords, order=1, mode='nearest')
                data_cart = interp_vals.reshape(X_grid.shape)
                data_cart_channels[ch] = data_cart
            # Calculate plasma emission intensity using all channels.
            # For example, we assume:
            #   Free-free emission: I_ff ~ ne * T^4  (channels: 2 = ne, 1 = T)
            #   Bound-free emission: I_bf ~ alpha_PI * BB(T)  (channels: 3 = alpha_PI, 1 = T)
            I_emission = self.calculate_emission(data_cart_channels)
            # Integrate I_emission along Y (axis=1) to obtain a 2D (X, Z) map.
            integrated_emission[i,:,:] = np.trapz(I_emission, y_vals, axis=1)

        # Process time steps concurrently.
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            executor.map(process_timestep, range(nt))

        return integrated_emission
    
    def calculate_emission(self, cartesian_data):
        # Assuming channels: 0=rho, 1=T, 2=ne, 3=alpha_PI, 4=alpha_Ben, 5= alpha_Bei
        # Free-free approximate emission: I_ff ~ ne * T^4.
        #I_ff = self._I_ff_approx(cartesian_data)
        
        # Bound-free approximate emission: I_bf ~ alpha_PI * Blackbody(T).
        #I_bf = cartesian_data[3,:,:,:] * self._BB_approx(cartesian_data[1,:,:,:], lam=555)
        #I_BB = self._BB_approx(cartesian_data[1,:,:,:], lam=555)
        # I = (alpha_Ben + alpha_Bei) * BB *exp(-alpha_Ben + alpha_Bei)
        coefs = cartesian_data[4,:,:,:]+cartesian_data[5,:,:,:]
        I = coefs*self._BB_approx(cartesian_data[1,:,:,:])*(np.exp(coefs))
        
        return I   # Adjust if you want to include bound-free contribution.
    
    def _I_ff_approx(self, cartesian_data):
        # Assuming channels: 0=rho, 1=T, 2=ne, 3=alpha_PI, 4=alpha_Ben, 5= alpha_Bei
        # A helper version for free-free emission alone.
        I_ff = cartesian_data[2,:,:,:] * (cartesian_data[1,:,:,:] ** 4)
        return I_ff
    
    def _BB_approx(self, T, lam_min=300, lam_max=800, num_wavelengths=10):
        """
        Calculate the integrated photon flux over a wavelength range using blackbody radiation.

        Parameters:
            T (np.ndarray): Temperature array (e.g., shape (256, 256, 256)).
            lam_min (float): Minimum wavelength in nm.
            lam_max (float): Maximum wavelength in nm.
            num_wavelengths (int): Number of wavelengths to evaluate between lam_min and lam_max.

        Returns:
            np.ndarray: Integrated photon flux with the same shape as T.
                       Units will be photons/(m²·sr·s).
        """
        h = 6.626e-34  # Planck's constant (J s)
        c = 3.0e8      # Speed of light (m/s)
        k = 1.38e-23   # Boltzmann's constant (J/K)

        # Create an array of wavelengths (in nm) and convert to meters.
        wavelengths = np.linspace(lam_min, lam_max, num_wavelengths)
        wavelengths_m = wavelengths * 1e-9  # convert nm to m

        # Reshape wavelengths_m for broadcasting with T.
        # New shape: (num_wavelengths, 1, 1, 1) for a 3D T array.
        wavelengths_m = wavelengths_m.reshape((num_wavelengths,) + (1,) * T.ndim)

        # Calculate energy spectral radiance using Planck's law:
        # B(λ,T) = (2hc²) / (λ⁵ (exp(hc/(λkT)) - 1))
        numerator = 2 * h * c**2
        denominator = (wavelengths_m ** 5) * (np.exp(h * c / (wavelengths_m * k * T)) - 1)
        spectral_radiance = numerator / denominator  # Units: W/(m²·sr·m)

        # Convert to photon spectral radiance:
        # Energy per photon: E = h*c/λ. Dividing by this gives:
        # photon_spectral_radiance with units: photons/(m²·sr·s·m)
        energy_per_photon = h * c / wavelengths_m
        photon_spectral_radiance = spectral_radiance / energy_per_photon

        # Integrate over wavelength (in meters) to obtain the total photon flux.
        # The result has units: photons/(m²·sr·s)
        photon_flux = np.trapz(photon_spectral_radiance, wavelengths_m, axis=0)

        return photon_flux
        
    def get_conditions(self, sim_data):
        return dict(sim_data.attrs.items())
        
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # If caching is enabled, check for the preprocessed file.
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"item_{idx}.pt")
            lock_path = cache_path + ".lock"
            with FileLock(lock_path):
                if os.path.exists(cache_path):
                    sample = torch.load(cache_path)
                    data = torch.tensor(sample['data'])
                    # Check frame count depending on dimensions.
                    if data.ndim == 4:  # (C, T, H, W)
                        frame_count = data.shape[1]
                    elif data.ndim == 3:  # (T, H, W)
                        frame_count = data.shape[0]
                    else:
                        frame_count = 0
                    if frame_count < 10:
                        raise IndexError(f"Video at index {idx} (from cache) has less than 10 frames: {frame_count}")
                    # If 3D, add a channel dimension.
                    if data.ndim == 3:
                        data = data.unsqueeze(0)
                    # Retain only the first 10 frames.
                    data = data[:, 0:10, :, :]

                    if self.data_transform is not None:
                        data = self.data_transform(data)
                    if self.target_transform is not None:
                        targets = self.target_transform(sample['targets'])
                    else:
                        targets = sample['targets']
                    return data, targets

        # Load from the source file if not cached.
        file_idx, group_key = self.index[idx]
        if self.files[file_idx] is None:
            self.files[file_idx] = h5py.File(self.hdf5_paths[file_idx], 'r')

        data_raw = self.get_resampled_timeseries(self.files[file_idx][group_key])

        if self.return_raw_data:
            data = data_raw
        else:
            data = self.simulate_image(data_raw)

        targets = self.get_conditions(self.files[file_idx][group_key])

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            with FileLock(lock_path):
                if not os.path.exists(cache_path):
                    torch.save({'data': data, 'targets': targets}, cache_path)

        # Check and ensure there are at least 10 frames.
        if data.ndim == 4:  # (C, T, H, W)
            frame_count = data.shape[1]
        elif data.ndim == 3:  # (T, H, W)
            frame_count = data.shape[0]
        else:
            frame_count = 0

        if frame_count < 10:
            raise IndexError(f"Video at index {idx} has less than 10 frames: {frame_count}")

        # If data is 3D, add a channel dimension.
        if data.ndim == 3:
            data = data.unsqueeze(0)

        # Retain only the first 10 frames.
        data = data[:, 0:10, :, :]

        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return data, targets