import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import r2_score

from tqdm import tqdm

class SpatioTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A spatiotemporal block that first applies a spatial-only conv,
        then a temporal-only conv.
        """
        super(SpatioTemporalBlock, self).__init__()
        # Spatial convolution: kernel_size=(1,3,3), no temporal mixing.
        self.spatial_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.GELU()
        
        # Temporal convolution: kernel_size=(3,1,1), no spatial mixing.
        self.temporal_conv = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0)
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.GELU()
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = self.spatial_conv(x)    # (B, out_channels, T, H, W) with spatial mixing
        x = self.bn1(x)
        x = self.act1(x)
        x = self.temporal_conv(x)   # (B, out_channels, T, H, W) with temporal mixing
        x = self.bn2(x)
        x = self.act2(x)
        return x
    
class SelfAttentionConv2p1D(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size_spatial=(1, 3, 3), kernel_size_temporal=(3, 1, 1),
                 stride=1, padding_spatial=(0, 1, 1), padding_temporal=(1, 0, 0),
                 attn_dim=None, downsample_factor=2):
        """
        (2+1)D convolution block with self attention.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size_spatial (tuple): Kernel size for spatial conv.
            kernel_size_temporal (tuple): Kernel size for temporal conv.
            stride (int or tuple): Stride for the convolutions.
            padding_spatial (tuple): Padding for the spatial conv.
            padding_temporal (tuple): Padding for the temporal conv.
            attn_dim (int, optional): Dimension for the query/key projections.
                                      Defaults to max(1, out_channels // 8) if None.
            downsample_factor (int, optional): Factor to downsample spatial dimensions.
                                               Defaults to 2.
        """
        super(SelfAttentionConv2p1D, self).__init__()
        
        if downsample_factor is None:
            downsample_factor = 2

        # Spatial convolution: mixes spatial dimensions only.
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, 
                                      kernel_size=kernel_size_spatial,
                                      stride=stride,
                                      padding=padding_spatial)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.GELU()
        
        # Temporal convolution: mixes temporal dimension only.
        self.temporal_conv = nn.Conv3d(out_channels, out_channels,
                                       kernel_size=kernel_size_temporal,
                                       stride=stride,
                                       padding=padding_temporal)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.GELU()
        
        # Optional pooling to reduce spatial dimensions before attention.
        self.downsample = nn.AvgPool3d(kernel_size=(1, downsample_factor, downsample_factor)) if downsample_factor > 1 else None
        
        # Set attention projection dimension.
        if attn_dim is None:
            attn_dim = max(1, out_channels // 8)
        self.attn_dim = attn_dim
        
        # 1x1x1 convolutions for query, key, and value projections.
        self.query_conv = nn.Conv3d(out_channels, attn_dim, kernel_size=1)
        self.key_conv   = nn.Conv3d(out_channels, attn_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        
        # Learnable scaling parameter for the attention output.
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor of shape (B, in_channels, T, H, W)
        Returns:
            Tensor of shape (B, out_channels, T, H, W)
        """
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.temporal_conv(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Only apply downsampling if spatial dimensions are large enough.
        if self.downsample is not None:
            _, _, _, H, W = x.shape
            kernel_h, kernel_w = self.downsample.kernel_size[1], self.downsample.kernel_size[2]
            if H >= kernel_h and W >= kernel_w:
                x = self.downsample(x)
        
        B, C, T, H, W = x.shape
        N = T * H * W  # Total number of spatiotemporal positions
        
        # Compute query, key, and value projections.
        query = self.query_conv(x).view(B, self.attn_dim, N).permute(0, 2, 1)
        key = self.key_conv(x).view(B, self.attn_dim, N)
        
        energy = torch.bmm(query, key)  # shape: (B, N, N)
        attn = F.softmax(energy, dim=-1)
        
        value = self.value_conv(x).view(B, C, N)
        out = torch.bmm(value, attn.permute(0, 2, 1))  # shape: (B, C, N)
        out = out.view(B, C, T, H, W)
        
        out = self.gamma * out + x
        return out

class CNN_Encoder(nn.Module):
    def __init__(self, input_channels, output_shape, num_stages=5, blocks_per_stage=2, start_filters=32, input_frames=10, attn_dim=None):
        """
        Args:
            input_channels (int): Number of input channels.
            output_shape (int): Dimension of the final projection.
            num_stages (int): Number of stages. Each stage stacks a number of SelfAttentionConv2p1D blocks, then downsamples spatially.
            blocks_per_stage (int): Number of SelfAttentionConv2p1D blocks to stack in each stage.
            start_filters (int): Number of filters in the first stage.
            input_frames (int): Number of input time frames.
            attn_dim (int, optional): Attention dimension for the self-attention blocks.
        """
        super(CNN_Encoder, self).__init__()
        layers = []
        in_channels = input_channels
        filters = start_filters

        # For each stage, stack blocks_per_stage self-attention blocks, then downsample spatially.
        for stage in range(num_stages):
            for block in range(blocks_per_stage):
                if block == 0:
                    # First block in the stage converts from in_channels to filters.
                    #layers.append(SelfAttentionConv2p1D(in_channels, filters, attn_dim=attn_dim))
                    layers.append(SpatioTemporalBlock(in_channels, filters))
                else:
                    #layers.append(SelfAttentionConv2p1D(filters, filters, attn_dim=attn_dim))
                    layers.append(SpatioTemporalBlock(filters, filters))
            # Downsample spatially (using a 3D conv that only downsamples H and W).
            layers.append(nn.Conv3d(filters, filters, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
            layers.append(nn.BatchNorm3d(filters))
            # Prepare for the next stage.
            in_channels = filters
            filters *= 2

        self.blocks = nn.Sequential(*layers)
        self.activation = nn.GELU()
        # Global average pooling over spatial dimensions.
        # After processing, features have shape (B, filters, T, H', W').
        # We average over H' and W', then flatten (T * filters) and project.
        self.project = nn.Linear(input_frames * in_channels, output_shape)

    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor of shape (B, C, T, H, W)
        Returns:
            Tensor of shape (B, output_shape)
        """
        x = self.blocks(x)  # shape: (B, out_channels, T, H', W')
        b, c, t, h, w = x.size()
        # Rearrange to (B, T, c, H', W') to separate temporal dimension.
        x = x.permute(0, 2, 1, 3, 4)
        # Global average pooling over spatial dimensions.
        x = torch.mean(x, dim=[3, 4])  # shape: (B, T, c)
        # Flatten temporal and channel dimensions.
        x = x.flatten(start_dim=1)    # shape: (B, T * c)
        x = self.activation(self.project(x))
        return x
    
class FeedForward(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers=None, 
                 activation=nn.ReLU, dropout=0.0):
        """
        Args:
            input_shape (int): Dimension of the input features.
            output_shape (int): Dimension of the output.
            hidden_layers (list of int, optional): List containing the number of nodes in each hidden layer.
                                                   If None, no hidden layers will be added.
            activation (nn.Module, optional): Activation function to use after each hidden layer (default: nn.ReLU).
            dropout (float, optional): Dropout probability after each activation (default: 0.0, no dropout).
        """
        super(FeedForward, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = []
        
        layers = []
        prev_dim = input_shape
        
        # Create each hidden layer
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_shape))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
class InverseSpatioTemporalBlock(nn.Module):
    """
    Mirrors the SpatioTemporalBlock by first undoing the temporal mixing 
    and then the spatial mixing using transposed convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super(InverseSpatioTemporalBlock, self).__init__()
        # Inverse temporal conv: transposed conv for kernel (3,1,1)
        self.temporal_deconv = nn.ConvTranspose3d(
            in_channels, in_channels,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0)
        )
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.act1 = nn.GELU()
        # Inverse spatial conv: transposed conv for kernel (1,3,3)
        self.spatial_deconv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.GELU()
    
    def forward(self, x):
        x = self.temporal_deconv(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.spatial_deconv(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class UpSampleBlock(nn.Module):
    """
    Upsamples spatial dimensions (inverting the encoder’s Conv3d downsampling)
    using a transposed convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            output_padding=(0, 1, 1)
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

#########################################
# CNN Decoder: Mirror of the Encoder
#########################################

class CNN_Decoder(nn.Module):
    def __init__(self, input_frames, bottleneck_channels, h_enc, w_enc,
                 num_stages, blocks_per_stage, output_channels):
        """
        Args:
            input_frames (int): Number of time frames (same as encoder).
            bottleneck_channels (int): Number of channels in the bottleneck feature map.
            h_enc, w_enc (int): Spatial dimensions after encoder downsampling.
            num_stages (int): Number of downsampling stages in the encoder.
            blocks_per_stage (int): Number of inverse spatiotemporal blocks per stage.
            output_channels (int): Number of channels in the output video.
        """
        super(CNN_Decoder, self).__init__()
        layers = []
        in_channels = bottleneck_channels
        # For each stage (mirroring encoder stages)
        for stage in range(num_stages):
            # Apply a series of inverse spatiotemporal blocks.
            for block in range(blocks_per_stage):
                # Optionally reduce channels at the beginning of each stage (except the first)
                if block == 0 and stage > 0:
                    out_channels = in_channels // 2
                    layers.append(InverseSpatioTemporalBlock(in_channels, out_channels))
                    in_channels = out_channels
                else:
                    layers.append(InverseSpatioTemporalBlock(in_channels, in_channels))
            # Upsample spatially if not at the final stage.
            if stage < num_stages:
                # In upsampling, if not the last stage, reduce the channel count by half.
                if stage < num_stages - 1:
                    out_channels = in_channels // 2
                else:
                    out_channels = in_channels
                layers.append(UpSampleBlock(in_channels, out_channels))
                in_channels = out_channels
        
        self.decoder_blocks = nn.Sequential(*layers)
        # Final convolution to map features to desired output channels (e.g. RGB)
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.decoder_blocks(x)
        x = self.final_conv(x)
        return x

#########################################
# PlumeGenerator: Combines Encoder, mlp_param, and Decoder
#########################################

class PlumeGenerator(nn.Module):
    def __init__(self, model_config):
        """
        model_config should include:
          - input_channels, cnn_output_shape, num_cnn_layers, blocks_per_stage, 
            start_filters, input_frames,
          - param_input_shape, param_hidden_layers, param_output_shape,
          - output_channels (for the generated video),
          - input_height and input_width (to compute spatial dims after encoder downsampling)
        """
        super(PlumeGenerator, self).__init__()
        # Encoder config (same as in PropertyPredictor)
        self.input_channels = model_config['input_channels']
        self.cnn_output_shape = model_config['cnn_output_shape']
        self.num_cnn_layers = model_config['num_cnn_layers']  # number of stages
        self.cnn_blocks_per_stage = model_config['blocks_per_stage']
        self.start_filters = model_config['start_filters']
        self.input_frames = model_config['input_frames']
        self.output_channels = model_config['output_channels']
        self.input_height = model_config['input_height']
        self.input_width = model_config['input_width']
        
        self.encoder = CNN_Encoder(
            input_channels=self.input_channels, 
            output_shape=self.cnn_output_shape,
            num_stages=self.num_cnn_layers, 
            blocks_per_stage=self.cnn_blocks_per_stage,
            start_filters=self.start_filters,  
            input_frames=self.input_frames
        )
        
        self.mlp_param = FeedForward(
            model_config['param_input_shape'], 
            model_config['param_output_shape'],
            hidden_layers=model_config['param_hidden_layers']
        )
        
        # Determine the spatial dimensions at the bottleneck.
        self.h_enc = self.input_height // (2 ** self.num_cnn_layers)
        self.w_enc = self.input_width // (2 ** self.num_cnn_layers)
        # Final number of channels after the encoder stages:
        self.bottleneck_channels = self.start_filters * (2 ** (self.num_cnn_layers - 1))
        
        # Inverse projection: map the param latent vector to a feature map
        self.inverse_project = nn.Linear(
            model_config['param_output_shape'],
            self.input_frames * self.bottleneck_channels * self.h_enc * self.w_enc
        )
        
        # Decoder: symmetric (inverse) CNN for video generation.
        self.decoder = CNN_Decoder(
            input_frames=self.input_frames,
            bottleneck_channels=self.bottleneck_channels,
            h_enc=self.h_enc,
            w_enc=self.w_enc,
            num_stages=self.num_cnn_layers,
            blocks_per_stage=self.cnn_blocks_per_stage,
            output_channels=self.output_channels
        )
    
    def encode(self, x, param=None):
        """
        Optionally encode an input video x and (if provided) a parameter tensor.
        This can be used for autoencoding tasks.
        """
        latent = self.encoder(x)
        if param is not None:
            param_encoded = self.mlp_param(param)
            latent = torch.cat((latent, param_encoded), dim=1)
        return latent
    
    def decode(self, param):
        """
        Generates a video from the parameter tensor.
        """
        # Process param through the same MLP that was used in the predictor.
        param_encoded = self.mlp_param(param)
        B = param_encoded.size(0)
        # Map the latent vector to a feature map matching the bottleneck dimensions.
        x = self.inverse_project(param_encoded)
        x = x.view(B, self.bottleneck_channels, self.input_frames, self.h_enc, self.w_enc)
        # Decode the feature map to reconstruct a video.
        x = self.decoder(x)
        return x
    
    def forward(self, x=None, param=None):
        """
        If an input video x is provided, you could use the encoder (and optionally combine with param)
        to obtain a latent representation. For video generation from parameters, simply pass param.
        Here we focus on generation from param.
        """
        if x is not None:
            # For an autoencoding variant, you might encode x then decode.
            latent = self.encode(x, param)
            # In this example we decode only from the param tensor.
            return self.decode(param)
        else:
            return self.decode(param)
    
class PropertyPredictor(nn.Module):
    def __init__(self,model_config):
        super(PropertyPredictor, self).__init__()
        # CNN encoder config
        self.input_channels = model_config['input_channels']
        self.cnn_output_shape = model_config['cnn_output_shape']
        self.num_cnn_layers = model_config['num_cnn_layers']
        self.cnn_blocks_per_stage = model_config['blocks_per_stage']
        self.start_filters = model_config['start_filters']
        self.input_frames = model_config['input_frames']
        
        # mlp parameter config
        self.param_input_shape = model_config['param_input_shape']
        self.param_hidden_layers = model_config['param_hidden_layers']
        self.param_output_shape = model_config['param_output_shape']        
        
        # mlp_head config
        self.hidden_layers = model_config['hidden_layers']
        self.output_shape = model_config['output_shape']
        
        self.encoder = CNN_Encoder(input_channels=self.input_channels, output_shape=self.cnn_output_shape,
                                     num_stages=self.num_cnn_layers, blocks_per_stage=self.cnn_blocks_per_stage,
                                     start_filters=self.start_filters,  input_frames=self.input_frames)
        
        if self.param_input_shape is not None:
            self.mlp_param = FeedForward(self.param_input_shape, self.param_output_shape,
                                        hidden_layers=self.param_hidden_layers)
            
        if self.param_input_shape is not None:
            self.mlp_head = FeedForward(self.cnn_output_shape+self.param_output_shape, self.output_shape,
                                        hidden_layers=self.hidden_layers)            
        else:
            self.mlp_head = FeedForward(self.cnn_output_shape, self.output_shape,
                                        hidden_layers=self.hidden_layers)
        
    def forward(self, x,y=None):
        # spatiotemporal encoder
        x = self.encoder(x)
        
        # mlp parameter encoder
        if y is not None:
            y = self.mlp_param(y)
            x = torch.concat((x,y),dim=1)
        
        # mlp prediction head
        x = self.mlp_head(x)
        
        return x

def train_regressor(model, train_loader, val_loader, device, checkpoint_filename, num_epochs, optimizer, criterion):
    """Train the regressor and track loss and R^2 for both training and validation."""
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_R2': [],
        'val_R2': []
    }

    best_val_loss = float('inf')
    model_save_path = checkpoint_filename  # Save the best model here

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_targets = []
        all_train_preds = []
        
        # Training loop
        for inputs, (targets, params) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            params = params.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, params)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accumulate predictions and targets.
            all_train_preds.append(outputs.detach().cpu())
            all_train_targets.append(targets.detach().cpu())
            
        train_loss /= len(train_loader)
        # Concatenate all batch predictions/targets.
        all_train_preds = torch.cat(all_train_preds, dim=0).numpy()
        all_train_targets = torch.cat(all_train_targets, dim=0).numpy()
        train_R2 = r2_score(all_train_targets, all_train_preds)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for inputs, (targets, params) in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                params = params.to(device)
                
                outputs = model(inputs, params)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                all_val_preds.append(outputs.detach().cpu())
                all_val_targets.append(targets.detach().cpu())
                
        val_loss /= len(val_loader)
        all_val_preds = torch.cat(all_val_preds, dim=0).numpy()
        all_val_targets = torch.cat(all_val_targets, dim=0).numpy()
        val_R2 = r2_score(all_val_targets, all_val_preds)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_R2'].append(train_R2)
        history['val_R2'].append(val_R2)
        
        # Print epoch summary (multiply R^2 by 100 to display as percentage if desired)
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train R^2: {train_R2*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val R^2: {val_R2*100:.2f}%")
        
        # Save history to CSV after every epoch.
        pd.DataFrame(history).to_csv(checkpoint_filename+'.csv', index=False)
        
        # Save model if validation loss improved.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved, saving model at epoch {epoch+1}")
            
    return history

def train_classifier(model, train_loader, val_loader, device, checkpoint_filename, num_epochs, optimizer, criterion):
    """Train the classifier and track loss and accuracy for both training and validation."""
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    best_val_loss = float('inf')
    model_save_path = checkpoint_filename  # Save the best model here

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training loop
        for inputs, (targets,params) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            params = params.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs,params)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate training accuracy
            # Assume outputs shape is (B, num_classes) and targets are one-hot encoded.
            predicted = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(targets, dim=1)
            train_correct += (predicted == true_labels).sum().item()
            train_total += inputs.size(0)
            
        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, (targets,params) in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                params = params.to(device)
                
                outputs = model(inputs,params)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                predicted = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(targets, dim=1)
                val_correct += (predicted == true_labels).sum().item()
                val_total += inputs.size(0)
                
        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        pd.DataFrame(history).to_csv(checkpoint_filename+'.csv',index=False) 
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved, saving model at epoch {epoch+1}")
            
    return history

def SSIM(pred, true):
    pred, true = pred.cpu().detach().numpy(), true.cpu().detach().numpy()
    ssim_total = 0.0
    count = 0
    for b in range(pred.shape[0]):
        for f in range(pred.shape[1]):
            # Rearrange image to (H, W, C)
            img_pred = pred[b, f]#.swapaxes(0, 2)
            img_true = true[b, f]#.swapaxes(0, 2)
            # Use channel_axis to indicate the channel dimension
            ssim_total += structural_similarity(img_pred, img_true, channel_axis=0, data_range=1.0)
            count += 1
    return ssim_total / count


def train_generator(model, train_loader, val_loader, device, checkpoint_filename, num_epochs, optimizer, criterion):
    """Train the generator and SSIM and MSE loss per epoch"""
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_SSIM': [],
        'val_SSIM': []
    }

    best_val_loss = float('inf')
    model_save_path = checkpoint_filename  # Save the best model here

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_ssim = 0.0
        
        # Training loop
        for inputs, (targets, params) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            params = params.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, torch.concat((targets,params),dim=1))
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ssim += SSIM(outputs,inputs)

            
        train_loss /= len(train_loader)
        train_ssim /= len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_ssim = 0.0

        with torch.no_grad():
            for inputs, (targets, params) in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                params = params.to(device)
                
                outputs = model(inputs, torch.concat((targets,params),dim=1))
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
                val_ssim += SSIM(outputs,inputs)
            
                
        val_loss /= len(val_loader)
        val_ssim /= len(val_loader)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_SSIM'].append(train_ssim)
        history['val_SSIM'].append(val_ssim)
        
        # Print epoch summary (multiply R^2 by 100 to display as percentage if desired)
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train SSIM: {train_ssim*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val SSIM: {val_ssim*100:.2f}%")
        
        # Save history to CSV after every epoch.
        pd.DataFrame(history).to_csv(checkpoint_filename+'.csv', index=False)
        
        # Save model if validation loss improved.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved, saving model at epoch {epoch+1}")
            
    return history