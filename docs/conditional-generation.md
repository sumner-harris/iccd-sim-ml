# Joint conditional generation, regression, and classification

The joint conditional VAE is an experimental surrogate for the expensive
plasma-to-image workflow. One jointly trained model supports three operations:

1. generate a canonical ICCD-like video from laser conditions and material
   properties;
2. regress material properties from a video and its laser conditions; and
3. classify a video among the element classes represented during training.

It is a data-driven surrogate, not a replacement for the physical solver. Its
predictions inherit the current forward model's continuum-LTE assumptions and
are not yet synthetic camera counts.

## Data passed to each branch

The public batch keys are:

- `video`: standardized video in `(B,C,T,H,W)` order;
- `laser_conditions`: `[laser_power_wcm, rspot]`;
- `material_properties`: `[cp_metal, h_vapor, kappa_metal,
  laser_reflectivity, mass_density_metal, t_boil, tcrit]`;
- `class_index`: integer element class;
- `times_s`: canonical physical frame times.

The ordering above is part of the model contract. `laser_power_wcm` is the
historical intensity-like simulation input, conventionally in `W cm^-2`; it is
not laser pulse energy. A joule-valued energy needs a defined spatial and
temporal pulse integration. `rspot` retains the source model's spot-radius
semantics.

The regression and classification heads receive only the video embedding and
laser conditions. They must never receive the true material-property vector
or class label as an input. The conditional prior and decoder receive both
laser conditions and material properties because those quantities define the
requested generation. This separation prevents the regression branch from
being handed its answer during joint training.

## Architecture

`JointConditionalVAE` contains:

- a shared factorized 3D video encoder;
- regression and classification heads driven by the deterministic video
  embedding and laser-condition embedding;
- a conditional prior
  `p(z | laser_conditions, material_properties)`;
- a training posterior
  `q(z | video, laser_conditions, material_properties)`; and
- a 3D conditional decoder with a fixed `(C,T,H,W)` output.

The decoder concatenates the sampled latent vector with an embedding of the
laser and material conditions, projects that vector to a small 3D seed, and
uses a geometric sequence of exact-size trilinear resizes followed by 3D
convolution, GroupNorm, and GELU. This avoids transposed-convolution
checkerboard artifacts and guarantees the configured output shape, including
odd sizes. At every resize-convolution stage, a learned FiLM-like affine
modulation derived from the condition embedding adjusts the normalized
channels. The final 3D convolution is linear.

Both latent distributions are diagonal Gaussians. If their parameters are
`mu_q, logvar_q` and `mu_p, logvar_p`, respectively, the per-sample divergence
is

```text
0.5 * sum(
    logvar_p - logvar_q
    + (exp(logvar_q) + (mu_q - mu_p)^2) / exp(logvar_p)
    - 1
)
```

The implementation bounds log variances before exponentiation. Mixed-precision
training should still evaluate this expression in float32.

## Canonical physical time is required

The decoder has a fixed output shape, whereas source simulations may contain
different numbers of frames at irregular times. Choose and version one
canonical physical time grid in seconds. Align source frames in an offline,
versioned preprocessing step. `JointPrecomputedVideoDataset` then verifies the
exact video shape and `times_s` values before samples can enter a batch; it
does not silently interpolate them in `__getitem__`.

Do not train on frame index as a proxy for time. A generated frame has a clear
physical meaning only when every training target at that index represents the
same time. Restrict the dataset to sequences covering the canonical interval,
or carry an explicit validity mask if partial coverage is intentionally
supported. See [data-and-splits.md](data-and-splits.md) for the required split
and preprocessing rules.

## Radiance transform and decoder output

Raw photon radiance spans many orders of magnitude. Fit
`VideoStandardizer(log1p=True)` on the training IDs only, then train the model
on

```text
(log1p(radiance) - training_mean) / training_scale
```

The decoder's final layer is linear. A sigmoid would impose an artificial
`[0,1]` range, while ReLU or softplus would incorrectly prohibit negative
standardized values. Convert generated values back to radiance with the frozen
training scaler, preferably in float64, and clamp only small negative values
caused by extrapolation or roundoff after `expm1`. Preserve the standardized
output as well so numerical failures remain auditable.

## One joint training objective

One optimizer step trains all three capabilities:

```text
loss = reconstruction_weight * reconstruction_loss
     + beta(step) * KL(q || p)
     + regression_weight * regression_loss
     + classification_weight * cross_entropy
```

Reconstruction is evaluated in standardized log-radiance space. Average it
over pixels so its magnitude does not change merely because the canonical
grid changes. The regression loss operates on train-standardized physical
properties, and classification consumes raw logits. Since most plume images
contain a large dark background, always compare an unweighted reconstruction
loss with a documented foreground-aware alternative before adopting the
latter.

`JointLossConfig` uses Smooth L1 for reconstruction and regression, cross
entropy for classification, and the conditional Gaussian KL above. Its
optional `temporal_gradient_weight` adds a Smooth L1 penalty on adjacent-frame
differences. `free_bits` is applied per latent dimension before reduction.
The training loop does not guess a schedule: the caller supplies a
`kl_warmup_weight` in `[0,1]`, and the effective coefficient is
`loss_config.kl_weight * kl_warmup_weight`.

Starting at full KL weight often causes posterior collapse. Use a recorded KL
warm-up from zero to the configured weight, monitor both total KL and active
latent dimensions, and consider free bits only if collapse remains visible.
Loss weights and the warm-up schedule are hyperparameters selected entirely
inside the training folds.

## Minimal API pattern

The maintained names are `JointCVAEConfig`, `JointConditionalVAE`,
`JointPrecomputedVideoDataset`, `JointLossConfig`, and
`train_joint_one_epoch`. A complete executable example will depend on the
chosen manifest, canonical time grid, and train-only scalers, but the intended
flow is:

```python
import torch
from torch.utils.data import DataLoader

from iccd_sim_ml.data import JointPrecomputedVideoDataset
from iccd_sim_ml.models import JointCVAEConfig, JointConditionalVAE
from iccd_sim_ml.training import JointLossConfig, train_joint_one_epoch

model_config = JointCVAEConfig(
    video_shape=(1, len(canonical_times_s), 96, 96),
    num_classes=len(class_to_index),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JointConditionalVAE(model_config).to(device)

train_dataset = JointPrecomputedVideoDataset(
    manifest,
    sample_ids=split.train,
    scalers=train_scalers,
    class_to_index=class_to_index,
    expected_video_shape=(1, len(canonical_times_s), 96, 96),
    expected_times_s=canonical_times_s,
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

loss_config = JointLossConfig()
kl_warmup_weight = min(1.0, (epoch + 1) / warmup_epochs)
metrics = train_joint_one_epoch(
    model,
    train_loader,
    optimizer,
    device=device,
    loss_config=loss_config,
    kl_warmup_weight=kl_warmup_weight,
)

batch = next(iter(train_loader))
video = batch["video"].to(device)
laser = batch["laser_conditions"].to(device)
properties = batch["material_properties"].to(device)

model.eval()
with torch.no_grad():
    property_prediction, class_logits = model.predict(video, laser)
    samples = model.generate(
        laser,
        properties,
        num_samples=16,
        temperature=1.0,
    )
```

`predict` returns `(property_prediction, class_logits)`. Generation always
returns `(B,S,C,T,H,W)`, including when `num_samples=1`; it never silently
squeezes the sample dimension. The default model configuration is mirrored in
[`configs/joint_cvae_template.json`](../configs/joint_cvae_template.json) and
can be loaded with `JointCVAEConfig.from_dict` after parsing the JSON.

## What uncertainty samples mean

Sampling the conditional prior produces variation learned from the training
data after accounting for the supplied laser and material descriptors. The
sample mean, quantiles, and pixelwise spread can summarize that conditional
latent variability. They do not automatically provide calibrated epistemic
uncertainty, confidence in extrapolation, measurement noise, or uncertainty
from missing physics.

Use `temperature=0` or the prior mean for a deterministic conditional result.
Use repeated prior samples at `temperature=1` for the learned distribution;
other temperatures are sensitivity controls and must be labeled as such. For
epistemic assessment, use held-out-material validation and independently
trained ensembles or another separately validated method.

## Evaluation and out-of-distribution limits

For unseen-material evaluation, hold out every simulation from complete
elements. The generator may then receive the held-out element's known physical
property vector and laser condition, but no held-out video may affect weights,
scalers, early stopping, or model selection. Evaluate temporal profiles,
spatial morphology, integrated radiance, and distributional coverage rather
than relying only on pixel MSE.

For classification, use a separate known-class split in which every reported
class was represented during training. A softmax score does not turn a
closed-set classifier into an unseen-element detector.

Conditional generation is best viewed as interpolation when both the laser
condition and material-property vector lie within well-supported training
regions. A point can be inside every one-dimensional feature range yet remain
far from the joint training manifold. Report nearest-neighbor or covariance-
aware distances in standardized condition space, flag extrapolation, and do
not present an unvalidated out-of-domain video as equivalent to a simulation.
The model does not enforce conservation laws, hydrodynamic equations, or
atomic kinetics.

## Checkpoint provenance

A usable checkpoint should include the joint model and loss configurations,
optimizer state, train-only scalers, class mapping, manifest and split IDs,
canonical `times_s`, canonical spatial shape, condition/property ordering,
simulator and atomic-reference versions, and the training epoch/step used by
the KL schedule. Without that information, generated videos cannot be
interpreted reproducibly.
