"""Joint conditional generation and material-property prediction.

The model deliberately keeps its two uses of material properties separate:
properties condition the variational generator, while the regression and
classification heads only receive information extracted from the video and
the laser conditions.  This prevents ground-truth properties from leaking
directly into either predictive output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional environment
    if exc.name == "torch":
        raise ImportError(
            "iccd_sim_ml.models requires PyTorch. Install the ML extras with "
            "`pip install iccd-sim-ml[ml]`."
        ) from exc
    raise

from .blocks import MLP
from .video import VideoEncoder3D, VideoEncoderConfig

DEFAULT_LASER_CONDITION_NAMES = ("laser_power_wcm", "rspot")
DEFAULT_PROPERTY_NAMES = (
    "cp_metal",
    "h_vapor",
    "kappa_metal",
    "laser_reflectivity",
    "mass_density_metal",
    "t_boil",
    "tcrit",
)


@dataclass(frozen=True)
class JointCVAEConfig:
    """Serializable architecture definition for :class:`JointConditionalVAE`.

    ``video_shape`` follows ``(channels, frames, height, width)``.  The model
    intentionally has a fixed output shape: sequences must be placed on a
    common physical time and image grid before batching.
    """

    video_shape: tuple[int, int, int, int] = (1, 16, 64, 64)
    encoder: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    laser_condition_names: tuple[str, ...] = DEFAULT_LASER_CONDITION_NAMES
    property_names: tuple[str, ...] = DEFAULT_PROPERTY_NAMES
    num_classes: int = 6
    latent_dim: int = 64
    laser_hidden: tuple[int, ...] = (32,)
    laser_embedding_dim: int = 32
    condition_hidden: tuple[int, ...] = (128,)
    condition_embedding_dim: int = 64
    posterior_hidden: tuple[int, ...] = (256,)
    prior_hidden: tuple[int, ...] = (128,)
    regression_hidden: tuple[int, ...] = (256, 64)
    classification_hidden: tuple[int, ...] = (128, 32)
    decoder_seed_shape: tuple[int, int, int] = (2, 4, 4)
    decoder_channels: tuple[int, ...] = (128, 64, 32, 16)
    group_norm_groups: int = 8
    dropout: float = 0.0
    log_variance_min: float = -12.0
    log_variance_max: float = 8.0

    def __post_init__(self) -> None:
        tuple_fields = (
            "video_shape",
            "laser_condition_names",
            "property_names",
            "laser_hidden",
            "condition_hidden",
            "posterior_hidden",
            "prior_hidden",
            "regression_hidden",
            "classification_hidden",
            "decoder_seed_shape",
            "decoder_channels",
        )
        for name in tuple_fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))

        if len(self.video_shape) != 4 or any(value <= 0 for value in self.video_shape):
            raise ValueError("video_shape must contain four positive values in (C,T,H,W) order")
        if self.encoder.input_channels != self.video_shape[0]:
            raise ValueError(
                "encoder.input_channels must equal the channel dimension of video_shape"
            )
        self._validate_names(self.laser_condition_names, "laser_condition_names")
        self._validate_names(self.property_names, "property_names")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        scalar_dimensions = {
            "latent_dim": self.latent_dim,
            "laser_embedding_dim": self.laser_embedding_dim,
            "condition_embedding_dim": self.condition_embedding_dim,
            "group_norm_groups": self.group_norm_groups,
        }
        for name, value in scalar_dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        hidden_fields = (
            "laser_hidden",
            "condition_hidden",
            "posterior_hidden",
            "prior_hidden",
            "regression_hidden",
            "classification_hidden",
            "decoder_channels",
        )
        for name in hidden_fields:
            if any(value <= 0 for value in getattr(self, name)):
                raise ValueError(f"{name} must contain only positive values")
        if len(self.decoder_channels) < 2:
            raise ValueError("decoder_channels must contain at least two stages")
        if len(self.decoder_seed_shape) != 3 or any(
            value <= 0 for value in self.decoder_seed_shape
        ):
            raise ValueError("decoder_seed_shape must contain three positive values")
        target_shape = self.video_shape[1:]
        if any(
            seed > target
            for seed, target in zip(self.decoder_seed_shape, target_shape, strict=True)
        ):
            raise ValueError("decoder_seed_shape cannot exceed the target (T,H,W) dimensions")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not (
            math.isfinite(self.log_variance_min)
            and math.isfinite(self.log_variance_max)
            and self.log_variance_min < self.log_variance_max
        ):
            raise ValueError("log-variance limits must be finite and strictly increasing")

    @staticmethod
    def _validate_names(names: tuple[str, ...], field_name: str) -> None:
        if not names:
            raise ValueError(f"{field_name} cannot be empty")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError(f"{field_name} must contain non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError(f"{field_name} must not contain duplicates")

    @property
    def laser_condition_dim(self) -> int:
        """Number of ordered laser-condition inputs."""

        return len(self.laser_condition_names)

    @property
    def material_property_dim(self) -> int:
        """Number of ordered material-property inputs and regression targets."""

        return len(self.property_names)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Persist derived dimensions as an additional schema guard.  Loading
        # rejects a name list whose width disagrees with these values.
        value["laser_condition_dim"] = self.laser_condition_dim
        value["material_property_dim"] = self.material_property_dim
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JointCVAEConfig:
        defaults = cls()
        config = cls(
            video_shape=tuple(value.get("video_shape", defaults.video_shape)),
            encoder=VideoEncoderConfig.from_dict(value.get("encoder", {})),
            laser_condition_names=tuple(
                value.get("laser_condition_names", defaults.laser_condition_names)
            ),
            property_names=tuple(value.get("property_names", defaults.property_names)),
            num_classes=int(value.get("num_classes", defaults.num_classes)),
            latent_dim=int(value.get("latent_dim", defaults.latent_dim)),
            laser_hidden=tuple(value.get("laser_hidden", defaults.laser_hidden)),
            laser_embedding_dim=int(value.get("laser_embedding_dim", defaults.laser_embedding_dim)),
            condition_hidden=tuple(value.get("condition_hidden", defaults.condition_hidden)),
            condition_embedding_dim=int(
                value.get("condition_embedding_dim", defaults.condition_embedding_dim)
            ),
            posterior_hidden=tuple(value.get("posterior_hidden", defaults.posterior_hidden)),
            prior_hidden=tuple(value.get("prior_hidden", defaults.prior_hidden)),
            regression_hidden=tuple(value.get("regression_hidden", defaults.regression_hidden)),
            classification_hidden=tuple(
                value.get("classification_hidden", defaults.classification_hidden)
            ),
            decoder_seed_shape=tuple(value.get("decoder_seed_shape", defaults.decoder_seed_shape)),
            decoder_channels=tuple(value.get("decoder_channels", defaults.decoder_channels)),
            group_norm_groups=int(value.get("group_norm_groups", defaults.group_norm_groups)),
            dropout=float(value.get("dropout", defaults.dropout)),
            log_variance_min=float(value.get("log_variance_min", defaults.log_variance_min)),
            log_variance_max=float(value.get("log_variance_max", defaults.log_variance_max)),
        )
        declared_laser_dim = int(value.get("laser_condition_dim", config.laser_condition_dim))
        declared_property_dim = int(
            value.get("material_property_dim", config.material_property_dim)
        )
        if declared_laser_dim != config.laser_condition_dim:
            raise ValueError("laser_condition_dim disagrees with laser_condition_names")
        if declared_property_dim != config.material_property_dim:
            raise ValueError("material_property_dim disagrees with property_names")
        return config


@dataclass(frozen=True)
class JointCVAEOutput:
    """Outputs needed by the three tasks and the joint CVAE objective."""

    reconstruction: torch.Tensor
    property_prediction: torch.Tensor
    class_logits: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_log_variance: torch.Tensor
    prior_mean: torch.Tensor
    prior_log_variance: torch.Tensor
    latent: torch.Tensor


def _group_count(channels: int, requested: int) -> int:
    """Return the largest valid GroupNorm group count up to ``requested``."""

    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0:
            return groups
    return 1  # pragma: no cover - every positive integer is divisible by one


def _resize_schedule(
    seed_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    steps: int,
) -> tuple[tuple[int, int, int], ...]:
    """Create monotonic geometric resize targets ending exactly at target."""

    sizes: list[tuple[int, int, int]] = []
    previous = seed_shape
    for step in range(1, steps + 1):
        fraction = step / steps
        current = []
        for seed, target, prior in zip(seed_shape, target_shape, previous, strict=True):
            if step == steps:
                value = target
            elif seed == target:
                value = target
            else:
                value = int(round(seed * (target / seed) ** fraction))
            current.append(min(target, max(prior, value)))
        previous = tuple(current)
        sizes.append(previous)
    return tuple(sizes)


class _ResizeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_groups: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.normalization = nn.GroupNorm(_group_count(out_channels, max_groups), out_channels)
        self.modulation = nn.Linear(condition_dim, 2 * out_channels)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        normalized = self.normalization(self.convolution(inputs))
        scale, shift = self.modulation(condition).chunk(2, dim=1)
        modulated = normalized * (1.0 + scale[:, :, None, None, None])
        modulated = modulated + shift[:, :, None, None, None]
        return self.activation(modulated)


class _VideoDecoder3D(nn.Module):
    """Decode a vector with exact-size resize-convolution stages."""

    def __init__(self, input_dim: int, config: JointCVAEConfig) -> None:
        super().__init__()
        self.seed_shape = config.decoder_seed_shape
        self.target_shape = config.video_shape[1:]
        self.seed_channels = config.decoder_channels[0]
        seed_values = self.seed_channels * math.prod(self.seed_shape)
        self.projection = nn.Linear(input_dim, seed_values)
        self.blocks = nn.ModuleList(
            _ResizeConvBlock(
                left,
                right,
                config.group_norm_groups,
                config.condition_embedding_dim,
            )
            for left, right in zip(
                config.decoder_channels[:-1], config.decoder_channels[1:], strict=True
            )
        )
        self.resize_shapes = _resize_schedule(self.seed_shape, self.target_shape, len(self.blocks))
        self.output = nn.Conv3d(
            config.decoder_channels[-1], config.video_shape[0], kernel_size=3, padding=1
        )

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        decoded = self.projection(inputs).reshape(batch_size, self.seed_channels, *self.seed_shape)
        for block, shape in zip(self.blocks, self.resize_shapes, strict=True):
            if decoded.shape[2:] != shape:
                decoded = F.interpolate(
                    decoded,
                    size=shape,
                    mode="trilinear",
                    align_corners=False,
                )
            decoded = block(decoded, condition)
        return self.output(decoded)


class JointConditionalVAE(nn.Module):
    """One model for conditional generation, regression, and classification."""

    def __init__(self, config: JointCVAEConfig | None = None) -> None:
        super().__init__()
        config = JointCVAEConfig() if config is None else config
        self.config = config
        self.video_encoder = VideoEncoder3D(config.encoder)
        self.laser_encoder = MLP(
            config.laser_condition_dim,
            config.laser_embedding_dim,
            hidden_dims=config.laser_hidden,
            dropout=config.dropout,
        )
        self.condition_encoder = MLP(
            config.laser_condition_dim + config.material_property_dim,
            config.condition_embedding_dim,
            hidden_dims=config.condition_hidden,
            dropout=config.dropout,
        )
        predictor_dim = config.encoder.embedding_dim + config.laser_embedding_dim
        self.property_head = MLP(
            predictor_dim,
            config.material_property_dim,
            hidden_dims=config.regression_hidden,
            dropout=config.dropout,
        )
        self.classification_head = MLP(
            predictor_dim,
            config.num_classes,
            hidden_dims=config.classification_hidden,
            dropout=config.dropout,
        )
        self.posterior = MLP(
            config.encoder.embedding_dim + config.condition_embedding_dim,
            2 * config.latent_dim,
            hidden_dims=config.posterior_hidden,
            dropout=config.dropout,
        )
        self.prior = MLP(
            config.condition_embedding_dim,
            2 * config.latent_dim,
            hidden_dims=config.prior_hidden,
            dropout=config.dropout,
        )
        self.decoder = _VideoDecoder3D(config.latent_dim + config.condition_embedding_dim, config)

    def _validate_video(self, video: torch.Tensor) -> None:
        expected = self.config.video_shape
        if video.ndim != 5 or tuple(video.shape[1:]) != expected:
            raise ValueError(
                f"Expected video shape (B,{','.join(map(str, expected))}), got {tuple(video.shape)}"
            )

    @staticmethod
    def _validate_matrix(
        value: torch.Tensor,
        *,
        name: str,
        width: int,
        batch_size: int | None = None,
    ) -> None:
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"Expected {name} shape (B,{width}), got {tuple(value.shape)}")
        if batch_size is not None and value.shape[0] != batch_size:
            raise ValueError(f"{name} batch size differs from the other model inputs")

    def _condition(
        self, laser_conditions: torch.Tensor, material_properties: torch.Tensor
    ) -> torch.Tensor:
        return self.condition_encoder(torch.cat((laser_conditions, material_properties), dim=1))

    def _distribution_parameters(
        self, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = parameters.chunk(2, dim=1)
        return mean, log_variance.clamp(
            min=self.config.log_variance_min,
            max=self.config.log_variance_max,
        )

    @staticmethod
    def _validate_temperature(temperature: float) -> float:
        value = float(temperature)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        return value

    @staticmethod
    def _sample(mean: torch.Tensor, log_variance: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0.0:
            return mean
        return mean + temperature * torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    def _validate_latent(self, latent: torch.Tensor, batch_size: int) -> None:
        if latent.ndim != 2 or latent.shape != (batch_size, self.config.latent_dim):
            raise ValueError(
                f"Expected latent shape ({batch_size},{self.config.latent_dim}), "
                f"got {tuple(latent.shape)}"
            )

    def _encode_predictors(
        self, video: torch.Tensor, laser_conditions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_embedding = self.video_encoder(video)
        laser_embedding = self.laser_encoder(laser_conditions)
        predictive_embedding = torch.cat((video_embedding, laser_embedding), dim=1)
        return (
            video_embedding,
            self.property_head(predictive_embedding),
            self.classification_head(predictive_embedding),
        )

    def forward(
        self,
        video: torch.Tensor,
        laser_conditions: torch.Tensor,
        material_properties: torch.Tensor,
        *,
        sample_posterior: bool | None = None,
        temperature: float = 1.0,
        latent: torch.Tensor | None = None,
    ) -> JointCVAEOutput:
        """Run all training branches with a single shared video encoding.

        ``sample_posterior=False`` is a convenience alias for
        ``temperature=0``.  An explicit ``latent`` overrides posterior
        sampling while posterior/prior parameters are still returned for the
        KL objective.
        """

        self._validate_video(video)
        batch_size = video.shape[0]
        self._validate_matrix(
            laser_conditions,
            name="laser_conditions",
            width=self.config.laser_condition_dim,
            batch_size=batch_size,
        )
        self._validate_matrix(
            material_properties,
            name="material_properties",
            width=self.config.material_property_dim,
            batch_size=batch_size,
        )
        temperature = self._validate_temperature(temperature)
        if sample_posterior is False:
            temperature = 0.0

        video_embedding, property_prediction, class_logits = self._encode_predictors(
            video, laser_conditions
        )
        condition_embedding = self._condition(laser_conditions, material_properties)
        posterior_mean, posterior_log_variance = self._distribution_parameters(
            self.posterior(torch.cat((video_embedding, condition_embedding), dim=1))
        )
        prior_mean, prior_log_variance = self._distribution_parameters(
            self.prior(condition_embedding)
        )
        if latent is None:
            latent = self._sample(posterior_mean, posterior_log_variance, temperature)
        else:
            self._validate_latent(latent, batch_size)
        reconstruction = self.decoder(
            torch.cat((latent, condition_embedding), dim=1), condition_embedding
        )
        return JointCVAEOutput(
            reconstruction=reconstruction,
            property_prediction=property_prediction,
            class_logits=class_logits,
            posterior_mean=posterior_mean,
            posterior_log_variance=posterior_log_variance,
            prior_mean=prior_mean,
            prior_log_variance=prior_log_variance,
            latent=latent,
        )

    def predict(
        self, video: torch.Tensor, laser_conditions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict material properties and class logits with one encoder pass."""

        self._validate_video(video)
        self._validate_matrix(
            laser_conditions,
            name="laser_conditions",
            width=self.config.laser_condition_dim,
            batch_size=video.shape[0],
        )
        _, properties, logits = self._encode_predictors(video, laser_conditions)
        return properties, logits

    def regress(self, video: torch.Tensor, laser_conditions: torch.Tensor) -> torch.Tensor:
        """Predict standardized material properties from video and laser inputs."""

        return self.predict(video, laser_conditions)[0]

    def classify(self, video: torch.Tensor, laser_conditions: torch.Tensor) -> torch.Tensor:
        """Return unnormalized element-class logits from video and laser inputs."""

        return self.predict(video, laser_conditions)[1]

    def reconstruct(
        self,
        video: torch.Tensor,
        laser_conditions: torch.Tensor,
        material_properties: torch.Tensor,
        *,
        temperature: float = 0.0,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct a video from its posterior (posterior mean by default)."""

        return self.forward(
            video,
            laser_conditions,
            material_properties,
            temperature=temperature,
            latent=latent,
        ).reconstruction

    def generate(
        self,
        laser_conditions: torch.Tensor,
        material_properties: torch.Tensor,
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate conditionally with shape ``(B,S,C,T,H,W)`` in all cases.

        ``temperature=0`` decodes the conditional-prior mean deterministically.
        An explicit latent may have shape ``(B,Z)`` (repeated ``S`` times) or
        ``(B,S,Z)``.  For the latter, the latent sample dimension is inferred
        when ``num_samples`` is left at its default value of one.
        """

        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        batch_size = laser_conditions.shape[0] if laser_conditions.ndim > 0 else 0
        self._validate_matrix(
            laser_conditions,
            name="laser_conditions",
            width=self.config.laser_condition_dim,
        )
        self._validate_matrix(
            material_properties,
            name="material_properties",
            width=self.config.material_property_dim,
            batch_size=batch_size,
        )
        temperature = self._validate_temperature(temperature)
        condition_embedding = self._condition(laser_conditions, material_properties)
        repeated_identical_latent = False

        if latent is None:
            prior_mean, prior_log_variance = self._distribution_parameters(
                self.prior(condition_embedding)
            )
            mean = prior_mean[:, None, :].expand(-1, num_samples, -1)
            log_variance = prior_log_variance[:, None, :].expand(-1, num_samples, -1)
            latents = self._sample(mean, log_variance, temperature)
            sample_count = num_samples
            repeated_identical_latent = temperature == 0.0
        elif latent.ndim == 2:
            self._validate_latent(latent, batch_size)
            sample_count = num_samples
            latents = latent[:, None, :].expand(-1, sample_count, -1)
            repeated_identical_latent = True
        elif latent.ndim == 3:
            expected = (batch_size, latent.shape[1], self.config.latent_dim)
            if tuple(latent.shape) != expected:
                raise ValueError(
                    "Expected latent shape (B,S,Z) with "
                    f"B={batch_size} and Z={self.config.latent_dim}, got {tuple(latent.shape)}"
                )
            sample_count = latent.shape[1]
            if num_samples not in {1, sample_count}:
                raise ValueError("num_samples disagrees with the explicit latent sample axis")
            latents = latent
        else:
            raise ValueError("latent must have shape (B,Z) or (B,S,Z)")

        # Decode an intentionally repeated latent once. Besides avoiding duplicate
        # work, this guarantees bitwise-identical deterministic samples on GPU;
        # batched convolution kernels can otherwise differ at round-off level by
        # batch position even when their inputs are identical.
        if repeated_identical_latent:
            decoder_input = torch.cat((latents[:, 0, :], condition_embedding), dim=1)
            generated = self.decoder(decoder_input, condition_embedding)
            return generated[:, None, ...].expand(
                -1, sample_count, *self.config.video_shape
            ).clone()

        repeated_condition = condition_embedding[:, None, :].expand(-1, sample_count, -1)
        decoder_input = torch.cat((latents, repeated_condition), dim=2).reshape(
            batch_size * sample_count, -1
        )
        generated = self.decoder(
            decoder_input,
            repeated_condition.reshape(batch_size * sample_count, -1),
        )
        return generated.reshape(batch_size, sample_count, *self.config.video_shape)
