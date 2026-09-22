"""3D video encoders and conditioned prediction heads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional environment
    if exc.name == "torch":
        raise ImportError(
            "iccd_sim_ml.models requires PyTorch. Install the ML extras with "
            "`pip install iccd-sim-ml[ml]`."
        ) from exc
    raise

from .blocks import MLP, SpatialDownsample, SpatioTemporalBlock


@dataclass(frozen=True)
class VideoEncoderConfig:
    input_channels: int = 1
    stage_channels: tuple[int, ...] = (32, 64, 128, 256)
    blocks_per_stage: int = 2
    embedding_dim: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_channels", tuple(self.stage_channels))
        if self.input_channels <= 0 or not self.stage_channels:
            raise ValueError("Encoder channel configuration must be non-empty and positive")
        if any(channels <= 0 for channels in self.stage_channels):
            raise ValueError("All stage channel counts must be positive")
        if self.blocks_per_stage <= 0 or self.embedding_dim <= 0:
            raise ValueError("blocks_per_stage and embedding_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VideoEncoderConfig:
        return cls(
            input_channels=int(value.get("input_channels", 1)),
            stage_channels=tuple(value.get("stage_channels", (32, 64, 128, 256))),
            blocks_per_stage=int(value.get("blocks_per_stage", 2)),
            embedding_dim=int(value.get("embedding_dim", 256)),
            dropout=float(value.get("dropout", 0.0)),
        )


@dataclass(frozen=True)
class VideoRegressorConfig:
    encoder: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    condition_dim: int = 2
    condition_hidden: tuple[int, ...] = (16, 16)
    condition_embedding_dim: int = 16
    head_hidden: tuple[int, ...] = (256, 64)
    num_targets: int = 7
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_hidden", tuple(self.condition_hidden))
        object.__setattr__(self, "head_hidden", tuple(self.head_hidden))
        if self.condition_dim < 0 or self.num_targets <= 0:
            raise ValueError("condition_dim must be non-negative and num_targets positive")
        if self.condition_dim > 0 and self.condition_embedding_dim <= 0:
            raise ValueError("A conditioned model needs a positive condition_embedding_dim")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VideoRegressorConfig:
        return cls(
            encoder=VideoEncoderConfig.from_dict(value.get("encoder", {})),
            condition_dim=int(value.get("condition_dim", 2)),
            condition_hidden=tuple(value.get("condition_hidden", (16, 16))),
            condition_embedding_dim=int(value.get("condition_embedding_dim", 16)),
            head_hidden=tuple(value.get("head_hidden", (256, 64))),
            num_targets=int(value.get("num_targets", 7)),
            dropout=float(value.get("dropout", 0.0)),
        )


@dataclass(frozen=True)
class VideoClassifierConfig:
    encoder: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    condition_dim: int = 2
    condition_hidden: tuple[int, ...] = (16, 16)
    condition_embedding_dim: int = 16
    head_hidden: tuple[int, ...] = (128, 32)
    num_classes: int = 6
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_hidden", tuple(self.condition_hidden))
        object.__setattr__(self, "head_hidden", tuple(self.head_hidden))
        if self.condition_dim < 0 or self.num_classes <= 1:
            raise ValueError("condition_dim must be non-negative and num_classes greater than one")
        if self.condition_dim > 0 and self.condition_embedding_dim <= 0:
            raise ValueError("A conditioned model needs a positive condition_embedding_dim")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VideoClassifierConfig:
        return cls(
            encoder=VideoEncoderConfig.from_dict(value.get("encoder", {})),
            condition_dim=int(value.get("condition_dim", 2)),
            condition_hidden=tuple(value.get("condition_hidden", (16, 16))),
            condition_embedding_dim=int(value.get("condition_embedding_dim", 16)),
            head_hidden=tuple(value.get("head_hidden", (128, 32))),
            num_classes=int(value.get("num_classes", 6)),
            dropout=float(value.get("dropout", 0.0)),
        )


class VideoEncoder3D(nn.Module):
    """Factorized 3D CNN accepting videos in ``(B,C,T,H,W)`` order."""

    def __init__(self, config: VideoEncoderConfig | None = None) -> None:
        super().__init__()
        config = VideoEncoderConfig() if config is None else config
        self.config = config
        stages: list[nn.Module] = []
        in_channels = config.input_channels
        for out_channels in config.stage_channels:
            blocks: list[nn.Module] = [SpatioTemporalBlock(in_channels, out_channels)]
            blocks.extend(
                SpatioTemporalBlock(out_channels, out_channels)
                for _ in range(config.blocks_per_stage - 1)
            )
            blocks.append(SpatialDownsample(out_channels))
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shape (B,C,T,H,W), got {tuple(video.shape)}")
        if video.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} input channels, got {video.shape[1]}"
            )
        return self.projection(self.pool(self.stages(video)))


class _ConditionedPredictor(nn.Module):
    def __init__(
        self,
        *,
        encoder_config: VideoEncoderConfig,
        condition_dim: int,
        condition_hidden: tuple[int, ...],
        condition_embedding_dim: int,
        head_hidden: tuple[int, ...],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = VideoEncoder3D(encoder_config)
        self.condition_dim = condition_dim
        if condition_dim > 0:
            self.condition_encoder: nn.Module | None = MLP(
                condition_dim,
                condition_embedding_dim,
                hidden_dims=condition_hidden,
                dropout=dropout,
            )
            head_input = encoder_config.embedding_dim + condition_embedding_dim
        else:
            self.condition_encoder = None
            head_input = encoder_config.embedding_dim
        self.head = MLP(
            head_input,
            output_dim,
            hidden_dims=head_hidden,
            dropout=dropout,
        )

    def forward(self, video: torch.Tensor, conditions: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(video)
        if self.condition_encoder is not None:
            if conditions is None:
                raise ValueError("This model requires a conditions tensor")
            if conditions.ndim != 2 or conditions.shape[1] != self.condition_dim:
                raise ValueError(
                    f"Expected conditions shape (B,{self.condition_dim}), "
                    f"got {tuple(conditions.shape)}"
                )
            if conditions.shape[0] != video.shape[0]:
                raise ValueError("Video and conditions batch sizes differ")
            encoded = torch.cat((encoded, self.condition_encoder(conditions)), dim=1)
        elif conditions is not None and conditions.numel() > 0:
            raise ValueError("This model was configured without conditioning features")
        return self.head(encoded)


class VideoRegressor(_ConditionedPredictor):
    """Predict continuous material properties from video and laser conditions."""

    def __init__(self, config: VideoRegressorConfig | None = None) -> None:
        config = VideoRegressorConfig() if config is None else config
        super().__init__(
            encoder_config=config.encoder,
            condition_dim=config.condition_dim,
            condition_hidden=config.condition_hidden,
            condition_embedding_dim=config.condition_embedding_dim,
            head_hidden=config.head_hidden,
            output_dim=config.num_targets,
            dropout=config.dropout,
        )
        self.config = config


class VideoClassifier(_ConditionedPredictor):
    """Return unnormalized element-class logits from video and conditions."""

    def __init__(self, config: VideoClassifierConfig | None = None) -> None:
        config = VideoClassifierConfig() if config is None else config
        super().__init__(
            encoder_config=config.encoder,
            condition_dim=config.condition_dim,
            condition_hidden=config.condition_hidden,
            condition_embedding_dim=config.condition_embedding_dim,
            head_hidden=config.head_hidden,
            output_dim=config.num_classes,
            dropout=config.dropout,
        )
        self.config = config
