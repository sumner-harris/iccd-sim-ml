"""Material-property regression from unordered sets of plume videos."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - optional ML dependency
    if exc.name == "torch":
        raise ImportError(
            "Set regressors require PyTorch. Install the ML extras with "
            "`pip install iccd-sim-ml[ml]`."
        ) from exc
    raise

from .blocks import MLP
from .video import VideoEncoder3D, VideoEncoderConfig

SetAggregator = Literal["deep_set", "set_transformer"]


@dataclass(frozen=True)
class MaterialSetRegressorConfig:
    """Configuration shared by Deep Sets and Set Transformer regressors."""

    aggregator: SetAggregator = "deep_set"
    encoder: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    condition_dim: int = 2
    condition_hidden: tuple[int, ...] = (16, 16)
    condition_embedding_dim: int = 16
    baseline_head_hidden: tuple[int, ...] = (256, 64)
    baseline_dropout: float = 0.0
    token_hidden: tuple[int, ...] = (256,)
    token_dim: int = 128
    set_hidden: tuple[int, ...] = (128,)
    head_hidden: tuple[int, ...] = (128, 64)
    num_targets: int = 7
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_feedforward_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_hidden", tuple(self.condition_hidden))
        object.__setattr__(self, "baseline_head_hidden", tuple(self.baseline_head_hidden))
        object.__setattr__(self, "token_hidden", tuple(self.token_hidden))
        object.__setattr__(self, "set_hidden", tuple(self.set_hidden))
        object.__setattr__(self, "head_hidden", tuple(self.head_hidden))
        if self.aggregator not in {"deep_set", "set_transformer"}:
            raise ValueError("aggregator must be 'deep_set' or 'set_transformer'")
        if self.condition_dim <= 0 or self.condition_embedding_dim <= 0:
            raise ValueError("condition dimensions must be positive")
        if self.token_dim <= 0 or self.num_targets <= 0:
            raise ValueError("token_dim and num_targets must be positive")
        if self.transformer_layers <= 0 or self.transformer_heads <= 0:
            raise ValueError("transformer layer and head counts must be positive")
        if self.token_dim % self.transformer_heads:
            raise ValueError("token_dim must be divisible by transformer_heads")
        if self.transformer_feedforward_dim <= 0:
            raise ValueError("transformer_feedforward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.baseline_dropout < 1.0:
            raise ValueError("baseline_dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MaterialSetRegressorConfig:
        defaults = cls()
        return cls(
            aggregator=str(value.get("aggregator", defaults.aggregator)),
            encoder=VideoEncoderConfig.from_dict(value.get("encoder", {})),
            condition_dim=int(value.get("condition_dim", defaults.condition_dim)),
            condition_hidden=tuple(value.get("condition_hidden", defaults.condition_hidden)),
            condition_embedding_dim=int(
                value.get("condition_embedding_dim", defaults.condition_embedding_dim)
            ),
            baseline_head_hidden=tuple(
                value.get("baseline_head_hidden", defaults.baseline_head_hidden)
            ),
            baseline_dropout=float(value.get("baseline_dropout", defaults.baseline_dropout)),
            token_hidden=tuple(value.get("token_hidden", defaults.token_hidden)),
            token_dim=int(value.get("token_dim", defaults.token_dim)),
            set_hidden=tuple(value.get("set_hidden", defaults.set_hidden)),
            head_hidden=tuple(value.get("head_hidden", defaults.head_hidden)),
            num_targets=int(value.get("num_targets", defaults.num_targets)),
            transformer_layers=int(value.get("transformer_layers", defaults.transformer_layers)),
            transformer_heads=int(value.get("transformer_heads", defaults.transformer_heads)),
            transformer_feedforward_dim=int(
                value.get("transformer_feedforward_dim", defaults.transformer_feedforward_dim)
            ),
            dropout=float(value.get("dropout", defaults.dropout)),
        )


class MaterialSetRegressor(nn.Module):
    """Encode each experiment, aggregate its material set, and regress properties.

    Inputs use ``videos=(B,K,C,T,H,W)``, ``conditions=(B,K,F)``, and an
    optional Boolean ``set_mask=(B,K)``. No positional encoding is used, so
    both supported aggregators are invariant to experiment order.
    """

    def __init__(self, config: MaterialSetRegressorConfig | None = None) -> None:
        super().__init__()
        config = MaterialSetRegressorConfig() if config is None else config
        self.config = config
        self._baseline_frozen = False
        self._pretrained_batchnorm_frozen = False
        self.encoder = VideoEncoder3D(config.encoder)
        self.condition_encoder = MLP(
            config.condition_dim,
            config.condition_embedding_dim,
            hidden_dims=config.condition_hidden,
            dropout=config.baseline_dropout,
        )
        experiment_embedding_dim = config.encoder.embedding_dim + config.condition_embedding_dim
        self.baseline_head = MLP(
            experiment_embedding_dim,
            config.num_targets,
            hidden_dims=config.baseline_head_hidden,
            dropout=config.baseline_dropout,
        )
        self.token_encoder = nn.Sequential(
            MLP(
                experiment_embedding_dim,
                config.token_dim,
                hidden_dims=config.token_hidden,
                dropout=config.dropout,
            ),
            nn.LayerNorm(config.token_dim),
            nn.GELU(),
        )

        if config.aggregator == "deep_set":
            self.transformer: nn.Module | None = None
            self.pooling_attention: nn.MultiheadAttention | None = None
            self.pooling_seed: nn.Parameter | None = None
            self.pooling_norm: nn.Module | None = None
            self.set_encoder = MLP(
                2 * config.token_dim,
                config.token_dim,
                hidden_dims=config.set_hidden,
                dropout=config.dropout,
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=config.token_dim,
                nhead=config.transformer_heads,
                dim_feedforward=config.transformer_feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=config.transformer_layers,
                enable_nested_tensor=False,
            )
            self.pooling_attention = nn.MultiheadAttention(
                config.token_dim,
                config.transformer_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.pooling_seed = nn.Parameter(torch.empty(1, 1, config.token_dim))
            nn.init.normal_(self.pooling_seed, mean=0.0, std=0.02)
            self.pooling_norm = nn.LayerNorm(config.token_dim)
            self.set_encoder = nn.Identity()

        self.head = MLP(
            config.token_dim,
            config.num_targets,
            hidden_dims=config.head_hidden,
            dropout=config.dropout,
        )
        final_residual_layer = next(
            module
            for module in reversed(tuple(self.head.modules()))
            if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(final_residual_layer.weight)
        nn.init.zeros_(final_residual_layer.bias)

    def freeze_pretrained_baseline(self) -> None:
        """Freeze and keep the transferred single-video regressor in eval mode."""

        self._baseline_frozen = True
        for module in (self.encoder, self.condition_encoder, self.baseline_head):
            module.requires_grad_(False)
            module.eval()

    def freeze_pretrained_batchnorm_statistics(self) -> None:
        """Keep transferred BatchNorm statistics fixed while fine-tuning weights.

        Set-training batches contain many experiments from one material, unlike
        the mixed-material batches used to train the standalone regressor.  This
        mode prevents those correlated bags from overwriting the pretrained
        encoder's running mean and variance without disabling weight gradients.
        """

        self._pretrained_batchnorm_frozen = True
        for module in (self.encoder, self.condition_encoder, self.baseline_head):
            for child in module.modules():
                if isinstance(child, nn.modules.batchnorm._BatchNorm):
                    child.eval()

    def unfreeze_pretrained_baseline(self, *, freeze_batchnorm_statistics: bool = False) -> None:
        """Make transferred weights trainable after a set-head warmup stage."""

        self._baseline_frozen = False
        self._pretrained_batchnorm_frozen = bool(freeze_batchnorm_statistics)
        for module in (self.encoder, self.condition_encoder, self.baseline_head):
            module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True) -> MaterialSetRegressor:
        super().train(mode)
        if self._baseline_frozen:
            self.encoder.eval()
            self.condition_encoder.eval()
            self.baseline_head.eval()
        elif self._pretrained_batchnorm_frozen:
            for module in (self.encoder, self.condition_encoder, self.baseline_head):
                for child in module.modules():
                    if isinstance(child, nn.modules.batchnorm._BatchNorm):
                        child.eval()
        return self

    def _validate_inputs(
        self,
        videos: torch.Tensor,
        conditions: torch.Tensor,
        set_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if videos.ndim != 6:
            raise ValueError(f"Expected videos shape (B,K,C,T,H,W), got {tuple(videos.shape)}")
        batch_size, set_size = videos.shape[:2]
        if conditions.shape != (batch_size, set_size, self.config.condition_dim):
            raise ValueError(
                f"Expected conditions shape ({batch_size},{set_size},"
                f"{self.config.condition_dim}), got {tuple(conditions.shape)}"
            )
        if set_mask is None:
            mask = torch.ones((batch_size, set_size), dtype=torch.bool, device=videos.device)
        else:
            if set_mask.shape != (batch_size, set_size):
                raise ValueError(
                    f"Expected set_mask shape ({batch_size},{set_size}), "
                    f"got {tuple(set_mask.shape)}"
                )
            mask = set_mask.to(device=videos.device, dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every material set must contain at least one valid experiment")
        return mask

    def encode_experiments(
        self,
        videos: torch.Tensor,
        conditions: torch.Tensor,
        set_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tokens, baseline predictions, and a validated Boolean mask."""

        mask = self._validate_inputs(videos, conditions, set_mask)
        batch_size, set_size = videos.shape[:2]
        flat_video = videos.flatten(0, 1)
        flat_conditions = conditions.flatten(0, 1)
        flat_mask = mask.flatten()
        valid_indices = flat_mask.nonzero(as_tuple=False).squeeze(1)
        video_embedding = self.encoder(flat_video.index_select(0, valid_indices))
        condition_embedding = self.condition_encoder(flat_conditions.index_select(0, valid_indices))
        experiment_embedding = torch.cat((video_embedding, condition_embedding), dim=1)
        valid_tokens = self.token_encoder(experiment_embedding)
        valid_predictions = self.baseline_head(experiment_embedding)
        flat_tokens = valid_tokens.new_zeros(
            (batch_size * set_size, self.config.token_dim)
        ).index_copy(0, valid_indices, valid_tokens)
        flat_predictions = valid_predictions.new_zeros(
            (batch_size * set_size, self.config.num_targets)
        ).index_copy(0, valid_indices, valid_predictions)
        return (
            flat_tokens.reshape(batch_size, set_size, -1),
            flat_predictions.reshape(batch_size, set_size, -1),
            mask,
        )

    def aggregate(self, tokens: torch.Tensor, set_mask: torch.Tensor) -> torch.Tensor:
        """Aggregate experiment tokens into one material embedding."""

        if self.config.aggregator == "deep_set":
            weights = set_mask.unsqueeze(-1).to(tokens.dtype)
            count = weights.sum(dim=1).clamp_min(1.0)
            mean = (tokens * weights).sum(dim=1) / count
            variance = ((tokens - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
            return self.set_encoder(torch.cat((mean, torch.sqrt(variance + 1.0e-6)), dim=1))

        assert self.transformer is not None
        assert self.pooling_attention is not None
        assert self.pooling_seed is not None
        assert self.pooling_norm is not None
        padding_mask = ~set_mask
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        query = self.pooling_seed.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.pooling_attention(
            query,
            encoded,
            encoded,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.pooling_norm(pooled[:, 0])

    def forward(
        self,
        videos: torch.Tensor,
        conditions: torch.Tensor,
        set_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction, _, _ = self.forward_with_individual(videos, conditions, set_mask)
        return prediction

    def forward_with_individual(
        self,
        videos: torch.Tensor,
        conditions: torch.Tensor,
        set_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return set prediction, per-experiment predictions, and valid mask."""

        tokens, baseline_predictions, mask = self.encode_experiments(videos, conditions, set_mask)
        weights = mask.unsqueeze(-1).to(baseline_predictions.dtype)
        baseline = (baseline_predictions * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        residual = self.head(self.aggregate(tokens, mask))
        return baseline + residual, baseline_predictions, mask


class DeepSetRegressor(MaterialSetRegressor):
    """Convenience wrapper enforcing the Deep Sets aggregator."""

    def __init__(self, config: MaterialSetRegressorConfig | None = None) -> None:
        selected = MaterialSetRegressorConfig() if config is None else config
        if selected.aggregator != "deep_set":
            raise ValueError("DeepSetRegressor requires aggregator='deep_set'")
        super().__init__(selected)


class SetTransformerRegressor(MaterialSetRegressor):
    """Convenience wrapper enforcing the Set Transformer aggregator."""

    def __init__(self, config: MaterialSetRegressorConfig | None = None) -> None:
        selected = (
            MaterialSetRegressorConfig(aggregator="set_transformer") if config is None else config
        )
        if selected.aggregator != "set_transformer":
            raise ValueError("SetTransformerRegressor requires aggregator='set_transformer'")
        super().__init__(selected)
