"""Reusable factorized spatiotemporal neural-network blocks."""

from __future__ import annotations

from collections.abc import Sequence

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


class SpatioTemporalBlock(nn.Module):
    """A residual (2+1)D block: spatial mixing followed by temporal mixing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel counts must be positive")
        self.spatial = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        return self.activation(self.temporal(self.spatial(inputs)) + residual)


class SpatialDownsample(nn.Module):
    """Halve image height and width without reducing the time axis."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class MLP(nn.Module):
    """A small configurable multilayer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dimensions = [input_dim, *hidden_dims, output_dim]
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("MLP dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
            layers.append(nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.append(nn.GELU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
