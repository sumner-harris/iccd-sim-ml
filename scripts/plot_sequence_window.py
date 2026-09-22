"""Plot every simulated frame in a requested time window on one radiance scale."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-ns", type=float, default=0.0)
    parser.add_argument("--stop-ns", type=float, required=True)
    parser.add_argument("--floor", type=float, default=1.0e-32)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as product:
        video = np.asarray(product["video"], dtype=np.float64)
        times_ns = np.asarray(product["times_s"], dtype=np.float64) * 1.0e9
        x_mm = np.asarray(product["x_m"], dtype=np.float64) * 1.0e3
        z_mm = np.asarray(product["z_m"], dtype=np.float64) * 1.0e3
    selected = np.flatnonzero((times_ns >= args.start_ns) & (times_ns <= args.stop_ns))
    if selected.size == 0:
        raise ValueError("No frames fall in the requested time window")

    columns = min(5, selected.size)
    rows = math.ceil(selected.size / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.1 * columns, 2.8 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    maximum = max(float(np.max(video[selected])), args.floor * 10.0)
    norm = LogNorm(vmin=args.floor, vmax=maximum, clip=True)
    image = None
    for axis in axes.ravel():
        axis.set_visible(False)
    for axis, index in zip(axes.ravel(), selected, strict=True):
        axis.set_visible(True)
        image = axis.imshow(
            np.maximum(video[index], args.floor).T,
            origin="lower",
            aspect="auto",
            extent=(x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]),
            cmap="inferno",
            norm=norm,
        )
        axis.set_title(f"{times_ns[index]:g} ns\npeak {np.max(video[index]):.2e}", fontsize=9)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("z (mm)")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.015)
        colorbar.set_label("photon radiance (photons s$^{-1}$ m$^{-2}$ sr$^{-1}$)")
    figure.suptitle(
        f"Synthetic Cu continuum emission, {args.start_ns:g}--{args.stop_ns:g} ns\n"
        f"shared logarithmic scale; values below {args.floor:.0e} shown at floor"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
