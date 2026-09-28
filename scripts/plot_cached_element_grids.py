"""Plot fixed-spot cached videos as power-density by time grids."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402


@dataclass(frozen=True)
class CachedCondition:
    record: dict[str, Any]
    path: Path
    power_wcm: float
    spot_m: float


def _even_indices(length: int, count: int) -> np.ndarray:
    """Return unique, endpoint-inclusive indices distributed over a sequence."""

    if not 1 <= count <= length:
        raise ValueError(f"count must be between 1 and {length}, got {count}")
    indices = np.floor(np.linspace(0, length - 1, count) + 0.5).astype(int)
    if np.unique(indices).size != count:
        raise RuntimeError("Even index selection unexpectedly produced duplicate indices")
    return indices


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spot-mm", type=float, default=1.25)
    parser.add_argument("--power-rows", type=int, default=10)
    parser.add_argument("--time-columns", type=int, default=10)
    parser.add_argument(
        "--display-percentile",
        type=float,
        default=99.9,
        help="Positive log-radiance percentile used as the per-element display ceiling.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _read_conditions(cache_dir: Path) -> dict[str, list[CachedCondition]]:
    manifest_path = cache_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[CachedCondition]] = defaultdict(list)
    for record in payload["records"]:
        path = Path(record["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        with np.load(path, allow_pickle=False) as product:
            conditions = np.asarray(product["conditions"], dtype=np.float64).reshape(-1)
        if conditions.size != 2 or not np.isfinite(conditions).all():
            raise ValueError(f"Expected two finite conditions in {path}, got {conditions}")
        grouped[str(record["element"])].append(
            CachedCondition(
                record=record,
                path=path,
                power_wcm=float(conditions[0]),
                spot_m=float(conditions[1]),
            )
        )
    return dict(grouped)


def _field_of_view(cache_dir: Path) -> tuple[float, float]:
    report = json.loads((cache_dir / "run_report.json").read_text(encoding="utf-8"))
    imaging = report["configuration"]["imaging"]
    return float(imaging["radial_max_m"]) * 1.0e3, float(imaging["axial_max_m"]) * 1.0e3


def _select_fixed_spot(
    records: list[CachedCondition], spot_m: float, power_rows: int
) -> list[CachedCondition]:
    matching = [row for row in records if np.isclose(row.spot_m, spot_m, rtol=0.0, atol=1.0e-8)]
    matching.sort(key=lambda row: row.power_wcm)
    powers = [row.power_wcm for row in matching]
    if len(powers) != len(set(powers)):
        raise ValueError("Fixed-spot selection contains duplicate laser power densities")
    if len(matching) < power_rows:
        raise ValueError(
            f"Only {len(matching)} simulations exist at spot={spot_m * 1e3:g} mm; "
            f"cannot select {power_rows} rows"
        )
    return [matching[index] for index in _even_indices(len(matching), power_rows)]


def _load_selected(
    records: list[CachedCondition], time_columns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    videos: list[np.ndarray] = []
    common_times: np.ndarray | None = None
    time_indices: np.ndarray | None = None
    for row in records:
        with np.load(row.path, allow_pickle=False) as product:
            video = np.asarray(product["video"], dtype=np.float32)
            times_s = np.asarray(product["times_s"], dtype=np.float64)
        if video.ndim != 3 or video.shape[0] != times_s.size:
            raise ValueError(f"Incompatible video/time shapes in {row.path}")
        if common_times is None:
            common_times = times_s
            time_indices = _even_indices(times_s.size, time_columns)
        elif not np.array_equal(times_s, common_times):
            raise ValueError(f"Time grid differs in {row.path}")
        if not np.isfinite(video).all() or np.any(video < 0.0):
            raise ValueError(f"Video is non-finite or negative in {row.path}")
        assert time_indices is not None
        videos.append(video[time_indices])
    assert common_times is not None and time_indices is not None
    return np.stack(videos), common_times[time_indices], time_indices


def _plot_element(
    element: str,
    records: list[CachedCondition],
    videos: np.ndarray,
    times_s: np.ndarray,
    *,
    radial_mm: float,
    axial_mm: float,
    display_percentile: float,
) -> tuple[plt.Figure, dict[str, Any]]:
    displayed = np.log10(1.0 + videos.astype(np.float64))
    positive = displayed[displayed > 0.0]
    ceiling = float(np.percentile(positive, display_percentile)) if positive.size else 1.0
    ceiling = max(ceiling, np.finfo(np.float32).eps)
    norm = Normalize(vmin=0.0, vmax=ceiling, clip=True)
    rows, columns = videos.shape[:2]
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.0 * columns + 1.2, 2.05 * rows + 1.0),
        squeeze=False,
        constrained_layout=True,
    )
    zero_panels = 0
    for row_index, record in enumerate(records):
        for column_index, time_s in enumerate(times_s):
            axis = axes[row_index, column_index]
            frame = videos[row_index, column_index]
            axis.imshow(
                displayed[row_index, column_index].T,
                origin="lower",
                aspect="auto",
                extent=(-radial_mm, radial_mm, 0.0, axial_mm),
                cmap="inferno",
                norm=norm,
                interpolation="nearest",
            )
            if not np.any(frame > 0.0):
                zero_panels += 1
                axis.text(
                    0.5,
                    0.5,
                    "zero",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )
            if row_index == 0:
                axis.set_title(f"{time_s * 1e6:.2f} us", fontsize=8)
            if column_index == 0:
                axis.set_ylabel(
                    f"{record.power_wcm / 1e6:g} MW/cm$^2$\nz (mm)", fontsize=7
                )
                axis.set_yticks((0.0, axial_mm / 2.0, axial_mm))
                axis.tick_params(axis="y", labelsize=6)
            else:
                axis.set_yticks([])
            if row_index == rows - 1:
                axis.set_xlabel("x (mm)", fontsize=7)
                axis.set_xticks((-radial_mm, 0.0, radial_mm))
                axis.tick_params(axis="x", labelsize=6)
            else:
                axis.set_xticks([])

    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap="inferno"),
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.82,
        pad=0.01,
    )
    colorbar.set_label(
        "display value: log10(1 + photon radiance)",
        fontsize=9,
    )
    figure.suptitle(
        f"{element}: synthetic ICCD continuum | spot radius {records[0].spot_m * 1e3:g} mm\n"
        f"shared per-element scale clipped at positive p{display_percentile:g}; "
        f"FOV x=+/-{radial_mm:g} mm, z=0--{axial_mm:g} mm",
        fontsize=12,
    )
    metadata = {
        "element": element,
        "spot_radius_mm": records[0].spot_m * 1.0e3,
        "power_density_MW_cm2": [row.power_wcm / 1.0e6 for row in records],
        "times_us": (times_s * 1.0e6).tolist(),
        "display_transform": "log10(1 + photon_radiance)",
        "display_positive_percentile": display_percentile,
        "display_ceiling": ceiling,
        "zero_panels": zero_panels,
        "panel_count": rows * columns,
        "sample_ids": [row.record["sample_id"] for row in records],
        "source_paths": [str(row.path) for row in records],
    }
    return figure, metadata


def main() -> int:
    args = _arguments()
    if args.spot_mm <= 0.0 or args.power_rows < 1 or args.time_columns < 1:
        raise ValueError("Spot size and grid dimensions must be positive")
    if not 0.0 < args.display_percentile <= 100.0:
        raise ValueError("--display-percentile must be in (0, 100]")
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = _read_conditions(cache_dir)
    radial_mm, axial_mm = _field_of_view(cache_dir)
    summary: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "spot_radius_mm": args.spot_mm,
        "power_rows": args.power_rows,
        "time_columns": args.time_columns,
        "radial_max_mm": radial_mm,
        "axial_max_mm": axial_mm,
        "elements": {},
    }
    csv_rows: list[dict[str, Any]] = []
    pdf_path = output_dir / "all_elements_fixed_spot_grids.pdf"
    with PdfPages(pdf_path) as pdf:
        for element in sorted(grouped):
            selected = _select_fixed_spot(
                grouped[element], args.spot_mm * 1.0e-3, args.power_rows
            )
            videos, times_s, time_indices = _load_selected(selected, args.time_columns)
            figure, metadata = _plot_element(
                element,
                selected,
                videos,
                times_s,
                radial_mm=radial_mm,
                axial_mm=axial_mm,
                display_percentile=args.display_percentile,
            )
            png_path = output_dir / f"{element}_fixed_spot_grid.png"
            figure.savefig(png_path, dpi=args.dpi)
            pdf.savefig(figure, dpi=args.dpi)
            plt.close(figure)
            metadata["png"] = png_path.name
            metadata["time_indices"] = time_indices.tolist()
            summary["elements"][element] = metadata
            for row_index, record in enumerate(selected):
                csv_rows.append(
                    {
                        "element": element,
                        "row": row_index,
                        "power_density_W_cm2": record.power_wcm,
                        "power_density_MW_cm2": record.power_wcm / 1.0e6,
                        "spot_radius_mm": record.spot_m * 1.0e3,
                        "sample_id": record.record["sample_id"],
                        "source_path": str(record.path),
                    }
                )
            print(f"Wrote {png_path}", flush=True)

    (output_dir / "plot_selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "plot_selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {pdf_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
