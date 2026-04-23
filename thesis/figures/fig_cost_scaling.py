import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from plot_style import frame_all_axes

EXPORT_PNG = True

DEFAULT_SUMMARY_CSV = (
    pathlib.Path(__file__).resolve().parents[1]
    / "data"
    / "cost_scan_dt_bins_20260318_0433"
    / "summary"
    / "param_scan_summary.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot runtime scaling vs n_bins for multiple dt values.")
    parser.add_argument("--summary-csv", type=pathlib.Path, default=DEFAULT_SUMMARY_CSV)
    return parser.parse_args()


def _parse_float(row: dict[str, str], key: str, line_no: int) -> float:
    raw = (row.get(key) or "").strip()
    if raw == "":
        raise ValueError(f"CSV line {line_no}: missing value for '{key}'")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"CSV line {line_no}: invalid float '{raw}' for '{key}'") from exc
    if not np.isfinite(value):
        raise ValueError(f"CSV line {line_no}: non-finite '{raw}' for '{key}'")
    return value


def _load_runtime_by_dt(summary_csv: pathlib.Path) -> dict[float, dict[str, np.ndarray]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = ("dt_ns", "n_bins", "runtime_wall_s_avg")
    missing = [key for key in required if key not in fieldnames]
    if missing:
        raise ValueError(f"Summary CSV missing columns: {', '.join(missing)}")
    if not rows:
        raise ValueError(f"Summary CSV has no rows: {summary_csv}")

    records: list[tuple[float, float, float]] = []
    for line_no, row in enumerate(rows, start=2):
        dt_ns = _parse_float(row, "dt_ns", line_no)
        n_bins = _parse_float(row, "n_bins", line_no)
        runtime = _parse_float(row, "runtime_wall_s_avg", line_no)
        records.append((round(dt_ns, 9), n_bins, runtime))

    dt_groups: dict[float, list[tuple[float, float]]] = {}
    for dt_ns, n_bins, runtime in records:
        dt_groups.setdefault(dt_ns, []).append((n_bins, runtime))

    result: dict[float, dict[str, np.ndarray]] = {}
    for dt_ns, items in dt_groups.items():
        arr = np.asarray(items, dtype=float)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        result[dt_ns] = {"n_bins": arr[:, 0], "runtime": arr[:, 1]}
    return result


def _ordered_dt_list(dts: list[float]) -> list[float]:
    preferred = [0.5, 1.0, 1.5, 2.0]
    ordered = [dt for dt in preferred if dt in dts]
    for dt in sorted(dts):
        if dt not in ordered:
            ordered.append(dt)
    return ordered


def main() -> None:
    args = _parse_args()
    data_by_dt = _load_runtime_by_dt(args.summary_csv)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.95,
            "axes.grid": True,
            "grid.alpha": 0.20,
        }
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.90)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]

    ordered_dts = _ordered_dt_list(list(data_by_dt.keys()))
    for idx, dt_ns in enumerate(ordered_dts):
        series = data_by_dt[dt_ns]
        ax.plot(
            series["n_bins"],
            series["runtime"],
            label=f"dt={dt_ns:g} ns",
            color=palette[idx % len(palette)],
            marker=markers[idx % len(markers)],
            lw=2.2,
            ms=6.0,
        )

    ax.set_xlabel("时间 bin 数")
    ax.set_ylabel("单次仿真用时 (s)")
    ax.set_title("不同时间步长下的单次仿真用时")
    ax.legend(loc="best", frameon=False, title="时间步长")

    base = pathlib.Path(__file__).with_suffix("")
    frame_all_axes(fig)
    fig.savefig(base.with_suffix(".pdf"), dpi=240)
    if EXPORT_PNG:
        fig.savefig(base.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
