import csv
import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "data"
    / "window_scan_server_output_20260221_0421"
    / "summary"
)
TRIALS_PATH = DATA_DIR / "window_scan_trials.csv"
MANIFEST_PATH = DATA_DIR / "run_manifest.json"
WINDOW_NS_SLICE = 100.0
PALETTE = {
    "arm_a": "#1F77B4",
    "arm_b": "#D62728",
    "delta": "#2F2F2F",
    "window": "#F2C14E",
    "grid": "#D9DEE7",
}


def _parse_optional_bin(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return int(float(text))


def _load_timing_config() -> tuple[float, float]:
    dt_ns = 1.0
    window_ns = 70.0
    if not MANIFEST_PATH.exists():
        print(
            f"[fig_accept_hist] warning: manifest not found: {MANIFEST_PATH}, "
            "fallback dt_ns=1.0, window_ns=70.0"
        )
        return dt_ns, window_ns

    with MANIFEST_PATH.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    dt_raw = (
        manifest.get("config", {})
        .get("emission", {})
        .get("dt_ns", 1.0)
    )
    window_raw = (
        manifest.get("config", {})
        .get("run", {})
        .get("window_ns", 70.0)
    )
    try:
        dt_ns = float(dt_raw)
    except (TypeError, ValueError):
        print(f"[fig_accept_hist] warning: invalid dt_ns={dt_raw!r}, fallback dt_ns=1.0")
    try:
        window_ns = float(window_raw)
    except (TypeError, ValueError):
        print(f"[fig_accept_hist] warning: invalid window_ns={window_raw!r}, fallback window_ns=70.0")
    return dt_ns, window_ns


def _set_paper_style() -> None:
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
            "font.size": 10.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 9.2,
            "axes.linewidth": 0.85,
            "axes.grid": True,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "grid.color": PALETTE["grid"],
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.015,
        0.975,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.6},
    )


def _load_click_distributions() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not TRIALS_PATH.exists():
        raise FileNotFoundError(f"window scan trials file not found: {TRIALS_PATH}")

    a_bins_abs: list[int] = []
    b_bins_abs: list[int] = []
    delta_bins: list[int] = []
    rows_in_slice = 0

    with TRIALS_PATH.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                window_ns = float(row.get("window_ns", ""))
            except (TypeError, ValueError):
                continue
            if not math.isclose(window_ns, WINDOW_NS_SLICE, rel_tol=0.0, abs_tol=1e-9):
                continue

            rows_in_slice += 1

            a_candidates = [
                _parse_optional_bin(row.get("H1_bin")),
                _parse_optional_bin(row.get("V1_bin")),
            ]
            b_candidates = [
                _parse_optional_bin(row.get("H2_bin")),
                _parse_optional_bin(row.get("V2_bin")),
            ]

            a_values = [x for x in a_candidates if x is not None]
            b_values = [x for x in b_candidates if x is not None]

            a_bins_abs.extend(a_values)
            b_bins_abs.extend(b_values)

            if a_values and b_values:
                # Use one relative delay per shot (first available click on each arm).
                delta_bins.append(a_values[0] - b_values[0])

    if rows_in_slice == 0:
        raise RuntimeError(
            f"no rows found with window_ns={WINDOW_NS_SLICE} in {TRIALS_PATH}"
        )
    if not a_bins_abs or not b_bins_abs:
        raise RuntimeError("no valid click bins extracted for A or B arm")
    if not delta_bins:
        raise RuntimeError("no valid cross-arm click pairs for relative delay histogram")

    return (
        np.asarray(a_bins_abs, dtype=float),
        np.asarray(b_bins_abs, dtype=float),
        np.asarray(delta_bins, dtype=float),
        rows_in_slice,
    )


def main() -> None:
    dt_ns, acceptance_window_ns = _load_timing_config()
    a_bins, b_bins, delta_bins, rows_in_slice = _load_click_distributions()

    a_times_ns = a_bins * dt_ns
    b_times_ns = b_bins * dt_ns
    delta_ns = delta_bins * dt_ns

    _set_paper_style()

    fig = plt.figure(figsize=(10.6, 4.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    abs_bins_min = int(np.floor(np.min(np.concatenate([a_bins, b_bins]))))
    abs_bins_max = int(np.ceil(np.max(np.concatenate([a_bins, b_bins]))))
    abs_edges = (np.arange(abs_bins_min, abs_bins_max + 2, 1) - 0.5) * dt_ns

    ax0.hist(
        a_times_ns,
        bins=abs_edges,
        alpha=0.58,
        color=PALETTE["arm_a"],
        edgecolor="none",
        label=f"A臂 (N={a_times_ns.size})",
    )
    ax0.hist(
        b_times_ns,
        bins=abs_edges,
        alpha=0.52,
        color=PALETTE["arm_b"],
        edgecolor="none",
        label=f"B臂 (N={b_times_ns.size})",
    )
    ax0.set_title("分臂绝对点击时间分布", pad=5.0)
    ax0.set_xlabel("点击时间 (ns)")
    ax0.set_ylabel("计数")
    ax0.legend(frameon=False, loc="upper right")
    _style_axis(ax0)
    _panel_label(ax0, "(a)")

    delta_bin_min = int(np.floor(np.min(delta_bins)))
    delta_bin_max = int(np.ceil(np.max(delta_bins)))
    delta_edges = (np.arange(delta_bin_min, delta_bin_max + 2, 1) - 0.5) * dt_ns
    window_half = float(abs(acceptance_window_ns))

    ax1.hist(
        delta_ns,
        bins=delta_edges,
        color=PALETTE["delta"],
        alpha=0.86,
        edgecolor="none",
        label=f"成对记录 (N={delta_ns.size})",
    )
    ax1.axvspan(
        -window_half,
        window_half,
        color=PALETTE["window"],
        alpha=0.20,
        linewidth=0.0,
        label=rf"接收窗口（$|\Delta t| \leq {window_half:g}\,\mathrm{{ns}}$）",
        zorder=0,
    )
    ax1.axvline(0.0, color=PALETTE["window"], lw=1.6, ls="--", label=r"$\Delta t=0$")
    ax1.set_title("相对双点击时延（A-B）", pad=5.0)
    ax1.set_xlabel(r"$\Delta t$ (ns)")
    ax1.set_ylabel("计数")
    ax1.legend(frameon=False, loc="upper right")
    _style_axis(ax1)
    _panel_label(ax1, "(b)")

    fig.suptitle(
        (
            "真实数据窗口扫描时序统计  "
            f"（切片 window_ns={WINDOW_NS_SLICE:g}, dt={dt_ns:g} ns, rows={rows_in_slice}）"
        ),
        fontsize=12.4,
        fontweight="bold",
    )

    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(
        "[fig_accept_hist] "
        f"rows={rows_in_slice}, A_clicks={a_times_ns.size}, "
        f"B_clicks={b_times_ns.size}, delta_pairs={delta_ns.size}"
    )


if __name__ == "__main__":
    main()
