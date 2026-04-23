from plot_style import frame_all_axes
import csv
import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

import fig_window_tradeoff as window_tradeoff


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
    "true": "#1F77B4",
    "fidelity": "#2A9D8F",
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
    dt_raw = manifest.get("config", {}).get("emission", {}).get("dt_ns", 1.0)
    window_raw = manifest.get("config", {}).get("run", {}).get("window_ns", 70.0)
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
            "font.size": 10.4,
            "axes.titlesize": 10.9,
            "axes.labelsize": 10.3,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "grid.color": PALETTE["grid"],
        }
    )


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.015,
        0.975,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
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
                delta_bins.append(a_values[0] - b_values[0])

    if rows_in_slice == 0:
        raise RuntimeError(f"no rows found with window_ns={WINDOW_NS_SLICE} in {TRIALS_PATH}")
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


def _load_tradeoff_summary() -> dict[str, np.ndarray]:
    return window_tradeoff._load_summary_data(window_tradeoff._default_summary_csv())


def main() -> None:
    dt_ns, acceptance_window_ns = _load_timing_config()
    a_bins, b_bins, delta_bins, rows_in_slice = _load_click_distributions()
    summary = _load_tradeoff_summary()

    a_times_ns = a_bins * dt_ns
    b_times_ns = b_bins * dt_ns
    delta_ns = delta_bins * dt_ns
    windows = summary["window_ns"]
    p_true = np.clip(summary["p_t"], 0.0, None)
    fidelity = np.clip(summary["f_t"], 0.0, 1.0)
    false_pct = 100.0 * np.clip(summary["false_frac"], 0.0, 1.0)
    pt11 = np.clip(summary["p_t11"], 0.0, 1.0)

    work_idx = int(np.argmin(np.abs(windows - acceptance_window_ns)))
    window_tradeoff._set_style()
    _set_paper_style()

    fig = plt.figure(figsize=(14.2, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.12], wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax2r = ax2.twinx()

    abs_bins_min = int(np.floor(np.min(np.concatenate([a_bins, b_bins]))))
    abs_bins_max = int(np.ceil(np.max(np.concatenate([a_bins, b_bins]))))
    abs_edges = (np.arange(abs_bins_min, abs_bins_max + 2, 1) - 0.5) * dt_ns
    ax0.hist(
        a_times_ns,
        bins=abs_edges,
        alpha=0.60,
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
    _panel_label(ax0, "(a)")

    delta_bin_min = int(np.floor(np.min(delta_bins)))
    delta_bin_max = int(np.ceil(np.max(delta_bins)))
    delta_edges = (np.arange(delta_bin_min, delta_bin_max + 2, 1) - 0.5) * dt_ns
    window_half = float(abs(acceptance_window_ns))
    ax1.hist(
        delta_ns,
        bins=delta_edges,
        color=PALETTE["delta"],
        alpha=0.88,
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
    ax1.axvline(0.0, color=PALETTE["window"], lw=1.5, ls="--", label=r"$\Delta t=0$")
    ax1.set_title("相对双点击时延（A-B）", pad=5.0)
    ax1.set_xlabel(r"$\Delta t$ (ns)")
    ax1.set_ylabel("计数")
    ax1.legend(frameon=False, loc="upper right")
    _panel_label(ax1, "(b)")

    dense_w = window_tradeoff._make_dense_grid(windows, density=8)
    sigma_x = max(0.8 * float(np.median(np.diff(windows))), 0.45) if windows.size > 1 else 0.45
    p_true_s = window_tradeoff._smooth_to_grid(
        windows,
        p_true,
        dense_w,
        sigma_x,
        lo=np.finfo(float).tiny,
    )
    fidelity_s = window_tradeoff._smooth_to_grid(windows, fidelity, dense_w, sigma_x, lo=0.0, hi=1.0)
    pt11_s = window_tradeoff._smooth_to_grid(windows, pt11, dense_w, sigma_x, lo=0.0, hi=1.0)
    false_pct_s = 100.0 * window_tradeoff._smooth_to_grid(
        windows,
        summary["false_frac"],
        dense_w,
        sigma_x,
        lo=0.0,
        hi=1.0,
    )

    scatter = ax2.scatter(
        windows,
        fidelity,
        c=false_pct,
        cmap="Reds",
        norm=Normalize(vmin=float(np.min(false_pct)), vmax=float(np.max(false_pct) + 0.1)),
        s=34.0,
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )
    ax2.plot(dense_w, fidelity_s, color=PALETTE["fidelity"], lw=2.1, label=r"$F_t$")
    ax2.plot(dense_w, pt11_s, color="#7C3AED", lw=1.8, ls="-.", label=r"$p_{t|11}$")
    ax2.scatter(
        [windows[work_idx]],
        [fidelity[work_idx]],
        marker="*",
        s=150,
        color=PALETTE["window"],
        edgecolors="#111827",
        linewidths=0.7,
        zorder=4,
    )
    ax2.axvline(acceptance_window_ns, color=PALETTE["window"], lw=1.45, ls="--")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("接收窗口 (ns)")
    ax2.set_ylabel(r"$F_t$ 与 $p_{t|11}$")
    ax2.set_title("窗口扫描工作区", pad=5.0)

    ax2r.plot(dense_w, 1e6 * p_true_s, color=PALETTE["true"], lw=1.9, label=r"$p_t$")
    ax2r.plot(dense_w, false_pct_s, color="#B91C1C", lw=1.6, ls="--", label="假成功占比 (%)")
    ax2r.scatter(windows, 1e6 * p_true, s=9, color=PALETTE["true"], alpha=0.20, linewidths=0, zorder=2)
    ax2r.scatter(windows, false_pct, s=9, color="#B91C1C", alpha=0.20, linewidths=0, zorder=2)
    ax2r.set_ylabel(r"$p_t$ ($\times 10^{-6}$) 与假成功占比 (%)")
    ax2r.text(
        float(windows[work_idx]) + 1.2,
        float(fidelity[work_idx]) + 0.035,
        f"推荐口径\n{acceptance_window_ns:.0f} ns",
        fontsize=8.2,
        color="#374151",
        ha="left",
        va="bottom",
    )
    handles0, labels0 = ax2.get_legend_handles_labels()
    handles1, labels1 = ax2r.get_legend_handles_labels()
    ax2.legend(handles0 + handles1, labels0 + labels1, frameon=False, fontsize=8.2, loc="lower left")
    cbar = fig.colorbar(scatter, ax=ax2, fraction=0.052, pad=0.03)
    cbar.set_label("假成功占比 (%)")
    _panel_label(ax2, "(c)")

    fig.suptitle(
        (
            "窗口选择的三联诊断图  "
            f"（切片 window_ns={WINDOW_NS_SLICE:g}, dt={dt_ns:g} ns, rows={rows_in_slice}）"
        ),
        fontsize=12.3,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.14, wspace=0.28)

    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    frame_all_axes(fig)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)

    print(
        "[fig_accept_hist] "
        f"rows={rows_in_slice}, A_clicks={a_times_ns.size}, "
        f"B_clicks={b_times_ns.size}, delta_pairs={delta_ns.size}"
    )


if __name__ == "__main__":
    main()
