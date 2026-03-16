import argparse
import csv
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


Z95 = 1.96
METRICS = ("p_s", "p_t", "p_f", "f_t", "false_frac", "p_t11")
SUMMARY_COLUMNS = {
    "window_ns": "window_ns",
    "p_s": "herald_rate_abs",
    "p_t": "p_success_true_abs_avg",
    "p_f": "p_success_false_abs_avg",
    "f_t": "fidelity_true_avg",
    "false_frac": "false_fraction_global",
}
RUNS_COLUMNS = {
    "p_s": "p_success_abs",
    "p_t": "p_success_true_abs",
    "p_f": "p_success_false_abs",
    "f_t": "fidelity_true",
    "false_frac": "false_fraction",
    "p_t11": "p_success_true_given_arrival",
}
PT11_CANDIDATES = (
    "p_success_true_given_arrival11_global",
)


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def sem(self) -> float:
        if self.n < 2:
            return float("nan")
        variance = self.m2 / (self.n - 1)
        return float(np.sqrt(variance / self.n))


def _default_summary_csv() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "data"
        / "window_scan_server_output_20260221_0421"
        / "summary"
        / "window_scan_summary.csv"
    )


def _default_runs_csv() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "data"
        / "window_scan_server_output_20260221_0421"
        / "summary"
        / "window_scan_runs.csv"
    )


def _window_key(value: float) -> float:
    return round(float(value), 6)


def _parse_float(row: dict[str, str], col: str, line_no: int) -> float:
    raw = (row.get(col) or "").strip()
    if raw == "":
        raise ValueError(f"Empty value for '{col}' at CSV line {line_no}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for '{col}' at CSV line {line_no}: '{raw}'") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite value for '{col}' at CSV line {line_no}: '{raw}'")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot window tradeoff using real summary/runs data.")
    parser.add_argument("--summary-csv", type=pathlib.Path, default=_default_summary_csv())
    parser.add_argument("--runs-csv", type=pathlib.Path, default=_default_runs_csv())
    parser.add_argument("--ci-cache", type=pathlib.Path, default=None)
    parser.add_argument("--work-point-ns", type=float, default=70.0)
    parser.add_argument("--no-ci", action="store_true", help="Disable run-level confidence intervals.")
    parser.add_argument(
        "--force-recompute-ci",
        action="store_true",
        help="Recompute CI from runs CSV even if cache exists.",
    )
    return parser.parse_args()


def _load_summary_data(csv_path: pathlib.Path) -> dict[str, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        missing = [col for col in SUMMARY_COLUMNS.values() if col not in fieldnames]
        if missing:
            raise ValueError(f"Summary CSV missing required columns: {', '.join(missing)}")

        pt11_col = next((col for col in PT11_CANDIDATES if col in fieldnames), None)
        if pt11_col is None:
            raise ValueError(
                "Summary CSV missing p_t|11 column; expected one of: "
                + ", ".join(PT11_CANDIDATES)
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"Summary CSV has no rows: {csv_path}")

    parsed: dict[str, list[float]] = {k: [] for k in SUMMARY_COLUMNS}
    parsed["p_t11"] = []

    for idx, row in enumerate(rows, start=2):
        for key, col in SUMMARY_COLUMNS.items():
            parsed[key].append(_parse_float(row, col, idx))
        parsed["p_t11"].append(_parse_float(row, pt11_col, idx))

    data = {key: np.asarray(values, dtype=float) for key, values in parsed.items()}
    order = np.argsort(data["window_ns"])
    return {key: values[order] for key, values in data.items()}


def _load_ci_cache(cache_path: pathlib.Path, windows: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            cache_windows = cache["window_ns"]
            if cache_windows.shape != windows.shape or not np.allclose(cache_windows, windows):
                return None
            ci: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for metric in METRICS:
                lo_key = f"{metric}_lo"
                hi_key = f"{metric}_hi"
                if lo_key not in cache or hi_key not in cache:
                    return None
                ci[metric] = (np.asarray(cache[lo_key], dtype=float), np.asarray(cache[hi_key], dtype=float))
            return ci
    except Exception:
        return None


def _save_ci_cache(cache_path: pathlib.Path, windows: np.ndarray, ci: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {"window_ns": windows}
    for metric in METRICS:
        lo, hi = ci[metric]
        payload[f"{metric}_lo"] = lo
        payload[f"{metric}_hi"] = hi
    np.savez(cache_path, **payload)


def _compute_run_ci(
    runs_csv: pathlib.Path,
    windows: np.ndarray,
    summary: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not runs_csv.exists():
        raise FileNotFoundError(f"Runs CSV not found for CI computation: {runs_csv}")

    stats_by_window: dict[float, dict[str, RunningStats]] = {}
    with runs_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        line_no = 1
        for row in reader:
            line_no += 1
            if line_no % 2_000_000 == 0:
                print(f"[fig_window_tradeoff] processed {line_no:,} run rows...")

            try:
                w = _window_key(_parse_float(row, "window_ns", line_no))
            except ValueError:
                continue

            bucket = stats_by_window.setdefault(
                w,
                {metric: RunningStats() for metric in METRICS},
            )
            for metric, col in RUNS_COLUMNS.items():
                raw = (row.get(col) or "").strip()
                if raw == "":
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if np.isfinite(value):
                    bucket[metric].update(value)

    ci: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    tiny = np.finfo(float).tiny
    unit_metrics = {"f_t", "false_frac", "p_t11"}

    for metric in METRICS:
        line_values = summary[metric]
        lo = np.empty_like(line_values)
        hi = np.empty_like(line_values)

        for i, w in enumerate(windows):
            bucket = stats_by_window.get(_window_key(w))
            if bucket is None or bucket[metric].n == 0:
                mean = float(line_values[i])
                delta = 0.0
            else:
                mean = bucket[metric].mean
                sem = bucket[metric].sem()
                delta = 0.0 if not np.isfinite(sem) else Z95 * sem

            low_val = mean - delta
            high_val = mean + delta

            if metric in unit_metrics:
                low_val = min(max(low_val, 0.0), 1.0)
                high_val = min(max(high_val, 0.0), 1.0)
            else:
                low_val = max(low_val, tiny)
                high_val = max(high_val, tiny)

            if high_val < low_val:
                high_val = low_val

            lo[i] = low_val
            hi[i] = high_val

        ci[metric] = (lo, hi)

    return ci


def _load_or_compute_ci(
    runs_csv: pathlib.Path,
    windows: np.ndarray,
    summary: dict[str, np.ndarray],
    cache_path: pathlib.Path | None,
    force_recompute: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if cache_path is None:
        cache_path = runs_csv.with_name("window_scan_ci_cache.npz")

    if (
        not force_recompute
        and cache_path.exists()
        and runs_csv.exists()
        and cache_path.stat().st_mtime >= runs_csv.stat().st_mtime
    ):
        cached = _load_ci_cache(cache_path, windows)
        if cached is not None:
            print(f"[fig_window_tradeoff] loaded CI cache: {cache_path}")
            return cached

    print(f"[fig_window_tradeoff] computing CI from runs: {runs_csv}")
    ci = _compute_run_ci(runs_csv, windows, summary)
    try:
        _save_ci_cache(cache_path, windows, ci)
        print(f"[fig_window_tradeoff] wrote CI cache: {cache_path}")
    except Exception as exc:
        print(f"[fig_window_tradeoff] warning: failed to write CI cache ({exc})")
    return ci


def _set_style() -> None:
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
            "axes.labelsize": 10.4,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.65,
        }
    )


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.015,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.3},
    )


def _line_handles_labels(*axes: plt.Axes) -> tuple[list, list[str]]:
    handles: list = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in axes:
        handles_raw, labels_raw = ax.get_legend_handles_labels()
        for hh, ll in zip(handles_raw, labels_raw):
            if ll in seen:
                continue
            seen.add(ll)
            handles.append(hh)
            labels.append(ll)
    return handles, labels


def _ci_fill(
    ax: plt.Axes,
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    scale: float,
    color: str,
    alpha: float,
) -> None:
    if np.allclose(lo, hi):
        return
    ax.fill_between(x, scale * lo, scale * hi, color=color, alpha=alpha, linewidth=0.0)


def _make_dense_grid(x: np.ndarray, density: int = 6) -> np.ndarray:
    if x.size < 2:
        return x.copy()
    n_pts = max(int((x.size - 1) * density) + 1, x.size)
    return np.linspace(float(x[0]), float(x[-1]), n_pts)


def _gaussian_smooth(x: np.ndarray, y: np.ndarray, sigma_x: float) -> np.ndarray:
    if x.size < 3 or sigma_x <= 0.0:
        return y.copy()
    out = np.empty_like(y, dtype=float)
    for i, xi in enumerate(x):
        dx = (x - xi) / sigma_x
        w = np.exp(-0.5 * dx * dx)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0 or not np.isfinite(w_sum):
            out[i] = y[i]
        else:
            out[i] = float(np.dot(w, y) / w_sum)
    return out


def _smooth_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_dense: np.ndarray,
    sigma_x: float,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    y_s = _gaussian_smooth(x, y, sigma_x)
    y_dense = np.interp(x_dense, x, y_s)
    if lo is not None:
        y_dense = np.maximum(y_dense, lo)
    if hi is not None:
        y_dense = np.minimum(y_dense, hi)
    return y_dense


def main() -> None:
    args = _parse_args()
    summary = _load_summary_data(args.summary_csv)
    w = summary["window_ns"]

    ci: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    if not args.no_ci:
        ci = _load_or_compute_ci(
            runs_csv=args.runs_csv,
            windows=w,
            summary=summary,
            cache_path=args.ci_cache,
            force_recompute=args.force_recompute_ci,
        )

    _set_style()

    fig = plt.figure(figsize=(12.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.08], wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax1r = ax1.twinx()

    c_true = "#1f77b4"
    c_false = "#c44e52"
    c_all = "#2f2f2f"
    c_fidelity = "#2a9d8f"
    c_pt11 = "#9467bd"
    c_work = "#f2c14e"

    tiny = np.finfo(float).tiny
    p_s = np.clip(summary["p_s"], tiny, None)
    p_t = np.clip(summary["p_t"], tiny, None)
    p_f = np.clip(summary["p_f"], tiny, None)
    f_t = np.clip(summary["f_t"], 0.0, 1.0)
    false_frac = np.clip(summary["false_frac"], 0.0, 1.0)
    p_t11 = np.clip(summary["p_t11"], 0.0, 1.0)
    base_step = float(np.median(np.diff(w))) if w.size > 1 else 1.0
    smooth_sigma = max(0.8 * base_step, 0.45)
    w_plot = _make_dense_grid(w, density=8)

    p_s_plot = _smooth_to_grid(w, p_s, w_plot, smooth_sigma, lo=tiny)
    p_t_plot = _smooth_to_grid(w, p_t, w_plot, smooth_sigma, lo=tiny)
    p_f_plot = _smooth_to_grid(w, p_f, w_plot, smooth_sigma, lo=tiny)
    p_s_plot = np.maximum(p_s_plot, p_t_plot)
    f_t_plot = _smooth_to_grid(w, f_t, w_plot, smooth_sigma, lo=0.0, hi=1.0)
    p_t11_plot = _smooth_to_grid(w, p_t11, w_plot, smooth_sigma, lo=0.0, hi=1.0)
    false_frac_plot = _smooth_to_grid(w, false_frac, w_plot, smooth_sigma, lo=0.0, hi=1.0)
    false_pct = 100.0 * false_frac
    false_pct_plot = 100.0 * false_frac_plot

    ax0.plot(w_plot, 1e6 * p_s_plot, color=c_all, lw=2.25, label=r"$p_s$：成功记录")
    ax0.plot(w_plot, 1e6 * p_t_plot, color=c_true, lw=2.2, label=r"$p_t$：真成功分量")
    ax0.plot(w_plot, 1e6 * p_f_plot, color=c_false, lw=2.0, ls="--", label=r"$p_f$：假成功分量")
    ax0.scatter(w, 1e6 * p_s, s=8, color=c_all, alpha=0.18, linewidths=0, zorder=3)
    ax0.scatter(w, 1e6 * p_t, s=8, color=c_true, alpha=0.18, linewidths=0, zorder=3)
    ax0.scatter(w, 1e6 * p_f, s=8, color=c_false, alpha=0.18, linewidths=0, zorder=3)

    if ci is not None:
        ps_lo = _smooth_to_grid(w, ci["p_s"][0], w_plot, smooth_sigma, lo=tiny)
        ps_hi = _smooth_to_grid(w, ci["p_s"][1], w_plot, smooth_sigma, lo=tiny)
        pt_lo = _smooth_to_grid(w, ci["p_t"][0], w_plot, smooth_sigma, lo=tiny)
        pt_hi = _smooth_to_grid(w, ci["p_t"][1], w_plot, smooth_sigma, lo=tiny)
        pf_lo = _smooth_to_grid(w, ci["p_f"][0], w_plot, smooth_sigma, lo=tiny)
        pf_hi = _smooth_to_grid(w, ci["p_f"][1], w_plot, smooth_sigma, lo=tiny)
        _ci_fill(ax0, w_plot, np.minimum(ps_lo, ps_hi), np.maximum(ps_lo, ps_hi), 1e6, c_all, 0.10)
        _ci_fill(ax0, w_plot, np.minimum(pt_lo, pt_hi), np.maximum(pt_lo, pt_hi), 1e6, c_true, 0.10)
        _ci_fill(ax0, w_plot, np.minimum(pf_lo, pf_hi), np.maximum(pf_lo, pf_hi), 1e6, c_false, 0.11)

    ax0.fill_between(w_plot, 1e6 * p_t_plot, 1e6 * p_s_plot, color=c_false, alpha=0.10)
    ax0.axvline(args.work_point_ns, color=c_work, lw=1.7, ls="--")
    ax0.set_yscale("log")
    ax0.set_xlabel("接收窗口 (ns)")
    ax0.set_ylabel(r"每次尝试概率 ($\times 10^{-6}$)")
    ax0.set_title("速率分解随接收窗口变化")
    ax0.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
    )
    _panel_label(ax0, "(a)")

    ax1.plot(
        w_plot,
        f_t_plot,
        color=c_fidelity,
        lw=2.25,
        label=r"$F_t$：条件保真度（真成功分量）",
    )
    ax1.plot(
        w_plot,
        p_t11_plot,
        color=c_pt11,
        lw=2.05,
        ls="-.",
        label=r"$p_{t|11}$：双光子到达条件真成功",
    )
    ax1.scatter(w, f_t, s=8, color=c_fidelity, alpha=0.18, linewidths=0, zorder=3)
    ax1.scatter(w, p_t11, s=8, color=c_pt11, alpha=0.18, linewidths=0, zorder=3)

    ax1r.plot(w_plot, false_pct_plot, color="#d62728", lw=2.1, label=r"$p_f/p_s$（假成功占比，右轴，%）")
    ax1r.scatter(w, false_pct, s=8, color="#d62728", alpha=0.18, linewidths=0, zorder=3)

    if ci is not None:
        ft_lo = _smooth_to_grid(w, ci["f_t"][0], w_plot, smooth_sigma, lo=0.0, hi=1.0)
        ft_hi = _smooth_to_grid(w, ci["f_t"][1], w_plot, smooth_sigma, lo=0.0, hi=1.0)
        pt11_lo = _smooth_to_grid(w, ci["p_t11"][0], w_plot, smooth_sigma, lo=0.0, hi=1.0)
        pt11_hi = _smooth_to_grid(w, ci["p_t11"][1], w_plot, smooth_sigma, lo=0.0, hi=1.0)
        ff_lo_ci = _smooth_to_grid(w, ci["false_frac"][0], w_plot, smooth_sigma, lo=0.0, hi=1.0)
        ff_hi_ci = _smooth_to_grid(w, ci["false_frac"][1], w_plot, smooth_sigma, lo=0.0, hi=1.0)
        _ci_fill(ax1, w_plot, np.minimum(ft_lo, ft_hi), np.maximum(ft_lo, ft_hi), 1.0, c_fidelity, 0.10)
        _ci_fill(ax1, w_plot, np.minimum(pt11_lo, pt11_hi), np.maximum(pt11_lo, pt11_hi), 1.0, c_pt11, 0.10)
        _ci_fill(
            ax1r,
            w_plot,
            np.minimum(ff_lo_ci, ff_hi_ci),
            np.maximum(ff_lo_ci, ff_hi_ci),
            100.0,
            "#d62728",
            0.12,
        )

    ax1.axvline(args.work_point_ns, color=c_work, lw=1.7, ls="--", label=f"{args.work_point_ns:g} ns")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("接收窗口 (ns)")
    ax1.set_ylabel(r"$F_t$ 与 $p_{t|11}$")

    ff_min = float(np.min(false_frac_plot))
    ff_max = float(np.max(false_frac_plot))
    ff_pad = max(5e-5, 0.12 * (ff_max - ff_min))
    ff_lo = max(0.0, ff_min - ff_pad)
    ff_hi = ff_max + ff_pad
    ax1r.set_ylim(100.0 * ff_lo, 100.0 * ff_hi)
    ax1r.set_ylabel(r"假成功占比 $p_f/p_s$ (%)", color="#d62728")
    ax1r.tick_params(axis="y", colors="#d62728")
    ax1r.spines["top"].set_visible(False)

    idx_ff_min = int(np.argmin(false_frac_plot))
    w_min = float(w_plot[idx_ff_min])
    ff_min_pct = float(false_pct_plot[idx_ff_min])
    ax1r.scatter([w_min], [ff_min_pct], s=20, color="#d62728", zorder=5)
    ax1r.annotate(
        f"局部最小值：{ff_min_pct:.3f}% @ {w_min:g} ns",
        xy=(w_min, ff_min_pct),
        xytext=(w_min + 8.0, ff_min_pct + 0.02),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 0.9},
        fontsize=8.2,
        color="#9a1b1b",
    )

    ax1.set_title("质量指标与假成功占比演化")
    legend_handles, legend_labels = _line_handles_labels(ax1, ax1r)
    ax1.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
    )
    _panel_label(ax1, "(b)")

    y_work = float(np.interp(args.work_point_ns, w_plot, f_t_plot))
    ax1.annotate(
        "推荐工作点",
        xy=(args.work_point_ns, y_work),
        xytext=(args.work_point_ns + 14.0, min(0.96, y_work + 0.15)),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
        fontsize=8.7,
    )

    fig.suptitle(
        "接收窗口权衡：事件速率、条件保真度与假成功占比",
        fontsize=11.6,
        fontweight="bold",
    )
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
