from plot_style import frame_all_axes
import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

EXPORT_PNG = False
from matplotlib.colors import LinearSegmentedColormap


PATTERN_ORDER = (
    "pattern_h1v2",
    "pattern_v1h2",
    "pattern_h1v1",
    "pattern_h2v2",
    "pattern_h1h2",
    "pattern_v1v2",
)
PATTERN_LABELS = {
    "pattern_h1v2": "H1V2",
    "pattern_v1h2": "V1H2",
    "pattern_h1v1": "H1V1",
    "pattern_h2v2": "H2V2",
    "pattern_h1h2": "H1H2",
    "pattern_v1v2": "V1V2",
}
PAIR_ORDER = ("H1+V2", "V1+H2", "H1+V1", "H2+V2")
BUCKET_STYLE = {
    "success": {"label": "全部成功记录", "color": "#111827", "lw": 1.8, "ls": "-"},
    "true": {"label": "真成功权重", "color": "#059669", "lw": 2.0, "ls": "-"},
    "false": {"label": "假成功权重", "color": "#dc2626", "lw": 1.8, "ls": "--"},
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _default_summary_dir() -> pathlib.Path:
    data_root = pathlib.Path(__file__).resolve().parents[1] / "data"
    candidates = sorted(
        data_root.glob("bsm_scan_summary_output_*/summary/bsm_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1].parent

    outputs_root = _repo_root() / "outputs"
    candidates = sorted(
        outputs_root.glob("*/summary/bsm_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1].parent

    return data_root / "bsm_scan_summary_output_latest" / "summary"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BSM pattern diagnostics from BSM_SCAN summary CSV files.")
    parser.add_argument("--summary-dir", type=pathlib.Path, default=_default_summary_dir())
    parser.add_argument("--bs-theta", type=float, default=None, help="BS mixing angle to visualize. Default: best fidelity point.")
    return parser.parse_args()


def _read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _parse_float(row: dict[str, str], key: str, line_no: int) -> float:
    raw = (row.get(key) or "").strip()
    if raw == "":
        raise ValueError(f"CSV line {line_no}: missing '{key}'")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"CSV line {line_no}: invalid float '{raw}' for '{key}'") from exc
    if not np.isfinite(value):
        raise ValueError(f"CSV line {line_no}: non-finite '{raw}' for '{key}'")
    return value


def _parse_bool(row: dict[str, str], key: str) -> bool:
    raw = str(row.get(key, "")).strip().lower()
    return raw in ("1", "true", "t", "yes")


def _select_theta(summary_rows: list[dict[str, str]], requested_theta: float | None) -> float:
    theta_vals = sorted({_parse_float(row, "bs_theta", idx) for idx, row in enumerate(summary_rows, start=2)})
    theta_arr = np.asarray(theta_vals, dtype=float)
    if requested_theta is not None:
        idx = int(np.argmin(np.abs(theta_arr - float(requested_theta))))
        if abs(float(theta_arr[idx]) - float(requested_theta)) > 5e-6:
            raise ValueError(f"Requested bs_theta={requested_theta} not found. Available: {theta_vals}")
        return float(theta_arr[idx])

    best_row = max(
        summary_rows,
        key=lambda row: (
            _parse_float(row, "fidelity_true_avg", 0),
            _parse_float(row, "herald_rate_abs", 0),
        ),
    )
    return _parse_float(best_row, "bs_theta", 0)


def _filter_theta_rows(rows: list[dict[str, str]], key: str, theta: float) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for line_no, row in enumerate(rows, start=2):
        value = _parse_float(row, key, line_no)
        if np.isclose(value, theta, atol=5e-6):
            out.append(row)
    if not out:
        raise ValueError(f"No rows found for {key}={theta}")
    return out


def _aggregate_pattern_components(trial_rows: list[dict[str, str]]) -> tuple[np.ndarray, int]:
    components = np.zeros((len(PATTERN_ORDER), 3), dtype=float)
    total_success = 0.0
    pattern_to_idx = {pattern: i for i, pattern in enumerate(PATTERN_ORDER)}
    for line_no, row in enumerate(trial_rows, start=2):
        if not _parse_bool(row, "success"):
            continue
        pattern = str(row.get("pattern", "")).strip()
        idx = pattern_to_idx.get(pattern)
        if idx is None:
            continue
        p_true = _parse_float(row, "p_true_given_record", line_no)
        p_bg = _parse_float(row, "p_bg_assist_given_record", line_no)
        p_intrinsic = _parse_float(row, "p_intrinsic_dark_assist_given_record", line_no)
        components[idx, 0] += max(p_true, 0.0)
        components[idx, 1] += max(p_bg, 0.0)
        components[idx, 2] += max(p_intrinsic, 0.0)
        total_success += 1.0

    if total_success <= 0.0:
        raise ValueError("No successful trial rows found for selected bs_theta")
    return components / total_success, int(round(total_success))


def _build_reliability_matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    deltas = sorted({int(round(_parse_float(row, "delta_bin", idx))) for idx, row in enumerate(rows, start=2)})
    if not deltas:
        raise ValueError("No delta_bin values in reliability heatmap rows")
    delta_to_idx = {delta: i for i, delta in enumerate(deltas)}
    pair_to_idx = {pair: i for i, pair in enumerate(PAIR_ORDER)}
    matrix = np.full((len(PAIR_ORDER), len(deltas)), np.nan, dtype=float)

    for line_no, row in enumerate(rows, start=2):
        pair = str(row.get("pair", "")).strip()
        i = pair_to_idx.get(pair)
        if i is None:
            continue
        delta = int(round(_parse_float(row, "delta_bin", line_no)))
        j = delta_to_idx.get(delta)
        if j is None:
            continue
        matrix[i, j] = float(np.clip(_parse_float(row, "p_true_given_record_avg", line_no), 0.0, 1.0))
    return np.asarray(deltas, dtype=int), matrix


def _build_delta_curves(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    deltas = sorted({int(round(_parse_float(row, "delta_bin", idx))) for idx, row in enumerate(rows, start=2)})
    if not deltas:
        raise ValueError("No delta_bin values in delta distribution rows")
    delta_to_idx = {delta: i for i, delta in enumerate(deltas)}
    curves = {bucket: np.zeros(len(deltas), dtype=float) for bucket in BUCKET_STYLE}

    for line_no, row in enumerate(rows, start=2):
        bucket = str(row.get("bucket", "")).strip()
        if bucket not in curves:
            continue
        delta = int(round(_parse_float(row, "delta_bin", line_no)))
        j = delta_to_idx.get(delta)
        if j is None:
            continue
        curves[bucket][j] = float(np.clip(_parse_float(row, "probability", line_no), 0.0, 1.0))
    return np.asarray(deltas, dtype=int), curves


def main() -> None:
    args = _parse_args()
    summary_dir = pathlib.Path(args.summary_dir)
    summary_rows = _read_csv_rows(summary_dir / "bsm_scan_summary.csv")
    trial_rows = _read_csv_rows(summary_dir / "bsm_scan_trials.csv")
    heatmap_rows = _read_csv_rows(summary_dir / "bsm_scan_record_reliability_heatmap.csv")
    delta_rows = _read_csv_rows(summary_dir / "bsm_scan_delta_bin_distribution.csv")

    theta = _select_theta(summary_rows, args.bs_theta)
    theta_summary = _filter_theta_rows(summary_rows, "bs_theta", theta)[0]
    theta_trials = _filter_theta_rows(trial_rows, "bs_theta", theta)
    theta_heatmap = _filter_theta_rows(heatmap_rows, "bs_theta", theta)
    theta_delta = _filter_theta_rows(delta_rows, "bs_theta", theta)

    pattern_components, success_count = _aggregate_pattern_components(theta_trials)
    heatmap_deltas, reliability = _build_reliability_matrix(theta_heatmap)
    delta_bins, bucket_curves = _build_delta_curves(theta_delta)

    bs_split_ratio = _parse_float(theta_summary, "bs_split_ratio", 0)
    fidelity_true = _parse_float(theta_summary, "fidelity_true_avg", 0)
    false_frac = _parse_float(theta_summary, "false_fraction_global", 0)
    herald_rate = _parse_float(theta_summary, "herald_rate_abs", 0)

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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.75,
        }
    )

    fig = plt.figure(figsize=(15.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.30, 1.16], wspace=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    x = np.arange(len(PATTERN_ORDER))
    y_true = 100.0 * pattern_components[:, 0]
    y_bg = 100.0 * pattern_components[:, 1]
    y_intrinsic = 100.0 * pattern_components[:, 2]
    ax0.bar(x, y_true, width=0.72, color="#1f77b4", label="真成功")
    ax0.bar(x, y_bg, width=0.72, bottom=y_true, color="#f28e2b", label="背景辅助假成功")
    ax0.bar(x, y_intrinsic, width=0.72, bottom=y_true + y_bg, color="#c44e52", label="内禀暗计数假成功")
    ax0.set_xticks(x, [PATTERN_LABELS[p] for p in PATTERN_ORDER])
    ax0.set_ylabel("成功记录占比 (%)")
    ax0.set_xlabel("点击模式")
    ax0.set_title("成功记录的模式组成")
    ax0.legend(frameon=False, fontsize=8.3, loc="upper right")

    masked_reliability = np.ma.masked_invalid(reliability)
    cmap = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#d90429"]).copy()
    cmap.set_bad(color="#f0f0f0", alpha=1.0)
    im = ax1.imshow(
        masked_reliability,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[float(heatmap_deltas.min()) - 0.5, float(heatmap_deltas.max()) + 0.5, -0.5, len(PAIR_ORDER) - 0.5],
        interpolation="nearest",
    )
    ax1.set_yticks(np.arange(len(PAIR_ORDER)), PAIR_ORDER)
    ax1.set_xlabel(r"$|\Delta n|$")
    ax1.set_ylabel("探测器对")
    ax1.set_title(r"真成功记录分布图 $P(\mathrm{genuine}\mid \mathrm{pair},|\Delta n|)$")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.052, pad=0.03)
    cbar.set_label("真成功记录概率")

    for bucket, style in BUCKET_STYLE.items():
        ax2.plot(
            delta_bins,
            100.0 * bucket_curves[bucket],
            color=style["color"],
            lw=style["lw"],
            ls=style["ls"],
            label=style["label"],
        )
    ax2.set_xlabel(r"$|\Delta n|$")
    ax2.set_ylabel("归一化占比 (%)")
    ax2.set_title(r"$|\Delta n|$ 在记录类别上的分布")
    ax2.legend(frameon=False, fontsize=8.4, loc="upper right")

    ax0.text(-0.16, 1.05, "(a)", transform=ax0.transAxes, fontsize=12.5, fontweight="bold", va="bottom", ha="left")
    ax1.text(-0.16, 1.05, "(b)", transform=ax1.transAxes, fontsize=12.5, fontweight="bold", va="bottom", ha="left")
    ax2.text(-0.16, 1.05, "(c)", transform=ax2.transAxes, fontsize=12.5, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(
        (
            "BSM 记录诊断 "
            + rf"$\theta_{{\mathrm{{BS}}}}={theta:.2f}$ rad "
            + rf"($R=\sin^2\theta={bs_split_ratio:.3f}$)："
            + rf"$F_t={fidelity_true:.3f}$，"
            + rf"$p_s={1e4 * herald_rate:.3f}\times10^{{-4}}$，"
            + rf"假成功占比={100.0 * false_frac:.2f}\%，"
            + f"成功记录数={success_count}"
        ),
        fontsize=12.2,
        fontweight="bold",
    )

    out_base = pathlib.Path(__file__).with_suffix("")
    frame_all_axes(fig)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=260)
    if EXPORT_PNG:
        fig.savefig(out_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
