import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

EXPORT_PNG = False


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
    parser = argparse.ArgumentParser(description="Plot real-data error budget from BSM_SCAN summary/trials CSV.")
    parser.add_argument("--summary-dir", type=pathlib.Path, default=_default_summary_dir())
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


def _load_summary(summary_rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    records = []
    for line_no, row in enumerate(summary_rows, start=2):
        records.append(
            (
                _parse_float(row, "bs_theta", line_no),
                _parse_float(row, "herald_rate_abs", line_no),
                _parse_float(row, "p_success_true_abs_avg", line_no),
                _parse_float(row, "p_success_false_abs_avg", line_no),
                _parse_float(row, "fidelity_true_avg", line_no),
                _parse_float(row, "false_fraction_global", line_no),
                _parse_float(row, "chsh_s_max_avg", line_no),
            )
        )
    arr = np.asarray(records, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    return {
        "theta": arr[:, 0],
        "rate_all": arr[:, 1],
        "rate_true": arr[:, 2],
        "rate_false": arr[:, 3],
        "fidelity": arr[:, 4],
        "false_fraction": arr[:, 5],
        "chsh": arr[:, 6],
    }


def _recommended_index(fidelity: np.ndarray, rate: np.ndarray) -> int:
    idx_candidates = np.flatnonzero(np.isclose(fidelity, float(np.max(fidelity)), rtol=0.0, atol=1e-12))
    if idx_candidates.size == 1:
        return int(idx_candidates[0])
    return int(idx_candidates[np.argmax(rate[idx_candidates])])


def _filter_theta_rows(rows: list[dict[str, str]], key: str, theta: float) -> list[dict[str, str]]:
    out = []
    for line_no, row in enumerate(rows, start=2):
        if np.isclose(_parse_float(row, key, line_no), theta, atol=5e-6):
            out.append(row)
    if not out:
        raise ValueError(f"No rows found for {key}={theta}")
    return out


def _false_budget_from_trials(theta_trials: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float], int]:
    false_bg = 0.0
    false_intrinsic = 0.0
    pattern_false_mass = {pattern: 0.0 for pattern in PATTERN_ORDER}
    success_count = 0

    for line_no, row in enumerate(theta_trials, start=2):
        if not _parse_bool(row, "success"):
            continue
        success_count += 1
        pattern = str(row.get("pattern", "")).strip()
        p_bg = _parse_float(row, "p_bg_assist_given_record", line_no)
        p_intrinsic = _parse_float(row, "p_intrinsic_dark_assist_given_record", line_no)
        false_bg += p_bg
        false_intrinsic += p_intrinsic
        if pattern in pattern_false_mass:
            pattern_false_mass[pattern] += (p_bg + p_intrinsic)

    total_false = false_bg + false_intrinsic
    if total_false <= 0.0:
        source_share = {"bg": 0.0, "intrinsic_dark": 0.0}
        pattern_share = {pattern: 0.0 for pattern in PATTERN_ORDER}
    else:
        source_share = {"bg": false_bg / total_false, "intrinsic_dark": false_intrinsic / total_false}
        pattern_share = {pattern: pattern_false_mass[pattern] / total_false for pattern in PATTERN_ORDER}
    return source_share, pattern_share, success_count


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=12.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> None:
    args = _parse_args()
    summary_dir = pathlib.Path(args.summary_dir)
    summary_rows = _read_csv_rows(summary_dir / "bsm_scan_summary.csv")
    trial_rows = _read_csv_rows(summary_dir / "bsm_scan_trials.csv")
    summary = _load_summary(summary_rows)

    theta = summary["theta"]
    rate_true = np.clip(summary["rate_true"], 0.0, None)
    rate_false = np.clip(summary["rate_false"], 0.0, None)
    fidelity = np.clip(summary["fidelity"], 0.0, 1.0)
    false_pct = 100.0 * np.clip(summary["false_fraction"], 0.0, 1.0)
    chsh = summary["chsh"]

    idx_rec = _recommended_index(fidelity, summary["rate_all"])
    theta_rec = float(theta[idx_rec])
    source_share, pattern_share, success_count = _false_budget_from_trials(_filter_theta_rows(trial_rows, "bs_theta", theta_rec))

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

    fig = plt.figure(figsize=(10.8, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.88], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    bar_width = 0.035
    rate_true_1e4 = 1e4 * rate_true
    rate_false_1e4 = 1e4 * rate_false
    ax0.bar(
        theta,
        rate_false_1e4,
        width=bar_width,
        color="#ef4444",
        alpha=0.36,
        edgecolor="none",
        zorder=1,
        label=r"假成功分量 $p_f$",
    )
    ax0.plot(
        theta,
        rate_true_1e4,
        color="#1f77b4",
        lw=2.1,
        marker="o",
        ms=4.1,
        zorder=3,
        label=r"真成功分量 $p_t$",
    )
    ax0.set_xlabel(r"BS 混合角 $\theta_{BS}$ (rad)")
    ax0.set_ylabel(r"每次尝试概率 ($\times 10^{-4}$)")
    ax0.set_title("事件概率预算随分束器角度变化")

    ax0_r = ax0.twinx()
    ax0_r.plot(theta, fidelity, color="#047857", lw=2.0, marker="o", ms=4.2, label=r"$F_t$")
    ax0_r.plot(theta, false_pct, color="#991b1b", lw=1.6, marker="s", ms=3.8, ls="--", label="假成功占比 (%)")
    ax0_r.plot(theta, chsh / 4.0, color="#111827", lw=1.3, marker="^", ms=3.2, ls="-.", label=r"$S_{\max}/4$")
    ax0_r.set_ylabel(r"质量轴：$F_t$、假成功占比 (\%)、$S_{\max}/4$")
    ax0.axvline(theta_rec, color="#9ca3af", lw=1.0, ls=":")

    # Inset: rate-fidelity operating-point frontier view (merged from old standalone figure).
    inset = inset_axes(ax0, width="44%", height="44%", loc="lower right", borderpad=1.0)
    rate_all_1e4 = 1e4 * np.clip(summary["rate_all"], np.finfo(float).tiny, None)
    inset_sc = inset.scatter(
        rate_all_1e4,
        fidelity,
        c=false_pct,
        s=32.0,
        cmap="Reds",
        alpha=0.92,
        edgecolors="#f8fafc",
        linewidths=0.45,
        zorder=2,
    )
    inset.scatter(
        [rate_all_1e4[idx_rec]],
        [fidelity[idx_rec]],
        marker="*",
        s=86.0,
        color="#f97316",
        edgecolors="#0f172a",
        linewidths=0.6,
        zorder=3,
    )
    inset.set_title("速率-保真度插图", fontsize=7.2)
    inset.set_xlabel(r"$p_s$ ($\times 10^{-4}$)", fontsize=6.8)
    inset.set_ylabel(r"$F_t$", fontsize=6.8)
    inset.tick_params(labelsize=6.4, length=2.2, pad=1.5)
    inset.grid(True, alpha=0.18, linewidth=0.6)
    cb_inset = fig.colorbar(inset_sc, ax=inset, fraction=0.16, pad=0.02)
    cb_inset.set_label("假成功 (%)", fontsize=6.4)
    cb_inset.ax.tick_params(labelsize=6.2, length=2.0)

    lines = []
    labels = []
    for axis in (ax0, ax0_r):
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    ax0.legend(lines, labels, frameon=False, fontsize=8.3, loc="upper left")
    _panel_label(ax0, "(a)")

    ax1.set_xlim(0.0, 100.0)
    ax1.set_ylim(-0.8, 1.8)
    ax1.set_yticks([1.0, 0.0], ["来源组成", "模式组成"])
    ax1.set_xlabel("假成功总量内占比 (%)")
    ax1.set_title(rf"$\theta_{{BS}}={theta_rec:.2f}$ rad 下假成功组成")

    source_items = [
        ("背景辅助", source_share["bg"], "#f59e0b"),
        ("内禀暗计数辅助", source_share["intrinsic_dark"], "#dc2626"),
    ]
    left = 0.0
    for key, frac, color in source_items:
        width = 100.0 * max(frac, 0.0)
        ax1.barh([1.0], [width], left=left, height=0.44, color=color, edgecolor="#111827", linewidth=0.8, label=key)
        if width >= 8.0:
            ax1.text(left + width / 2.0, 1.0, f"{width:.1f}%", ha="center", va="center", fontsize=8.6, color="white")
        left += width

    pattern_colors = {
        "pattern_h1v2": "#1f77b4",
        "pattern_v1h2": "#2ca02c",
        "pattern_h1v1": "#9467bd",
        "pattern_h2v2": "#8c564b",
        "pattern_h1h2": "#17becf",
        "pattern_v1v2": "#bcbd22",
    }
    left = 0.0
    for pattern in PATTERN_ORDER:
        width = 100.0 * max(pattern_share[pattern], 0.0)
        ax1.barh(
            [0.0],
            [width],
            left=left,
            height=0.44,
            color=pattern_colors[pattern],
            edgecolor="#111827",
            linewidth=0.6,
            label=PATTERN_LABELS[pattern],
        )
        if width >= 7.0:
            ax1.text(left + width / 2.0, 0.0, f"{width:.1f}%", ha="center", va="center", fontsize=7.8, color="white")
        left += width

    handles, labels = ax1.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax1.legend(unique.values(), unique.keys(), frameon=False, fontsize=7.8, loc="lower right", ncol=2)
    _panel_label(ax1, "(b)")

    fig.suptitle(
        (
            "真实 BSM 扫描的误差预算视图："
            + rf"$\theta_{{BS}}={theta_rec:.2f}$ rad，"
            + rf"$F_t={fidelity[idx_rec]:.3f}$，"
            + rf"$p_t={1e4 * rate_true[idx_rec]:.3f}\times10^{{-4}}$，"
            + rf"$p_f={1e4 * rate_false[idx_rec]:.4f}\times10^{{-4}}$，"
            + f"成功记录数={success_count}"
        ),
        fontsize=12.1,
        fontweight="bold",
    )

    out_base = pathlib.Path(__file__).with_suffix("")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=260)
    if EXPORT_PNG:
        fig.savefig(out_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

