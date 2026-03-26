import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from plot_style import frame_all_axes

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
    parser = argparse.ArgumentParser(description="Plot error-budget figure from BSM_SCAN summary/trials CSV.")
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


def _false_contribution_matrix_from_trials(theta_trials: list[dict[str, str]]) -> tuple[np.ndarray, int]:
    matrix = np.zeros((2, len(PATTERN_ORDER)), dtype=float)
    pattern_index = {pattern: idx for idx, pattern in enumerate(PATTERN_ORDER)}
    success_count = 0

    for line_no, row in enumerate(theta_trials, start=2):
        if not _parse_bool(row, "success"):
            continue
        success_count += 1
        pattern = str(row.get("pattern", "")).strip()
        idx = pattern_index.get(pattern)
        if idx is None:
            continue
        matrix[0, idx] += _parse_float(row, "p_bg_assist_given_record", line_no)
        matrix[1, idx] += _parse_float(row, "p_intrinsic_dark_assist_given_record", line_no)

    total_false = float(np.sum(matrix))
    if total_false > 0.0:
        matrix = 100.0 * matrix / total_false
    return matrix, success_count


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.02, label, transform=ax.transAxes, fontsize=12.0, fontweight="bold", ha="left", va="bottom")


def _label_last_point(ax: plt.Axes, x: np.ndarray, y: np.ndarray, text: str, color: str, dy: float = 0.0) -> None:
    ax.text(float(x[-1]) + 0.01, float(y[-1]) + dy, text, color=color, fontsize=8.8, va="center", clip_on=False)


def _set_numeric_xlim(ax: plt.Axes, values: np.ndarray, *, pad_fraction: float = 0.035) -> None:
    x_min = float(np.min(values))
    x_max = float(np.max(values))
    span = x_max - x_min
    pad = pad_fraction * span if span > 0.0 else max(abs(x_min), 1.0) * pad_fraction
    ax.set_xlim(x_min - pad, x_max + pad)

def main() -> None:
    args = _parse_args()
    summary_dir = pathlib.Path(args.summary_dir)
    summary_rows = _read_csv_rows(summary_dir / "bsm_scan_summary.csv")
    trial_rows = _read_csv_rows(summary_dir / "bsm_scan_trials.csv")
    summary = _load_summary(summary_rows)

    theta = summary["theta"]
    rate_all = np.clip(summary["rate_all"], 0.0, None)
    rate_true = np.clip(summary["rate_true"], 0.0, None)
    rate_false = np.clip(summary["rate_false"], 0.0, None)
    fidelity = np.clip(summary["fidelity"], 0.0, 1.0)
    false_pct = 100.0 * np.clip(summary["false_fraction"], 0.0, 1.0)
    chsh = summary["chsh"]

    idx_rec = _recommended_index(fidelity, rate_all)
    theta_rec = float(theta[idx_rec])
    false_share_matrix, success_count = _false_contribution_matrix_from_trials(_filter_theta_rows(trial_rows, "bs_theta", theta_rec))
    source_share_pct = np.sum(false_share_matrix, axis=1)
    pattern_share_pct = np.sum(false_share_matrix, axis=0)

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
            "grid.alpha": 0.16,
            "grid.linewidth": 0.70,
        }
    )

    fig = plt.figure(figsize=(13.8, 3.85))
    ax_a = fig.add_axes([0.070, 0.22, 0.255, 0.60])
    ax_b = fig.add_axes([0.413, 0.22, 0.178, 0.60])
    cax_b = fig.add_axes([0.599, 0.22, 0.011, 0.60])
    ax_c = fig.add_axes([0.710, 0.22, 0.205, 0.60])

    rate_true_1e4 = 1e4 * rate_true
    rate_false_1e4 = 1e4 * rate_false
    ax_a.plot(theta, rate_true_1e4, color="#1f77b4", lw=2.1, marker="o", ms=3.8)
    ax_a.plot(theta, rate_false_1e4, color="#d62728", lw=1.7, marker="s", ms=3.4)
    ax_a.axvline(theta_rec, color="#9ca3af", lw=1.0, ls="--")
    _set_numeric_xlim(ax_a, theta, pad_fraction=0.04)
    ax_a.set_xlabel(r"$\theta_{\mathrm{BS}}$ (rad)")
    ax_a.set_ylabel(r"每次尝试概率 ($\times 10^{-4}$)")
    ax_a.set_title("角度扫描")
    ax_a_r = ax_a.twinx()
    ax_a_r.grid(False)
    ax_a_r.plot(theta, fidelity, color="#047857", lw=2.0, marker="o", ms=3.6)
    ax_a_r.plot(theta, false_pct / 100.0, color="#991b1b", lw=1.5, marker="s", ms=3.2, ls="--")
    ax_a_r.plot(theta, chsh / 4.0, color="#111827", lw=1.3, marker="^", ms=3.0, ls="-.")
    ax_a_r.set_ylabel(r"$F_t$、假成功占比、$S_{\max}/4$")
    _label_last_point(ax_a, theta, rate_true_1e4, r"$p_t$", "#1f77b4", dy=0.01)
    _label_last_point(ax_a, theta, rate_false_1e4, r"$p_f$", "#d62728", dy=-0.005)
    _label_last_point(ax_a_r, theta, fidelity, r"$F_t$", "#047857", dy=0.012)
    ax_a.text(theta_rec + 0.01, ax_a.get_ylim()[0] + 0.86 * (ax_a.get_ylim()[1] - ax_a.get_ylim()[0]), "选取工作点", fontsize=8.2, color="#4b5563")
    handles_a = [
        plt.Line2D([], [], color="#1f77b4", marker="o", lw=2.1, ms=3.8, label=r"$p_t$"),
        plt.Line2D([], [], color="#d62728", marker="s", lw=1.7, ms=3.4, label=r"$p_f$"),
        plt.Line2D([], [], color="#047857", marker="o", lw=2.0, ms=3.6, label=r"$F_t$"),
        plt.Line2D([], [], color="#991b1b", marker="s", lw=1.5, ms=3.2, ls="--", label="假成功占比"),
        plt.Line2D([], [], color="#111827", marker="^", lw=1.3, ms=3.0, ls="-.", label=r"$S_{\max}/4$"),
    ]
    ax_a.legend(handles=handles_a, frameon=False, fontsize=8.0, loc="upper left")
    _panel_label(ax_a, "(a)")

    scatter = ax_b.scatter(
        1e4 * np.clip(rate_all, np.finfo(float).tiny, None),
        fidelity,
        c=false_pct,
        s=48.0,
        cmap="Reds",
        norm=Normalize(vmin=float(np.min(false_pct)), vmax=float(np.max(false_pct) if np.max(false_pct) > np.min(false_pct) else np.min(false_pct) + 1.0)),
        edgecolors="white",
        linewidths=0.65,
        zorder=2,
    )
    ax_b.scatter(
        [1e4 * rate_all[idx_rec]],
        [fidelity[idx_rec]],
        marker="*",
        s=130,
        color="#f59e0b",
        edgecolors="#111827",
        linewidths=0.7,
        zorder=3,
    )
    rate_all_1e4 = 1e4 * np.clip(rate_all, np.finfo(float).tiny, None)
    _set_numeric_xlim(ax_b, rate_all_1e4, pad_fraction=0.05)
    ax_b.set_xlabel(r"宣告成功率 $p_s$ ($\times 10^{-4}$)")
    ax_b.set_ylabel(r"真成功条件保真度 $F_t$")
    ax_b.set_title("速率-保真度工作点前沿")
    cbar = fig.colorbar(scatter, cax=cax_b)
    cbar.set_label("假成功占比 (%)")
    _panel_label(ax_b, "(b)")

    active_idx = [idx for idx, value in enumerate(pattern_share_pct) if value > 0.05]
    x_pos = np.arange(len(active_idx), dtype=float)
    bar_width = 0.32
    bg_vals = false_share_matrix[0, active_idx]
    intrinsic_vals = false_share_matrix[1, active_idx]
    total_vals = pattern_share_pct[active_idx]
    pattern_labels = [PATTERN_LABELS[PATTERN_ORDER[idx]] for idx in active_idx]
    bars_bg = ax_c.bar(
        x_pos - bar_width / 2.0,
        bg_vals,
        width=bar_width,
        color="#f59e0b",
        edgecolor="#111827",
        linewidth=0.75,
        label=f"背景辅助 ({source_share_pct[0]:.1f}%)",
        zorder=3,
    )
    bars_intrinsic = ax_c.bar(
        x_pos + bar_width / 2.0,
        intrinsic_vals,
        width=bar_width,
        color="#dc2626",
        edgecolor="#111827",
        linewidth=0.75,
        label=f"内禀暗计数 ({source_share_pct[1]:.1f}%)",
        zorder=3,
    )
    total_peak = max(1.0, float(np.max(total_vals)))
    ax_c.set_ylim(0.0, total_peak * 1.25)
    ax_c.set_xlim(-0.6, len(active_idx) - 0.4)
    ax_c.set_xticks(x_pos, pattern_labels)
    ax_c.tick_params(axis="x", labelsize=8.0)
    ax_c.set_ylabel("对总假成功的贡献 (%)")
    ax_c.set_xlabel("点击模式")
    ax_c.set_title(rf"$\theta_{{\mathrm{{BS}}}}={theta_rec:.2f}$ rad 下点击模式贡献")
    ax_c.grid(axis="y", alpha=0.18, linewidth=0.70)
    ax_c.grid(axis="x", visible=False)
    for x_center, total in zip(x_pos, total_vals):
        ax_c.text(x_center, total + 0.45, f"总 {total:.1f}%", ha="center", va="bottom", fontsize=7.4, color="#4b5563")
    for bars in (bars_bg, bars_intrinsic):
        for bar in bars:
            height = float(bar.get_height())
            if height >= 0.15:
                ax_c.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.20,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    color="#111827",
                )
    ax_c.legend(frameon=False, fontsize=7.3, loc="upper right")
    _panel_label(ax_c, "(c)")

    fig.suptitle(
        (
            "真实 BSM 扫描的误差预算："
            + rf"$\theta_{{\mathrm{{BS}}}}={theta_rec:.2f}$ rad，"
            + rf"$F_t={fidelity[idx_rec]:.3f}$，"
            + rf"$p_t={1e4 * rate_true[idx_rec]:.3f}\times10^{{-4}}$，"
            + rf"$p_f={1e4 * rate_false[idx_rec]:.4f}\times10^{{-4}}$，"
            + f"成功记录数={success_count}"
        ),
        fontsize=12.0,
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
