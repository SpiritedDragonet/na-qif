import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _default_summary_csv() -> pathlib.Path:
    data_root = pathlib.Path(__file__).resolve().parents[1] / "data"
    candidates = sorted(
        data_root.glob("bsm_scan_summary_output_*/summary/bsm_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    outputs_root = _repo_root() / "outputs"
    candidates = sorted(
        outputs_root.glob("*/summary/bsm_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    return data_root / "bsm_scan_summary_output_latest" / "summary" / "bsm_scan_summary.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rate-fidelity frontier from BSM_SCAN summary CSV.")
    parser.add_argument("--summary-csv", type=pathlib.Path, default=_default_summary_csv())
    return parser.parse_args()


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


def _load_summary(summary_csv: pathlib.Path) -> dict[str, np.ndarray]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = (
        "bs_theta",
        "bs_split_ratio",
        "herald_rate_abs",
        "fidelity_true_avg",
        "false_fraction_global",
        "chsh_s_max_avg",
    )
    missing = [key for key in required if key not in fieldnames]
    if missing:
        raise ValueError(f"Summary CSV missing columns: {', '.join(missing)}")
    if not rows:
        raise ValueError(f"Summary CSV has no rows: {summary_csv}")

    records = []
    for line_no, row in enumerate(rows, start=2):
        records.append(
            (
                _parse_float(row, "bs_theta", line_no),
                _parse_float(row, "bs_split_ratio", line_no),
                _parse_float(row, "herald_rate_abs", line_no),
                _parse_float(row, "fidelity_true_avg", line_no),
                _parse_float(row, "false_fraction_global", line_no),
                _parse_float(row, "chsh_s_max_avg", line_no),
            )
        )
    arr = np.asarray(records, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return {
        "theta": arr[:, 0],
        "split_ratio": arr[:, 1],
        "rate": arr[:, 2],
        "fidelity": arr[:, 3],
        "false_fraction": arr[:, 4],
        "chsh": arr[:, 5],
    }


def _pareto_front(rate: np.ndarray, fidelity: np.ndarray) -> np.ndarray:
    order = np.argsort(rate)
    keep = []
    best_fidelity = -np.inf
    for idx in order[::-1]:
        fi = float(fidelity[idx])
        if fi > best_fidelity:
            keep.append(int(idx))
            best_fidelity = fi
    keep_arr = np.asarray(keep, dtype=int)
    return keep_arr[np.argsort(rate[keep_arr])]


def _recommended_index(fidelity: np.ndarray, rate: np.ndarray) -> int:
    idx_candidates = np.flatnonzero(np.isclose(fidelity, float(np.max(fidelity)), rtol=0.0, atol=1e-12))
    if idx_candidates.size == 1:
        return int(idx_candidates[0])
    return int(idx_candidates[np.argmax(rate[idx_candidates])])


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
    summary = _load_summary(args.summary_csv)

    theta = summary["theta"]
    rate = np.clip(summary["rate"], np.finfo(float).tiny, None)
    fidelity = np.clip(summary["fidelity"], 0.0, 1.0)
    false_pct = 100.0 * np.clip(summary["false_fraction"], 0.0, 1.0)
    rate_1e4 = 1e4 * rate
    rate_attempt_pct = 100.0 * rate

    idx_front = _pareto_front(rate, fidelity)
    idx_rec = _recommended_index(fidelity, rate)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.75,
        }
    )

    fig = plt.figure(figsize=(10.8, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.26)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    sc = ax0.scatter(
        rate_1e4,
        fidelity,
        c=false_pct,
        s=76.0,
        cmap="Reds",
        alpha=0.95,
        edgecolors="#f8fafc",
        linewidths=0.8,
        zorder=3,
    )
    ax0.plot(
        rate_1e4[idx_front],
        fidelity[idx_front],
        color="#1e3a8a",
        lw=2.0,
        marker="o",
        ms=4.2,
        label="Pareto frontier",
        zorder=4,
    )
    ax0.scatter(
        [rate_1e4[idx_rec]],
        [fidelity[idx_rec]],
        marker="*",
        s=210,
        color="#f97316",
        edgecolors="#0f172a",
        linewidths=0.8,
        label=rf"best point ($\theta_{{BS}}={theta[idx_rec]:.2f}$ rad)",
        zorder=5,
    )
    ax0.set_xlabel(r"Herald probability $p_s$ ($\times 10^{-4}$)")
    ax0.set_ylabel(r"Conditional fidelity $F_t$")
    xpad = 0.06 * max(float(np.max(rate_1e4) - np.min(rate_1e4)), 0.01)
    ypad = 0.08 * max(float(np.max(fidelity) - np.min(fidelity)), 0.01)
    ax0.set_xlim(float(np.min(rate_1e4)) - xpad, float(np.max(rate_1e4)) + xpad)
    ax0.set_ylim(float(np.min(fidelity)) - ypad, float(np.max(fidelity)) + ypad)
    ax0.set_title("Operating points in the rate-fidelity plane")
    ax0.legend(frameon=False, fontsize=8.3, loc="lower right")
    _panel_label(ax0, "(a)")
    cbar = fig.colorbar(sc, ax=ax0, fraction=0.048, pad=0.03)
    cbar.set_label("Spurious-share (%)")

    ax1.plot(theta, fidelity, color="#047857", lw=2.2, marker="o", ms=4.6, label=r"Conditional fidelity $F_t$")
    ax1.axvline(theta[idx_rec], color="#9ca3af", lw=1.0, ls=":")
    ax1.set_xlabel(r"BS mixing angle $\theta_{BS}$ (rad)")
    ax1.set_ylabel(r"Conditional fidelity $F_t$", color="#047857")
    ax1.tick_params(axis="y", colors="#047857")
    ax1.set_title(r"Metric trends versus $\theta_{BS}$")

    ax1_r = ax1.twinx()
    ax1_r.plot(theta, rate_attempt_pct, color="#1d4ed8", lw=1.8, marker="s", ms=3.9, label=r"Herald rate $p_s$ (%)")
    ax1_r.plot(theta, false_pct, color="#dc2626", lw=1.8, marker="^", ms=3.9, ls="--", label="Spurious-share (%)")
    ax1_r.set_ylabel("Percentage (%)", color="#991b1b")
    ax1_r.tick_params(axis="y", colors="#991b1b")

    lines = []
    labels = []
    for axis in (ax1, ax1_r):
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    ax1.legend(lines, labels, frameon=False, fontsize=8.3, loc="upper left")
    _panel_label(ax1, "(b)")

    fig.suptitle(
        (
            "Rate-Fidelity Working-Point Frontier under Beam-Splitter-Angle Scan "
            + rf"(best point: $\theta_{{BS}}={theta[idx_rec]:.2f}$ rad, "
            + rf"$F_t={fidelity[idx_rec]:.3f}$, "
            + rf"$p_s={rate_1e4[idx_rec]:.3f}\times10^{{-4}}$)"
        ),
        fontsize=12.3,
        fontweight="bold",
    )

    out_base = pathlib.Path(__file__).with_suffix("")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=260)
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
