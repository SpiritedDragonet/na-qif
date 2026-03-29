import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from plot_style import frame_all_axes

EXPORT_PNG = False


def _default_summary_dir() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "data"
        / "bsm_scan_summary_output_20260223_2121"
        / "summary"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot two-panel error-budget figure from BSM_SCAN summary CSV.")
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
    }


def _recommended_index(fidelity: np.ndarray, rate: np.ndarray) -> int:
    idx_candidates = np.flatnonzero(np.isclose(fidelity, float(np.max(fidelity)), rtol=0.0, atol=1e-12))
    if idx_candidates.size == 1:
        return int(idx_candidates[0])
    return int(idx_candidates[np.argmax(rate[idx_candidates])])


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.03, label, transform=ax.transAxes, fontsize=12.0, fontweight="bold", ha="left", va="bottom")


def _set_numeric_xlim(ax: plt.Axes, values: np.ndarray, *, pad_fraction: float = 0.04) -> None:
    x_min = float(np.min(values))
    x_max = float(np.max(values))
    span = x_max - x_min
    pad = pad_fraction * span if span > 0.0 else max(abs(x_min), 1.0) * pad_fraction
    ax.set_xlim(x_min - pad, x_max + pad)


def main() -> None:
    args = _parse_args()
    summary_dir = pathlib.Path(args.summary_dir)
    summary_rows = _read_csv_rows(summary_dir / "bsm_scan_summary.csv")
    summary = _load_summary(summary_rows)

    theta = summary["theta"]
    rate_all = np.clip(summary["rate_all"], 0.0, None)
    rate_true = np.clip(summary["rate_true"], 0.0, None)
    rate_false = np.clip(summary["rate_false"], 0.0, None)
    fidelity = np.clip(summary["fidelity"], 0.0, 1.0)
    false_pct = 100.0 * np.clip(summary["false_fraction"], 0.0, 1.0)

    idx_rec = _recommended_index(fidelity, rate_all)
    theta_rec = float(theta[idx_rec])

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
            "axes.edgecolor": "black",
            "axes.linewidth": 0.95,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.70,
        }
    )

    fig = plt.figure(figsize=(10.9, 4.0))
    ax_a = fig.add_axes([0.075, 0.20, 0.44, 0.64])
    ax_b = fig.add_axes([0.61, 0.20, 0.28, 0.64])
    cax_b = fig.add_axes([0.905, 0.20, 0.015, 0.64])

    rate_true_1e4 = 1e4 * rate_true
    rate_false_1e4 = 1e4 * rate_false
    ax_a.plot(theta, rate_true_1e4, color="#1f77b4", lw=2.1, marker="o", ms=3.8, label=r"$p_t$")
    ax_a.plot(theta, rate_false_1e4, color="#d62728", lw=1.8, marker="s", ms=3.4, label=r"$p_f$")
    ax_a.axvline(theta_rec, color="#9ca3af", lw=1.0, ls="--")
    _set_numeric_xlim(ax_a, theta, pad_fraction=0.04)
    ax_a.set_xlabel(r"$\theta_{\mathrm{BS}}$ (rad)")
    ax_a.set_ylabel(r"每次尝试概率 ($\times 10^{-4}$)")
    ax_a.set_title("角度扫描下的事件预算")
    ax_a_r = ax_a.twinx()
    ax_a_r.grid(False)
    ax_a_r.plot(theta, fidelity, color="#047857", lw=2.0, marker="o", ms=3.5, label=r"$F_t$")
    ax_a_r.plot(theta, false_pct, color="#991b1b", lw=1.55, marker="s", ms=3.0, ls="--", label="假成功占比")
    ax_a_r.set_ylabel(r"$F_t$ 与假成功占比 (%)")
    ax_a.text(
        theta_rec + 0.01,
        ax_a.get_ylim()[0] + 0.82 * (ax_a.get_ylim()[1] - ax_a.get_ylim()[0]),
        "推荐工作点",
        fontsize=8.2,
        color="#4b5563",
    )
    handles_a = [
        plt.Line2D([], [], color="#1f77b4", marker="o", lw=2.1, ms=3.8, label=r"$p_t$"),
        plt.Line2D([], [], color="#d62728", marker="s", lw=1.8, ms=3.4, label=r"$p_f$"),
        plt.Line2D([], [], color="#047857", marker="o", lw=2.0, ms=3.5, label=r"$F_t$"),
        plt.Line2D([], [], color="#991b1b", marker="s", lw=1.55, ms=3.0, ls="--", label="假成功占比"),
    ]
    ax_a.legend(handles=handles_a, frameon=False, fontsize=8.2, loc="upper left")
    _panel_label(ax_a, "(a)")

    scatter = ax_b.scatter(
        1e4 * np.clip(rate_all, np.finfo(float).tiny, None),
        fidelity,
        c=false_pct,
        s=52.0,
        cmap="Reds",
        norm=Normalize(
            vmin=float(np.min(false_pct)),
            vmax=float(np.max(false_pct) if np.max(false_pct) > np.min(false_pct) else np.min(false_pct) + 1.0),
        ),
        edgecolors="white",
        linewidths=0.65,
        zorder=2,
    )
    ax_b.scatter(
        [1e4 * rate_all[idx_rec]],
        [fidelity[idx_rec]],
        marker="*",
        s=135,
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

    fig.suptitle(
        (
            "真实 BSM 扫描的误差预算："
            + rf"$\theta_{{\mathrm{{BS}}}}={theta_rec:.2f}$ rad，"
            + rf"$F_t={fidelity[idx_rec]:.3f}$，"
            + rf"$p_t={1e4 * rate_true[idx_rec]:.3f}\times10^{{-4}}$，"
            + rf"$p_f={1e4 * rate_false[idx_rec]:.4f}\times10^{{-4}}$"
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
