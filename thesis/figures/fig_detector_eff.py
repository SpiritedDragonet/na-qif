import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASELINE_ETA_DET = 0.85
DEFAULT_BASELINE_BG_HZ = 165.0


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _default_summary_csv() -> pathlib.Path:
    data_root = pathlib.Path(__file__).resolve().parents[1] / "data"
    candidates = sorted(
        data_root.glob("*/summary/detector_bg_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    outputs_root = _repo_root() / "outputs"
    candidates = sorted(
        outputs_root.glob("*/summary/detector_bg_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    return data_root / "detector_bg_scan_server_output_latest" / "summary" / "detector_bg_scan_summary.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot detector efficiency-background scan from summary CSV.")
    parser.add_argument("--summary-csv", type=pathlib.Path, default=_default_summary_csv())
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
        raise ValueError(f"CSV line {line_no}: non-finite value '{raw}' for '{key}'")
    return value


def _load_grid(summary_csv: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    success_key = "herald_rate_abs" if "herald_rate_abs" in fieldnames else "p_success_abs_avg"
    required = ("eta_det", "bg_rate_mean_hz", "fidelity_true_avg", success_key)
    missing = [key for key in required if key not in fieldnames]
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {', '.join(missing)}")
    if not rows:
        raise ValueError(f"Summary CSV has no rows: {summary_csv}")

    eta_vals = sorted(
        {
            round(_parse_float(row, "eta_det", idx), 9)
            for idx, row in enumerate(rows, start=2)
        }
    )
    bg_vals = sorted(
        {
            round(_parse_float(row, "bg_rate_mean_hz", idx), 9)
            for idx, row in enumerate(rows, start=2)
        }
    )
    eta = np.asarray(eta_vals, dtype=float)
    bg = np.asarray(bg_vals, dtype=float)

    fidelity = np.full((bg.size, eta.size), np.nan, dtype=float)
    success = np.full((bg.size, eta.size), np.nan, dtype=float)
    eta_index = {value: idx for idx, value in enumerate(eta_vals)}
    bg_index = {value: idx for idx, value in enumerate(bg_vals)}

    for line_no, row in enumerate(rows, start=2):
        e = round(_parse_float(row, "eta_det", line_no), 9)
        b = round(_parse_float(row, "bg_rate_mean_hz", line_no), 9)
        i = bg_index[b]
        j = eta_index[e]
        fidelity[i, j] = _parse_float(row, "fidelity_true_avg", line_no)
        success[i, j] = _parse_float(row, success_key, line_no)

    if np.isnan(fidelity).any() or np.isnan(success).any():
        raise ValueError(
            "Summary grid is incomplete: missing entries for some (eta_det, bg_rate_mean_hz) points."
        )

    return eta, bg, fidelity, success


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
        }
    )

    eta, bg_levels, fidelity, success = _load_grid(args.summary_csv)

    fig = plt.figure(figsize=(10.4, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    im = ax0.imshow(
        fidelity,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=float(np.nanmin(fidelity)),
        vmax=float(np.nanmax(fidelity)),
        extent=[eta.min(), eta.max(), bg_levels.min(), bg_levels.max()],
        interpolation="bicubic",
    )
    fidelity_levels = np.linspace(float(np.nanmin(fidelity)), float(np.nanmax(fidelity)), 6)
    cs = ax0.contour(
        eta,
        bg_levels,
        fidelity,
        levels=fidelity_levels,
        colors="white",
        linewidths=0.9,
        alpha=0.85,
    )
    ax0.clabel(cs, fmt="%.3f", inline=True, fontsize=8)
    ax0.set_xlabel(r"探测效率 $\eta_d$")
    ax0.set_ylabel(r"背景计数率 $R_{\mathrm{bg}}$ (Hz)")
    ax0.set_title(r"条件保真度热图 $F_t(\eta_d,\mathrm{bg})$")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.05, pad=0.03)
    cbar.set_label(r"$F_t$")
    ax0.scatter(
        [DEFAULT_BASELINE_ETA_DET],
        [DEFAULT_BASELINE_BG_HZ],
        s=46,
        marker="o",
        color="#f2c14e",
        edgecolors="#1f1f1f",
        linewidths=0.8,
        zorder=5,
    )
    _panel_label(ax0, "(a)")

    # 先画全部切片（浅灰）保留全信息，再突出代表性切片避免图例过密。
    for i, bg in enumerate(bg_levels):
        ax1.plot(
            eta,
            1e6 * success[i, :],
            lw=0.9,
            color="#a7a7a7",
            alpha=0.28,
            zorder=1,
        )
    if bg_levels.size <= 6:
        highlight_idx = np.arange(bg_levels.size, dtype=int)
    else:
        highlight_idx = np.unique(np.round(np.linspace(0, bg_levels.size - 1, 5)).astype(int))
    cmap = plt.get_cmap("magma")
    for k, i in enumerate(highlight_idx):
        bg = bg_levels[i]
        color = cmap(0.18 + 0.72 * (k / max(1, len(highlight_idx) - 1)))
        ax1.plot(
            eta,
            1e6 * success[i, :],
            lw=2.0,
            color=color,
            label=rf"$R_{{\mathrm{{bg}}}}={int(round(bg))}$ Hz",
            zorder=3,
        )
    ax1.set_xlabel(r"探测效率 $\eta_d$")
    ax1.set_ylabel(r"宣告概率 $p_s$（每次尝试 $\times10^{-6}$）")
    ax1.set_title(r"固定 $R_{\mathrm{bg}}$ 下的宣告概率切片")
    ax1.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    _panel_label(ax1, "(b)")

    fig.suptitle(
        "探测效率与背景权衡（DETECTOR_BG_SCAN 汇总）",
        fontsize=12.1,
        fontweight="bold",
    )
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
