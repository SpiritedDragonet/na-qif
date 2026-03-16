import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASELINE_ETA_Q = 0.57
DEFAULT_BASELINE_NOISE_CPS_PER_MHZ = 41.1
DEFAULT_SMOOTH_SIGMA = 0.85
DEFAULT_SMOOTH_BLEND = 0.65


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _default_summary_csv() -> pathlib.Path:
    data_root = pathlib.Path(__file__).resolve().parents[1] / "data"
    candidates = sorted(
        data_root.glob("*/summary/qfc_eff_noise_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    outputs_root = _repo_root() / "outputs"
    candidates = sorted(
        outputs_root.glob("*/summary/qfc_eff_noise_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    return data_root / "qfc_eff_noise_scan_server_output_latest" / "summary" / "qfc_eff_noise_scan_summary.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot QFC efficiency-noise landscape from summary CSV.")
    parser.add_argument("--summary-csv", type=pathlib.Path, default=_default_summary_csv())
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=DEFAULT_SMOOTH_SIGMA,
        help="Gaussian smoothing sigma in grid-cell units (0 disables smoothing).",
    )
    parser.add_argument(
        "--smooth-blend",
        type=float,
        default=DEFAULT_SMOOTH_BLEND,
        help="Blend factor between raw and smoothed grids in [0, 1].",
    )
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

    required = (
        "qfc_eta",
        "qfc_noise_sd_cps_per_mhz",
        "fidelity_true_avg",
        "p_success_abs_avg",
    )
    missing = [key for key in required if key not in fieldnames]
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {', '.join(missing)}")
    if not rows:
        raise ValueError(f"Summary CSV has no rows: {summary_csv}")

    eta_vals = sorted(
        {
            round(_parse_float(row, "qfc_eta", idx), 9)
            for idx, row in enumerate(rows, start=2)
        }
    )
    noise_vals = sorted(
        {
            round(_parse_float(row, "qfc_noise_sd_cps_per_mhz", idx), 9)
            for idx, row in enumerate(rows, start=2)
        }
    )
    eta = np.asarray(eta_vals, dtype=float)
    noise = np.asarray(noise_vals, dtype=float)

    fidelity = np.full((noise.size, eta.size), np.nan, dtype=float)
    p_success = np.full((noise.size, eta.size), np.nan, dtype=float)
    eta_index = {value: idx for idx, value in enumerate(eta_vals)}
    noise_index = {value: idx for idx, value in enumerate(noise_vals)}

    for line_no, row in enumerate(rows, start=2):
        e = round(_parse_float(row, "qfc_eta", line_no), 9)
        n = round(_parse_float(row, "qfc_noise_sd_cps_per_mhz", line_no), 9)
        i = noise_index[n]
        j = eta_index[e]
        fidelity[i, j] = _parse_float(row, "fidelity_true_avg", line_no)
        p_success[i, j] = _parse_float(row, "p_success_abs_avg", line_no)

    return eta, noise, fidelity, p_success


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray([1.0], dtype=float)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return np.asarray([1.0], dtype=float)
    return kernel / kernel_sum


def _convolve1d_nan(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if arr.size == 0:
        return arr.copy()
    pad = int(len(kernel) // 2)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    arr_pad = np.pad(arr, pad_width, mode="edge")
    val_pad = np.isfinite(arr_pad).astype(float)
    arr_pad = np.nan_to_num(arr_pad, nan=0.0)

    out = np.zeros_like(arr, dtype=float)
    norm = np.zeros_like(arr, dtype=float)
    for shift, weight in enumerate(kernel):
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(shift, shift + arr.shape[axis])
        sl = tuple(slicer)
        out += weight * arr_pad[sl]
        norm += weight * val_pad[sl]
    return np.where(norm > 1e-12, out / norm, np.nan)


def _smooth_field(field: np.ndarray, sigma: float, blend: float) -> np.ndarray:
    blend = float(np.clip(blend, 0.0, 1.0))
    if sigma <= 0.0 or blend <= 0.0:
        return field.copy()

    kernel = _gaussian_kernel1d(float(sigma))
    smoothed = _convolve1d_nan(_convolve1d_nan(field, kernel, axis=0), kernel, axis=1)
    out = field.copy()
    valid = np.isfinite(field) & np.isfinite(smoothed)
    out[valid] = (1.0 - blend) * field[valid] + blend * smoothed[valid]
    return out


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
            "axes.grid": False,
        }
    )

    eta_q, noise_sd, fidelity, p_success = _load_grid(args.summary_csv)
    fidelity_plot = _smooth_field(fidelity, sigma=args.smooth_sigma, blend=args.smooth_blend)
    p_success_plot = _smooth_field(p_success, sigma=args.smooth_sigma, blend=args.smooth_blend)

    fidelity_m = np.ma.masked_invalid(fidelity_plot)
    p_success_m = np.ma.masked_invalid(p_success_plot)

    fig = plt.figure(figsize=(9.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.26)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    extent = [float(eta_q.min()), float(eta_q.max()), float(noise_sd.min()), float(noise_sd.max())]

    im0 = ax0.imshow(
        fidelity_m,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        interpolation="bicubic",
    )
    cs0 = ax0.contour(
        eta_q,
        noise_sd,
        fidelity_m,
        levels=6,
        colors="white",
        linewidths=0.8,
        alpha=0.85,
    )
    ax0.clabel(cs0, fmt="%.3f", inline=True, fontsize=8)
    ax0.set_xlabel(r"QFC 效率 $\eta_q$")
    ax0.set_ylabel(r"QFC 噪声标准差 (cps/MHz)")
    ax0.set_title(r"条件保真度热图 $F_t(\eta_q,\sigma_{\rm QFC})$")
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.05, pad=0.03)
    cb0.set_label(r"$F_t$")
    _panel_label(ax0, "(a)")

    im1 = ax1.imshow(
        1e6 * p_success_m,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        interpolation="bicubic",
    )
    cs1 = ax1.contour(
        eta_q,
        noise_sd,
        1e6 * p_success_m,
        levels=6,
        colors="white",
        linewidths=0.8,
        alpha=0.85,
    )
    ax1.clabel(cs1, fmt="%.3f", inline=True, fontsize=8)
    ax1.set_xlabel(r"QFC 效率 $\eta_q$")
    ax1.set_ylabel(r"QFC 噪声标准差 (cps/MHz)")
    ax1.set_title(r"宣告概率热图 $p_s(\eta_q,\sigma_{\rm QFC})$")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.05, pad=0.03)
    cb1.set_label(r"$p_s \times 10^{-6}$")
    _panel_label(ax1, "(b)")

    eta_mesh, noise_mesh = np.meshgrid(eta_q, noise_sd, indexing="xy")
    ax0.scatter(eta_mesh, noise_mesh, s=5, color="white", alpha=0.30, linewidths=0.0, zorder=3)
    ax1.scatter(eta_mesh, noise_mesh, s=5, color="white", alpha=0.25, linewidths=0.0, zorder=3)

    ax0.scatter(
        [DEFAULT_BASELINE_ETA_Q],
        [DEFAULT_BASELINE_NOISE_CPS_PER_MHZ],
        s=48,
        marker="o",
        color="#f2c14e",
        edgecolors="#1f1f1f",
        linewidths=0.8,
    )
    ax1.scatter(
        [DEFAULT_BASELINE_ETA_Q],
        [DEFAULT_BASELINE_NOISE_CPS_PER_MHZ],
        s=48,
        marker="o",
        color="#f2c14e",
        edgecolors="#1f1f1f",
        linewidths=0.8,
    )

    fig.suptitle(
        "QFC 效率-噪声权衡（QFC_EFF_NOISE_SCAN 汇总）",
        fontsize=12.3,
        fontweight="bold",
    )
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
