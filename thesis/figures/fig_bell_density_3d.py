import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "axes.linewidth": 0.85,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _find_density_json() -> pathlib.Path:
    data_root = pathlib.Path(__file__).resolve().parents[1] / "data"
    candidates = []
    for path in data_root.glob("sim_single_run_*"):
        density_path = path / "results" / "result_sim_run_000000" / "raw" / "declared_density_matrix.json"
        if density_path.exists():
            candidates.append(density_path)
    if not candidates:
        raise FileNotFoundError(f"未找到 declared_density_matrix.json: {data_root}")
    return sorted(candidates)[-1]


def _load_density_matrix(path: pathlib.Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "rho_ff_real" in payload and "rho_ff_imag" in payload:
        rho_real = np.array(payload["rho_ff_real"], dtype=float)
        rho_imag = np.array(payload["rho_ff_imag"], dtype=float)
    elif "rho_raw_real" in payload and "rho_raw_imag" in payload:
        rho_real = np.array(payload["rho_raw_real"], dtype=float)
        rho_imag = np.array(payload["rho_raw_imag"], dtype=float)
    else:
        raise KeyError(f"{path} 缺少 rho_ff/rho_raw 字段")

    rho = rho_real + 1j * rho_imag
    if rho.shape != (4, 4):
        raise ValueError(f"密度矩阵形状异常: {rho.shape}")
    return rho


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.7},
    )


def _annotate_matrix(ax: plt.Axes, mat: np.ndarray, vmax: float) -> None:
    threshold = max(0.015, 0.08 * vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat[i, j])
            if abs(value) < threshold:
                continue
            color = "white" if abs(value) > 0.52 * vmax else "#111111"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8.4, color=color)


def _plot_component(ax: plt.Axes, mat: np.ndarray, title: str, vmax: float) -> mpl.image.AxesImage:
    labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")
    ax.set_title(title, pad=4.0)
    ax.set_xticks(range(4), labels, rotation=22, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("ket basis")
    ax.set_ylabel("bra basis")
    ax.set_xticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _annotate_matrix(ax, mat, vmax)
    return im


def main() -> None:
    _set_style()

    density_path = _find_density_json()
    rho = _load_density_matrix(density_path)
    rho_re = np.real(rho)
    rho_im = np.imag(rho)
    vmax = max(float(np.max(np.abs(rho_re))), float(np.max(np.abs(rho_im))), 0.15)

    fig = plt.figure(figsize=(9.8, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    im0 = _plot_component(ax0, rho_re, r"Re$(\rho_{\mathrm{cond}})$", vmax)
    _plot_component(ax1, rho_im, r"Im$(\rho_{\mathrm{cond}})$", vmax)
    _panel_label(ax0, "(a)")
    _panel_label(ax1, "(b)")

    cbar = fig.colorbar(im0, ax=[ax0, ax1], fraction=0.048, pad=0.015, shrink=0.92)
    cbar.set_label(r"Matrix element value")

    fig.suptitle("Heralded Bell-state density matrix (2D heatmap view)", y=1.02, fontweight="bold")

    out_png = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
