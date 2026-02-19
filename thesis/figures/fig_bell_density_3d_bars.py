import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
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


def _panel_label(ax, label: str) -> None:
    ax.text2D(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.6},
    )


def _plot_component(ax, component: np.ndarray, title: str, zlim: float, cmap, norm) -> None:
    n = component.shape[0]
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xpos = (xx.ravel() - 0.28).astype(float)
    ypos = (yy.ravel() - 0.28).astype(float)
    dx = np.full_like(xpos, 0.56, dtype=float)
    dy = np.full_like(ypos, 0.56, dtype=float)

    values = component.ravel().astype(float)
    zbase = np.where(values >= 0.0, 0.0, values)
    heights = np.abs(values)
    colors = cmap(norm(values))

    ax.bar3d(
        xpos,
        ypos,
        zbase,
        dx,
        dy,
        heights,
        color=colors,
        shade=True,
        alpha=0.95,
        edgecolor=(0.22, 0.22, 0.22, 0.45),
        linewidth=0.24,
        zsort="average",
    )
    ax.view_init(elev=26, azim=-52)
    ax.set_box_aspect((1.0, 1.0, 0.76))
    ax.set_title(title, pad=8.0)
    ax.set_zlim(-zlim, zlim)
    ax.set_zticks([-zlim, -0.5 * zlim, 0.0, 0.5 * zlim, zlim])

    labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, rotation=-16, ha="right")
    ax.set_xlabel("ket")
    ax.set_ylabel("bra")
    ax.set_zlabel(r"value")

    ax.xaxis.pane.set_facecolor((0.965, 0.972, 0.988, 1.0))
    ax.yaxis.pane.set_facecolor((0.965, 0.972, 0.988, 1.0))
    ax.zaxis.pane.set_facecolor((0.982, 0.986, 0.995, 1.0))
    ax.xaxis.pane.set_edgecolor((0.82, 0.85, 0.90, 1.0))
    ax.yaxis.pane.set_edgecolor((0.82, 0.85, 0.90, 1.0))
    ax.zaxis.pane.set_edgecolor((0.82, 0.85, 0.90, 1.0))

    top_idx = np.argsort(np.abs(values))[-4:]
    for idx in top_idx:
        v = float(values[idx])
        if abs(v) < max(0.03, 0.09 * zlim):
            continue
        tx = xpos[idx] + 0.28
        ty = ypos[idx] + 0.28
        tz = v + (0.05 * zlim if v >= 0.0 else -0.08 * zlim)
        ax.text(tx, ty, tz, f"{v:+.2f}", ha="center", va="center", fontsize=7.8)


def main() -> None:
    _set_style()

    density_path = _find_density_json()
    rho = _load_density_matrix(density_path)
    rho_re = np.real(rho)
    rho_im = np.imag(rho)
    zlim = max(float(np.max(np.abs(rho_re))), float(np.max(np.abs(rho_im))), 0.15)

    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-zlim, vmax=zlim)

    fig = plt.figure(figsize=(10.9, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.16)
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")

    _plot_component(ax0, rho_re, r"Re$(\rho_{\mathrm{cond}})$", zlim, cmap, norm)
    _plot_component(ax1, rho_im, r"Im$(\rho_{\mathrm{cond}})$", zlim, cmap, norm)
    _panel_label(ax0, "(a)")
    _panel_label(ax1, "(b)")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax0, ax1], fraction=0.035, pad=0.02, shrink=0.86)
    cbar.set_label("Matrix element value")

    fig.subplots_adjust(left=0.02, right=0.95, top=0.88, bottom=0.05, wspace=0.14)
    fig.suptitle("Heralded Bell-state density matrix (3D bar view)", fontsize=12.5, fontweight="bold")

    out_png = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
