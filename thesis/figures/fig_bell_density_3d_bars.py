import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _rho_bell_conditional() -> np.ndarray:
    # Same representative heralded state as the 2D heatmap figure.
    return np.array(
        [
            [0.015 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.486 + 0.0j, -0.458 + 0.024j, 0.0 + 0.0j],
            [0.0 + 0.0j, -0.458 - 0.024j, 0.486 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.013 + 0.0j],
        ],
        dtype=np.complex128,
    )


def _plot_component(ax, component: np.ndarray, title: str, zlim: float) -> None:
    n = component.shape[0]
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xpos = xx.ravel().astype(float)
    ypos = yy.ravel().astype(float)
    zpos = np.zeros_like(xpos)
    dx = np.full_like(xpos, 0.62, dtype=float)
    dy = np.full_like(ypos, 0.62, dtype=float)
    dz = component.ravel()

    norm = Normalize(vmin=-zlim, vmax=zlim)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(norm(dz))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True, zsort="average")
    ax.view_init(elev=24, azim=-52)
    ax.set_zlim(-zlim, zlim)
    ax.set_zticks([-zlim, -0.5 * zlim, 0.0, 0.5 * zlim, zlim])
    ax.set_title(title)

    labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    ax.set_xticks(np.arange(n) + 0.31)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n) + 0.31)
    ax.set_yticklabels(labels, rotation=-15, ha="right", fontsize=8)
    ax.set_xlabel("ket")
    ax.set_ylabel("bra")
    ax.set_zlabel(r"$\rho$")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(r"$\rho_{ij}$")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    rho = _rho_bell_conditional()
    rho_re = np.real(rho)
    rho_im = np.imag(rho)
    zlim = 0.52

    fig = plt.figure(figsize=(10.8, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")

    _plot_component(ax0, rho_re, r"Re$(\rho_{\mathrm{cond}})$", zlim)
    _plot_component(ax1, rho_im, r"Im$(\rho_{\mathrm{cond}})$", zlim)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.10, wspace=0.20)
    fig.suptitle("Heralded Bell-state density matrix (3D bars)", fontsize=12.4, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

