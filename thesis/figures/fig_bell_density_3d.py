import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _rho_bell_conditional() -> np.ndarray:
    # A representative heralded two-qubit state near |Psi-> with small leakage.
    return np.array(
        [
            [0.015 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.486 + 0.0j, -0.458 + 0.024j, 0.0 + 0.0j],
            [0.0 + 0.0j, -0.458 - 0.024j, 0.486 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.013 + 0.0j],
        ],
        dtype=np.complex128,
    )


def _annotate_matrix(ax: plt.Axes, mat: np.ndarray, vmax: float) -> None:
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            value = mat[i, j]
            if abs(value) < 0.006:
                continue
            color = "white" if abs(value) > 0.45 * vmax else "#111111"
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8.2, color=color)


def _plot_component(ax: plt.Axes, mat: np.ndarray, title: str, vmax: float) -> None:
    labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")
    ax.set_title(title)
    ax.set_xticks(range(4), labels, rotation=20, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("ket index")
    ax.set_ylabel("bra index")
    _annotate_matrix(ax, mat, vmax)
    return im


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    rho = _rho_bell_conditional()
    rho_re = np.real(rho)
    rho_im = np.imag(rho)
    vmax = 0.52

    fig = plt.figure(figsize=(9.8, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    im0 = _plot_component(ax0, rho_re, r"Re$(\rho_{\mathrm{cond}})$", vmax)
    im1 = _plot_component(ax1, rho_im, r"Im$(\rho_{\mathrm{cond}})$", vmax)
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.05, pad=0.02)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.05, pad=0.02)
    cbar0.set_label(r"Re$(\rho_{ij})$")
    cbar1.set_label(r"Im$(\rho_{ij})$")

    fig.suptitle("Heralded Bell-state density matrix (2D heatmaps)", fontsize=12.1, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
