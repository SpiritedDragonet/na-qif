import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _metrics(eta_q: np.ndarray, bg_cps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qq, bb = np.meshgrid(eta_q, bg_cps, indexing="xy")
    p_true = 1.35e-6 * qq**2
    p_false = 0.08e-6 + 0.006e-6 * bb
    p_all = p_true + p_false
    fidelity = 0.97 - 0.13 * (1.0 - qq) - 0.00055 * bb
    fidelity = np.clip(fidelity, 0.5, 0.99)
    return fidelity, p_all


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )

    eta_q = np.linspace(0.3, 0.9, 180)
    bg_cps = np.linspace(0.0, 260.0, 180)
    fidelity, p_all = _metrics(eta_q, bg_cps)

    fig = plt.figure(figsize=(9.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    im0 = ax0.imshow(
        fidelity,
        origin="lower",
        aspect="auto",
        extent=[eta_q.min(), eta_q.max(), bg_cps.min(), bg_cps.max()],
        cmap="viridis",
        vmin=0.55,
        vmax=0.97,
    )
    cs0 = ax0.contour(
        eta_q,
        bg_cps,
        fidelity,
        levels=[0.70, 0.75, 0.80, 0.85, 0.90],
        colors="white",
        linewidths=0.9,
        alpha=0.8,
    )
    ax0.clabel(cs0, fmt="%.2f", inline=True, fontsize=8)
    ax0.set_xlabel(r"QFC efficiency $\eta_q$")
    ax0.set_ylabel("Background rate (cps)")
    ax0.set_title("Conditional fidelity map")
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.048, pad=0.03)
    cb0.set_label(r"$F_t$")

    im1 = ax1.imshow(
        1e6 * p_all,
        origin="lower",
        aspect="auto",
        extent=[eta_q.min(), eta_q.max(), bg_cps.min(), bg_cps.max()],
        cmap="magma",
    )
    cs1 = ax1.contour(
        eta_q,
        bg_cps,
        1e6 * p_all,
        levels=[0.2, 0.4, 0.8, 1.2, 1.6],
        colors="white",
        linewidths=0.9,
        alpha=0.8,
    )
    ax1.clabel(cs1, fmt="%.1f", inline=True, fontsize=8)
    ax1.set_xlabel(r"QFC efficiency $\eta_q$")
    ax1.set_ylabel("Background rate (cps)")
    ax1.set_title(r"Herald probability map ($\times10^{-6}$)")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.048, pad=0.03)
    cb1.set_label(r"$p_s \times 10^{-6}$")

    ax0.scatter([0.57], [165.0], s=55, marker="o", color="#f2c14e", edgecolors="#1f1f1f", linewidths=0.8)
    ax0.text(0.585, 172.0, "baseline", fontsize=8.5, color="#1f1f1f")
    ax1.scatter([0.57], [165.0], s=55, marker="o", color="#f2c14e", edgecolors="#1f1f1f", linewidths=0.8)

    fig.suptitle("QFC efficiency-background operating landscape", fontsize=12.5, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
