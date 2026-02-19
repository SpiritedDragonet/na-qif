import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )

    eta = np.linspace(0.60, 0.95, 180)
    bg_levels = np.array([80.0, 120.0, 160.0, 200.0, 240.0])
    ee, bb = np.meshgrid(eta, bg_levels, indexing="xy")

    success = np.clip(1.55e-6 * ee**2 * np.exp(-0.0018 * (bb - 80.0)), 0.08e-6, None)
    fidelity = np.clip(0.79 + 0.17 * (ee - 0.60) / 0.35 - 0.00058 * (bb - 80.0), 0.58, 0.96)

    fig = plt.figure(figsize=(10.4, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    im = ax0.imshow(
        fidelity,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.58,
        vmax=0.96,
        extent=[eta.min(), eta.max(), bg_levels.min(), bg_levels.max()],
    )
    cs = ax0.contour(
        eta,
        bg_levels,
        fidelity,
        levels=[0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        colors="white",
        linewidths=0.9,
        alpha=0.85,
    )
    ax0.clabel(cs, fmt="%.2f", inline=True, fontsize=8)
    ax0.set_xlabel(r"detector efficiency $\eta_d$")
    ax0.set_ylabel("background rate (cps)")
    ax0.set_title(r"Conditional fidelity map $F_t(\eta_d,\mathrm{bg})$")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.05, pad=0.03)
    cbar.set_label(r"$F_t$")

    for i, bg in enumerate(bg_levels):
        ax1.plot(
            eta,
            1e6 * success[i, :],
            lw=2.0,
            label=f"bg={int(bg)} cps",
        )
    ax1.set_xlabel(r"detector efficiency $\eta_d$")
    ax1.set_ylabel(r"herald probability ($\times10^{-6}$)")
    ax1.set_title("Success-rate sensitivity")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("Detector efficiency and background: fidelity map with rate slices", fontsize=12.1, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
