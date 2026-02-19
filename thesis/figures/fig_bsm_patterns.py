import pathlib

import matplotlib.pyplot as plt
import numpy as np


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

    patterns = ["H1V2", "V1H2", "H1V1", "H2V2", "H1H2", "V1V2"]
    p_true = np.array([0.245, 0.238, 0.214, 0.209, 0.041, 0.038])
    p_dark = np.array([0.034, 0.031, 0.047, 0.046, 0.028, 0.027])
    p_bg = np.array([0.021, 0.018, 0.029, 0.028, 0.019, 0.018])

    x = np.arange(len(patterns))

    delta_bins = np.arange(0, 33)
    reliability = np.zeros((len(patterns), delta_bins.size))
    centers = np.array([9.0, 9.0, 5.0, 5.0, 3.0, 3.0])
    widths = np.array([6.5, 6.5, 4.8, 4.8, 3.0, 3.0])
    base = np.array([0.92, 0.92, 0.87, 0.86, 0.73, 0.73])
    floor = np.array([0.50, 0.50, 0.46, 0.46, 0.25, 0.25])
    for i in range(len(patterns)):
        g = np.exp(-0.5 * ((delta_bins - centers[i]) / widths[i]) ** 2)
        reliability[i] = floor[i] + (base[i] - floor[i]) * g

    fig = plt.figure(figsize=(9.4, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.bar(x, p_true, width=0.72, color="#1f77b4", label="true signal")
    ax0.bar(x, p_bg, width=0.72, bottom=p_true, color="#f28e2b", label="background-assisted")
    ax0.bar(x, p_dark, width=0.72, bottom=p_true + p_bg, color="#c44e52", label="intrinsic dark-assisted")
    ax0.set_xticks(x, patterns, rotation=0)
    ax0.set_ylabel("Pattern probability")
    ax0.set_xlabel("BSM click pattern")
    ax0.set_ylim(0.0, 0.36)
    ax0.set_title("Herald-pattern composition")
    ax0.legend(frameon=False, fontsize=8.6, loc="upper right")

    im = ax1.imshow(
        reliability,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.2,
        vmax=0.95,
        extent=[delta_bins.min(), delta_bins.max(), -0.5, len(patterns) - 0.5],
    )
    ax1.set_yticks(np.arange(len(patterns)), patterns)
    ax1.set_xlabel(r"$\Delta$bin")
    ax1.set_ylabel("Pattern")
    ax1.set_title(r"Record reliability $P(\mathrm{true}\mid \mathrm{pattern},\Delta\mathrm{bin})$")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.05, pad=0.03)
    cbar.set_label("True-herald probability")

    fig.suptitle("BSM pattern diagnostics with true/false decomposition", fontsize=12.4, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
