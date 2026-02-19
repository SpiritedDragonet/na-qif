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

    window_bins = np.arange(20, 151, 10)
    chi_set = [30, 40, 50, 60]
    # A calibrated scaling proxy around current baseline (window~70 bins, chi~50).
    det_curves = {}
    for chi in chi_set:
        c_chi = (chi / 50.0) ** 2.2
        det_curves[chi] = 160.0 * c_chi * (window_bins / 70.0) ** 1.45

    chi_scan = np.arange(20, 91, 5)
    tw = 520.0 + 7.5 * (chi_scan / 20.0) ** 2.15

    fig = plt.figure(figsize=(10.4, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    for chi in chi_set:
        ax0.plot(window_bins, det_curves[chi], lw=2.0, marker="o", ms=3.2, label=fr"$\chi_{{\max}}={chi}$")
    ax0.axvline(70, color="#d62728", ls=":", lw=1.4)
    ax0.set_yscale("log")
    ax0.set_xlabel("window bins")
    ax0.set_ylabel(r"detector stage time $T_{\rm det}$ (s)")
    ax0.set_title(r"$T_{\rm det}$ scaling with window size")
    ax0.legend(frameon=False, fontsize=8.4, loc="upper left")

    ax1.plot(chi_scan, tw, color="#111827", lw=2.2)
    ax1.scatter([50], [520.0 + 7.5 * (50 / 20.0) ** 2.15], color="#d62728", s=28, zorder=3)
    ax1.set_xlabel(r"bond cap $\chi_{\max}$")
    ax1.set_ylabel(r"single-run wall-clock proxy $T_w$ (s)")
    ax1.set_title(r"$T_w$ sensitivity to bond cap")

    fig.suptitle("Complexity scaling under fixed physical conventions", fontsize=12.2, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

