import pathlib

import matplotlib.pyplot as plt
import numpy as np


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
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
        }
    )

    window_bin_cap = np.array([40, 60, 80, 100, 120], dtype=float)
    det_runtime_proxy = 0.78 * window_bin_cap - 8.0
    wall_clock_proxy = 0.036 * window_bin_cap + 0.25

    fig = plt.figure(figsize=(9.6, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.27)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.plot(window_bin_cap, det_runtime_proxy, color="#1f77b4", lw=2.3)
    ax0.scatter(window_bin_cap, det_runtime_proxy, color="#1f77b4", s=26, zorder=3)
    ax0.set_xlabel("Window bin upper bound")
    ax0.set_ylabel("Detection-stage runtime (a.u.)")
    ax0.set_title(r"$T_{\rm det}$ vs window bin upper bound")
    _panel_label(ax0, "(a)")

    ax1.plot(window_bin_cap, wall_clock_proxy, color="#d62728", lw=2.3)
    ax1.scatter(window_bin_cap, wall_clock_proxy, color="#d62728", s=26, zorder=3)
    ax1.set_xlabel("Window bin upper bound")
    ax1.set_ylabel("Wall-clock proxy (a.u.)")
    ax1.set_title(r"$T_w$ proxy vs window bin upper bound")
    _panel_label(ax1, "(b)")

    fig.suptitle(
        "Complexity Scaling with Window Bin Upper Bound",
        fontsize=12.1,
        fontweight="bold",
    )

    base = pathlib.Path(__file__).with_suffix("")
    fig.savefig(base.with_suffix(".pdf"), dpi=240)
    fig.savefig(base.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
