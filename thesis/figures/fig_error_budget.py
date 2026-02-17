import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _waterfall_series():
    labels = ["ideal", "source", "QFC+mem", "fiber", "detector", "final"]
    # Normalized marginal degradation for a representative baseline.
    drops = [0.0, 0.028, 0.036, 0.018, 0.024, 0.0]
    ft_ideal = 0.945
    levels = [ft_ideal]
    for d in drops[1:-1]:
        levels.append(levels[-1] - d)
    levels.append(levels[-1])
    return labels, np.array(drops), np.array(levels)


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

    labels, drops, levels = _waterfall_series()
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(10.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    bar_bottom = np.r_[0.0, levels[:-2], 0.0]
    bar_height = np.r_[levels[0], -drops[1:-1], levels[-1]]
    colors = ["#0ea5e9", "#f97316", "#f97316", "#f97316", "#f97316", "#8b5cf6"]
    ax0.bar(x, bar_height, bottom=bar_bottom, color=colors, width=0.72, edgecolor="#111827", linewidth=0.8)
    ax0.plot(x, levels, "o-", color="#111827", lw=1.2, ms=4.5, label=r"$F_t$ trajectory")
    ax0.set_xticks(x, labels, rotation=18, ha="right")
    ax0.set_ylabel(r"conditional fidelity $F_t$")
    ax0.set_ylim(0.80, 0.97)
    ax0.set_title("Fidelity waterfall by noise groups")
    ax0.legend(frameon=False, fontsize=8.5, loc="lower left")

    false_components = np.array([0.52, 0.31, 0.17])
    false_labels = ["intrinsic dark", "background", "other trigger"]
    false_colors = ["#dc2626", "#f59e0b", "#64748b"]
    left = 0.0
    for frac, lab, col in zip(false_components, false_labels, false_colors):
        ax1.barh([0], [frac], left=left, color=col, edgecolor="#111827", linewidth=0.8, label=lab, height=0.46)
        ax1.text(left + frac / 2, 0, f"{100*frac:.0f}%", ha="center", va="center", fontsize=8.8, color="white")
        left += frac
    ax1.set_xlim(0.0, 1.0)
    ax1.set_yticks([0], ["$p_f$ decomposition"])
    ax1.set_xlabel("fraction")
    ax1.set_title("False-herald composition")
    ax1.legend(frameon=False, fontsize=8.4, loc="lower right")

    fig.suptitle("Error budget summary: fidelity waterfall and false-herald decomposition", fontsize=12.2, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

