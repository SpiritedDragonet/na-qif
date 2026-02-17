import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, text, fc):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc,
        edgecolor="#1f2937",
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.7)


def _arrow(ax, p0, p1):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#374151",
        )
    )


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    w = 0.18
    h = 0.22
    y = 0.56
    xs = [0.03, 0.23, 0.43, 0.63, 0.83]

    _box(
        ax,
        xs[0],
        y,
        w,
        h,
        "Wavepacket anchor\nObs: peak/width/tail\nFit: $\\Omega(t),g,\\kappa$",
        "#e0f2fe",
    )
    _box(
        ax,
        xs[1],
        y,
        w,
        h,
        "Window layer\nObs: click histogram\nFit: $\\Delta t_w$",
        "#dcfce7",
    )
    _box(
        ax,
        xs[2],
        y,
        w,
        h,
        "HOM layer\nObs: dip depth/width\nFit: $v_{\\mathrm{HOM}},\\sigma_\\phi$",
        "#fef9c3",
    )
    _box(
        ax,
        xs[3],
        y,
        w,
        h,
        "BSM layer\nObs: mode patterns\nFit: residual mix, bg split",
        "#fee2e2",
    )
    _box(
        ax,
        xs[4],
        y,
        w,
        h,
        "Task metrics\nValidate:\n$\\{p_s,p_t,p_f,F_t,S\\}$",
        "#ede9fe",
    )

    for i in range(4):
        _arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1] - 0.01, y + h / 2))

    _box(
        ax,
        0.12,
        0.16,
        0.76,
        0.24,
        "Layered objective: each stage optimizes only stage-matched observables, then passes constraints downstream.\n"
        "This avoids full-parameter degeneracy and improves auditability of residual mismatch.",
        "#f8fafc",
    )
    _arrow(ax, (0.50, y), (0.50, 0.41))

    ax.set_title("Layered calibration workflow and observable-to-parameter mapping", fontsize=12.3, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

