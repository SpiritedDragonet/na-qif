import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _add_box(ax, xy, w, h, text, fc="#f8fafc", ec="#1f2937", fs=9.0):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _add_arrow(ax, p0, p1):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#374151",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(11.2, 4.2), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    y = 0.52
    w = 0.155
    h = 0.24
    xs = [0.03, 0.22, 0.41, 0.60, 0.79]

    _add_box(
        ax,
        (xs[0], y),
        w,
        h,
        "Input setup\n$\\theta, N, \\Delta t, \\Delta t_w$\n+ role-aligned config",
        fc="#e0f2fe",
    )
    _add_box(
        ax,
        (xs[1], y),
        w,
        h,
        "Emission TEBD\natom-cavity gate\nbin update + SWAP",
        fc="#dcfce7",
    )
    _add_box(
        ax,
        (xs[2], y),
        w,
        h,
        "QFC + memory gate\n$5\\mathrm{D}$ and $5\\times 3\\mathrm{D}$\nlocal step on bins",
        fc="#fef9c3",
    )
    _add_box(
        ax,
        (xs[3], y),
        w,
        h,
        "Fiber + BSM adjoint\nJones/PDL/phase\npushback to effects",
        fc="#fee2e2",
    )
    _add_box(
        ax,
        (xs[4], y),
        w,
        h,
        "POVM contraction\nsuccess split\n$\\{p_s,p_t,p_f,F_t,S\\}$",
        fc="#ede9fe",
    )

    for k in range(4):
        _add_arrow(ax, (xs[k] + w, y + h / 2), (xs[k + 1] - 0.01, y + h / 2))

    _add_box(
        ax,
        (0.20, 0.14),
        0.60,
        0.20,
        "Main loop over time bins: emission update $\\rightarrow$ memory step $\\rightarrow$ effect pushback\n"
        "Outputs include task metrics and intermediate observables (wavepacket, HOM, pattern distribution).",
        fc="#f8fafc",
        fs=8.9,
    )
    _add_arrow(ax, (0.50, y), (0.50, 0.35))

    ax.set_title("End-to-end simulation pipeline (single-run core)", fontsize=12.4, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

