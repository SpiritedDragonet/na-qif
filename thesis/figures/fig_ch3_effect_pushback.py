import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


EDGE = "#1f2937"
TEXT = "#111827"
MUTED = "#475569"
ORANGE = "#ffedd5"
ORANGE_DARK = "#ea580c"
AMBER = "#fef3c7"
AMBER_DARK = "#d97706"
GRAY = "#f8fafc"
GRAY_DARK = "#64748b"
PURPLE = "#ede9fe"
PURPLE_DARK = "#7c3aed"
BLUE = "#dbeafe"
BLUE_DARK = "#2563eb"
GREEN = "#dcfce7"
GREEN_DARK = "#16a34a"


def _setup_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    formula: str,
    subtitle: str,
    fc: str,
    ec: str,
    formula_size: float = 13.0,
    subtitle_size: float = 7.5,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, formula, ha="center", va="center", fontsize=formula_size, color=TEXT)
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=subtitle_size, color=MUTED)


def _simple_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str,
    fontsize: float = 9.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.20,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=TEXT, linespacing=1.15)


def _arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    label: str | None = None,
    color: str = GRAY_DARK,
    lw: float = 1.25,
    rad: float = 0.0,
    label_y: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text(
            (p0[0] + p1[0]) / 2,
            (p0[1] + p1[1]) / 2 + label_y,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color=MUTED,
        )


def main() -> None:
    _setup_fonts()
    fig, ax = plt.subplots(figsize=(13.2, 4.2), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(0.035, 0.930, "测量侧效应算符推回", ha="left", va="center", fontsize=13.5, color=TEXT)
    ax.text(
        0.965,
        0.930,
        r"$E_m^{in}=\Phi^\dagger(F_m),\quad \Phi=\mathcal{M}_{det}\circ Ad_{U_{BS}}\circ\mathcal{L}\circ\mathcal{V}$",
        ha="right",
        va="center",
        fontsize=12.0,
        color=TEXT,
    )

    y = 0.555
    w = 0.128
    h = 0.170
    xs = [0.035, 0.198, 0.361, 0.524, 0.687, 0.850]
    nodes = [
        (r"$F_m$", "探测端记录效应", ORANGE, ORANGE_DARK, 15.0),
        (r"$\mathcal{M}_{det}^{\dagger}(F_m)$", "探测器响应", AMBER, AMBER_DARK, 10.8),
        (r"$Ad_{U_{BS}}^{\dagger}(\cdot)$", "分束器基变换", GRAY, GRAY_DARK, 11.3),
        (r"$\mathcal{L}^{\dagger}(\cdot)$", "光纤损耗与偏振", GRAY, GRAY_DARK, 12.4),
        (r"$V^{\dagger}(\cdot)V$", "嵌入空间回投影", GREEN, GREEN_DARK, 12.4),
        (r"$E_m^{in}$", "输入端效应", PURPLE, PURPLE_DARK, 15.0),
    ]
    for x, (formula, subtitle, fc, ec, size) in zip(xs, nodes):
        _box(ax, x, y, w, h, formula=formula, subtitle=subtitle, fc=fc, ec=ec, formula_size=size)

    for i in range(len(xs) - 1):
        _arrow(
            ax,
            (xs[i] + w + 0.006, y + h / 2),
            (xs[i + 1] - 0.006, y + h / 2),
        )

    _simple_box(
        ax,
        0.090,
        0.245,
        0.205,
        0.118,
        "状态侧输出\n$\\rho_{in}$",
        fc=BLUE,
        ec=BLUE_DARK,
        fontsize=10.0,
    )
    _simple_box(
        ax,
        0.650,
        0.220,
        0.280,
        0.168,
        "$p_m=Tr(E_m^{in}\\rho_{in})$\n条件概率与后验权重入口",
        fc="#ffffff",
        ec=EDGE,
        fontsize=10.2,
    )
    _arrow(ax, (0.748, y), (0.765, 0.388), color=PURPLE_DARK, lw=1.20, rad=-0.10)
    _arrow(ax, (0.295, 0.305), (0.650, 0.305), color=BLUE_DARK, lw=1.20, label="MPS 收缩输入", label_y=0.035)

    ax.text(
        0.500,
        0.080,
        "物理正向通道从输入端走向探测端；效应算符推回按相反顺序把点击记录写成输入端 POVM。图中主线只表示测量侧，状态侧通过收缩公式接入。",
        ha="center",
        va="center",
        fontsize=8.6,
        color=MUTED,
    )

    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".png"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
