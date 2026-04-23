import math
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


EDGE = "#1f2937"
TEXT = "#111827"
MUTED = "#475569"
BLUE = "#dbeafe"
BLUE_DARK = "#2563eb"
GREEN = "#bbf7d0"
GREEN_DARK = "#16a34a"
ORANGE = "#fb923c"
ORANGE_LIGHT = "#ffedd5"
RED = "#fecaca"
RED_DARK = "#dc2626"
PURPLE = "#ede9fe"
PURPLE_DARK = "#7c3aed"
GRAY = "#f8fafc"


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
    text: str,
    *,
    fc: str = GRAY,
    ec: str = EDGE,
    fontsize: float = 8.0,
    lw: float = 1.0,
    radius: float = 0.018,
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=TEXT)
    return (x, y, w, h)


def _arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    color: str = MUTED,
    lw: float = 1.15,
    rad: float = 0.0,
    ms: float = 11.0,
    style: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def _section(ax, x: float, y: float, w: float, h: float, title: str, fc: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.026",
        facecolor=fc,
        edgecolor="#cbd5e1",
        linewidth=0.95,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.032, title, ha="left", va="top", fontsize=10.0, color=TEXT)


def _node_icon(ax, x: float, y: float, label: str, *, small: bool = False) -> tuple[float, float]:
    r = 0.020 if small else 0.026
    ax.add_patch(Circle((x, y), r, facecolor=BLUE, edgecolor=BLUE_DARK, linewidth=1.1))
    ax.text(x, y - (0.040 if small else 0.050), label, ha="center", va="center", fontsize=7.3, color=MUTED)
    return (x, y)


def _detector(ax, x: float, y: float, label: str) -> None:
    ax.add_patch(Rectangle((x, y), 0.022, 0.032, facecolor="#e2e8f0", edgecolor=EDGE, linewidth=0.8))
    ax.text(x + 0.011, y + 0.016, label, ha="center", va="center", fontsize=6.3, color=TEXT)


def _cavity(ax, x: float, y: float) -> None:
    ax.add_patch(Rectangle((x - 0.016, y - 0.033), 0.008, 0.066, facecolor="#cbd5e1", edgecolor=EDGE, linewidth=0.8))
    ax.add_patch(Rectangle((x + 0.008, y - 0.033), 0.008, 0.066, facecolor="#cbd5e1", edgecolor=EDGE, linewidth=0.8))
    ax.add_patch(Circle((x, y), 0.018, facecolor="#bfdbfe", edgecolor=BLUE_DARK, linewidth=1.0))


def _qfc(ax, x: float, y: float, *, direction: str = "right") -> None:
    _box(ax, x - 0.036, y - 0.030, 0.072, 0.060, "QFC", fc=GREEN, ec=GREEN_DARK, fontsize=7.8)
    if direction == "right":
        _arrow(ax, (x - 0.056, y), (x - 0.036, y), color=BLUE_DARK, lw=1.6, ms=8.5)
        _arrow(ax, (x + 0.036, y), (x + 0.060, y), color=ORANGE, lw=1.9, ms=8.5)
    else:
        _arrow(ax, (x + 0.056, y), (x + 0.036, y), color=BLUE_DARK, lw=1.6, ms=8.5)
        _arrow(ax, (x - 0.036, y), (x - 0.060, y), color=ORANGE, lw=1.9, ms=8.5)
    ax.text(x, y + 0.044, "780 -> 1517 nm", fontsize=6.7, ha="center", va="center", color=MUTED)


def _fiber(ax, x0: float, x1: float, y: float, *, label: str, direction: str) -> None:
    xs = [x0 + (x1 - x0) * i / 80 for i in range(81)]
    ys = [y + 0.010 * math.sin(i * math.pi / 5) for i in range(81)]
    ax.plot(xs, ys, color=ORANGE, linewidth=2.1)
    if direction == "right":
        _arrow(ax, (x1 - 0.030, y), (x1 - 0.004, y), color=ORANGE, lw=1.5, ms=8.5)
    else:
        _arrow(ax, (x0 + 0.030, y), (x0 + 0.004, y), color=ORANGE, lw=1.5, ms=8.5)
    ax.text((x0 + x1) / 2, y - 0.035, label, ha="center", va="center", fontsize=7.2, color=MUTED)


def _top_panel(ax) -> None:
    _section(ax, 0.035, 0.670, 0.430, 0.250, "方案选择层", "#f8fafc")
    _section(ax, 0.535, 0.670, 0.430, 0.250, "本文采用路线", "#fff7ed")

    _node_icon(ax, 0.095, 0.790, "节点 A", small=True)
    _node_icon(ax, 0.260, 0.790, "节点 B", small=True)
    _box(ax, 0.165, 0.745, 0.065, 0.055, "单击\n宣告", fc="#ffffff", ec="#cbd5e1", fontsize=7.1)
    _arrow(ax, (0.115, 0.790), (0.165, 0.772), color=ORANGE, lw=1.4)
    _arrow(ax, (0.240, 0.790), (0.230, 0.772), color=ORANGE, lw=1.4)
    _box(ax, 0.070, 0.690, 0.105, 0.050, "相位参考\n进入约束", fc=PURPLE, ec=PURPLE_DARK, fontsize=7.1)
    _box(ax, 0.240, 0.690, 0.120, 0.050, "链路相位噪声\n压低可见度", fc="#ffffff", ec="#cbd5e1", fontsize=7.0)

    _node_icon(ax, 0.595, 0.790, "节点 A", small=True)
    _node_icon(ax, 0.830, 0.790, "节点 B", small=True)
    bsm = _box(ax, 0.690, 0.744, 0.080, 0.064, "部分\nBSM", fc=RED, ec=RED_DARK, fontsize=7.4)
    _arrow(ax, (0.615, 0.790), (bsm[0], 0.776), color=ORANGE, lw=1.6)
    _arrow(ax, (0.810, 0.790), (bsm[0] + bsm[2], 0.776), color=ORANGE, lw=1.6)
    _box(ax, 0.590, 0.690, 0.125, 0.050, "时频 / 偏振 / 空间\n模式匹配", fc="#ffffff", ec="#fdba74", fontsize=6.9)
    _box(ax, 0.775, 0.690, 0.105, 0.050, "符合点击\n宣告成功", fc=ORANGE_LIGHT, ec=ORANGE, fontsize=7.1)
    _box(ax, 0.890, 0.735, 0.055, 0.065, "本文\n采用", fc="#fef3c7", ec="#f59e0b", fontsize=7.2)
    _arrow(ax, (0.770, 0.776), (0.890, 0.768), color=ORANGE, lw=1.2)


def _bottom_panel(ax) -> None:
    _section(ax, 0.035, 0.090, 0.930, 0.515, "本文端到端链路对象", "#f8fbff")

    ax.text(0.110, 0.548, "节点 A", ha="center", va="center", fontsize=8.6, color=TEXT)
    ax.text(0.890, 0.548, "节点 B", ha="center", va="center", fontsize=8.6, color=TEXT)
    ax.text(0.500, 0.548, "中继测量站", ha="center", va="center", fontsize=8.6, color=TEXT)

    for x, side in ((0.120, "A"), (0.880, "B")):
        _cavity(ax, x, 0.425)
        ax.text(x, 0.485, f"Rb 原子-腔\n节点 {side}", ha="center", va="center", fontsize=7.2, color=MUTED)
        _qfc(ax, x + (0.110 if side == "A" else -0.110), 0.425, direction=("right" if side == "A" else "left"))

    _fiber(ax, 0.265, 0.415, 0.425, label="电信光纤", direction="right")
    _fiber(ax, 0.585, 0.735, 0.425, label="电信光纤", direction="left")

    _box(ax, 0.435, 0.390, 0.055, 0.070, "BS", fc="#fef3c7", ec="#f59e0b", fontsize=8.0)
    _box(ax, 0.505, 0.390, 0.075, 0.070, "偏振分析\n+ 滤波", fc=RED, ec=RED_DARK, fontsize=7.0)
    _arrow(ax, (0.415, 0.425), (0.435, 0.425), color=ORANGE, lw=1.8)
    _arrow(ax, (0.565, 0.425), (0.585, 0.425), color=ORANGE, lw=1.8, style="<|-")
    _arrow(ax, (0.490, 0.425), (0.505, 0.425), color=ORANGE, lw=1.5)

    for i, (dy, label) in enumerate(((0.050, "H1"), (0.016, "V1"), (-0.018, "H2"), (-0.052, "V2"))):
        _arrow(ax, (0.580, 0.425), (0.635, 0.425 + dy), color="#334155", lw=1.0, rad=(i - 1.5) * 0.06)
        _detector(ax, 0.640, 0.409 + dy, label)

    _box(ax, 0.390, 0.250, 0.105, 0.055, "两光子干涉", fc=ORANGE_LIGHT, ec=ORANGE, fontsize=7.4)
    _box(ax, 0.510, 0.250, 0.115, 0.055, "partial BSM\n成功模式", fc=RED, ec=RED_DARK, fontsize=7.2)
    _arrow(ax, (0.495, 0.277), (0.510, 0.277), color=ORANGE, lw=1.0)

    stage_y = 0.140
    stages = [
        (0.055, 0.190, "源端发射\n与波包整形", BLUE, BLUE_DARK),
        (0.205, 0.135, "QFC\n与滤波", GREEN, GREEN_DARK),
        (0.350, 0.300, "光纤传播\n与链路损耗", ORANGE_LIGHT, ORANGE),
        (0.665, 0.250, "中继站干涉\n与探测", RED, RED_DARK),
    ]
    for x, w, text, fc, ec in stages:
        _box(ax, x, stage_y, w, 0.060, text, fc=fc, ec=ec, fontsize=7.0)

    ax.text(
        0.500,
        0.050,
        "建模边界：状态侧保留发射、QFC 和滤波记忆；测量侧处理光纤、分束器、探测器与成功效应",
        ha="center",
        va="center",
        fontsize=8.0,
        color=MUTED,
    )


def main() -> None:
    _setup_fonts()
    fig, ax = plt.subplots(figsize=(13.0, 7.0), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _top_panel(ax)
    _bottom_panel(ax)

    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".png"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
