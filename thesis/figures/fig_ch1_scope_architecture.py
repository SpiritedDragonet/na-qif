import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


EDGE = "#1f2937"
TEXT = "#111827"
MUTED = "#475569"
BLUE = "#dbeafe"
BLUE_DARK = "#2563eb"
CYAN = "#dcfce7"
GREEN = "#16a34a"
ORANGE = "#fb923c"
ORANGE_LIGHT = "#ffedd5"
PURPLE = "#ede9fe"
PURPLE_DARK = "#7c3aed"
GRAY = "#f8fafc"
GRAY_DARK = "#64748b"


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
    lw: float = 1.0,
    fontsize: float = 8.4,
    radius: float = 0.018,
    color: str = TEXT,
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
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color)
    return (x, y, w, h)


def _arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    color: str = MUTED,
    lw: float = 1.15,
    rad: float = 0.0,
    ms: float = 12.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def _band(ax, x: float, y: float, w: float, h: float, title: str, fc: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=fc,
        edgecolor="#cbd5e1",
        linewidth=0.95,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.035, title, ha="left", va="top", fontsize=10.2, color=TEXT)


def _atom_array(ax, x: float, y: float, cols: int = 4, rows: int = 3, *, color: str = BLUE_DARK) -> None:
    dx = 0.018
    dy = 0.021
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Circle((x + c * dx, y + r * dy), 0.0055, facecolor="#bfdbfe", edgecolor=color, linewidth=0.8))


def _node(ax, x: float, y: float, label: str) -> dict[str, tuple[float, float]]:
    _box(ax, x, y, 0.205, 0.215, label, fc="#ffffff", ec="#94a3b8", fontsize=8.5, radius=0.022)
    ax.text(x + 0.032, y + 0.168, "本地寄存器", fontsize=7.3, ha="left", va="center", color=MUTED)
    _atom_array(ax, x + 0.035, y + 0.090)
    ax.add_patch(Circle((x + 0.138, y + 0.112), 0.018, facecolor="#bbf7d0", edgecolor=GREEN, linewidth=1.2))
    ax.text(x + 0.138, y + 0.078, "通信原子", fontsize=7.1, ha="center", va="center", color=MUTED)
    ax.add_patch(Rectangle((x + 0.164, y + 0.098), 0.010, 0.045, facecolor="#e2e8f0", edgecolor=EDGE, linewidth=0.8))
    ax.add_patch(Rectangle((x + 0.184, y + 0.098), 0.010, 0.045, facecolor="#e2e8f0", edgecolor=EDGE, linewidth=0.8))
    ax.plot([x + 0.174, x + 0.184], [y + 0.120, y + 0.120], color=ORANGE, linewidth=2.3)
    ax.text(x + 0.181, y + 0.155, "腔接口", fontsize=7.1, ha="center", va="center", color=MUTED)
    return {
        "port": (x + 0.197, y + 0.120),
        "register": (x + 0.070, y + 0.120),
        "comm": (x + 0.138, y + 0.112),
    }


def main() -> None:
    _setup_fonts()

    fig, ax = plt.subplots(figsize=(12.4, 5.7), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _band(ax, 0.025, 0.115, 0.255, 0.79, "可扩展性压力", "#f8fafc")
    _band(ax, 0.315, 0.115, 0.395, 0.79, "模块化 / 分布式体系结构", "#f8fbff")
    _band(ax, 0.745, 0.115, 0.230, 0.79, "量子接口任务边界", "#fcfbff")

    pressure_items = [
        ("物理比特规模", 0.735),
        ("并行测量吞吐", 0.635),
        ("经典反馈延迟", 0.535),
        ("标定维护成本", 0.435),
        ("误差传播控制", 0.335),
    ]
    target = _box(ax, 0.105, 0.205, 0.095, 0.070, "模块化\n需求", fc=BLUE, ec=BLUE_DARK, fontsize=8.1)
    for text, y in pressure_items:
        b = _box(ax, 0.060, y, 0.170, 0.060, text, fc="#ffffff", ec="#cbd5e1", fontsize=8.0)
        _arrow(ax, (b[0] + b[2] / 2, b[1]), (target[0] + target[2] / 2, target[1] + target[3]), color=GRAY_DARK, lw=0.85)

    _arrow(ax, (0.205, 0.240), (0.326, 0.505), color=BLUE_DARK, lw=1.4, rad=-0.12)
    ax.text(0.285, 0.395, "拆分为可维护节点", fontsize=7.8, ha="center", va="center", color=BLUE_DARK, rotation=19)

    node_a = _node(ax, 0.345, 0.470, "处理节点 A")
    node_b = _node(ax, 0.345, 0.205, "处理节点 B")
    _box(ax, 0.565, 0.360, 0.105, 0.070, "光子信道", fc=ORANGE_LIGHT, ec=ORANGE, fontsize=8.0)
    _arrow(ax, node_a["port"], (0.565, 0.395), color=ORANGE, lw=2.0, rad=-0.12)
    _arrow(ax, node_b["port"], (0.565, 0.395), color=ORANGE, lw=2.0, rad=0.12)

    ax.add_patch(Circle((0.635, 0.540), 0.017, facecolor=PURPLE, edgecolor=PURPLE_DARK, linewidth=1.2))
    ax.add_patch(Circle((0.635, 0.255), 0.017, facecolor=PURPLE, edgecolor=PURPLE_DARK, linewidth=1.2))
    ax.plot([0.635, 0.635], [0.272, 0.523], color=PURPLE_DARK, linewidth=1.7, linestyle=(0, (4, 3)))
    ax.text(0.650, 0.398, "可宣告\nBell 对", fontsize=8.0, ha="left", va="center", color=PURPLE_DARK)

    _box(
        ax,
        0.445,
        0.790,
        0.170,
        0.060,
        "本地门、纠错、读出\n留在节点内部",
        fc="#eef2ff",
        ec="#a5b4fc",
        fontsize=7.6,
    )
    _arrow(ax, (0.530, 0.790), (0.530, 0.690), color="#6366f1", lw=0.9)

    main_task = _box(
        ax,
        0.795,
        0.515,
        0.135,
        0.090,
        "远程纠缠产生\n本文主线",
        fc="#fed7aa",
        ec=ORANGE,
        fontsize=8.5,
    )
    side_tasks = [
        ("态转移 / 存储", 0.790, 0.720, (0.835, 0.605), (0.842, 0.720), 0.12),
        ("远程门", 0.865, 0.650, (0.900, 0.595), (0.920, 0.650), -0.10),
        ("错误探测", 0.865, 0.370, (0.900, 0.515), (0.920, 0.425), 0.10),
        ("资源注入", 0.790, 0.300, (0.835, 0.515), (0.842, 0.355), -0.12),
    ]
    for text, x, y, p0, p1, rad in side_tasks:
        _box(ax, x, y, 0.105, 0.055, text, fc="#ffffff", ec="#c4b5fd", fontsize=7.6)
        _arrow(ax, p0, p1, color=PURPLE_DARK, lw=0.75, ms=8.0, rad=rad)

    _arrow(ax, (0.670, 0.398), (main_task[0], main_task[1] + main_task[3] / 2), color=ORANGE, lw=1.8)
    _box(
        ax,
        0.765,
        0.170,
        0.185,
        0.070,
        "本文对象：中性原子\n远程纠缠接口链路",
        fc="#fef3c7",
        ec="#f59e0b",
        fontsize=8.0,
    )

    ax.text(
        0.500,
        0.055,
        "体系结构位置：可扩展计算压力  ->  模块化节点  ->  光子接口产生远程 Bell 对",
        ha="center",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )

    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".png"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
