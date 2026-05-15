import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEVICE_COLORS = {
    "source": "#dbeafe",
    "window": "#dcfce7",
    "hom": "#fef3c7",
    "bsm": "#fee2e2",
    "note": "#f8fafc",
}
EDGE = "#1f2937"
ARROW = "#475569"


def _box(ax, x, y, w, h, text, fc, *, fontsize: float = 17.0, bold: bool = False) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=EDGE,
        linewidth=1.15,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        linespacing=1.18,
    )


def _arrow(ax, p0, p1, *, lw: float = 1.35) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=14.0,
            linewidth=lw,
            color=ARROW,
        )
    )


def _column(ax, *, x: float, color: str, title: str, observable: str, params: str) -> None:
    w = 0.205
    _box(ax, x, 0.70, w, 0.18, title, color, fontsize=17.2, bold=True)
    _box(ax, x, 0.43, w, 0.18, observable, "#ffffff", fontsize=16.0)
    _box(ax, x, 0.16, w, 0.18, params, "#ffffff", fontsize=16.0)
    _arrow(ax, (x + w / 2.0, 0.70), (x + w / 2.0, 0.61))
    _arrow(ax, (x + w / 2.0, 0.43), (x + w / 2.0, 0.34))


def main() -> None:
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

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    xs = [0.025, 0.273, 0.521, 0.769]
    _column(
        ax,
        x=xs[0],
        color=DEVICE_COLORS["source"],
        title="源端驱动与腔发射",
        observable="观测量\n单臂波包\n峰位、宽度、尾部",
        params="锚定参数\n$\\Omega(t)$\n$g$、$\\kappa$",
    )
    _column(
        ax,
        x=xs[1],
        color=DEVICE_COLORS["window"],
        title="时间窗与到达统计",
        observable="观测量\n绝对点击时序\n$\\Delta t$ 直方图",
        params="锚定参数\n$\\Delta t_w$\n符合口径",
    )
    _column(
        ax,
        x=xs[2],
        color=DEVICE_COLORS["hom"],
        title="两光子干涉",
        observable="观测量\nHOM 谷深、宽度\n中心偏移",
        params="锚定参数\n$v_{\\mathrm{HOM}}$\n$\\sigma_\\phi$、$\\tau_0$",
    )
    _column(
        ax,
        x=xs[3],
        color=DEVICE_COLORS["bsm"],
        title="BSM 与条件态",
        observable="观测量\n模式组成、$F_t$\nCHSH、速率",
        params="锚定参数\n背景/暗计数\n残余混合、工作点",
    )

    w = 0.205
    for left, right in zip(xs[:-1], xs[1:]):
        _arrow(ax, (left + w + 0.004, 0.79), (right - 0.018, 0.79), lw=1.45)

    _box(
        ax,
        0.05,
        0.025,
        0.90,
        0.085,
        "层间传递规则：继承已锚定参数与不确定度，固定进入后续拟合。",
        DEVICE_COLORS["note"],
        fontsize=14.6,
    )

    ax.set_title("分层标定中的装置环节、观测量与参数锚定", fontsize=21.0, fontweight="bold", pad=5)
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
