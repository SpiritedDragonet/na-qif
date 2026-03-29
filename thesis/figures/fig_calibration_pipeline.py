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


def _box(ax, x, y, w, h, text, fc, *, fontsize: float = 8.6, bold: bool = False) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=EDGE,
        linewidth=1.05,
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
    )


def _arrow(ax, p0, p1, *, lw: float = 1.2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=11.5,
            linewidth=lw,
            color=ARROW,
        )
    )


def _column(ax, *, x: float, color: str, title: str, observable: str, params: str) -> None:
    w = 0.205
    _box(ax, x, 0.67, w, 0.17, title, color, fontsize=8.7, bold=True)
    _box(ax, x, 0.42, w, 0.15, observable, "#ffffff", fontsize=8.35)
    _box(ax, x, 0.17, w, 0.15, params, "#ffffff", fontsize=8.3)
    _arrow(ax, (x + w / 2.0, 0.67), (x + w / 2.0, 0.57))
    _arrow(ax, (x + w / 2.0, 0.42), (x + w / 2.0, 0.32))


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

    fig, ax = plt.subplots(figsize=(11.2, 5.7), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    xs = [0.03, 0.275, 0.52, 0.765]
    _column(
        ax,
        x=xs[0],
        color=DEVICE_COLORS["source"],
        title="源端驱动与腔发射",
        observable="观测量\n单臂波包峰位、宽度、尾部",
        params="锚定参数\n$\\Omega(t)$、$g$、$\\kappa$",
    )
    _column(
        ax,
        x=xs[1],
        color=DEVICE_COLORS["window"],
        title="时间窗与到达统计",
        observable="观测量\n绝对点击时序、$\\Delta t$ 直方图",
        params="锚定参数\n$\\Delta t_w$、符合口径",
    )
    _column(
        ax,
        x=xs[2],
        color=DEVICE_COLORS["hom"],
        title="两光子干涉",
        observable="观测量\nHOM 谷深、宽度、中心偏移",
        params="锚定参数\n$v_{\\mathrm{HOM}}$、$\\sigma_\\phi$、$\\tau_0$",
    )
    _column(
        ax,
        x=xs[3],
        color=DEVICE_COLORS["bsm"],
        title="BSM 与条件态",
        observable="观测量\n模式组成、$F_t$、CHSH、速率",
        params="锚定参数\n背景/暗计数、残余混合、工作点",
    )

    w = 0.205
    for left, right in zip(xs[:-1], xs[1:]):
        _arrow(ax, (left + w, 0.755), (right - 0.012, 0.755), lw=1.25)

    _box(
        ax,
        0.10,
        0.01,
        0.80,
        0.10,
        "层间传递规则：下一层只继承前一层已锚定的参数与不确定度，不回跳到全局重拟合。",
        DEVICE_COLORS["note"],
        fontsize=8.4,
    )

    ax.set_title("分层标定中的装置环节、观测量与参数锚定", fontsize=12.3, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
