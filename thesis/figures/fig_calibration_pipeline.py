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
        "波包锚定\n观测：峰值/宽度/尾部\n拟合：$\\Omega(t),g,\\kappa$",
        "#e0f2fe",
    )
    _box(
        ax,
        xs[1],
        y,
        w,
        h,
        "窗口层\n观测：点击直方图\n拟合：$\\Delta t_w$",
        "#dcfce7",
    )
    _box(
        ax,
        xs[2],
        y,
        w,
        h,
        "HOM层\n观测：谷深/宽度\n拟合：$v_{\\mathrm{HOM}},\\sigma_\\phi$",
        "#fef9c3",
    )
    _box(
        ax,
        xs[3],
        y,
        w,
        h,
        "BSM层\n观测：模式分布\n拟合：残余混合、背景占比",
        "#fee2e2",
    )
    _box(
        ax,
        xs[4],
        y,
        w,
        h,
        "任务指标\n验证：\n$\\{p_s,p_t,p_f,F_t,S\\}$",
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
        "分层目标：每一层只优化该层对应观测量，并将约束传递到下一层。\n"
        "该策略可避免全参数退化并提高残差不匹配的可审计性。",
        "#f8fafc",
    )
    _arrow(ax, (0.50, y), (0.50, 0.41))

    ax.set_title("分层标定流程与观测量-参数映射", fontsize=12.3, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

