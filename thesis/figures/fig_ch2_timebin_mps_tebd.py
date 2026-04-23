import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


EDGE = "#1f2937"
TEXT = "#111827"
MUTED = "#475569"
BLUE = "#dbeafe"
BLUE_DARK = "#2563eb"
CYAN = "#cffafe"
CYAN_DARK = "#0891b2"
GREEN = "#bbf7d0"
GREEN_DARK = "#16a34a"
AMBER = "#fde68a"
AMBER_DARK = "#d97706"
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
    radius: float = 0.014,
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=TEXT, linespacing=1.18)
    return (x, y, w, h)


def _arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    color: str = MUTED,
    lw: float = 1.1,
    rad: float = 0.0,
    ms: float = 10.0,
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


def _panel(ax, x: float, y: float, w: float, h: float, label: str, title: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.024",
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        linewidth=0.95,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.035, label, ha="left", va="top", fontsize=10.0, color=TEXT)
    ax.text(x + 0.058, y + h - 0.035, title, ha="left", va="top", fontsize=9.4, color=TEXT)


def _draw_panel_a(ax, x: float, y: float, w: float, h: float) -> None:
    _panel(ax, x, y, w, h, "(a)", "连续输出场到时间 bin")
    xs = np.linspace(x + 0.045, x + 0.160, 160)
    center = x + 0.100
    wave = y + 0.530 * h + 0.080 * h * np.exp(-((xs - center) / 0.035) ** 2) * np.sin((xs - x) * 210)
    ax.plot(xs, wave, color=BLUE_DARK, linewidth=1.7)
    ax.text(x + 0.100, y + 0.690 * h, "连续输出场 $b(t)$", ha="center", va="center", fontsize=8.0, color=MUTED)

    bin_y = y + 0.260 * h
    bin_x0 = x + 0.045
    bin_w = 0.025
    for i in range(8):
        fc = CYAN if i == 3 else "#e0f2fe"
        ec = CYAN_DARK if i == 3 else "#93c5fd"
        ax.add_patch(Rectangle((bin_x0 + i * bin_w, bin_y), bin_w * 0.84, 0.130 * h, facecolor=fc, edgecolor=ec, linewidth=0.9))
    ax.text(bin_x0 + 3.5 * bin_w, bin_y - 0.045, "时间 bin 链", ha="center", va="center", fontsize=8.0, color=MUTED)
    ax.text(bin_x0 + 3.42 * bin_w, bin_y + 0.160 * h, "当前窗口", ha="center", va="center", fontsize=7.5, color=CYAN_DARK)

    _arrow(ax, (x + 0.115, y + 0.450 * h), (x + 0.123, y + 0.395 * h), color=CYAN_DARK, lw=1.0)
    ax.text(
        x + 0.187,
        y + 0.455 * h,
        r"$\Delta B_k \propto \int_{t_k}^{t_{k+1}} b(t)\,dt$",
        ha="center",
        va="center",
        fontsize=8.2,
        color=TEXT,
    )
    _box(ax, x + 0.205, y + 0.230 * h, 0.070, 0.060, "$\\Delta t$", fc=GRAY, ec="#cbd5e1", fontsize=7.9)


def _site(ax, x: float, y: float, text: str, *, fc: str, ec: str, w: float = 0.040, h: float = 0.060) -> None:
    _box(ax, x, y, w, h, text, fc=fc, ec=ec, fontsize=6.9, radius=0.010)


def _draw_panel_b(ax, x: float, y: float, w: float, h: float) -> None:
    _panel(ax, x, y, w, h, "(b)", "双臂 MPS 站点排布")
    row_y = [y + 0.570 * h, y + 0.330 * h]
    labels = ["A 臂", "B 臂"]
    for yy, lab in zip(row_y, labels):
        ax.text(x + 0.035, yy + 0.030, lab, ha="center", va="center", fontsize=7.8, color=MUTED)
        _site(ax, x + 0.050, yy, "emitter\n$d=12$", fc=BLUE, ec=BLUE_DARK, w=0.052, h=0.067)
        for i in range(4):
            _site(ax, x + 0.115 + i * 0.036, yy + 0.006, "bin", fc=CYAN, ec=CYAN_DARK, w=0.028, h=0.052)
        ax.text(x + 0.257, yy + 0.032, "...", ha="center", va="center", fontsize=9.0, color=MUTED)
        _site(ax, x + 0.270, yy + 0.006, "memory\n$d=3$", fc=AMBER, ec=AMBER_DARK, w=0.043, h=0.052)
        ax.plot([x + 0.102, x + 0.115], [yy + 0.034, yy + 0.034], color=MUTED, linewidth=1.0)
        ax.plot([x + 0.263, x + 0.270], [yy + 0.034, yy + 0.034], color=MUTED, linewidth=1.0)

    active = Rectangle((x + 0.183, y + 0.275 * h), 0.060, 0.405 * h, facecolor="#fef3c7", edgecolor="#f59e0b", linewidth=1.2, alpha=0.35)
    ax.add_patch(active)
    ax.text(x + 0.213, y + 0.710 * h, "活跃窗口", ha="center", va="center", fontsize=7.6, color=AMBER_DARK)
    legend = _box(ax, x + 0.055, y + 0.080 * h, 0.150, 0.083, "bin $d=5$: 真空 + 780/1517 nm", fc=GRAY, ec="#cbd5e1", fontsize=7.0)
    _box(ax, legend[0] + legend[2] + 0.010, y + 0.080 * h, 0.085, 0.083, "$\\chi$ 控制", fc=PURPLE, ec=PURPLE_DARK, fontsize=7.0)


def _gate(ax, x: float, y: float, w: float, text: str, *, fc: str, ec: str) -> None:
    _box(ax, x, y, w, 0.050, text, fc=fc, ec=ec, fontsize=7.4, radius=0.012)


def _draw_panel_c(ax, x: float, y: float, w: float, h: float) -> None:
    _panel(ax, x, y, w, h, "(c)", "单时间步局部门更新")
    site_y = y + 0.560 * h
    sites = [
        ("emitter", x + 0.030, BLUE, BLUE_DARK),
        ("bin $k$", x + 0.095, CYAN, CYAN_DARK),
        ("memory", x + 0.160, AMBER, AMBER_DARK),
        ("bin $k+1$", x + 0.225, CYAN, CYAN_DARK),
    ]
    for text, sx, fc, ec in sites:
        _site(ax, sx, site_y, text, fc=fc, ec=ec, w=0.050, h=0.062)
    for sx0, sx1 in ((0.080, 0.095), (0.145, 0.160), (0.210, 0.225)):
        ax.plot([x + sx0, x + sx1], [site_y + 0.031, site_y + 0.031], color=MUTED, linewidth=1.0)

    _gate(ax, x + 0.055, y + 0.415 * h, 0.082, "$U_{emit}$", fc=BLUE, ec=BLUE_DARK)
    _gate(ax, x + 0.105, y + 0.315 * h, 0.074, "$U_{QFC}$", fc=GREEN, ec=GREEN_DARK)
    _gate(ax, x + 0.140, y + 0.215 * h, 0.090, "$U_{filt}$", fc=AMBER, ec=AMBER_DARK)
    _gate(ax, x + 0.225, y + 0.315 * h, 0.058, "SWAP", fc=GRAY, ec="#94a3b8")
    _gate(ax, x + 0.075, y + 0.100 * h, 0.170, "SVD 回写与 $\\chi$ 截断", fc=PURPLE, ec=PURPLE_DARK)

    _arrow(ax, (x + 0.095, site_y), (x + 0.095, y + 0.465 * h), color=BLUE_DARK, lw=0.9)
    _arrow(ax, (x + 0.145, site_y), (x + 0.145, y + 0.365 * h), color=GREEN_DARK, lw=0.9)
    _arrow(ax, (x + 0.185, site_y), (x + 0.185, y + 0.265 * h), color=AMBER_DARK, lw=0.9)
    _arrow(ax, (x + 0.254, y + 0.315 * h), (x + 0.254, y + 0.205 * h), color=MUTED, lw=0.9)
    _arrow(ax, (x + 0.185, y + 0.215 * h), (x + 0.185, y + 0.150 * h), color=PURPLE_DARK, lw=0.9)
    ax.text(x + 0.160, y + 0.690 * h, "局域门作用", ha="center", va="center", fontsize=7.7, color=MUTED)
    ax.text(x + 0.160, y + 0.045 * h, "活跃区移动到下一 bin", ha="center", va="center", fontsize=7.3, color=MUTED)


def main() -> None:
    _setup_fonts()
    fig, ax = plt.subplots(figsize=(13.0, 4.9), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _draw_panel_a(ax, 0.025, 0.105, 0.295, 0.800)
    _draw_panel_b(ax, 0.350, 0.105, 0.315, 0.800)
    _draw_panel_c(ax, 0.695, 0.105, 0.280, 0.800)

    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".png"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
