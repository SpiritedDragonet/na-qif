import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


def _add_box(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = "#f8fafc",
    ec: str = "#1f2937",
    fs: float = 8.7,
    lw: float = 1.1,
) -> tuple[float, float, float, float]:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fs)
    return (x, y, w, h)


def _add_decision(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = "#f8fafc",
    ec: str = "#1f2937",
    fs: float = 8.5,
    lw: float = 1.1,
) -> tuple[float, float, float, float]:
    cx = x + w / 2.0
    cy = y + h / 2.0
    diamond = Polygon(
        [(cx, y + h), (x + w, cy), (cx, y), (x, cy)],
        closed=True,
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)
    return (x, y, w, h)


def _add_lane(
    ax,
    *,
    y0: float,
    y1: float,
    label: str,
    fc: str,
) -> None:
    lane = FancyBboxPatch(
        (0.02, y0),
        0.96,
        y1 - y0,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        linewidth=0.9,
        edgecolor="#cbd5e1",
        facecolor=fc,
    )
    ax.add_patch(lane)
    ax.text(
        0.03,
        y1 - 0.03,
        label,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#334155",
    )


def _right(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2], box[1] + box[3] / 2.0)


def _left(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0], box[1] + box[3] / 2.0)


def _top(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3])


def _bottom(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1])


def _add_arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    rad: float = 0.0,
    color: str = "#334155",
    lw: float = 1.1,
    ls: str = "-",
) -> None:
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(13.0, 6.2), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _add_lane(
        ax,
        y0=0.56,
        y1=0.94,
        label="State-side physics pipeline",
        fc="#f8fbff",
    )
    _add_lane(
        ax,
        y0=0.20,
        y1=0.53,
        label="Measurement-side statistics pipeline",
        fc="#fcfcff",
    )

    state_y = 0.66
    state_h = 0.17
    state_w = 0.145
    state_xs = [0.05, 0.225, 0.40, 0.575, 0.75]

    b_in = _add_box(
        ax,
        x=state_xs[0],
        y=state_y,
        w=state_w,
        h=state_h,
        text="Input setup\n$n_{bins},\\,\\Delta t,\\,\\chi_{max}$\nnoise + scan parameters",
        fc="#e0f2fe",
    )
    b_em = _add_box(
        ax,
        x=state_xs[1],
        y=state_y,
        w=state_w,
        h=state_h,
        text="Emission TEBD\natom-cavity gates\nbin evolution + SWAP",
        fc="#dcfce7",
    )
    b_qfc = _add_box(
        ax,
        x=state_xs[2],
        y=state_y,
        w=state_w,
        h=state_h,
        text="QFC + filter memory\n$5\\mathrm{D}$ with $5\\times3\\mathrm{D}$\nstate-side local updates",
        fc="#fef9c3",
    )
    b_fiber = _add_box(
        ax,
        x=state_xs[3],
        y=state_y,
        w=state_w,
        h=state_h,
        text="Fiber realization\nJones / PDL / phase\nHeisenberg parameters",
        fc="#fee2e2",
    )
    b_dephase = _add_box(
        ax,
        x=state_xs[4],
        y=state_y,
        w=state_w,
        h=state_h,
        text="Atomic dephasing\n$T_{wait}, T_2 \\rightarrow p_{dephase}$\nstate update",
        fc="#fde68a",
    )

    for left_box, right_box in ((b_in, b_em), (b_em, b_qfc), (b_qfc, b_fiber), (b_fiber, b_dephase)):
        _add_arrow(ax, _right(left_box), (right_box[0] - 0.01, _left(right_box)[1]))

    meas_y = 0.30
    meas_h = 0.15
    meas_w = 0.12
    tail_w = 0.10
    # Measurement lane is arranged right-to-left to keep cross-lane arrows short.
    meas_xs = [0.86, 0.71, 0.56, 0.41]

    b_arrive = _add_box(
        ax,
        x=meas_xs[0],
        y=meas_y,
        w=meas_w,
        h=meas_h,
        text="Arrival statistics\n$p_{arrive},\\,p_{11}$\n+ same-arm terms",
        fc="#e2e8f0",
    )
    b_effect = _add_box(
        ax,
        x=meas_xs[1],
        y=meas_y,
        w=meas_w,
        h=meas_h,
        text="POVM effects build\nper-bin $5\\mathrm{D}$ effects\n$+\\,v_{res}$",
        fc="#ede9fe",
    )
    b_bg = _add_box(
        ax,
        x=meas_xs[2],
        y=meas_y,
        w=meas_w,
        h=meas_h,
        text="Background OR map\ndetector-side\nmerge",
        fc="#ffe4e6",
    )
    b_enum = _add_box(
        ax,
        x=meas_xs[3],
        y=meas_y,
        w=meas_w,
        h=meas_h,
        text="Success enumeration\nStage 1-4\nall / true / all-signal / raw+ff",
        fc="#f3e8ff",
    )
    b_sample = _add_box(
        ax,
        x=0.13,
        y=meas_y,
        w=tail_w,
        h=meas_h,
        text="Sampling + gating\n$n_{samples}$\nwindow_bins",
        fc="#ccfbf1",
    )
    b_mode_sel = _add_decision(
        ax,
        x=0.355,
        y=0.315,
        w=0.050,
        h=0.120,
        text="enum\nmode",
        fc="#f8fafc",
        fs=8.3,
    )
    b_no_dark = _add_box(
        ax,
        x=0.245,
        y=0.385,
        w=0.090,
        h=0.065,
        text="no dark\nbranch",
        fc="#ecfeff",
        fs=8.2,
    )
    b_dark = _add_box(
        ax,
        x=0.245,
        y=0.255,
        w=0.090,
        h=0.065,
        text="dark\nbranch",
        fc="#fff7ed",
        fs=8.2,
    )
    b_out = _add_box(
        ax,
        x=0.01,
        y=meas_y,
        w=tail_w,
        h=meas_h,
        text="Outputs\n$p_s,p_t,p_f,p_{t|11}$\n$F_t,S_{max},\\rho_{declared}$",
        fc="#dbeafe",
    )

    for right_box, left_box in (
        (b_arrive, b_effect),
        (b_effect, b_bg),
        (b_bg, b_enum),
    ):
        _add_arrow(ax, _left(right_box), (left_box[0] + left_box[2] + 0.01, _right(left_box)[1]))

    _add_arrow(ax, _left(b_enum), (b_mode_sel[0] + b_mode_sel[2], b_mode_sel[1] + b_mode_sel[3] / 2.0))
    _add_arrow(
        ax,
        (b_mode_sel[0], b_mode_sel[1] + b_mode_sel[3] * 0.72),
        _right(b_no_dark),
        rad=0.06,
    )
    _add_arrow(
        ax,
        (b_mode_sel[0], b_mode_sel[1] + b_mode_sel[3] * 0.28),
        _right(b_dark),
        rad=-0.06,
    )
    _add_arrow(
        ax,
        _left(b_no_dark),
        (b_sample[0] + b_sample[2] + 0.01, b_sample[1] + b_sample[3] * 0.62),
        rad=-0.10,
    )
    _add_arrow(
        ax,
        _left(b_dark),
        (b_sample[0] + b_sample[2] + 0.01, b_sample[1] + b_sample[3] * 0.38),
        rad=0.10,
    )
    _add_arrow(ax, _left(b_sample), (b_out[0] + b_out[2] + 0.01, _right(b_out)[1]))
    _add_arrow(ax, (b_dephase[0] + b_dephase[2] * 0.20, b_dephase[1]), _top(b_arrive), rad=0.05, color="#475569")

    ax.set_title(
        "End-to-end single-run core: state-side evolution and measurement-side statistics",
        fontsize=12.2,
        fontweight="bold",
    )
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()

