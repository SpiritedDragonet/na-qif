from plot_style import frame_all_axes
import argparse
import json
import pathlib
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

DEFAULT_DENSITY_JSON = (
    pathlib.Path(__file__).resolve().parents[1]
    / "data"
    / "sim_single_run_local_output_20260223_2115"
    / "results"
    / "result_sim_run_000000"
    / "raw"
    / "declared_density_matrix.json"
)


def _set_style() -> None:
    # 在多分图布局下仍保持较大字号，避免缩放后难以辨认。
    mpl.rcParams.update(
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
            "font.size": 13.2,
            "axes.titlesize": 14.2,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 11.8,
            "ytick.labelsize": 11.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _load_payload(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 格式异常: {path}")
    return payload


def _as_matrix(real_part: Any, imag_part: Any, label: str) -> np.ndarray:
    rho = np.array(real_part, dtype=float) + 1j * np.array(imag_part, dtype=float)
    if rho.shape != (4, 4):
        raise ValueError(f"{label} 的密度矩阵形状异常: {rho.shape}")
    return rho


def _load_bell_matrix_map(payload: dict[str, Any], real_key: str, imag_key: str) -> dict[str, np.ndarray]:
    real_map = payload.get(real_key)
    imag_map = payload.get(imag_key)
    if not isinstance(real_map, dict) or not isinstance(imag_map, dict):
        return {}
    labels = sorted(set(real_map.keys()) & set(imag_map.keys()))
    out: dict[str, np.ndarray] = {}
    for label in labels:
        out[str(label)] = _as_matrix(real_map[label], imag_map[label], f"{real_key}.{label}")
    return out


def _soft_diverging_cmap() -> LinearSegmentedColormap:
    # 负值蓝、零附近近白、正值柔和黄。
    return LinearSegmentedColormap.from_list(
        "blue_white_yellow",
        ["#2F6FA3", "#F8FAFD", "#F4D35E"],
        N=256,
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    text_fn = getattr(ax, "text2D", ax.text)
    text_fn(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.8,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "pad": 1.7},
    )


def _annotate_matrix(ax: plt.Axes, mat: np.ndarray, vmax: float) -> None:
    threshold = max(0.02, 0.11 * vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat[i, j])
            if abs(value) < threshold:
                continue
            color = "white" if abs(value) > 0.55 * vmax else "#111111"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=10.5, color=color)


def _plot_heatmap(
    ax: plt.Axes,
    component: np.ndarray,
    title: str,
    vmax: float,
    cmap: LinearSegmentedColormap,
) -> None:
    labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    ax.imshow(component, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")
    ax.set_title(title, pad=6.0)
    ax.set_xticks(range(4), labels, rotation=19, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("ket", labelpad=6.0, loc="right")
    ax.set_ylabel("bra", labelpad=6.0, loc="top")
    ax.set_xticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.05, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _annotate_matrix(ax, component, vmax)


def _plot_bar3d(
    ax: plt.Axes,
    component: np.ndarray,
    title: str,
    vmax: float,
    cmap: LinearSegmentedColormap,
) -> None:
    n = component.shape[0]
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xpos = (xx.ravel() - 0.28).astype(float)
    ypos = (yy.ravel() - 0.28).astype(float)
    dx = np.full_like(xpos, 0.56, dtype=float)
    dy = np.full_like(ypos, 0.56, dtype=float)

    values = component.ravel().astype(float)
    zbase = np.where(values >= 0.0, 0.0, values)
    heights = np.abs(values)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    colors = cmap(norm(values))

    ax.bar3d(
        xpos,
        ypos,
        zbase,
        dx,
        dy,
        heights,
        color=colors,
        shade=True,
        alpha=0.96,
        edgecolor=(0.24, 0.24, 0.24, 0.42),
        linewidth=0.24,
        zsort="average",
    )
    ax.view_init(elev=24, azim=-54)
    ax.set_box_aspect((1.0, 1.0, 0.78))
    ax.set_title(title, pad=6.0)
    ax.set_zlim(-vmax, vmax)
    ax.set_zticks([-vmax, -0.5 * vmax, 0.0, 0.5 * vmax, vmax])

    labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=17, ha="right")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, rotation=-13, ha="right")
    ax.set_xlabel("ket", labelpad=6.0)
    ax.set_ylabel("bra", labelpad=8.0)
    ax.set_zlabel("", labelpad=6.0)
    ax.xaxis.set_label_coords(1.02, -0.04)
    ax.yaxis.set_label_coords(-0.04, 1.02)

    ax.xaxis.pane.set_facecolor((0.965, 0.972, 0.988, 1.0))
    ax.yaxis.pane.set_facecolor((0.965, 0.972, 0.988, 1.0))
    ax.zaxis.pane.set_facecolor((0.982, 0.986, 0.995, 1.0))
    ax.xaxis.pane.set_edgecolor((0.82, 0.85, 0.90, 1.0))
    ax.yaxis.pane.set_edgecolor((0.82, 0.85, 0.90, 1.0))
    ax.zaxis.pane.set_edgecolor((0.82, 0.85, 0.90, 1.0))


def main() -> None:
    _set_style()
    parser = argparse.ArgumentParser(
        description="Plot 12-panel density-matrix figure: 2D/3D, real/imag, raw Psi+/Psi- and ff delta."
    )
    parser.add_argument(
        "--density-json",
        type=pathlib.Path,
        default=DEFAULT_DENSITY_JSON,
        help="Path to declared_density_matrix.json; default points to thesis/data/sim_single_run_local_output_20260223_2115.",
    )
    args = parser.parse_args()

    density_path = args.density_json
    if not density_path.exists():
        raise FileNotFoundError(f"未找到密度矩阵数据: {density_path}")
    payload = _load_payload(density_path)
    rho_raw_by_bell = _load_bell_matrix_map(payload, "rho_raw_by_bell_real", "rho_raw_by_bell_imag")
    rho_ff_by_bell = _load_bell_matrix_map(payload, "rho_ff_by_bell_real", "rho_ff_by_bell_imag")

    required = ("Psi+", "Psi-")
    if not (all(label in rho_raw_by_bell for label in required) and all(label in rho_ff_by_bell for label in required)):
        raise KeyError(
            "declared_density_matrix.json 缺少 by-bell 字段："
            "rho_raw_by_bell_{real,imag} / rho_ff_by_bell_{real,imag} (Psi+/Psi-)."
        )

    rho_raw_psip = rho_raw_by_bell["Psi+"]
    rho_raw_psim = rho_raw_by_bell["Psi-"]
    delta_ff = rho_ff_by_bell["Psi+"] - rho_ff_by_bell["Psi-"]
    raw_vmax = max(
        float(np.max(np.abs(np.real(rho_raw_psip)))),
        float(np.max(np.abs(np.imag(rho_raw_psip)))),
        float(np.max(np.abs(np.real(rho_raw_psim)))),
        float(np.max(np.abs(np.imag(rho_raw_psim)))),
        0.10,
    )
    delta_vmax = max(
        float(np.max(np.abs(np.real(delta_ff)))),
        float(np.max(np.abs(np.imag(delta_ff)))),
        0.03,
    )
    rows = [
        ("Raw $\\Psi^{+}$", rho_raw_psip, raw_vmax),
        ("Raw $\\Psi^{-}$", rho_raw_psim, raw_vmax),
        (
            "$\\Delta\\rho_{\\mathrm{ff}}=\\rho^{\\Psi+}_{\\mathrm{ff}}-\\rho^{\\Psi-}_{\\mathrm{ff}}$",
            delta_ff,
            delta_vmax,
        ),
    ]
    cmap = _soft_diverging_cmap()

    fig = plt.figure(figsize=(24.0, 14.2))
    gs = fig.add_gridspec(3, 4, wspace=0.20, hspace=0.26)

    panel_idx = 0
    for r, (row_name, mat, vmax) in enumerate(rows):
        re_comp = np.real(mat)
        im_comp = np.imag(mat)
        titles = [
            rf"{row_name}：实部 2D",
            rf"{row_name}：实部 3D",
            rf"{row_name}：虚部 2D",
            rf"{row_name}：虚部 3D",
        ]

        ax = fig.add_subplot(gs[r, 0])
        _plot_heatmap(ax, re_comp, titles[0], vmax, cmap)
        _panel_label(ax, f"({chr(ord('a') + panel_idx)})")
        panel_idx += 1

        ax = fig.add_subplot(gs[r, 1], projection="3d")
        _plot_bar3d(ax, re_comp, titles[1], vmax, cmap)
        _panel_label(ax, f"({chr(ord('a') + panel_idx)})")
        panel_idx += 1

        ax = fig.add_subplot(gs[r, 2])
        _plot_heatmap(ax, im_comp, titles[2], vmax, cmap)
        _panel_label(ax, f"({chr(ord('a') + panel_idx)})")
        panel_idx += 1

        ax = fig.add_subplot(gs[r, 3], projection="3d")
        _plot_bar3d(ax, im_comp, titles[3], vmax, cmap)
        _panel_label(ax, f"({chr(ord('a') + panel_idx)})")
        panel_idx += 1

    # 颜色条右移，避免遮挡最右侧 3D 图纵轴文本。
    fig.subplots_adjust(left=0.045, right=0.905, top=0.92, bottom=0.072, wspace=0.19, hspace=0.24)

    sm_raw = ScalarMappable(norm=Normalize(vmin=-raw_vmax, vmax=raw_vmax), cmap=cmap)
    sm_raw.set_array([])
    cax_raw = fig.add_axes([0.932, 0.55, 0.020, 0.32])
    cbar_raw = fig.colorbar(sm_raw, cax=cax_raw)
    cbar_raw.set_label("原始矩阵", fontsize=13.0)
    cbar_raw.ax.tick_params(labelsize=11.2)

    sm_delta = ScalarMappable(norm=Normalize(vmin=-delta_vmax, vmax=delta_vmax), cmap=cmap)
    sm_delta.set_array([])
    cax_delta = fig.add_axes([0.932, 0.14, 0.020, 0.32])
    cbar_delta = fig.colorbar(sm_delta, cax=cax_delta)
    cbar_delta.set_label(r"$\Delta\rho_{\mathrm{ff}}$", fontsize=13.0)
    cbar_delta.ax.tick_params(labelsize=11.2)

    fig.suptitle(
        "宣告成功 Bell 后验态：实部/虚部 2D/3D 十二联图对比",
        y=0.965,
        fontsize=17.0,
        fontweight="bold",
    )

    out_pdf = pathlib.Path(__file__).with_suffix(".pdf")
    frame_all_axes(fig)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()


