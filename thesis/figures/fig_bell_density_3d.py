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
            "font.size": 11.8,
            "axes.titlesize": 12.6,
            "axes.labelsize": 11.8,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
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
    return LinearSegmentedColormap.from_list(
        "blue_white_yellow",
        ["#2F6FA3", "#F8FAFD", "#F4D35E"],
        N=256,
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.6,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "pad": 1.4},
    )


def _annotate_matrix(ax: plt.Axes, mat: np.ndarray, vmax: float) -> None:
    threshold = max(0.02, 0.11 * vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat[i, j])
            if abs(value) < threshold:
                continue
            color = "white" if abs(value) > 0.52 * vmax else "#111111"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9.7, color=color)


def _plot_heatmap(
    ax: plt.Axes,
    component: np.ndarray,
    title: str,
    vmax: float,
    cmap: LinearSegmentedColormap,
) -> None:
    labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    ax.imshow(component, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")
    ax.set_title(title, pad=5.5)
    ax.set_xticks(range(4), labels, rotation=19, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("ket", labelpad=6.0, loc="right")
    ax.set_ylabel("bra", labelpad=8.0, loc="top")
    ax.set_xticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.05, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    _annotate_matrix(ax, component, vmax)


def main() -> None:
    _set_style()
    parser = argparse.ArgumentParser(
        description="Plot 2D Bell-density heatmaps for raw Psi+/Psi- and ff delta."
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

    fig = plt.figure(figsize=(11.6, 13.6))
    gs = fig.add_gridspec(3, 2, wspace=0.18, hspace=0.28)

    panel_idx = 0
    for r, (row_name, mat, vmax) in enumerate(rows):
        re_comp = np.real(mat)
        im_comp = np.imag(mat)
        titles = [
            rf"{row_name}：实部",
            rf"{row_name}：虚部",
        ]

        ax = fig.add_subplot(gs[r, 0])
        _plot_heatmap(ax, re_comp, titles[0], vmax, cmap)
        _panel_label(ax, f"({chr(ord('a') + panel_idx)})")
        panel_idx += 1

        ax = fig.add_subplot(gs[r, 1])
        _plot_heatmap(ax, im_comp, titles[1], vmax, cmap)
        _panel_label(ax, f"({chr(ord('a') + panel_idx)})")
        panel_idx += 1

    fig.subplots_adjust(left=0.075, right=0.885, top=0.93, bottom=0.06, wspace=0.16, hspace=0.23)

    sm_raw = ScalarMappable(norm=Normalize(vmin=-raw_vmax, vmax=raw_vmax), cmap=cmap)
    sm_raw.set_array([])
    cax_raw = fig.add_axes([0.905, 0.55, 0.022, 0.29])
    cbar_raw = fig.colorbar(sm_raw, cax=cax_raw)
    cbar_raw.set_label("原始后验态", fontsize=11.6)
    cbar_raw.ax.tick_params(labelsize=10.4)

    sm_delta = ScalarMappable(norm=Normalize(vmin=-delta_vmax, vmax=delta_vmax), cmap=cmap)
    sm_delta.set_array([])
    cax_delta = fig.add_axes([0.905, 0.16, 0.022, 0.24])
    cbar_delta = fig.colorbar(sm_delta, cax=cax_delta)
    cbar_delta.set_label(r"$\Delta\rho_{\mathrm{ff}}$", fontsize=11.6)
    cbar_delta.ax.tick_params(labelsize=10.4)

    fig.suptitle(
        "宣告成功 Bell 后验态的二维热图比较",
        y=0.965,
        fontsize=15.2,
        fontweight="bold",
    )

    out_pdf = pathlib.Path(__file__).with_suffix(".pdf")
    frame_all_axes(fig)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
