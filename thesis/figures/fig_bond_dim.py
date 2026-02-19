import csv
import pathlib
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.9,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.linewidth": 0.85,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _find_single_run_raw_dir() -> pathlib.Path:
    data_root = pathlib.Path(__file__).resolve().parents[1] / "data"
    candidates = []
    for path in data_root.glob("sim_single_run_*"):
        raw_dir = path / "results" / "result_sim_run_000000" / "raw"
        if raw_dir.exists():
            candidates.append(raw_dir)
    if not candidates:
        raise FileNotFoundError(f"未找到单跑 raw 目录: {data_root}")
    return sorted(candidates)[-1]


def _stage_key(path: pathlib.Path) -> tuple[int, str]:
    match = re.search(r"step_(\d+)_([^.]*)_bond_dims\.csv$", path.name)
    if not match:
        return (999, path.stem)
    step = int(match.group(1))
    title = match.group(2).replace("_", " ").strip()
    return (step, title)


def _short_stage_label(step: int, stage_name: str) -> str:
    lower = stage_name.lower()
    if "after emission" in lower:
        core = "Emission"
    elif "after qfc" in lower:
        core = "QFC+Memory"
    elif "after fiber" in lower:
        core = "Fiber params"
    elif "after bs" in lower:
        core = "BS params"
    else:
        core = stage_name.title()
    return f"S{step}: {core}"


def _load_bond_dims(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    bond_idx = []
    chi = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bond_idx.append(int(row["bond_index"]))
            chi.append(float(row["chi"]))
    if not bond_idx:
        raise ValueError(f"空的 bond-dim 文件: {path}")
    return np.array(bond_idx, dtype=float), np.array(chi, dtype=float)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "pad": 1.7},
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#D9DEE7", alpha=0.65, linewidth=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    _set_style()

    raw_dir = _find_single_run_raw_dir()
    files = sorted(raw_dir.glob("run000_debug_step_*_bond_dims.csv"), key=_stage_key)
    if not files:
        raise FileNotFoundError(f"{raw_dir} 下未找到 bond_dims.csv")

    series = []
    for file in files:
        step, stage_name = _stage_key(file)
        x, y = _load_bond_dims(file)
        label = _short_stage_label(step, stage_name)
        series.append((step, label, x, y))

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.95, len(series)))

    fig = plt.figure(figsize=(11.4, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.16)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    max_vals = []
    p95_vals = []
    stage_labels = []

    for color, (step, label, x, y) in zip(colors, series):
        ax0.plot(x, y, color=color, linewidth=2.0, label=label)
        max_vals.append(float(np.max(y)))
        p95_vals.append(float(np.percentile(y, 95)))
        stage_labels.append(label)

    ax0.set_xlabel("Bond index")
    ax0.set_ylabel("Bond dimension $\\chi$")
    ax0.set_title("Bond-dimension profile across core stages", pad=4.0)
    _style_axis(ax0)
    ax0.legend(frameon=False, ncol=2, loc="upper left")
    _panel_label(ax0, "(a)")

    ypos = np.arange(len(stage_labels), dtype=float)
    ax1.barh(ypos + 0.17, max_vals, height=0.32, color="#4C78A8", label="max $\\chi$")
    ax1.barh(ypos - 0.17, p95_vals, height=0.32, color="#F58518", label="95th pct $\\chi$")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(stage_labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("$\\chi$")
    ax1.set_title("Stage summary", pad=4.0)
    _style_axis(ax1)
    ax1.legend(frameon=False, loc="lower right")
    _panel_label(ax1, "(b)")

    fig.suptitle("MPS bond growth diagnostics (real single-run data)", y=1.02, fontweight="bold")

    out_png = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
