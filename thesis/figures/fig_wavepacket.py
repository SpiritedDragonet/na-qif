import csv
import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ATOM_LABELS = ["|0>", "|1>", "|u>", "|e>"]
PALETTE = {
    "atom_0": "#4C78A8",
    "atom_1": "#F58518",
    "atom_u": "#54A24B",
    "atom_e": "#E45756",
    "arm_a": "#1F77B4",
    "arm_b": "#D62728",
    "qfc": "#FF7F0E",
    "fiber": "#2CA02C",
    "bs": "#9467BD",
    "grid": "#D9DEE7",
    "window": "#F2C14E",
}


def _set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.85,
            "lines.linewidth": 2.0,
            "figure.titlesize": 12.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _find_single_run_root(data_root: pathlib.Path) -> pathlib.Path:
    candidates = []
    for path in data_root.glob("sim_single_run_*"):
        result_root = path / "results" / "result_sim_run_000000"
        if (result_root / "raw").exists() and (result_root / "plots").exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"未找到单跑数据目录: {data_root}")
    return sorted(candidates)[-1]


def _load_manifest(run_root: pathlib.Path) -> dict:
    manifest_path = run_root / "summary" / "run_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_stage_table(states_csv: pathlib.Path) -> tuple[dict, list[str]]:
    by_arm: dict[str, dict[int, dict[str, object]]] = {"A": {}, "B": {}}
    label_order: list[str] = []
    seen_labels: set[str] = set()
    with states_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arm = row.get("arm", "")
            if arm not in by_arm:
                continue
            bin_idx = int(row["bin_index"])
            time_ns = float(row["time_ns_increasing"])
            label = row["state_label"].strip()
            prob = float(row["probability"])
            item = by_arm[arm].setdefault(bin_idx, {"time_ns": time_ns, "states": {}})
            item["states"][label] = prob
            if label not in seen_labels:
                seen_labels.add(label)
                label_order.append(label)
    if not by_arm["A"] or not by_arm["B"]:
        raise ValueError(f"CSV 缺少 A/B 臂数据: {states_csv}")
    return by_arm, label_order


def _extract_series(by_arm: dict, arm: str, reducer) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(by_arm[arm].items(), key=lambda kv: kv[0])
    t_ns = np.array([float(item["time_ns"]) for _, item in ordered], dtype=float)
    values = np.array([float(reducer(item["states"])) for _, item in ordered], dtype=float)
    return t_ns, values


def _extract_dual_arm_photon_intensity(by_arm: dict, label_order: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    photon_labels = [lbl for lbl in label_order if lbl not in ATOM_LABELS and lbl != "|vac>"]
    if not photon_labels:
        raise ValueError("未找到光子态标签，无法构建波包强度。")
    t_a, intensity_a = _extract_series(
        by_arm,
        "A",
        lambda states: sum(float(states.get(lbl, 0.0)) for lbl in photon_labels),
    )
    t_b, intensity_b = _extract_series(
        by_arm,
        "B",
        lambda states: sum(float(states.get(lbl, 0.0)) for lbl in photon_labels),
    )
    if not np.allclose(t_a, t_b):
        raise ValueError("A/B 臂时间轴不一致。")
    return t_a, intensity_a, intensity_b


def _extract_atomic_population(by_arm: dict, label: str, arm: str = "A") -> tuple[np.ndarray, np.ndarray]:
    return _extract_series(by_arm, arm, lambda states: float(states.get(label, 0.0)))


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color=PALETTE["grid"], alpha=0.65, linewidth=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _lock_time_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 100.0)
    ax.set_xbound(0.0, 100.0)
    ax.margins(x=0.0)
    ax.set_xticks(np.arange(0.0, 101.0, 20.0))


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
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.8},
    )


def _plot_dual_arm(
    ax: plt.Axes,
    t_ns: np.ndarray,
    arm_a: np.ndarray,
    arm_b: np.ndarray,
    title: str,
    y_max: float,
    window_ns: float,
    show_legend: bool = False,
) -> None:
    ax.fill_between(t_ns, 0.0, arm_a, color=PALETTE["arm_a"], alpha=0.14, linewidth=0.0)
    ax.fill_between(t_ns, 0.0, arm_b, color=PALETTE["arm_b"], alpha=0.12, linewidth=0.0)
    ax.plot(t_ns, arm_a, color=PALETTE["arm_a"], label="Arm A")
    ax.plot(t_ns, arm_b, color=PALETTE["arm_b"], label="Arm B")
    ax.axvspan(0.0, window_ns, color=PALETTE["window"], alpha=0.08, linewidth=0.0)
    _lock_time_axis(ax)
    ax.set_ylim(0.0, y_max)
    ax.set_ylabel("Photon prob./bin")
    ax.set_title(title, pad=4.0)
    _style_axis(ax)
    if show_legend:
        ax.legend(frameon=False, loc="upper right")


def main() -> None:
    _set_paper_style()

    data_root = pathlib.Path(__file__).resolve().parents[1] / "data"
    run_root = _find_single_run_root(data_root)
    manifest = _load_manifest(run_root)
    result_root = run_root / "results" / "result_sim_run_000000"
    plots_dir = result_root / "plots"

    stage_files = {
        "emission": plots_dir / "run000_1_after_emission_states.csv",
        "qfc": plots_dir / "run000_2_after_qfc_states.csv",
        "fiber": plots_dir / "run000_3_after_fiber_states.csv",
        "bs": plots_dir / "run000_4_after_bs_states.csv",
    }
    for key, path in stage_files.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少 {key} 阶段数据: {path}")

    em_table, em_labels = _load_stage_table(stage_files["emission"])
    qfc_table, qfc_labels = _load_stage_table(stage_files["qfc"])
    fib_table, fib_labels = _load_stage_table(stage_files["fiber"])
    bs_table, bs_labels = _load_stage_table(stage_files["bs"])

    t_ns, em_a, em_b = _extract_dual_arm_photon_intensity(em_table, em_labels)
    _, qfc_a, qfc_b = _extract_dual_arm_photon_intensity(qfc_table, qfc_labels)
    _, fib_a, fib_b = _extract_dual_arm_photon_intensity(fib_table, fib_labels)
    _, bs_a, bs_b = _extract_dual_arm_photon_intensity(bs_table, bs_labels)

    missing_atomic = [lbl for lbl in ATOM_LABELS if lbl not in set(em_labels)]
    if missing_atomic:
        raise ValueError(f"发射阶段 CSV 缺少原子态列: {missing_atomic}")

    t_atom, p0 = _extract_atomic_population(em_table, "|0>", arm="A")
    _, p1 = _extract_atomic_population(em_table, "|1>", arm="A")
    _, pu = _extract_atomic_population(em_table, "|u>", arm="A")
    _, pe = _extract_atomic_population(em_table, "|e>", arm="A")
    if not np.allclose(t_atom, t_ns):
        raise ValueError("发射阶段原子态与光子态时间轴不一致。")

    dt_ns = float(np.median(np.diff(t_ns))) if t_ns.size > 1 else 1.0
    n_bins = int(t_ns.size)
    window_ns = float(manifest.get("config", {}).get("run", {}).get("window_ns", 70.0))
    em_cfg = manifest.get("config", {}).get("emission", {})
    arm_a_cfg = em_cfg.get("arm_A", {})
    ham_rate_unit = str(em_cfg.get("hamiltonian_rate_unit", "rad_s"))
    g_raw = float(arm_a_cfg.get("g", 0.0))
    g_hz = g_raw / (2.0 * np.pi) if ham_rate_unit == "rad_s" else g_raw
    kappa_ex = float(arm_a_cfg.get("kappa_ex", 0.0))
    kappa_in = float(arm_a_cfg.get("kappa_in", 0.0))
    kappa_tot_hz = kappa_ex + kappa_in
    g_over_kappa = g_hz / max(kappa_tot_hz, 1e-12)

    integ_emit = np.cumsum(em_a) * dt_ns
    integ_qfc = np.cumsum(qfc_a) * dt_ns
    integ_fib = np.cumsum(fib_a) * dt_ns
    integ_bs = np.cumsum(bs_a) * dt_ns

    norm_base = max(float(integ_emit[-1]), 1e-12)
    retain_emit = integ_emit / norm_base
    retain_qfc = integ_qfc / norm_base
    retain_fib = integ_fib / norm_base
    retain_bs = integ_bs / norm_base

    energy_a = np.cumsum(fib_a)
    energy_b = np.cumsum(fib_b)
    energy_a /= max(float(energy_a[-1]), 1e-12)
    energy_b /= max(float(energy_b[-1]), 1e-12)

    wave_ymax = max(
        float(np.max(em_a)),
        float(np.max(em_b)),
        float(np.max(qfc_a)),
        float(np.max(qfc_b)),
        float(np.max(fib_a)),
        float(np.max(fib_b)),
        float(np.max(bs_a)),
        float(np.max(bs_b)),
        1e-6,
    )
    wave_ymax *= 1.16

    fig = plt.figure(figsize=(13.8, 10.8), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1.0], hspace=0.22, wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)
    ax_info = fig.add_subplot(gs[3, 0])
    ax_d = fig.add_subplot(gs[0, 1], sharex=ax_a, sharey=ax_b)
    ax_e = fig.add_subplot(gs[1, 1], sharex=ax_a, sharey=ax_b)
    ax_f = fig.add_subplot(gs[2, 1], sharex=ax_a, sharey=ax_b)
    ax_g = fig.add_subplot(gs[3, 1], sharex=ax_a)

    # (a) 原子四能级（真实导出）
    ax_a.plot(t_atom, p0, color=PALETTE["atom_0"], label="|0>")
    ax_a.plot(t_atom, p1, color=PALETTE["atom_1"], label="|1>")
    ax_a.plot(t_atom, pu, color=PALETTE["atom_u"], label="|u>")
    ax_a.plot(t_atom, pe, color=PALETTE["atom_e"], label="|e>")
    ax_a.axvspan(0.0, window_ns, color=PALETTE["window"], alpha=0.08, linewidth=0.0)
    _lock_time_axis(ax_a)
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_ylabel("Population")
    ax_a.set_title("Atomic populations (Arm A, directly exported)", pad=4.0)
    _style_axis(ax_a)
    ax_a.legend(frameon=False, ncol=4, loc="upper right")
    ax_a.text(
        0.02,
        0.08,
        (
            f"$g/\\kappa_{{tot}}={g_over_kappa:.3f}$ (restored cavity parameter set).\n"
            "In bad-cavity regime, $|u\\rangle\\!\\to\\!|e\\rangle$ is followed by fast radiative loss to $|0\\rangle,|1\\rangle$;\n"
            "higher finesse (smaller $\\kappa$) increases coherent $u\\leftrightarrow e$ exchange, approaching Rabi-like oscillation."
        ),
        transform=ax_a.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.7,
        bbox={"facecolor": "white", "edgecolor": "#D8DEE9", "alpha": 0.86, "boxstyle": "round,pad=0.32"},
    )
    _panel_label(ax_a, "(a)")

    # (b) after emission
    _plot_dual_arm(
        ax_b,
        t_ns,
        em_a,
        em_b,
        "After emission: dual-arm wavepacket",
        y_max=wave_ymax,
        window_ns=window_ns,
        show_legend=True,
    )
    _panel_label(ax_b, "(b)")

    # (c) 单臂积分保留（归一化到发射阶段）
    ax_c.plot(t_ns, retain_emit, color=PALETTE["arm_a"], label=f"Emission ({retain_emit[-1]:.3f})")
    ax_c.plot(t_ns, retain_qfc, color=PALETTE["qfc"], label=f"After QFC ({retain_qfc[-1]:.3f})")
    ax_c.plot(t_ns, retain_fib, color=PALETTE["fiber"], label=f"After fiber ({retain_fib[-1]:.3f})")
    ax_c.plot(t_ns, retain_bs, color=PALETTE["bs"], label=f"After BS ({retain_bs[-1]:.3f})")
    ax_c.axvspan(0.0, window_ns, color=PALETTE["window"], alpha=0.08, linewidth=0.0)
    _lock_time_axis(ax_c)
    ax_c.set_ylim(0.0, max(1.03, float(np.max(retain_emit)) * 1.05))
    ax_c.set_ylabel("Retention ratio")
    ax_c.set_xlabel("Time (ns)")
    ax_c.set_title("Single-arm cumulative retention (Arm A, normalized)", pad=4.0)
    _style_axis(ax_c)
    ax_c.legend(frameon=False, loc="upper left", ncol=2)
    _panel_label(ax_c, "(c)")

    # 左下信息框
    ax_info.axis("off")
    info_lines = [
        "Run Snapshot (Real Data)",
        f"Dataset: {run_root.name}",
        f"Time bins: N={n_bins}, dt={dt_ns:.1f} ns, total={n_bins * dt_ns:.1f} ns",
        f"Acceptance window: {window_ns:.1f} ns",
        f"Retention @100 ns: QFC={retain_qfc[-1]:.3f}, fiber={retain_fib[-1]:.3f}, BS={retain_bs[-1]:.3f}",
    ]
    ax_info.text(
        0.0,
        0.98,
        "\n".join(info_lines),
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=9.0,
        bbox={
            "facecolor": "#F7F9FC",
            "edgecolor": "#D8DEE9",
            "alpha": 0.96,
            "boxstyle": "round,pad=0.45",
        },
    )

    # (d)(e)(f) 三阶段对比
    _plot_dual_arm(ax_d, t_ns, qfc_a, qfc_b, "After QFC", y_max=wave_ymax, window_ns=window_ns, show_legend=False)
    _panel_label(ax_d, "(d)")
    _plot_dual_arm(
        ax_e,
        t_ns,
        fib_a,
        fib_b,
        "Before BS (after fiber)",
        y_max=wave_ymax,
        window_ns=window_ns,
        show_legend=False,
    )
    _panel_label(ax_e, "(e)")
    _plot_dual_arm(ax_f, t_ns, bs_a, bs_b, "After BS", y_max=wave_ymax, window_ns=window_ns, show_legend=False)
    _panel_label(ax_f, "(f)")

    # (g) 能量捕获轮廓
    ax_g.plot(t_ns, energy_a, color=PALETTE["arm_a"], label="Arm A cumulative")
    ax_g.plot(t_ns, energy_b, color=PALETTE["arm_b"], label="Arm B cumulative")
    ax_g.axhline(0.65, linestyle="--", linewidth=1.2, color="#666666", label="65% level")
    ax_g.axvline(window_ns, linestyle="--", linewidth=1.2, color=PALETTE["window"], label=f"{window_ns:.0f} ns edge")
    ax_g.fill_between(t_ns, 0.0, energy_a, color=PALETTE["arm_a"], alpha=0.08, linewidth=0.0)
    ax_g.fill_between(t_ns, 0.0, energy_b, color=PALETTE["arm_b"], alpha=0.07, linewidth=0.0)
    _lock_time_axis(ax_g)
    ax_g.set_ylim(0.0, 1.02)
    ax_g.set_xlabel("Time (ns)")
    ax_g.set_ylabel("Cumulative energy")
    ax_g.set_title("Energy capture profile (after fiber)", pad=4.0)
    _style_axis(ax_g)
    ax_g.legend(frameon=False, loc="lower right")
    _panel_label(ax_g, "(g)")

    for ax in (ax_a, ax_b, ax_d, ax_e, ax_f):
        ax.tick_params(labelbottom=False)

    fig.suptitle(
        "Dual-arm wavepacket temporal structure and acceptance-window placement",
        y=1.01,
        fontweight="bold",
    )

    out_png = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
