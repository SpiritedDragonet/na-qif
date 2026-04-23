from plot_style import frame_all_axes
import csv
import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ATOM_LABELS = ["|0>", "|1>", "|u>", "|e>"]
FONT_SCALE = 1.75
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
            "font.size": 9.2 * FONT_SCALE,
            "axes.titlesize": 10.2 * FONT_SCALE,
            "axes.labelsize": 10.0 * FONT_SCALE,
            "xtick.labelsize": 8.6 * FONT_SCALE,
            "ytick.labelsize": 8.6 * FONT_SCALE,
            "legend.fontsize": 8.2 * FONT_SCALE,
            "axes.linewidth": 0.85,
            "lines.linewidth": 2.0,
            "figure.titlesize": 12.8 * FONT_SCALE,
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


def _auto_wave_ymax(arm_a: np.ndarray, arm_b: np.ndarray) -> float:
    peak = max(float(np.max(arm_a)), float(np.max(arm_b)), 1e-9)
    return peak * 1.18


def _omega_peak_to_rad_s(omega_peak: float, unit: str) -> float:
    unit_norm = str(unit).strip().lower()
    return float(omega_peak) * (2.0 * np.pi) if unit_norm == "hz" else float(omega_peak)


def _drive_envelope(
    t_ns: np.ndarray,
    *,
    t0_ns: float,
    sigma_ns: float,
    waveform: str,
) -> np.ndarray:
    if sigma_ns <= 0.0:
        return np.zeros_like(t_ns, dtype=float)
    x = (t_ns - float(t0_ns)) / float(sigma_ns)
    wf = str(waveform).strip().lower()
    if wf == "gaussian":
        return np.exp(-0.5 * x * x)
    if wf == "sech":
        return 1.0 / np.cosh(x)
    if wf == "square":
        return (np.abs(x) <= 1.0).astype(float)
    return np.zeros_like(t_ns, dtype=float)


def _compute_drive_intensity_curves(t_ns: np.ndarray, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    emission = manifest.get("config", {}).get("emission", {})
    arm_a_cfg = emission.get("arm_A", {})
    arm_b_cfg = emission.get("arm_B", {})
    sigma_ns = float(emission.get("sigma", 0.0))
    delay_ns_raw = emission.get("delay_ns", 0.0)
    delay_ns = 0.0 if delay_ns_raw is None else float(delay_ns_raw)
    t0_raw = emission.get("t0_ns")
    if t0_raw is None:
        t0_ns = 0.5 * float(t_ns[0] + t_ns[-1]) if t_ns.size > 0 else 0.0
    else:
        t0_ns = float(t0_raw)
    unit = str(emission.get("hamiltonian_rate_unit", "rad_s"))
    omega_peak_a = _omega_peak_to_rad_s(float(arm_a_cfg.get("omega_peak", 0.0)), unit)
    omega_peak_b = _omega_peak_to_rad_s(float(arm_b_cfg.get("omega_peak", 0.0)), unit)
    wf_a = str(emission.get("drive_waveform_A", "gaussian"))
    wf_b = str(emission.get("drive_waveform_B", "gaussian"))
    half_delay = 0.5 * delay_ns
    t0_a = t0_ns - half_delay
    t0_b = t0_ns + half_delay
    env_a = _drive_envelope(t_ns, t0_ns=t0_a, sigma_ns=sigma_ns, waveform=wf_a)
    env_b = _drive_envelope(t_ns, t0_ns=t0_b, sigma_ns=sigma_ns, waveform=wf_b)
    intensity_a = (omega_peak_a * env_a) ** 2
    intensity_b = (omega_peak_b * env_b) ** 2
    return intensity_a, intensity_b


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
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
    ax.plot(t_ns, arm_a, color=PALETTE["arm_a"], label="A臂")
    ax.plot(t_ns, arm_b, color=PALETTE["arm_b"], label="B臂")
    ax.axvspan(0.0, window_ns, color=PALETTE["window"], alpha=0.08, linewidth=0.0)
    _lock_time_axis(ax)
    ax.set_ylim(0.0, y_max)
    ax.set_ylabel("光子概率")
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
    drive_a, drive_b = _compute_drive_intensity_curves(t_ns, manifest)

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

    wave_ymax_em = _auto_wave_ymax(em_a, em_b)
    wave_ymax_qfc = _auto_wave_ymax(qfc_a, qfc_b)
    wave_ymax_fib = _auto_wave_ymax(fib_a, fib_b)
    wave_ymax_bs = _auto_wave_ymax(bs_a, bs_b)

    fig = plt.figure(figsize=(17.5, 11.8), constrained_layout=False)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.45, 1.0], hspace=0.55, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)
    ax_d = fig.add_subplot(gs[3, 0])
    ax_e = fig.add_subplot(gs[0, 1], sharex=ax_a)
    ax_f = fig.add_subplot(gs[1, 1], sharex=ax_a)
    ax_g = fig.add_subplot(gs[2, 1], sharex=ax_a)
    ax_h = fig.add_subplot(gs[3, 1], sharex=ax_a)

    # (a) 原子四能级（真实导出）
    ax_a.plot(t_atom, p0, color=PALETTE["atom_0"], label="|0>")
    ax_a.plot(t_atom, p1, color=PALETTE["atom_1"], label="|1>")
    ax_a.plot(t_atom, pu, color=PALETTE["atom_u"], label="|u>")
    ax_a.plot(t_atom, pe, color=PALETTE["atom_e"], label="|e>")
    ax_a.axvspan(0.0, window_ns, color=PALETTE["window"], alpha=0.08, linewidth=0.0)
    _lock_time_axis(ax_a)
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_ylabel("占据概率")
    ax_a.set_title("原子占据（A臂，直接导出）", pad=4.0)
    _style_axis(ax_a)
    ax_a.legend(frameon=False, ncol=4, loc="upper right")
    ax_a.text(
        0.02,
        0.96,
        (
            f"$g/\\kappa_{{tot}}={g_over_kappa:.3f}$（恢复的腔参数组）。\n"
            "坏腔：$|u\\rangle\\!\\to\\!|e\\rangle$ 后迅速辐射损失。\n"
            "更高精细度（更小 $\\kappa$）：$u\\leftrightarrow e$ 相干交换更强。"
        ),
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=9.6,
        bbox={"facecolor": "white", "edgecolor": "#D8DEE9", "alpha": 0.86, "boxstyle": "round,pad=0.32"},
    )
    _panel_label(ax_a, "(a)")

    # (b) after emission
    _plot_dual_arm(
        ax_b,
        t_ns,
        em_a,
        em_b,
        "发射后：双臂波包与驱动强度",
        y_max=wave_ymax_em,
        window_ns=window_ns,
        show_legend=True,
    )
    ax_b_r = ax_b.twinx()
    ax_b_r.plot(t_ns, drive_a, color=PALETTE["arm_a"], linestyle="--", linewidth=1.8, label="A臂驱动")
    ax_b_r.plot(t_ns, drive_b, color=PALETTE["arm_b"], linestyle="--", linewidth=1.8, label="B臂驱动")
    drive_peak = max(float(np.max(drive_a)), float(np.max(drive_b)), 1e-30)
    ax_b_r.set_ylim(0.0, drive_peak * 1.12)
    ax_b_r.set_ylabel(r"驱动强度 $|\Omega|^2$ (rad$^2$/s$^2$)")
    ax_b_r.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_b_r.grid(False)
    ax_b_r.spines["top"].set_visible(False)
    ax_b_r.legend(frameon=False, loc="upper left")
    n_bins = int(t_ns.size)
    t_start_ns = 0.0
    t_end_ns = float(n_bins * dt_ns)
    ax_b.text(
        0.02,
        0.06,
        (
            f"时间轴视图（非时间仓索引）：Δt={dt_ns:g} ns/仓；"
            f"内部时间仓顺序反向，即 bin {n_bins} <-> [{t_start_ns:g},{dt_ns:g}) ns，"
            f"bin 1 <-> [{t_end_ns - dt_ns:g},{t_end_ns:g}) ns"
        ),
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#222222",
        bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 1.6},
    )
    _panel_label(ax_b, "(b)")

    # (c) 单臂积分保留（归一化到发射阶段）
    ax_c.plot(t_ns, retain_emit, color=PALETTE["arm_a"], label=f"发射后（{retain_emit[-1]:.3f}）")
    ax_c.plot(t_ns, retain_qfc, color=PALETTE["qfc"], label=f"QFC 后（{retain_qfc[-1]:.3f}）")
    ax_c.plot(t_ns, retain_fib, color=PALETTE["fiber"], label=f"光纤后（{retain_fib[-1]:.3f}）")
    ax_c.plot(t_ns, retain_bs, color=PALETTE["bs"], label=f"BS 后（{retain_bs[-1]:.3f}）")
    ax_c.axvspan(0.0, window_ns, color=PALETTE["window"], alpha=0.08, linewidth=0.0)
    _lock_time_axis(ax_c)
    ax_c.set_ylim(0.0, max(1.03, float(np.max(retain_emit)) * 1.05))
    ax_c.set_ylabel("保留比例")
    ax_c.set_xlabel("时间 (ns)")
    ax_c.set_title("单臂累计保留（A臂，归一化）", pad=4.0)
    _style_axis(ax_c)
    ax_c.legend(frameon=False, loc="upper left", ncol=2)
    _panel_label(ax_c, "(c)")

    # (d) 阶段总积分（波包积分）对比
    integ_emit_b = np.cumsum(em_b) * dt_ns
    integ_qfc_b = np.cumsum(qfc_b) * dt_ns
    integ_fib_b = np.cumsum(fib_b) * dt_ns
    integ_bs_b = np.cumsum(bs_b) * dt_ns
    stage_names = ["发射", "QFC", "光纤", "BS"]
    stage_x = np.arange(len(stage_names), dtype=float)
    stage_integral_a = np.array([integ_emit[-1], integ_qfc[-1], integ_fib[-1], integ_bs[-1]], dtype=float)
    stage_integral_b = np.array([integ_emit_b[-1], integ_qfc_b[-1], integ_fib_b[-1], integ_bs_b[-1]], dtype=float)
    ax_d.plot(stage_x, stage_integral_a, color=PALETTE["arm_a"], marker="o", label="A臂")
    ax_d.plot(stage_x, stage_integral_b, color=PALETTE["arm_b"], marker="s", label="B臂")
    ax_d.fill_between(stage_x, 0.0, stage_integral_a, color=PALETTE["arm_a"], alpha=0.07, linewidth=0.0)
    ax_d.fill_between(stage_x, 0.0, stage_integral_b, color=PALETTE["arm_b"], alpha=0.06, linewidth=0.0)
    ax_d.set_xticks(stage_x, stage_names)
    ax_d.tick_params(axis="x", labelrotation=12)
    ax_d.set_ylabel("光子概率积分")
    ax_d.set_title("分阶段总光子概率", pad=4.0)
    _style_axis(ax_d)
    ax_d.legend(frameon=False, loc="upper right")
    _panel_label(ax_d, "(d)")

    # (e)(f)(g) 三阶段双臂波包
    _plot_dual_arm(
        ax_e,
        t_ns,
        qfc_a,
        qfc_b,
        "QFC 后",
        y_max=wave_ymax_qfc,
        window_ns=window_ns,
        show_legend=False,
    )
    _panel_label(ax_e, "(e)")
    _plot_dual_arm(
        ax_f,
        t_ns,
        fib_a,
        fib_b,
        "BS 前（光纤后）",
        y_max=wave_ymax_fib,
        window_ns=window_ns,
        show_legend=False,
    )
    _panel_label(ax_f, "(f)")
    _plot_dual_arm(
        ax_g,
        t_ns,
        bs_a,
        bs_b,
        "BS 后",
        y_max=wave_ymax_bs,
        window_ns=window_ns,
        show_legend=False,
    )
    _panel_label(ax_g, "(g)")

    # (h) 能量捕获轮廓
    ax_h.plot(t_ns, energy_a, color=PALETTE["arm_a"], label="A臂累计")
    ax_h.plot(t_ns, energy_b, color=PALETTE["arm_b"], label="B臂累计")
    ax_h.axhline(0.65, linestyle="--", linewidth=1.2, color="#666666", label="65% 水平")
    ax_h.axvline(window_ns, linestyle="--", linewidth=1.2, color=PALETTE["window"], label=f"{window_ns:.0f} ns 边界")
    ax_h.fill_between(t_ns, 0.0, energy_a, color=PALETTE["arm_a"], alpha=0.08, linewidth=0.0)
    ax_h.fill_between(t_ns, 0.0, energy_b, color=PALETTE["arm_b"], alpha=0.07, linewidth=0.0)
    _lock_time_axis(ax_h)
    ax_h.set_ylim(0.0, 1.02)
    ax_h.set_xlabel("时间 (ns)")
    ax_h.set_ylabel("累计能量")
    ax_h.set_title("能量捕获轮廓（光纤后）", pad=4.0)
    _style_axis(ax_h)
    ax_h.legend(frameon=False, loc="lower right")
    _panel_label(ax_h, "(h)")

    # 共享 x 轴时 Matplotlib 默认只保留底部刻度；这里强制显示 a/b/e/f/g 的刻度数字。
    for ax in (ax_a, ax_b, ax_e, ax_f, ax_g):
        ax.tick_params(axis="x", labelbottom=True)
        plt.setp(ax.get_xticklabels(), visible=True)

    fig.suptitle(
        "双臂波包时域结构与接收窗口位置",
        y=0.985,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        (
            f"坐标约定：所有时域面板使用物理时间 t (ns)。"
            f"单个时间仓宽度为 Δt={dt_ns:g} ns。"
        ),
        ha="center",
        va="bottom",
        fontsize=9.0,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.10)

    out_pdf = pathlib.Path(__file__).with_suffix(".pdf")
    frame_all_axes(fig)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
