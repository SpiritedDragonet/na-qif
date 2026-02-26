# -*- coding: utf-8 -*-
"""
Single-run simulation workflow and summary helpers.
"""

from __future__ import annotations

import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Iterator
from collections import Counter
from copy import deepcopy

import numpy as np

from ..simulation import (
    compute_fidelity_with_bell,
)
from ..visualization import plot_dual_arm_heatmap
from ..physics.gates import bs_gate_6d
from .common import (
    SimConfig,
    PipelineHooks,
    TimingTracer,
    run_trial_physics_core,
    run_detection_core_from_pipe,
    write_click_records,
    write_declared_density_matrix,
    _build_run_parameter_store,
)

# Debug toggle (default False)
DEBUG_MODE = False

SIM_TASK_METRIC_KEYS = (
    "window_ns",
    "p_arrive",
    "p_arrive_11",
    "p_arrive_same_arm",
    "p_arrive_20",
    "p_arrive_02",
    "p_success_abs",
    "p_success_true_abs",
    "p_success_false_abs",
    "p_success_true_given_arrival",
    "fidelity_all",
    "fidelity_true",
    "fidelity_false",
    "false_fraction",
    "corr_exx",
    "corr_eyy",
    "corr_ezz",
    "chsh_s_max",
    "corr_exx_ff",
    "corr_eyy_ff",
    "corr_ezz_ff",
    "chsh_s_max_ff",
    "mps_chi_min",
    "mps_chi_mean",
    "mps_chi_max",
    "mps_trunc_total_calls",
    "mps_trunc_total_eps",
    "mps_trunc_total_eps_max",
    "p_success_intrinsic_dark_assisted",
    "p_success_bg_assisted",
)

def save_debug_info(
    mps,
    n_bins: int,
    stage: str,
    output_dir: Path,
    step_index: int,
    run_tag: Optional[str] = None,
    stage_context: Optional[dict] = None,
):
    """
    保存调试信息到文件。

    Parameters
    ----------
    mps : MPSState
        当前MPS态
    n_bins : int
        时间仓数量
    stage : str
        当前阶段名称
    output_dir : Path
        输出目录
    step_index : int
        步骤索引
    """
    from atom_sim.simulation.detection import (
        extract_qubit_state, compute_fidelity_with_bell
    )

    info = {}
    info['stage'] = stage
    info['step'] = step_index
    stage_lower = str(stage).strip().lower()
    stage_note = None
    if "after fiber" in stage_lower:
        stage_note = (
            "注: 光纤在当前实现中仅采样 Heisenberg 端参数，不直接改写态端 MPS；"
            "因此态快照可与上一阶段一致，但最终测量统计会变化。"
        )
    elif "after bs" in stage_lower:
        stage_note = (
            "注: BS 已并入测量端，不直接作用于态；"
            "该快照主要反映退相干后态。若本次轨迹未采到相位翻转，"
            "与 After Fiber 可近似一致，但探测端口统计仍会受 BS 参数影响。"
        )

    # MPS维度信息
    chi_list = mps._mps.chi
    d_list = mps.d
    trunc_stats = mps.get_truncation_stats()
    info['n_sites'] = len(d_list)
    info['n_bins'] = n_bins
    info['bond_dimensions'] = f'chi_min={min(chi_list)}, chi_max={max(chi_list)}, chi_mean={np.mean(chi_list):.1f}'
    info['local_dimensions'] = f'first_5={d_list[:5]}, last_5={d_list[-5:]}'
    info['trunc_total_calls'] = int(trunc_stats["total_calls"])
    info['trunc_total_eps'] = float(trunc_stats["total_eps"])
    info['trunc_total_eps_max'] = float(trunc_stats["total_eps_max"])

    # 原子态信息
    qubit_state, p_qubit = extract_qubit_state(mps)
    info['qubit_state_diag'] = np.diag(qubit_state).real.tolist()
    info['p_qubit'] = p_qubit
    if p_qubit > 0:
        qubit_state_cond = qubit_state / p_qubit
        info['qubit_purity'] = float(np.real(np.trace(qubit_state_cond @ qubit_state_cond)))
    else:
        info['qubit_purity'] = 0.0

    # Bell态保真度
    for bell in ['Psi+', 'Psi-', 'Phi+', 'Phi-']:
        f_full = compute_fidelity_with_bell(qubit_state, bell)
        f_cond = f_full / p_qubit if p_qubit > 0 else 0.0
        info[f'fidelity_{bell.replace("+", "p").replace("-", "m")}_full'] = f_full
        info[f'fidelity_{bell.replace("+", "p").replace("-", "m")}_cond'] = f_cond

    # 保存到文件
    prefix = f"{run_tag}_" if run_tag else ""
    info_file = output_dir / f'{prefix}debug_step_{step_index:02d}_{stage.replace(" ", "_").lower()}.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f'调试信息 - {stage}\n')
        f.write('='*60 + '\n\n')
        if stage_note:
            f.write(stage_note + '\n\n')
        if stage_context:
            f.write('阶段口径/参数:\n')
            for key in sorted(stage_context.keys()):
                value = stage_context[key]
                if isinstance(value, float):
                    value_str = f"{value:.6e}"
                else:
                    value_str = str(value)
                f.write(f'  {key}: {value_str}\n')
            f.write('\n')
        f.write('MPS维度信息:\n')
        f.write(f'  n_sites = {info["n_sites"]}\n')
        f.write(f'  n_bins = {info["n_bins"]}\n')
        f.write(f'  {info["bond_dimensions"]}\n')
        f.write(f'  {info["local_dimensions"]}\n\n')
        f.write('数值可信度:\n')
        f.write(f'  trunc_total_calls = {info["trunc_total_calls"]}\n')
        f.write(f'  trunc_total_eps = {info["trunc_total_eps"]:.6e}\n')
        f.write(f'  trunc_total_eps_max = {info["trunc_total_eps_max"]:.6e}\n\n')
        f.write('原子态信息:\n')
        f.write(f'  对角元: {info["qubit_state_diag"]}\n')
        f.write(f'  p_qubit: {info["p_qubit"]:.4f}\n')
        f.write(f'  纯度(条件化): {info["qubit_purity"]:.4f}\n\n')
        f.write('Bell态保真度:\n')
        f.write(f'  Psi+ (full/cond) = {info["fidelity_Psip_full"]:.4f} / {info["fidelity_Psip_cond"]:.4f}\n')
        f.write(f'  Psi- (full/cond) = {info["fidelity_Psim_full"]:.4f} / {info["fidelity_Psim_cond"]:.4f}\n')
        f.write(f'  Phi+ (full/cond) = {info["fidelity_Phip_full"]:.4f} / {info["fidelity_Phip_cond"]:.4f}\n')
        f.write(f'  Phi- (full/cond) = {info["fidelity_Phim_full"]:.4f} / {info["fidelity_Phim_cond"]:.4f}\n')

    # 额外导出完整键维信息，便于论文图直接读取原始数值
    bond_file = output_dir / f'{prefix}debug_step_{step_index:02d}_{stage.replace(" ", "_").lower()}_bond_dims.csv'
    with open(bond_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "bond_index",
            "left_site",
            "right_site",
            "left_local_dim",
            "right_local_dim",
            "chi",
        ])
        for bond_index, chi in enumerate(chi_list):
            left_site = int(bond_index)
            right_site = int(bond_index + 1)
            left_local_dim = int(d_list[left_site]) if left_site < len(d_list) else ""
            right_local_dim = int(d_list[right_site]) if right_site < len(d_list) else ""
            writer.writerow([
                int(bond_index),
                left_site,
                right_site,
                left_local_dim,
                right_local_dim,
                int(chi),
            ])

    print(f'  调试信息已保存: {info_file.name}')
    print(f'  键维数据已保存: {bond_file.name}')

def _run_single_trial(
    rng: Optional[np.random.Generator],
    config: SimConfig,
    delay_ns: Optional[float],
    delay_jitter_ns: Optional[float],
    verbose: bool,
    debug: bool,
    hooks: Optional[PipelineHooks],
    emission_diagnostics: bool,
    should_abort=None,
):
    """
    目的：抽出最小可复用的物理流程（发射->QFC->滤波->光纤->退相干->BS并入测量）。
    规则：delay_ns/delay_jitter_ns 优先采用显式传入，否则取配置默认。
    """
    return run_trial_physics_core(
        rng=rng,
        config=config,
        delay_ns=delay_ns,
        delay_jitter_ns=delay_jitter_ns,
        verbose=verbose,
        debug=debug,
        hooks=hooks,
        emission_diagnostics=emission_diagnostics,
        should_abort=should_abort,
    )

def _run_single_simulation_core(
    output_dir: Path,
    run_index: int,
    config: SimConfig,
    show_plots: bool,
    force_plot: bool = False,
    plot_dir: Optional[Path] = None,
    run_tag: Optional[str] = None,
    seed: Optional[int] = None,
    should_abort=None,
):
    run_wall_start = time.perf_counter()
    trace_enabled = bool(DEBUG_MODE)
    timer = TimingTracer(enabled=trace_enabled)
    run_cfg = config.run
    n_runs = run_cfg.runs
    shots_per_run = run_cfg.shots_per_run
    plot_all = run_cfg.plot_all
    plot_enabled = run_cfg.plot_enabled
    enum_mode = run_cfg.enum_mode
    run_tag = f"run{run_index:03d}"
    success_metrics = None
    stage_total = 6
    # 目的：落盘保存成功率/保真度等关键指标，便于复现与后处理。
    def _write_success_metrics_detail(
        output_dir: Path,
        run_tag: str,
        metrics: dict,
    ) -> Path:
        output_path = output_dir / f"{run_tag}_success_metrics.txt"
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write("success_metrics\n")
            file.write("=" * 60 + "\n")
            if "eta_det" in metrics:
                file.write(f"eta_det = {metrics['eta_det']:.6f}\n")
            if "eta_det_map" in metrics:
                file.write(f"eta_det_map = {metrics['eta_det_map']}\n")
            if "window_bins" in metrics:
                file.write(f"window_bins = {metrics['window_bins']}\n")
            if "window_ns" in metrics:
                file.write(f"window_ns = {metrics['window_ns']:.3f}\n")
            if "dark_rate_intrinsic_hz" in metrics:
                file.write(f"dark_rate_intrinsic_hz = {metrics['dark_rate_intrinsic_hz']:.3f}\n")
            if "dark_rate_bg_hz" in metrics:
                file.write(f"dark_rate_bg_hz = {metrics['dark_rate_bg_hz']:.3f}\n")
            if "detector_gate_ns" in metrics:
                file.write(f"detector_gate_ns = {metrics['detector_gate_ns']:.6f}\n")
            if "bins_per_gate" in metrics:
                file.write(f"bins_per_gate = {metrics['bins_per_gate']}\n")
            if "p_dark_intrinsic_gate" in metrics:
                file.write(f"p_dark_intrinsic_gate = {metrics['p_dark_intrinsic_gate']:.8f}\n")
            if "p_bg_gate" in metrics:
                file.write(f"p_bg_gate = {metrics['p_bg_gate']:.8f}\n")
            if "p_noise_gate" in metrics:
                file.write(f"p_noise_gate = {metrics['p_noise_gate']:.8f}\n")
            if "p_dark_intrinsic" in metrics:
                file.write(f"p_dark_intrinsic = {metrics['p_dark_intrinsic']:.8f}\n")
            if "p_dark_intrinsic_map" in metrics:
                file.write(f"p_dark_intrinsic_map = {metrics['p_dark_intrinsic_map']}\n")
            if "p_bg" in metrics:
                file.write(f"p_bg = {metrics['p_bg']:.8f}\n")
            if "p_bg_map" in metrics:
                file.write(f"p_bg_map = {metrics['p_bg_map']}\n")
            if "p_noise" in metrics:
                file.write(f"p_noise = {metrics['p_noise']:.8f}\n")
            if "t_wait_us" in metrics:
                file.write(f"t_wait_us = {metrics['t_wait_us']:.3f}\n")
            if "t2_us" in metrics:
                file.write(f"t2_us = {metrics['t2_us']:.3f}\n")
            if "p_dephase" in metrics:
                file.write(f"p_dephase = {metrics['p_dephase']:.6f}\n")
            if "p_qubit_emit" in metrics:
                file.write(f"p_qubit_emit = {metrics['p_qubit_emit']:.6f}\n")
            if "v_res" in metrics:
                file.write(f"v_res = {metrics['v_res']:.6f}\n")
            if "qfc_theta_H" in metrics:
                file.write(f"qfc_theta_H = {metrics['qfc_theta_H']:.6f}\n")
            if "qfc_theta_V" in metrics:
                file.write(f"qfc_theta_V = {metrics['qfc_theta_V']:.6f}\n")
            if metrics.get("p_arrive") is not None:
                file.write(f"p_arrive = {metrics['p_arrive']:.8f}\n")
            else:
                file.write("p_arrive = N/A\n")
            if metrics.get("p_arrive_11") is not None:
                file.write(f"p_arrive_11 = {metrics['p_arrive_11']:.8f}\n")
            else:
                file.write("p_arrive_11 = N/A\n")
            if metrics.get("p_arrive_same_arm") is not None:
                file.write(f"p_arrive_same_arm = {metrics['p_arrive_same_arm']:.8f}\n")
            else:
                file.write("p_arrive_same_arm = N/A\n")
            if metrics.get("p_arrive_20") is not None:
                file.write(f"p_arrive_20 = {metrics['p_arrive_20']:.8f}\n")
            else:
                file.write("p_arrive_20 = N/A\n")
            if metrics.get("p_arrive_02") is not None:
                file.write(f"p_arrive_02 = {metrics['p_arrive_02']:.8f}\n")
            else:
                file.write("p_arrive_02 = N/A\n")
            if metrics.get("p_success_no_dark_abs") is not None:
                file.write(f"p_success_no_dark_abs = {metrics['p_success_no_dark_abs']:.8f}\n")
                file.write(f"fidelity_no_dark = {metrics['fidelity_no_dark']:.6f}\n")
            else:
                file.write("p_success_no_dark_abs = N/A\n")
                file.write("fidelity_no_dark = N/A\n")
            if metrics.get("p_success_abs") is not None:
                file.write(f"p_success_abs = {metrics['p_success_abs']:.8f}\n")
            else:
                file.write("p_success_abs = N/A\n")
            if metrics.get("fidelity_all") is not None:
                file.write(f"fidelity_all = {metrics['fidelity_all']:.6f}\n")
            else:
                file.write("fidelity_all = N/A\n")
            if metrics.get("p_false_approx") is not None:
                file.write(f"p_false_approx = {metrics['p_false_approx']:.8f}\n")
                file.write(f"false_fraction_approx = {metrics['false_fraction_approx']:.6f}\n")
            else:
                file.write("p_false_approx = N/A\n")
                file.write("false_fraction_approx = N/A\n")

            if metrics.get("p_success_true_abs") is not None:
                file.write(f"p_success_true_abs = {metrics['p_success_true_abs']:.8f}\n")
            else:
                file.write("p_success_true_abs = N/A\n")
            if metrics.get("p_success_false_abs") is not None:
                file.write(f"p_success_false_abs = {metrics['p_success_false_abs']:.8f}\n")
            else:
                file.write("p_success_false_abs = N/A\n")
            if metrics.get("p_success_signal_heuristic") is not None:
                file.write(f"p_success_signal_heuristic = {metrics['p_success_signal_heuristic']:.8f}\n")
            else:
                file.write("p_success_signal_heuristic = N/A\n")
            if metrics.get("p_success_same_arm_heuristic") is not None:
                file.write(f"p_success_same_arm_heuristic = {metrics['p_success_same_arm_heuristic']:.8f}\n")
            else:
                file.write("p_success_same_arm_heuristic = N/A\n")
            if metrics.get("p_success_intrinsic_dark_assisted") is not None:
                file.write(
                    f"p_success_intrinsic_dark_assisted = {metrics['p_success_intrinsic_dark_assisted']:.8f}\n"
                )
            else:
                file.write("p_success_intrinsic_dark_assisted = N/A\n")
            if metrics.get("p_success_bg_assisted") is not None:
                file.write(f"p_success_bg_assisted = {metrics['p_success_bg_assisted']:.8f}\n")
            else:
                file.write("p_success_bg_assisted = N/A\n")
            if metrics.get("p_success_true_given_arrival") is not None:
                file.write(f"p_success_true_given_arrival = {metrics['p_success_true_given_arrival']:.8f}\n")
            else:
                file.write("p_success_true_given_arrival = N/A\n")
            if metrics.get("false_fraction") is not None:
                file.write(f"false_fraction = {metrics['false_fraction']:.6f}\n")
            else:
                file.write("false_fraction = N/A\n")
            if metrics.get("fidelity_true") is not None:
                file.write(f"fidelity_true = {metrics['fidelity_true']:.6f}\n")
            else:
                file.write("fidelity_true = N/A\n")
            if metrics.get("fidelity_false") is not None:
                file.write(f"fidelity_false = {metrics['fidelity_false']:.6f}\n")
            else:
                file.write("fidelity_false = N/A\n")
            if metrics.get("corr_exx") is not None:
                file.write(f"corr_exx = {metrics['corr_exx']:.6f}\n")
            else:
                file.write("corr_exx = N/A\n")
            if metrics.get("corr_eyy") is not None:
                file.write(f"corr_eyy = {metrics['corr_eyy']:.6f}\n")
            else:
                file.write("corr_eyy = N/A\n")
            if metrics.get("corr_ezz") is not None:
                file.write(f"corr_ezz = {metrics['corr_ezz']:.6f}\n")
            else:
                file.write("corr_ezz = N/A\n")
            if metrics.get("chsh_s_max") is not None:
                file.write(f"chsh_s_max = {metrics['chsh_s_max']:.6f}\n")
            else:
                file.write("chsh_s_max = N/A\n")
            if metrics.get("corr_exx_ff") is not None:
                file.write(f"corr_exx_ff = {metrics['corr_exx_ff']:.6f}\n")
            else:
                file.write("corr_exx_ff = N/A\n")
            if metrics.get("corr_eyy_ff") is not None:
                file.write(f"corr_eyy_ff = {metrics['corr_eyy_ff']:.6f}\n")
            else:
                file.write("corr_eyy_ff = N/A\n")
            if metrics.get("corr_ezz_ff") is not None:
                file.write(f"corr_ezz_ff = {metrics['corr_ezz_ff']:.6f}\n")
            else:
                file.write("corr_ezz_ff = N/A\n")
            if metrics.get("chsh_s_max_ff") is not None:
                file.write(f"chsh_s_max_ff = {metrics['chsh_s_max_ff']:.6f}\n")
            else:
                file.write("chsh_s_max_ff = N/A\n")
            if metrics.get("mps_chi_min") is not None:
                file.write(f"mps_chi_min = {int(metrics['mps_chi_min'])}\n")
            else:
                file.write("mps_chi_min = N/A\n")
            if metrics.get("mps_chi_mean") is not None:
                file.write(f"mps_chi_mean = {metrics['mps_chi_mean']:.6f}\n")
            else:
                file.write("mps_chi_mean = N/A\n")
            if metrics.get("mps_chi_max") is not None:
                file.write(f"mps_chi_max = {int(metrics['mps_chi_max'])}\n")
            else:
                file.write("mps_chi_max = N/A\n")
            if metrics.get("mps_trunc_total_calls") is not None:
                file.write(f"mps_trunc_total_calls = {int(metrics['mps_trunc_total_calls'])}\n")
            else:
                file.write("mps_trunc_total_calls = N/A\n")
            if metrics.get("mps_trunc_total_eps") is not None:
                file.write(f"mps_trunc_total_eps = {metrics['mps_trunc_total_eps']:.6e}\n")
            else:
                file.write("mps_trunc_total_eps = N/A\n")
            if metrics.get("mps_trunc_total_eps_max") is not None:
                file.write(f"mps_trunc_total_eps_max = {metrics['mps_trunc_total_eps_max']:.6e}\n")
            else:
                file.write("mps_trunc_total_eps_max = N/A\n")
        return output_path

    # 目的：绘图占位与清理逻辑集中，避免散落多个函数。
    # 目的：占用/判断是否允许绘图；规则：plot_all 或首次抢占成功。
    def _plot_gate_allow() -> bool:
        # --------------------------------------------------------------
        # 逻辑：默认只保留一个 run 的图，避免 I/O 爆炸。
        # plot_all=True 或 force_plot=True 时不做限制；
        # 否则仅 run_index==0 允许绘图。
        # --------------------------------------------------------------
        if not plot_enabled:
            return False
        if plot_all:
            return True
        if force_plot:
            return True
        return run_index == 0

    print("\n" + "=" * 80)
    print(f"Run {run_index}/{n_runs} ({run_tag})")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("运行发射 + QFC + 分束器 + 探测仿真...")

    run_rng = np.random.default_rng(seed)

    stage_map = {
        "发射": 1,
        "QFC + 滤波记忆(态端)": 2,
        "光纤信道 (Heisenberg 参数)": 3,
        "分束器(测量端) + 诊断/可视化": 4,
        "成功事件统计 (POVM)": 5,
        "POVM抽样": 6,
    }

    # 目的：输出阶段日志；公式：显示 "阶段 idx / 总阶段数"。
    def _on_stage(label: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        idx = stage_map[label]
        print(f"[{run_tag} {ts}] [阶段 {idx}/{stage_total}] {label}")

    def _merge_detection_timings(prefix: str, timing_map: Optional[dict]) -> None:
        if not trace_enabled or not timing_map:
            return
        key_alias = {
            "povm_effects": "povm_effects",
            "povm_enumeration": "povm_enumeration",
            "povm_sampling": "povm_sampling",
            "detection_total": "detection_total",
        }
        timer.merge_timing_map(
            timing_map,
            prefix=f"{prefix}_",
            key_alias=key_alias,
        )
        timer.add("povm_effects_total", timing_map.get("povm_effects", 0.0))
        timer.add("povm_enumeration_total", timing_map.get("povm_enumeration", 0.0))
        timer.add("povm_sampling_total", timing_map.get("povm_sampling", 0.0))
        timer.add("detection_total_all", timing_map.get("detection_total", 0.0))

    # 目的：统一生成阶段可视化与调试快照，减少重复代码。
    def _make_plot_hook(
        stage_name: str,
        file_suffix: str,
        show_atomic: bool,
        use_emission_obj: bool,
        use_time_grid: bool,
        debug_stage: str,
        step_index: int,
        bs_unitary: Optional[np.ndarray] = None,
    ):
        # --------------------------------------------------------------
        # 统一封装可视化 hook：
        #   - 只在“允许绘图”的 run 上执行
        #   - 输出命名规则统一，便于汇总
        #   - bs_unitary 用于 Heisenberg 端口“after BS”热图
        # --------------------------------------------------------------
        def _hook(
            emission,
            fiber_sample=None,
            qfc_params=None,
            *_args,
        ):
            if _plot_gate_allow():
                with timer.span("hook_plot_total"):
                    print(f"\n生成{stage_name}的可视化图...")
                    plot_path = plot_dir / f"{run_tag}_{file_suffix}.png"
                    plot_data_path = plot_dir / f"{run_tag}_{file_suffix}_states.csv"
                    target = emission if use_emission_obj else emission.mps
                    kwargs = dict(
                        save_path=str(plot_path),
                        show_atomic=show_atomic,
                        stage_name=stage_name,
                        show=show_plots,
                        export_csv_path=str(plot_data_path),
                    )
                    if use_time_grid:
                        kwargs["time_grid"] = {"dt_s": emission.dt_s}
                    if bs_unitary is not None:
                        kwargs["bs_unitary"] = bs_unitary
                    kwargs["qfc_params"] = qfc_params
                    kwargs["fiber_sample"] = fiber_sample
                    plot_dual_arm_heatmap(target, **kwargs)
            if trace_enabled:
                with timer.span("hook_debug_snapshot_total"):
                    stage_context = {
                        "representation": (
                            "heisenberg"
                            if "heisenberg" in debug_stage.strip().lower()
                            else "state_side"
                        ),
                        "acts_on_state": (
                            False if "heisenberg" in debug_stage.strip().lower() else True
                        ),
                    }
                    if qfc_params is not None and len(qfc_params) >= 2:
                        stage_context["qfc_theta_H"] = float(qfc_params[0])
                        stage_context["qfc_theta_V"] = float(qfc_params[1])
                    if fiber_sample is not None and len(fiber_sample) >= 9:
                        _, _, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase, phase_slope, phase_jitter_std = fiber_sample
                        stage_context["fiber_eta_H_A"] = float(eta_H_A)
                        stage_context["fiber_eta_V_A"] = float(eta_V_A)
                        stage_context["fiber_eta_H_B"] = float(eta_H_B)
                        stage_context["fiber_eta_V_B"] = float(eta_V_B)
                        stage_context["fiber_phase_drift_rad"] = float(phase)
                        stage_context["fiber_phase_slope_rad_per_bin"] = float(phase_slope)
                        stage_context["fiber_phase_jitter_std_rad"] = float(phase_jitter_std)
                    if bs_unitary is not None:
                        bs_theta = float(config.detector.bs_theta)
                        stage_context["bs_theta_rad"] = bs_theta
                        stage_context["bs_split_ratio"] = float(np.sin(bs_theta) ** 2)
                    save_debug_info(
                        mps=emission.mps,
                        n_bins=emission.get_n_bins(),
                        stage=debug_stage,
                        output_dir=output_dir,
                        step_index=step_index,
                        run_tag=run_tag,
                        stage_context=stage_context,
                    )
        return _hook

    _after_emission = _make_plot_hook(
        stage_name="After Emission",
        file_suffix="1_after_emission",
        show_atomic=True,
        use_emission_obj=True,
        use_time_grid=False,
        debug_stage="After Emission",
        step_index=1,
    )
    _after_qfc_filter = _make_plot_hook(
        stage_name="After QFC (State-side)",
        file_suffix="2_after_qfc",
        show_atomic=False,
        use_emission_obj=False,
        use_time_grid=True,
        debug_stage="After QFC (State-side)",
        step_index=2,
    )
    _after_fiber = _make_plot_hook(
        stage_name="After Fiber (Heisenberg)",
        file_suffix="3_after_fiber",
        show_atomic=False,
        use_emission_obj=False,
        use_time_grid=True,
        debug_stage="After Fiber (Heisenberg)",
        step_index=3,
    )
    _after_bs = _make_plot_hook(
        stage_name="After BS (Heisenberg)",
        file_suffix="4_after_bs",
        show_atomic=False,
        use_emission_obj=False,
        use_time_grid=True,
        debug_stage="After BS (Heisenberg)",
        step_index=4,
        bs_unitary=bs_gate_6d(config.detector.bs_theta),
    )

    pipe = _run_single_trial(
        rng=run_rng,
        config=config,
        delay_ns=None,
        delay_jitter_ns=None,
        verbose=True,
        debug=DEBUG_MODE,
        emission_diagnostics=(plot_enabled or DEBUG_MODE),
        hooks=PipelineHooks(
            on_stage=_on_stage,
            after_emission=_after_emission,
            after_qfc_filter=_after_qfc_filter,
            after_fiber=_after_fiber,
            after_bs=_after_bs,
        ),
        should_abort=should_abort,
    )
    if pipe.timings:
        timer.merge_timing_map(pipe.timings)

    result = pipe.emission
    p_qubit_emit = pipe.p_qubit_emit
    t_wait_us = pipe.t_wait_us
    t2_us = pipe.t2_us
    p_dephase = pipe.p_dephase

    # =========================================================================
    # 探测
    # =========================================================================
    # 探测参数（基于 Nature 2022 实验设置）
    # 符合窗口：默认采用论文中数据分析窗口 70 ns
    coincidence_window_ns = float(config.run.window_ns)
    bin_dt_s = result.dt_s
    with timer.span("build_param_store"):
        param_store = _build_run_parameter_store(
            config=config,
            emission_bin_dt_s=bin_dt_s,
            coincidence_window_ns=coincidence_window_ns,
            rng=run_rng,
        )
    budget = param_store.noise_budget
    eta_det = param_store.eta_det
    eta_det_map = param_store.eta_det_map
    p_dark_intrinsic_map = param_store.p_dark_intrinsic_bin_map
    p_bg_detector_map = param_store.p_bg_bin_map
    print(
        f"\n探测器本底暗计数率: {budget.dark_rate_intrinsic_hz:.3f} Hz -> "
        f"p_dark_gate={budget.p_dark_intrinsic_gate:.3e}, p_dark_bin={budget.p_dark_intrinsic_bin:.3e}"
    )
    print(
        f"背景噪声参数: mean={budget.bg_rate_mean_hz:.3f} Hz, std={budget.bg_rate_std_hz:.3f} Hz, "
        f"sampled={budget.dark_rate_bg_hz:.3f} Hz"
    )
    print(f"源背景噪声概率: p_bg_gate={budget.p_bg_gate:.3e}, p_bg_bin={budget.p_bg_bin:.3e}")
    print(
        f"合并噪声概率(仅预算展示): p_noise_gate={budget.p_noise_gate:.3e}, p_noise_bin={budget.p_noise_bin:.3e}"
    )
    print(
        f"点击时间窗 window_bins = {param_store.window_bins} "
        f"(~{param_store.window_bins * result.dt_s * 1e9:.1f} ns), gate={budget.detection_gate_ns:.3f} ns"
    )
    print(f"探测器效率映射 eta_det_map: {eta_det_map}")
    print(f"暗计数/bin 映射 p_dark_intrinsic_map: {p_dark_intrinsic_map}")
    print(f"背景/bin 映射 p_bg_map: {p_bg_detector_map}")

    _on_stage("成功事件统计 (POVM)")
    print(f"\n成功事件枚举模式: {enum_mode}")
    enum_no_dark = None
    enum_main = None
    samples = []
    run_stats = {
        "shots": 0,
        "success": 0,
        "bell": Counter(),
        "clicks": Counter(),
    }
    click_records = []

    if enum_mode == "no-dark":
        print("\n枚举成功事件（无暗计数）...")
        _on_stage("POVM抽样")
        print("\n运行探测和BSM（POVM抽样）...")
        _, pipeline = run_detection_core_from_pipe(
            pipe=pipe,
            config=config,
            rng=run_rng,
            coincidence_window_ns=coincidence_window_ns,
            shots_per_run=shots_per_run,
            compute_metrics=True,
            verbose=True,
            bs_theta=float(config.detector.bs_theta),
            param_store=param_store,
            p_dark_intrinsic_map={det: 0.0 for det in p_dark_intrinsic_map},
            p_bg_detector_map={det: 0.0 for det in p_bg_detector_map},
            should_abort=should_abort,
        )
        enum_no_dark = pipeline.metrics
        enum_main = enum_no_dark
        samples = pipeline.samples
        _merge_detection_timings("main", pipeline.timings)
    else:
        if enum_mode == "both":
            print("\n枚举成功事件（无暗计数基线）...")
            _, enum_pipeline = run_detection_core_from_pipe(
                pipe=pipe,
                config=config,
                rng=run_rng,
                coincidence_window_ns=coincidence_window_ns,
                shots_per_run=0,
                compute_metrics=True,
                verbose=True,
                bs_theta=float(config.detector.bs_theta),
                param_store=param_store,
                p_dark_intrinsic_map={det: 0.0 for det in p_dark_intrinsic_map},
                p_bg_detector_map={det: 0.0 for det in p_bg_detector_map},
                should_abort=should_abort,
            )
            enum_no_dark = enum_pipeline.metrics
            _merge_detection_timings("baseline", enum_pipeline.timings)
        print("\n枚举成功事件（含暗计数）...")
        _on_stage("POVM抽样")
        print("\n运行探测和BSM（POVM抽样）...")
        _, pipeline = run_detection_core_from_pipe(
            pipe=pipe,
            config=config,
            rng=run_rng,
            coincidence_window_ns=coincidence_window_ns,
            shots_per_run=shots_per_run,
            compute_metrics=True,
            verbose=True,
            bs_theta=float(config.detector.bs_theta),
            param_store=param_store,
            p_dark_intrinsic_map=p_dark_intrinsic_map,
            p_bg_detector_map=p_bg_detector_map,
            should_abort=should_abort,
        )
        enum_main = pipeline.metrics
        samples = pipeline.samples
        _merge_detection_timings("main", pipeline.timings)

    with timer.span("metrics_assemble"):
        chi_list = result.mps.get_bond_dimensions()
        trunc_stats = result.mps.get_truncation_stats()

        # 汇总统计量（跨 shots）
        success_metrics = {
            "eta_det": eta_det,
            "eta_det_map": eta_det_map,
            "window_bins": param_store.window_bins,
            "window_ns": coincidence_window_ns,
            "detector_gate_ns": budget.detection_gate_ns,
            "bins_per_gate": budget.bins_per_gate,
            "dark_rate_intrinsic_hz": budget.dark_rate_intrinsic_hz,
            "dark_rate_bg_hz": budget.dark_rate_bg_hz,
            "p_dark_intrinsic_gate": budget.p_dark_intrinsic_gate,
            "p_bg_gate": budget.p_bg_gate,
            "p_noise_gate": budget.p_noise_gate,
            "p_dark_intrinsic": budget.p_dark_intrinsic_bin,
            "p_dark_intrinsic_map": p_dark_intrinsic_map,
            "p_bg": budget.p_bg_bin,
            "p_bg_map": p_bg_detector_map,
            "p_noise": budget.p_noise_bin,
            "t_wait_us": t_wait_us,
            "t2_us": t2_us,
            "p_dephase": p_dephase,
            "p_qubit_emit": p_qubit_emit,
            "v_res": float(param_store.v_res),
            "bs_theta": float(config.detector.bs_theta),
            "bs_split_ratio": float(np.sin(config.detector.bs_theta) ** 2),
            "qfc_theta_H": pipe.qfc_theta_H,
            "qfc_theta_V": pipe.qfc_theta_V,
            "p_arrive": enum_main.p_arrive,
            "p_arrive_11": enum_main.p_arrive_11,
            "p_arrive_20": enum_main.p_arrive_20,
            "p_arrive_02": enum_main.p_arrive_02,
            "p_arrive_same_arm": enum_main.p_arrive_same_arm,
            "p_success_abs": enum_main.p_success,
            "p_success_true_abs": enum_main.p_success_true,
            "p_success_false_abs": enum_main.p_success_false,
            "p_success_signal_heuristic": enum_main.p_success_signal_heuristic,
            "p_success_same_arm_heuristic": enum_main.p_success_same_arm_heuristic,
            "p_success_intrinsic_dark_assisted": enum_main.p_success_intrinsic_dark_assisted,
            "p_success_bg_assisted": enum_main.p_success_bg_assisted,
            "p_success_true_given_arrival": enum_main.p_success_given_arrival,
            "fidelity_all": enum_main.fidelity_declared,
            "fidelity_true": enum_main.fidelity_true,
            "fidelity_false": enum_main.fidelity_false,
            "p_success_no_dark_abs": enum_no_dark.p_success if enum_no_dark is not None else None,
            "fidelity_no_dark": enum_no_dark.fidelity_declared if enum_no_dark is not None else None,
            "corr_exx": enum_main.corr_exx,
            "corr_eyy": enum_main.corr_eyy,
            "corr_ezz": enum_main.corr_ezz,
            "chsh_s_max": enum_main.chsh_s_max,
            "corr_exx_ff": enum_main.corr_exx_ff,
            "corr_eyy_ff": enum_main.corr_eyy_ff,
            "corr_ezz_ff": enum_main.corr_ezz_ff,
            "chsh_s_max_ff": enum_main.chsh_s_max_ff,
            "mps_chi_min": int(min(chi_list)) if chi_list else 0,
            "mps_chi_mean": float(np.mean(chi_list)) if chi_list else 0.0,
            "mps_chi_max": int(max(chi_list)) if chi_list else 0,
            "mps_trunc_total_calls": int(trunc_stats["total_calls"]),
            "mps_trunc_total_eps": float(trunc_stats["total_eps"]),
            "mps_trunc_total_eps_max": float(trunc_stats["total_eps_max"]),
            "fiber_length_km": config.fiber.length_km,
            "fiber_attenuation_db_per_km": config.fiber.attenuation_db_per_km,
            "fiber_eta_std": config.fiber.eta_std,
            "fiber_pdl_sigma": config.fiber.pdl_sigma,
            "fiber_phase_drift_std": config.fiber.phase_drift_std,
            "fiber_phase_slope_std": config.fiber.phase_slope_std,
            "fiber_phase_jitter_std": config.fiber.phase_jitter_std,
            "fiber_polarization_model": config.fiber.polarization_model,
            "fiber_polarization_sigma": config.fiber.polarization_sigma,
        }
        # 误判占比：false / all
        success_metrics["false_fraction"] = (
            success_metrics["p_success_false_abs"] / success_metrics["p_success_abs"]
            if success_metrics["p_success_abs"] > 0
            else 0.0
        )
        if enum_no_dark is not None:
            # 粗略估计：将暗计数引入的“额外成功”视为 false
            success_metrics["p_false_approx"] = max(
                0.0, success_metrics["p_success_abs"] - success_metrics["p_success_no_dark_abs"]
            )
            success_metrics["false_fraction_approx"] = (
                success_metrics["p_false_approx"] / success_metrics["p_success_abs"]
                if success_metrics["p_success_abs"] > 0
                else 0.0
            )
        else:
            success_metrics["p_false_approx"] = None
            success_metrics["false_fraction_approx"] = None

    with timer.span("io_success_metrics_write"):
        success_path = _write_success_metrics_detail(output_dir, run_tag, success_metrics)
    print(f"  Success metrics saved: {success_path.name}")

    with timer.span("samples_postprocess_total"):
        for shot_index, det_result in enumerate(samples, start=1):
            print(f"\n[shot {shot_index}/{shots_per_run}]")

            # 打印结果
            if det_result.success:
                print("\n  BSM成功!")
                print(f"  宣告的Bell态: {det_result.bell_state}")

                # 计算与期望Bell态的保真度（未归一化）
                fidelity_full = compute_fidelity_with_bell(det_result.qubit_state, det_result.bell_state)
                rho = det_result.qubit_state
                trace_rho = float(np.trace(rho).real)
                fidelity_cond = (fidelity_full / trace_rho) if trace_rho > 0 else 0.0
                print(f"  F_full(|{det_result.bell_state}>): {fidelity_full:.4e}")
                print(f"  F_cond(|{det_result.bell_state}>): {fidelity_cond:.4f}")

                # 计算与所有Bell态的保真度以供参考
                print("\n  与所有Bell态的保真度:")
                for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
                    f_full = compute_fidelity_with_bell(det_result.qubit_state, bell)
                    f_cond = (f_full / trace_rho) if trace_rho > 0 else 0.0
                    marker = " <-- 宣告的" if bell == det_result.bell_state else ""
                    print(f"    F_full(|{bell}>): {f_full:.4e}, F_cond: {f_cond:.4f}{marker}")

                # 打印量子比特态
                print("\n  量子比特密度矩阵（量子比特子空间）:")
                print(f"    Tr(rho) = {trace_rho:.4e}")
                print(f"    纯度(未归一化) = {np.trace(rho @ rho).real:.4f}")
            else:
                print("\n  BSM失败 - 未找到成功模式")
                print(f"  点击数量: {len(det_result.clicks)}")

            # 保存探测后的调试信息
            if trace_enabled:
                print("\n保存探测后调试信息...")
                if shots_per_run == 1:
                    det_file = output_dir / f'{run_tag}_debug_detection_result.txt'
                else:
                    det_file = output_dir / f'{run_tag}_shot{shot_index:03d}_debug_detection_result.txt'
                with timer.span("io_detection_debug_write"):
                    with open(det_file, 'w', encoding='utf-8') as file:
                        file.write('探测结果\n')
                        file.write('='*60 + '\n\n')
                        file.write('口径说明:\n')
                        file.write('  1) 该文件展示单次抽样得到的条件态（未归一化）。\n')
                        file.write('  2) success_metrics 来自全枚举统计，口径不同且不受本文件格式影响。\n\n')
                        file.write(f'成功: {det_result.success}\n')
                        file.write(f'Bell态: {det_result.bell_state}\n')
                        file.write(f'点击次数: {len(det_result.clicks)}\n')
                        if det_result.clicks:
                            file.write(
                                "点击详情: "
                                f"{[(c.detector, c.bin_index, bool(getattr(c, 'is_dark', False)), getattr(c, 'source', 'signal')) for c in det_result.clicks]}"
                                "\n"
                            )

                            file.write('\n量子比特密度矩阵:\n')
                            rho = det_result.qubit_state
                            trace_rho = float(np.trace(rho).real)
                            file.write('  基: |00>, |01>, |10>, |11>\n')
                            file.write(f'  Tr(rho) = {trace_rho:.6e}\n')
                            wrote_any = False
                            for i in range(4):
                                for j in range(4):
                                    val = rho[i, j]
                                    if abs(val) > 1e-14:
                                        file.write(
                                            f'  rho[{i},{j}] = {val.real:.6e}{val.imag:+.6e}j\n'
                                        )
                                        wrote_any = True
                            if not wrote_any:
                                file.write('  (所有矩阵元绝对值均 < 1e-14)\n')

                            purity_raw = float(np.trace(rho @ rho).real)
                            file.write(f'\n纯度(未归一化): {purity_raw:.6e}\n')
                            if trace_rho > 1e-15:
                                rho_cond = rho / trace_rho
                                purity_cond = float(np.trace(rho_cond @ rho_cond).real)
                                file.write(f'纯度(条件化): {purity_cond:.6f}\n')
                            else:
                                file.write('纯度(条件化): N/A (Tr(rho)≈0)\n')

                            file.write('\nBell态保真度:\n')
                            for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
                                fid_full = compute_fidelity_with_bell(rho, bell)
                                fid_cond = (fid_full / trace_rho) if trace_rho > 1e-15 else 0.0
                                marker = " <-- 探测到的" if bell == det_result.bell_state else ""
                                file.write(
                                    f'  F_full({bell}) = {fid_full:.6e}, '
                                    f'F_cond = {fid_cond:.6f}{marker}\n'
                                )

                print(f"  调试信息已保存: {det_file.name}")

            click_pairs = [
                (
                    c.detector,
                    c.bin_index,
                    bool(getattr(c, "is_dark", False)),
                    str(getattr(c, "source", "signal")),
                )
                for c in det_result.clicks
            ]
            click_records.append(
                {
                    "shot_index": shot_index - 1,
                    "success": bool(det_result.success),
                    "bell": det_result.bell_state,
                    "clicks": click_pairs,
                    "p_true_given_record": float(getattr(det_result, "p_true_given_record", 0.0)),
                    "p_bg_assist_given_record": float(
                        getattr(det_result, "p_bg_assist_given_record", 0.0)
                    ),
                    "p_intrinsic_dark_assist_given_record": float(
                        getattr(det_result, "p_intrinsic_dark_assist_given_record", 0.0)
                    ),
                }
            )
            declared = det_result.bell_state if det_result.success else "失败"
            if click_pairs:
                print(f"  宣告结果: {declared} | 点击记录: {click_pairs}")
            else:
                print(f"  宣告结果: {declared} | 点击记录: 无")

            run_stats["shots"] += 1
            run_stats["clicks"][len(det_result.clicks)] += 1
            if det_result.success:
                run_stats["success"] += 1
                if det_result.bell_state:
                    run_stats["bell"][det_result.bell_state] += 1

    with timer.span("io_declared_density_write"):
        write_declared_density_matrix(
            output_dir,
            rho_raw=getattr(enum_main, "rho_declared_raw", None),
            rho_ff=getattr(enum_main, "rho_declared_ff", None),
            trace_raw=float(getattr(enum_main, "trace_declared_raw", 0.0)),
            trace_ff=float(getattr(enum_main, "trace_declared_ff", 0.0)),
            rho_raw_by_bell=getattr(enum_main, "rho_declared_raw_by_bell", None),
            rho_ff_by_bell=getattr(enum_main, "rho_declared_ff_by_bell", None),
            trace_raw_by_bell=getattr(enum_main, "trace_declared_raw_by_bell", None),
            trace_ff_by_bell=getattr(enum_main, "trace_declared_ff_by_bell", None),
        )

    if trace_enabled:
        timer.set("run_wall_total", time.perf_counter() - run_wall_start)

    print(f"\n完成! 文件已保存至: {output_dir}/")
    if trace_enabled:
        timings = timer.snapshot()
        if shots_per_run > 0:
            det_total = float(timings.get("main_detection_total", 0.0))
            if det_total > 0.0:
                timings["main_detection_per_shot"] = det_total / shots_per_run

        timing_order = [
            ("emission", "发射"),
            ("qfc_filter_memory", "QFC+滤波记忆"),
            ("fiber", "光纤"),
            ("dephase", "退相干"),
            ("sanity_checks", "Sanity检查"),
            ("baseline_povm_effects", "POVM构建(基线)"),
            ("baseline_povm_enumeration", "POVM枚举(基线)"),
            ("baseline_povm_sampling", "POVM抽样(基线)"),
            ("baseline_detection_total", "探测总计(基线)"),
            ("main_povm_effects", "POVM构建(主流程)"),
            ("main_povm_enumeration", "POVM枚举(主流程)"),
            ("main_povm_sampling", "POVM抽样(主流程)"),
            ("main_detection_total", "探测总计(主流程)"),
            ("detection_total_all", "探测总计(合计)"),
            ("hook_plot_total", "Hook绘图"),
            ("hook_debug_snapshot_total", "Hook快照"),
            ("build_param_store", "参数预算"),
            ("metrics_assemble", "指标汇总"),
            ("io_success_metrics_write", "指标写盘"),
            ("io_declared_density_write", "后验态写盘"),
            ("io_detection_debug_write", "探测调试写盘"),
            ("samples_postprocess_total", "样本后处理"),
        ]
        parts = []
        for key, label in timing_order:
            if key in timings:
                value = float(timings[key])
                parts.append(f"{label}={value:.2f}s")
        if parts:
            print("\n[调试耗时] " + " | ".join(parts))
        if "run_wall_total" in timings:
            core_base_keys = ("emission", "qfc_filter_memory", "fiber", "dephase")
            core_sum = sum(float(timings[k]) for k in core_base_keys if k in timings)
            core_sum += float(timings.get("baseline_detection_total", 0.0))
            core_sum += float(timings.get("main_detection_total", 0.0))
            wall = float(timings["run_wall_total"])
            overhead_profiled_keys = (
                "sanity_checks",
                "hook_plot_total",
                "hook_debug_snapshot_total",
                "build_param_store",
                "metrics_assemble",
                "io_success_metrics_write",
                "io_declared_density_write",
                "io_detection_debug_write",
                "samples_postprocess_total",
            )
            overhead_profiled = sum(float(timings.get(k, 0.0)) for k in overhead_profiled_keys)
            residual = max(0.0, wall - core_sum - overhead_profiled)
            print(
                f"[调试总览] 核心阶段(去重)={core_sum:.2f}s | "
                f"run墙钟={wall:.2f}s | "
                f"额外开销(已计)={overhead_profiled:.2f}s | "
                f"额外开销(残差)={residual:.2f}s"
            )
    return run_stats, success_metrics, click_records


def iter_sim_core_tasks(config: SimConfig) -> Iterator[dict]:
    for run_index in range(config.run.runs):
        yield {
            "id": f"sim_run_{run_index:06d}",
            "experiment": "SIM",
            "run_index": run_index,
            "payload": {},
        }


def build_sim_task_metrics(run_stats: dict, run_index: int, success_metrics: dict) -> dict:
    metrics = {
        "shots": run_stats["shots"],
        "success": run_stats["success"],
        "run_index": run_index,
    }
    if not isinstance(success_metrics, dict):
        return metrics
    for key in SIM_TASK_METRIC_KEYS:
        if key in success_metrics:
            metrics[key] = success_metrics.get(key)
    return metrics


def run_sim_task(
    task: dict,
    config: SimConfig,
    raw_dir: Path,
    plots_dir: Path,
    task_id: str,
    should_abort=None,
) -> dict:
    seed_raw = task.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None
    run_index = int(task.get("run_index", 1))
    payload = task.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("SCHEMA_ERROR: SIM task payload 必须是对象")

    run_config = config
    force_plot = False
    if payload:
        run_config = deepcopy(config)
        if "n_bins" in payload:
            run_config.emission.n_bins = int(payload["n_bins"])
        if "dt_ns" in payload:
            run_config.emission.dt_ns = float(payload["dt_ns"])
        if "delay_ns" in payload:
            run_config.emission.delay_ns = (
                None if payload["delay_ns"] is None else float(payload["delay_ns"])
            )
        if "delay_jitter_ns" in payload:
            run_config.emission.delay_jitter_ns = float(payload["delay_jitter_ns"])
        if "plot_enabled" in payload:
            run_config.run.plot_enabled = bool(payload["plot_enabled"])
        if "force_plot" in payload:
            force_plot = bool(payload["force_plot"])

    run_stats, success_metrics, click_records = _run_single_simulation_core(
        output_dir=raw_dir,
        run_index=run_index,
        config=run_config,
        show_plots=run_config.run.plot_all,
        force_plot=force_plot,
        plot_dir=plots_dir,
        run_tag=task_id,
        seed=seed,
        should_abort=should_abort,
    )
    write_click_records(raw_dir, click_records)
    metrics = build_sim_task_metrics(run_stats, run_index, success_metrics)
    return metrics
