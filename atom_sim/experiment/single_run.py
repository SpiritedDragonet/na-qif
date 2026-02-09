# -*- coding: utf-8 -*-
"""
Single-run simulation workflow and summary helpers.
"""

from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter

import numpy as np

from ..simulation import (
    run_detection_pipeline,
    compute_fidelity_with_bell,
    compute_pauli_correlators_and_chsh,
)
from ..visualization import plot_dual_arm_heatmap
from ..physics.gates import bs_gate_6d
from .common import (
    SimConfig,
    PipelineHooks,
    run_emission_to_bs,
    _build_parameter_snapshot,
    _build_run_parameter_store,
    _build_detection_kwargs,
    _compute_t_wait_us_from_length,
)

# Debug toggle (default False)
DEBUG_MODE = False

def save_debug_info(
    mps,
    n_bins: int,
    stage: str,
    output_dir: Path,
    step_index: int,
    run_tag: Optional[str] = None,
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

    # MPS维度信息
    chi_list = mps._mps.chi
    d_list = mps.d
    info['n_sites'] = len(d_list)
    info['n_bins'] = n_bins
    info['bond_dimensions'] = f'chi_min={min(chi_list)}, chi_max={max(chi_list)}, chi_mean={np.mean(chi_list):.1f}'
    info['local_dimensions'] = f'first_5={d_list[:5]}, last_5={d_list[-5:]}'

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
        f.write('MPS维度信息:\n')
        f.write(f'  n_sites = {info["n_sites"]}\n')
        f.write(f'  n_bins = {info["n_bins"]}\n')
        f.write(f'  {info["bond_dimensions"]}\n')
        f.write(f'  {info["local_dimensions"]}\n\n')
        f.write('原子态信息:\n')
        f.write(f'  对角元: {info["qubit_state_diag"]}\n')
        f.write(f'  p_qubit: {info["p_qubit"]:.4f}\n')
        f.write(f'  纯度(条件化): {info["qubit_purity"]:.4f}\n\n')
        f.write('Bell态保真度:\n')
        f.write(f'  Psi+ (full/cond) = {info["fidelity_Psip_full"]:.4f} / {info["fidelity_Psip_cond"]:.4f}\n')
        f.write(f'  Psi- (full/cond) = {info["fidelity_Psim_full"]:.4f} / {info["fidelity_Psim_cond"]:.4f}\n')
        f.write(f'  Phi+ (full/cond) = {info["fidelity_Phip_full"]:.4f} / {info["fidelity_Phip_cond"]:.4f}\n')
        f.write(f'  Phi- (full/cond) = {info["fidelity_Phim_full"]:.4f} / {info["fidelity_Phim_cond"]:.4f}\n')

    print(f'  调试信息已保存: {info_file.name}')

def _run_single_trial(
    rng: Optional[np.random.Generator],
    config: SimConfig,
    delay_ns: Optional[float],
    delay_jitter_ns: Optional[float],
    verbose: bool,
    debug: bool,
    hooks: Optional[PipelineHooks],
    emission_diagnostics: bool,
):
    """
    目的：抽出最小可复用的物理流程（发射->QFC->滤波->光纤->退相干->BS并入测量）。
    规则：delay_ns/delay_jitter_ns 优先采用显式传入，否则取配置默认。
    """
    # 该函数只负责“物理链路”部分，不包含探测统计；
    # 便于 SIM/HOM 复用并减少重复代码。
    run_rng = rng or np.random.default_rng()
    t_wait_us = _compute_t_wait_us_from_length(
        length_km=config.fiber.length_km,
        fiber_group_velocity_mps=config.run.fiber_group_velocity_mps,
        t_wait_overhead_us=config.run.t_wait_overhead_us,
        t_wait_length_scale=config.run.t_wait_length_scale,
    )
    pipe = run_emission_to_bs(
        emission=config.emission,
        rng=run_rng,
        fiber=config.fiber,
        qfc=config.qfc,
        delay_ns=delay_ns,
        delay_jitter_ns=delay_jitter_ns,
        verbose=verbose,
        hooks=hooks,
        t_wait_us=t_wait_us,
        record_timings=debug,
        emission_diagnostics=emission_diagnostics,
    )
    return pipe

def _run_single_simulation_core(
    output_dir: Path,
    run_index: int,
    config: SimConfig,
    show_plots: bool,
    plot_dir: Optional[Path] = None,
    run_tag: Optional[str] = None,
    seed: Optional[int] = None,
):
    run_wall_start = time.perf_counter()
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
            if "apply_filter_780" in metrics:
                file.write(f"apply_filter_780 = {int(bool(metrics['apply_filter_780']))}\n")
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
            if metrics.get("p_success_signal_approx") is not None:
                file.write(f"p_success_signal_approx = {metrics['p_success_signal_approx']:.8f}\n")
            else:
                file.write("p_success_signal_approx = N/A\n")
            if metrics.get("p_success_same_arm_approx") is not None:
                file.write(f"p_success_same_arm_approx = {metrics['p_success_same_arm_approx']:.8f}\n")
            else:
                file.write("p_success_same_arm_approx = N/A\n")
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
        return output_path

    # 目的：绘图占位与清理逻辑集中，避免散落多个函数。
    # 目的：占用/判断是否允许绘图；规则：plot_all 或首次抢占成功。
    def _plot_gate_allow() -> bool:
        # --------------------------------------------------------------
        # 逻辑：默认只保留一个 run 的图，避免 I/O 爆炸。
        # plot_all=True 时不做限制；否则仅 run_index==0 允许绘图。
        # --------------------------------------------------------------
        if not plot_enabled:
            return False
        if plot_all:
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
            apply_filter_780: bool = True,
            *_args,
        ):
            if _plot_gate_allow():
                print(f"\n生成{stage_name}的可视化图...")
                plot_path = plot_dir / f"{run_tag}_{file_suffix}.png"
                target = emission if use_emission_obj else emission.mps
                kwargs = dict(
                    save_path=str(plot_path),
                    show_atomic=show_atomic,
                    stage_name=stage_name,
                    show=show_plots,
                )
                if use_time_grid:
                    kwargs["time_grid"] = {"dt_s": emission.dt_s}
                if bs_unitary is not None:
                    kwargs["bs_unitary"] = bs_unitary
                kwargs["qfc_params"] = qfc_params
                kwargs["fiber_sample"] = fiber_sample
                kwargs["apply_filter_780"] = apply_filter_780
                plot_dual_arm_heatmap(target, **kwargs)
            if DEBUG_MODE:
                save_debug_info(
                    mps=emission.mps,
                    n_bins=emission.get_n_bins(),
                    stage=debug_stage,
                    output_dir=output_dir,
                    step_index=step_index,
                    run_tag=run_tag,
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

    timings = {} if DEBUG_MODE else None
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
    )
    if DEBUG_MODE and pipe.timings:
        timings.update(pipe.timings)

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
    p_bg_source_map = param_store.p_bg_bin_map
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
    print(f"背景/bin 映射 p_bg_map: {p_bg_source_map}")

    parameter_snapshot = _build_parameter_snapshot(config, param_store)
    # 重要：BS 已并入测量端。这里传入 U_BS，用 U^† E U 计算点击分布。
    bs_unitary = bs_gate_6d(config.detector.bs_theta)
    detect_common = _build_detection_kwargs(
        pipe=pipe,
        param_store=param_store,
        rng=run_rng,
        verbose=True,
        bs_unitary=bs_unitary,
        bs_theta=config.detector.bs_theta,
    )

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
        detect_start = time.perf_counter() if DEBUG_MODE else None
        pipeline = run_detection_pipeline(
            **detect_common,
            p_dark_intrinsic={det: 0.0 for det in p_dark_intrinsic_map},
            p_bg_source={det: 0.0 for det in p_bg_source_map},
            n_samples=shots_per_run,
            compute_metrics=True,
        )
        enum_no_dark = pipeline.metrics
        enum_main = enum_no_dark
        samples = pipeline.samples
        if DEBUG_MODE and timings is not None and pipeline.timings:
            timings["povm_effects"] = pipeline.timings.get("povm_effects", 0.0)
            timings["povm_enumeration"] = pipeline.timings.get("povm_enumeration", 0.0)
            timings["povm_sampling"] = pipeline.timings.get("povm_sampling", 0.0)
            timings["detection_total"] = pipeline.timings.get("detection_total", 0.0)
    else:
        if enum_mode == "both":
            print("\n枚举成功事件（无暗计数基线）...")
            enum_pipeline = run_detection_pipeline(
                **detect_common,
                p_dark_intrinsic={det: 0.0 for det in p_dark_intrinsic_map},
                p_bg_source={det: 0.0 for det in p_bg_source_map},
                n_samples=0,
                compute_metrics=True,
            )
            enum_no_dark = enum_pipeline.metrics
            if DEBUG_MODE and timings is not None and enum_pipeline.timings:
                timings["povm_effects"] = enum_pipeline.timings.get("povm_effects", 0.0)
                timings["povm_enumeration"] = enum_pipeline.timings.get("povm_enumeration", 0.0)
        print("\n枚举成功事件（含暗计数）...")
        _on_stage("POVM抽样")
        print("\n运行探测和BSM（POVM抽样）...")
        detect_start = time.perf_counter() if DEBUG_MODE else None
        pipeline = run_detection_pipeline(
            **detect_common,
            p_dark_intrinsic=p_dark_intrinsic_map,
            p_bg_source=p_bg_source_map,
            n_samples=shots_per_run,
            compute_metrics=True,
        )
        enum_main = pipeline.metrics
        samples = pipeline.samples
        if DEBUG_MODE and timings is not None and pipeline.timings:
            timings["povm_effects"] = pipeline.timings.get("povm_effects", 0.0)
            timings["povm_enumeration"] = pipeline.timings.get("povm_enumeration", 0.0)
            timings["povm_sampling"] = pipeline.timings.get("povm_sampling", 0.0)
            timings["detection_total"] = pipeline.timings.get("detection_total", 0.0)

    corr_exx_vals = []
    corr_eyy_vals = []
    corr_ezz_vals = []
    chsh_vals = []
    for det_result in samples:
        if not det_result.success:
            continue
        corr = compute_pauli_correlators_and_chsh(det_result.qubit_state)
        corr_exx_vals.append(float(corr["corr_exx"]))
        corr_eyy_vals.append(float(corr["corr_eyy"]))
        corr_ezz_vals.append(float(corr["corr_ezz"]))
        chsh_vals.append(float(corr["chsh_s_max"]))

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
        "p_bg_map": p_bg_source_map,
        "p_noise": budget.p_noise_bin,
        "t_wait_us": t_wait_us,
        "t2_us": t2_us,
        "p_dephase": p_dephase,
        "p_qubit_emit": p_qubit_emit,
        "v_res": detect_common["v_res"],
        "bs_theta": float(config.detector.bs_theta),
        "bs_split_ratio": float(np.sin(config.detector.bs_theta) ** 2),
        "qfc_theta_H": pipe.qfc_theta_H,
        "qfc_theta_V": pipe.qfc_theta_V,
        "apply_filter_780": pipe.apply_filter_780,
        "p_arrive": enum_main.p_arrive,
        "p_arrive_11": enum_main.p_arrive_11,
        "p_arrive_20": enum_main.p_arrive_20,
        "p_arrive_02": enum_main.p_arrive_02,
        "p_arrive_same_arm": enum_main.p_arrive_same_arm,
        "p_success_abs": enum_main.p_success,
        "p_success_true_abs": enum_main.p_success_true,
        "p_success_false_abs": enum_main.p_success_false,
        "p_success_signal_approx": enum_main.p_success_signal_approx,
        "p_success_same_arm_approx": enum_main.p_success_same_arm_approx,
        "p_success_intrinsic_dark_assisted": enum_main.p_success_intrinsic_dark_assisted,
        "p_success_bg_assisted": enum_main.p_success_bg_assisted,
        "p_success_true_given_arrival": enum_main.p_success_given_arrival,
        "fidelity_all": enum_main.fidelity_declared,
        "fidelity_true": enum_main.fidelity_true,
        "fidelity_false": enum_main.fidelity_false,
        "p_success_no_dark_abs": enum_no_dark.p_success if enum_no_dark is not None else None,
        "fidelity_no_dark": enum_no_dark.fidelity_declared if enum_no_dark is not None else None,
        "corr_exx": float(np.mean(corr_exx_vals)) if corr_exx_vals else 0.0,
        "corr_eyy": float(np.mean(corr_eyy_vals)) if corr_eyy_vals else 0.0,
        "corr_ezz": float(np.mean(corr_ezz_vals)) if corr_ezz_vals else 0.0,
        "chsh_s_max": float(np.mean(chsh_vals)) if chsh_vals else 0.0,
        "parameter_snapshot": parameter_snapshot,
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

    success_path = _write_success_metrics_detail(output_dir, run_tag, success_metrics)
    print(f"  Success metrics saved: {success_path.name}")

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
        if DEBUG_MODE:
            print("\n保存探测后调试信息...")
            if shots_per_run == 1:
                det_file = output_dir / f'{run_tag}_debug_detection_result.txt'
            else:
                det_file = output_dir / f'{run_tag}_shot{shot_index:03d}_debug_detection_result.txt'
            with open(det_file, 'w', encoding='utf-8') as file:
                file.write('探测结果\n')
                file.write('='*60 + '\n\n')
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
                    file.write('  基: |00>, |01>, |10>, |11>\n')
                    for i in range(4):
                        for j in range(4):
                            val = rho[i, j]
                            if abs(val) > 1e-10:
                                file.write(f'  rho[{i},{j}] = {val:.4f}\n')

                    file.write(f'\n纯度: {np.trace(rho @ rho).real:.4f}\n')

                    file.write('\nBell态保真度:\n')
                    for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
                        fid = compute_fidelity_with_bell(rho, bell)
                        marker = " <-- 探测到的" if bell == det_result.bell_state else ""
                        file.write(f'  F({bell}) = {fid:.4f}{marker}\n')

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

    if DEBUG_MODE and timings is not None and detect_start is not None:
        timings["detection_total"] = time.perf_counter() - detect_start
        if shots_per_run > 0:
            timings["detection_per_shot"] = timings["detection_total"] / shots_per_run

    if DEBUG_MODE and timings is not None:
        timings["run_wall_total"] = time.perf_counter() - run_wall_start

    print(f"\n完成! 文件已保存至: {output_dir}/")
    if DEBUG_MODE and timings:
        timing_order = [
            ("emission", "发射"),
            ("qfc_filter_memory", "QFC+滤波记忆"),
            ("fiber", "光纤"),
            ("dephase", "退相干"),
            ("povm_effects", "POVM构建"),
            ("povm_enumeration", "POVM枚举"),
            ("povm_sampling", "POVM抽样"),
            ("detection_total", "探测总计"),
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
            if "detection_total" in timings:
                core_sum += float(timings["detection_total"])
            else:
                core_sum += sum(
                    float(timings[k])
                    for k in ("povm_effects", "povm_enumeration", "povm_sampling")
                    if k in timings
                )
            wall = float(timings["run_wall_total"])
            overhead = max(0.0, wall - core_sum)
            print(
                f"[调试总览] 核心阶段(去重)={core_sum:.2f}s | "
                f"run墙钟={wall:.2f}s | 额外开销={overhead:.2f}s"
            )
    return run_stats, success_metrics, click_records
