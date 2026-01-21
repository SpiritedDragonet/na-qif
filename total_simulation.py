# -*- coding: utf-8 -*-
"""
完整仿真流程：双原子发射 -> 时间仓波包 -> QFC -> 分束器 -> 探测

仿真阶段：
1. 双原子发射（780nm光子到时间仓）
2. QFC频率转换（780nm -> 1517nm）
3. 780nm滤波器（滤除未转换的光子）
4. 分束器（A_n与B_n干涉）
5. 双光子探测与Bell态测量

链结构（新架构）：
    初始：[A1, B1, ..., AN, BN, atomA, atomB]  (仓在前，原子在后)
    发射后：[atomA, atomB, A1, B1, ..., AN, BN]  (原子向左移动到最前)

原子向左移动，依次与每个仓对相互作用，不再需要SWAP conveyor belt。
"""

import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Optional, Tuple
from collections import Counter

# Add project root to path (for running as standalone script)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from atom_sim.config import TimeGrid, EmitParams
from atom_sim.simulation import (
    run_dual_atom_emission, EmissionResult, apply_qfc, apply_780_filter, apply_fiber_channel,
    apply_bs, project_to_1517,
    # 探测
    run_two_photon_detection, enumerate_success_events,
    compute_fidelity_with_bell, compute_photon_statistics,
)
from atom_sim.visualization import plot_dual_arm_heatmap
from atom_sim.visualization.wavepacket import plot_cross_bin_joint_heatmap
from atom_sim.physics import FiberChannelParams


class Tee:
    """同时输出到多个流（用于日志与控制台同步）。"""
    def __init__(self, *streams):
        self._streams = streams
        self._tty_streams = [
            s for s in streams if getattr(s, "isatty", lambda: False)()
        ]

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        for stream in self._tty_streams:
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return bool(self._tty_streams)


def _parse_run_params(argv) -> Tuple[int, int, int]:
    if len(argv) < 2:
        return 1, 1, 1
    if len(argv) > 4:
        print("用法: python total_simulation.py [N_runs] [shots_per_run] [jobs]")
        sys.exit(1)
    try:
        n_runs = int(argv[1])
    except ValueError:
        print("用法: python total_simulation.py [N_runs] [shots_per_run] [jobs]")
        sys.exit(1)
    if n_runs < 1:
        print("N_runs 必须 >= 1")
        sys.exit(1)
    shots_per_run = 1
    if len(argv) >= 3:
        try:
            shots_per_run = int(argv[2])
        except ValueError:
            print("用法: python total_simulation.py [N_runs] [shots_per_run] [jobs]")
            sys.exit(1)
        if shots_per_run < 1:
            print("shots_per_run 必须 >= 1")
            sys.exit(1)
    jobs = 1
    if len(argv) >= 4:
        try:
            jobs = int(argv[3])
        except ValueError:
            print("用法: python total_simulation.py [N_runs] [shots_per_run] [jobs]")
            sys.exit(1)
        if jobs < 1:
            print("jobs 必须 >= 1")
            sys.exit(1)
    return n_runs, shots_per_run, jobs


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
        compute_photon_statistics,
        extract_spin_state, compute_fidelity_with_bell
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

    # 光子统计
    stats = compute_photon_statistics(mps, n_bins, verbose=False)
    info['photon_stats'] = stats

    # 原子态信息
    spin_state = extract_spin_state(mps, n_bins)
    info['spin_state_diag'] = np.diag(spin_state).real.tolist()
    info['spin_purity'] = float(np.real(np.trace(spin_state @ spin_state)))

    # Bell态保真度
    for bell in ['Psi+', 'Psi-', 'Phi+', 'Phi-']:
        info[f'fidelity_{bell.replace("+", "p").replace("-", "m")}'] = compute_fidelity_with_bell(spin_state, bell)

    # 保存到文件
    prefix = f"{run_tag}_" if run_tag else ""
    info_file = output_dir / f'{prefix}debug_step_{step_index:02d}_{stage.replace(" ", "_").lower()}.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f'调试信息 - {stage}\n')
        f.write('='*60 + '\n\n')
        f.write(f'MPS维度信息:\n')
        f.write(f'  n_sites = {info["n_sites"]}\n')
        f.write(f'  n_bins = {info["n_bins"]}\n')
        f.write(f'  {info["bond_dimensions"]}\n')
        f.write(f'  {info["local_dimensions"]}\n\n')
        f.write(f'光子统计:\n')
        f.write(f'  总期望光子数 = {stats["n_total"]:.4f}\n')
        f.write(f'  780nm: H={stats.get("n_780_H", 0):.4f}, V={stats.get("n_780_V", 0):.4f}, total={stats.get("n_780_total", 0):.4f}\n')
        f.write(f'  1517nm: H={stats.get("n_1517_H", 0):.4f}, V={stats.get("n_1517_V", 0):.4f}, total={stats.get("n_1517_total", 0):.4f}\n')
        f.write(f'  期望损耗光子数 = {stats["loss_prob"]:.4f}\n\n')
        f.write(f'原子态信息:\n')
        f.write(f'  对角元: {info["spin_state_diag"]}\n')
        f.write(f'  纯度: {info["spin_purity"]:.4f}\n\n')
        f.write(f'Bell态保真度:\n')
        f.write(f'  Psi+ = {info["fidelity_Psip"]:.4f}\n')
        f.write(f'  Psi- = {info["fidelity_Psim"]:.4f}\n')
        f.write(f'  Phi+ = {info["fidelity_Phip"]:.4f}\n')
        f.write(f'  Phi- = {info["fidelity_Phim"]:.4f}\n')

    print(f'  调试信息已保存: {info_file.name}')


def _append_click_summary(
    summary_path: Path,
    run_index: int,
    shot_index: int,
    det_result,
):
    clicks = [(c.detector, c.bin_index) for c in det_result.clicks]
    with open(summary_path, 'a', encoding='utf-8') as file:
        file.write(
            f'{run_index}\t{shot_index}\t{det_result.success}\t'
            f'{det_result.bell_state}\t{len(clicks)}\t{clicks}\n'
        )


def _init_stats() -> dict:
    return {
        "shots": 0,
        "success": 0,
        "bell": Counter(),
        "clicks": Counter(),
    }


def _merge_stats(dst: dict, src: dict) -> None:
    dst["shots"] += src["shots"]
    dst["success"] += src["success"]
    dst["bell"].update(src["bell"])
    dst["clicks"].update(src["clicks"])


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "-"
    parts = []
    for key in sorted(counter.keys()):
        parts.append(f"{key}:{counter[key]}")
    return ",".join(parts)


def _append_run_summary(
    summary_path: Path,
    run_id,
    stats: dict,
) -> None:
    shots = stats["shots"]
    success = stats["success"]
    success_rate = (success / shots) if shots > 0 else 0.0
    bell_str = _format_counter(stats["bell"])
    click_str = _format_counter(stats["clicks"])
    with open(summary_path, 'a', encoding='utf-8') as file:
        file.write(
            f'{run_id}\t{shots}\t{success}\t{success_rate:.4f}\t'
            f'{bell_str}\t{click_str}\n'
        )


def _write_overall_summary(
    summary_path: Path,
    stats: dict,
    n_runs: int,
    shots_per_run: int,
) -> None:
    shots = stats["shots"]
    success = stats["success"]
    success_rate = (success / shots) if shots > 0 else 0.0
    bell_str = _format_counter(stats["bell"])
    click_str = _format_counter(stats["clicks"])
    with open(summary_path, 'w', encoding='utf-8') as file:
        file.write("Overall summary\n")
        file.write("=" * 60 + "\n")
        file.write(f"runs = {n_runs}\n")
        file.write(f"shots_per_run = {shots_per_run}\n")
        file.write(f"total_shots = {shots}\n")
        file.write(f"success = {success}\n")
        file.write(f"success_rate = {success_rate:.4f}\n")
        file.write(f"bell_counts = {bell_str}\n")
        file.write(f"click_count_dist = {click_str}\n")


def _write_extra_data(
    output_path: Path,
    fiber_sample,
    pre_bs_arm_a: np.ndarray,
    pre_bs_arm_b: np.ndarray,
    cross_bin_joint: np.ndarray,
) -> None:
    def _format_matrix(mat: np.ndarray) -> str:
        return np.array2string(
            mat,
            precision=6,
            suppress_small=True,
            separator=", ",
        )

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("fiber_sample\n")
        if fiber_sample is not None:
            U_A, U_B, eta, phase = fiber_sample
            file.write(f"eta = {eta:.6f}\n")
            file.write(f"phase = {phase:.6f}\n")
            file.write(f"U_A = {_format_matrix(U_A)}\n")
            file.write(f"U_B = {_format_matrix(U_B)}\n")
        else:
            file.write("fiber_sample = None\n")

        file.write("\npre_bs_photon_per_bin\n")
        file.write("bin\tarmA\tarmB\n")
        for idx in range(len(pre_bs_arm_a)):
            file.write(f"{idx}\t{pre_bs_arm_a[idx]:.6f}\t{pre_bs_arm_b[idx]:.6f}\n")

        file.write("\ncross_bin_joint_after_bs\n")
        file.write(f"shape = {cross_bin_joint.shape}\n")
        np.savetxt(file, cross_bin_joint, fmt="%.6e")


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
        if "p_dark" in metrics:
            file.write(f"p_dark = {metrics['p_dark']:.6f}\n")
        file.write(f"p_arrive = {metrics['p_arrive']:.8f}\n")
        file.write("\nmethod_1_two_runs\n")
        file.write(f"p_success_no_dark = {metrics['p_success_no_dark']:.8f}\n")
        file.write(f"fidelity_no_dark = {metrics['fidelity_no_dark']:.6f}\n")
        file.write(f"p_success_all = {metrics['p_success_all']:.8f}\n")
        file.write(f"fidelity_all = {metrics['fidelity_all']:.6f}\n")
        file.write(f"p_false_approx = {metrics['p_false_approx']:.8f}\n")
        file.write(f"false_fraction_approx = {metrics['false_fraction_approx']:.6f}\n")

        file.write("\nmethod_2_kraus_tag\n")
        file.write(f"p_success_true = {metrics['p_success_true']:.8f}\n")
        file.write(f"p_success_false = {metrics['p_success_false']:.8f}\n")
        file.write(f"p_success_given_arrival = {metrics['p_success_given_arrival']:.8f}\n")
        file.write(f"false_fraction = {metrics['false_fraction']:.6f}\n")
        file.write(f"fidelity_true = {metrics['fidelity_true']:.6f}\n")
        file.write(f"fidelity_false = {metrics['fidelity_false']:.6f}\n")
    return output_path


def _append_success_metrics_summary(
    summary_path: Path,
    run_id,
    metrics: dict,
) -> None:
    with open(summary_path, 'a', encoding='utf-8') as file:
        file.write(
            f"{run_id}\t{metrics['p_arrive']:.8f}\t{metrics['p_success_all']:.8f}\t"
            f"{metrics['p_success_true']:.8f}\t{metrics['p_success_false']:.8f}\t"
            f"{metrics['p_success_given_arrival']:.8f}\t{metrics['fidelity_all']:.6f}\t"
            f"{metrics['fidelity_true']:.6f}\t{metrics['fidelity_false']:.6f}\t"
            f"{metrics['p_success_no_dark']:.8f}\t{metrics['fidelity_no_dark']:.6f}\t"
            f"{metrics['p_false_approx']:.8f}\t{metrics['false_fraction']:.6f}\t"
            f"{metrics['false_fraction_approx']:.6f}\n"
        )


def _init_success_metrics_accumulator() -> dict:
    return {
        "runs": 0,
        "p_arrive_sum": 0.0,
        "p_success_sum": 0.0,
        "p_success_true_sum": 0.0,
        "p_success_false_sum": 0.0,
        "p_success_no_dark_sum": 0.0,
        "fidelity_weighted_sum": 0.0,
        "fidelity_true_weighted_sum": 0.0,
        "fidelity_false_weighted_sum": 0.0,
        "fidelity_no_dark_weighted_sum": 0.0,
    }


def _accumulate_success_metrics(acc: dict, metrics: dict) -> None:
    acc["runs"] += 1
    acc["p_arrive_sum"] += metrics["p_arrive"]
    acc["p_success_sum"] += metrics["p_success_all"]
    acc["p_success_true_sum"] += metrics["p_success_true"]
    acc["p_success_false_sum"] += metrics["p_success_false"]
    acc["p_success_no_dark_sum"] += metrics["p_success_no_dark"]
    acc["fidelity_weighted_sum"] += metrics["p_success_all"] * metrics["fidelity_all"]
    acc["fidelity_true_weighted_sum"] += metrics["p_success_true"] * metrics["fidelity_true"]
    acc["fidelity_false_weighted_sum"] += metrics["p_success_false"] * metrics["fidelity_false"]
    acc["fidelity_no_dark_weighted_sum"] += metrics["p_success_no_dark"] * metrics["fidelity_no_dark"]


def _finalize_success_metrics(acc: dict) -> dict:
    runs = max(acc["runs"], 1)
    p_arrive = acc["p_arrive_sum"] / runs
    p_success_all = acc["p_success_sum"] / runs
    p_success_true = acc["p_success_true_sum"] / runs
    p_success_false = acc["p_success_false_sum"] / runs
    p_success_no_dark = acc["p_success_no_dark_sum"] / runs
    p_success_given_arrival = (acc["p_success_true_sum"] / acc["p_arrive_sum"]) if acc["p_arrive_sum"] > 0 else 0.0

    fidelity_all = acc["fidelity_weighted_sum"] / acc["p_success_sum"] if acc["p_success_sum"] > 0 else 0.0
    fidelity_true = acc["fidelity_true_weighted_sum"] / acc["p_success_true_sum"] if acc["p_success_true_sum"] > 0 else 0.0
    fidelity_false = acc["fidelity_false_weighted_sum"] / acc["p_success_false_sum"] if acc["p_success_false_sum"] > 0 else 0.0
    fidelity_no_dark = acc["fidelity_no_dark_weighted_sum"] / acc["p_success_no_dark_sum"] if acc["p_success_no_dark_sum"] > 0 else 0.0

    p_false_approx = max(0.0, p_success_all - p_success_no_dark)
    false_fraction = (p_success_false / p_success_all) if p_success_all > 0 else 0.0
    false_fraction_approx = (p_false_approx / p_success_all) if p_success_all > 0 else 0.0

    return {
        "p_arrive": p_arrive,
        "p_success_all": p_success_all,
        "p_success_true": p_success_true,
        "p_success_false": p_success_false,
        "p_success_given_arrival": p_success_given_arrival,
        "fidelity_all": fidelity_all,
        "fidelity_true": fidelity_true,
        "fidelity_false": fidelity_false,
        "p_success_no_dark": p_success_no_dark,
        "fidelity_no_dark": fidelity_no_dark,
        "p_false_approx": p_false_approx,
        "false_fraction": false_fraction,
        "false_fraction_approx": false_fraction_approx,
    }

def _append_clicks_file(
    target_path: Path,
    run_clicks_path: Path,
) -> None:
    if not run_clicks_path.exists():
        return
    with open(target_path, 'a', encoding='utf-8') as dst:
        with open(run_clicks_path, 'r', encoding='utf-8') as src:
            for idx, line in enumerate(src):
                if idx == 0 and line.startswith("run\t"):
                    continue
                dst.write(line)


def _run_single_simulation_core(
    output_dir: Path,
    run_index: int,
    n_runs: int,
    run_clicks_path: Path,
    shots_per_run: int,
    show_plots: bool,
):
    run_tag = f"run{run_index:03d}"
    success_metrics = None
    print("\n" + "=" * 80)
    print(f"Run {run_index}/{n_runs} ({run_tag})")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("运行发射 + QFC + 分束器 + 探测仿真...")

    run_rng = np.random.default_rng()
    fiber_params = FiberChannelParams()

    # 运行发射
    # 使用合理的物理参数
    delay_jitter_ns = 0.0  # 设置为 >0 可启用A/B延迟随机抖动（ns）
    result = run_dual_atom_emission(
        n_bins=30,  # 仓数
        dt_ns=0.2,   # 时间步长
        chi_max=50,
        gamma_peak_A=0.5,  # 发射率
        gamma_peak_B=0.5,
        sigma=10.0,  # 波包宽度（纳秒）
        delay_ns=5.0,  # B相对于A延迟5ns（半个波包宽度）
        delay_jitter_ns=delay_jitter_ns,
        rng=run_rng,
        verbose=True,
    )

    # 保存发射后的可视化
    print("\n生成发射后的可视化图...")
    plot_dual_arm_heatmap(
        result,
        save_path=str(output_dir / f"{run_tag}_1_after_emission.png"),
        show_atomic=True,
        stage_name="After Emission",
        show=show_plots,
    )

    # 保存调试信息
    print("\n保存调试信息...")
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After Emission',
        output_dir=output_dir,
        step_index=1,
        run_tag=run_tag,
    )

    # 应用QFC
    print("\n应用QFC...")
    apply_qfc(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        theta_H=np.pi/4,  # 50% 转换
        theta_V=np.pi/4,
        verbose=True,
    )

    # 应用780nm滤波器（滤除未转换的780nm光子）
    print("\n应用780nm滤波器...")
    apply_780_filter(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )

    # 投影到纯1517nm子空间（18D -> 6D），大幅加速后续计算
    print("\n投影到1517nm子空间...")
    project_to_1517(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )

    # 保存QFC+滤波后的可视化
    print("\n生成QFC+滤波后的可视化图...")
    plot_dual_arm_heatmap(
        result.mps,
        save_path=str(output_dir / f"{run_tag}_2_after_qfc.png"),
        show_atomic=False,
        stage_name="After QFC + 780nm Filter",
        time_grid=result.time_grid,
        show=show_plots,
    )

    # 保存调试信息
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After QFC + Filter',
        output_dir=output_dir,
        step_index=2,
        run_tag=run_tag,
    )

    # 应用光纤信道（偏振漂移 + 损耗）
    print("\n应用光纤信道...")
    result.mps, fiber_sample = apply_fiber_channel(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        fiber_params=fiber_params,
        rng=run_rng,
        verbose=True,
    )

    # 保存光纤传输后的可视化
    print("\n生成光纤传输后的可视化...")
    plot_dual_arm_heatmap(
        result.mps,
        save_path=str(output_dir / f"{run_tag}_3_after_fiber.png"),
        show_atomic=False,
        stage_name="After Fiber Channel",
        time_grid=result.time_grid,
        show=show_plots,
    )

    # 保存光纤传输后的调试信息
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After Fiber Channel',
        output_dir=output_dir,
        step_index=3,
        run_tag=run_tag,
    )

    # 诊断：检查BS前每个arm的光子分布
    print("\n诊断：检查BS前（含光纤）每个arm的光子分布...")
    n_bins = result.get_n_bins()
    total_A = 0.0
    total_B = 0.0
    pre_bs_arm_a = np.zeros(n_bins)
    pre_bs_arm_b = np.zeros(n_bins)
    for n in range(n_bins):
        site_A = 2 + 2 * n  # Arm A的bin n
        site_B = 2 + 2 * n + 1  # Arm B的bin n

        # 获取单个site的约化密度矩阵
        rho_A = result.mps.get_reduced_density([site_A])
        rho_B = result.mps.get_reduced_density([site_B])

        # 6D基: vac=0, H=1, V=2, 2H=3, 2V=4, HV=5
        photon_count = [0, 1, 1, 2, 2, 2]
        p_A = sum(rho_A[i, i].real * photon_count[i] for i in range(6))
        p_B = sum(rho_B[i, i].real * photon_count[i] for i in range(6))
        pre_bs_arm_a[n] = p_A
        pre_bs_arm_b[n] = p_B
        total_A += p_A
        total_B += p_B

        if p_A > 0.01 or p_B > 0.01:
            print(f"  bin {n}: Arm_A={p_A:.4f}, Arm_B={p_B:.4f}")

    print(f"  总计: Arm_A={total_A:.4f}, Arm_B={total_B:.4f}")


    # =========================================================================
    # 应用分束器（BS），使A_n与B_n在每个仓处干涉
    # =========================================================================
    print("\n应用分束器（BS）...")
    apply_bs(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )

    # 保存BS后的可视化
    print("\n生成BS后的可视化...")
    plot_dual_arm_heatmap(
        result.mps,
        save_path=str(output_dir / f"{run_tag}_4_after_bs.png"),
        show_atomic=False,
        stage_name="After Beam Splitter",
        time_grid=result.time_grid,
        show=show_plots,
    )

    print("\n生成BS后的跨bin联合分布热图...")
    joint_after_bs = plot_cross_bin_joint_heatmap(
        result.mps,
        n_bins=result.get_n_bins(),
        save_path=str(output_dir / f"{run_tag}_4b_after_bs_cross_bin_joint.png"),
        arm_pair=("A", "B"),
        normalize=False,
        show=show_plots,
    )

    # 保存BS后的调试信息
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After BS',
        output_dir=output_dir,
        step_index=4,
        run_tag=run_tag,
    )

    extra_data_path = output_dir / f"{run_tag}_extra_data.txt"
    _write_extra_data(
        extra_data_path,
        fiber_sample=fiber_sample,
        pre_bs_arm_a=pre_bs_arm_a,
        pre_bs_arm_b=pre_bs_arm_b,
        cross_bin_joint=joint_after_bs,
    )
    print(f"  Extra data saved: {extra_data_path.name}")

    # =========================================================================
    # 【深入分析After BS】检查双光子态的分布和BS门的作用
    # =========================================================================
    print("\n" + "="*80)
    print("【深入分析After BS】检查双光子态的分布")
    print("="*80)

    n_bins = result.get_n_bins()

    # 统计全局光子数
    total_photons_global = 0.0
    total_two_photon_states = 0.0  # 双光子态总概率

    # 1517子空间基：vac=0, H=1, V=2, 2H=3, 2V=4, HV=5
    state_names = ['vac', 'H', 'V', '2H', '2V', 'HV']

    print("\n【逐bin分析】")
    for n in range(n_bins):
        site_A = 2 + 2 * n
        site_B = 2 + 2 * n + 1
        rho_AB = result.mps.get_reduced_density([site_A, site_B])

        # 计算这个bin的总光子数
        bin_photons = 0.0
        bin_two_photon = 0.0

        # 遍历所有36个基态
        for i_A in range(6):
            for i_B in range(6):
                prob = rho_AB[i_A, i_B, i_A, i_B].real

                # 计算光子数
                n_A = 0 if i_A == 0 else (1 if i_A in [1, 2] else 2)
                n_B = 0 if i_B == 0 else (1 if i_B in [1, 2] else 2)

                bin_photons += prob * (n_A + n_B)

                # 统计双光子态
                if n_A + n_B == 2:
                    bin_two_photon += prob

        total_photons_global += bin_photons
        total_two_photon_states += bin_two_photon

        # 只打印有意义的bin
        if bin_photons > 0.01:
            print(f"\nBin {n}:")
            print(f"  总光子数: {bin_photons:.6f}")
            print(f"  双光子态概率: {bin_two_photon:.6f}")

            # 打印主要的态分量
            print(f"  主要态分量:")
            for i_A in range(6):
                for i_B in range(6):
                    prob = rho_AB[i_A, i_B, i_A, i_B].real
                    if prob > 0.001:
                        print(f"    |{state_names[i_A]},{state_names[i_B]}>: {prob:.6f}")

    print(f"\n【全局统计】")
    print(f"  总光子数（所有bin）: {total_photons_global:.6f}")
    print(f"  双光子态总概率: {total_two_photon_states:.6f}")
    print(f"  非双光子态概率: {1.0 - total_two_photon_states:.6f}")

    # 检查是否有多光子态或单光子态
    if total_two_photon_states < 0.95:
        print(f"警告：双光子态概率 < 95%，存在单光子或多光子分量")

    print("="*80)

    # 诊断：检查BS后每个bin的两端口关联
    print("\n诊断：检查BS后的两端口关联...")
    n_bins = result.get_n_bins()
    for n in range(n_bins):
        site_A = 2 + 2 * n
        site_B = 2 + 2 * n + 1
        rho_AB = result.mps.get_reduced_density([site_A, site_B])

        # 检查各种双光子态的概率
        # 6D基: vac=0, H=1, V=2, 2H=3, 2V=4, HV=5
        p_vac_vac = rho_AB[0, 0, 0, 0].real
        p_H_vac = rho_AB[1, 0, 1, 0].real
        p_V_vac = rho_AB[2, 0, 2, 0].real
        p_vac_H = rho_AB[0, 1, 0, 1].real
        p_vac_V = rho_AB[0, 2, 0, 2].real
        p_H_H = rho_AB[1, 1, 1, 1].real
        p_V_V = rho_AB[2, 2, 2, 2].real
        p_H_V = rho_AB[1, 2, 1, 2].real
        p_V_H = rho_AB[2, 1, 2, 1].real
        p_2H_vac = rho_AB[3, 0, 3, 0].real
        p_2V_vac = rho_AB[4, 0, 4, 0].real
        p_HV_vac = rho_AB[5, 0, 5, 0].real
        p_vac_2H = rho_AB[0, 3, 0, 3].real
        p_vac_2V = rho_AB[0, 4, 0, 4].real
        p_vac_HV = rho_AB[0, 5, 0, 5].real

        # 只打印有意义的bin
        total_nonvac = 1 - p_vac_vac
        if total_nonvac > 0.01:
            print(f"  bin {n}: P(non-vac)={total_nonvac:.4f}")
            if p_H_V + p_V_H > 1e-6:
                print(f"    BSM成功态: P(H,V)={p_H_V:.6f}, P(V,H)={p_V_H:.6f}")
            if p_H_H + p_V_V > 1e-6:
                print(f"    同极化: P(H,H)={p_H_H:.6f}, P(V,V)={p_V_V:.6f}")
            if p_2H_vac + p_2V_vac + p_HV_vac > 1e-6:
                print(f"    bunching port1: P(2H,0)={p_2H_vac:.6f}, P(2V,0)={p_2V_vac:.6f}, P(HV,0)={p_HV_vac:.6f}")
            if p_vac_2H + p_vac_2V + p_vac_HV > 1e-6:
                print(f"    bunching port2: P(0,2H)={p_vac_2H:.6f}, P(0,2V)={p_vac_2V:.6f}, P(0,HV)={p_vac_HV:.6f}")

    # 计算归一化前的光子统计
    print("\n计算BS后的光子统计...")
    photon_stats = compute_photon_statistics(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )

    # =========================================================================
    # 探测
    # =========================================================================
    # 探测参数
    eta_det = 1.0
    p_dark = 0.0

    # 诊断：检查每个bin的光子分布
    print("\n诊断：检查每个bin的光子分布...")
    n_bins = result.get_n_bins()

    # 先检查第一个bin的rho形状
    site_A = 2
    site_B = 3
    rho_AB = result.mps.get_reduced_density([site_A, site_B])
    print(f"  rho_AB shape: {rho_AB.shape}")
    print("\n枚举成功事件（无暗计数）...")
    enum_no_dark = enumerate_success_events(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        eta_det=eta_det,
        p_dark=0.0,
        verbose=True,
    )
    if p_dark > 0.0:
        print("\n枚举成功事件（含暗计数）...")
        enum_with_dark = enumerate_success_events(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            p_dark=p_dark,
            verbose=True,
        )
    else:
        enum_with_dark = enum_no_dark

    success_metrics = {
        "eta_det": eta_det,
        "p_dark": p_dark,
        "p_arrive": enum_with_dark.p_arrive,
        "p_success_all": enum_with_dark.p_success,
        "p_success_true": enum_with_dark.p_success_true,
        "p_success_false": enum_with_dark.p_success_false,
        "p_success_given_arrival": enum_with_dark.p_success_given_arrival,
        "fidelity_all": enum_with_dark.fidelity_declared,
        "fidelity_true": enum_with_dark.fidelity_true,
        "fidelity_false": enum_with_dark.fidelity_false,
        "p_success_no_dark": enum_no_dark.p_success,
        "fidelity_no_dark": enum_no_dark.fidelity_declared,
    }
    success_metrics["p_false_approx"] = max(
        0.0, success_metrics["p_success_all"] - success_metrics["p_success_no_dark"]
    )
    success_metrics["false_fraction"] = (
        success_metrics["p_success_false"] / success_metrics["p_success_all"]
        if success_metrics["p_success_all"] > 0
        else 0.0
    )
    success_metrics["false_fraction_approx"] = (
        success_metrics["p_false_approx"] / success_metrics["p_success_all"]
        if success_metrics["p_success_all"] > 0
        else 0.0
    )

    success_path = _write_success_metrics_detail(output_dir, run_tag, success_metrics)
    print(f"  Success metrics saved: {success_path.name}")

    # 使用逐bin Kraus测量方法运行探测和BSM（可多次采样）
    print("\n运行探测和BSM（逐bin Kraus测量）...")
    run_stats = _init_stats()
    with open(run_clicks_path, 'w', encoding='utf-8') as file:
        file.write("run\tshot\tsuccess\tbell\tclick_count\tevents\n")
    for shot_index in range(1, shots_per_run + 1):
        print(f"\n[shot {shot_index}/{shots_per_run}]")
        det_result = run_two_photon_detection(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            p_dark=p_dark,
            #rng=np.random.default_rng(seed=19),
            rng=np.random.default_rng(),
            verbose=True,
        )

        # 打印结果
        if det_result.success:
            print(f"\n  BSM成功!")
            print(f"  宣告的Bell态: {det_result.bell_state}")
            print(f"  点击: {[(c.detector, c.bin_index) for c in det_result.clicks]}")

            # 计算与期望Bell态的保真度
            fidelity = compute_fidelity_with_bell(det_result.spin_state, det_result.bell_state)
            print(f"  与|{det_result.bell_state}>的保真度: {fidelity:.4f}")

            # 计算与所有Bell态的保真度以供参考
            print(f"\n  与所有Bell态的保真度:")
            for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
                f = compute_fidelity_with_bell(det_result.spin_state, bell)
                marker = " <-- 宣告的" if bell == det_result.bell_state else ""
                print(f"    F(|{bell}>): {f:.4f}{marker}")

            # 打印自旋态
            print(f"\n  自旋密度矩阵（量子比特子空间）:")
            rho = det_result.spin_state
            print(f"    Tr(rho) = {np.trace(rho).real:.4f}")
            print(f"    纯度 = {np.trace(rho @ rho).real:.4f}")
        else:
            print(f"\n  BSM失败 - 未找到成功模式")
            print(f"  点击数量: {len(det_result.clicks)}")
            if det_result.clicks:
                print(f"  点击: {[(c.detector, c.bin_index) for c in det_result.clicks]}")

        # 保存探测后的调试信息
        print("\n保存探测后调试信息...")
        if shots_per_run == 1:
            det_file = output_dir / f'{run_tag}_debug_detection_result.txt'
        else:
            det_file = output_dir / f'{run_tag}_shot{shot_index:03d}_debug_detection_result.txt'
        with open(det_file, 'w', encoding='utf-8') as file:
            file.write(f'探测结果\n')
            file.write('='*60 + '\n\n')
            file.write(f'成功: {det_result.success}\n')
            file.write(f'Bell态: {det_result.bell_state}\n')
            file.write(f'点击次数: {len(det_result.clicks)}\n')
            if det_result.clicks:
                file.write(f'点击详情: {[(c.detector, c.bin_index) for c in det_result.clicks]}\n')

                file.write(f'\n自旋密度矩阵:\n')
                rho = det_result.spin_state
                file.write(f'  基: |00>, |01>, |10>, |11>\n')
                for i in range(4):
                    for j in range(4):
                        val = rho[i, j]
                        if abs(val) > 1e-10:
                            file.write(f'  rho[{i},{j}] = {val:.4f}\n')

                file.write(f'\n纯度: {np.trace(rho @ rho).real:.4f}\n')

                file.write(f'\nBell态保真度:\n')
                for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
                    fid = compute_fidelity_with_bell(rho, bell)
                    marker = " <-- 探测到的" if bell == det_result.bell_state else ""
                    file.write(f'  F({bell}) = {fid:.4f}{marker}\n')

        print(f"  调试信息已保存: {det_file.name}")

        run_stats["shots"] += 1
        run_stats["clicks"][len(det_result.clicks)] += 1
        if det_result.success:
            run_stats["success"] += 1
            if det_result.bell_state:
                run_stats["bell"][det_result.bell_state] += 1

        _append_click_summary(run_clicks_path, run_index, shot_index, det_result)

    print(f"\n完成! 文件已保存至: {output_dir}/")
    return run_stats, success_metrics


def _run_single_simulation(
    output_dir: Path,
    run_index: int,
    n_runs: int,
    run_clicks_path: Path,
    shots_per_run: int,
    log_path: Optional[Path] = None,
    mirror_console: bool = True,
    show_plots: bool = True,
):
    run_tag = f"run{run_index:03d}"
    if log_path is None:
        print(f"正在处理: {run_tag} | clicks: {run_clicks_path.name}")
        return _run_single_simulation_core(
            output_dir,
            run_index,
            n_runs,
            run_clicks_path,
            shots_per_run,
            show_plots,
        )
    with open(log_path, 'w', encoding='utf-8') as log_file:
        if mirror_console:
            tee_out = Tee(sys.stdout, log_file)
            tee_err = Tee(sys.stderr, log_file)
        else:
            tee_out = Tee(log_file)
            tee_err = Tee(log_file)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_out, tee_err
        try:
            print(f"正在处理: {run_tag} | log: {log_path.name} | clicks: {run_clicks_path.name}")
            return _run_single_simulation_core(
                output_dir,
                run_index,
                n_runs,
                run_clicks_path,
                shots_per_run,
                show_plots,
            )
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _run_single_simulation_task(args):
    output_dir, run_index, n_runs, shots_per_run, mirror_console, show_plots = args
    run_tag = f"run{run_index:03d}"
    log_path = output_dir / f"{run_tag}_console.log"
    run_clicks_path = output_dir / f"{run_tag}_clicks.txt"
    run_stats, success_metrics = _run_single_simulation(
        output_dir,
        run_index,
        n_runs,
        run_clicks_path,
        shots_per_run,
        log_path=log_path,
        mirror_console=mirror_console,
        show_plots=show_plots,
    )
    return run_index, run_clicks_path, run_stats, success_metrics


def main():
    """主函数：运行发射 + QFC + 分束器 + 探测仿真。"""
    n_runs, shots_per_run, jobs = _parse_run_params(sys.argv)

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统一的点击结果汇总文件
    clicks_summary_path = output_dir / "all_clicks_summary.txt"
    with open(clicks_summary_path, 'w', encoding='utf-8') as file:
        file.write("run\tshot\tsuccess\tbell\tclick_count\tevents\n")

    runs_summary_path = output_dir / "runs_summary.txt"
    with open(runs_summary_path, 'w', encoding='utf-8') as file:
        file.write("run\tshots\tsuccess\tsuccess_rate\tbell_counts\tclick_count_dist\n")

    success_summary_path = output_dir / "success_metrics_summary.txt"
    with open(success_summary_path, 'w', encoding='utf-8') as file:
        file.write(
            "run\tp_arrive\tp_success_all\tp_success_true\tp_success_false\t"
            "p_success_given_arrival\tfidelity_all\tfidelity_true\tfidelity_false\t"
            "p_success_no_dark\tfidelity_no_dark\tp_false_approx\tfalse_fraction\tfalse_fraction_approx\n"
        )

    print(f"Output directory: {output_dir}")
    print(f"将运行 {n_runs} 次仿真，每次 {shots_per_run} 次探测采样...")
    jobs = max(1, min(jobs, n_runs, os.cpu_count() or 1))
    print(f"并行进程数: {jobs}")

    if jobs > 1:
        focus_run = int(np.random.default_rng().integers(1, n_runs + 1))
        print(f"并行模式: 仅显示 run{focus_run:03d} 的实时输出（若非无屏幕环境将显示图像）")
        run_order = [focus_run] + [i for i in range(1, n_runs + 1) if i != focus_run]
    else:
        focus_run = 1
        run_order = list(range(1, n_runs + 1))

    overall_stats = _init_stats()
    overall_success = _init_success_metrics_accumulator()

    results = {}
    if jobs == 1:
        for run_index in range(1, n_runs + 1):
            run_index, run_clicks_path, run_stats, success_metrics = _run_single_simulation_task(
                (output_dir, run_index, n_runs, shots_per_run, True, True)
            )
            results[run_index] = (run_clicks_path, run_stats, success_metrics)
    else:
        tasks = []
        for run_index in run_order:
            mirror_console = run_index == focus_run
            show_plots = mirror_console
            tasks.append((output_dir, run_index, n_runs, shots_per_run, mirror_console, show_plots))
        print("已提交全部并行任务，非前台 run 的输出写入各自日志。")
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_map = {executor.submit(_run_single_simulation_task, task): task[1] for task in tasks}
            for future in as_completed(future_map):
                run_index, run_clicks_path, run_stats, success_metrics = future.result()
                results[run_index] = (run_clicks_path, run_stats, success_metrics)
                print(f"[完成] run{run_index:03d}", flush=True)

    for run_index in sorted(results.keys()):
        run_clicks_path, run_stats, success_metrics = results[run_index]
        if run_stats is not None:
            _append_run_summary(runs_summary_path, run_index, run_stats)
            _merge_stats(overall_stats, run_stats)
        if success_metrics is not None:
            _append_success_metrics_summary(success_summary_path, run_index, success_metrics)
            _accumulate_success_metrics(overall_success, success_metrics)
        if run_clicks_path is not None:
            _append_clicks_file(clicks_summary_path, run_clicks_path)

    _append_run_summary(runs_summary_path, "TOTAL", overall_stats)
    _write_overall_summary(
        output_dir / "overall_summary.txt",
        overall_stats,
        n_runs,
        shots_per_run,
    )
    total_success_metrics = _finalize_success_metrics(overall_success)
    _append_success_metrics_summary(success_summary_path, "TOTAL", total_success_metrics)

if __name__ == "__main__":
    main()
