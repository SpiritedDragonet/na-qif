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
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Optional, Tuple

# Add project root to path (for running as standalone script)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from atom_sim.config import TimeGrid, EmitParams
from atom_sim.simulation import (
    run_emission_only, EmissionResult, apply_qfc, apply_780_filter, apply_fiber_channel,
    apply_bs, project_to_1517, postselect_two_photon,
    # 探测
    run_two_photon_detection, compute_fidelity_with_bell, compute_photon_statistics,
)
from atom_sim.visualization import plot_dual_arm_heatmap, plot_dual_arm_heatmap_phase
from atom_sim.physics import FiberChannelParams


def run_dual_atom_emission(
    n_bins: int = 200,
    dt_ns: float = 0.2,
    chi_max: int = 50,
    Alpha_A: Optional[np.ndarray] = None,
    Alpha_B: Optional[np.ndarray] = None,
    gamma_peak_A: float = 0.2,
    gamma_peak_B: float = 0.2,
    t0_A: Optional[float] = None,
    t0_B: Optional[float] = None,
    sigma: float = 12.0,
    delay_bins_B: int = 0,
    verbose: bool = True,
) -> EmissionResult:
    """
    运行双原子发射仿真。

    原子向左移动，依次与每个时间仓相互作用。

    Parameters
    ----------
    n_bins : int
        时间仓数量
    dt_ns : float
        时间步长（纳秒）
    chi_max : int
        MPS最大键维度
    Alpha_A : np.ndarray, optional
        原子A的2x2偏振矩阵
    Alpha_B : np.ndarray, optional
        原子B的2x2偏振矩阵
    gamma_peak_A : float
        原子A的峰值发射率
    gamma_peak_B : float
        原子B的峰值发射率
    t0_A : float, optional
        原子A的峰值时间（纳秒）
    t0_B : float, optional
        原子B的峰值时间（纳秒）
    sigma : float
        高斯发射轮廓的宽度参数（纳秒）
    delay_bins_B : int
        原子B发射延迟的仓数（负数表示B晚于A）
        例如-10表示A先开始发射，10个仓后B才开始
    verbose : bool
        是否打印进度信息

    Returns
    -------
    EmissionResult
        仿真结果容器
    """
    # 创建时间网格
    time_grid = TimeGrid(dt=dt_ns * 1e-9, N=n_bins)  # 将ns转换为秒
    t = time_grid.t

    # 设置默认峰值为时间窗口中心
    if t0_A is None:
        t0_A = n_bins * dt_ns / 2
    if t0_B is None:
        t0_B = n_bins * dt_ns / 2

    # 创建高斯发射率函数
    def gamma_A_func(t_sec):
        t_ns = t_sec * 1e9  # 转换为ns进行计算
        return gamma_peak_A * np.exp(-0.5 * ((t_ns - t0_A) / sigma) ** 2)

    def gamma_B_func(t_sec):
        t_ns = t_sec * 1e9
        return gamma_peak_B * np.exp(-0.5 * ((t_ns - t0_B) / sigma) ** 2)

    # 设置默认Alpha矩阵（单位矩阵=无偏振混合）
    if Alpha_A is None:
        Alpha_A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    if Alpha_B is None:
        Alpha_B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

    # 创建发射参数
    emit_params = EmitParams(
        gamma_A=gamma_A_func,
        gamma_B=gamma_B_func,
        Alpha_A=Alpha_A,
        Alpha_B=Alpha_B,
    )

    # 使用仿真层运行仿真
    result = run_emission_only(
        time_grid=time_grid,
        emit_params=emit_params,
        chi_max=chi_max,
        verbose=verbose,
        delay_bins_B=delay_bins_B,
    )

    return result


def save_debug_info(
    mps,
    n_bins: int,
    stage: str,
    output_dir: Path,
    step_index: int,
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
    info_file = output_dir / f'debug_step_{step_index:02d}_{stage.replace(" ", "_").lower()}.txt'
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
        f.write(f'  损耗概率 = {stats["loss_prob"]:.4f}\n\n')
        f.write(f'原子态信息:\n')
        f.write(f'  对角元: {info["spin_state_diag"]}\n')
        f.write(f'  纯度: {info["spin_purity"]:.4f}\n\n')
        f.write(f'Bell态保真度:\n')
        f.write(f'  Psi+ = {info["fidelity_Psip"]:.4f}\n')
        f.write(f'  Psi- = {info["fidelity_Psim"]:.4f}\n')
        f.write(f'  Phi+ = {info["fidelity_Phip"]:.4f}\n')
        f.write(f'  Phi- = {info["fidelity_Phim"]:.4f}\n')

    print(f'  调试信息已保存: {info_file.name}')


def main():
    """主函数：运行发射 + QFC + 分束器 + 探测仿真。"""
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("运行发射 + QFC + 分束器 + 探测仿真...")

    # 运行发射
    # 使用合理的物理参数
    result = run_dual_atom_emission(
        n_bins=30,  # 仓数
        dt_ns=0.2,   # 时间步长
        chi_max=50,
        gamma_peak_A=0.5,  # 发射率
        gamma_peak_B=0.5,
        sigma=10.0,  # 波包宽度（纳秒）
        delay_bins_B=0,  # 无延迟（测试）
        verbose=True,
    )

    # 保存发射后的可视化
    print("\n生成发射后的可视化图...")
    plot_dual_arm_heatmap(
        result,
        save_path=str(output_dir / "1_after_emission.png"),
        show_atomic=True,
        stage_name="After Emission"
    )

    # 保存调试信息
    print("\n保存调试信息...")
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After Emission',
        output_dir=output_dir,
        step_index=1,
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
        save_path=str(output_dir / "2_after_qfc.png"),
        show_atomic=False,
        stage_name="After QFC + 780nm Filter",
        time_grid=result.time_grid,
    )

    # 保存调试信息
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After QFC + Filter',
        output_dir=output_dir,
        step_index=2,
    )

    # 诊断：检查BS前每个arm的光子分布
    print("\n诊断：检查BS前每个arm的光子分布...")
    n_bins = result.get_n_bins()
    total_A = 0.0
    total_B = 0.0
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
        save_path=str(output_dir / "3_after_bs.png"),
        show_atomic=False,
        stage_name="After Beam Splitter",
        time_grid=result.time_grid,
    )

    # 保存BS后的调试信息
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After BS',
        output_dir=output_dir,
        step_index=3,
    )

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

    # 投影到两光子子空间（post-selection）
    # 这会剔除"至少一路光子被吸收"的分量
    print("\n投影到两光子子空间...")
    result.mps, two_photon_prob = postselect_two_photon(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )
    print(f"  两光子到达概率: {two_photon_prob:.6f}")
    print(f"  已更新MPS为条件态（只含两光子到达的分量）")

    # 保存归一化后的可视化
    print("\n生成归一化后的可视化...")
    plot_dual_arm_heatmap(
        result.mps,
        save_path=str(output_dir / "4_after_normalization.png"),
        show_atomic=False,
        stage_name="After Normalization (Two-Photon Branch)",
        time_grid=result.time_grid,
    )

    # 保存归一化后的调试信息
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After Normalization',
        output_dir=output_dir,
        step_index=4,
    )

    # 验证归一化后的光子统计
    photon_stats_norm = compute_photon_statistics(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )

    # =========================================================================
    # 探测
    # =========================================================================
    # 探测参数
    eta_det = 1

    # 诊断：检查每个bin的光子分布
    print("\n诊断：检查每个bin的光子分布...")
    n_bins = result.get_n_bins()

    # 先检查第一个bin的rho形状
    site_A = 2
    site_B = 3
    rho_AB = result.mps.get_reduced_density([site_A, site_B])
    print(f"  rho_AB shape: {rho_AB.shape}")

    # 统计总光子数和BSM相关态
    total_photons_all = 0.0
    bsm_success_prob_psi_minus = 0.0  # Ψ⁻ 成功概率
    bsm_success_prob_psi_plus = 0.0   # Ψ⁺ 成功概率

    print("\n【BSM成功率精确计算】")
    print("="*70)
    print("考虑所有可能的两光子态对BSM成功的贡献")
    print("="*70)

    # 6D基态索引
    VAC, H, V, HH, VV, HV = 0, 1, 2, 3, 4, 5
    state_names = ['vac', 'H', 'V', '2H', '2V', 'HV']

    # =========================================================================
    # 第一部分：同bin两光子态的贡献
    # =========================================================================
    print("\n【第一部分】同bin两光子态的贡献")
    print("-"*70)

    same_bin_contribution = 0.0

    for n in range(n_bins):
        site_A = 2 + 2 * n  # Port 1
        site_B = 2 + 2 * n + 1  # Port 2

        # 获取两个site的约化密度矩阵
        rho_AB = result.mps.get_reduced_density([site_A, site_B])

        # 计算光子数
        photon_count = [0, 1, 1, 2, 2, 2]
        total_A = 0.0
        total_B = 0.0
        for i in range(6):
            for j in range(6):
                p = rho_AB[i, j, i, j].real
                total_A += p * photon_count[i]
                total_B += p * photon_count[j]
        total_photons_all += total_A + total_B

        # 分析BSM成功态
        # Ψ⁻ 模式: {H1, V2} 或 {V1, H2}
        p_H1V2 = rho_AB[H, V, H, V].real  # |H⟩_port1 |V⟩_port2
        p_V1H2 = rho_AB[V, H, V, H].real  # |V⟩_port1 |H⟩_port2

        # Ψ⁺ 模式: {H1, V1} 或 {H2, V2}
        p_H1V1 = rho_AB[HV, VAC, HV, VAC].real  # |HV⟩_port1 |vac⟩_port2
        p_H2V2 = rho_AB[VAC, HV, VAC, HV].real  # |vac⟩_port1 |HV⟩_port2

        bin_psi_minus = p_H1V2 + p_V1H2
        bin_psi_plus = p_H1V1 + p_H2V2
        bin_total = bin_psi_minus + bin_psi_plus

        bsm_success_prob_psi_minus += bin_psi_minus
        bsm_success_prob_psi_plus += bin_psi_plus
        same_bin_contribution += bin_total

        if bin_total > 1e-6:
            print(f"  bin {n:2d}: Ψ⁻={bin_psi_minus:.6f}, Ψ⁺={bin_psi_plus:.6f}, "
                  f"total={bin_total:.6f}")

            # 打印主要态分量（用于调试）
            if total_A > 0.01 or total_B > 0.01:
                print(f"    光子分布: port1={total_A:.4f}, port2={total_B:.4f}")
                for i in range(6):
                    for j in range(6):
                        p = rho_AB[i, j, i, j].real
                        if p > 0.001:
                            print(f"      |{state_names[i]},{state_names[j]}>: {p:.6f}")

    print(f"\n  同bin总贡献: {same_bin_contribution:.6f}")

    # 使用逐bin Kraus测量方法运行探测和BSM
    print("\n运行探测和BSM（逐bin Kraus测量）...")
    det_result = run_two_photon_detection(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        eta_det=eta_det,
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
    with open(output_dir / 'debug_detection_result.txt', 'w', encoding='utf-8') as file:
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

    print(f"  调试信息已保存: debug_detection_result.txt")

    print(f"\n完成! 文件已保存至: {output_dir}/")


if __name__ == "__main__":
    main()
