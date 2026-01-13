# -*- coding: utf-8 -*-
"""
演示：原子发射 → 时间仓波包仿真

本演示模拟完整过程：
1. 原子从激发态 |e> 开始
2. 对每个时间仓 n：
   - 发射门将原子耦合到780nm光子
   - QFC 将 780nm → 1517nm 转换
3. 结果：具有特定形状的时间仓波包

物理原理：
---------
原��（三能级）：
  |e>: 激发态 (5P_{3/2}, F'=0, m_F=0)
  |0>: 基态 (5S_{1/2}, F=1, m_F=+1)
  |1>: 基态 (5S_{1/2}, F=1, m_F=-1)

选择定则：
  |e> → |0>: sigma+ 光子（H 偏振）
  |e> → |1>: sigma- 光子（V 偏振）

780nm 子空间 (3D): |vac>, |H>, |V>
1517nm 子空间 (6D): |vac>, |H>, |V>, |2H>, |2V>, |HV>
仓空间: 780(3D) × 1517(6D) = 18D
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 将父目录添加到路径

from atom_sim.core.mps import MPSState
from atom_sim.config import TimeGrid, EmitParams, QFCParams
from atom_sim.physics.gates import emission_gate, qfc_gate
from atom_sim.visualization import (
    plot_intensity_envelope,
    plot_single_photon_prob,
    plot_wavepacket,
    extract_wavepacket,
)


def gaussian_pulse(t: np.ndarray, t0: float, sigma: float) -> np.ndarray:
    """
    高斯发射率分布。

    Parameters
    ----------
    t : np.ndarray
        时间点
    t0 : float
        峰值时间
    sigma : float
        宽度（标准差）

    Returns
    -------
    np.ndarray
        每个时间点的 Gamma 值
    """
    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def simulate_emission_wavepacket(
    n_bins: int = 200,
    dt: float = 0.1,
    t0: float = 10.0,
    sigma: float = 3.0,
    theta_H: float = np.pi / 4,  # H 偏振的 QFC 角度
    theta_V: float = np.pi / 4,  # V 偏振的 QFC 角度
    alpha_H_plus: float = 1.0 / np.sqrt(2),
    alpha_H_minus: float = 1.0 / np.sqrt(2),
    alpha_V_plus: float = 1.0 / np.sqrt(2),
    alpha_V_minus: float = -1.0 / np.sqrt(2),
    chi_max: int = 50,
):
    """
    模拟发射到时间仓波包。

    链布局: atom_A - atom_B - A1 - B1 - A2 - B2 - ... - AN - BN
    为简单起见，我们只模拟一个臂（A）和一个原子。

    Parameters
    ----------
    n_bins : int
        时间仓数量
    dt : float
        时间仓宽度
    t0 : float
        发射脉冲的峰值时间
    sigma : float
        脉冲宽度
    theta_H, theta_V : float
        QFC 转换角度
    alpha_* : float
        偏振映射系数
    chi_max : int
        最大键维度

    Returns
    -------
    MPSState
        所有发射后的最终 MPS 状态
    """
    print("=" * 70)
    print("模拟原子发射 → 时间仓波包")
    print("=" * 70)

    # 创建时间网格
    time_grid = TimeGrid(dt=dt, N=n_bins)
    t = time_grid.t

    # 创建高斯脉冲分布
    gamma_values = gaussian_pulse(t, t0, sigma)

    print(f"\n时间网格:")
    print(f"  N_bins = {n_bins}")
    print(f"  dt = {dt}")
    print(f"  总时间 = {n_bins * dt}")
    print(f"  t 范围: [{t[0]:.1f}, {t[-1]:.1f}]")

    print(f"\n发射脉冲分布:")
    print(f"  峰值在 t0 = {t0}")
    print(f"  宽度 sigma = {sigma}")
    print(f"  峰值 gamma = {gamma_values.max():.4f}")

    # 设置偏振矩阵
    Alpha = np.array([
        [alpha_H_plus, alpha_H_minus],
        [alpha_V_plus, alpha_V_minus]
    ], dtype=complex)

    print(f"\n偏振映射（Alpha 矩阵）:")
    print(f"  [[{alpha_H_plus:.3f}, {alpha_H_minus:.3f}],")
    print(f"   [{alpha_V_plus:.3f}, {alpha_V_minus:.3f}]]")

    # 创建发射参数
    emit_params = EmitParams(
        gamma_A=gamma_values,  # 时间相关
        Alpha_A=Alpha,
    )

    # QFC 参数
    qfc_params = QFCParams(theta_H=theta_H, theta_V=theta_V)

    # ========================================================================
    # 初始化 MPS: atom(3D) - A1(18D) - A2(18D) - ... - AN(18D)
    # ========================================================================

    print(f"\n初始化 MPS...")
    print(f"  链布局: atom - A1 - A2 - ... - AN")
    print(f"  原子维度: 3D (|0>, |1>, |e>)")
    print(f"  仓维度: 18D (780 x 1517)")

    local_dims = [3] + [18] * n_bins

    # 初始状态: 原子在激发态 |e>（索引 2）
    # 所有仓在真空态（每个 18D 仓空间的索引 0）
    init_state = [2] + [0] * n_bins

    mps = MPSState(local_dims=local_dims, init_state=init_state, max_bond=chi_max)

    print(f"  MPS 已创建: L={mps.L}, d={mps.d[:5]}...")
    print(f"  初始 chi = {mps.get_bond_dimensions()[:5]}...")

    # ========================================================================
    # 处理每个仓: 发射 + QFC
    # ========================================================================

    print(f"\n处理 {n_bins} 个时间仓...")

    # 预计算 QFC 门（所有仓相同）
    U_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)
    print(f"  QFC 门已计算（转换率 sin^2 theta = {np.sin(theta_H)**2:.3f}）")

    for n in range(n_bins):
        # Site 0 是原子，site 1+n 是仓 A_n
        tn = t[n]
        gamma_n = gamma_values[n]

        # 跳过发射率很小的仓
        if gamma_n < 1e-6:
            continue

        # 为该时间步创建发射门
        U_emit = emission_gate(
            gamma=gamma_n,
            dt=dt,
            Alpha=Alpha,
            which_atom='A'
        )

        # 应用发射门（原子到仓 n）
        # 注意：emission_gate 返回 27D（原子 x 780），需要嵌入到 18D 仓
        # 为简单起见，我们使用 27D 门但只应用到原子 + 780 部分
        mps.apply_bond_op(0, U_emit)

        # 应用 QFC（仓内的 780 → 1517 转换）
        mps.apply_one_site_gate(1 + n, U_qfc)

        if (n + 1) % 50 == 0 or n == 0 or n == n_bins - 1:
            chi = mps.get_bond_dimensions()
            print(f"  仓 {n+1:3d}/{n_bins}: gamma={gamma_n:.4f}, chi={chi[1]}")

    print(f"\n发射完成!")
    print(f"最终键维度: {mps.get_bond_dimensions()}")
    print(f"最终态归一化: {mps.norm():.6f}")

    return mps, time_grid


def visualize_results(mps: MPSState, time_grid: TimeGrid):
    """
    使用可视化模块可视化波包。

    Parameters
    ----------
    mps : MPSState
        最终 MPS 状态
    time_grid : TimeGrid
        时间网格
    """
    print("\n" + "=" * 70)
    print("可视化波包")
    print("=" * 70)

    n_bins = time_grid.N

    # 创建带子图的图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========================================================================
    # 图 1: 单光子概率（归一化）
    # ========================================================================

    ax = axes[0, 0]
    data_A, _ = extract_single_photon_prob(mps, n_bins, polarized=False)

    # 归一化以显示形状
    data_A_norm = data_A / (data_A.sum() + 1e-15)

    x = time_grid.t
    ax.plot(x, data_A_norm, '-', color='tab:blue', linewidth=2, label='单光子概率')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('概率（归一化）')
    ax.set_title('单光子波包（归一化）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # 图 2: 强度包络
    # ========================================================================

    ax = axes[0, 1]
    data_A, _ = extract_intensity_envelope(mps, n_bins, polarized=False)

    ax.plot(x, data_A, '-', color='tab:orange', linewidth=2, label='<N>')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('强度 / <N>')
    ax.set_title('强度包络')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # 图 3: 偏振分辨波包
    # ========================================================================

    ax = axes[1, 0]
    data_A, _ = extract_single_photon_prob(mps, n_bins, polarized=True)

    # 归一化
    total = data_A.sum(axis=0, keepdims=True) + 1e-15
    data_A_norm = data_A / total

    ax.plot(x, data_A_norm[:, 0], '--', label='H 偏振', color='tab:blue', alpha=0.7)
    ax.plot(x, data_A_norm[:, 1], '-', label='V 偏振', color='tab:red', alpha=0.7)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('概率')
    ax.set_title('偏振分辨波包')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # 图 4: 累积概率
    # ========================================================================

    ax = axes[1, 1]
    cumulative = np.cumsum(data_A_norm)
    ax.plot(x, cumulative, '-', color='tab:green', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('累积概率')
    ax.set_title('累积分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wavepacket_results.png', dpi=150)
    print("\n图像已保存到: wavepacket_results.png")

    return fig


def main():
    """运行完整的发射 → 波包仿真。"""

    # ========================================================================
    # 仿真参数
    # ========================================================================

    print("\n" + "=" * 70)
    print("演示：原子发射 → 时间仓波包")
    print("=" * 70)

    # 时间参数
    n_bins = 200
    dt = 0.1
    t0 = 10.0  # 仿真中期的峰值
    sigma = 3.0  # 脉冲宽度

    # QFC: 50% 转换率（sin^2 theta = 0.5）
    theta_H = np.arcsin(np.sqrt(0.5)) / 2  # 使 sin^2(theta) = 0.5
    theta_V = theta_H

    # 偏振: 圆偏振（sigma+/- 正交）
    # 这会在原子态和偏振之间产生纠缠
    alpha_H_plus = 1.0 / np.sqrt(2)
    alpha_H_minus = 1.0 / np.sqrt(2)
    alpha_V_plus = 1j / np.sqrt(2)  # +90 度相位
    alpha_V_minus = -1j / np.sqrt(2)

    print(f"\n仿真参数:")
    print(f"  n_bins = {n_bins}")
    print(f"  dt = {dt}")
    print(f"  脉冲: 高斯分布，t0={t0}, sigma={sigma}")
    print(f"  QFC: theta_H={theta_H:.3f}, theta_V={theta_V:.3f}")
    print(f"  偏振: 圆偏振（sigma+/-）")

    # ========================================================================
    # 运行仿真
    # ========================================================================

    mps, time_grid = simulate_emission_wavepacket(
        n_bins=n_bins,
        dt=dt,
        t0=t0,
        sigma=sigma,
        theta_H=theta_H,
        theta_V=theta_V,
        alpha_H_plus=alpha_H_plus,
        alpha_H_minus=alpha_H_minus,
        alpha_V_plus=alpha_V_plus,
        alpha_V_minus=alpha_V_minus,
        chi_max=50,
    )

    # ========================================================================
    # 可视化结果
    # ========================================================================

    fig = visualize_results(mps, time_grid)

    # ========================================================================
    # 汇总统计
    # ========================================================================

    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    # 提取波包数据
    data_A, _ = extract_single_photon_prob(mps, n_bins, polarized=False)

    total_prob = data_A.sum()
    print(f"\n总单光子概率: {total_prob:.6f}")
    print(f"  （应接近预期发射概率）")

    # 预期发射概率
    theta_peak = np.sqrt(dt * 1.0)  # gamma=1 时的峰值 theta
    expected_p = np.sin(theta_peak) ** 2
    print(f"\n预期单仓发射概率（峰值）: {expected_p:.6f}")

    # 峰值仓
    peak_idx = np.argmax(data_A)
    peak_prob = data_A[peak_idx]
    peak_time = time_grid.t[peak_idx]
    print(f"\n峰值仓: {peak_idx + 1}")
    print(f"峰值时间: {peak_time:.2f} s")
    print(f"峰值概率: {peak_prob:.6f}")

    # 宽度（FWHM 近似）
    threshold = peak_prob / 2
    above_threshold = data_A > threshold
    if np.any(above_threshold):
        fwhm = (above_threshold.sum()) * dt
        print(f"FWHM（近似）: {fwhm:.2f} s")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
