# -*- coding: utf-8 -*-
"""
测试：原子发射 → 时间仓波包（无 QFC，仅原子发射）

调试测试，验证发射仅在 780nm 子空间产生光子。
"""

import sys
from pathlib import Path
import numpy as np

# 将父目录添加到路径

from atom_sim.core.mps import MPSState
from atom_sim.config import TimeGrid, EmitParams, QFCParams
from atom_sim.physics.gates import emission_gate, qfc_gate, swap_gate
from atom_sim.visualization import telecom_ops_bin18, plot_bin_state_heatmap


def test_emission_wavepacket():
    """
    不使用 QFC 的发射测试，以验证物理过程正确。

    发射应仅在 780nm 子空间产生光子，
    对 1517nm 子空间的泄漏应为零（还没有光纤！）。
    """
    print("=" * 70)
    print("测试：原子发射 → 780nm 波包（无 QFC）")
    print("=" * 70)

    # Parameters
    n_bins = 200  # More bins for high resolution
    dt_ns = 0.2  # High time resolution (0.2ns per bin)
    chi_max = 50  # Higher bond dimension for more accurate simulation

    print(f"\n参数:")
    print(f"  n_bins = {n_bins}")
    print(f"  dt = {dt_ns} ns（高分辨率）")
    print(f"  总时间 = {n_bins * dt_ns:.1f} ns")
    print(f"  chi_max = {chi_max}")

    # 时间网格（纳秒）
    time_grid = TimeGrid(dt=dt_ns, N=n_bins)
    t = time_grid.t  # 时间（ns）

    # 高斯发射率分布（宽度 ~28ns FWHM）
    t0 = n_bins * dt_ns / 2  # 中心峰值（20ns）
    sigma = 12.0  # 宽度参数: FWHM ≈ 2.35*sigma ≈ 28ns
    gamma_peak = 0.2  # 峰值发射率（单步概率 ≈ 4%）

    # 创建时间相关的发射率
    gamma_values = gamma_peak * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    print(f"\n发射参数:")
    print(f"  高斯脉冲: t0={t0:.1f}ns, sigma={sigma:.1f}ns")
    print(f"  峰值 gamma = {gamma_peak}（单步概率 ≈ {gamma_peak * dt_ns:.3f}）")
    print(f"  预期 FWHM ≈ {2.35 * sigma:.1f} ns")

    # 打印某些仓的 gamma 值以供验证
    peak_idx = np.argmin(np.abs(t - t0))
    print(f"  峰值处 Gamma（t={t[peak_idx]:.1f}ns）: {gamma_values[peak_idx]:.3f}")

    # 简单的 H/V 映射（调试时不产生纠缠）
    Alpha = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    print(f"  Alpha = 简单（sigma+ -> H, sigma- -> V）")

    print(f"\n注意：不应用 QFC - 仅测试到 780nm 的原子发射")

    # ========================================================================
    # 初始化 MPS
    # ========================================================================

    print(f"\n初始化 MPS...")
    local_dims = [3] + [18] * n_bins  # 原子 + n_bins
    init_state = [2] + [0] * n_bins  # |e> + 真空

    mps = MPSState(local_dims=local_dims, init_state=init_state, max_bond=chi_max)

    print(f"  MPS: L={mps.L}, d={mps.d[:3]}... chi={mps.get_bond_dimensions()[:3]}...")

    # ========================================================================
    # 仓空间结构说明
    # ========================================================================

    print(f"\n仓空间结构（18D = 780(3D) x 1517(6D）:")
    print(f"  780 基矢: |vac>, |H>, |V>")
    print(f"  1517 基矢: |vac>, |H>, |V>, |2H>, |2V>, |HV>")
    print(f"  索引公式: index = i_780 * 6 + i_1517")
    print(f"    780=vac (i_780=0): 索引 0-5")
    print(f"    780=H   (i_780=1): 索引 6-11")
    print(f"    780=V   (i_780=2): 索引 12-17")

    # ========================================================================
    # SWAP传送带: 正确的time-bin发射模型
    # ========================================================================
    #
    # 根据专家建议，正确的time-bin发射模型需要：
    # - 第n步让原子与第n个bin耦合
    # - 使用SWAP让原子沿链移动，这样每个bin只被作用一次
    # - 这样光子会分布在各个bin上，而不是集中在一个bin反复振荡
    #
    # 关键公式（专家答复中的第7节）：
    # 对 n=1,2,...,N:
    #   1. 在(atom, bin_n)上施加发射门
    #   2. 记录该bin的占据概率 p_n = Tr[rho_bin_n, Pi_780]
    #   3. 施加SWAP，把原子与该bin交换位置
    #
    # 这样：
    #   - 每个bin只被作用一次，不会有"再吸收"
    #   - 光子幅度会分布在多个bins上
    #   - 柱状图全为非负，单调递增
    #
    # ========================================================================

    print(f"\n使用 SWAP 传送带处理 {n_bins} 个仓...")
    print(f"（每个仓只耦合一次，无再吸收）")

    # 存储每个bin的发射概率
    # per_bin_prob[n] = 第n个bin的780nm光子占据概率
    per_bin_prob_H = np.zeros(n_bins)
    per_bin_prob_V = np.zeros(n_bins)
    per_bin_prob_total = np.zeros(n_bins)

    # 追踪原子位置（初始在site 0）
    atom_position = 0  # atom当前在第几个位置

    # SWAP门（3D原子 × 18D bin）
    # 维度: d1=3, d2=18 -> dim=54
    U_swap = swap_gate(3, 18)

    # 对每个时间步
    for n in range(n_bins):
        # 当前原子位置是 atom_position，要耦合的bin是 atom_position + 1
        bin_idx = atom_position + 1

        # 获取这个时间步的发射率
        gamma_n = float(gamma_values[n])

        if gamma_n >= 1e-6:
            # 构造发射门（作用在atom和当前bin上）
            U_emit = emission_gate(
                gamma=gamma_n,
                dt=dt_ns,
                Alpha=Alpha,
                which_atom='A'
            )

            # 应用发射门到(atom, bin_idx)
            mps.apply_bond_op(atom_position, U_emit)

        # 记录当前bin的占据概率（之后这个bin不会再被触碰）
        # 注意：需要trace掉除了当前bin和原子的所有其他sites
        # 但由于只有这两个site有纠缠，我们可以直接取当前bin的约化密度矩阵
        rho_current_bin = mps.get_reduced_density([bin_idx])

        # 780nm子空间占据概率
        # bin空间: index = i_780 * 6 + i_1517
        # 780=H: indices 6-11, 780=V: indices 12-17
        p_H = rho_current_bin[6:12, 6:12].sum().real
        p_V = rho_current_bin[12:18, 12:18].sum().real

        per_bin_prob_H[n] = p_H
        per_bin_prob_V[n] = p_V
        per_bin_prob_total[n] = p_H + p_V

        # 如果还有更多bins需要处理，进行SWAP
        # SWAP把原子和当前bin交换位置
        if atom_position + 1 < len(mps.d) - 1:  # 确保不会移出链
            mps.swap_sites(atom_position)  # 交换atom_position和atom_position+1
            atom_position += 1  # 原子现在移动到了下一个位置

    print(f"  完成！")
    print(f"  最终原子位置: {atom_position}")
    print(f"  最终 chi: {mps.get_bond_dimensions()}")
    print(f"  归一化: {mps.norm():.6f}")

    # ========================================================================
    # 1517子空间泄漏检查
    # ========================================================================
    # 由于每个bin只被作用一次，没有"再吸收"，因此可以直接检查所有bins
    print(f"\n1517 子空间泄漏检查:")

    # 检查所有bins的1517占据（应该全为零）
    # 注意：经过SWAP传送带后，原子移到了最后，所以bins在sites 0 到 n_bins-1
    total_1517_prob = 0.0
    for i in range(n_bins):
        rho_bin = mps.get_reduced_density([i])  # SWAP后bins在最前面
        # 检查维度：如果这个site是3D（原子），跳过1517检查
        if rho_bin.shape[0] == 3:
            continue  # 这是原子位置，不是bin
        # 1517=H: index 1, 1517=V: index 2 (当780=vac时)
        # 但需要注意：由于780子空间是3D，1517子空间是6D
        # index = i_780 * 6 + i_1517
        # 1517=H在各个780子空间中的索引：
        #   当780=vac(i_780=0): index = 0*6 + 1 = 1
        #   当780=H(i_780=1): index = 1*6 + 1 = 7
        #   当780=V(i_780=2): index = 2*6 + 1 = 13
        # 1517=V在各个780子空间中的索引：2, 8, 14
        if rho_bin.shape[0] >= 14:  # 确保是18D bin空间
            p_1517_H = rho_bin[1, 1].real + rho_bin[7, 7].real + rho_bin[13, 13].real
            p_1517_V = rho_bin[2, 2].real + rho_bin[8, 8].real + rho_bin[14, 14].real
            total_1517_prob += p_1517_H + p_1517_V

    print(f"  所有仓的总 1517 概率: {total_1517_prob:.6e}")
    if total_1517_prob > 1e-10:
        print(f"  错误：检测到非零 1517 概率！")
    else:
        print(f"  OK：无 1517 子空间泄漏（符合预期）")

    # ========================================================================
    # Wave Packet Analysis (780nm subspace)
    # ========================================================================
    #
    # 关键物理理解：
    # - 使用SWAP传送带后，每个bin只被耦合一次
    # - per_bin_prob[n] 是第n个bin的**最终**780nm光子占据概率
    # - 这些概率不会随时间变化（没有再吸收）
    # - 波包形状由发射率轮廓 gamma(t) 和原子衰减共同决定
    #
    # ========================================================================

    print("\n" + "=" * 70)
    print("波包分析（780nm 子空间）")
    print("=" * 70)

    # 计算累积概率（前n个bins的总发射概率）
    cumulative_prob = np.cumsum(per_bin_prob_total)

    total_prob = cumulative_prob[-1]  # 最终总概率
    peak_idx = np.argmax(per_bin_prob_total)
    peak_time = t[peak_idx]
    peak_prob = per_bin_prob_total[peak_idx]

    print(f"\n780nm 单光子概率:")
    print(f"  总发射: {total_prob:.6f}")
    print(f"  峰值单仓概率: {peak_prob:.6f} 位于仓 {peak_idx + 1}（t={peak_time:.1f}ns）")

    # 打印gamma峰值附近的值
    gamma_peak_idx = np.argmax(gamma_values)
    print(f"\n  Gamma 峰值附近（仓 {gamma_peak_idx + 1}, t={t[gamma_peak_idx]:.1f}ns）:")
    for i in range(max(0, gamma_peak_idx - 2), min(n_bins, gamma_peak_idx + 3)):
        print(f"    仓 {i + 1}（t={t[i]:.1f}ns）: gamma={gamma_values[i]:.3f}, "
              f"per_bin={per_bin_prob_total[i]:.6f}")

    # 打印18-25ns区间的详细值
    print(f"\n  详细值（t=18ns 到 25ns）:")
    for i in range(n_bins):
        if 18 <= t[i] <= 25:
            print(f"    仓 {i + 1}（t={t[i]:.1f}ns）: gamma={gamma_values[i]:.6f}, "
                  f"per_bin={per_bin_prob_total[i]:.6f}, H={per_bin_prob_H[i]:.6f}, V={per_bin_prob_V[i]:.6f}")

    # ========================================================================
    # 可视化
    # ========================================================================

    print("\n绘制结果...")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 每个bin的发射概率（柱状图）
    # 这才是正确的"time-bin波包"表示！
    ax = axes[0]
    ax.bar(t - dt_ns/2, per_bin_prob_total, width=dt_ns, alpha=0.7, label='总计', color='purple')
    ax.bar(t - dt_ns/2, per_bin_prob_H, width=dt_ns, alpha=0.5, label='H 偏振', color='blue')
    ax.bar(t - dt_ns/2, per_bin_prob_V, width=dt_ns, alpha=0.5, label='V 偏振', color='red', bottom=per_bin_prob_H)
    # 叠加gamma轮廓用于对比
    ax2 = ax.twinx()
    ax2.plot(t, gamma_values, ':', color='gray', alpha=0.5, label='Gamma 轮廓')
    ax2.set_ylabel('Gamma（发射率）')
    ax.set_xlabel('时间')
    ax.set_ylabel('每仓概率')
    ax.set_title('780nm 每时间仓发射（SWAP 传送带）')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 右图: 累积发射概率
    ax = axes[1]
    ax.plot(t, cumulative_prob, '-', linewidth=2, label='累积', color='purple')
    ax.set_xlabel('时间')
    ax.set_ylabel('累积概率')
    ax.set_title('累积发射概率')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('test_emission_wavepacket.png', dpi=100)
    print("  已保存到: test_emission_wavepacket.png")

    # ========================================================================
    # 仓状态热图：18 个直积态 (780 × 1517)
    # ========================================================================
    # 780(3D) × 1517(6D) = 18D
    # 780态: |vac>, |H>, |V> (单光子)
    # 1517态: |vac>, |H>, |V>, |2H>, |2V>, |HV> (最多双光子)

    print("  生成仓状态热图...")

    import matplotlib as mpl
    mpl.rcParams['image.interpolation'] = 'nearest'  # 无抗锯齿

    # 使用更大的figsize并调整subplot参数以容纳y轴标签和双x轴
    fig, ax = plt.subplots(figsize=(18, 10))

    # 计算合适的vmax：让发射态（H,V）的变化可见
    # vac,vac态会接近饱和，但这是可以接受的
    # 使用 per_bin_prob_max 的1.2倍作为vmax
    vmax_scale = max(0.05, per_bin_prob_total.max() * 1.5)

    plot_bin_state_heatmap(
        mps,
        arm='A',
        n_bins=n_bins,
        time_grid=time_grid,
        group_by='780',  # 按780态分组
        vmax=vmax_scale,  # 设置合适的vmax以显示发射态的变化
        figsize=(18, 10),
        ax=ax,
        atom_at_end=True,  # SWAP传送带后原子在末尾
    )
    ax.set_title(f'仓状态概率 - 18 个态（780(3D) x 1517(6D））- vmax={vmax_scale:.3f}', fontsize=12)

    # 调整布局：为左侧y轴标签、顶部x轴和右侧colorbar留出空间
    plt.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.08)

    plt.savefig('test_emission_heatmap.png', dpi=150, bbox_inches='tight')
    print("  热图已保存到: test_emission_heatmap.png")

    # ========================================================================
    # 一致性检查
    # ========================================================================

    print("\n" + "=" * 70)
    print("一致性检查")
    print("=" * 70)

    # 检查原子状态
    # 注意：经过SWAP传送带后，原子从site 0移到了site 199
    rho_atom = mps.get_reduced_density([atom_position])
    p_excited = rho_atom[2, 2].real
    p_g0 = rho_atom[0, 0].real
    p_g1 = rho_atom[1, 1].real

    print(f"\n原子状态（位于 site {atom_position}）:")
    print(f"  P(|e>) = {p_excited:.6f}")
    print(f"  P(|0>) = {p_g0:.6f}")
    print(f"  P(|1>) = {p_g1:.6f}")
    print(f"  总计: {p_excited + p_g0 + p_g1:.6f}")

    # 预期总发射概率（所有仓求和）
    # 对于小角度：p_emit ~ gamma * dt，但这只是近似
    # 由于原子衰减，实际概率饱和于 1
    print(f"\n波包统计:")
    print(f"  总发射概率: {total_prob:.3f}")
    print(f"  峰值仓: {peak_idx + 1}（t={t[peak_idx]:.1f}ns）")
    print(f"  峰值: {peak_prob:.3f}")

    # 检查主要是单光子还是多光子
    if total_prob > 1.1:
        multi_photon = total_prob - 1.0
        print(f"  多光子概率: ~{multi_photon:.3f}")
    elif total_prob < 0.9:
        print(f"  警告：发射概率较低")

    # 波包宽度（FWHM 近似）
    # 对per_bin概率计算FWHM
    threshold = peak_prob / 2
    above_threshold = per_bin_prob_total > threshold
    if np.any(above_threshold):
        fwhm = above_threshold.sum() * dt_ns
        print(f"  波包 FWHM: ~{fwhm:.1f} ns")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    test_emission_wavepacket()
