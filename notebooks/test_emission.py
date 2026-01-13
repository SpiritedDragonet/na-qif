# -*- coding: utf-8 -*-
"""
简单 TEBD 测试：原子发射（使用 L0 层 MPSState）

物理原理：
---------
原子（三能级）：
  |e>: 激发态
  |0>, |1>: 基态

选择定则：
  |e> → |0>: sigma+ 光子（H 偏振）
  |e> |1>: sigma- 光子（V 偏振）

目的：测试 MPSState.apply_two_site_gate() 用于 TEBD 演化
"""

import sys
from pathlib import Path
import numpy as np

# 将父目录添加到路径以导入
# 注意：__file__ 在某些环境中不可用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 导入 L0 层（核心）MPS 基础设施
from atom_sim.core.mps import MPSState


def test_emission():
    """运行发射测试。"""

    # ========================================================================
    # 第 1 部分：系统设置
    # ========================================================================

    print("=" * 70)
    print("第 1 部分：系统设置")
    print("=" * 70)

    print("\n原子基矢（3D）: |0> [idx 0], |1> [idx 1], |e> [idx 2]")
    print("780nm 光子基矢（3D）: |vac> [idx 0], |H> [idx 1], |V> [idx 2]")

    # ========================================================================
    # 第 2 部分：构建发射门（U_emit）
    # ========================================================================

    print("\n" + "=" * 70)
    print("第 2 部分：构建发射门（U_emit）")
    print("=" * 70)

    # 门参数
    dt = 0.1
    g = 1.0  # 耦合强度
    theta = np.sqrt(dt) * g

    # 维度
    d_atom = 3
    d_photon = 3
    d_combined = d_atom * d_photon

    print(f"\n门构造: |e, vac> -> cos(theta)|e, vac> + sin(theta)/sqrt(2)(|0, H> + |1, V>)")
    print(f"  theta = sqrt(dt) * g = {theta:.4f}")

    # 构建 U_emit 矩阵
    U_emit = np.eye(d_combined, dtype=complex)

    # 索引映射: idx = atom_idx * d_photon + photon_idx
    e_vac_idx = 2 * d_photon + 0  # |e, vac>
    target_H_idx = 0 * d_photon + 1  # |0, H>
    target_V_idx = 1 * d_photon + 2  # |1, V>

    print(f"\n组合空间索引（9D）:")
    print(f"|e, vac> -> idx {e_vac_idx}")
    print(f"|0, H>   -> idx {target_H_idx}")
    print(f"|1, V>   -> idx {target_V_idx}")

    # 设置矩阵元素
    U_emit[e_vac_idx, e_vac_idx] = np.cos(theta)
    U_emit[target_H_idx, e_vac_idx] = np.sin(theta) / np.sqrt(2)
    U_emit[target_V_idx, e_vac_idx] = np.sin(theta) / np.sqrt(2)

    # ========================================================================
    # 第 3 部分：创建初始 MPS 状态
    # ========================================================================

    print("\n" + "=" * 70)
    print("第 3 部分：创建初始 MPS 状态")
    print("=" * 70)

    print("\n创建 MPS，local_dims = [3, 3]...")
    print("初始状态: 原子在 |e>（索引 2），光子在 |vac>（索引 0）")

    mps = MPSState(local_dims=[d_atom, d_photon], init_state=[2, 0], max_bond=10)

    print(f"\nMPS 已创建:")
    print(f"  L = {mps.L} 个格点")
    print(f"  d = {mps.d}")
    print(f"  chi = {mps.get_bond_dimensions()}")
    print(f"  归一化 = {mps.norm():.6f}")

    # 验证初始状态
    rho_atom_init = mps.get_atom_state(system_site=0)
    print("\n原子密度矩阵（应在索引 2 显示 |e><e|）:")
    print(rho_atom_init)

    # ========================================================================
    # 第 4 部分：应用发射门
    # ========================================================================

    print("\n" + "=" * 70)
    print("第 4 部分：通过 TEBD 应用发射门")
    print("=" * 70)

    print("\n正在向格点 (0, 1) 应用发射门 U^(emit)...")

    mps.apply_two_site_gate(site_left=0, gate=U_emit, truncate=True)

    print("发射门已应用！")
    print(f"门作用后的键维度: {mps.get_bond_dimensions()}")
    print(f"状态归一化: {mps.norm():.6f}")

    # ========================================================================
    # 第 5 部分：分析结果
    # ========================================================================

    print("\n" + "=" * 70)
    print("第 5 部分：分析结果")
    print("=" * 70)

    rho_atom = mps.get_atom_state(system_site=0)
    print("\n原子约化密度矩阵:")
    print(rho_atom)

    rho_photon = mps.get_reduced_density(sites=[1])
    print("\n光子约化密度矩阵:")
    print(rho_photon)

    # 概率
    p_remain = rho_atom[2, 2].real
    p_H = rho_atom[0, 0].real
    p_V = rho_atom[1, 1].real
    p_emit = p_H + p_V

    print(f"\n概率:")
    print(f"  P(原子在 |e>)        = {p_remain:.6f}")
    print(f"  P(原子在 |0>, H 发射)  = {p_H:.6f}")
    print(f"  P(原子在 |1>, V 发射)  = {p_V:.6f}")
    print(f"  P(发射)            = {p_emit:.6f}")

    # 一致性检查
    p_photon_H = rho_photon[1, 1].real
    p_photon_V = rho_photon[2, 2].real
    print(f"\n一致性检查:")
    print(f"  P(发射|原子)   = {p_emit:.6f}")
    print(f"  P(发射|光子) = {p_photon_H + p_photon_V:.6f}")
    print(f"  差值 = {abs(p_emit - (p_photon_H + p_photon_V)):.2e}")

    # ========================================================================
    # 第 6 部分：纠缠分析
    # ========================================================================

    print("\n" + "=" * 70)
    print("第 6 部分：纠缠分析")
    print("=" * 70)

    chi = mps.get_bond_dimensions()
    print(f"\n键维度: {chi}")
    print(f"Schmidt 秩 = {chi[0]}")
    print(f"\n预期 Schmidt 秩 = 3（叠加态中的 3 个正交项）")

    # ========================================================================
    # 汇总
    # ========================================================================

    print("\n" + "=" * 70)
    print("汇总：使用的 L0 层函数")
    print("=" * 70)
    print("\n  1. MPSState.__init__()         - 创建 MPS")
    print("  2. MPSState.apply_two_site_gate() - 应用双格点酉算符")
    print("  3. MPSState.get_atom_state()      - 获取约化密度矩阵")
    print("  4. MPSState.get_reduced_density() - 获取任意格点的约化态")
    print("  5. MPSState.norm()                - 获取状态归一化")
    print("  6. MPSState.get_bond_dimensions() - 获取纠缠信息")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_emission()
