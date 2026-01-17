# -*- coding: utf-8 -*-
"""
测试BS门是否正确产生Hong-Ou-Mandel干涉效应
"""

import numpy as np
import sys
sys.path.insert(0, r'G:\BProj\Quantum_simulation')

from atom_sim.physics.gates import bs_gate_6d
from atom_sim.hilbert.basis import SUBSPACE_1517

# 1517子空间基顺序：vac, H, V, 2H, 2V, HV
# 索引：0=vac, 1=H, 2=V, 3=2H, 4=2V, 5=HV

print("="*70)
print("测试BS门的Hong-Ou-Mandel干涉效应")
print("="*70)

# 获取BS门（36x36，作用于1517_A × 1517_B）
U_bs = bs_gate_6d()

print(f"\nBS门维度: {U_bs.shape}")
print(f"BS门是否幺正: {np.allclose(U_bs @ U_bs.conj().T, np.eye(36))}")

# =========================================================================
# 测试1：|H>_A |H>_B 输入（两个相同偏振光子）
# =========================================================================
print("\n" + "="*70)
print("【测试1】输入态: |H>_A ⊗ |H>_B （两个H偏振光子）")
print("="*70)
print("预期：HOM干涉 → 应该看到 |2H,vac> 和 |vac,2H> 各50%")
print("      不应该看到 |H,H> （HOM dip）")

# 构造输入态 |H>_A ⊗ |H>_B
psi_in = np.zeros(36, dtype=complex)
idx_H_H = 1 * 6 + 1  # A在H(1), B在H(1)
psi_in[idx_H_H] = 1.0

# 应用BS门
psi_out = U_bs @ psi_in

# 检查输出
print("\n输出态的非零分量：")
for i in range(36):
    if np.abs(psi_out[i]) > 1e-6:
        i_A = i // 6
        i_B = i % 6
        basis_names = ['vac', 'H', 'V', '2H', '2V', 'HV']
        prob = np.abs(psi_out[i])**2
        print(f"  |{basis_names[i_A]}>_A ⊗ |{basis_names[i_B]}>_B: "
              f"振幅={psi_out[i]:.4f}, 概率={prob:.4f}")

# 关键检查
idx_2H_vac = 3 * 6 + 0  # |2H>_A ⊗ |vac>_B
idx_vac_2H = 0 * 6 + 3  # |vac>_A ⊗ |2H>_B
idx_H_H_out = 1 * 6 + 1  # |H>_A ⊗ |H>_B

prob_2H_vac = np.abs(psi_out[idx_2H_vac])**2
prob_vac_2H = np.abs(psi_out[idx_vac_2H])**2
prob_H_H = np.abs(psi_out[idx_H_H_out])**2

print(f"\n关键概率：")
print(f"  P(|2H,vac>) = {prob_2H_vac:.6f}  （期望 ~0.5）")
print(f"  P(|vac,2H>) = {prob_vac_2H:.6f}  （期望 ~0.5）")
print(f"  P(|H,H>)    = {prob_H_H:.6f}  （期望 ~0，HOM dip）")

if prob_H_H > 0.01:
    print("\n❌ 错误：|H,H> 态有显著概率，HOM干涉失败！")
else:
    print("\n✓ 正确：|H,H> 态被抑制，HOM干涉成功")

# =========================================================================
# 测试2：|H>_A |V>_B 输入（不同偏振光子）
# =========================================================================
print("\n" + "="*70)
print("【测试2】输入态: |H>_A ⊗ |V>_B （不同偏振光子）")
print("="*70)
print("预期：无HOM干涉 → 应该看到 |H,V>, |V,H> 各50%")

# 构造输入态 |H>_A ⊗ |V>_B
psi_in = np.zeros(36, dtype=complex)
idx_H_V = 1 * 6 + 2  # A在H(1), B在V(2)
psi_in[idx_H_V] = 1.0

# 应用BS门
psi_out = U_bs @ psi_in

# 检查输出
print("\n输出态的非零分量：")
for i in range(36):
    if np.abs(psi_out[i]) > 1e-6:
        i_A = i // 6
        i_B = i % 6
        basis_names = ['vac', 'H', 'V', '2H', '2V', 'HV']
        prob = np.abs(psi_out[i])**2
        print(f"  |{basis_names[i_A]}>_A ⊗ |{basis_names[i_B]}>_B: "
              f"振幅={psi_out[i]:.4f}, 概率={prob:.4f}")

idx_H_V_out = 1 * 6 + 2  # |H>_A ⊗ |V>_B
idx_V_H_out = 2 * 6 + 1  # |V>_A ⊗ |H>_B

prob_H_V = np.abs(psi_out[idx_H_V_out])**2
prob_V_H = np.abs(psi_out[idx_V_H_out])**2

print(f"\n关键概率：")
print(f"  P(|H,V>) = {prob_H_V:.6f}  （期望 ~0.5）")
print(f"  P(|V,H>) = {prob_V_H:.6f}  （期望 ~0.5）")

# =========================================================================
# 测试3：检查BS门的生成元
# =========================================================================
print("\n" + "="*70)
print("【测试3】检查BS门的生成元形式")
print("="*70)

from atom_sim.hilbert.operators import annihilation_op, creation_op
from scipy.linalg import logm

# 重构生成元
c_H = annihilation_op(SUBSPACE_1517, mode_id=0)
c_V = annihilation_op(SUBSPACE_1517, mode_id=1)
c_H_dag = creation_op(SUBSPACE_1517, mode_id=0)
c_V_dag = creation_op(SUBSPACE_1517, mode_id=1)

I = np.eye(6, dtype=complex)

c_H_A = np.kron(c_H, I)
c_H_B = np.kron(I, c_H)
c_H_dag_A = np.kron(c_H_dag, I)
c_H_dag_B = np.kron(I, c_H_dag)

c_V_A = np.kron(c_V, I)
c_V_B = np.kron(I, c_V)
c_V_dag_A = np.kron(c_V_dag, I)
c_V_dag_B = np.kron(I, c_V_dag)

# 标准BS生成元：G = θ * (c_A^† c_B - c_A c_B^†)
theta = np.pi / 4
G_H = theta * (c_H_dag_A @ c_H_B - c_H_A @ c_H_dag_B)
G_V = theta * (c_V_dag_A @ c_V_B - c_V_A @ c_V_dag_B)
G_total = G_H + G_V

# 从U_bs提取生成元
G_extracted = logm(U_bs)

# 检查是否匹配（考虑到相位自由度）
diff = np.linalg.norm(G_extracted - G_total)
print(f"生成元差异: ||G_extracted - G_expected|| = {diff:.6e}")

if diff < 1e-10:
    print("✓ 生成元正确")
else:
    print("⚠ 生成元可能有问题")

print("\n" + "="*70)
print("测试完成")
print("="*70)
