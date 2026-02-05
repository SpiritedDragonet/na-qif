# -*- coding: utf-8 -*-
"""
带缓存的幺正门工厂

本模块提供用于构造时间仓仿真中使用的幺正门的工厂函数。
不随每仓变化的门会被缓存。
"""

from typing import Tuple, Optional
from functools import lru_cache
import numpy as np
from scipy.linalg import expm

from ..hilbert.basis import (
    SUBSPACE_1517,
    project_6d_to_3d,
)
from ..hilbert.operators import (
    annihilation_op,
    creation_op,
    atom_transition,
)


# 缓存的门（这些计算昂贵且不随每仓变化）

@lru_cache(maxsize=8)
def qfc_gate(theta_H: float = 0.0, theta_V: float = 0.0) -> np.ndarray:
    """
    量子频率转换门 U_qfc（5D bin）。

    通过类分束器耦合将780nm光子转换为1517nm：
        对每个偏振在 (|H_780>, |H_1517>) 与 (|V_780>, |V_1517>) 上做二维旋转，
        其中 sin^2(theta) = 转换概率。

    Parameters
    ----------
    theta_H : float
        H偏振的转换角（sin²(theta) = 转换概率）
    theta_V : float
        V偏振的转换角

    Returns
    -------
    np.ndarray
        作用于bin空间的5x5幺正矩阵

    Examples
    --------
    >>> U = qfc_gate(theta_H=np.pi/4, theta_V=np.pi/4)  # 50%转换
    """
    # ------------------------------------------------------------------
    # 5D bin 基序：
    #   0: |vac>
    #   1: |H_780>
    #   2: |V_780>
    #   3: |H_1517>
    #   4: |V_1517>
    #
    # 在 (1,3) 与 (2,4) 子空间做二维旋转。
    # ------------------------------------------------------------------
    U = np.eye(5, dtype=complex)

    cH = np.cos(theta_H)
    sH = np.sin(theta_H)
    cV = np.cos(theta_V)
    sV = np.sin(theta_V)

    # H 偏振：|H_780> <-> |H_1517>
    U[1, 1] = cH
    U[1, 3] = -sH
    U[3, 1] = sH
    U[3, 3] = cH

    # V 偏振：|V_780> <-> |V_1517>
    U[2, 2] = cV
    U[2, 4] = -sV
    U[4, 2] = sV
    U[4, 4] = cV

    return U



@lru_cache(maxsize=4)
def bs_gate_6d() -> np.ndarray:
    """
    6D 输出端口（1517nm）的50/50分束器门（36x36）。

    该门用于测量端共轭：BS 后的端口需要容纳 2 光子态。

    Returns
    -------
    np.ndarray
        1517_A × 1517_B空间的36x36幺正矩阵（每个站点6×6）
    """
    # ------------------------------------------------------------------
    # 该 BS 只作用在 1517nm 子空间 (6D)：
    #   - H/V 各自做 50/50 beamsplitter
    #   - 多光子态 (|2H>, |HV>, |2V>) 通过算符指数自然包含
    # ------------------------------------------------------------------
    return _bs_gate_1517()


@lru_cache(maxsize=4)
def bs_gate_3d() -> np.ndarray:
    """
    3D 输出端口（vac/H/V）的 50/50 分束器门（9x9）。
    """
    return project_6d_to_3d(bs_gate_6d())


def _permute_factors_matrix(dims: Tuple[int, ...], perm: Tuple[int, ...]) -> np.ndarray:
    """构造张量因子重排的置换矩阵。"""
    dim = int(np.prod(dims))
    P = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        multi = np.unravel_index(idx, dims)
        permuted = tuple(multi[p] for p in perm)
        jdx = np.ravel_multi_index(permuted, dims)
        P[jdx, idx] = 1.0
    return P


def _bs_gate_9d_dist_from_6d(U_6d: np.ndarray) -> np.ndarray:
    """
    从 6D BS 推导“可区分标签”的 9D × 9D 两端口 BS（81x81）。

    逻辑：
      - 先从 6D BS 提取 3D 单光子块（9x9）
      - 对标签 a / b 分别作用：U_a ⊗ U_b
      - 用置换矩阵从 (aA,aB,bA,bB) 重排回 (aA,bA,aB,bB)
    """
    U_3d = project_6d_to_3d(U_6d)
    U_ab = np.kron(U_3d, U_3d)
    # perm: (aA, aB, bA, bB) -> (aA, bA, aB, bB)
    P_swap = _permute_factors_matrix((3, 3, 3, 3), (0, 2, 1, 3))
    return P_swap @ U_ab @ P_swap.T


@lru_cache(maxsize=4)
def _bs_gate_9d_dist_cached() -> np.ndarray:
    return _bs_gate_9d_dist_from_6d(bs_gate_6d())


def bs_gate_9d_dist(bs_unitary_6d: Optional[np.ndarray] = None) -> np.ndarray:
    """
    9D 标签空间下的可区分 BS（81x81）。
    若传入 6D BS，则使用该门推导；否则使用默认 6D BS。
    """
    if bs_unitary_6d is None:
        return _bs_gate_9d_dist_cached()
    return _bs_gate_9d_dist_from_6d(bs_unitary_6d)


@lru_cache(maxsize=4)
def _bs_gate_1517() -> np.ndarray:
    """
    内部函数：1517_A × 1517_B上的50/50分束器（36x36）。

    这是仅作用于通信子空间的核心BS门。

    Returns
    -------
    np.ndarray
        1517_A × 1517空间的36x36幺正矩阵（每个站点6×6）
    """
    def make_generator(mode_id: int) -> np.ndarray:
        """为单个偏振模式构造BS生成元。"""
        c = annihilation_op(SUBSPACE_1517, mode_id)  # 6x6
        c_dag = creation_op(SUBSPACE_1517, mode_id)  # 6x6

        # 在联合1517_A × 1517_B空间（36D）上构造算符
        c_A = np.kron(c, np.eye(6, dtype=complex))
        c_B = np.kron(np.eye(6, dtype=complex), c)
        c_dag_A = np.kron(c_dag, np.eye(6, dtype=complex))
        c_dag_B = np.kron(np.eye(6, dtype=complex), c_dag)

        # BS生成元：G = θ * (c_A^† c_B - c_A c_B^†)
        # For a 50:50 beam splitter, sin^2(theta) = 0.5 -> theta = pi/4
        theta = np.pi / 4
        G = theta * (c_dag_A @ c_B - c_A @ c_dag_B)
        return G

    # 为H和V偏振生成生成元
    G_H = make_generator(mode_id=0)  # H偏振
    G_V = make_generator(mode_id=1)  # V偏振

    # 总生成元（对两种偏振求和）
    G_total = G_H + G_V

    # 指数化得到幺正
    U_bs = expm(G_total)

    return U_bs


def emission_gate(
    gamma: float,
    dt: float,
    Alpha: np.ndarray,
    phase: float = 0.0,
    H_sys: Optional[np.ndarray] = None,
    bin_first: bool = False
) -> np.ndarray:
    """
    原子-光子纠缠的发射门 U_emit（嵌入5D bin空间）。

        U_emit = exp(√(dt) * (L ⊗ b^†_780 - L^† ⊗ b_780))

    其中 L = √gamma * (alpha_+ * S_+ + alpha_- * S_-)
    且 S_± 是原子跃迁算符。

    门嵌入5D bin空间：仅作用于 (vac, H_780, V_780) 子块，
    对 (H_1517, V_1517) 子块保持单位。

    这在原子态和发射光子偏振之间创建纠缠。
    发射的光子在780nm子空间中，稍后可通过QFC
    转换为1517nm。

    Parameters
    ----------
    gamma : float
        此时间步的单通道发射率（总发射率的一半）
    dt : float
        时间仓宽度
    Alpha : np.ndarray
        从原子跃迁到H/V的2x2偏振映射矩阵
        [[alpha_H+, alpha_H-], [alpha_V+, alpha_V-]]
    phase : float
        发射波包的相位（会同时作用于H/V通道）
    H_sys : np.ndarray, optional
        原子系统哈密顿量（4x4），用于在单步门中同时加入驱动与失谐
    bin_first : bool
        如果为 True，返回 I_1517 ⊗ U_12x12（作用于 bin × atom）
        如果为 False，返回 U_12x12 ⊗ I_1517（作用于 atom × bin）

    Returns
    -------
    np.ndarray
        20x20 幺正矩阵
        - bin_first=False: 作用在 原子(4D) × bin(5D)
        - bin_first=True: 作用在 bin(5D) × 原子(4D)

    Examples
    --------
    >>> # 示例：圆偏振映射
    >>> Alpha = np.array([[1, 0], [0, 1]])  # σ+ -> H, σ- -> V
    >>> U = emission_gate(gamma=0.1, dt=1.0, Alpha=Alpha)
    """
    # ------------------------------------------------------------------
    # 发射门的结构（碰撞模型离散化）：
    #   U_emit = exp[ sqrt(dt) * (L ⊗ b^† - L^† ⊗ b)  - i dt (H_sys ⊗ I) ]
    #
    # 其中：
    #   L = sqrt(gamma) * (alpha_+ S_+ + alpha_- S_-)
    #   b^† 是 780nm 光子的产生算符（单光子截断）
    #
    # 该门在“原子 × 780”上是 12x12，再嵌入到 5D bin 得到 20x20。
    # ------------------------------------------------------------------
    # 原子跃迁算符
    S_plus = atom_transition('+')  # |0><e|
    S_minus = atom_transition('-')  # |1><e|

    # 提取Alpha矩阵元素
    alpha_H_plus = Alpha[0, 0]
    alpha_H_minus = Alpha[0, 1]
    alpha_V_plus = Alpha[1, 0]
    alpha_V_minus = Alpha[1, 1]

    # 在原子(4D)上构造L算符
    # L = √gamma * (alpha_H+ * S_+ + alpha_H- * S_-) 用于H偏振
    # V偏振同理
    sqrt_gamma = np.sqrt(gamma)
    phase_factor = np.exp(1j * phase) if phase != 0.0 else 1.0

    L_H = phase_factor * sqrt_gamma * (alpha_H_plus * S_plus + alpha_H_minus * S_minus)
    L_V = phase_factor * sqrt_gamma * (alpha_V_plus * S_plus + alpha_V_minus * S_minus)

    # 780上的光子算符（3D：vac, H, V）
    # b^†_H = |H><vac|
    bH_dag = np.zeros((3, 3), dtype=complex)
    bH_dag[1, 0] = 1.0
    bH = bH_dag.conj().T

    bV_dag = np.zeros((3, 3), dtype=complex)
    bV_dag[2, 0] = 1.0
    bV = bV_dag.conj().T

    # 生成元：G = √dt * (L_H ⊗ b_H^† + L_V ⊗ b_V^† - h.c.)
    sqrt_dt = np.sqrt(dt)

    G_H = sqrt_dt * (np.kron(L_H, bH_dag) - np.kron(L_H.conj().T, bH))
    G_V = sqrt_dt * (np.kron(L_V, bV_dag) - np.kron(L_V.conj().T, bV))
    G_12x12 = G_H + G_V

    d_atom = L_H.shape[0]
    if H_sys is not None:
        if H_sys.shape != (d_atom, d_atom):
            raise ValueError(f"H_sys 维度应为 ({d_atom},{d_atom})，实际为 {H_sys.shape}")
        G_sys = -1j * dt * np.kron(H_sys, np.eye(3, dtype=complex))
        G_12x12 = G_12x12 + G_sys

    # 指数化得到原子×780上的幺正
    U_12x12 = expm(G_12x12)

    # U_12x12 作用在 atom(4D) × 780(3D) 上，形状 (12, 12)
    # Reshape 为 (d_atom, d_780, d_atom, d_780)
    U_12x12_4d = U_12x12.reshape(d_atom, 3, d_atom, 3)

    if bin_first:
        # bin × atom: (5D bin) × atom
        # 在 bin-first 索引下嵌入 780 子块，其余 1517 分量保持单位
        U_20 = np.eye(5 * d_atom, dtype=complex)
        for iatom in range(d_atom):
            for i780 in range(3):
                row = i780 * d_atom + iatom
                for jatom in range(d_atom):
                    for j780 in range(3):
                        col = j780 * d_atom + jatom
                        U_20[row, col] = U_12x12_4d[iatom, i780, jatom, j780]
    else:
        # atom × bin: atom × (5D bin)
        # 在 bin-last 索引下嵌入 780 子块，其余 1517 分量保持单位
        U_20 = np.eye(d_atom * 5, dtype=complex)
        for iatom in range(d_atom):
            for i780 in range(3):
                row = iatom * 5 + i780
                for jatom in range(d_atom):
                    for j780 in range(3):
                        col = jatom * 5 + j780
                        U_20[row, col] = U_12x12_4d[iatom, i780, jatom, j780]

    return U_20
