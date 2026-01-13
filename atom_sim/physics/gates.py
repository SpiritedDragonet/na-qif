# -*- coding: utf-8 -*-
"""
带缓存的幺正门工厂

本模块提供用于构造时间仓仿真中使用的幺正门的工厂函数。
不随每仓变化的门会被缓存。
"""

from typing import Optional, Tuple
from functools import lru_cache
import numpy as np
from scipy.linalg import expm

from ..hilbert.basis import (
    SubSpace,
    ProductSpace,
    subspace_gate,
    SUBSPACE_780,
    SUBSPACE_1517,
    BIN_SPACE,
    get_bin_space,
    get_system_space,
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
    量子频率转换门 U_qfc。

    通过类分束器耦合将780nm光子转换为1517nm：
        U_qfc = exp(theta_H * (b_H c_H^† - b_H^† c_H) + theta_V * (b_V c_V^��� - b_V^† c_V))

    这是作用于18D bin空间（780 x 1517）的单格点幺正。

    Parameters
    ----------
    theta_H : float
        H偏振的转换角（sin²(theta) = 转换概率）
    theta_V : float
        V偏振的转换角

    Returns
    -------
    np.ndarray
        作用于bin空间的18x18幺正矩阵

    Examples
    --------
    >>> U = qfc_gate(theta_H=np.pi/4, theta_V=np.pi/4)  # 50%转换
    """
    # 获取湮灭/产生算符
    bH = annihilation_op(SUBSPACE_780, mode_id=0)
    bH_dag = creation_op(SUBSPACE_780, mode_id=0)
    bV = annihilation_op(SUBSPACE_780, mode_id=1)
    bV_dag = creation_op(SUBSPACE_780, mode_id=1)

    cH = annihilation_op(SUBSPACE_1517, mode_id=0)
    cH_dag = creation_op(SUBSPACE_1517, mode_id=0)
    cV = annihilation_op(SUBSPACE_1517, mode_id=1)
    cV_dag = creation_op(SUBSPACE_1517, mode_id=1)

    # 在780子空间上构建生成元（通过张量积作用于1517中的c）
    # G = -i * theta_H * (b_H c_H^† - b_H^† c_H) - i * theta_V * (b_V c_V^† - b_V^† c_V)
    # 总生成元作用于780 x 1517积空间

    # 需要正确嵌入算符
    # b作用于780，c作用于1517，所以b ⊗ c^†作用于积空间

    I_780 = np.eye(3, dtype=complex)
    I_1517 = np.eye(6, dtype=complex)

    # b_H ⊗ I_1517
    bH_full = np.kron(bH, I_1517)
    # I_780 ⊗ c_H^†
    cH_dag_full = np.kron(I_780, cH_dag)
    # b_H^† ⊗ I_1517
    bH_dag_full = np.kron(bH_dag, I_1517)
    # I_780 ⊗ c_H
    cH_full = np.kron(I_780, cH)

    # V模式同理
    bV_full = np.kron(bV, I_1517)
    cV_dag_full = np.kron(I_780, cV_dag)
    bV_dag_full = np.kron(bV_dag, I_1517)
    cV_full = np.kron(I_780, cV)

    # 生成元：theta * (b c^† - b^† c)
    # 这是反厄米的，所以exp(G)是幺正的
    G_H = theta_H * (bH_full @ cH_dag_full - bH_dag_full @ cH_full)
    G_V = theta_V * (bV_full @ cV_dag_full - bV_dag_full @ cV_full)

    G = G_H + G_V

    # 指数化得到幺正
    U = expm(G)

    return U


def filter_780_gate() -> np.ndarray:
    """
    780nm滤波器投影算符：移除所有780nm光子同时保持1517nm完整。

    这是一个投影算符（非幺正），作用于18D bin空间：
        P_filter = |vac><vac|_780 ⊗ I_1517

    它有效地将任何780nm光子态投影到真空，保持1517nm态。
    由于这是投影算符，不保持模长 - 应用后必须重新归一化态。

    Returns
    -------
    np.ndarray
        18x18投影矩阵
    """
    from ..hilbert.basis import SUBSPACE_780, SUBSPACE_1517

    dim_780 = SUBSPACE_780.dim  # 3: vac, H, V
    dim_1517 = SUBSPACE_1517.dim  # 6: vac, H, V, 2H, 2V, HV

    # 780子空���中的|vac><vac|
    P_vac_780 = np.zeros((dim_780, dim_780), dtype=complex)
    P_vac_780[0, 0] = 1.0  # 只有|vac><vac|存活

    # 1517子空间中的单位矩阵
    I_1517 = np.eye(dim_1517, dtype=complex)

    # 张量积：|vac><vac|_780 ⊗ I_1517
    P_filter = np.kron(P_vac_780, I_1517)

    return P_filter


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


def bs_gate_bin18() -> np.ndarray:
    """
    18D bin的50/50分束器门（324x324）。

    这是应该与当前MPS结构一起使用的版本，
    其中每个bin是18D（780 × 1517）。

    BS仅作用于1517nm子空间，保持780nm不变。
    对于每个固定的(780_A, 780_B)配置，它将36x36 BS门
    应用于1517_A × 1517_B子空间。

    Returns
    -------
    np.ndarray
        bin_A × bin_B空间的324x324幺正矩阵（每个站点18×18）
    """
    # 获取1517_A × 1517_B上的核心36x36 BS门
    U_bs_1517 = _bs_gate_1517()  # 36x36

    # 嵌入完整的324x324空间
    # 每个bin是780(3D) × 1517(6D) = 18D
    # 联合空间是18 × 18 = 324D
    dim_780 = 3
    dim_1517 = 6
    dim_bin = 18
    dim_full = 324

    U_full = np.zeros((dim_full, dim_full), dtype=complex)

    # 对于每个(780_A, 780_B)配置，将BS应用于1517子空间
    # 780部分不变（单位），只有1517被混合
    for i_780_A in range(dim_780):
        for i_780_B in range(dim_780):
            # 对于此固定的780配置，遍历所有1517组合
            for i_1517_A_out in range(dim_1517):
                for i_1517_B_out in range(dim_1517):
                    # 输出bin索引
                    idx_A_out = i_780_A * dim_1517 + i_1517_A_out
                    idx_B_out = i_780_B * dim_1517 + i_1517_B_out
                    row = idx_A_out * dim_bin + idx_B_out

                    for i_1517_A_in in range(dim_1517):
                        for i_1517_B_in in range(dim_1517):
                            # 输入bin索引（相同780，不同1517）
                            idx_A_in = i_780_A * dim_1517 + i_1517_A_in
                            idx_B_in = i_780_B * dim_1517 + i_1517_B_in
                            col = idx_A_in * dim_bin + idx_B_in

                            # 获取此1517跃迁的BS矩阵元
                            i_1517_out = i_1517_A_out * dim_1517 + i_1517_B_out
                            i_1517_in = i_1517_A_in * dim_1517 + i_1517_B_in
                            U_full[row, col] = U_bs_1517[i_1517_out, i_1517_in]

    return U_full


@lru_cache(maxsize=16)
def jones_gate(U: Tuple[Tuple[complex, complex], Tuple[complex, complex]]) -> np.ndarray:
    """
    琼斯偏振旋转门 U_pol。

    将2x2琼斯矩阵应用于通信（1517nm）H/V子空间：
        (c_H', c_V')^T = U * (c_H, c_V)^T

    Parameters
    ----------
    U : Tuple[Tuple[complex, complex], Tuple[complex, complex]]
        2x2琼斯矩阵，以嵌套元组形式用于哈希（可缓存）
        格式：((u00, u01), (u10, u11))

    Returns
    -------
    np.ndarray
        作用于1517子空间的6x6幺正矩阵（在真空和双光子态上为单位）

    Examples
    --------
    >>> # 45度半波片
    >>> import numpy as np
    >>> U_hwp = ((1, 0), (0, -1))
    >>> U = jones_gate(U_hwp)
    """
    u00, u01 = U[0]
    u10, u11 = U[1]

    # 1517基：vac, H, V, 2H, 2V, HV
    # 琼斯旋转作用于单光子H/V子空间
    # 对于多光子态，它作用为U⊗U于适当的张量幂

    # 构建6x6矩阵
    op = np.zeros((6, 6), dtype=complex)

    # 真空不变
    op[0, 0] = 1.0

    # 单光子子空间：(H, V) -> U @ (H, V)
    op[1, 1] = u00  # H -> u00*H + u10*V
    op[2, 1] = u10
    op[1, 2] = u01  # V -> u01*H + u11*V
    op[2, 2] = u11

    # 双光子子空间：U作用为U ⊗ U
    # |2H> = |HH> -> (u00*H + u10*V) ⊗ (u00*H + u10*V)
    # = u00²|HH> + u00*u10|HV> + u10*u00|VH> + u10²|VV>
    # 但由于我们有不可区分光子，|HV> = |VH>

    # |2H> -> u00²|2H> + 2*u00*u10|HV> + u10²|2V>
    op[3, 3] = u00 * u00  # |2H>
    op[5, 3] = 2 * u00 * u10  # |HV>
    op[4, 3] = u10 * u10  # |2V>

    # |2V> -> u01²|2H> + 2*u01*u11|HV> + u11²|2V>
    op[3, 4] = u01 * u01
    op[5, 4] = 2 * u01 * u11
    op[4, 4] = u11 * u11

    # |HV> -> (u00*H + u10*V) ⊗ (u01*H + u11*V)
    # = u00*u01|HH> + (u00*u11 + u10*u01)|HV> + u10*u11|VV>
    op[3, 5] = u00 * u01
    op[5, 5] = u00 * u11 + u10 * u01
    op[4, 5] = u10 * u11

    return op


def jones_gate_from_array(U_array: np.ndarray) -> np.ndarray:
    """
    便捷包装器，用numpy数组调用jones_gate。
    返回嵌入18D bin空间的门（I_780 ⊗ U_1517）。

    Parameters
    ----------
    U_array : np.ndarray
        2x2琼斯矩阵

    Returns
    -------
    np.ndarray
        作用于完整bin空间（780 × 1517）的18x18幺正矩阵
    """
    U_tuple = (
        (complex(U_array[0, 0]), complex(U_array[0, 1])),
        (complex(U_array[1, 0]), complex(U_array[1, 1]))
    )
    U_1517 = jones_gate(U_tuple)  # 6x6

    # 嵌入18D bin空间：I_780 ⊗ U_1517
    I_780 = np.eye(3, dtype=complex)
    return np.kron(I_780, U_1517)  # 18x18


@lru_cache(maxsize=2)
def swap_gate(d1: int, d2: int) -> np.ndarray:
    """
    用于交换两个站点的SWAP门。

    W |s> ⊗ |t> = |t> ⊗ |s>

    用于"传送带"协议以沿链移动系统站点。

    Parameters
    ----------
    d1 : int
        第一个站点的维度
    d2 : int
        第二个站点的维度

    Returns
    -------
    np.ndarray
        (d1*d2, d1*d2)置换矩阵

    Examples
    --------
    >>> W = swap_gate(9, 18)  # 交换系统（9D）与bin（18D）
    """
    # 显式构造SWAP矩阵
    dim = d1 * d2
    W = np.zeros((dim, dim), dtype=complex)

    for i in range(d1):
        for j in range(d2):
            # |i> ⊗ |j> -> |j> ⊗ |i>
            row_idx = i * d2 + j
            col_idx = j * d1 + i
            W[row_idx, col_idx] = 1.0

    return W


def emission_gate(
    gamma: float,
    dt: float,
    Alpha: np.ndarray,
    which_atom: str = 'A'
) -> np.ndarray:
    """
    原子-光子纠缠的发射门 U_emit（嵌入bin空间）。

        U_emit = exp(√(dt) * (L ⊗ b^†_780 - L^† ⊗ b_780))

    其中 L = √gamma * (alpha_+ * S_+ + alpha_- * S_-)
    且 S_± 是原子跃迁算符。

    门嵌入18D bin空间为U_9x9 ⊗ I_1517，
    其中9×9门作用于原子(3D) × 780(3D)，I_1517是通信子空间上的单位。

    这在原子态和发射光子偏振之间创建纠缠。
    发射的光子在780nm子空间中，稍后可通过QFC
    转换为1517nm。

    Parameters
    ----------
    gamma : float
        此时间步的发射率
    dt : float
        时间仓宽度
    Alpha : np.ndarray
        从原子跃迁到H/V的2x2偏振映射矩阵
        [[alpha_H+, alpha_H-], [alpha_V+, alpha_V-]]
    which_atom : str
        哪个原子（'A' 或 'B'）

    Returns
    -------
    np.ndarray
        作用于原子(3D) × bin(18D=780×1517)的(54, 54)幺正矩阵

    Examples
    --------
    >>> # 示例：圆偏振映射
    >>> Alpha = np.array([[1, 0], [0, 1]])  # σ+ -> H, σ- -> V
    >>> U = emission_gate(gamma=0.1, dt=1.0, Alpha=Alpha, which_atom='A')
    """
    # 原子跃迁算符
    S_plus = atom_transition('+')  # |0><e|
    S_minus = atom_transition('-')  # |1><e|

    # 提取Alpha矩阵元素
    alpha_H_plus = Alpha[0, 0]
    alpha_H_minus = Alpha[0, 1]
    alpha_V_plus = Alpha[1, 0]
    alpha_V_minus = Alpha[1, 1]

    # 在原子(3D)上构造L算符
    # L = √gamma * (alpha_H+ * S_+ + alpha_H- * S_-) 用于H偏振
    # V偏振同理
    sqrt_gamma = np.sqrt(gamma)

    L_H = sqrt_gamma * (alpha_H_plus * S_plus + alpha_H_minus * S_minus)
    L_V = sqrt_gamma * (alpha_V_plus * S_plus + alpha_V_minus * S_minus)

    # 780上的光子算符（3D：vac, H, V）
    # b^†_H = |H><vac|
    bH_dag = np.zeros((3, 3), dtype=complex)
    bH_dag[1, 0] = 1.0
    bH = bH_dag.conj().T

    bV_dag = np.zeros((3, 3), dtype=complex)
    bV_dag[2, 0] = 1.0
    bV = bV_dag.conj().T

    # 生成元：G = √dt * (L_H ⊗ b_H^† + L_V ⊗ b_V^† - h.c.)
    I_atom = np.eye(3, dtype=complex)
    I_780 = np.eye(3, dtype=complex)

    sqrt_dt = np.sqrt(dt)

    G_H = sqrt_dt * (np.kron(L_H, bH_dag) - np.kron(L_H.conj().T, bH))
    G_V = sqrt_dt * (np.kron(L_V, bV_dag) - np.kron(L_V.conj().T, bV))

    G_9x9 = G_H + G_V

    # 指数化得到原子×780上的幺正
    U_9x9 = expm(G_9x9)

    # 1517子空间（6D）上的单位
    I_1517 = np.eye(6, dtype=complex)

    # 嵌入原子×bin(780×1517)空间：U_54 = U_9x9 ⊗ I_1517
    # 这给出(9×9) ⊗ (6×6) = (54, 54)
    U_54 = np.kron(U_9x9, I_1517)

    return U_54
