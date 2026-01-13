# -*- coding: utf-8 -*-
"""
非幺正演化的Kraus信道

本模块提供各种量子信道的Kraus算符集：
- 振幅阻尼（光纤损耗）
- 探测POVM（开/关测量）
- 原子退相干
"""

from typing import List, Tuple
import numpy as np

from ..hilbert.basis import SUBSPACE_1517, SUBSPACE_780, BIN_SPACE


def loss_channel_both_subspaces(
    eta_780: float,
    eta_H_1517: float,
    eta_V_1517: float
) -> List[np.ndarray]:
    """
    作用于780和1517两个子空间的组合损耗信道。

    用于QFC应用：780nm通道通常 eta_780=0（100%过滤），
    而1517nm通道具有正常的传输损耗。

    Kraus算符是张量积：K_780^(k) ⊗ K_1517^(j)

    Parameters
    ----------
    eta_780 : float
        780nm子空间的透过率（0 = 100%损耗/过滤）
    eta_H_1517 : float
        1517nm H偏振的透过率
    eta_V_1517 : float
        1517nm V偏振的透过率

    Returns
    -------
    List[np.ndarray]
        作用于18D bin空间的Kraus算符列表
    """
    # 获取每个子空间的Kraus算符（未嵌入）
    K_780_list = loss_channel_780_general(eta_780)  # 3x3 矩阵
    K_1517_list = _loss_channel_1517_raw(eta_H_1517, eta_V_1517)  # 6x6 矩阵

    # 构成所有张量积组合
    K_combined = []
    for K_780 in K_780_list:
        for K_1517 in K_1517_list:
            # K_780 是 (3,3), K_1517 是 (6,6), 结果是 (18,18)
            K_combined.append(np.kron(K_780, K_1517))

    return K_combined


def _loss_channel_1517_raw(eta_H: float, eta_V: float) -> List[np.ndarray]:
    """
    原始1517nm损耗信道（6x6矩阵，未嵌入18D）。

    这是loss_channel_both_subspaces使用的内部函数。

    Parameters
    ----------
    eta_H : float
        H偏振的透过率
    eta_V : float
        V偏振的透过率

    Returns
    -------
    List[np.ndarray]
        仅作用于1517子空间的6x6 Kraus算符列表
    """
    K_list_1517 = []

    basis = [
        (0, 0),  # 0: vac
        (1, 0),  # 1: H
        (0, 1),  # 2: V
        (2, 0),  # 3: 2H
        (0, 2),  # 4: 2V
        (1, 1),  # 5: HV
    ]

    for kH in range(3):
        for kV in range(3):
            K = np.zeros((6, 6), dtype=complex)

            for i, (nH, nV) in enumerate(basis):
                if nH < kH or nV < kV:
                    continue

                nH_new = nH - kH
                nV_new = nV - kV

                target = (nH_new, nV_new)
                if target in basis:
                    j = basis.index(target)

                    from math import comb
                    coeff_H = np.sqrt(comb(nH, kH)) * (eta_H ** ((nH - kH) / 2)) * ((1 - eta_H) ** (kH / 2))
                    coeff_V = np.sqrt(comb(nV, kV)) * (eta_V ** ((nV - kV) / 2)) * ((1 - eta_V) ** (kV / 2))
                    K[j, i] = coeff_H * coeff_V

            K_list_1517.append(K)

    # 移除全零算符
    K_list_1517 = [K for K in K_list_1517 if np.any(K != 0)]
    return K_list_1517


def loss_channel_780_general(eta: float) -> List[np.ndarray]:
    """
    780nm子空间的通用损耗信道（每个模式最多1个光子）。

    对于eta=0：100%损耗（3个Kraus算符：|vac><vac|, |vac><H|, |vac><V|）
    对于eta=1：无损耗（K_0 = I）

    Parameters
    ----------
    eta : float
        透过率 (0 <= eta <= 1)

    Returns
    -------
    List[np.ndarray]
        780子空间的Kraus算符（3x3矩阵）
    """
    # 基：|vac>, |H>, |V>

    K_list = []

    if eta == 0.0:
        # 100%损耗：3个Kraus算符
        # K_0 = |vac><vac| （真空保持真空）
        K0 = np.zeros((3, 3), dtype=complex)
        K0[0, 0] = 1.0
        K_list.append(K0)

        # K_1 = |vac><H| （H光子损耗 -> 真空）
        K1 = np.zeros((3, 3), dtype=complex)
        K1[0, 1] = 1.0
        K_list.append(K1)

        # K_2 = |vac><V| （V光子损耗 -> 真空）
        K2 = np.zeros((3, 3), dtype=complex)
        K2[0, 2] = 1.0
        K_list.append(K2)
    elif eta == 1.0:
        # 无损耗：单位矩阵
        K_list.append(np.eye(3, dtype=complex))
    else:
        # 部分损耗：K_0（无损耗）和K_H、K_V（每个模式的损耗）
        K0 = np.zeros((3, 3), dtype=complex)
        K0[0, 0] = 1.0
        K0[1, 1] = np.sqrt(eta)
        K0[2, 2] = np.sqrt(eta)
        K_list.append(K0)

        # 每个模式的损耗算符
        loss_amp = np.sqrt(1 - eta)

        K_H = np.zeros((3, 3), dtype=complex)
        K_H[0, 1] = loss_amp
        K_list.append(K_H)

        K_V = np.zeros((3, 3), dtype=complex)
        K_V[0, 2] = loss_amp
        K_list.append(K_V)

    return K_list


def loss_channel_1517(eta_H: float, eta_V: float) -> List[np.ndarray]:
    """
    1517nm通信子空间（6D）的振幅阻尼，
    嵌入在18D bin空间中（I_780 ⊗ K_1517）。

    独立处理两个偏振模式，可能有不同的损耗。

    Parameters
    ----------
    eta_H : float
        H偏振的透过率
    eta_V : float
        V偏振的透过率

    Returns
    -------
    List[np.ndarray]
        作用于18D bin空间（780 × 1517）的Kraus算符列表
    """
    # 1517基：vac, H, V, 2H, 2V, HV
    # 占有数：(0,0), (1,0), (0,1), (2,0), (0,2), (1,1)

    # 需要为H和V模式上所有可能的损耗组合构造Kraus算符
    # 对于小截断，显式枚举

    K_list_1517 = []

    # 带有占有数元组的基
    basis = [
        (0, 0),  # 0: vac
        (1, 0),  # 1: H
        (0, 1),  # 2: V
        (2, 0),  # 3: 2H
        (0, 2),  # 4: 2V
        (1, 1),  # 5: HV
    ]

    # 对于每个可能的损耗结果（从H损耗kH个光子，从V损耗kV个）
    for kH in range(3):  # 可损耗0、1或2个H光子
        for kV in range(3):
            K = np.zeros((6, 6), dtype=complex)

            for i, (nH, nV) in enumerate(basis):
                if nH < kH or nV < kV:
                    continue  # 不能损耗超过已有的光子

                nH_new = nH - kH
                nV_new = nV - kV

                # 找到目标索引
                target = (nH_new, nV_new)
                if target in basis:
                    j = basis.index(target)

                    # 计算系数
                    # 独立H和V损耗的乘积
                    from math import comb
                    coeff_H = np.sqrt(comb(nH, kH)) * (eta_H ** ((nH - kH) / 2)) * ((1 - eta_H) ** (kH / 2))
                    coeff_V = np.sqrt(comb(nV, kV)) * (eta_V ** ((nV - kV) / 2)) * ((1 - eta_V) ** (kV / 2))
                    K[j, i] = coeff_H * coeff_V

            K_list_1517.append(K)

    # 移除全零算符
    K_list_1517 = [K for K in K_list_1517 if np.any(K != 0)]

    # 将每个Kraus算符嵌入18D bin空间：I_780 ⊗ K_1517
    I_780 = np.eye(3, dtype=complex)
    K_list_embedded = [np.kron(I_780, K) for K in K_list_1517]

    return K_list_embedded


def detection_channel(
    eta_det: float = 1.0,
    p_dark: float = 0.0
) -> Tuple[List[np.ndarray], List[int]]:
    """
    光子数测量的开/关探测POVM。

    模拟单光子探测器，具有：
    - 效率 eta_det
    - 每仓暗计数概率 p_dark

    POVM元素：
        E_0 = (1-p_dark) * sum_n (1-eta_det)^n |n><n|  （无点击）
        E_1 = I - E_0  （有点击）

    Kraus算符为 M_r = sqrt(E_r)。

    Parameters
    ----------
    eta_det : float
        探测效率 (0 <= eta_det <= 1)
    p_dark : float
        暗计数概率 (0 <= p_dark <= 1)

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        (Kraus算符，结果标签)
        结果0 = 无点击，结果1 = 有点击

    Examples
    --------
    >>> K, outcomes = detection_channel(eta_det=0.9, p_dark=0.001)
    >>> # K[0] = 无点击Kraus算符，K[1] = 有点击Kraus算符
    """
    if not 0 <= eta_det <= 1:
        raise ValueError(f"eta_det必须在[0, 1]内，得到 {eta_det}")
    if not 0 <= p_dark <= 1:
        raise ValueError(f"p_dark必须在[0, 1]内，得到 {p_dark}")

    # 对于6D 1517子空间（n_max = 2）
    dim = 6

    # 无点击POVM元素
    # E_0 = (1-p_dark) * [(1-eta)^0*|0><0| + (1-eta)^1*|1><1| + (1-eta)^2*|2><2|]
    # 但我们有多个单光子态和多光子态

    # 基：vac, H, V, 2H, 2V, HV
    # 需要每个基态的光子数
    n_per_state = np.array([0, 1, 1, 2, 2, 2])  # 总光子数

    # 无点击算符（对角）
    E0 = np.zeros((dim, dim), dtype=complex)
    for i, n in enumerate(n_per_state):
        prob_no_click = (1 - p_dark) * ((1 - eta_det) ** n)
        E0[i, i] = prob_no_click

    # 点击算符
    I = np.eye(dim, dtype=complex)
    E1 = I - E0

    # Kraus算符是矩阵平方根
    # 对于对角算符，就是对角元素的平方根
    M0 = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        M0[i, i] = np.sqrt(E0[i, i]) if E0[i, i] > 0 else 0

    # E1可能非对角（由于I - E0），但由于E0是对角的，E1也是对角的
    M1 = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        M1[i, i] = np.sqrt(E1[i, i]) if E1[i, i] > 0 else 0

    return [M0, M1], [0, 1]


def detection_povm_single_site(
    eta_det: float = 1.0,
    p_dark: float = 0.0
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    单个bin站点的开/关探测POVM（H和V探测器）。

    作用于18D bin空间（780 x 1517）。由于780nm被过滤，
    探测只响应1517nm光子。

    每个站点有两个探测器（H和V），给出4种可能的结果：
        (0, 0): 都不点击
        (1, 0): 只有H点击
        (0, 1): 只有V点击
        (1, 1): 两个都点击

    Parameters
    ----------
    eta_det : float
        探测效率 (0 <= eta_det <= 1)
    p_dark : float
        每个探测器每个仓的暗计数概率

    Returns
    -------
    Tuple[List[np.ndarray], List[Tuple[int, int]]]
        (Kraus算符 [4 x (18,18)], 结果标签 [(d_H, d_V)])

    Notes
    -----
    1517nm基：vac, H, V, 2H, 2V, HV，光子数为：
        - vac: n_H=0, n_V=0
        - H:   n_H=1, n_V=0
        - V:   n_H=0, n_V=1
        - 2H:  n_H=2, n_V=0
        - 2V:  n_H=0, n_V=2
        - HV:  n_H=1, n_V=1

    对于效率为eta的开/关探测器：
        P(无点击 | n个光子) = (1-eta)^n * (1-p_dark)  (忽略n>0时的暗计数)
        P(点击 | n个光子) = 1 - (1-eta)^n + 小的暗计数修正
    """
    if not 0 <= eta_det <= 1:
        raise ValueError(f"eta_det必须在[0, 1]内，得到 {eta_det}")
    if not 0 <= p_dark <= 1:
        raise ValueError(f"p_dark必须在[0, 1]内，得到 {p_dark}")

    # 1517nm基光子数 (n_H, n_V)
    photon_numbers = [
        (0, 0),  # vac
        (1, 0),  # H
        (0, 1),  # V
        (2, 0),  # 2H
        (0, 2),  # 2V
        (1, 1),  # HV
    ]

    # 为1517nm子空间（6D）构建POVM元素
    # E_{d_H, d_V} = 对于每个基态 P(d_H | n_H) * P(d_V | n_V)

    E_list_1517 = []
    outcomes = []

    for d_H in range(2):  # 0 = 无点击，1 = 有点击
        for d_V in range(2):
            E = np.zeros((6, 6), dtype=complex)
            for i, (n_H, n_V) in enumerate(photon_numbers):
                # 给定(n_H, n_V)个光子时结果(d_H, d_V)的概率
                if d_H == 0:  # H无点击
                    if n_H == 0:
                        P_H = 1 - p_dark  # 无光子，无暗计数
                    else:
                        P_H = (1 - eta_det) ** n_H  # 所有光子都被漏掉
                else:  # H有点击
                    if n_H == 0:
                        P_H = p_dark  # 仅暗计数
                    else:
                        P_H = 1 - (1 - eta_det) ** n_H  # 至少一个被探测到

                if d_V == 0:  # V无点击
                    if n_V == 0:
                        P_V = 1 - p_dark
                    else:
                        P_V = (1 - eta_det) ** n_V
                else:  # V有点击
                    if n_V == 0:
                        P_V = p_dark
                    else:
                        P_V = 1 - (1 - eta_det) ** n_V

                E[i, i] = P_H * P_V

            E_list_1517.append(E)
            outcomes.append((d_H, d_V))

    # Kraus算符：M = sqrt(E)（对角，所以逐元素平方根）
    M_list_1517 = []
    for E in E_list_1517:
        M = np.zeros_like(E)
        for i in range(6):
            M[i, i] = np.sqrt(max(0, E[i, i].real))
        M_list_1517.append(M)

    # 嵌入18D bin空间：I_780 ⊗ M_1517
    # 光纤过滤后，780nm是真空，所以只需780上的单位
    I_780 = np.eye(3, dtype=complex)
    M_list_embedded = [np.kron(I_780, M) for M in M_list_1517]

    return M_list_embedded, outcomes


def detection_channel_two_mode(
    eta_det: float = 1.0,
    p_dark: float = 0.0
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    两个输出端口的开/关探测POVM（例如在分束器后）。

    返回用于在两个站点（A和B）探测光子的Kraus算符，
    每个站点有H和V偏振探测器。总共4个探测器，16种结果。

    Kraus算符是张量积：M_A ⊗ M_B
    其中M_A和M_B是单站点探测算符。

    Parameters
    ----------
    eta_det : float
        探测效率（所有探测器相同）
    p_dark : float
        每个探测器的暗计数概率

    Returns
    -------
    Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]
        (Kraus算符 [16 x (324,324)], 结果标签)
        每个结果为 (dA_H, dA_V, dB_H, dB_V)，其中d=0表示无点击，d=1表示有点击

    Notes
    -----
    对于BSM（贝尔态测量），相关的结果有：
        - (1,0,0,1) 或 (0,1,1,0): Psi+ 信号
        - (0,1,0,1) 或 (1,0,1,0): Psi- 信号
        - 其他模式：无成功信号

    Examples
    --------
    >>> K, outcomes = detection_channel_two_mode(eta_det=0.9)
    >>> # K有16个算符，每个对应一种点击模式
    >>> # outcomes[i]给出K[i]的(dA_H, dA_V, dB_H, dB_V)
    """
    # 获取单站点探测算符（4个算符对应4种结果）
    M_single, outcomes_single = detection_povm_single_site(eta_det, p_dark)
    # M_single[i]是18x18，outcomes_single[i]是(d_H, d_V)

    # 为所有16种组合构建张量积
    K_list = []
    outcomes = []

    for iA, (dA_H, dA_V) in enumerate(outcomes_single):
        for iB, (dB_H, dB_V) in enumerate(outcomes_single):
            # 张量积：M_A ⊗ M_B (324 x 324)
            K = np.kron(M_single[iA], M_single[iB])
            K_list.append(K)
            outcomes.append((dA_H, dA_V, dB_H, dB_V))

    return K_list, outcomes


def dephasing_channel(
    p_phi: float,
    dim: int = 3
) -> List[np.ndarray]:
    """
    原子量子比特的纯退相干信道。

        E(rho) = (1 - p_phi) * rho + p_phi * Z * rho * Z

    其中 Z = |0><0| - |1><1| 在 {|0>, |1>} 子空间中翻转相位。

    Parameters
    ----------
    p_phi : float
        退相干概率 (0 <= p_phi <= 1)
    dim : int
        原子子空间的维度（默认：3表示|0>, |1>, |e>）

    Returns
    -------
    List[np.ndarray]
        Kraus算符 [K0, K1]，其中：
        K0 = sqrt(1 - p_phi) * I
        K1 = sqrt(p_phi) * Z

    Examples
    --------
    >>> K = dephasing_channel(p_phi=0.01)
    """
    if not 0 <= p_phi <= 1:
        raise ValueError(f"p_phi必须在[0, 1]内，得到 {p_phi}")

    K0 = np.sqrt(1 - p_phi) * np.eye(dim, dtype=complex)

    K1 = np.zeros((dim, dim), dtype=complex)
    # Z = |0><0| - |1><1| 在 {|0>, |1>, |e>} 基中
    K1[0, 0] = 1.0   # |0><0|
    K1[1, 1] = -1.0  # -|1><1|
    # |e>不受退相干影响
    K1[2, 2] = 1.0 if dim >= 3 else 0
    K1 = np.sqrt(p_phi) * K1

    return [K0, K1]


def dephasing_channel_from_rate(
    gamma_phi: float,
    tau: float,
    dim: int = 3
) -> List[np.ndarray]:
    """
    从连续退相干率导出退相干信道。

    p_phi = 1 - exp(-gamma_phi * tau)

    Parameters
    ----------
    gamma_phi : float
        退相干率（1/时间）
    tau : float
        持续时间
    dim : int
        原子子空间的维度

    Returns
    -------
    List[np.ndarray]
        Kraus算符
    """
    p_phi = 1 - np.exp(-gamma_phi * tau)
    return dephasing_channel(p_phi, dim)


# =============================================================================
# 光纤信道参数（用于真实光纤传输仿真）
# =============================================================================

class FiberChannelParams:
    """
    带随机偏振漂移的光纤信道传输参数。

    此类模拟：
    - 琼斯矩阵偏振漂移（SU(2)随机矩阵）
    - 两臂之间的相位漂移
    - 带小波动的损耗
    - PMD（偏振模色散）

    每条轨迹从分布中采样新的随机参数。

    Parameters
    ----------
    U_mean_A : np.ndarray
        A臂的平均琼斯矩阵（2x2幺正）
    U_mean_B : np.ndarray
        B臂的平均琼斯矩阵（2x2幺正）
    polarization_model : str
        "haar" - 完全随机SU(2)（未补偿光纤）
        "perturb" - 围绕平均值的小随机旋转（补偿光纤）
        "euler" - 随机欧拉角（中等）
    polarization_sigma : float
        "perturb"模型：旋转角的标准差（弧度）
    eta_mean : float
        平均透过率（0到1）
    eta_std : float
        透过率的标准差
    phase_drift_std : float
        两臂之间相位漂移的标准差（弧度）
    pmd_enabled : bool
        是否包含PMD效应
    pmd_delay_bins : int
        PMD延迟，以仓数为单位（整数位移）

    Examples
    --------
    >>> # 未补偿的长光纤
    >>> params = FiberChannelParams(polarization_model="haar")
    >>> # 带小漂移的补偿光纤
    >>> params = FiberChannelParams(polarization_model="perturb", polarization_sigma=0.1)
    >>> # 为一条轨迹采样
    >>> U_A, U_B, eta, phase = params.sample_all(rng)
    """

    def __init__(
        self,
        U_mean_A: np.ndarray = None,
        U_mean_B: np.ndarray = None,
        polarization_model: str = "perturb",
        polarization_sigma: float = 0.1,
        eta_mean: float = 0.6,
        eta_std: float = 0.02,
        phase_drift_std: float = 0.2,
        pmd_enabled: bool = False,
        pmd_delay_bins: int = 0,
    ):
        if U_mean_A is None:
            U_mean_A = np.eye(2, dtype=complex)
        if U_mean_B is None:
            U_mean_B = np.eye(2, dtype=complex)

        self.U_mean_A = np.asarray(U_mean_A, dtype=complex)
        self.U_mean_B = np.asarray(U_mean_B, dtype=complex)
        self.polarization_model = polarization_model
        self.polarization_sigma = polarization_sigma
        self.eta_mean = eta_mean
        self.eta_std = eta_std
        self.phase_drift_std = phase_drift_std
        self.pmd_enabled = pmd_enabled
        self.pmd_delay_bins = pmd_delay_bins

    def sample_jones_A(self, rng: np.random.Generator) -> np.ndarray:
        """为A臂采样琼斯矩阵。"""
        return self._sample_jones(self.U_mean_A, rng)

    def sample_jones_B(self, rng: np.random.Generator) -> np.ndarray:
        """为B臂采样琼斯矩阵。"""
        return self._sample_jones(self.U_mean_B, rng)

    def _sample_jones(self, U_mean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """给定平均矩阵，采样琼斯矩阵。"""
        model = self.polarization_model

        if model == "haar":
            # 从Haar测度采样的完全随机SU(2)
            # 使用四元数参数化
            x = rng.standard_normal(4)
            x = x / np.linalg.norm(x)
            a, b, c, d = x
            U = np.array([
                [a + 1j*b, c + 1j*d],
                [-c + 1j*d, a - 1j*b]
            ], dtype=complex)

        elif model == "perturb":
            # 围绕平均值的小随机旋转
            # 在布洛赫球上生成随机轴
            axis = rng.standard_normal(3)
            axis = axis / np.linalg.norm(axis)

            # 随机旋转角
            delta_theta = rng.normal(0, self.polarization_sigma)

            # 构建旋转：U = U_mean @ exp(i * delta_theta * (axis·sigma/2))
            from scipy.linalg import expm
            sigma = [
                np.array([[0, 1], [1, 0]], dtype=complex),   # sigma_x
                np.array([[0, -1j], [1j, 0]], dtype=complex), # sigma_y
                np.array([[1, 0], [0, -1]], dtype=complex)   # sigma_z
            ]
            generator = sum(a * s for a, s in zip(axis, sigma)) / 2
            delta_U = expm(1j * delta_theta * generator)
            U = U_mean @ delta_U

        elif model == "euler":
            # 随机欧拉角
            # U = R_z(alpha) @ R_y(beta) @ R_z(gamma)
            alpha = rng.uniform(0, 2*np.pi)
            beta = rng.uniform(0, np.pi)
            gamma = rng.uniform(0, 2*np.pi)

            Rz_a = np.array([
                [np.exp(-1j*alpha/2), 0],
                [0, np.exp(1j*alpha/2)]
            ], dtype=complex)
            Ry_b = np.array([
                [np.cos(beta/2), -np.sin(beta/2)],
                [np.sin(beta/2), np.cos(beta/2)]
            ], dtype=complex)
            Rz_g = np.array([
                [np.exp(-1j*gamma/2), 0],
                [0, np.exp(1j*gamma/2)]
            ], dtype=complex)
            U = Rz_a @ Ry_b @ Rz_g

        else:
            raise ValueError(f"未知的 polarization_model: {model}")

        return U

    def sample_eta(self, rng: np.random.Generator) -> float:
        """从截断正态分布采样透过率。"""
        eta = rng.normal(self.eta_mean, self.eta_std)
        return np.clip(eta, 0, 1)

    def sample_phase_drift(self, rng: np.random.Generator) -> float:
        """采样两臂之间的相位漂移（弧度）。"""
        return rng.normal(0, self.phase_drift_std)

    def sample_all(self, rng: np.random.Generator) -> tuple:
        """
        为一条轨迹采样所有参数。

        Returns
        -------
        tuple
            (U_A, U_B, eta, phase_drift)，其中：
            - U_A: A臂的琼斯矩阵（2x2）
            - U_B: B臂的琼斯矩阵（2x2，可能有相位漂移）
            - eta: 透过率（0到1）
            - phase_drift: 两臂之间的相对相位（弧度）
        """
        U_A = self.sample_jones_A(rng)
        U_B = self.sample_jones_B(rng)
        eta = self.sample_eta(rng)
        phase = self.sample_phase_drift(rng)

        # 对B臂应用相位漂移（全局相位影响干涉）
        U_B = np.exp(1j * phase) * U_B

        return U_A, U_B, eta, phase
