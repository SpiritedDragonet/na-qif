# -*- coding: utf-8 -*-
"""
非幺正演化的Kraus信道

本模块提供各种量子信道的Kraus算符集：
- 振幅阻尼（光纤损耗）
- 光纤漂移参数（琼斯+损耗）
"""

from typing import List
import numpy as np


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


def loss_channel_1517_raw(eta_H: float, eta_V: float) -> List[np.ndarray]:
    """
    1517nm通信子空间（6D）的振幅阻尼Kraus算符（不嵌入18D）。

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
    return _loss_channel_1517_raw(eta_H, eta_V)


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


# =============================================================================
# 光纤信道参数（用于真实光纤传输仿真）
# =============================================================================

class FiberChannelParams:
    """
    带随机偏振漂移的光纤信道传输参数。

    此类模拟：
    - 琼斯矩阵偏振漂移（SU(2)随机矩阵）
    - 两臂之间的相位漂移
    - 带小波动的损耗（含小幅度PDL）
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
    pdl_sigma : float
        PDL强度（H/V透过率相对差异的标准差，线性）
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
    >>> U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase = params.sample_all(rng)
    """

    def __init__(
        self,
        U_mean_A: np.ndarray = None,
        U_mean_B: np.ndarray = None,
        polarization_model: str = "perturb",
        polarization_sigma: float = 0.1,
        eta_mean: float = 0.6,
        eta_std: float = 0.02,
        pdl_sigma: float = 0.0,
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
        self.pdl_sigma = pdl_sigma
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

    def sample_eta_hv(self, rng: np.random.Generator) -> tuple:
        """
        采样单臂的H/V透过率（包含小PDL）。

        使用相对差异delta：eta_H = eta*(1+delta), eta_V = eta*(1-delta)。
        """
        eta = self.sample_eta(rng)
        # 小PDL：H/V透过率的相对差异
        delta = rng.normal(0.0, self.pdl_sigma)
        delta = np.clip(delta, -0.5, 0.5)
        eta_H = np.clip(eta * (1.0 + delta), 0.0, 1.0)
        eta_V = np.clip(eta * (1.0 - delta), 0.0, 1.0)
        return eta_H, eta_V, delta

    def sample_phase_drift(self, rng: np.random.Generator) -> float:
        """采样两臂之间的相位漂移（弧度）。"""
        return rng.normal(0, self.phase_drift_std)

    def sample_all(self, rng: np.random.Generator) -> tuple:
        """
        为一条轨迹采样所有参数。

        Returns
        -------
        tuple
            (U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase_drift)，其中：
            - U_A: A臂的琼斯矩阵（2x2）
            - U_B: B臂的琼斯矩阵（2x2，可能有相位漂移）
            - eta_H_A/eta_V_A: A臂H/V透过率（0到1）
            - eta_H_B/eta_V_B: B臂H/V透过率（0到1）
            - phase_drift: 两臂之间的相对相位（弧度）
        """
        U_A = self.sample_jones_A(rng)
        U_B = self.sample_jones_B(rng)
        eta_H_A, eta_V_A, _ = self.sample_eta_hv(rng)
        eta_H_B, eta_V_B, _ = self.sample_eta_hv(rng)
        phase = self.sample_phase_drift(rng)

        # 对B臂应用相位漂移（全局相位影响干涉）
        U_B = np.exp(1j * phase) * U_B

        return U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase
