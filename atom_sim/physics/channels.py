# -*- coding: utf-8 -*-
"""
非幺正演化的Kraus信道

本模块提供各种量子信道的Kraus算符集：
- 振幅阻尼（光纤损耗）
- 光纤漂移参数（琼斯+损耗）
"""

from typing import List
import numpy as np


def kraus_from_collapse_ops(c_ops: List[np.ndarray], dt_s: float) -> List[np.ndarray]:
    """
    由 collapse 算符列表构造单步 Kraus（一级离散化）。

    给定 Lindblad 形式的 collapse 算符 C_j，本函数构造：
        K0 = I - 0.5 * dt * Σ_j C_j^† C_j
        Kj = sqrt(dt) * C_j

    该构造在小 dt 下保持数值稳定，适合 time-bin 碰撞模型中的逐步抽样。

    Parameters
    ----------
    c_ops : List[np.ndarray]
        collapse 算符列表，每个形状为 (d, d)
    dt_s : float
        单步时间（秒）

    Returns
    -------
    List[np.ndarray]
        Kraus 列表 [K0, K1, ...]
    """
    if dt_s <= 0.0:
        raise ValueError("dt_s 必须 > 0")
    if not c_ops:
        return []

    dim = int(c_ops[0].shape[0])
    if c_ops[0].shape != (dim, dim):
        raise ValueError("collapse 算符必须是方阵")
    for index, op in enumerate(c_ops, start=1):
        if op.shape != (dim, dim):
            raise ValueError(f"collapse 算符第 {index} 个维度不一致: {op.shape} != ({dim}, {dim})")

    sum_cdagger_c = np.zeros((dim, dim), dtype=complex)
    for op in c_ops:
        sum_cdagger_c += op.conj().T @ op

    k0 = np.eye(dim, dtype=complex) - 0.5 * float(dt_s) * sum_cdagger_c
    k_list = [k0]
    factor = np.sqrt(float(dt_s))
    for op in c_ops:
        k_list.append(factor * op)

    # 数值健康检查：一级离散近似下应接近 I。
    completeness = np.zeros((dim, dim), dtype=complex)
    for kraus in k_list:
        completeness += kraus.conj().T @ kraus
    deviation = np.linalg.norm(completeness - np.eye(dim, dtype=complex))
    if deviation > 1e-4:
        raise ValueError(
            f"kraus_from_collapse_ops: 完备性偏差过大 ({deviation:.3e})，"
            "请减小 dt 或检查 collapse 参数量级"
        )

    return k_list


def loss_channel_both_subspaces(
    eta_780: float,
    eta_H_1517: float,
    eta_V_1517: float
) -> List[np.ndarray]:
    """
    5D bin 上的组合损耗信道（单光子截断）。

    用于 QFC/过滤：780nm 通道通常 eta_780=0（100%过滤），
    1517nm 通道具有正常损耗。由于每臂每个 bin 最多 1 光子，
    Kraus 可以直接在 5D 基序上构造：
        |vac>, |H_780>, |V_780>, |H_1517>, |V_1517>

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
        作用于5D bin空间的Kraus算符列表
    """
    # ------------------------------------------------------------------
    # 5D 基序：vac, H_780, V_780, H_1517, V_1517
    # 由于单光子截断，损耗仅把单光子态映射到真空。
    # ------------------------------------------------------------------
    K_list = []

    K0 = np.zeros((5, 5), dtype=complex)
    K0[0, 0] = 1.0
    K0[1, 1] = np.sqrt(eta_780)
    K0[2, 2] = np.sqrt(eta_780)
    K0[3, 3] = np.sqrt(eta_H_1517)
    K0[4, 4] = np.sqrt(eta_V_1517)
    K_list.append(K0)

    # 780 损耗分支
    K_h780 = np.zeros((5, 5), dtype=complex)
    K_h780[0, 1] = np.sqrt(1.0 - eta_780)
    K_list.append(K_h780)

    K_v780 = np.zeros((5, 5), dtype=complex)
    K_v780[0, 2] = np.sqrt(1.0 - eta_780)
    K_list.append(K_v780)

    # 1517 损耗分支
    K_h1517 = np.zeros((5, 5), dtype=complex)
    K_h1517[0, 3] = np.sqrt(1.0 - eta_H_1517)
    K_list.append(K_h1517)

    K_v1517 = np.zeros((5, 5), dtype=complex)
    K_v1517[0, 4] = np.sqrt(1.0 - eta_V_1517)
    K_list.append(K_v1517)

    # 移除全零算符（例如 eta=1 时）
    return [K for K in K_list if np.any(K != 0)]


def loss_channel_1517_single_photon(eta_H: float, eta_V: float) -> List[np.ndarray]:
    """
    1517nm 单光子子空间的损耗信道（3x3）。

    基序：|vac>, |H>, |V>
    """
    K_list = []

    K0 = np.zeros((3, 3), dtype=complex)
    K0[0, 0] = 1.0
    K0[1, 1] = np.sqrt(eta_H)
    K0[2, 2] = np.sqrt(eta_V)
    K_list.append(K0)

    K_H = np.zeros((3, 3), dtype=complex)
    K_H[0, 1] = np.sqrt(1.0 - eta_H)
    K_list.append(K_H)

    K_V = np.zeros((3, 3), dtype=complex)
    K_V[0, 2] = np.sqrt(1.0 - eta_V)
    K_list.append(K_V)

    return [K for K in K_list if np.any(K != 0)]


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

    每条轨迹从分布中采样新的随机参数。

    Parameters
    ----------
    U_mean_A : np.ndarray
        A臂的平均琼斯矩阵（2x2幺正）
    U_mean_B : np.ndarray
        B臂的平均琼斯矩阵（2x2幺正）
    polarization_model : str
        "fixed" - 恒等/固定琼斯矩阵（无随机）
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
    phase_slope_std : float
        相位斜率的标准差（弧度/每个bin），用于模拟频率失配
    phase_jitter_std : float
        单个bin相位噪声的标准差（弧度），用于模拟时间相关相位噪声
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
        phase_slope_std: float = 0.0,
        phase_jitter_std: float = 0.0,
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
        self.phase_slope_std = phase_slope_std
        self.phase_jitter_std = phase_jitter_std

    def sample_jones_A(self, rng: np.random.Generator) -> np.ndarray:
        """为A臂采样琼斯矩阵。"""
        return self._sample_jones(self.U_mean_A, rng)

    def sample_jones_B(self, rng: np.random.Generator) -> np.ndarray:
        """为B臂采样琼斯矩阵。"""
        return self._sample_jones(self.U_mean_B, rng)

    def _sample_jones(self, U_mean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """给定平均矩阵，采样琼斯矩阵。"""
        model = self.polarization_model

        if model == "fixed":
            # 关闭噪声时直接返回均值矩阵
            U = np.array(U_mean, dtype=complex)

        elif model == "haar":
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

    def sample_phase_slope(self, rng: np.random.Generator) -> float:
        """采样相位斜率（弧度/每个bin）。"""
        return rng.normal(0, self.phase_slope_std)

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
