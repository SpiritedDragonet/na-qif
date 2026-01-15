# -*- coding: utf-8 -*-
"""
双光子探测与Bell态测量

实现：逐bin Kraus测量，遍历所有bins，无早停。

物理模型：
- 每个时间仓应用16结果Kraus测量（两端口×两偏振×有无click）
- 记录完整的观测序列（包括no-click）
- BSM成功判据：恰好2次click且满足Bell态模式
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from ..core.mps import MPSState
from ..hilbert.basis import SUBSPACE_780, SUBSPACE_1517


@dataclass
class DetectionEvent:
    """单次探测事件。"""
    detector: str  # "H1", "V1", "H2", "V2"
    bin_index: int
    site: int


@dataclass
class TwoPhotonDetectionResult:
    """双光子探测结果。"""
    clicks: List[DetectionEvent]
    success: bool
    bell_state: str  # "Psi+", "Psi-", "" (if not success)
    spin_state: np.ndarray  # 4x4 density matrix


def build_detection_kraus_6d(eta: float) -> Tuple[List[np.ndarray], List[str]]:
    """
    构造6D探测 Kraus 算符（桶式SNSPD模型）。

    这是优化版本，用于project_to_1517后的6D bin空间。
    比18D版本快9倍（36x36 vs 324x324）。

    物理模型：
    - 每个端口有H/V两个独立的桶式探测器
    - 桶式探测器：不数分辨，破坏性探测
    - 单端口8个Kraus算符（分解以满足完备性）
    - 两端口64个Kraus算符

    完备性：∑ K_μ† K_μ = I 严格满足

    Parameters
    ----------
    eta : float
        探测效率

    Returns
    -------
    kraus_list : List[np.ndarray]
        64个36x36的Kraus算符
    outcome_names : List[str]
        64个结果名称（物理结果可能重复）
    """
    # 1517nm 基：vac=0, H=1, V=2, 2H=3, 2V=4, HV=5
    # 构造8个单端口6x6 Kraus算符（分解以满足完备性）

    # K_00: no click - 保留态但缩幅
    K00_6d = np.diag([
        1.0,                # |vac>: always no click
        np.sqrt(1 - eta),   # |H>: H doesn't click
        np.sqrt(1 - eta),   # |V>: V doesn't click
        (1 - eta),          # |2H>: both H photons don't click
        (1 - eta),          # |2V>: both V photons don't click
        (1 - eta),          # |HV>: neither clicks
    ]).astype(complex)

    # H-only click 分解为3个正交Kraus
    # K_10a: |H> -> |vac>
    K10a_6d = np.zeros((6, 6), dtype=complex)
    K10a_6d[0, 1] = np.sqrt(eta)

    # K_10b: |2H> -> |vac> (桶式：所有H光子被吸收)
    K10b_6d = np.zeros((6, 6), dtype=complex)
    K10b_6d[0, 3] = np.sqrt(1 - (1 - eta)**2)

    # K_10c: |HV> -> |V> (H被吸收，V留下)
    K10c_6d = np.zeros((6, 6), dtype=complex)
    K10c_6d[2, 5] = np.sqrt(eta * (1 - eta))

    # V-only click 分解为3个正交Kraus
    # K_01a: |V> -> |vac>
    K01a_6d = np.zeros((6, 6), dtype=complex)
    K01a_6d[0, 2] = np.sqrt(eta)

    # K_01b: |2V> -> |vac>
    K01b_6d = np.zeros((6, 6), dtype=complex)
    K01b_6d[0, 4] = np.sqrt(1 - (1 - eta)**2)

    # K_01c: |HV> -> |H> (V被吸收，H留下)
    K01c_6d = np.zeros((6, 6), dtype=complex)
    K01c_6d[1, 5] = np.sqrt(eta * (1 - eta))

    # K_11: H+V both click - only from |HV>
    K11_6d = np.zeros((6, 6), dtype=complex)
    K11_6d[0, 5] = eta

    # 单端口Kraus（直接6x6，无需嵌入）
    port_kraus = [K00_6d, K10a_6d, K10b_6d, K10c_6d, K01a_6d, K01b_6d, K01c_6d, K11_6d]
    # 物理结果名称：none, H, H, H, V, V, V, H+V
    port_names = ["none", "H", "H", "H", "V", "V", "V", "H+V"]

    kraus_list = []
    outcome_names = []

    # 构建64个两端口Kraus算符 (36x36)
    for i1, K1 in enumerate(port_kraus):
        for i2, K2 in enumerate(port_kraus):
            K_two = np.kron(K1, K2)
            kraus_list.append(K_two)

            # 从探测器click构建结果名称
            clicks = []
            name1 = port_names[i1]
            name2 = port_names[i2]
            if name1 != "none":
                if "H" in name1:
                    clicks.append("H1")
                if "V" in name1:
                    clicks.append("V1")
            if name2 != "none":
                if "H" in name2:
                    clicks.append("H2")
                if "V" in name2:
                    clicks.append("V2")
            outcome_names.append("+".join(clicks) if clicks else "none")

    return kraus_list, outcome_names


def build_detection_kraus_18d(eta: float) -> Tuple[List[np.ndarray], List[str]]:
    """
    构造探测 Kraus 算符（桶式SNSPD模型）。

    物理模型：
    - 每个端口有H/V两个独立的桶式探测器
    - 桶式探测器：不数分辨，破坏性探测
    - 单端口8个Kraus算符（分解以满足完备性）
    - 两端口64个Kraus算符

    完备性：∑ K_μ† K_μ = I 严格满足

    关键修正：
    - click Kraus必须分解成正交部分，避免不同输入态映射到同一输出态
    - |HV> 在 H-only click 时应映射到 |V>（而非|vac>）
    - |HV> 在 V-only click 时应映射到 |H>（而非|vac>）

    Parameters
    ----------
    eta : float
        探测效率

    Returns
    -------
    kraus_list : List[np.ndarray]
        64个324x324的Kraus算符
    outcome_names : List[str]
        64个结果名称（物理结果可能重复）
    """
    # 1517nm 基：vac=0, H=1, V=2, 2H=3, 2V=4, HV=5
    I_780 = np.eye(3, dtype=complex)

    # 构造8个单端口6x6 Kraus算符（分解以满足完备性）

    # K_00: no click - 保留态但缩幅
    K00_6d = np.diag([
        1.0,                # |vac>: always no click
        np.sqrt(1 - eta),   # |H>: H doesn't click
        np.sqrt(1 - eta),   # |V>: V doesn't click
        (1 - eta),          # |2H>: both H photons don't click
        (1 - eta),          # |2V>: both V photons don't click
        (1 - eta),          # |HV>: neither clicks
    ]).astype(complex)

    # H-only click 分解为3个正交Kraus
    # K_10a: |H> -> |vac>
    K10a_6d = np.zeros((6, 6), dtype=complex)
    K10a_6d[0, 1] = np.sqrt(eta)

    # K_10b: |2H> -> |vac> (桶式：所有H光子被吸收)
    K10b_6d = np.zeros((6, 6), dtype=complex)
    K10b_6d[0, 3] = np.sqrt(1 - (1 - eta)**2)

    # K_10c: |HV> -> |V> (H被吸收，V留下)
    K10c_6d = np.zeros((6, 6), dtype=complex)
    K10c_6d[2, 5] = np.sqrt(eta * (1 - eta))

    # V-only click 分解为3个正交Kraus
    # K_01a: |V> -> |vac>
    K01a_6d = np.zeros((6, 6), dtype=complex)
    K01a_6d[0, 2] = np.sqrt(eta)

    # K_01b: |2V> -> |vac>
    K01b_6d = np.zeros((6, 6), dtype=complex)
    K01b_6d[0, 4] = np.sqrt(1 - (1 - eta)**2)

    # K_01c: |HV> -> |H> (V被吸收，H留下)
    K01c_6d = np.zeros((6, 6), dtype=complex)
    K01c_6d[1, 5] = np.sqrt(eta * (1 - eta))

    # K_11: H+V both click - only from |HV>
    K11_6d = np.zeros((6, 6), dtype=complex)
    K11_6d[0, 5] = eta

    # 嵌入到18D (780nm x 1517nm)
    port_kraus_6d = [K00_6d, K10a_6d, K10b_6d, K10c_6d, K01a_6d, K01b_6d, K01c_6d, K11_6d]
    port_kraus = [np.kron(I_780, K) for K in port_kraus_6d]
    # 物理结果名称：none, H, H, H, V, V, V, H+V
    port_names = ["none", "H", "H", "H", "V", "V", "V", "H+V"]

    kraus_list = []
    outcome_names = []

    # 构建64个两端口Kraus算符 (324x324)
    for i1, K1 in enumerate(port_kraus):
        for i2, K2 in enumerate(port_kraus):
            K_two = np.kron(K1, K2)
            kraus_list.append(K_two)

            # 从探测器click构建结果名称
            clicks = []
            name1 = port_names[i1]
            name2 = port_names[i2]
            if name1 != "none":
                if "H" in name1:
                    clicks.append("H1")
                if "V" in name1:
                    clicks.append("V1")
            if name2 != "none":
                if "H" in name2:
                    clicks.append("H2")
                if "V" in name2:
                    clicks.append("V2")
            outcome_names.append("+".join(clicks) if clicks else "none")

    return kraus_list, outcome_names


def run_two_photon_detection(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 0.85,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> TwoPhotonDetectionResult:
    """
    逐bin Kraus测量（方案1：遍历所有bins，无早停）。

    物理语义：条件在完整观测记录（click + no-click）下的原子后验态。

    自动检测bin维度（6D或18D）并使用相应的Kraus算符。

    Parameters
    ----------
    mps : MPSState
        输入MPS态
    n_bins : int
        时间仓数量
    eta_det : float
        探测效率
    rng : np.random.Generator, optional
        随机数生成器
    verbose : bool
        是否打印详细信息

    Returns
    -------
    TwoPhotonDetectionResult
        包含clicks、BSM成功与否、Bell态、原子自旋态
    """
    if rng is None:
        rng = np.random.default_rng()

    if verbose:
        print("\n" + "=" * 60)
        print("双光子探测（逐bin扫描）")
        print("=" * 60)

    # 检测bin维度并选择相应的Kraus算符
    bin_dim = mps.d[2]  # 第一个bin的维度
    if bin_dim == 6:
        kraus_list, outcome_names = build_detection_kraus_6d(eta_det)
        if verbose:
            print(f"  Using 6D Kraus operators (36x36) - optimized!")
    elif bin_dim == 18:
        kraus_list, outcome_names = build_detection_kraus_18d(eta_det)
        if verbose:
            print(f"  Using 18D Kraus operators (324x324)")
    else:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 6 or 18.")

    mps_work = mps.copy()
    clicks = []

    # 遍历所有bins，不预扫描，不早停
    for n in range(n_bins):
        site_1 = 2 + 2 * n  # A_n
        site_2 = 2 + 2 * n + 1  # B_n

        outcome_idx = mps_work.apply_two_site_kraus(
            site_left=site_1, kraus_ops=kraus_list, rng=rng,
        )

        outcome = outcome_names[outcome_idx]

        if outcome != "none":
            if verbose:
                print(f"  bin {n}: {outcome}")
            for det in outcome.split("+"):
                site = site_1 if det in ["H1", "V1"] else site_2
                clicks.append(DetectionEvent(
                    detector=det, bin_index=n, site=site,
                ))

    success, bell_state = check_bsm_success(clicks)

    # 提取探测后的原子态
    # 所有bins已被测量，直接trace得到原子的后验态
    spin_state = extract_spin_state(mps_work, n_bins)

    if verbose:
        print(f"\n  结果：")
        print(f"    总点击数：{len(clicks)}")
        if clicks:
            print(f"    点击：{[(c.detector, c.bin_index) for c in clicks]}")
        print(f"    BSM成功：{success}")
        if success:
            print(f"    Bell态：{bell_state}")

    return TwoPhotonDetectionResult(
        clicks=clicks,
        success=success,
        bell_state=bell_state,
        spin_state=spin_state,
    )


def extract_spin_state(mps: MPSState, n_bins: int) -> np.ndarray:
    """
    提取双原子自旋密度矩阵（量子比特子空间）。

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量（保留参数以保持接口兼容）

    Returns
    -------
    np.ndarray
        4x4量子比特密度矩阵，基顺序：|00>, |01>, |10>, |11>
    """
    site_A, site_B = 0, 1
    rho_full = mps.get_reduced_density([site_A, site_B])
    if rho_full.ndim == 4:
        rho_full = rho_full.reshape(9, 9)

    # 3D原子基：|e>=0, |0>=1, |1>=2
    # 提取量子比特子空间：|0>, |1> → indices [1, 3] in single atom
    # 双原子：|00>=0, |01>=1, |10>=2, |11>=3
    # 完整9x9基顺序：
    #   |e,e>=0, |e,0>=1, |e,1>=2,
    #   |0,e>=3, |0,0>=4, |0,1>=5,
    #   |1,e>=6, |1,0>=7, |1,1>=8
    qubit_indices = [4, 5, 7, 8]  # |00>, |01>, |10>, |11>

    rho_qubit = np.zeros((4, 4), dtype=complex)
    for i, qi in enumerate(qubit_indices):
        for j, qj in enumerate(qubit_indices):
            rho_qubit[i, j] = rho_full[qi, qj]

    trace = np.trace(rho_qubit)
    if trace > 1e-10:
        rho_qubit = rho_qubit / trace
    return rho_qubit


def check_bsm_success(clicks: List[DetectionEvent]) -> Tuple[bool, str]:
    """
    检查BSM成功。

    判据：恰好2次click，在同一个bin中，且满足Bell态模式。

    Parameters
    ----------
    clicks : List[DetectionEvent]
        探测事件列表

    Returns
    -------
    success : bool
        是否成功
    bell_state : str
        Bell态名称："Psi+", "Psi-", 或 ""（如果不成功）
    """
    if len(clicks) != 2:
        return False, ""

    # 检查两个click是否在同一个bin中
    if clicks[0].bin_index != clicks[1].bin_index:
        return False, ""

    detectors = {clicks[0].detector, clicks[1].detector}

    # Psi-: {H1, V2} or {V1, H2} - 跨端口不同偏振
    if detectors == {"H1", "V2"} or detectors == {"V1", "H2"}:
        return True, "Psi-"

    # Psi+: {H1, V1} or {H2, V2} - 同端口不同偏振
    if detectors == {"H1", "V1"} or detectors == {"H2", "V2"}:
        return True, "Psi+"

    return False, ""


def compute_fidelity_with_bell(spin_state: np.ndarray, target_bell: str) -> float:
    """
    计算与Bell态的保真度。

    Parameters
    ----------
    spin_state : np.ndarray
        4x4密度矩阵
    target_bell : str
        目标Bell态："Phi+", "Phi-", "Psi+", "Psi-"

    Returns
    -------
    float
        保真度 F = <Bell|rho|Bell>
    """
    bell_states = {
        "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),
        "Phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),
        "Psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),
        "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),
    }
    if target_bell not in bell_states:
        raise ValueError(f"未知的Bell态：{target_bell}")
    psi = bell_states[target_bell]
    return float(np.real(psi.conj() @ spin_state @ psi))


def compute_photon_statistics(mps: MPSState, n_bins: int, verbose: bool = False) -> dict:
    """
    计算光子统计（同时计算780nm和1517nm光子）。

    自动检测bin维度（6D或18D）并使用相应的算符。

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量
    verbose : bool
        是否打印详细信息

    Returns
    -------
    dict
        包含 'n_total', 'n_H', 'n_V', 'loss_prob',
        以及 'n_780_H', 'n_780_V', 'n_1517_H', 'n_1517_V'
    """
    from ..hilbert.operators import annihilation_op

    # 检测bin维度
    bin_dim = mps.d[2]  # 第一个bin的维度

    if bin_dim == 6:
        # 6D空间：只有1517nm光子
        J_H_1517 = annihilation_op(SUBSPACE_1517, mode_id=0)
        J_V_1517 = annihilation_op(SUBSPACE_1517, mode_id=1)

        n_780_H = n_780_V = 0.0  # 6D空间中没有780nm光子
        n_1517_H = n_1517_V = 0.0

        for n in range(n_bins):
            for site in [2 + 2 * n, 2 + 2 * n + 1]:
                rho = mps.get_reduced_density([site])
                # 1517nm 光子
                n_1517_H += np.real(np.trace(J_H_1517.conj().T @ J_H_1517 @ rho))
                n_1517_V += np.real(np.trace(J_V_1517.conj().T @ J_V_1517 @ rho))

    elif bin_dim == 18:
        # 18D空间：780nm和1517nm光子都有
        # 1517nm 光子算符（在18D空间中）
        J_H_1517 = annihilation_op(SUBSPACE_1517, mode_id=0)
        J_V_1517 = annihilation_op(SUBSPACE_1517, mode_id=1)
        I_780 = np.eye(3, dtype=complex)
        J_H_1517_18 = np.kron(I_780, J_H_1517)
        J_V_1517_18 = np.kron(I_780, J_V_1517)

        # 780nm 光子算符（在18D空间中）
        J_H_780 = annihilation_op(SUBSPACE_780, mode_id=0)
        J_V_780 = annihilation_op(SUBSPACE_780, mode_id=1)
        I_1517 = np.eye(6, dtype=complex)
        J_H_780_18 = np.kron(J_H_780, I_1517)
        J_V_780_18 = np.kron(J_V_780, I_1517)

        n_780_H = n_780_V = 0.0
        n_1517_H = n_1517_V = 0.0

        for n in range(n_bins):
            for site in [2 + 2 * n, 2 + 2 * n + 1]:
                rho = mps.get_reduced_density([site])
                # 780nm 光子
                n_780_H += np.real(np.trace(J_H_780_18.conj().T @ J_H_780_18 @ rho))
                n_780_V += np.real(np.trace(J_V_780_18.conj().T @ J_V_780_18 @ rho))
                # 1517nm 光子
                n_1517_H += np.real(np.trace(J_H_1517_18.conj().T @ J_H_1517_18 @ rho))
                n_1517_V += np.real(np.trace(J_V_1517_18.conj().T @ J_V_1517_18 @ rho))
    else:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 6 or 18.")

    n_total = n_780_H + n_780_V + n_1517_H + n_1517_V

    stats = {
        'n_total': n_total,
        'n_H': n_780_H + n_1517_H,
        'n_V': n_780_V + n_1517_V,
        'n_780_H': n_780_H,
        'n_780_V': n_780_V,
        'n_780_total': n_780_H + n_780_V,
        'n_1517_H': n_1517_H,
        'n_1517_V': n_1517_V,
        'n_1517_total': n_1517_H + n_1517_V,
        'loss_prob': max(0.0, 2.0 - n_total),
    }

    if verbose:
        print(f"\n  光子统计：")
        print(f"    总期望光子数：{stats['n_total']:.4f}")
        print(f"    780nm: H={stats['n_780_H']:.4f}, V={stats['n_780_V']:.4f}, total={stats['n_780_total']:.4f}")
        print(f"    1517nm: H={stats['n_1517_H']:.4f}, V={stats['n_1517_V']:.4f}, total={stats['n_1517_total']:.4f}")
        print(f"    损耗概率：{stats['loss_prob']:.4f}")

    return stats
