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

    # 3D原子基：|0>=0, |1>=1, |e>=2
    # 提取量子比特子空间：|0>, |1> → indices [0, 1] in single atom
    # 双原子：|00>=0, |01>=1, |10>=2, |11>=3
    # 完整9x9基顺序：
    #   |0,0>=0, |0,1>=1, |0,e>=2,
    #   |1,0>=3, |1,1>=4, |1,e>=5,
    #   |e,0>=6, |e,1>=7, |e,e>=8
    qubit_indices = [0, 1, 3, 4]  # |00>, |01>, |10>, |11>

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

    判据：恰好2次click（可在不同bin），且探测器组合满足Bell态模式。

    BSM成功模式：
    - Ψ⁻: {H1, V2} 或 {V1, H2} - 跨端口不同偏振
    - Ψ⁺: {H1, V1} 或 {H2, V2} - 同端口不同偏振

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

    # 注意：两个click可以在不同bin中，不要求同bin！
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


def _compute_photon_statistics_global(mps: MPSState, n_bins: int, bin_dim: int, verbose: bool) -> dict:
    """
    使用全局MPO方法计算光子统计（正确处理强关联态）。

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量
    bin_dim : int
        bin的维度（6或18）
    verbose : bool
        是否打印详细信息

    Returns
    -------
    dict
        光子统计信息
    """
    from ..hilbert.operators import annihilation_op

    # 构建全局光子数算符的MPO
    n_sites = len(mps.d)

    # 为每个站点准备单位算符和光子数算符
    if bin_dim == 6:
        # 6D空间：只有1517nm光子
        J_H_1517 = annihilation_op(SUBSPACE_1517, mode_id=0)
        J_V_1517 = annihilation_op(SUBSPACE_1517, mode_id=1)
        n_H_op = J_H_1517.conj().T @ J_H_1517
        n_V_op = J_V_1517.conj().T @ J_V_1517

        # 构建全局光子数算符MPO
        n_1517_H = _build_sum_mpo(mps, n_bins, n_H_op, is_6d=True)
        n_1517_V = _build_sum_mpo(mps, n_bins, n_V_op, is_6d=True)

        n_780_H = n_780_V = 0.0

    elif bin_dim == 18:
        # 18D空间：780nm和1517nm光子都有
        # 1517nm 光子算符
        J_H_1517 = annihilation_op(SUBSPACE_1517, mode_id=0)
        J_V_1517 = annihilation_op(SUBSPACE_1517, mode_id=1)
        I_780 = np.eye(3, dtype=complex)
        J_H_1517_18 = np.kron(I_780, J_H_1517)
        J_V_1517_18 = np.kron(I_780, J_V_1517)
        n_1517_H_op = J_H_1517_18.conj().T @ J_H_1517_18
        n_1517_V_op = J_V_1517_18.conj().T @ J_V_1517_18

        # 780nm 光子算符
        J_H_780 = annihilation_op(SUBSPACE_780, mode_id=0)
        J_V_780 = annihilation_op(SUBSPACE_780, mode_id=1)
        I_1517 = np.eye(6, dtype=complex)
        J_H_780_18 = np.kron(J_H_780, I_1517)
        J_V_780_18 = np.kron(J_V_780, I_1517)
        n_780_H_op = J_H_780_18.conj().T @ J_H_780_18
        n_780_V_op = J_V_780_18.conj().T @ J_V_780_18

        # 构建全局光子数算符MPO
        n_780_H = _build_sum_mpo(mps, n_bins, n_780_H_op, is_6d=False)
        n_780_V = _build_sum_mpo(mps, n_bins, n_780_V_op, is_6d=False)
        n_1517_H = _build_sum_mpo(mps, n_bins, n_1517_H_op, is_6d=False)
        n_1517_V = _build_sum_mpo(mps, n_bins, n_1517_V_op, is_6d=False)
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
        print(f"\n  光子统计（全局MPO方法）：")
        print(f"    总期望光子数：{stats['n_total']:.4f}")
        print(f"    780nm: H={stats['n_780_H']:.4f}, V={stats['n_780_V']:.4f}, total={stats['n_780_total']:.4f}")
        print(f"    1517nm: H={stats['n_1517_H']:.4f}, V={stats['n_1517_V']:.4f}, total={stats['n_1517_total']:.4f}")
        print(f"    损耗概率：{stats['loss_prob']:.4f}")

    return stats


def _build_sum_mpo(mps: MPSState, n_bins: int, local_op: np.ndarray, is_6d: bool) -> float:
    """
    构建并应用求和MPO：sum_i O_i，其中O_i是第i个bin上的局域算符。

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量
    local_op : np.ndarray
        局域算符（作用在单个bin上）
    is_6d : bool
        是否是6D空间（否则是18D）

    Returns
    -------
    float
        期望值 <sum_i O_i>
    """
    # 直接计算期望值，不构建完整MPO
    # 对于求和算符，期望值 = sum_i <O_i>_global
    # 其中 <O_i>_global 需要用全局态计算

    total = 0.0
    n_sites = len(mps.d)

    # 对每个bin（每个bin有2个站点：左臂和右臂）
    # 修复：使用两站点约化密度矩阵计算bin对的光子数，避免重复计数
    for n in range(n_bins):
        site_A = 2 + 2 * n      # 左臂
        site_B = 2 + 2 * n + 1  # 右臂

        # 获取两站点约化密度矩阵
        rho_AB = mps.get_reduced_density([site_A, site_B])

        # rho_AB 的形状应该是 (d_A * d_B, d_A * d_B)
        dim = local_op.shape[0]
        expected_dim = dim * dim

        # 如果形状不对，需要reshape
        if rho_AB.shape[0] != expected_dim:
            # 可能是 (d_A, d_B, d_A, d_B) 形状，需要reshape
            rho_AB = rho_AB.reshape(expected_dim, expected_dim)

        # 构建两站点算符：O_A ⊗ I_B + I_A ⊗ O_B
        I = np.eye(dim, dtype=complex)
        O_A = np.kron(local_op, I)  # 作用在左臂
        O_B = np.kron(I, local_op)  # 作用在右臂
        O_total = O_A + O_B

        # 计算期望值：Tr(rho * O)
        expectation = np.trace(rho_AB @ O_total)
        total += np.real(expectation)

    return total


def _compute_global_expectation(mps: MPSState, operators: list) -> float:
    """
    计算全局算符的期望值 <psi|O_0 ⊗ O_1 ⊗ ... ⊗ O_n|psi>。

    Parameters
    ----------
    mps : MPSState
        MPS态
    operators : list of np.ndarray
        每个站点上的算符列表

    Returns
    -------
    float
        期望值
    """
    # 使用MPS收缩技术计算期望值
    # <psi|O|psi> = sum_{s,s'} conj(A[s]) * O[s,s'] * A[s']

    n_sites = len(operators)

    # 从左到右收缩
    # L[i] 表示从左边收缩到第i个站点的结果
    L = np.array([[1.0]], dtype=complex)  # 初始：标量1

    for i in range(n_sites):
        # 获取第i个站点的MPS张量
        A_tenpy = mps._mps.get_B(i, form=None)  # TeNPy Array
        A = A_tenpy.to_ndarray()  # 转换为numpy数组: shape (chi_left, d, chi_right)
        O = operators[i]  # shape: (d, d)

        # 收缩：L_new = sum_{s,s'} L * conj(A[:, s, :]) * O[s, s'] * A[:, s', :]
        # L: (chi_L_prev, chi_L_prev)
        # A: (chi_L, d, chi_R)
        # O: (d, d)
        # 结果: (chi_R, chi_R)
        chi_left, d, chi_right = A.shape

        L_new = np.zeros((chi_right, chi_right), dtype=complex)
        for s in range(d):
            for s_prime in range(d):
                # L_new += conj(A[:, s, :]).T @ L @ A[:, s_prime, :] * O[s, s_prime]
                L_new += A[:, s_prime, :].T @ L @ A[:, s, :].conj() * O[s, s_prime]

        L = L_new

    # 最后L应该是一个标量（或1x1矩阵）
    result = np.trace(L)
    return float(np.real(result))


def compute_photon_statistics(mps: MPSState, n_bins: int, verbose: bool = False) -> dict:
    """
    计算光子统计（同时计算780nm和1517nm光子）。

    使用全局MPO方法，正确处理投影后的强关联态。

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
    # 检测bin维度
    bin_dim = mps.d[2]  # 第一个bin的维度

    # 使用全局MPO方法计算（正确处理强关联态）
    return _compute_photon_statistics_global(mps, n_bins, bin_dim, verbose)
