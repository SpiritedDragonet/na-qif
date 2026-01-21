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
from itertools import product
from typing import Tuple, List, Optional
from dataclasses import dataclass
from collections import Counter

from tenpy.linalg.np_conserved import Array

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


@dataclass
class SuccessEnumerationResult:
    """枚举成功事件的统计结果。"""
    p_arrive: float
    p_success: float
    p_success_true: float
    p_success_false: float
    p_success_given_arrival: float
    fidelity_declared: float
    fidelity_true: float
    fidelity_false: float
    spin_state: np.ndarray  # success条件下的4x4密度矩阵
    spin_state_true: np.ndarray
    spin_state_false: np.ndarray
    bell_weights: Counter
    success_events: List[Tuple[str, int, int, float]]  # (bell, bin1, bin2, weight)


def _order_detectors(detectors: List[str]) -> List[str]:
    order = {"H": 0, "V": 1}
    return sorted(detectors, key=lambda d: order[d])


def _order_two_port_detectors(detectors: List[str]) -> Tuple[str, ...]:
    order = {"H1": 0, "V1": 1, "H2": 2, "V2": 3}
    return tuple(sorted(detectors, key=lambda d: order[d]))


def _split_with_dark(
    kraus: np.ndarray,
    detectors: List[str],
    p_dark: float,
) -> List[Tuple[np.ndarray, List[str], List[str]]]:
    if not 0 <= p_dark <= 1:
        raise ValueError(f"p_dark必须在[0, 1]内，得到 {p_dark}")

    base_detectors = _order_detectors(detectors)
    if p_dark <= 0:
        return [(kraus, base_detectors, [])]

    off_detectors = [d for d in ("H", "V") if d not in base_detectors]
    entries = []
    for mask in product([0, 1], repeat=len(off_detectors)):
        prob = 1.0
        dark_detectors = []
        for det, use_dark in zip(off_detectors, mask):
            if use_dark:
                prob *= p_dark
                dark_detectors.append(det)
            else:
                prob *= (1 - p_dark)
        if prob <= 0:
            continue
        combined_detectors = _order_detectors(base_detectors + dark_detectors)
        entries.append((np.sqrt(prob) * kraus, combined_detectors, _order_detectors(dark_detectors)))
    return entries


def _build_port_kraus_entries_6d(
    eta: float,
    p_dark: float,
    embed_780: bool = False,
) -> List[Tuple[np.ndarray, List[str], List[str]]]:
    # 1517nm 基：vac=0, H=1, V=2, 2H=3, 2V=4, HV=5
    K00_6d = np.diag([
        1.0,                # |vac>: always no click
        np.sqrt(1 - eta),   # |H>: H doesn't click
        np.sqrt(1 - eta),   # |V>: V doesn't click
        (1 - eta),          # |2H>: both H photons don't click
        (1 - eta),          # |2V>: both V photons don't click
        (1 - eta),          # |HV>: neither clicks
    ]).astype(complex)

    K10a_6d = np.zeros((6, 6), dtype=complex)
    K10a_6d[0, 1] = np.sqrt(eta)

    K10b_6d = np.zeros((6, 6), dtype=complex)
    K10b_6d[0, 3] = np.sqrt(1 - (1 - eta) ** 2)

    K10c_6d = np.zeros((6, 6), dtype=complex)
    K10c_6d[2, 5] = np.sqrt(eta * (1 - eta))

    K01a_6d = np.zeros((6, 6), dtype=complex)
    K01a_6d[0, 2] = np.sqrt(eta)

    K01b_6d = np.zeros((6, 6), dtype=complex)
    K01b_6d[0, 4] = np.sqrt(1 - (1 - eta) ** 2)

    K01c_6d = np.zeros((6, 6), dtype=complex)
    K01c_6d[1, 5] = np.sqrt(eta * (1 - eta))

    K11_6d = np.zeros((6, 6), dtype=complex)
    K11_6d[0, 5] = eta

    base_entries = [
        (K00_6d, []),
        (K10a_6d, ["H"]),
        (K10b_6d, ["H"]),
        (K10c_6d, ["H"]),
        (K01a_6d, ["V"]),
        (K01b_6d, ["V"]),
        (K01c_6d, ["V"]),
        (K11_6d, ["H", "V"]),
    ]

    entries = []
    for K, detectors in base_entries:
        for K_split, det_split, dark_split in _split_with_dark(K, detectors, p_dark):
            if embed_780:
                I_780 = np.eye(3, dtype=complex)
                K_split = np.kron(I_780, K_split)
            entries.append((K_split, det_split, dark_split))
    return entries


def _build_detection_kraus(
    eta: float,
    p_dark: float,
    embed_780: bool,
) -> Tuple[List[np.ndarray], List[List[str]], List[List[str]]]:
    port_entries = _build_port_kraus_entries_6d(eta, p_dark, embed_780=embed_780)

    kraus_list = []
    outcome_detectors = []
    outcome_dark = []

    for K1, det1, dark1 in port_entries:
        for K2, det2, dark2 in port_entries:
            K_two = np.kron(K1, K2)

            dets = []
            dark_dets = []
            for det in ("H", "V"):
                if det in det1:
                    dets.append(f"{det}1")
                if det in dark1:
                    dark_dets.append(f"{det}1")
            for det in ("H", "V"):
                if det in det2:
                    dets.append(f"{det}2")
                if det in dark2:
                    dark_dets.append(f"{det}2")

            kraus_list.append(K_two)
            outcome_detectors.append(dets)
            outcome_dark.append(dark_dets)

    return kraus_list, outcome_detectors, outcome_dark


def build_detection_kraus_6d(
    eta: float,
    p_dark: float = 0.0,
) -> Tuple[List[np.ndarray], List[List[str]], List[List[str]]]:
    """
    构造6D探测 Kraus 算符（桶式SNSPD模型）。

    这是优化版本，用于project_to_1517后的6D bin空间。
    比18D版本快9倍（36x36 vs 324x324）。

    物理模型：
    - 每个端口有H/V两个独立的桶式探测器
    - 桶式探测器：不数分辨，破坏性探测
    - 单端口8个Kraus算符（分解以满足完备性）
    - p_dark=0 时，两端口64个Kraus算符；p_dark>0 会进一步拆分

    完备性：∑ K_μ† K_μ = I 严格满足

    Parameters
    ----------
    eta : float
        探测效率
    p_dark : float
        每个探测器每个bin的暗计数概率

    Returns
    -------
    kraus_list : List[np.ndarray]
        两端口Kraus算符列表
    outcome_detectors : List[List[str]]
        每个Kraus对应的探测器点击列表（如 ["H1", "V2"]）
    outcome_dark : List[List[str]]
        每个Kraus对应的暗计数点击列表（为空表示无暗计数）
    """
    return _build_detection_kraus(eta, p_dark, embed_780=False)


def build_detection_kraus_18d(
    eta: float,
    p_dark: float = 0.0,
) -> Tuple[List[np.ndarray], List[List[str]], List[List[str]]]:
    """
    构造探测 Kraus 算符（桶式SNSPD模型）。

    物理模型：
    - 每个端口有H/V两个独立的桶式探测器
    - 桶式探测器：不数分辨，破坏性探测
    - 单端口8个Kraus算符（分解以满足完备性）
    - p_dark=0 时，两端口64个Kraus算符；p_dark>0 会进一步拆分

    完备性：∑ K_μ† K_μ = I 严格满足

    关键修正：
    - click Kraus必须分解成正交部分，避免不同输入态映射到同一输出态
    - |HV> 在 H-only click 时应映射到 |V>（而非|vac>）
    - |HV> 在 V-only click 时应映射到 |H>（而非|vac>）

    Parameters
    ----------
    eta : float
        探测效率
    p_dark : float
        每个探测器每个bin的暗计数概率

    Returns
    -------
    kraus_list : List[np.ndarray]
        两端口Kraus算符列表
    outcome_detectors : List[List[str]]
        每个Kraus对应的探测器点击列表（如 ["H1", "V2"]）
    outcome_dark : List[List[str]]
        每个Kraus对应的暗计数点击列表（为空表示无暗计数）
    """
    return _build_detection_kraus(eta, p_dark, embed_780=True)


def run_two_photon_detection(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 0.85,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    *,
    p_dark: float = 0.0,
) -> TwoPhotonDetectionResult:
    """
    逐bin Kraus测量（方案1：遍历所有bins，无早停）。

    物理语义：条件在完整观测记录（click + no-click）下的原子后验态。

    自动检测bin维度（6D或18D）并使用相应的Kraus算符。
    Kraus概率由两站点约化密度矩阵计算，避免正交中心位置依赖。

    Parameters
    ----------
    mps : MPSState
        输入MPS态
    n_bins : int
        时间仓数量
    eta_det : float
        探测效率
    p_dark : float
        每个探测器每个bin的暗计数概率
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
        kraus_list, outcome_detectors, _ = build_detection_kraus_6d(eta_det, p_dark)
        if verbose:
            print(f"  Using 6D Kraus operators (36x36) - optimized!")
    elif bin_dim == 18:
        kraus_list, outcome_detectors, _ = build_detection_kraus_18d(eta_det, p_dark)
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

        # 先规一化到规范形式，再用rho计算Kraus概率（避免正交中心问题）
        mps_work._mps.canonical_form_finite(renormalize=True)
        rho_AB = mps_work.get_reduced_density([site_1, site_2])

        outcome_idx = mps_work.apply_two_site_kraus(
            site_left=site_1,
            kraus_ops=kraus_list,
            rng=rng,
            probs_from_rho=True,
            rho=rho_AB,
        )

        detectors = outcome_detectors[outcome_idx]

        if detectors:
            if verbose:
                print(f"  bin {n}: {'+'.join(detectors)}")
            for det in detectors:
                site = site_1 if det in ["H1", "V1"] else site_2
                clicks.append(DetectionEvent(
                    detector=det, bin_index=n, site=site,
                ))

    success, bell_state = check_bsm_success(clicks)

    # 提取探测后的原子态
    # 所有bins已被测量，直接trace得到原子的后验态
    mps_work._mps.canonical_form_finite(renormalize=True)
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


def check_bsm_success(
    clicks: List[DetectionEvent],
    window_bins: Optional[int] = None,
) -> Tuple[bool, str]:
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

    if window_bins is not None:
        if abs(clicks[0].bin_index - clicks[1].bin_index) > window_bins:
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


def _infer_bin_start(mps: MPSState) -> int:
    if len(mps.d) >= 2 and mps.d[0] == 3 and mps.d[1] == 3:
        return 2
    return 0


def _get_bin_sites(mps: MPSState, n_bins: int) -> List[int]:
    bin_start = _infer_bin_start(mps)
    sites = []
    for n in range(n_bins):
        site_A = bin_start + 2 * n
        site_B = bin_start + 2 * n + 1
        if site_B >= mps.L:
            raise ValueError(f"n_bins={n_bins} 超出MPS长度 {mps.L}")
        sites.append(site_A)
        sites.append(site_B)
    return sites


def _build_number_ops(bin_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if bin_dim == 6:
        n_vals = np.array([0, 1, 1, 2, 2, 2], dtype=float)
        n_op = np.diag(n_vals)
    elif bin_dim == 18:
        n_780 = np.diag([0, 1, 1]).astype(float)
        n_1517 = np.diag([0, 1, 1, 2, 2, 2]).astype(float)
        n_op = np.kron(n_780, np.eye(6)) + np.kron(np.eye(3), n_1517)
        n_vals = np.diag(n_op).real
    else:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 6 or 18.")

    n2_vals = 0.5 * n_vals * (n_vals - 1.0)
    n2_op = np.diag(n2_vals)
    return n_op, n2_op


def compute_two_photon_arrival_prob(
    mps: MPSState,
    n_bins: int,
    verbose: bool = False,
) -> float:
    """
    计算双光子均到达探测器的概率（总光子数=2）。

    通过计算 <N_total(N_total-1)/2> 得到精确的两光子概率。
    """
    mps._mps.canonical_form_finite(renormalize=True)
    bin_start = _infer_bin_start(mps)
    bin_dim = mps.d[bin_start]
    n_op, n2_op = _build_number_ops(bin_dim)
    n2_pair_op = np.kron(n_op, n_op)

    sites = _get_bin_sites(mps, n_bins)

    p2_local = 0.0
    for site in sites:
        rho = mps.get_reduced_density([site])
        p2_local += np.trace(rho @ n2_op).real

    p2_pair = 0.0
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            site_i = sites[i]
            site_j = sites[j]
            rho_ij = mps.get_reduced_density([site_i, site_j])
            if rho_ij.ndim == 4:
                rho_ij = rho_ij.reshape(bin_dim * bin_dim, bin_dim * bin_dim)
            p2_pair += np.trace(rho_ij @ n2_pair_op).real

    p2 = p2_local + p2_pair
    if verbose:
        print(f"  两光子到达概率 p_arrive={p2:.6f}")
    return float(max(0.0, p2))


def _kraus_thetas_from_mps(
    mps: MPSState,
    site_left: int,
    kraus_ops: List[np.ndarray],
) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    d1, d2 = mps.d[site_left], mps.d[site_left + 1]
    theta = mps._mps.get_theta(site_left, n=2)
    theta_np = theta.to_ndarray()

    probs = np.zeros(len(kraus_ops), dtype=float)
    thetas = [None] * len(kraus_ops)
    for idx, K in enumerate(kraus_ops):
        K_mat = np.asarray(K)
        if K_mat.ndim == 2:
            K_mat = K_mat.reshape(d1 * d2, d1 * d2)
        K_4d = K_mat.reshape(d1, d2, d1, d2)
        K_theta = np.einsum('ijkl,aklb->aijb', K_4d, theta_np)
        p_mu = float(np.linalg.norm(K_theta) ** 2)
        probs[idx] = max(p_mu, 0.0)
        if p_mu > 1e-15:
            thetas[idx] = K_theta / np.sqrt(p_mu)
    return probs, thetas


def _set_two_site_theta(
    mps: MPSState,
    site_left: int,
    theta_selected: np.ndarray,
) -> None:
    theta_arr = Array.from_ndarray_trivial(theta_selected, labels=['vL', 'p0', 'p1', 'vR'])
    theta_combined = theta_arr.combine_legs(
        [['vL', 'p0'], ['p1', 'vR']],
        new_axes=[0, 1],
        qconj=[+1, -1],
    )
    mps._mps.set_svd_theta(
        site_left,
        theta_combined,
        trunc_par={'chi_max': mps.max_bond, 'svd_min': 1e-13},
    )
    mps._mps.norm = 1.0


def enumerate_success_events(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 0.85,
    p_dark: float = 0.0,
    window_bins: Optional[int] = None,
    verbose: bool = False,
) -> SuccessEnumerationResult:
    """
    枚举所有双点击成功事件，计算真实成功率与成功态。

    该函数不做随机采样，仅枚举成功事件集合并严格求和。
    """
    p_arrive = compute_two_photon_arrival_prob(mps, n_bins, verbose=verbose)
    if p_arrive <= 0.0:
        if verbose:
            print("  p_arrive≈0，跳过成功事件枚举")
        zero_spin = np.zeros((4, 4), dtype=complex)
        return SuccessEnumerationResult(
            p_arrive=p_arrive,
            p_success=0.0,
            p_success_true=0.0,
            p_success_false=0.0,
            p_success_given_arrival=0.0,
            fidelity_declared=0.0,
            fidelity_true=0.0,
            fidelity_false=0.0,
            spin_state=zero_spin,
            spin_state_true=zero_spin,
            spin_state_false=zero_spin,
            bell_weights=Counter(),
            success_events=[],
        )

    bin_start = _infer_bin_start(mps)
    bin_dim = mps.d[bin_start]
    if bin_dim == 6:
        kraus_list, outcome_detectors, outcome_dark = build_detection_kraus_6d(eta_det, p_dark)
        if verbose:
            print("  Using 6D Kraus operators (36x36) - deterministic")
    elif bin_dim == 18:
        kraus_list, outcome_detectors, outcome_dark = build_detection_kraus_18d(eta_det, p_dark)
        if verbose:
            print("  Using 18D Kraus operators (324x324) - deterministic")
    else:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 6 or 18.")

    detector_map = {}
    for K, detectors, dark_detectors in zip(kraus_list, outcome_detectors, outcome_dark):
        key = _order_two_port_detectors(detectors)
        detector_map.setdefault(key, []).append((K, dark_detectors))

    entry_cache = {}
    for key, entries in detector_map.items():
        entry_cache[key] = (entries, [K for K, _ in entries])

    empty_key = _order_two_port_detectors([])
    required_keys = [
        empty_key,
        _order_two_port_detectors(["H1"]),
        _order_two_port_detectors(["V1"]),
        _order_two_port_detectors(["H2"]),
        _order_two_port_detectors(["V2"]),
        _order_two_port_detectors(["H1", "V2"]),
        _order_two_port_detectors(["V1", "H2"]),
        _order_two_port_detectors(["H1", "V1"]),
        _order_two_port_detectors(["H2", "V2"]),
    ]
    for key in required_keys:
        if key not in entry_cache:
            raise ValueError(f"Missing detection outcome for detectors={list(key)}")

    mps._mps.canonical_form_finite(renormalize=True)

    bin_sites = [
        (bin_start + 2 * n, bin_start + 2 * n + 1)
        for n in range(n_bins)
    ]

    p_success = 0.0
    p_success_true = 0.0
    p_success_false = 0.0
    rho_accum = np.zeros((4, 4), dtype=complex)
    rho_accum_true = np.zeros((4, 4), dtype=complex)
    rho_accum_false = np.zeros((4, 4), dtype=complex)
    fidelity_weighted = 0.0
    fidelity_true_weighted = 0.0
    fidelity_false_weighted = 0.0
    bell_weights = Counter()
    success_events = []

    def _apply_event(det_by_bin: dict) -> List[Tuple[MPSState, float, bool]]:
        branches = [(mps.copy(), 1.0, False)]
        for n in range(n_bins):
            req_key = det_by_bin.get(n, empty_key)
            entries, ops = entry_cache[req_key]
            site_1, site_2 = bin_sites[n]
            if len(entries) == 1:
                (_, dark_detectors) = entries[0]
                new_branches = []
                for mps_branch, weight, has_dark in branches:
                    probs, thetas = _kraus_thetas_from_mps(mps_branch, site_1, ops)
                    p_mu = probs[0]
                    if p_mu <= 1e-15:
                        continue
                    _set_two_site_theta(mps_branch, site_1, thetas[0])
                    new_branches.append((mps_branch, weight * p_mu, has_dark or bool(dark_detectors)))
                branches = new_branches
            else:
                new_branches = []
                for mps_branch, weight, has_dark in branches:
                    probs, thetas = _kraus_thetas_from_mps(mps_branch, site_1, ops)
                    for (entry, p_mu, theta_selected) in zip(entries, probs, thetas):
                        if p_mu <= 1e-15:
                            continue
                        _, dark_detectors = entry
                        mps_next = mps_branch.copy()
                        _set_two_site_theta(mps_next, site_1, theta_selected)
                        new_branches.append(
                            (mps_next, weight * p_mu, has_dark or bool(dark_detectors))
                        )
                branches = new_branches
            if not branches:
                break
        return branches

    def _accumulate(
        bell_state: str,
        bin1: int,
        bin2: int,
        branches: List[Tuple[MPSState, float, bool]],
    ) -> None:
        nonlocal p_success, p_success_true, p_success_false
        nonlocal rho_accum, rho_accum_true, rho_accum_false
        nonlocal fidelity_weighted, fidelity_true_weighted, fidelity_false_weighted
        for mps_branch, weight, has_dark in branches:
            if weight <= 0:
                continue
            p_success += weight
            bell_weights[bell_state] += weight
            spin_state = extract_spin_state(mps_branch, n_bins)
            rho_accum += weight * spin_state
            fidelity = compute_fidelity_with_bell(spin_state, bell_state)
            fidelity_weighted += weight * fidelity
            success_events.append((bell_state, bin1, bin2, weight))
            if has_dark:
                p_success_false += weight
                rho_accum_false += weight * spin_state
                fidelity_false_weighted += weight * fidelity
            else:
                p_success_true += weight
                rho_accum_true += weight * spin_state
                fidelity_true_weighted += weight * fidelity

    patterns = [
        ("Psi-", ("H1", "V2")),
        ("Psi-", ("V1", "H2")),
        ("Psi+", ("H1", "V1")),
        ("Psi+", ("H2", "V2")),
    ]

    for bell_state, (det_a, det_b) in patterns:
        key_pair = _order_two_port_detectors([det_a, det_b])
        key_a = _order_two_port_detectors([det_a])
        key_b = _order_two_port_detectors([det_b])

        for n in range(n_bins):
            branches = _apply_event({n: key_pair})
            _accumulate(bell_state, n, n, branches)

        for i in range(n_bins - 1):
            j_end = n_bins
            if window_bins is not None:
                j_end = min(n_bins, i + window_bins + 1)
            for j in range(i + 1, j_end):
                branches = _apply_event({i: key_a, j: key_b})
                _accumulate(bell_state, i, j, branches)
                branches = _apply_event({i: key_b, j: key_a})
                _accumulate(bell_state, i, j, branches)

    if p_success > 0:
        rho_success = rho_accum / p_success
        fidelity_declared = fidelity_weighted / p_success
    else:
        rho_success = np.zeros((4, 4), dtype=complex)
        fidelity_declared = 0.0
    if p_success_true > 0:
        rho_success_true = rho_accum_true / p_success_true
        fidelity_true = fidelity_true_weighted / p_success_true
    else:
        rho_success_true = np.zeros((4, 4), dtype=complex)
        fidelity_true = 0.0

    if p_success_false > 0:
        rho_success_false = rho_accum_false / p_success_false
        fidelity_false = fidelity_false_weighted / p_success_false
    else:
        rho_success_false = np.zeros((4, 4), dtype=complex)
        fidelity_false = 0.0

    p_success_given_arrival = (p_success_true / p_arrive) if p_arrive > 0 else 0.0

    return SuccessEnumerationResult(
        p_arrive=p_arrive,
        p_success=p_success,
        p_success_true=p_success_true,
        p_success_false=p_success_false,
        p_success_given_arrival=p_success_given_arrival,
        fidelity_declared=fidelity_declared,
        fidelity_true=fidelity_true,
        fidelity_false=fidelity_false,
        spin_state=rho_success,
        spin_state_true=rho_success_true,
        spin_state_false=rho_success_false,
        bell_weights=bell_weights,
        success_events=success_events,
    )


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
        print(f"    期望损耗光子数：{stats['loss_prob']:.4f}")

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
    # 对于求和算符，期望值 = sum_i <O_i>
    # 其中 <O_i> 是第i个站点上的局域期望值

    total = 0.0
    n_sites = len(mps.d)

    # MPS结构：[atomA, atomB, A1, B1, A2, B2, ..., An, Bn]
    # 前2个站点是原子，后面是bin站点（交替左臂和右臂）
    # 我们需要对所有bin站点求和（跳过原子站点）

    for site_idx in range(2, n_sites):
        # 获取单站点约化密度矩阵
        rho = mps.get_reduced_density([site_idx])

        # 获取站点维度
        dim = mps.d[site_idx]

        # 检查local_op的维度是否与站点维度匹配
        if local_op.shape[0] != dim:
            # 跳过维度不匹配的站点
            continue

        # 如果rho的形状不对，需要reshape
        if len(rho.shape) == 2 and rho.shape[0] == dim and rho.shape[1] == dim:
            # 已经是正确的形状
            pass
        else:
            # 需要reshape
            rho = rho.reshape(dim, dim)

        # 计算期望值：Tr(rho * O)
        expectation = np.trace(np.dot(rho, local_op))
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
        包含 'n_total', 'n_H', 'n_V', 'loss_prob'(期望损耗光子数),
        以及 'n_780_H', 'n_780_V', 'n_1517_H', 'n_1517_V'
    """
    # 检测bin维度
    bin_dim = mps.d[2]  # 第一个bin的维度

    # 使用全局MPO方法计算（正确处理强关联态）
    return _compute_photon_statistics_global(mps, n_bins, bin_dim, verbose)
