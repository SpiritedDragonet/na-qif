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
    spin_state: np.ndarray  # 4x4 qubit-block density matrix (unnormalized)


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

# 条件量在此阈值以下视为无效，避免数值噪声放大。
P_ARRIVE_EPS = 1e-8


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
            entries.append((K_split, det_split, dark_split))
    return entries


def _build_detection_kraus(
    eta: float,
    p_dark: float,
) -> Tuple[List[np.ndarray], List[List[str]], List[List[str]]]:
    port_entries = _build_port_kraus_entries_6d(eta, p_dark)

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
    return _build_detection_kraus(eta, p_dark)


def run_two_photon_detection(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 0.85,
    window_bins: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    kraus_cache: Optional[Tuple[List[np.ndarray], List[List[str]]]] = None,
    *,
    p_dark: float = 0.0,
) -> TwoPhotonDetectionResult:
    """
    逐bin Kraus测量（方案1：遍历所有bins，无早停）。

    物理语义：条件在完整观测记录（click + no-click）下的原子后验态。

    当前仅支持6D bin空间（project_to_1517之后）。
    Kraus概率由两站点约化密度矩阵计算，避免正交中心位置依赖。

    Parameters
    ----------
    mps : MPSState
        输入MPS态
    n_bins : int
        时间仓数量
    eta_det : float
        探测效率
    window_bins : int, optional
        点击时间窗（bin差阈值）。None表示不限制。
    p_dark : float
        每个探测器每个bin的暗计数概率
    kraus_cache : Optional[Tuple[List[np.ndarray], List[List[str]]]]
        预构建的Kraus列表与点击映射（用于同一run内复用以节省算力）。
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

    # 当前仅支持投影到1517后的6D bin空间
    bin_dim = mps.d[2]  # 第一个bin的维度
    if bin_dim != 6:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 6.")
    if kraus_cache is None:
        kraus_list, outcome_detectors, _ = build_detection_kraus_6d(eta_det, p_dark)
    else:
        kraus_list, outcome_detectors = kraus_cache
    if verbose:
        print("  Using 6D Kraus operators (36x36) - optimized!")

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
            rho=rho_AB,
            rng=rng,
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

    success, bell_state = check_bsm_success(clicks, window_bins=window_bins)

    # 提取探测后的原子态
    # 所有bins已被测量，直接trace得到原子的后验态
    mps_work._mps.canonical_form_finite(renormalize=True)
    spin_state, _ = extract_spin_state(mps_work, n_bins)

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


def extract_spin_state(mps: MPSState, n_bins: int) -> Tuple[np.ndarray, float]:
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
    Tuple[np.ndarray, float]
        (rho_qubit, p_qubit)
        rho_qubit: 4x4量子比特子块（未归一化）
        p_qubit: Tr(rho_qubit)，表示留在量子比特子空间的概率
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

    p_qubit = float(np.real(np.trace(rho_qubit)))
    return rho_qubit, p_qubit


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


def _build_photon_number_projectors(bin_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bin_dim == 6:
        pi0 = np.diag([1, 0, 0, 0, 0, 0]).astype(complex)
        pi1 = np.diag([0, 1, 1, 0, 0, 0]).astype(complex)
        pi2 = np.diag([0, 0, 0, 1, 1, 1]).astype(complex)
        return pi0, pi1, pi2
    raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 6.")


def _build_p2_mpo_tensor(pi0: np.ndarray, pi1: np.ndarray, pi2: np.ndarray) -> np.ndarray:
    dim = pi0.shape[0]
    w = np.zeros((3, 3, dim, dim), dtype=complex)
    w[0, 0] = pi0
    w[0, 1] = pi1
    w[0, 2] = pi2
    w[1, 1] = pi0
    w[1, 2] = pi1
    w[2, 2] = pi0
    return w


def _apply_env_left_mpo(
    env_left: np.ndarray,
    B: np.ndarray,
    Bc: np.ndarray,
    mpo_tensor: np.ndarray,
) -> np.ndarray:
    return np.einsum('aij,ipk,jql,abpq->bkl', env_left, B, Bc, mpo_tensor, optimize=True)


def compute_two_photon_arrival_prob(
    mps: MPSState,
    n_bins: int,
    verbose: bool = False,
) -> float:
    """
    计算双光子均到达探测器的概率（总光子数=2）。

    使用P2投影的MPO（bond dimension=3）计算 <P2>。
    """
    # 在当前模型中 <P2> 与 <N(N-1)/2> 等价，但MPO可线性收缩，避免O(N^2)。
    mps._mps.canonical_form_finite(renormalize=True)
    mps._mps.norm = 1.0
    bin_start = _infer_bin_start(mps)
    bin_dim = mps.d[bin_start]
    bin_sites = set(_get_bin_sites(mps, n_bins))
    pi0, pi1, pi2 = _build_photon_number_projectors(bin_dim)
    w_bin = _build_p2_mpo_tensor(pi0, pi1, pi2)
    w_identity_cache: dict[int, np.ndarray] = {}

    env = np.zeros((3, 1, 1), dtype=complex)
    env[0, 0, 0] = 1.0

    for site in range(mps.L):
        B = mps._mps.get_B(site, form='B').to_ndarray()
        Bc = B.conj()
        if site in bin_sites:
            w = w_bin
        else:
            dim = mps.d[site]
            if dim not in w_identity_cache:
                pi0_id = np.eye(dim, dtype=complex)
                pi1_zero = np.zeros((dim, dim), dtype=complex)
                pi2_zero = np.zeros((dim, dim), dtype=complex)
                w_identity_cache[dim] = _build_p2_mpo_tensor(pi0_id, pi1_zero, pi2_zero)
            w = w_identity_cache[dim]
        env = _apply_env_left_mpo(env, B, Bc, w)

    p2 = float(env[2, 0, 0].real)
    if verbose:
        print(f"  两光子到达概率 p_arrive={p2:.6f}")
    return float(max(0.0, p2))


def _build_detection_effects(
    kraus_list: List[np.ndarray],
    outcome_detectors: List[List[str]],
    outcome_dark: List[List[str]],
) -> Tuple[dict, dict]:
    effects_all = {}
    effects_true = {}
    for K, detectors, dark_detectors in zip(kraus_list, outcome_detectors, outcome_dark):
        key = _order_two_port_detectors(detectors)
        K_mat = np.asarray(K)
        E = K_mat.conj().T @ K_mat
        effects_all[key] = effects_all.get(key, 0) + E
        if not dark_detectors:
            effects_true[key] = effects_true.get(key, 0) + E
    return effects_all, effects_true


def _bell_projector_full(target_bell: str) -> np.ndarray:
    bell_states = {
        "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),
        "Phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),
        "Psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),
        "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),
    }
    if target_bell not in bell_states:
        raise ValueError(f"未知的Bell态：{target_bell}")
    psi = bell_states[target_bell]
    proj_qubit = np.outer(psi, psi.conj())
    proj_full = np.zeros((9, 9), dtype=complex)
    qubit_indices = [0, 1, 3, 4]
    for i, qi in enumerate(qubit_indices):
        for j, qj in enumerate(qubit_indices):
            proj_full[qi, qj] = proj_qubit[i, j]
    return proj_full


def _prepare_grouped_mps_pairs(mps: MPSState) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    psi = mps._mps.copy()
    if psi.L % 2 != 0:
        raise ValueError("MPS sites 数量必须为偶数，才能按 (atomA,atomB),(A1,B1),... 分组")
    psi.group_sites(n=2)
    psi.canonical_form_finite(renormalize=True)
    psi.norm = 1.0
    B_list = []
    Bc_list = []
    for i in range(psi.L):
        B = psi.get_B(i, form='B').to_ndarray()
        B_list.append(B)
        Bc_list.append(B.conj())
    return B_list, Bc_list


def _apply_env_left(
    B: np.ndarray,
    Bc: np.ndarray,
    op: np.ndarray,
    env_left: np.ndarray,
) -> np.ndarray:
    return np.einsum('ij,ipk,jql,pq->kl', env_left, B, Bc, op, optimize=True)


def _apply_env_right(
    B: np.ndarray,
    Bc: np.ndarray,
    op: np.ndarray,
    env_right: np.ndarray,
) -> np.ndarray:
    return np.einsum('ipk,jql,pq,kl->ij', B, Bc, op, env_right, optimize=True)


def _build_left_envs(
    B_list: List[np.ndarray],
    Bc_list: List[np.ndarray],
    atom_op: np.ndarray,
    bin_no_op: np.ndarray,
) -> List[np.ndarray]:
    L = len(B_list)
    left_envs = [None] * (L + 1)
    left_envs[0] = np.array([[1.0 + 0.0j]])
    left_envs[1] = _apply_env_left(B_list[0], Bc_list[0], atom_op, left_envs[0])
    for s in range(1, L):
        left_envs[s + 1] = _apply_env_left(B_list[s], Bc_list[s], bin_no_op, left_envs[s])
    return left_envs


def _build_right_envs(
    B_list: List[np.ndarray],
    Bc_list: List[np.ndarray],
    bin_no_op: np.ndarray,
) -> List[np.ndarray]:
    L = len(B_list)
    right_envs = [None] * (L + 1)
    right_envs[L] = np.array([[1.0 + 0.0j]])
    for s in range(L - 1, 0, -1):
        right_envs[s] = _apply_env_right(B_list[s], Bc_list[s], bin_no_op, right_envs[s + 1])
    return right_envs


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
    p_arrive_eps = P_ARRIVE_EPS
    p_arrive = compute_two_photon_arrival_prob(mps, n_bins, verbose=verbose)
    if p_arrive < p_arrive_eps:
        p_arrive = 0.0
    if p_arrive <= p_arrive_eps and p_dark <= 0.0:
        if verbose:
            print(f"  p_arrive<{p_arrive_eps:.1e} 且 p_dark=0，跳过成功事件枚举")
        return SuccessEnumerationResult(
            p_arrive=p_arrive,
            p_success=0.0,
            p_success_true=0.0,
            p_success_false=0.0,
            p_success_given_arrival=0.0,
            fidelity_declared=0.0,
            fidelity_true=0.0,
            fidelity_false=0.0,
        )

    bin_start = _infer_bin_start(mps)
    bin_dim = mps.d[bin_start]
    if bin_dim != 6:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 6.")
    kraus_list, outcome_detectors, outcome_dark = build_detection_kraus_6d(eta_det, p_dark)
    if verbose:
        print("  Using 6D Kraus operators (36x36) - POVM contraction")

    effects_all, effects_true = _build_detection_effects(kraus_list, outcome_detectors, outcome_dark)
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
        if key not in effects_all:
            raise ValueError(f"Missing detection outcome for detectors={list(key)}")

    if verbose:
        print("  POVM准备: 分组MPS并构建环境")
    B_list, Bc_list = _prepare_grouped_mps_pairs(mps)
    grouped_bins = len(B_list) - 1
    if grouped_bins != n_bins:
        raise ValueError(f"n_bins={n_bins} 与分组后bin数量 {grouped_bins} 不一致")

    dim_atom = B_list[0].shape[1]
    if dim_atom != 9:
        raise ValueError(f"Atom pair site dimension {dim_atom} != 9")

    def _get_effect(effects: dict, key: Tuple[str, ...], dim: int) -> np.ndarray:
        if key in effects:
            return effects[key]
        return np.zeros((dim, dim), dtype=complex)

    dim_pair = kraus_list[0].shape[0]
    E_no = _get_effect(effects_all, empty_key, dim_pair)

    atom_I = np.eye(dim_atom, dtype=complex)
    right_envs = _build_right_envs(B_list, Bc_list, E_no)
    left_envs_id = _build_left_envs(B_list, Bc_list, atom_I, E_no)

    bell_projectors = {bell: _bell_projector_full(bell) for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]}
    left_envs_bell = {
        bell: _build_left_envs(B_list, Bc_list, proj, E_no)
        for bell, proj in bell_projectors.items()
    }

    def _contract_env(env_mid: np.ndarray, env_right: np.ndarray) -> float:
        return float(np.einsum('ij,ij->', env_mid, env_right).real)

    def _sum_same_bin(
        left_envs: List[np.ndarray],
        op_pair: np.ndarray,
    ) -> float:
        total = 0.0
        for s in range(1, n_bins + 1):
            env_mid = _apply_env_left(B_list[s], Bc_list[s], op_pair, left_envs[s])
            weight = _contract_env(env_mid, right_envs[s + 1])
            total += weight
        return total

    def _sum_diff_bins(
        left_envs: List[np.ndarray],
        op_a: np.ndarray,
        op_b: np.ndarray,
    ) -> float:
        total = 0.0
        for i in range(1, n_bins):
            env_mid = _apply_env_left(B_list[i], Bc_list[i], op_a, left_envs[i])
            j_end = n_bins
            if window_bins is not None:
                j_end = min(n_bins, i + window_bins)
            for j in range(i + 1, j_end + 1):
                env_j = _apply_env_left(B_list[j], Bc_list[j], op_b, env_mid)
                weight = _contract_env(env_j, right_envs[j + 1])
                total += weight
                if j < j_end:
                    env_mid = _apply_env_left(B_list[j], Bc_list[j], E_no, env_mid)
        return total

    patterns = [
        ("Psi-", ("H1", "V2")),
        ("Psi-", ("V1", "H2")),
        ("Psi+", ("H1", "V1")),
        ("Psi+", ("H2", "V2")),
    ]

    p_success_all = 0.0
    p_success_true = 0.0
    fidelity_weighted_all = 0.0
    fidelity_weighted_true = 0.0

    for idx, (bell_state, (det_a, det_b)) in enumerate(patterns, start=1):
        if verbose:
            print(f"  POVM累加: {bell_state} ({idx}/{len(patterns)})")
        key_pair = _order_two_port_detectors([det_a, det_b])
        key_a = _order_two_port_detectors([det_a])
        key_b = _order_two_port_detectors([det_b])

        E_pair_all = _get_effect(effects_all, key_pair, dim_pair)
        E_pair_true = _get_effect(effects_true, key_pair, dim_pair)
        E_a_all = _get_effect(effects_all, key_a, dim_pair)
        E_b_all = _get_effect(effects_all, key_b, dim_pair)
        E_a_true = _get_effect(effects_true, key_a, dim_pair)
        E_b_true = _get_effect(effects_true, key_b, dim_pair)

        weight_same_all = _sum_same_bin(left_envs_id, E_pair_all)
        weight_same_true = _sum_same_bin(left_envs_id, E_pair_true)

        weight_diff_all = _sum_diff_bins(left_envs_id, E_a_all, E_b_all)
        weight_diff_all += _sum_diff_bins(left_envs_id, E_b_all, E_a_all)
        weight_diff_true = _sum_diff_bins(left_envs_id, E_a_true, E_b_true)
        weight_diff_true += _sum_diff_bins(left_envs_id, E_b_true, E_a_true)

        p_success_all += weight_same_all + weight_diff_all
        p_success_true += weight_same_true + weight_diff_true
        fidelity_weighted_all += _sum_same_bin(left_envs_bell[bell_state], E_pair_all)
        fidelity_weighted_all += _sum_diff_bins(left_envs_bell[bell_state], E_a_all, E_b_all)
        fidelity_weighted_all += _sum_diff_bins(left_envs_bell[bell_state], E_b_all, E_a_all)

        fidelity_weighted_true += _sum_same_bin(left_envs_bell[bell_state], E_pair_true)
        fidelity_weighted_true += _sum_diff_bins(left_envs_bell[bell_state], E_a_true, E_b_true)
        fidelity_weighted_true += _sum_diff_bins(left_envs_bell[bell_state], E_b_true, E_a_true)


    p_success_all = float(max(0.0, p_success_all))
    p_success_true = float(max(0.0, p_success_true))
    p_success_false = float(max(0.0, p_success_all - p_success_true))

    fidelity_declared = (fidelity_weighted_all / p_success_all) if p_success_all > 0 else 0.0
    fidelity_true = (fidelity_weighted_true / p_success_true) if p_success_true > 0 else 0.0
    fidelity_false = (
        (fidelity_weighted_all - fidelity_weighted_true) / p_success_false
        if p_success_false > 0
        else 0.0
    )

    p_success_given_arrival = (p_success_true / p_arrive) if p_arrive > p_arrive_eps else 0.0

    return SuccessEnumerationResult(
        p_arrive=p_arrive,
        p_success=p_success_all,
        p_success_true=p_success_true,
        p_success_false=p_success_false,
        p_success_given_arrival=p_success_given_arrival,
        fidelity_declared=fidelity_declared,
        fidelity_true=fidelity_true,
        fidelity_false=fidelity_false,
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
        'loss_expected': max(0.0, 2.0 - n_total),
    }

    if verbose:
        print(f"\n  光子统计（全局MPO方法）：")
        print(f"    总期望光子数：{stats['n_total']:.4f}")
        print(f"    780nm: H={stats['n_780_H']:.4f}, V={stats['n_780_V']:.4f}, total={stats['n_780_total']:.4f}")
        print(f"    1517nm: H={stats['n_1517_H']:.4f}, V={stats['n_1517_V']:.4f}, total={stats['n_1517_total']:.4f}")
        print(f"    期望损耗光子数：{stats['loss_expected']:.4f}")

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
        包含 'n_total', 'n_H', 'n_V', 'loss_expected'(期望损耗光子数),
        以及 'n_780_H', 'n_780_V', 'n_1517_H', 'n_1517_V'
    """
    # 检测bin维度
    bin_dim = mps.d[2]  # 第一个bin的维度

    # 使用全局MPO方法计算（正确处理强关联态）
    return _compute_photon_statistics_global(mps, n_bins, bin_dim, verbose)
