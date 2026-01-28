# -*- coding: utf-8 -*-
"""
双光子探测与Bell态测量

实现：基于POVM的双点击事件抽样与后验态计算。

物理模型：
- 每个时间仓定义桶式探测POVM（含效率与暗计数）
- 通过MPO收缩枚举双点击事件的精确权重分布
- 从该分布抽样得到本次点击记录，并计算对应原子后验态
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
class TwoClickRecord:
    """双点击记录（用于POVM抽样）。"""
    detector_a: str
    detector_b: str
    bin_a: int
    bin_b: int
    weight: float


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


def _get_effect(effects: dict, key: Tuple[str, ...], dim: int) -> np.ndarray:
    if key in effects:
        return effects[key]
    return np.zeros((dim, dim), dtype=complex)


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


def build_detection_effects_6d(
    eta: float,
    p_dark: float = 0.0,
) -> Tuple[dict, dict]:
    """
    构造6D探测 POVM effects（包含暗计数拆分）。

    Returns
    -------
    effects_all : dict
        所有点击记录的 effect（含暗计数）
    effects_true : dict
        不含暗计数的 effect（仅真实点击）
    """
    kraus_list, outcome_detectors, outcome_dark = build_detection_kraus_6d(eta, p_dark)
    return _build_detection_effects(kraus_list, outcome_detectors, outcome_dark)


def run_two_photon_detection(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 0.85,
    window_bins: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    effects_cache: Optional[Tuple[dict, dict]] = None,
    *,
    p_dark: float = 0.0,
) -> TwoPhotonDetectionResult:
    """
    基于POVM枚举双点击事件分布，并抽样得到单次点击记录及其后验原子态。

    物理语义：先计算两次点击事件的精确权重分布，再抽样得到本次点击记录；
    随后用同一个POVM effect 计算该记录对应的原子后验态。
    """
    if rng is None:
        rng = np.random.default_rng()

    if verbose:
        print("\n" + "=" * 60)
        print("双光子探测（POVM抽样）")
        print("=" * 60)

    bin_start = _infer_bin_start(mps)
    bin_dim = mps.d[bin_start]
    if bin_dim != 6:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 6.")

    if effects_cache is None:
        effects_all, _ = build_detection_effects_6d(eta_det, p_dark)
    else:
        effects_all = effects_cache[0] if isinstance(effects_cache, (tuple, list)) else effects_cache

    if verbose:
        print("  Using 6D POVM effects (36x36) - sampling two-click records")

    B_list, Bc_list = _prepare_grouped_mps_pairs(mps)
    grouped_bins = len(B_list) - 1
    if grouped_bins != n_bins:
        raise ValueError(f"n_bins={n_bins} 与分组后bin数量 {grouped_bins} 不一致")

    dim_atom = B_list[0].shape[1]
    if dim_atom != 16:
        raise ValueError(f"Atom pair site dimension {dim_atom} != 16")

    dim_pair = bin_dim * bin_dim
    empty_key = _order_two_port_detectors([])
    E_no = _get_effect(effects_all, empty_key, dim_pair)
    atom_I = np.eye(dim_atom, dtype=complex)
    right_envs = _build_right_envs(B_list, Bc_list, E_no)
    left_envs_id = _build_left_envs(B_list, Bc_list, atom_I, E_no)

    patterns = [
        ("Psi-", ("H1", "V2")),
        ("Psi-", ("V1", "H2")),
        ("Psi+", ("H1", "V1")),
        ("Psi+", ("H2", "V2")),
        ("", ("H1", "H2")),
        ("", ("V1", "V2")),
    ]

    weight_eps = 1e-14
    records: List[TwoClickRecord] = []
    for _, (det_a, det_b) in patterns:
        key_pair = _order_two_port_detectors([det_a, det_b])
        key_a = _order_two_port_detectors([det_a])
        key_b = _order_two_port_detectors([det_b])

        E_pair = _get_effect(effects_all, key_pair, dim_pair)
        E_a = _get_effect(effects_all, key_a, dim_pair)
        E_b = _get_effect(effects_all, key_b, dim_pair)

        records.extend(_collect_same_bin_records(
            B_list, Bc_list, left_envs_id, right_envs,
            E_pair, det_a, det_b, n_bins, weight_eps,
        ))
        records.extend(_collect_diff_bin_records(
            B_list, Bc_list, left_envs_id, right_envs,
            E_a, E_b, E_no, det_a, det_b, n_bins, weight_eps,
        ))
        records.extend(_collect_diff_bin_records(
            B_list, Bc_list, left_envs_id, right_envs,
            E_b, E_a, E_no, det_b, det_a, n_bins, weight_eps,
        ))

    if not records:
        zero_spin = np.zeros((4, 4), dtype=complex)
        return TwoPhotonDetectionResult(
            clicks=[],
            success=False,
            bell_state="",
            spin_state=zero_spin,
        )

    weights = np.array([max(0.0, r.weight) for r in records], dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= weight_eps:
        zero_spin = np.zeros((4, 4), dtype=complex)
        return TwoPhotonDetectionResult(
            clicks=[],
            success=False,
            bell_state="",
            spin_state=zero_spin,
        )

    probs = weights / total_weight
    pick = int(rng.choice(len(records), p=probs))
    record = records[pick]

    clicks = [
        DetectionEvent(
            detector=record.detector_a,
            bin_index=record.bin_a,
            site=_detector_site(record.detector_a, record.bin_a),
        ),
        DetectionEvent(
            detector=record.detector_b,
            bin_index=record.bin_b,
            site=_detector_site(record.detector_b, record.bin_b),
        ),
    ]
    # 窗口仅用于判定成功，抽样仍覆盖所有双点击记录。
    success, bell_state = check_bsm_success(clicks, window_bins=window_bins)
    spin_state = _compute_record_qubit_state(
        B_list=B_list,
        Bc_list=Bc_list,
        right_envs=right_envs,
        op_no=E_no,
        record=record,
        effects_all=effects_all,
        dim_atom=dim_atom,
        dim_pair=dim_pair,
    )

    if verbose:
        print("\n  结果：")
        print("    抽样自双点击分布（条件在两次点击）")
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
    dim_atom = mps.d[0]
    if dim_atom != 4:
        raise ValueError(f"Unexpected atom dimension: {dim_atom}. Expected 4.")

    rho_full = mps.get_reduced_density([site_A, site_B])
    if rho_full.ndim == 4:
        rho_full = rho_full.reshape(dim_atom * dim_atom, dim_atom * dim_atom)

    # 4D原子基顺序：|0>, |1>, |e>, |u>
    # 提取量子比特子空间：|0>, |1>
    # 双原子基序：|i,j> 的扁平索引为 i * dim_atom + j
    qubit_indices = [
        0 * dim_atom + 0,  # |00>
        0 * dim_atom + 1,  # |01>
        1 * dim_atom + 0,  # |10>
        1 * dim_atom + 1,  # |11>
    ]

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
    if len(mps.d) >= 2 and mps.d[0] == 4 and mps.d[1] == 4:
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
    dim_atom = 4
    proj_full = np.zeros((dim_atom * dim_atom, dim_atom * dim_atom), dtype=complex)
    qubit_indices = [
        0 * dim_atom + 0,
        0 * dim_atom + 1,
        1 * dim_atom + 0,
        1 * dim_atom + 1,
    ]
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


def _contract_env_real(env_mid: np.ndarray, env_right: np.ndarray) -> float:
    return float(np.einsum('ij,ij->', env_mid, env_right).real)


def _contract_env_complex(env_mid: np.ndarray, env_right: np.ndarray) -> complex:
    return np.einsum('ij,ij->', env_mid, env_right)


def _detector_site(detector: str, bin_index: int) -> int:
    site_left = 2 + 2 * bin_index
    site_right = 2 + 2 * bin_index + 1
    return site_left if detector in ("H1", "V1") else site_right


def _collect_same_bin_records(
    B_list: List[np.ndarray],
    Bc_list: List[np.ndarray],
    left_envs: List[np.ndarray],
    right_envs: List[np.ndarray],
    op_pair: np.ndarray,
    det_a: str,
    det_b: str,
    n_bins: int,
    weight_eps: float,
) -> List[TwoClickRecord]:
    records = []
    for s in range(1, n_bins + 1):
        env_mid = _apply_env_left(B_list[s], Bc_list[s], op_pair, left_envs[s])
        weight = _contract_env_real(env_mid, right_envs[s + 1])
        if weight > weight_eps:
            records.append(TwoClickRecord(det_a, det_b, s - 1, s - 1, weight))
    return records


def _collect_diff_bin_records(
    B_list: List[np.ndarray],
    Bc_list: List[np.ndarray],
    left_envs: List[np.ndarray],
    right_envs: List[np.ndarray],
    op_first: np.ndarray,
    op_second: np.ndarray,
    op_no: np.ndarray,
    det_first: str,
    det_second: str,
    n_bins: int,
    weight_eps: float,
) -> List[TwoClickRecord]:
    records = []
    for i in range(1, n_bins):
        env_mid = _apply_env_left(B_list[i], Bc_list[i], op_first, left_envs[i])
        j_end = n_bins
        for j in range(i + 1, j_end + 1):
            env_j = _apply_env_left(B_list[j], Bc_list[j], op_second, env_mid)
            weight = _contract_env_real(env_j, right_envs[j + 1])
            if weight > weight_eps:
                records.append(TwoClickRecord(det_first, det_second, i - 1, j - 1, weight))
            if j < j_end:
                env_mid = _apply_env_left(B_list[j], Bc_list[j], op_no, env_mid)
        # restore env_mid not needed beyond loop
    return records


def _contract_record(
    B_list: List[np.ndarray],
    Bc_list: List[np.ndarray],
    left_envs: List[np.ndarray],
    right_envs: List[np.ndarray],
    op_no: np.ndarray,
    record: TwoClickRecord,
    op_pair: np.ndarray,
    op_a: np.ndarray,
    op_b: np.ndarray,
) -> complex:
    if record.bin_a == record.bin_b:
        s = record.bin_a + 1
        env_mid = _apply_env_left(B_list[s], Bc_list[s], op_pair, left_envs[s])
        return _contract_env_complex(env_mid, right_envs[s + 1])

    if record.bin_a < record.bin_b:
        i = record.bin_a + 1
        j = record.bin_b + 1
        op_first = op_a
        op_second = op_b
    else:
        i = record.bin_b + 1
        j = record.bin_a + 1
        op_first = op_b
        op_second = op_a

    env_mid = _apply_env_left(B_list[i], Bc_list[i], op_first, left_envs[i])
    for s in range(i + 1, j):
        env_mid = _apply_env_left(B_list[s], Bc_list[s], op_no, env_mid)
    env_mid = _apply_env_left(B_list[j], Bc_list[j], op_second, env_mid)
    return _contract_env_complex(env_mid, right_envs[j + 1])


def _compute_record_qubit_state(
    B_list: List[np.ndarray],
    Bc_list: List[np.ndarray],
    right_envs: List[np.ndarray],
    op_no: np.ndarray,
    record: TwoClickRecord,
    effects_all: dict,
    dim_atom: int,
    dim_pair: int,
) -> np.ndarray:
    single_dim = int(round(np.sqrt(dim_atom)))
    if single_dim * single_dim != dim_atom:
        raise ValueError(f"Unexpected atom-pair dimension: {dim_atom}")
    qubit_indices = [
        0 * single_dim + 0,
        0 * single_dim + 1,
        1 * single_dim + 0,
        1 * single_dim + 1,
    ]
    op_pair = _get_effect(
        effects_all,
        _order_two_port_detectors([record.detector_a, record.detector_b]),
        dim_pair,
    )
    op_a = _get_effect(
        effects_all,
        _order_two_port_detectors([record.detector_a]),
        dim_pair,
    )
    op_b = _get_effect(
        effects_all,
        _order_two_port_detectors([record.detector_b]),
        dim_pair,
    )

    sigma = np.zeros((4, 4), dtype=complex)
    for i, qi in enumerate(qubit_indices):
        for j, qj in enumerate(qubit_indices):
            atom_op = np.zeros((dim_atom, dim_atom), dtype=complex)
            atom_op[qi, qj] = 1.0
            left_envs = _build_left_envs(B_list, Bc_list, atom_op, op_no)
            sigma[i, j] = _contract_record(
                B_list,
                Bc_list,
                left_envs,
                right_envs,
                op_no,
                record,
                op_pair,
                op_a,
                op_b,
            )
    return sigma


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
    if dim_atom != 16:
        raise ValueError(f"Atom pair site dimension {dim_atom} != 16")

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

    def _sum_same_bin(
        left_envs: List[np.ndarray],
        op_pair: np.ndarray,
    ) -> float:
        total = 0.0
        for s in range(1, n_bins + 1):
            env_mid = _apply_env_left(B_list[s], Bc_list[s], op_pair, left_envs[s])
            weight = _contract_env_real(env_mid, right_envs[s + 1])
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
                weight = _contract_env_real(env_j, right_envs[j + 1])
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
        print("\n  光子统计（全局MPO方法）：")
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
