# -*- coding: utf-8 -*-
"""
事件驱动探测仿真（量子跃迁方法）

本模块实现专家文档中描述的物理正确的双光子探测，
遵循量子跃迁/量子轨迹方法。

核心概念：
-------------
1. 探测器alpha在时间仓n的跃迁算符 J_{alpha,n}
2. 首次命中/首次跃迁采样：恰好0、1或2次点击
3. 方法B：基于阈值的累积采样用于时间排序
4. 每次探测事件后MPS坍缩

探测模式（BS + PBS后）：
---------------------------------
- H1: 端口1 H偏振（来自 c_{H,n}）
- V1: 端口1 V偏振（来自 c_{V,n}）
- H2: 端口2 H偏振（来自 d_{H,n}）
- V2: 端口2 V偏振（来自 d_{V,n}）

BSM成功模式（基于BS门相位约定的理论推导）：
--------------------------------------------------
- Psi-: (H1, V2) 或 (V1, H2) - 跨端口不同偏振（反聚束态）
- Psi+: (H1, V1) 或 (H2, V2) - 同端口不同偏振（聚束态）
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np

from ..core.mps import MPSState
from ..hilbert.basis import SUBSPACE_1517


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class DetectionEvent:
    """
    单次探测事件。

    Attributes
    ----------
    detector : str
        探测器标签："H1", "V1", "H2", "V2"
    bin_index : int
        发生点击的时间仓索引
    site : int
        MPS格点索引（内部使用）
    """
    detector: str
    bin_index: int
    site: int


@dataclass
class TwoPhotonDetectionResult:
    """
    双光子探测试验的结果。

    Attributes
    ----------
    clicks : List[DetectionEvent]
        探测事件列表（0、1或2次点击）
    success : bool
        是否找到BSM成功模式
    bell_state : str
        若成功则为"Psi+"或"Psi-"，否则为空字符串
    spin_state : np.ndarray
        探测后的4x4自旋密度矩阵 rho_AB
        基：|00>, |01>, |10>, |11>，其中0=down, 1=up
    spin_amplitudes : np.ndarray
        4D复振幅向量（如果可提取纯态）
    p_click_first : float
        首次点击概率（用于诊断）
    p_click_second : float
        第二次点击概率（用于诊断）
    """
    clicks: List[DetectionEvent]
    success: bool
    bell_state: str
    spin_state: np.ndarray
    spin_amplitudes: Optional[np.ndarray] = None
    p_click_first: float = 0.0
    p_click_second: float = 0.0


# =============================================================================
# 跃迁算符（探测用的湮灭算符）
# =============================================================================

def build_jump_operators_1517() -> dict:
    """
    为1517nm探测模式构造跃迁（湮灭）算符。

    复用 hilbert.operators.annihilation_op 实现，避免重复造轮子。

    Returns
    -------
    dict
        {"H": J_H, "V": J_V} - 每个都是1517nm子空间的6x6矩阵
    """
    from ..hilbert.operators import annihilation_op
    from ..hilbert.basis import SUBSPACE_1517

    J_H = annihilation_op(SUBSPACE_1517, mode_id=0)  # H偏振
    J_V = annihilation_op(SUBSPACE_1517, mode_id=1)  # V偏振

    return {"H": J_H, "V": J_V}


def build_jump_operators_18d() -> dict:
    """
    构造嵌入18D bin空间（780 x 1517）的跃迁算符。

    由于780nm被过滤，我们只探测1517nm光子。
    J = I_780 ⊗ J_1517

    Returns
    -------
    dict
        {"H": J_H, "V": J_V} - 每个都是18x18矩阵
    """
    J_1517 = build_jump_operators_1517()
    I_780 = np.eye(3, dtype=complex)

    J_H_18 = np.kron(I_780, J_1517["H"])
    J_V_18 = np.kron(I_780, J_1517["V"])

    return {"H": J_H_18, "V": J_V_18}


# =============================================================================
# 核心探测算法（方法B：事件驱动量子跃迁）
# =============================================================================

def compute_click_probability(
    mps: MPSState,
    site: int,
    jump_op: np.ndarray,
) -> float:
    """
    计算特定格点和偏振的点击概率。

    对于跃迁算符J（湮灭算符），点击概率为：
    p = <psi| J^dagger J |psi> = <psi| n |psi>

    其中n是该模式的数算符。这给出该模式的期望光子数。

    注意：所有模式的总和可以超过1，因为：
    - 不同模式中可以存在多个光子
    - 同一个光子在H和V模式中都可以有振幅

    Parameters
    ----------
    mps : MPSState
        当前MPS态
    site : int
        格点索引
    jump_op : np.ndarray
        跃迁算符（湮灭算符），形状 (d, d)

    Returns
    -------
    float
        该模式的期望光子数（点击"率"）
    """
    # 获取格点的约化密度矩阵
    rho = mps.get_reduced_density([site])

    # p = Tr(J^dagger J rho) = <n>
    JdJ = jump_op.conj().T @ jump_op
    p = np.real(np.trace(JdJ @ rho))

    return max(0.0, p)  # 确保非负


def compute_all_click_probabilities(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 1.0,
    jump_ops_18d: dict = None,
) -> np.ndarray:
    """
    计算所有仓的4个探测器的点击概率。

    Parameters
    ----------
    mps : MPSState
        当前MPS态（布局：A1, B1, A2, B2, ..., AN, BN, atomA, atomB）
    n_bins : int
        时间仓数量
    eta_det : float
        探测效率
    jump_ops_18d : dict, optional
        预计算的跃迁算符

    Returns
    -------
    np.ndarray
        形状为 (n_bins, 4) - 每个仓的[H1, V1, H2, V2]概率
    """
    if jump_ops_18d is None:
        jump_ops_18d = build_jump_operators_18d()

    J_H = jump_ops_18d["H"]
    J_V = jump_ops_18d["V"]

    probs = np.zeros((n_bins, 4))

    for n in range(n_bins):
        site_1 = 2 * n      # 端口1（来自臂A）
        site_2 = 2 * n + 1  # 端口2（来自臂B）

        # site_1处的H1, V1（端口1）
        probs[n, 0] = eta_det * compute_click_probability(mps, site_1, J_H)  # H1
        probs[n, 1] = eta_det * compute_click_probability(mps, site_1, J_V)  # V1

        # site_2处的H2, V2（端口2）
        probs[n, 2] = eta_det * compute_click_probability(mps, site_2, J_H)  # H2
        probs[n, 3] = eta_det * compute_click_probability(mps, site_2, J_V)  # V2

    return probs


def apply_jump_and_collapse(
    mps: MPSState,
    site: int,
    jump_op: np.ndarray,
) -> Tuple[MPSState, float]:
    """
    将跃迁算符应用于MPS并坍缩态。

    |psi'> = J |psi> / ||J|psi>||

    Parameters
    ----------
    mps : MPSState
        当前MPS态（将被就地修改）
    site : int
        发生探测的格点索引
    jump_op : np.ndarray
        跃迁算符（18x18）

    Returns
    -------
    Tuple[MPSState, float]
        （坍缩的MPS，此跃迁的概率）
    """
    from tenpy.linalg.np_conserved import Array

    # 获取格点张量
    theta = mps._mps.get_theta(site, n=1)  # (vL, p, vR)
    theta_np = theta.to_ndarray()

    d = mps.d[site]
    J = jump_op.reshape(d, d)

    # 应用跃迁：J @ theta（在物理指标上收缩）
    # theta形状：(chiL, d, chiR)
    J_theta = np.einsum('ij,ajb->aib', J, theta_np)

    # 计算模的平方（=概率）
    p = np.linalg.norm(J_theta) ** 2

    if p < 1e-15:
        # 此跃迁概率为零 - 态变为无效
        raise ValueError(f"格点{site}处的跃迁概率为零")

    # 归一化
    J_theta_normalized = J_theta / np.sqrt(p)

    # 写回MPS
    theta_arr = Array.from_ndarray_trivial(J_theta_normalized, labels=['vL', 'p', 'vR'])
    mps._mps.set_B(site, theta_arr, form='Th')
    mps._mps.canonical_form_finite(renormalize=True)

    return mps, p


def sample_first_click_method_b(
    mps: MPSState,
    n_bins: int,
    eta_det: float,
    rng: np.random.Generator,
    jump_ops_18d: dict = None,
    verbose: bool = False,
) -> Tuple[Optional[DetectionEvent], float, np.ndarray]:
    """
    使用方法B（阈值累积）采样首次点击。

    对于双光子态，总期望光子数sum(p_{alpha,n})
    约等于2（每个臂一个光子）。我们将其归一化以获得
    首次点击的概率分布。

    算法：
    1. 计算所有模式的期望光子数
    2. 归一化得到概率分布
    3. 抽取阈值 u ~ U(0,1)
    4. 按时间顺序累积概率 C = sum p_{alpha,n}
    5. 首个C >= u的仓即为点击位置

    Parameters
    ----------
    mps : MPSState
        当前MPS态
    n_bins : int
        时间仓数量
    eta_det : float
        探测效率
    rng : np.random.Generator
        随机数生成器
    jump_ops_18d : dict, optional
        预计算的跃迁算符
    verbose : bool
        是否打印调试信息

    Returns
    -------
    Tuple[Optional[DetectionEvent], float, np.ndarray]
        (event, total_photon_number, all_probs_normalized)
        若无点击（效率损耗）则event为None
    """
    if jump_ops_18d is None:
        jump_ops_18d = build_jump_operators_18d()

    # 计算所有点击概率（期望光子数）
    all_probs = compute_all_click_probabilities(mps, n_bins, eta_det, jump_ops_18d)

    # 总期望光子数（对于双光子态应约为2）
    total_photon_number = all_probs.sum()

    if verbose:
        print(f"  首次点击：total_photon_number = {total_photon_number:.6f}")

    # 归一化得到首次点击的概率分布
    if total_photon_number < 1e-15:
        # 完全没有光子
        return None, 0.0, all_probs

    # 首先决定是否发生点击（基于探测效率）
    # 对于理想探测器（eta=1），若有光子则首次点击应总是发生
    # 概率已经按eta_det缩放
    p_click = min(1.0, total_photon_number)  # 至少一次点击的概率

    u = rng.uniform(0, 1)
    if u > p_click:
        if verbose:
            print(f"  无首次点击（u={u:.4f} > p_click={p_click:.4f}）")
        return None, total_photon_number, all_probs

    # 归一化概率以选择模式
    probs_normalized = all_probs / total_photon_number

    # 抽取模式选择的阈值
    u_mode = rng.uniform(0, 1)

    # 找到累积和首次超过u_mode的仓
    C = 0.0
    detector_labels = ["H1", "V1", "H2", "V2"]

    for n in range(n_bins):
        p_bin = probs_normalized[n].sum()

        if C + p_bin >= u_mode:
            # 此仓有点击 - 按比例选择探测器
            p_in_bin = probs_normalized[n]
            if p_in_bin.sum() < 1e-15:
                C += p_bin
                continue

            p_in_bin_renorm = p_in_bin / p_in_bin.sum()

            det_idx = rng.choice(4, p=p_in_bin_renorm)
            detector = detector_labels[det_idx]

            # 确定格点
            site = 2 * n if det_idx < 2 else 2 * n + 1

            event = DetectionEvent(
                detector=detector,
                bin_index=n,
                site=site,
            )

            if verbose:
                print(f"  首次点击：{detector} 在仓{n}（p={all_probs[n, det_idx]:.6f}）")

            return event, total_photon_number, probs_normalized

        C += p_bin

    # 若概率求和正确则不应到达此处
    return None, total_photon_number, probs_normalized


def sample_second_click_method_b(
    mps: MPSState,
    n_bins: int,
    eta_det: float,
    rng: np.random.Generator,
    first_event: DetectionEvent,
    jump_ops_18d: dict = None,
    verbose: bool = False,
) -> Tuple[Optional[DetectionEvent], float]:
    """
    在首次点击发生后采样第二次点击。

    首次跃迁后，MPS已坍缩为单光子态。
    总期望光子数现在应约为1。

    Parameters
    ----------
    mps : MPSState
        坍缩的MPS（单光子态 + 原子）
    n_bins : int
        时间仓数量
    eta_det : float
        探测效率
    rng : np.random.Generator
        随机数生成器
    first_event : DetectionEvent
        首次探测事件
    jump_ops_18d : dict, optional
        预计算的跃迁算符
    verbose : bool
        是否打印调试信息

    Returns
    -------
    Tuple[Optional[DetectionEvent], float]
        (event, total_photon_number)
        若无第二次点击则event为None
    """
    if jump_ops_18d is None:
        jump_ops_18d = build_jump_operators_18d()

    # 坍缩后重新计算概率
    all_probs = compute_all_click_probabilities(mps, n_bins, eta_det, jump_ops_18d)

    # 总期望光子数（对于单光子态应约为1）
    total_photon_number = all_probs.sum()

    if verbose:
        print(f"  第二次点击：total_photon_number = {total_photon_number:.6f}")

    if total_photon_number < 1e-15:
        return None, 0.0

    # 第二次点击的概率
    p_click = min(1.0, total_photon_number)

    # 为"是否发生点击"抽取随机数
    u = rng.uniform(0, 1)
    if u > p_click:
        if verbose:
            print(f"  无第二次点击（u={u:.4f} > p_click={p_click:.4f}）")
        return None, total_photon_number

    # 归一化并选择模式
    probs_normalized = all_probs / total_photon_number
    u_mode = rng.uniform(0, 1)

    C = 0.0
    detector_labels = ["H1", "V1", "H2", "V2"]

    for n in range(n_bins):
        p_bin = probs_normalized[n].sum()

        if C + p_bin >= u_mode:
            p_in_bin = probs_normalized[n]
            if p_in_bin.sum() < 1e-15:
                C += p_bin
                continue

            p_in_bin_renorm = p_in_bin / p_in_bin.sum()
            det_idx = rng.choice(4, p=p_in_bin_renorm)
            detector = detector_labels[det_idx]

            site = 2 * n if det_idx < 2 else 2 * n + 1

            event = DetectionEvent(
                detector=detector,
                bin_index=n,
                site=site,
            )

            if verbose:
                print(f"  第二次点击：{detector} 在仓{n}（p={all_probs[n, det_idx]:.6f}）")

            return event, total_photon_number

        C += p_bin

    return None, total_photon_number


# =============================================================================
# 主探测函数
# =============================================================================

def run_two_photon_detection(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 0.85,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> TwoPhotonDetectionResult:
    """
    使用量子跃迁方法运行完整的双光子探测。

    这实现了物理正确的探测仿真：
    1. 使用方法B采样首次点击
    2. 用跃迁算符坍缩MPS
    3. 从坍缩态采样第二次点击
    4. 再次坍缩并提取自旋态

    Parameters
    ----------
    mps : MPSState
        BS后的MPS态（布局：atomA, atomB, A1, B1, A2, B2, ...）
    n_bins : int
        时间仓数量
    eta_det : float
        探测效率（典型SNSPD：0.85）
    rng : np.random.Generator, optional
        随机数生成器
    verbose : bool
        是否打印进度

    Returns
    -------
    TwoPhotonDetectionResult
        包括点击、成功状态和自旋态的探测结果
    """
    if rng is None:
        rng = np.random.default_rng()

    if verbose:
        print("\n" + "=" * 60)
        print("双光子探测（量子跃迁方法）")
        print("=" * 60)
        print(f"  eta_det = {eta_det:.3f}")
        print(f"  n_bins = {n_bins}")

    # 构造跃迁算符
    jump_ops = build_jump_operators_18d()

    # 创建MPS副本用于坍缩操作
    mps_work = mps.copy()

    clicks = []

    # --- 首次点击 ---
    first_event, p1_total, _ = sample_first_click_method_b(
        mps=mps_work,
        n_bins=n_bins,
        eta_det=eta_det,
        rng=rng,
        jump_ops_18d=jump_ops,
        verbose=verbose,
    )

    if first_event is None:
        # 无首次点击 - 返回空结果
        spin_state = extract_spin_state(mps_work, n_bins)
        return TwoPhotonDetectionResult(
            clicks=[],
            success=False,
            bell_state="",
            spin_state=spin_state,
            p_click_first=p1_total,
            p_click_second=0.0,
        )

    clicks.append(first_event)

    # 应用跃迁算符并坍缩
    pol1 = "H" if "H" in first_event.detector else "V"
    J1 = jump_ops[pol1]
    mps_work, _ = apply_jump_and_collapse(mps_work, first_event.site, J1)

    # --- 第二次点击 ---
    second_event, p2_total = sample_second_click_method_b(
        mps=mps_work,
        n_bins=n_bins,
        eta_det=eta_det,
        rng=rng,
        first_event=first_event,
        jump_ops_18d=jump_ops,
        verbose=verbose,
    )

    if second_event is not None:
        clicks.append(second_event)

        # 应用第二次跃迁算符（仅用于概率计算）
        pol2 = "H" if "H" in second_event.detector else "V"
        J2 = jump_ops[pol2]
        mps_work, _ = apply_jump_and_collapse(mps_work, second_event.site, J2)

    # 检查BSM成功
    success, bell_state = check_bsm_success(clicks)

    # 提取最终自旋态
    # 对于有两次点击的BSM成功，在原始MPS上使用条件提取
    # 这正确地投影到被探测的光子模式
    if len(clicks) == 2:
        spin_state = extract_conditional_spin_state(
            mps=mps,  # 使用原始MPS，而非坍缩后的
            n_bins=n_bins,
            click1=clicks[0],
            click2=clicks[1],
        )
    else:
        # 对于0或1次点击，使用坍缩后的MPS
        spin_state = extract_spin_state(mps_work, n_bins)

    if verbose:
        print(f"\n  结果：")
        print(f"    点击：{[(c.detector, c.bin_index) for c in clicks]}")
        print(f"    成功：{success}")
        if success:
            print(f"    贝尔态：{bell_state}")

    return TwoPhotonDetectionResult(
        clicks=clicks,
        success=success,
        bell_state=bell_state,
        spin_state=spin_state,
        p_click_first=p1_total,
        p_click_second=p2_total,
    )


# =============================================================================
# 辅助函数
# =============================================================================

def extract_spin_state(mps: MPSState, n_bins: int) -> np.ndarray:
    """
    从MPS提取双原子自旋密度矩阵。

    探测后，原子位于格点2*n_bins和2*n_bins + 1。

    Parameters
    ----------
    mps : MPSState
        探测后的MPS态
    n_bins : int
        时间仓数量

    Returns
    -------
    np.ndarray
        计算基|00>, |01>, |10>, |11>中的4x4密度矩阵
        其中0 = |down>（3D原子中的索引0），1 = |up>（3D原子中的索引1）
    """
    # New layout: atoms are at sites 0 and 1
    site_A = 0
    site_B = 1

    # 获取完整的9x9双原子密度矩阵
    rho_full = mps.get_reduced_density([site_A, site_B])

    # rho_full形状应为(3, 3, 3, 3)或(9, 9)，取决于实现
    # 若需要则重塑为(9, 9)
    if rho_full.ndim == 4:
        rho_full = rho_full.reshape(9, 9)

    # 提取4x4量子比特子空间（每个原子的|0>, |1>，忽略|e>）
    # 完整基：|00>, |01>, |0e>, |10>, |11>, |1e>, |e0>, |e1>, |ee>
    # （行优先：第一个索引是原子A，第二个是原子B）
    # 量子比特基：|00>（索引0），|01>（索引1），|10>（索引3），|11>（索引4）
    qubit_indices = [0, 1, 3, 4]

    rho_qubit = np.zeros((4, 4), dtype=complex)
    for i, qi in enumerate(qubit_indices):
        for j, qj in enumerate(qubit_indices):
            rho_qubit[i, j] = rho_full[qi, qj]

    # 归一化（以防|e>中有布居）
    trace = np.trace(rho_qubit)
    if trace > 1e-10:
        rho_qubit = rho_qubit / trace

    return rho_qubit


def extract_conditional_spin_state(
    mps: MPSState,
    n_bins: int,
    click1: DetectionEvent,
    click2: Optional[DetectionEvent] = None,
) -> np.ndarray:
    """
    提取以探测事件为条件的自旋态。

    当光子在特定模式被探测时，自旋态是光子湮灭后的
    测后态。

    此函数：
    1. 在被探测的模式上应用湮灭（跃迁）算符
    2. 归一化结果态
    3. 提取原子约化密度矩阵

    Parameters
    ----------
    mps : MPSState
        BS后的MPS态（原始，未修改）
    n_bins : int
        时间仓数量
    click1 : DetectionEvent
        首次探测事件
    click2 : DetectionEvent, optional
        第二次探测事件（如果有两次点击）

    Returns
    -------
    np.ndarray
        条件探测的4x4自旋密度矩阵
    """
    from tenpy.linalg.np_conserved import Array

    # 在副本上工作
    mps_cond = mps.copy()

    # 获取跃迁算符
    jump_ops = build_jump_operators_18d()

    def get_jump_op_for_detector(detector: str) -> Tuple[int, np.ndarray]:
        """获取探测器的格点索引和跃迁算符。"""
        pol = "H" if "H" in detector else "V"
        return jump_ops[pol]

    # 对click1应用跃迁算符
    pol1 = "H" if "H" in click1.detector else "V"
    J1 = jump_ops[pol1]

    # 获取格点张量并应用跃迁
    theta1 = mps_cond._mps.get_theta(click1.site, n=1)
    theta1_np = theta1.to_ndarray()
    d1 = theta1_np.shape[1]
    J1_d = J1.reshape(d1, d1)
    J1_theta = np.einsum('ij,ajb->aib', J1_d, theta1_np)

    # 检查模是否非零
    norm1 = np.linalg.norm(J1_theta)
    if norm1 < 1e-15:
        # 此探测模式无振幅 - 返回最大混合态
        return np.eye(4, dtype=complex) / 4

    J1_theta /= norm1
    theta1_arr = Array.from_ndarray_trivial(J1_theta, labels=['vL', 'p', 'vR'])
    mps_cond._mps.set_B(click1.site, theta1_arr, form='Th')

    # 若存在click2则应用其跃迁算符
    if click2 is not None:
        pol2 = "H" if "H" in click2.detector else "V"
        J2 = jump_ops[pol2]

        theta2 = mps_cond._mps.get_theta(click2.site, n=1)
        theta2_np = theta2.to_ndarray()
        d2 = theta2_np.shape[1]
        J2_d = J2.reshape(d2, d2)
        J2_theta = np.einsum('ij,ajb->aib', J2_d, theta2_np)

        norm2 = np.linalg.norm(J2_theta)
        if norm2 < 1e-15:
            return np.eye(4, dtype=complex) / 4

        J2_theta /= norm2
        theta2_arr = Array.from_ndarray_trivial(J2_theta, labels=['vL', 'p', 'vR'])
        mps_cond._mps.set_B(click2.site, theta2_arr, form='Th')

    # 将MPS置于规范形式
    mps_cond._mps.canonical_form_finite(renormalize=True)

    # 提取原子态
    return extract_spin_state(mps_cond, n_bins)


def check_bsm_success(clicks: List[DetectionEvent]) -> Tuple[bool, str]:
    """
    检查探测模式是否指示BSM成功（严格：仅限同一仓）。

    BSM成功要求：
    1. 恰好两次点击
    2. 两次点击在同一时间仓（严格）
    3. 正确的探测器模式：
       - Psi+: (H1, V2) 或 (V1, H2) - 跨端口不同偏振
       - Psi-: (H1, H2) 或 (V1, V2) - 跨端口相同偏振

    Parameters
    ----------
    clicks : List[DetectionEvent]
        探测事件列表

    Returns
    -------
    Tuple[bool, str]
        (success, bell_state)
    """
    return check_bsm_success_coincidence(clicks, coincidence_window=0)


def check_bsm_success_coincidence(
    clicks: List[DetectionEvent],
    coincidence_window: int = 0,
) -> Tuple[bool, str]:
    """
    检查探测模式是否在符合窗口内指示BSM成功。

    在真实实验中，BSM成功要求：
    1. 恰好两次点击
    2. 两次点击在符合窗口内（|bin1 - bin2| <= window）
    3. 正确的探测器模式（基于BS门相位约定的理论推导）：

       理论分析（G = θ*(c_A†c_B - c_A c_B†), θ=π/4）：
       - Psi- 经过BS -> 反聚束态 |H>_A|V>_B - |V>_A|H>_B
         * H1+V2探测 (|H>_A|V>_B)：端口1的H，端口2的V
         * V1+H2探测 (|V>_A|H>_B)：端口1的V，端口2的H
       - Psi+ 经过BS -> 聚束态 |HV>_A - |HV>_B
         * H1+V1探测 (|HV>_A)：端口1同时有H和V
         * H2+V2探测 (|HV>_B)：端口2同时有H和V

       因此：
       - Psi-: (H1, V2) 或 (V1, H2) - 跨端口不同偏振
       - Psi+: (H1, V1) 或 (H2, V2) - 同端口不同偏振

    Parameters
    ----------
    clicks : List[DetectionEvent]
        探测事件列表
    coincidence_window : int
        符合的最大仓分离（默认0 = 仅同一仓）
        典型值：0-3仓（0.2 ns/仓，所以3仓 = 0.6 ns窗口）

    Returns
    -------
    Tuple[bool, str]
        (success, bell_state)
    """
    if len(clicks) != 2:
        return False, ""

    # 检查点击是否在符合窗口内
    bin_diff = abs(clicks[0].bin_index - clicks[1].bin_index)
    if bin_diff > coincidence_window:
        return False, f""  # 时间上相距太远

    detectors = {clicks[0].detector, clicks[1].detector}

    # Psi-模式：跨端口不同偏振（反聚束态）
    if detectors == {"H1", "V2"} or detectors == {"V1", "H2"}:
        return True, "Psi-"

    # Psi+模式：同端口不同偏振（聚束态）
    if detectors == {"H1", "V1"} or detectors == {"H2", "V2"}:
        return True, "Psi+"

    # 跨端口相同偏振（H1, H2）或（V1, V2）不对应Bell态投影
    return False, ""


def compute_photon_statistics(
    mps: MPSState,
    n_bins: int,
    verbose: bool = False,
) -> dict:
    """
    计算MPS态的光子数统计。

    这有助于量化：
    - 拥有0、1或2个光子的概率
    - 损耗概率（真空中的光子）
    - H/V偏振的分布

    Parameters
    ----------
    mps : MPSState
        BS后的MPS态（布局：atomA, atomB, A1, B1, A2, B2, ...）
    n_bins : int
        时间仓数量
    verbose : bool
        是否打印详细统计

    Returns
    -------
    dict
        包含光子统计的字典：
        - 'n_total': 总期望光子数
        - 'p_0photon': 0个光子的概率
        - 'p_1photon': 恰好1个光子的概率
        - 'p_2photon': 2+个光子的概率
        - 'loss_prob': 至少一个光子损耗的概率
        - 'n_H': 期望H偏振光子数
        - 'n_V': 期望V偏振光子数
    """
    # 获取跃迁算符
    jump_ops = build_jump_operators_18d()
    J_H = jump_ops["H"]
    J_V = jump_ops["V"]

    # 计算期望光子数
    n_H_total = 0.0
    n_V_total = 0.0

    for n in range(n_bins):
        # 端口1（格点2+2n）和端口2（格点2+2n+1）- 跳过原子（格点0,1）
        for site in [2 + 2 * n, 2 + 2 * n + 1]:
            rho = mps.get_reduced_density([site])

            # n_H = Tr(J_H^dagger J_H rho)
            JdJ_H = J_H.conj().T @ J_H
            n_H = np.real(np.trace(JdJ_H @ rho))
            n_H_total += n_H

            # n_V = Tr(J_V^dagger J_V rho)
            JdJ_V = J_V.conj().T @ J_V
            n_V = np.real(np.trace(JdJ_V @ rho))
            n_V_total += n_V

    n_total = n_H_total + n_V_total

    # 对于拥有0、1或2个光子的态：
    # - n_total ≈ 0 表示真空（都损耗或从未发射）
    # - n_total ≈ 1 表示一个光子到达，一个损耗
    # - n_total ≈ 2 表示两个光子都到达
    # 我们使用n_total作为概率的代理

    # 损耗概率：少于2个光子到达的概率
    # 这约为：p_loss ≈ 2 - n_total（当n_total <= 2时）
    loss_prob = max(0.0, 2.0 - n_total)

    stats = {
        'n_total': n_total,
        'n_H': n_H_total,
        'n_V': n_V_total,
        'loss_prob': loss_prob,
    }

    if verbose:
        print(f"\n  光子统计：")
        print(f"    总期望光子数：{n_total:.4f}")
        print(f"    H偏振：{n_H_total:.4f}")
        print(f"    V偏振：{n_V_total:.4f}")
        print(f"    损耗概率（< 2光子）：{loss_prob:.4f}")
        print(f"    双光子到达概率：{2 - loss_prob:.4f}")

    return stats


def compute_fidelity_with_bell(spin_state: np.ndarray, target_bell: str) -> float:
    """
    计算自旋态与目标贝尔态的保真度。

    Parameters
    ----------
    spin_state : np.ndarray
        计算基中的4x4密度矩阵
    target_bell : str
        "Psi+", "Psi-", "Phi+"或"Phi-"

    Returns
    -------
    float
        保真度 F = <target|rho|target>
    """
    # 计算基中的贝尔态 |00>, |01>, |10>, |11>
    bell_states = {
        "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),   # (|00> + |11>)/sqrt(2)
        "Phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),  # (|00> - |11>)/sqrt(2)
        "Psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),   # (|01> + |10>)/sqrt(2)
        "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),  # (|01> - |10>)/sqrt(2)
    }

    if target_bell not in bell_states:
        raise ValueError(f"未知的贝尔态：{target_bell}")

    psi = bell_states[target_bell]
    fidelity = np.real(psi.conj() @ spin_state @ psi)

    return float(fidelity)
