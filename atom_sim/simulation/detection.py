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
from ..physics.gates import qfc_gate
from ..physics.channels import (
    loss_channel_both_subspaces,
    loss_channel_1517_single_photon,
)


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


@dataclass
class DetectionPipelineResult:
    """一次准备、多用途输出：成功枚举 + 抽样结果。"""
    p_arrive: float
    metrics: Optional[SuccessEnumerationResult]
    samples: List[TwoPhotonDetectionResult]

# 条件量在此阈值以下视为无效，避免数值噪声放大。
P_ARRIVE_EPS = 1e-8


def _order_two_port_detectors(detectors: List[str]) -> Tuple[str, ...]:
    order = {"H1": 0, "V1": 1, "H2": 2, "V2": 3}
    return tuple(sorted(detectors, key=lambda d: order[d]))


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
    # ------------------------------------------------------------------
    # 该函数的核心目标：
    #   - 为每个“点击模式”(outcome) 生成 POVM effect E_r
    #   - E_r = sum_mu K_{r,mu}^\dagger K_{r,mu}
    # 其中 K_{r,mu} 是桶式探测器 (on/off) 的 Kraus 分解。
    #
    # 物理含义（单端口）：
    #   - 输入基：|vac>, |H>, |V>, |2H>, |2V>, |HV>
    #   - 输出结果：H click / V click / 双击 / 无点击
    #   - 暗计数：在“本应不点击”的 detector 上独立叠加
    #
    # 最终输出两个字典：
    #   effects_all: 含暗计数的 E_r
    #   effects_true: 仅真实点击 (无暗计数贡献) 的 E_r
    # ------------------------------------------------------------------
    def _order_detectors(detectors: List[str]) -> List[str]:
        order = {"H": 0, "V": 1}
        return sorted(detectors, key=lambda d: order[d])

    def _split_with_dark(
        kraus: np.ndarray,
        detectors: List[str],
        p_dark_local: float,
    ) -> List[Tuple[np.ndarray, List[str], List[str]]]:
        # 将“暗计数”作为独立 Bernoulli 过程附加到 Kraus：
        # - 对于未点击的 detector，按 p_dark_local 拆成“暗点击”分支
        # - 这里等价于在 Kraus 前乘以 sqrt(prob)
        if not 0 <= p_dark_local <= 1:
            raise ValueError(f"p_dark必须在[0, 1]内，得到 {p_dark_local}")

        base_detectors = _order_detectors(detectors)
        if p_dark_local <= 0:
            return [(kraus, base_detectors, [])]

        off_detectors = [d for d in ("H", "V") if d not in base_detectors]
        entries = []
        for mask in product([0, 1], repeat=len(off_detectors)):
            prob = 1.0
            dark_detectors = []
            for det, use_dark in zip(off_detectors, mask):
                if use_dark:
                    prob *= p_dark_local
                    dark_detectors.append(det)
                else:
                    prob *= (1 - p_dark_local)
            if prob <= 0:
                continue
            combined_detectors = _order_detectors(base_detectors + dark_detectors)
            entries.append((np.sqrt(prob) * kraus, combined_detectors, _order_detectors(dark_detectors)))
        return entries

    def _build_port_kraus_entries_6d(
        eta_local: float,
        p_dark_local: float,
    ) -> List[Tuple[np.ndarray, List[str], List[str]]]:
        # 单端口 Kraus 分解：使用“桶式探测器”的最简模型。
        # 注意：这是“测量结果”层面的 Kraus，而非物理光电倍增的细节模型。
        #
        # 1517nm 6D 基：[vac, H, V, 2H, 2V, HV]
        # - K00: 无点击
        # - K10*, K01*：H / V 单击
        # - K11: H 与 V 同时点击
        # 1517nm 基：vac=0, H=1, V=2, 2H=3, 2V=4, HV=5
        K00_6d = np.diag([
            1.0,                # |vac>: always no click
            np.sqrt(1 - eta_local),   # |H>: H doesn't click
            np.sqrt(1 - eta_local),   # |V>: V doesn't click
            (1 - eta_local),          # |2H>: both H photons don't click
            (1 - eta_local),          # |2V>: both V photons don't click
            (1 - eta_local),          # |HV>: neither clicks
        ]).astype(complex)

        K10a_6d = np.zeros((6, 6), dtype=complex)
        K10a_6d[0, 1] = np.sqrt(eta_local)

        K10b_6d = np.zeros((6, 6), dtype=complex)
        K10b_6d[0, 3] = np.sqrt(1 - (1 - eta_local) ** 2)

        K10c_6d = np.zeros((6, 6), dtype=complex)
        K10c_6d[2, 5] = np.sqrt(eta_local * (1 - eta_local))

        K01a_6d = np.zeros((6, 6), dtype=complex)
        K01a_6d[0, 2] = np.sqrt(eta_local)

        K01b_6d = np.zeros((6, 6), dtype=complex)
        K01b_6d[0, 4] = np.sqrt(1 - (1 - eta_local) ** 2)

        K01c_6d = np.zeros((6, 6), dtype=complex)
        K01c_6d[1, 5] = np.sqrt(eta_local * (1 - eta_local))

        K11_6d = np.zeros((6, 6), dtype=complex)
        K11_6d[0, 5] = eta_local

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
            for K_split, det_split, dark_split in _split_with_dark(K, detectors, p_dark_local):
                entries.append((K_split, det_split, dark_split))
        return entries

    port_entries = _build_port_kraus_entries_6d(eta, p_dark)
    kraus_list: List[np.ndarray] = []
    outcome_detectors: List[List[str]] = []
    outcome_dark: List[List[str]] = []

    # ------------------------------------------------------------------
    # 双端口组合：
    # 将 port1 与 port2 的 Kraus 通过张量积合成 (36x36)。
    # 结果 detector 标签用 H1/V1, H2/V2 统一编码。
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 将 Kraus 分支聚合成 effect：E_r = sum K^\dagger K
    # 同时拆出“无暗计数”的 pure-true effect。
    # ------------------------------------------------------------------
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


def _proj_3_from_6() -> np.ndarray:
    """6D -> 3D 投影：取 {vac,H,V}。"""
    P = np.zeros((3, 6), dtype=complex)
    P[0, 0] = 1.0
    P[1, 1] = 1.0
    P[2, 2] = 1.0
    return P


def _embed_5_from_3() -> np.ndarray:
    """3D -> 5D 嵌入：{vac,H,V} -> {vac,H_1517,V_1517}。"""
    P = np.zeros((5, 3), dtype=complex)
    P[0, 0] = 1.0
    P[3, 1] = 1.0
    P[4, 2] = 1.0
    return P


def _project_6d_to_3d(op_6d: np.ndarray) -> np.ndarray:
    """将 36x36 双端口算符投影到 3D×3D (9x9)。"""
    op_6d = np.asarray(op_6d, dtype=complex)
    if op_6d.shape != (36, 36):
        raise ValueError(f"op_6d shape {op_6d.shape} != (36,36)")
    P = _proj_3_from_6()
    Pi = np.kron(P, P)
    return Pi @ op_6d @ Pi.conj().T


def _embed_3d_to_5d(op_3d: np.ndarray) -> np.ndarray:
    """将 9x9 双端口算符嵌入到 5D×5D (25x25)。"""
    op_3d = np.asarray(op_3d, dtype=complex)
    if op_3d.shape != (9, 9):
        raise ValueError(f"op_3d shape {op_3d.shape} != (9,9)")
    P = _embed_5_from_3()
    Pi = np.kron(P, P)
    return Pi @ op_3d @ Pi.conj().T


def _jones_3d(U_2x2: np.ndarray) -> np.ndarray:
    """把 2x2 琼斯矩阵嵌入到 3D：diag(1, U_2x2)。"""
    U = np.asarray(U_2x2, dtype=complex)
    if U.shape != (2, 2):
        raise ValueError(f"Jones matrix shape {U.shape} != (2,2)")
    U3 = np.eye(3, dtype=complex)
    U3[1:, 1:] = U
    return U3


def _apply_unitary_adjoint(effects: dict, U: np.ndarray) -> dict:
    """对所有 effect 做 E <- U^† E U。"""
    if not effects:
        return {}
    U = np.asarray(U, dtype=complex)
    U_dag = U.conj().T
    return {k: U_dag @ E @ U for k, E in effects.items()}


def _apply_local_channel_adjoint(
    effects: dict,
    K_list_A: List[np.ndarray],
    K_list_B: List[np.ndarray],
) -> dict:
    """
    将局域信道 (A,B) 的对偶映射作用到所有 effect：

      E' = sum_{mu,nu} (K_A^mu ⊗ K_B^nu)^\dagger E (K_A^mu ⊗ K_B^nu)

    注意：该函数仅做线性变换，不做归一化或截断。
    """
    if not effects:
        return {}
    K_pairs = [np.kron(KA, KB) for KA in K_list_A for KB in K_list_B]
    new_effects = {}
    for key, E in effects.items():
        acc = np.zeros_like(E)
        for K in K_pairs:
            acc += K.conj().T @ E @ K
        new_effects[key] = acc
    return new_effects


def _apply_channel_adjoint_single(op: np.ndarray, K_list: List[np.ndarray]) -> np.ndarray:
    """单端口对偶映射：E <- sum K^† E K。"""
    acc = np.zeros_like(op)
    for K in K_list:
        acc += K.conj().T @ op @ K
    return acc


def _build_arrival_projectors_5d(
    theta_H: float,
    theta_V: float,
    eta_H_A: float,
    eta_V_A: float,
    eta_H_B: float,
    eta_V_B: float,
    apply_filter_780: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    构造用于 p_arrive 统计的 (pi0, pi1, pi2)（5D bin）。

    逻辑：
      - 在 3D telecom 空间上定义 {0/1/2} 光子投影
      - 推入光纤损耗对偶（单端口）
      - 嵌入到 5D 账本
      - 推入 780 过滤与 QFC 对偶（单端口）
    """
    pi0_3 = np.diag([1, 0, 0]).astype(complex)
    pi1_3 = np.diag([0, 1, 1]).astype(complex)
    pi2_3 = np.zeros((3, 3), dtype=complex)

    K_filter = None
    if apply_filter_780:
        K_filter = loss_channel_both_subspaces(
            eta_780=0.0,
            eta_H_1517=1.0,
            eta_V_1517=1.0,
        )

    U_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)
    U_qfc_dag = U_qfc.conj().T
    P_5_from_3 = _embed_5_from_3()

    def _build_one_arm(eta_H: float, eta_V: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 光纤损耗（单光子 3D）
        K_loss = loss_channel_1517_single_photon(eta_H, eta_V)
        p0 = _apply_channel_adjoint_single(pi0_3, K_loss)
        p1 = _apply_channel_adjoint_single(pi1_3, K_loss)
        p2 = _apply_channel_adjoint_single(pi2_3, K_loss)

        # 嵌入到 5D 账本
        p0 = P_5_from_3 @ p0 @ P_5_from_3.conj().T
        p1 = P_5_from_3 @ p1 @ P_5_from_3.conj().T
        p2 = P_5_from_3 @ p2 @ P_5_from_3.conj().T

        # 780 过滤对偶
        if K_filter is not None:
            p0 = _apply_channel_adjoint_single(p0, K_filter)
            p1 = _apply_channel_adjoint_single(p1, K_filter)
            p2 = _apply_channel_adjoint_single(p2, K_filter)

        # QFC 对偶
        p0 = U_qfc_dag @ p0 @ U_qfc
        p1 = U_qfc_dag @ p1 @ U_qfc
        p2 = U_qfc_dag @ p2 @ U_qfc

        return p0, p1, p2

    proj_A = _build_one_arm(eta_H_A, eta_V_A)
    proj_B = _build_one_arm(eta_H_B, eta_V_B)
    return proj_A, proj_B


def run_detection_pipeline(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 0.85,
    p_dark: float = 0.0,
    window_bins: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    n_samples: int = 1,
    compute_metrics: bool = False,
    bs_unitary: Optional[np.ndarray] = None,
    fiber_sample: Optional[tuple] = None,
    apply_filter_780: bool = True,
    theta_H: float = np.pi / 4,
    theta_V: float = np.pi / 4,
) -> DetectionPipelineResult:
    """
    POVM探测流水线：单次准备即可同时枚举成功率与抽样双点击。

    若提供 bs_unitary，则在测量端使用 U^† E U 处理点击 POVM，
    等价于不对态显式作用 BS（Heisenberg 绘景）。

    5D 方案下，QFC/过滤/光纤/BS 全部推入 POVM 对偶映射。
    """
    if rng is None:
        rng = np.random.default_rng()

    # 约定：若链前两个站点为原子(4D)，bin 从 index=2 开始；
    # 否则认为整个链都是 bin。
    bin_start = 2 if (len(mps.d) >= 2 and mps.d[0] == 4 and mps.d[1] == 4) else 0
    bin_dim = mps.d[bin_start]
    if bin_dim != 5:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 5.")

    # ------------------------------------------------------------------
    # bs_unitary 必须是 6D 两端口 (36x36)，用于测量端共轭。
    # ------------------------------------------------------------------
    bs_unitary_6d = None
    if bs_unitary is not None:
        bs_unitary = np.asarray(bs_unitary, dtype=complex)
        if bs_unitary.shape != (36, 36):
            raise ValueError(
                f"bs_unitary shape {bs_unitary.shape} != (36,36) for 6D output ports"
            )
        bs_unitary_6d = bs_unitary

    if verbose and n_samples > 0:
        print("\n" + "=" * 60)
        print("双光子探测（POVM抽样）")
        print("=" * 60)

    arrival_verbose = verbose and compute_metrics
    # ------------------------------------------------------------------
    # 两光子到达概率 p_arrive：
    # 用 MPO 收缩实现对“总光子数=2”的投影。
    # 形式上相当于：
    #   p_arrive = Tr[ Π_{n}(P0 + P1 + P2) ρ ] 中选出 2 光子总数项
    # 这里用小维度的“计数器” MPO (3x3) 来追踪 0/1/2 光子数。
    # ------------------------------------------------------------------
    mps._mps.canonical_form_finite(renormalize=True)
    mps._mps.norm = 1.0
    bin_sites = set()
    for n in range(n_bins):
        site_A = bin_start + 2 * n
        site_B = bin_start + 2 * n + 1
        if site_B >= mps.L:
            raise ValueError(f"n_bins={n_bins} 超出MPS长度 {mps.L}")
        bin_sites.add(site_A)
        bin_sites.add(site_B)

    # 构造 p_arrive 的 0/1/2 光子投影（已推入 QFC/过滤/损耗对偶）
    if fiber_sample is None:
        U_A = np.eye(2, dtype=complex)
        U_B = np.eye(2, dtype=complex)
        eta_H_A = eta_V_A = eta_H_B = eta_V_B = 1.0
        phase_slope = 0.0
        phase_jitter_std = 0.0
    else:
        try:
            U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, _phase, phase_slope, phase_jitter_std = fiber_sample
        except ValueError as exc:
            raise ValueError("fiber_sample 格式不正确，无法解析光纤参数") from exc

    (pi0_A, pi1_A, pi2_A), (pi0_B, pi1_B, pi2_B) = _build_arrival_projectors_5d(
        theta_H=theta_H,
        theta_V=theta_V,
        eta_H_A=float(eta_H_A),
        eta_V_A=float(eta_V_A),
        eta_H_B=float(eta_H_B),
        eta_V_B=float(eta_V_B),
        apply_filter_780=apply_filter_780,
    )

    w_bin_A = np.zeros((3, 3, bin_dim, bin_dim), dtype=complex)
    w_bin_A[0, 0] = pi0_A
    w_bin_A[0, 1] = pi1_A
    w_bin_A[0, 2] = pi2_A
    w_bin_A[1, 1] = pi0_A
    w_bin_A[1, 2] = pi1_A
    w_bin_A[2, 2] = pi0_A

    w_bin_B = np.zeros((3, 3, bin_dim, bin_dim), dtype=complex)
    w_bin_B[0, 0] = pi0_B
    w_bin_B[0, 1] = pi1_B
    w_bin_B[0, 2] = pi2_B
    w_bin_B[1, 1] = pi0_B
    w_bin_B[1, 2] = pi1_B
    w_bin_B[2, 2] = pi0_B

    w_identity_cache: dict[int, np.ndarray] = {}
    env = np.zeros((3, 1, 1), dtype=complex)
    env[0, 0, 0] = 1.0
    for site in range(mps.L):
        B = mps._mps.get_B(site, form='B').to_ndarray()
        Bc = B.conj()
        if site in bin_sites:
            is_A = ((site - bin_start) % 2 == 0)
            w = w_bin_A if is_A else w_bin_B
        else:
            dim = mps.d[site]
            if dim not in w_identity_cache:
                pi0_id = np.eye(dim, dtype=complex)
                pi1_zero = np.zeros((dim, dim), dtype=complex)
                pi2_zero = np.zeros((dim, dim), dtype=complex)
                w_id = np.zeros((3, 3, dim, dim), dtype=complex)
                w_id[0, 0] = pi0_id
                w_id[0, 1] = pi1_zero
                w_id[0, 2] = pi2_zero
                w_id[1, 1] = pi0_id
                w_id[1, 2] = pi1_zero
                w_id[2, 2] = pi0_id
                w_identity_cache[dim] = w_id
            w = w_identity_cache[dim]
        env = np.einsum('aij,ipk,jql,abpq->bkl', env, B, Bc, w, optimize=True)
    p_arrive = float(env[2, 0, 0].real)
    if arrival_verbose:
        print(f"  两光子到达概率 p_arrive={p_arrive:.6f}")
    if p_arrive < P_ARRIVE_EPS:
        p_arrive = 0.0

    # 若几乎不可能到达且无暗计数，则可直接短路，避免数值噪声。
    if p_arrive <= P_ARRIVE_EPS and p_dark <= 0.0:
        if verbose:
            print(f"  p_arrive<{P_ARRIVE_EPS:.1e} 且 p_dark=0，跳过POVM收缩")
        metrics = None
        if compute_metrics:
            metrics = SuccessEnumerationResult(
                p_arrive=p_arrive,
                p_success=0.0,
                p_success_true=0.0,
                p_success_false=0.0,
                p_success_given_arrival=0.0,
                fidelity_declared=0.0,
                fidelity_true=0.0,
                fidelity_false=0.0,
            )
        samples = []
        if n_samples > 0:
            zero_spin = np.zeros((4, 4), dtype=complex)
            samples = [
                TwoPhotonDetectionResult(
                    clicks=[],
                    success=False,
                    bell_state="",
                    spin_state=zero_spin,
                )
                for _ in range(n_samples)
            ]
        return DetectionPipelineResult(
            p_arrive=p_arrive,
            metrics=metrics,
            samples=samples,
        )

    # ------------------------------------------------------------------
    # 构造“点击记录 → effect”的映射（5D）：
    #   1) 6D 探测 POVM（输出端口）
    #   2) 共轭并入 BS
    #   3) 投影到 3D (BS 输入子空间)
    #   4) 推入光纤（损耗 + Jones/相位）
    #   5) 嵌入到 5D 账本
    #   6) 推入 780 过滤 + QFC
    # ------------------------------------------------------------------
    effects_all_6d, effects_true_6d = build_detection_effects_6d(eta_det, p_dark)

    if bs_unitary_6d is not None:
        effects_all_6d = _apply_unitary_adjoint(effects_all_6d, bs_unitary_6d)
        effects_true_6d = _apply_unitary_adjoint(effects_true_6d, bs_unitary_6d)

    effects_all_3d = {k: _project_6d_to_3d(E) for k, E in effects_all_6d.items()}
    effects_true_3d = {k: _project_6d_to_3d(E) for k, E in effects_true_6d.items()}

    # 光纤损耗（单光子 3D）
    K_A_3 = loss_channel_1517_single_photon(float(eta_H_A), float(eta_V_A))
    K_B_3 = loss_channel_1517_single_photon(float(eta_H_B), float(eta_V_B))
    effects_all_3d = _apply_local_channel_adjoint(effects_all_3d, K_A_3, K_B_3)
    effects_true_3d = _apply_local_channel_adjoint(effects_true_3d, K_A_3, K_B_3)

    # 780 过滤（5D）与 QFC（5D）
    K_filter = None
    if apply_filter_780:
        K_filter = loss_channel_both_subspaces(
            eta_780=0.0,
            eta_H_1517=1.0,
            eta_V_1517=1.0,
        )
    U_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)
    U_qfc_pair = np.kron(U_qfc, U_qfc)

    # 逐 bin 构造 effect（包含相位斜率/抖动）
    effects_all_by_bin: List[dict] = []
    effects_true_by_bin: List[dict] = []

    phase_center = 0.5 * (n_bins - 1)
    use_phase_profile = abs(phase_slope) > 0.0 or phase_jitter_std > 0.0

    U_A_3 = _jones_3d(U_A)
    for n in range(n_bins):
        phase_n = phase_slope * (n - phase_center)
        if phase_jitter_std > 0.0:
            phase_n += rng.normal(0.0, phase_jitter_std)
        if use_phase_profile or abs(phase_n) > 0.0:
            U_B_n = np.exp(1j * phase_n) * U_B
        else:
            U_B_n = U_B

        U_B_3 = _jones_3d(U_B_n)
        U_pair_3 = np.kron(U_A_3, U_B_3)

        eff_all_3 = _apply_unitary_adjoint(effects_all_3d, U_pair_3)
        eff_true_3 = _apply_unitary_adjoint(effects_true_3d, U_pair_3)

        eff_all_5 = {k: _embed_3d_to_5d(E) for k, E in eff_all_3.items()}
        eff_true_5 = {k: _embed_3d_to_5d(E) for k, E in eff_true_3.items()}

        if K_filter is not None:
            eff_all_5 = _apply_local_channel_adjoint(eff_all_5, K_filter, K_filter)
            eff_true_5 = _apply_local_channel_adjoint(eff_true_5, K_filter, K_filter)

        eff_all_5 = _apply_unitary_adjoint(eff_all_5, U_qfc_pair)
        eff_true_5 = _apply_unitary_adjoint(eff_true_5, U_qfc_pair)

        effects_all_by_bin.append(eff_all_5)
        effects_true_by_bin.append(eff_true_5)

    if verbose and n_samples > 0:
        dim_pair = bin_dim * bin_dim
        print(f"  使用{bin_dim}D POVM effects ({dim_pair}x{dim_pair}) - 抽样双点击记录")

    empty_key = _order_two_port_detectors([])
    if not effects_all_by_bin:
        raise ValueError("空的探测effect，无法进行POVM计算")
    dim_pair = next(iter(effects_all_by_bin[0].values())).shape[0]
    zero_effect = np.zeros((dim_pair, dim_pair), dtype=complex)
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
        if key not in effects_all_by_bin[0]:
            raise ValueError(f"缺少探测结果: detectors={list(key)}")

    E_no_list = [
        effects_all_by_bin[idx].get(empty_key, zero_effect) for idx in range(n_bins)
    ]

    def _prepare_grouped_pairs(state: MPSState) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # 把 (atomA,atomB),(A1,B1),... 两两成组：
        # - 便于一次性对每个 bin 的 (A_n,B_n) 做 2-site effect 收缩
        # - 这样 left/right environment 的构造更简洁
        psi = state._mps.copy()
        if psi.L % 2 != 0:
            raise ValueError("MPS sites 数量必须为偶数，才能按 (atomA,atomB),(A1,B1),... 分组")
        psi.group_sites(n=2)
        psi.canonical_form_finite(renormalize=True)
        psi.norm = 1.0
        B_list_local = []
        Bc_list_local = []
        for i in range(psi.L):
            B = psi.get_B(i, form='B').to_ndarray()
            B_list_local.append(B)
            Bc_list_local.append(B.conj())
        return B_list_local, Bc_list_local

    B_list, Bc_list = _prepare_grouped_pairs(mps)
    grouped_bins = len(B_list) - 1
    if grouped_bins != n_bins:
        raise ValueError(f"n_bins={n_bins} 与分组后bin数量 {grouped_bins} 不一致")

    dim_atom = B_list[0].shape[1]
    if dim_atom != 16:
        raise ValueError(f"Atom pair site dimension {dim_atom} != 16")

    atom_I = np.eye(dim_atom, dtype=complex)
    L = len(B_list)

    def _apply_env_left(
        B: np.ndarray,
        Bc: np.ndarray,
        op: np.ndarray,
        env_left: np.ndarray,
    ) -> np.ndarray:
        # 左环境推进：env' = <env| B op B^†
        # 这里的 op 是当前 site 的测量算符（或单位）。
        return np.einsum('ij,ipk,jql,pq->kl', env_left, B, Bc, op, optimize=True)

    def _apply_env_right(
        B: np.ndarray,
        Bc: np.ndarray,
        op: np.ndarray,
        env_right: np.ndarray,
    ) -> np.ndarray:
        # 右环境推进：env' = B op B^† |env>
        return np.einsum('ipk,jql,pq,kl->ij', B, Bc, op, env_right, optimize=True)

    def _build_left_envs(atom_op: np.ndarray) -> List[np.ndarray]:
        # 构造左环境序列，便于在任意 bin 插入测量算符。
        # 左边第一站点是 atom-pair，需要插入 atom_op（如 Bell 投影或 I）。
        left_envs_local = [None] * (L + 1)
        left_envs_local[0] = np.array([[1.0 + 0.0j]])
        left_envs_local[1] = _apply_env_left(B_list[0], Bc_list[0], atom_op, left_envs_local[0])
        for s in range(1, L):
            bin_idx = s - 1
            left_envs_local[s + 1] = _apply_env_left(
                B_list[s],
                Bc_list[s],
                E_no_list[bin_idx],
                left_envs_local[s],
            )
        return left_envs_local

    def _build_right_envs() -> List[np.ndarray]:
        # 构造右环境序列：默认所有 bin 都是“无点击”(E_no)。
        right_envs_local = [None] * (L + 1)
        right_envs_local[L] = np.array([[1.0 + 0.0j]])
        for s in range(L - 1, 0, -1):
            bin_idx = s - 1
            right_envs_local[s] = _apply_env_right(
                B_list[s],
                Bc_list[s],
                E_no_list[bin_idx],
                right_envs_local[s + 1],
            )
        return right_envs_local

    right_envs = _build_right_envs()
    left_envs_id = _build_left_envs(atom_I)

    metrics = None
    if compute_metrics:
        if verbose:
            dim_pair = bin_dim * bin_dim
            print(f"  使用{bin_dim}D Kraus operators ({dim_pair}x{dim_pair}) - POVM收缩")

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
            dim_atom_local = 4
            proj_full = np.zeros(
                (dim_atom_local * dim_atom_local, dim_atom_local * dim_atom_local),
                dtype=complex,
            )
            qubit_indices = [
                0 * dim_atom_local + 0,
                0 * dim_atom_local + 1,
                1 * dim_atom_local + 0,
                1 * dim_atom_local + 1,
            ]
            for i, qi in enumerate(qubit_indices):
                for j, qj in enumerate(qubit_indices):
                    proj_full[qi, qj] = proj_qubit[i, j]
            return proj_full

        bell_projectors = {bell: _bell_projector_full(bell) for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]}
        left_envs_bell = {
            bell: _build_left_envs(proj)
            for bell, proj in bell_projectors.items()
        }

        def _sum_same_bin(
            left_envs: List[np.ndarray],
            effects_by_bin: List[dict],
            key_pair: Tuple[str, ...],
        ) -> float:
            # 同 bin 双击：对每个 bin 插入对应的二端口 effect
            total = 0.0
            for s in range(1, n_bins + 1):
                op_pair = effects_by_bin[s - 1].get(key_pair, zero_effect)
                env_mid = _apply_env_left(B_list[s], Bc_list[s], op_pair, left_envs[s])
                weight = float(np.einsum('ij,ij->', env_mid, right_envs[s + 1]).real)
                total += weight
            return total

        def _sum_diff_bins(
            left_envs: List[np.ndarray],
            effects_by_bin: List[dict],
            key_a: Tuple[str, ...],
            key_b: Tuple[str, ...],
        ) -> float:
            # 不同 bin 双击：先插入 detector A，再在后续 bin 插入 detector B
            total = 0.0
            for i in range(1, n_bins):
                op_a = effects_by_bin[i - 1].get(key_a, zero_effect)
                env_mid = _apply_env_left(B_list[i], Bc_list[i], op_a, left_envs[i])
                j_end = n_bins
                if window_bins is not None:
                    j_end = min(n_bins, i + window_bins)
                for j in range(i + 1, j_end + 1):
                    op_b = effects_by_bin[j - 1].get(key_b, zero_effect)
                    env_j = _apply_env_left(B_list[j], Bc_list[j], op_b, env_mid)
                    weight = float(np.einsum('ij,ij->', env_j, right_envs[j + 1]).real)
                    total += weight
                    if j < j_end:
                        env_mid = _apply_env_left(B_list[j], Bc_list[j], E_no_list[j - 1], env_mid)
            return total

        # BSM 成功模式（按你当前定义）：
        # Psi-: 交叉端口不同偏振
        # Psi+: 同端口不同偏振
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

            weight_same_all = _sum_same_bin(left_envs_id, effects_all_by_bin, key_pair)
            weight_same_true = _sum_same_bin(left_envs_id, effects_true_by_bin, key_pair)

            weight_diff_all = _sum_diff_bins(left_envs_id, effects_all_by_bin, key_a, key_b)
            weight_diff_all += _sum_diff_bins(left_envs_id, effects_all_by_bin, key_b, key_a)
            weight_diff_true = _sum_diff_bins(left_envs_id, effects_true_by_bin, key_a, key_b)
            weight_diff_true += _sum_diff_bins(left_envs_id, effects_true_by_bin, key_b, key_a)

            p_success_all += weight_same_all + weight_diff_all
            p_success_true += weight_same_true + weight_diff_true
            fidelity_weighted_all += _sum_same_bin(left_envs_bell[bell_state], effects_all_by_bin, key_pair)
            fidelity_weighted_all += _sum_diff_bins(left_envs_bell[bell_state], effects_all_by_bin, key_a, key_b)
            fidelity_weighted_all += _sum_diff_bins(left_envs_bell[bell_state], effects_all_by_bin, key_b, key_a)

            fidelity_weighted_true += _sum_same_bin(left_envs_bell[bell_state], effects_true_by_bin, key_pair)
            fidelity_weighted_true += _sum_diff_bins(left_envs_bell[bell_state], effects_true_by_bin, key_a, key_b)
            fidelity_weighted_true += _sum_diff_bins(left_envs_bell[bell_state], effects_true_by_bin, key_b, key_a)

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

        p_success_given_arrival = (p_success_true / p_arrive) if p_arrive > P_ARRIVE_EPS else 0.0

        metrics = SuccessEnumerationResult(
            p_arrive=p_arrive,
            p_success=p_success_all,
            p_success_true=p_success_true,
            p_success_false=p_success_false,
            p_success_given_arrival=p_success_given_arrival,
            fidelity_declared=fidelity_declared,
            fidelity_true=fidelity_true,
            fidelity_false=fidelity_false,
        )

    samples: List[TwoPhotonDetectionResult] = []
    if n_samples <= 0:
        return DetectionPipelineResult(p_arrive=p_arrive, metrics=metrics, samples=samples)

    # 抽样用的“候选双点击模式集合”：
    # - 覆盖 Psi± 与部分失败模式
    patterns_records = [
        ("Psi-", ("H1", "V2")),
        ("Psi-", ("V1", "H2")),
        ("Psi+", ("H1", "V1")),
        ("Psi+", ("H2", "V2")),
        ("", ("H1", "H2")),
        ("", ("V1", "V2")),
    ]

    weight_eps = 1e-14
    records: List[TwoClickRecord] = []

    def _collect_same_bin_records(
        det_a: str,
        det_b: str,
    ) -> List[TwoClickRecord]:
        # 逐 bin 收集“同 bin 双击”的权重分布
        key_pair = _order_two_port_detectors([det_a, det_b])
        records_local = []
        for s in range(1, n_bins + 1):
            op_pair = effects_all_by_bin[s - 1].get(key_pair, zero_effect)
            env_mid = _apply_env_left(B_list[s], Bc_list[s], op_pair, left_envs_id[s])
            weight = float(np.einsum('ij,ij->', env_mid, right_envs[s + 1]).real)
            if weight > weight_eps:
                records_local.append(TwoClickRecord(det_a, det_b, s - 1, s - 1, weight))
        return records_local

    def _collect_diff_bin_records(
        det_first: str,
        det_second: str,
    ) -> List[TwoClickRecord]:
        # 逐 (i<j) 收集“跨 bin 双击”的权重分布
        key_first = _order_two_port_detectors([det_first])
        key_second = _order_two_port_detectors([det_second])
        records_local = []
        for i in range(1, n_bins):
            op_first = effects_all_by_bin[i - 1].get(key_first, zero_effect)
            env_mid = _apply_env_left(B_list[i], Bc_list[i], op_first, left_envs_id[i])
            for j in range(i + 1, n_bins + 1):
                op_second = effects_all_by_bin[j - 1].get(key_second, zero_effect)
                env_j = _apply_env_left(B_list[j], Bc_list[j], op_second, env_mid)
                weight = float(np.einsum('ij,ij->', env_j, right_envs[j + 1]).real)
                if weight > weight_eps:
                    records_local.append(TwoClickRecord(det_first, det_second, i - 1, j - 1, weight))
                if j < n_bins:
                    env_mid = _apply_env_left(B_list[j], Bc_list[j], E_no_list[j - 1], env_mid)
        return records_local

    # 汇总所有可选双点击记录，组成离散分布供抽样
    for _, (det_a, det_b) in patterns_records:
        records.extend(_collect_same_bin_records(det_a, det_b))
        records.extend(_collect_diff_bin_records(det_a, det_b))
        records.extend(_collect_diff_bin_records(det_b, det_a))

    if not records:
        samples = []
        if n_samples > 0:
            zero_spin = np.zeros((4, 4), dtype=complex)
            samples = [
                TwoPhotonDetectionResult(
                    clicks=[],
                    success=False,
                    bell_state="",
                    spin_state=zero_spin,
                )
                for _ in range(n_samples)
            ]
        return DetectionPipelineResult(
            p_arrive=p_arrive,
            metrics=metrics,
            samples=samples,
        )

    # 将权重归一化为概率分布 p_r，随后按 p_r 抽样
    weights = np.array([max(0.0, r.weight) for r in records], dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= weight_eps:
        samples = []
        if n_samples > 0:
            zero_spin = np.zeros((4, 4), dtype=complex)
            samples = [
                TwoPhotonDetectionResult(
                    clicks=[],
                    success=False,
                    bell_state="",
                    spin_state=zero_spin,
                )
                for _ in range(n_samples)
            ]
        return DetectionPipelineResult(
            p_arrive=p_arrive,
            metrics=metrics,
            samples=samples,
        )

    probs = weights / total_weight

    def _contract_record(
        left_envs: List[np.ndarray],
        record_local: TwoClickRecord,
    ) -> complex:
        # 计算某一条“点击记录 r”的未归一化原子态矩阵元：
        #   sigma_ij = Tr[ (|i><j| ⊗ E_r) ρ ]
        if record_local.bin_a == record_local.bin_b:
            s = record_local.bin_a + 1
            key_pair = _order_two_port_detectors([record_local.detector_a, record_local.detector_b])
            op_pair = effects_all_by_bin[record_local.bin_a].get(key_pair, zero_effect)
            env_mid = _apply_env_left(B_list[s], Bc_list[s], op_pair, left_envs[s])
            return np.einsum('ij,ij->', env_mid, right_envs[s + 1])

        if record_local.bin_a < record_local.bin_b:
            i = record_local.bin_a + 1
            j = record_local.bin_b + 1
            key_first = _order_two_port_detectors([record_local.detector_a])
            key_second = _order_two_port_detectors([record_local.detector_b])
            op_first = effects_all_by_bin[record_local.bin_a].get(key_first, zero_effect)
            op_second = effects_all_by_bin[record_local.bin_b].get(key_second, zero_effect)
        else:
            i = record_local.bin_b + 1
            j = record_local.bin_a + 1
            key_first = _order_two_port_detectors([record_local.detector_b])
            key_second = _order_two_port_detectors([record_local.detector_a])
            op_first = effects_all_by_bin[record_local.bin_b].get(key_first, zero_effect)
            op_second = effects_all_by_bin[record_local.bin_a].get(key_second, zero_effect)

        env_mid = _apply_env_left(B_list[i], Bc_list[i], op_first, left_envs[i])
        for s in range(i + 1, j):
            env_mid = _apply_env_left(B_list[s], Bc_list[s], E_no_list[s - 1], env_mid)
        env_mid = _apply_env_left(B_list[j], Bc_list[j], op_second, env_mid)
        return np.einsum('ij,ij->', env_mid, right_envs[j + 1])

    single_dim = int(round(np.sqrt(dim_atom)))
    if single_dim * single_dim != dim_atom:
        raise ValueError(f"Unexpected atom-pair dimension: {dim_atom}")
    qubit_indices = [
        0 * single_dim + 0,
        0 * single_dim + 1,
        1 * single_dim + 0,
        1 * single_dim + 1,
    ]
    left_envs_qubit: Optional[List[List[List[np.ndarray]]]] = None

    def _ensure_left_envs_qubit() -> None:
        # 为原子 4x4 子空间的每个 |i><j| 构造 left_env
        # 便于高效组装 4x4 原子后验态矩阵
        nonlocal left_envs_qubit
        if left_envs_qubit is not None:
            return
        left_envs_qubit = [[None for _ in range(4)] for _ in range(4)]
        for i, qi in enumerate(qubit_indices):
            for j, qj in enumerate(qubit_indices):
                atom_op = np.zeros((dim_atom, dim_atom), dtype=complex)
                atom_op[qi, qj] = 1.0
                left_envs_qubit[i][j] = _build_left_envs(atom_op)

    def _compute_record_qubit_state(record_local: TwoClickRecord) -> np.ndarray:
        # 返回该记录对应的原子 4x4 未归一化密度矩阵
        # 注意：这是 effect-only 的 Lüders 更新结果
        _ensure_left_envs_qubit()
        sigma = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            for j in range(4):
                left_envs = left_envs_qubit[i][j]
                sigma[i, j] = _contract_record(
                    left_envs,
                    record_local,
                )
        return sigma

    for sample_index in range(1, n_samples + 1):
        pick = int(rng.choice(len(records), p=probs))
        record = records[pick]
        base_a = bin_start + 2 * record.bin_a
        base_b = bin_start + 2 * record.bin_b
        site_a = base_a if record.detector_a in ("H1", "V1") else base_a + 1
        site_b = base_b if record.detector_b in ("H1", "V1") else base_b + 1
        clicks = [
            DetectionEvent(
                detector=record.detector_a,
                bin_index=record.bin_a,
                site=site_a,
            ),
            DetectionEvent(
                detector=record.detector_b,
                bin_index=record.bin_b,
                site=site_b,
            ),
        ]
        success = False
        bell_state = ""
        if window_bins is None or abs(record.bin_a - record.bin_b) <= window_bins:
            detectors = {record.detector_a, record.detector_b}
            if detectors == {"H1", "V2"} or detectors == {"V1", "H2"}:
                success = True
                bell_state = "Psi-"
            elif detectors == {"H1", "V1"} or detectors == {"H2", "V2"}:
                success = True
                bell_state = "Psi+"
        spin_state = _compute_record_qubit_state(record)

        if verbose:
            if n_samples > 1:
                print(f"\n  [POVM抽样 {sample_index}/{n_samples}]")
            print("\n  结果：")
            print("    抽样自双点击分布（条件在两次点击）")
            print(f"    点击：{[(c.detector, c.bin_index) for c in clicks]}")
            print(f"    BSM成功：{success}")
            if success:
                print(f"    Bell态：{bell_state}")

        samples.append(
            TwoPhotonDetectionResult(
                clicks=clicks,
                success=success,
                bell_state=bell_state,
                spin_state=spin_state,
            )
        )

    return DetectionPipelineResult(p_arrive=p_arrive, metrics=metrics, samples=samples)


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
    def _build_sum_mpo(local_op: np.ndarray) -> float:
        total = 0.0
        n_sites = len(mps.d)

        # MPS结构：[atomA, atomB, A1, B1, A2, B2, ..., An, Bn]
        # 前2个站点是原子，后面是bin站点（交替左臂和右臂）
        for site_idx in range(2, n_sites):
            rho = mps.get_reduced_density([site_idx])
            dim = mps.d[site_idx]

            if local_op.shape[0] != dim:
                continue

            if len(rho.shape) != 2 or rho.shape[0] != dim or rho.shape[1] != dim:
                rho = rho.reshape(dim, dim)

            expectation = np.trace(np.dot(rho, local_op))
            total += np.real(expectation)
        return total

    def _compute_photon_statistics_global(bin_dim: int) -> dict:
        if bin_dim != 5:
            raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 5.")

        # 5D 基序：vac, H_780, V_780, H_1517, V_1517
        n_780_H_op = np.diag([0, 1, 0, 0, 0]).astype(complex)
        n_780_V_op = np.diag([0, 0, 1, 0, 0]).astype(complex)
        n_1517_H_op = np.diag([0, 0, 0, 1, 0]).astype(complex)
        n_1517_V_op = np.diag([0, 0, 0, 0, 1]).astype(complex)

        n_780_H = _build_sum_mpo(n_780_H_op)
        n_780_V = _build_sum_mpo(n_780_V_op)
        n_1517_H = _build_sum_mpo(n_1517_H_op)
        n_1517_V = _build_sum_mpo(n_1517_V_op)

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

    bin_dim = mps.d[2]
    return _compute_photon_statistics_global(bin_dim)
