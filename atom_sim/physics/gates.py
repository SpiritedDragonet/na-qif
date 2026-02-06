# -*- coding: utf-8 -*-
"""
带缓存的幺正门工厂

本模块提供用于构造时间仓仿真中使用的幺正门的工厂函数。
不随每仓变化的门会被缓存。
"""

from typing import Tuple, Optional, List, Callable
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.linalg import expm

from ..hilbert.basis import (
    SUBSPACE_1517,
    embed_3d_to_5d,
    embed_5_from_3,
    embed_9_from_6,
    jones_3d,
    project_6d_to_3d,
    reduce_9d_effects_to_6d,
)
from ..hilbert.operators import (
    annihilation_op,
    creation_op,
    atom_transition,
)
from .channels import (
    loss_channel_both_subspaces,
    loss_channel_1517_single_photon,
)


# 缓存的门（这些计算昂贵且不随每仓变化）

@lru_cache(maxsize=8)
def qfc_gate(theta_H: float = 0.0, theta_V: float = 0.0) -> np.ndarray:
    """
    量子频率转换门 U_qfc（5D bin）。

    通过类分束器耦合将780nm光子转换为1517nm：
        对每个偏振在 (|H_780>, |H_1517>) 与 (|V_780>, |V_1517>) 上做二维旋转，
        其中 sin^2(theta) = 转换概率。

    Parameters
    ----------
    theta_H : float
        H偏振的转换角（sin²(theta) = 转换概率）
    theta_V : float
        V偏振的转换角

    Returns
    -------
    np.ndarray
        作用于bin空间的5x5幺正矩阵

    Examples
    --------
    >>> U = qfc_gate(theta_H=np.pi/4, theta_V=np.pi/4)  # 50%转换
    """
    # ------------------------------------------------------------------
    # 5D bin 基序：
    #   0: |vac>
    #   1: |H_780>
    #   2: |V_780>
    #   3: |H_1517>
    #   4: |V_1517>
    #
    # 在 (1,3) 与 (2,4) 子空间做二维旋转。
    # ------------------------------------------------------------------
    U = np.eye(5, dtype=complex)

    cH = np.cos(theta_H)
    sH = np.sin(theta_H)
    cV = np.cos(theta_V)
    sV = np.sin(theta_V)

    # H 偏振：|H_780> <-> |H_1517>
    U[1, 1] = cH
    U[1, 3] = -sH
    U[3, 1] = sH
    U[3, 3] = cH

    # V 偏振：|V_780> <-> |V_1517>
    U[2, 2] = cV
    U[2, 4] = -sV
    U[4, 2] = sV
    U[4, 4] = cV

    return U



@lru_cache(maxsize=4)
def bs_gate_6d() -> np.ndarray:
    """
    6D 输出端口（1517nm）的50/50分束器门（36x36）。

    该门用于测量端共轭：BS 后的端口需要容纳 2 光子态。

    Returns
    -------
    np.ndarray
        1517_A × 1517_B空间的36x36幺正矩阵（每个站点6×6）
    """
    # ------------------------------------------------------------------
    # 该 BS 只作用在 1517nm 子空间 (6D)：
    #   - H/V 各自做 50/50 beamsplitter
    #   - 多光子态 (|2H>, |HV>, |2V>) 通过算符指数自然包含
    # ------------------------------------------------------------------
    return _bs_gate_1517()


def _permute_factors_matrix(dims: Tuple[int, ...], perm: Tuple[int, ...]) -> np.ndarray:
    """构造张量因子重排的置换矩阵。"""
    dim = int(np.prod(dims))
    P = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        multi = np.unravel_index(idx, dims)
        permuted = tuple(multi[p] for p in perm)
        jdx = np.ravel_multi_index(permuted, dims)
        P[jdx, idx] = 1.0
    return P


def _bs_gate_9d_dist_from_6d(U_6d: np.ndarray) -> np.ndarray:
    """
    从 6D BS 推导“可区分标签”的 9D × 9D 两端口 BS（81x81）。

    逻辑：
      - 先从 6D BS 提取 3D 单光子块（9x9）
      - 对标签 a / b 分别作用：U_a ⊗ U_b
      - 用置换矩阵从 (aA,aB,bA,bB) 重排回 (aA,bA,aB,bB)
    """
    U_3d = project_6d_to_3d(U_6d)
    U_ab = np.kron(U_3d, U_3d)
    # perm: (aA, aB, bA, bB) -> (aA, bA, aB, bB)
    P_swap = _permute_factors_matrix((3, 3, 3, 3), (0, 2, 1, 3))
    return P_swap @ U_ab @ P_swap.T


@lru_cache(maxsize=4)
def _bs_gate_9d_dist_cached() -> np.ndarray:
    return _bs_gate_9d_dist_from_6d(bs_gate_6d())


def bs_gate_9d_dist(bs_unitary_6d: Optional[np.ndarray] = None) -> np.ndarray:
    """
    9D 标签空间下的可区分 BS（81x81）。
    若传入 6D BS，则使用该门推导；否则使用默认 6D BS。
    """
    if bs_unitary_6d is None:
        return _bs_gate_9d_dist_cached()
    return _bs_gate_9d_dist_from_6d(bs_unitary_6d)


def order_two_port_detectors(detectors: List[str]) -> Tuple[str, ...]:
    """双端口探测器标签排序（H1,V1,H2,V2）。"""
    order = {"H1": 0, "V1": 1, "H2": 2, "V2": 3}
    return tuple(sorted(detectors, key=lambda detector: order[detector]))


def build_detection_effects_6d(
    eta: float,
    p_dark: float = 0.0,
) -> Tuple[dict, dict, dict]:
    """构造 6D 端口空间的探测 POVM（含暗计数拆分）。"""

    def _order_detectors(detectors: List[str]) -> List[str]:
        order = {"H": 0, "V": 1}
        return sorted(detectors, key=lambda detector: order[detector])

    def _split_with_dark(
        kraus: np.ndarray,
        detectors: List[str],
        p_dark_local: float,
    ) -> List[Tuple[np.ndarray, List[str], List[str]]]:
        if not 0 <= p_dark_local <= 1:
            raise ValueError(f"p_dark必须在[0, 1]内，得到 {p_dark_local}")

        base_detectors = _order_detectors(detectors)
        if p_dark_local <= 0:
            return [(kraus, base_detectors, [])]

        off_detectors = [detector for detector in ("H", "V") if detector not in base_detectors]
        entries = []
        for mask in product([0, 1], repeat=len(off_detectors)):
            prob = 1.0
            dark_detectors = []
            for detector, use_dark in zip(off_detectors, mask):
                if use_dark:
                    prob *= p_dark_local
                    dark_detectors.append(detector)
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
        k00_6d = np.diag([
            1.0,
            np.sqrt(1 - eta_local),
            np.sqrt(1 - eta_local),
            (1 - eta_local),
            (1 - eta_local),
            (1 - eta_local),
        ]).astype(complex)

        k10a_6d = np.zeros((6, 6), dtype=complex)
        k10a_6d[0, 1] = np.sqrt(eta_local)
        k10b_6d = np.zeros((6, 6), dtype=complex)
        k10b_6d[0, 3] = np.sqrt(1 - (1 - eta_local) ** 2)
        k10c_6d = np.zeros((6, 6), dtype=complex)
        k10c_6d[2, 5] = np.sqrt(eta_local * (1 - eta_local))

        k01a_6d = np.zeros((6, 6), dtype=complex)
        k01a_6d[0, 2] = np.sqrt(eta_local)
        k01b_6d = np.zeros((6, 6), dtype=complex)
        k01b_6d[0, 4] = np.sqrt(1 - (1 - eta_local) ** 2)
        k01c_6d = np.zeros((6, 6), dtype=complex)
        k01c_6d[1, 5] = np.sqrt(eta_local * (1 - eta_local))

        k11_6d = np.zeros((6, 6), dtype=complex)
        k11_6d[0, 5] = eta_local

        base_entries = [
            (k00_6d, []),
            (k10a_6d, ["H"]),
            (k10b_6d, ["H"]),
            (k10c_6d, ["H"]),
            (k01a_6d, ["V"]),
            (k01b_6d, ["V"]),
            (k01c_6d, ["V"]),
            (k11_6d, ["H", "V"]),
        ]

        entries = []
        for kraus, detectors in base_entries:
            for kraus_split, detectors_split, dark_split in _split_with_dark(
                kraus,
                detectors,
                p_dark_local,
            ):
                entries.append((kraus_split, detectors_split, dark_split))
        return entries

    port_entries = _build_port_kraus_entries_6d(eta, p_dark)
    kraus_list: List[np.ndarray] = []
    outcome_detectors: List[List[str]] = []
    outcome_dark: List[List[str]] = []

    for k1, det1, dark1 in port_entries:
        for k2, det2, dark2 in port_entries:
            k_two = np.kron(k1, k2)

            detectors = []
            dark_detectors = []
            for detector in ("H", "V"):
                if detector in det1:
                    detectors.append(f"{detector}1")
                if detector in dark1:
                    dark_detectors.append(f"{detector}1")
            for detector in ("H", "V"):
                if detector in det2:
                    detectors.append(f"{detector}2")
                if detector in dark2:
                    dark_detectors.append(f"{detector}2")

            kraus_list.append(k_two)
            outcome_detectors.append(detectors)
            outcome_dark.append(dark_detectors)

    effects_all = {}
    effects_true = {}
    effects_by_darkmask = {}
    for kraus, detectors, dark_detectors in zip(kraus_list, outcome_detectors, outcome_dark):
        key = order_two_port_detectors(detectors)
        mask = order_two_port_detectors(dark_detectors)
        effect = np.asarray(kraus).conj().T @ np.asarray(kraus)
        effects_all[key] = effects_all.get(key, 0) + effect
        if not dark_detectors:
            effects_true[key] = effects_true.get(key, 0) + effect
        mask_map = effects_by_darkmask.setdefault(key, {})
        mask_map[mask] = mask_map.get(mask, 0) + effect
    return effects_all, effects_true, effects_by_darkmask


def build_detection_effects_9d(
    eta: float,
    p_dark: float = 0.0,
) -> Tuple[dict, dict, dict]:
    """构造 9D 标签空间的探测 POVM（含暗计数拆分）。"""

    def _order_detectors(detectors: List[str]) -> List[str]:
        order = {"H": 0, "V": 1}
        return sorted(detectors, key=lambda detector: order[detector])

    basis = [(a, b) for a in (0, 1, 2) for b in (0, 1, 2)]
    dim = 9

    effects_by_mask_port = {}
    for idx, (a_pol, b_pol) in enumerate(basis):
        n_h = int(a_pol == 1) + int(b_pol == 1)
        n_v = int(a_pol == 2) + int(b_pol == 2)
        p_h_click = 1.0 - (1.0 - eta) ** n_h
        p_v_click = 1.0 - (1.0 - eta) ** n_v

        base_outcomes = [
            ([], (1.0 - p_h_click) * (1.0 - p_v_click)),
            (["H"], p_h_click * (1.0 - p_v_click)),
            (["V"], (1.0 - p_h_click) * p_v_click),
            (["H", "V"], p_h_click * p_v_click),
        ]

        for base_detectors, p_true in base_outcomes:
            if p_true <= 0:
                continue
            base_detectors = _order_detectors(base_detectors)
            off_detectors = [detector for detector in ("H", "V") if detector not in base_detectors]
            for mask in product([0, 1], repeat=len(off_detectors)):
                prob = p_true
                dark_detectors = []
                for detector, use_dark in zip(off_detectors, mask):
                    if use_dark:
                        prob *= p_dark
                        dark_detectors.append(detector)
                    else:
                        prob *= (1 - p_dark)
                if prob <= 0:
                    continue
                dark_detectors = _order_detectors(dark_detectors)
                final_detectors = _order_detectors(base_detectors + dark_detectors)
                mask_map = effects_by_mask_port.setdefault(tuple(final_detectors), {})
                effect = mask_map.get(tuple(dark_detectors))
                if effect is None:
                    effect = np.zeros((dim, dim), dtype=complex)
                    mask_map[tuple(dark_detectors)] = effect
                effect[idx, idx] += prob

    port_entries = []
    for key, mask_map in effects_by_mask_port.items():
        for mask, effect in mask_map.items():
            port_entries.append((effect, list(key), list(mask)))

    effects_all = {}
    effects_true = {}
    effects_by_darkmask = {}
    for e1, det1, dark1 in port_entries:
        for e2, det2, dark2 in port_entries:
            e_two = np.kron(e1, e2)
            detectors = []
            dark_detectors = []
            for detector in ("H", "V"):
                if detector in det1:
                    detectors.append(f"{detector}1")
                if detector in dark1:
                    dark_detectors.append(f"{detector}1")
            for detector in ("H", "V"):
                if detector in det2:
                    detectors.append(f"{detector}2")
                if detector in dark2:
                    dark_detectors.append(f"{detector}2")

            key = order_two_port_detectors(detectors)
            mask = order_two_port_detectors(dark_detectors)
            effects_all[key] = effects_all.get(key, 0) + e_two
            if not dark_detectors:
                effects_true[key] = effects_true.get(key, 0) + e_two
            mask_map = effects_by_darkmask.setdefault(key, {})
            mask_map[mask] = mask_map.get(mask, 0) + e_two

    return effects_all, effects_true, effects_by_darkmask


def mix_effects(effects_int: dict, effects_dist: dict, v_res: float) -> dict:
    """按 v_res 混合不可区分与可区分 effect。"""
    if v_res >= 1.0:
        return effects_int
    if v_res <= 0.0:
        return effects_dist
    keys = set(effects_int.keys()) | set(effects_dist.keys())
    mixed = {}
    for key in keys:
        effect_int = effects_int.get(key)
        effect_dist = effects_dist.get(key)
        if effect_int is None:
            mixed[key] = (1.0 - v_res) * effect_dist
        elif effect_dist is None:
            mixed[key] = v_res * effect_int
        else:
            mixed[key] = v_res * effect_int + (1.0 - v_res) * effect_dist
    return mixed


def mix_effects_masked(effects_int: dict, effects_dist: dict, v_res: float) -> dict:
    """按 v_res 混合按暗计数 mask 分组的 effect。"""
    if v_res >= 1.0:
        return effects_int
    if v_res <= 0.0:
        return effects_dist
    keys = set(effects_int.keys()) | set(effects_dist.keys())
    mixed = {}
    for key in keys:
        map_int = effects_int.get(key, {})
        map_dist = effects_dist.get(key, {})
        masks = set(map_int.keys()) | set(map_dist.keys())
        mask_map = {}
        for mask in masks:
            effect_int = map_int.get(mask)
            effect_dist = map_dist.get(mask)
            if effect_int is None:
                mask_map[mask] = (1.0 - v_res) * effect_dist
            elif effect_dist is None:
                mask_map[mask] = v_res * effect_int
            else:
                mask_map[mask] = v_res * effect_int + (1.0 - v_res) * effect_dist
        mixed[key] = mask_map
    return mixed


def map_effects_masked(effects_by_mask: dict, fn: Callable[[np.ndarray], np.ndarray]) -> dict:
    """对按暗计数 mask 分组的 effect 做逐项映射。"""
    if not effects_by_mask:
        return {}
    new_effects = {}
    for key, mask_map in effects_by_mask.items():
        mapped_masks = {}
        for mask, effect in mask_map.items():
            mapped_masks[mask] = fn(effect)
        new_effects[key] = mapped_masks
    return new_effects


def apply_unitary_adjoint(effects: dict, unitary: np.ndarray) -> dict:
    """对 effect 集合施加 E <- U^† E U。"""
    if not effects:
        return {}
    unitary = np.asarray(unitary, dtype=complex)
    unitary_dag = unitary.conj().T
    return {key: unitary_dag @ effect @ unitary for key, effect in effects.items()}


def apply_unitary_adjoint_masked(effects_by_mask: dict, unitary: np.ndarray) -> dict:
    """对按暗计数 mask 分组的 effect 施加 E <- U^† E U。"""
    if not effects_by_mask:
        return {}
    unitary = np.asarray(unitary, dtype=complex)
    unitary_dag = unitary.conj().T
    return map_effects_masked(effects_by_mask, lambda effect: unitary_dag @ effect @ unitary)


def apply_local_channel_adjoint(
    effects: dict,
    kraus_list_a: List[np.ndarray],
    kraus_list_b: List[np.ndarray],
) -> dict:
    """对双端口 effect 集合施加局域信道对偶映射。"""
    if not effects:
        return {}
    kraus_pairs = [np.kron(k_a, k_b) for k_a in kraus_list_a for k_b in kraus_list_b]
    new_effects = {}
    for key, effect in effects.items():
        acc = np.zeros_like(effect)
        for kraus in kraus_pairs:
            acc += kraus.conj().T @ effect @ kraus
        new_effects[key] = acc
    return new_effects


def apply_local_channel_adjoint_masked(
    effects_by_mask: dict,
    kraus_list_a: List[np.ndarray],
    kraus_list_b: List[np.ndarray],
) -> dict:
    """对按暗计数 mask 分组的 effect 施加局域信道对偶映射。"""
    if not effects_by_mask:
        return {}
    kraus_pairs = [np.kron(k_a, k_b) for k_a in kraus_list_a for k_b in kraus_list_b]

    def _apply(effect: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(effect)
        for kraus in kraus_pairs:
            acc += kraus.conj().T @ effect @ kraus
        return acc

    return map_effects_masked(effects_by_mask, _apply)


def apply_channel_adjoint_single(op: np.ndarray, kraus_list: List[np.ndarray]) -> np.ndarray:
    """单端口对偶映射：E <- sum K^† E K。"""
    acc = np.zeros_like(op)
    for kraus in kraus_list:
        acc += kraus.conj().T @ op @ kraus
    return acc


def build_arrival_projectors_5d(
    theta_H: float,
    theta_V: float,
    eta_H_A: float,
    eta_V_A: float,
    eta_H_B: float,
    eta_V_B: float,
    apply_filter_780: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """构造用于到达概率统计的 (pi0,pi1,pi2)（5D 账本）。"""
    pi0_3 = np.diag([1, 0, 0]).astype(complex)
    pi1_3 = np.diag([0, 1, 1]).astype(complex)
    pi2_3 = np.zeros((3, 3), dtype=complex)

    k_filter = None
    if apply_filter_780:
        k_filter = loss_channel_both_subspaces(
            eta_780=0.0,
            eta_H_1517=1.0,
            eta_V_1517=1.0,
        )

    u_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)
    u_qfc_dag = u_qfc.conj().T
    p_5_from_3 = embed_5_from_3()

    def _build_one_arm(eta_h: float, eta_v: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_loss = loss_channel_1517_single_photon(eta_h, eta_v)
        p0 = apply_channel_adjoint_single(pi0_3, k_loss)
        p1 = apply_channel_adjoint_single(pi1_3, k_loss)
        p2 = apply_channel_adjoint_single(pi2_3, k_loss)

        p0 = p_5_from_3 @ p0 @ p_5_from_3.conj().T
        p1 = p_5_from_3 @ p1 @ p_5_from_3.conj().T
        p2 = p_5_from_3 @ p2 @ p_5_from_3.conj().T

        if k_filter is not None:
            p0 = apply_channel_adjoint_single(p0, k_filter)
            p1 = apply_channel_adjoint_single(p1, k_filter)
            p2 = apply_channel_adjoint_single(p2, k_filter)

        p0 = u_qfc_dag @ p0 @ u_qfc
        p1 = u_qfc_dag @ p1 @ u_qfc
        p2 = u_qfc_dag @ p2 @ u_qfc
        return p0, p1, p2

    proj_a = _build_one_arm(eta_H_A, eta_V_A)
    proj_b = _build_one_arm(eta_H_B, eta_V_B)
    return proj_a, proj_b


def build_detection_effects_5d_by_bin(
    n_bins: int,
    eta_det: float,
    p_dark: float,
    bs_unitary_6d: Optional[np.ndarray],
    v_res: float,
    U_A: np.ndarray,
    U_B: np.ndarray,
    eta_H_A: float,
    eta_V_A: float,
    eta_H_B: float,
    eta_V_B: float,
    apply_filter_780: bool,
    theta_H: float,
    theta_V: float,
    phase_slope: float,
    phase_jitter_std: float,
    rng: np.random.Generator,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """构造逐 bin 的 5D 双端口 POVM effects（all/true/masked）。"""
    effects_all_6d, effects_true_6d, effects_mask_6d = build_detection_effects_6d(eta_det, p_dark)

    if bs_unitary_6d is not None:
        effects_all_6d = apply_unitary_adjoint(effects_all_6d, bs_unitary_6d)
        effects_true_6d = apply_unitary_adjoint(effects_true_6d, bs_unitary_6d)
        effects_mask_6d = apply_unitary_adjoint_masked(effects_mask_6d, bs_unitary_6d)

    if v_res < 1.0:
        effects_all_6d_int = effects_all_6d
        effects_true_6d_int = effects_true_6d
        effects_mask_6d_int = effects_mask_6d

        effects_all_9d, effects_true_9d, effects_mask_9d = build_detection_effects_9d(eta_det, p_dark)
        if bs_unitary_6d is not None:
            u_dist_9d = bs_gate_9d_dist(bs_unitary_6d)
            effects_all_9d = apply_unitary_adjoint(effects_all_9d, u_dist_9d)
            effects_true_9d = apply_unitary_adjoint(effects_true_9d, u_dist_9d)
            effects_mask_9d = apply_unitary_adjoint_masked(effects_mask_9d, u_dist_9d)

        w = embed_9_from_6()
        w_pair = np.kron(w, w)
        effects_all_6d_dist = reduce_9d_effects_to_6d(effects_all_9d, w_pair)
        effects_true_6d_dist = reduce_9d_effects_to_6d(effects_true_9d, w_pair)
        effects_mask_6d_dist = map_effects_masked(
            effects_mask_9d,
            lambda effect: w_pair.conj().T @ effect @ w_pair,
        )

        effects_all_6d = mix_effects(effects_all_6d_int, effects_all_6d_dist, v_res)
        effects_true_6d = mix_effects(effects_true_6d_int, effects_true_6d_dist, v_res)
        effects_mask_6d = mix_effects_masked(effects_mask_6d_int, effects_mask_6d_dist, v_res)

    effects_all_3d = {key: project_6d_to_3d(effect) for key, effect in effects_all_6d.items()}
    effects_true_3d = {key: project_6d_to_3d(effect) for key, effect in effects_true_6d.items()}
    effects_mask_3d = map_effects_masked(effects_mask_6d, project_6d_to_3d)

    k_a_3 = loss_channel_1517_single_photon(float(eta_H_A), float(eta_V_A))
    k_b_3 = loss_channel_1517_single_photon(float(eta_H_B), float(eta_V_B))
    effects_all_3d = apply_local_channel_adjoint(effects_all_3d, k_a_3, k_b_3)
    effects_true_3d = apply_local_channel_adjoint(effects_true_3d, k_a_3, k_b_3)
    effects_mask_3d = apply_local_channel_adjoint_masked(effects_mask_3d, k_a_3, k_b_3)

    k_filter = None
    if apply_filter_780:
        k_filter = loss_channel_both_subspaces(
            eta_780=0.0,
            eta_H_1517=1.0,
            eta_V_1517=1.0,
        )

    u_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)
    u_qfc_pair = np.kron(u_qfc, u_qfc)

    effects_all_by_bin: List[dict] = []
    effects_true_by_bin: List[dict] = []
    effects_mask_by_bin: List[dict] = []

    phase_center = 0.5 * (n_bins - 1)
    use_phase_profile = abs(phase_slope) > 0.0 or phase_jitter_std > 0.0
    u_a_3 = jones_3d(U_A)
    for n in range(n_bins):
        phase_n = phase_slope * (n - phase_center)
        if phase_jitter_std > 0.0:
            phase_n += rng.normal(0.0, phase_jitter_std)
        if use_phase_profile or abs(phase_n) > 0.0:
            u_b_n = np.exp(1j * phase_n) * U_B
        else:
            u_b_n = U_B

        u_b_3 = jones_3d(u_b_n)
        u_pair_3 = np.kron(u_a_3, u_b_3)

        eff_all_3 = apply_unitary_adjoint(effects_all_3d, u_pair_3)
        eff_true_3 = apply_unitary_adjoint(effects_true_3d, u_pair_3)
        eff_mask_3 = apply_unitary_adjoint_masked(effects_mask_3d, u_pair_3)

        eff_all_5 = {key: embed_3d_to_5d(effect) for key, effect in eff_all_3.items()}
        eff_true_5 = {key: embed_3d_to_5d(effect) for key, effect in eff_true_3.items()}
        eff_mask_5 = map_effects_masked(eff_mask_3, embed_3d_to_5d)

        if k_filter is not None:
            eff_all_5 = apply_local_channel_adjoint(eff_all_5, k_filter, k_filter)
            eff_true_5 = apply_local_channel_adjoint(eff_true_5, k_filter, k_filter)
            eff_mask_5 = apply_local_channel_adjoint_masked(eff_mask_5, k_filter, k_filter)

        eff_all_5 = apply_unitary_adjoint(eff_all_5, u_qfc_pair)
        eff_true_5 = apply_unitary_adjoint(eff_true_5, u_qfc_pair)
        eff_mask_5 = apply_unitary_adjoint_masked(eff_mask_5, u_qfc_pair)

        effects_all_by_bin.append(eff_all_5)
        effects_true_by_bin.append(eff_true_5)
        effects_mask_by_bin.append(eff_mask_5)

    return effects_all_by_bin, effects_true_by_bin, effects_mask_by_bin


@lru_cache(maxsize=4)
def _bs_gate_1517() -> np.ndarray:
    """
    内部函数：1517_A × 1517_B上的50/50分束器（36x36）。

    这是仅作用于通信子空间的核心BS门。

    Returns
    -------
    np.ndarray
        1517_A × 1517空间的36x36幺正矩阵（每个站点6×6）
    """
    def make_generator(mode_id: int) -> np.ndarray:
        """为单个偏振模式构造BS生成元。"""
        c = annihilation_op(SUBSPACE_1517, mode_id)  # 6x6
        c_dag = creation_op(SUBSPACE_1517, mode_id)  # 6x6

        # 在联合1517_A × 1517_B空间（36D）上构造算符
        c_A = np.kron(c, np.eye(6, dtype=complex))
        c_B = np.kron(np.eye(6, dtype=complex), c)
        c_dag_A = np.kron(c_dag, np.eye(6, dtype=complex))
        c_dag_B = np.kron(np.eye(6, dtype=complex), c_dag)

        # BS生成元：G = θ * (c_A^† c_B - c_A c_B^†)
        # For a 50:50 beam splitter, sin^2(theta) = 0.5 -> theta = pi/4
        theta = np.pi / 4
        G = theta * (c_dag_A @ c_B - c_A @ c_dag_B)
        return G

    # 为H和V偏振生成生成元
    G_H = make_generator(mode_id=0)  # H偏振
    G_V = make_generator(mode_id=1)  # V偏振

    # 总生成元（对两种偏振求和）
    G_total = G_H + G_V

    # 指数化得到幺正
    U_bs = expm(G_total)

    return U_bs


def emission_gate(
    gamma: float,
    dt: float,
    Alpha: np.ndarray,
    phase: float = 0.0,
    H_sys: Optional[np.ndarray] = None,
    bin_first: bool = False
) -> np.ndarray:
    """
    原子-光子纠缠的发射门 U_emit（嵌入5D bin空间）。

        U_emit = exp(√(dt) * (L ⊗ b^†_780 - L^† ⊗ b_780))

    其中 L = √gamma * (alpha_+ * S_+ + alpha_- * S_-)
    且 S_± 是原子跃迁算符。

    门嵌入5D bin空间：仅作用于 (vac, H_780, V_780) 子块，
    对 (H_1517, V_1517) 子块保持单位。

    这在原子态和发射光子偏振之间创建纠缠。
    发射的光子在780nm子空间中，稍后可通过QFC
    转换为1517nm。

    Parameters
    ----------
    gamma : float
        此时间步的单通道发射率（总发射率的一半）
    dt : float
        时间仓宽度
    Alpha : np.ndarray
        从原子跃迁到H/V的2x2偏振映射矩阵
        [[alpha_H+, alpha_H-], [alpha_V+, alpha_V-]]
    phase : float
        发射波包的相位（会同时作用于H/V通道）
    H_sys : np.ndarray, optional
        原子系统哈密顿量（4x4），用于在单步门中同时加入驱动与失谐
    bin_first : bool
        如果为 True，返回 I_1517 ⊗ U_12x12（作用于 bin × atom）
        如果为 False，返回 U_12x12 ⊗ I_1517（作用于 atom × bin）

    Returns
    -------
    np.ndarray
        20x20 幺正矩阵
        - bin_first=False: 作用在 原子(4D) × bin(5D)
        - bin_first=True: 作用在 bin(5D) × 原子(4D)

    Examples
    --------
    >>> # 示例：圆偏振映射
    >>> Alpha = np.array([[1, 0], [0, 1]])  # σ+ -> H, σ- -> V
    >>> U = emission_gate(gamma=0.1, dt=1.0, Alpha=Alpha)
    """
    # ------------------------------------------------------------------
    # 发射门的结构（碰撞模型离散化）：
    #   U_emit = exp[ sqrt(dt) * (L ⊗ b^† - L^† ⊗ b)  - i dt (H_sys ⊗ I) ]
    #
    # 其中：
    #   L = sqrt(gamma) * (alpha_+ S_+ + alpha_- S_-)
    #   b^† 是 780nm 光子的产生算符（单光子截断）
    #
    # 该门在“原子 × 780”上是 12x12，再嵌入到 5D bin 得到 20x20。
    # ------------------------------------------------------------------
    # 原子跃迁算符
    S_plus = atom_transition('+')  # |0><e|
    S_minus = atom_transition('-')  # |1><e|

    # 提取Alpha矩阵元素
    alpha_H_plus = Alpha[0, 0]
    alpha_H_minus = Alpha[0, 1]
    alpha_V_plus = Alpha[1, 0]
    alpha_V_minus = Alpha[1, 1]

    # 在原子(4D)上构造L算符
    # L = √gamma * (alpha_H+ * S_+ + alpha_H- * S_-) 用于H偏振
    # V偏振同理
    sqrt_gamma = np.sqrt(gamma)
    phase_factor = np.exp(1j * phase) if phase != 0.0 else 1.0

    L_H = phase_factor * sqrt_gamma * (alpha_H_plus * S_plus + alpha_H_minus * S_minus)
    L_V = phase_factor * sqrt_gamma * (alpha_V_plus * S_plus + alpha_V_minus * S_minus)

    # 780上的光子算符（3D：vac, H, V）
    # b^†_H = |H><vac|
    bH_dag = np.zeros((3, 3), dtype=complex)
    bH_dag[1, 0] = 1.0
    bH = bH_dag.conj().T

    bV_dag = np.zeros((3, 3), dtype=complex)
    bV_dag[2, 0] = 1.0
    bV = bV_dag.conj().T

    # 生成元：G = √dt * (L_H ⊗ b_H^† + L_V ⊗ b_V^† - h.c.)
    sqrt_dt = np.sqrt(dt)

    G_H = sqrt_dt * (np.kron(L_H, bH_dag) - np.kron(L_H.conj().T, bH))
    G_V = sqrt_dt * (np.kron(L_V, bV_dag) - np.kron(L_V.conj().T, bV))
    G_12x12 = G_H + G_V

    d_atom = L_H.shape[0]
    if H_sys is not None:
        if H_sys.shape != (d_atom, d_atom):
            raise ValueError(f"H_sys 维度应为 ({d_atom},{d_atom})，实际为 {H_sys.shape}")
        G_sys = -1j * dt * np.kron(H_sys, np.eye(3, dtype=complex))
        G_12x12 = G_12x12 + G_sys

    # 指数化得到原子×780上的幺正
    U_12x12 = expm(G_12x12)

    # U_12x12 作用在 atom(4D) × 780(3D) 上，形状 (12, 12)
    # Reshape 为 (d_atom, d_780, d_atom, d_780)
    U_12x12_4d = U_12x12.reshape(d_atom, 3, d_atom, 3)

    if bin_first:
        # bin × atom: (5D bin) × atom
        # 在 bin-first 索引下嵌入 780 子块，其余 1517 分量保持单位
        U_20 = np.eye(5 * d_atom, dtype=complex)
        for iatom in range(d_atom):
            for i780 in range(3):
                row = i780 * d_atom + iatom
                for jatom in range(d_atom):
                    for j780 in range(3):
                        col = j780 * d_atom + jatom
                        U_20[row, col] = U_12x12_4d[iatom, i780, jatom, j780]
    else:
        # atom × bin: atom × (5D bin)
        # 在 bin-last 索引下嵌入 780 子块，其余 1517 分量保持单位
        U_20 = np.eye(d_atom * 5, dtype=complex)
        for iatom in range(d_atom):
            for i780 in range(3):
                row = iatom * 5 + i780
                for jatom in range(d_atom):
                    for j780 in range(3):
                        col = jatom * 5 + j780
                        U_20[row, col] = U_12x12_4d[iatom, i780, jatom, j780]

    return U_20
