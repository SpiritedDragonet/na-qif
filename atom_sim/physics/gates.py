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
    loss_channel_1517_single_photon,
)


# 缓存的门（这些计算昂贵且不随每仓变化）

@lru_cache(maxsize=32)
def qfc_gate(
    theta_H: float = 0.0,
    theta_V: float = 0.0,
    phi_H: float = 0.0,
    phi_V: float = 0.0,
) -> np.ndarray:
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
    phase_H = np.exp(1j * float(phi_H))
    phase_V = np.exp(1j * float(phi_V))

    # H 偏振：|H_780> <-> |H_1517>
    U[1, 1] = cH
    U[1, 3] = -phase_H.conjugate() * sH
    U[3, 1] = phase_H * sH
    U[3, 3] = cH

    # V 偏振：|V_780> <-> |V_1517>
    U[2, 2] = cV
    U[2, 4] = -phase_V.conjugate() * sV
    U[4, 2] = phase_V * sV
    U[4, 4] = cV

    return U


def filter_cavity_rt(
    fwhm_hz: float,
    dt_s: float,
    detuning_hz: float = 0.0,
) -> Tuple[complex, float]:
    """由滤波腔线宽/失谐/离散步长计算单步记忆系数 (r, t)。"""
    fwhm_hz = float(fwhm_hz)
    dt_s = float(dt_s)
    detuning_hz = float(detuning_hz)
    if fwhm_hz <= 0.0:
        raise ValueError(f"fwhm_hz 必须 > 0，得到 {fwhm_hz}")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s 必须 > 0，得到 {dt_s}")

    # 单极点滤波器离散化：
    #   r = exp[-(pi*Delta_nu + i*2pi*delta) * dt]
    # 其中 Delta_nu 为强度 FWHM，delta 为失谐。
    r = np.exp(-(np.pi * fwhm_hz + 1j * 2.0 * np.pi * detuning_hz) * dt_s)
    t = float(np.sqrt(max(0.0, 1.0 - abs(r) ** 2)))
    return complex(r), t


def filter_cavity_step_unitary_5d3d(
    r: complex,
    t: float,
) -> np.ndarray:
    """构造 (bin_5d ⊗ mem_3d) 的单步记忆门（15x15）。"""
    r = complex(r)
    t = float(t)
    if t < 0.0:
        raise ValueError(f"t 必须 >= 0，得到 {t}")
    if abs(abs(r) ** 2 + t * t - 1.0) > 1e-8:
        raise ValueError("(r, t) 不满足 |r|^2 + t^2 = 1，无法构造幺正步进门")

    unitary = np.eye(15, dtype=complex)

    # 5D bin 基序：|vac>, |H_780>, |V_780>, |H_1517>, |V_1517>
    # 3D mem 基序：|vac>, |H>, |V>
    def _index(bin_level: int, mem_level: int) -> int:
        return int(bin_level * 3 + mem_level)

    h_bin_mem = _index(3, 0)  # |H_1517, vac_mem>
    h_vac_mem = _index(0, 1)  # |vac_bin, H_mem>
    v_bin_mem = _index(4, 0)  # |V_1517, vac_mem>
    v_vac_mem = _index(0, 2)  # |vac_bin, V_mem>

    # H 子块
    unitary[h_bin_mem, h_bin_mem] = r
    unitary[h_bin_mem, h_vac_mem] = -t
    unitary[h_vac_mem, h_bin_mem] = t
    unitary[h_vac_mem, h_vac_mem] = np.conj(r)

    # V 子块
    unitary[v_bin_mem, v_bin_mem] = r
    unitary[v_bin_mem, v_vac_mem] = -t
    unitary[v_vac_mem, v_bin_mem] = t
    unitary[v_vac_mem, v_vac_mem] = np.conj(r)

    return unitary



@lru_cache(maxsize=32)
def bs_gate_6d(theta: float = np.pi / 4) -> np.ndarray:
    """
    6D 输出端口（1517nm）的分束器门（36x36）。

    该门用于测量端共轭：BS 后的端口需要容纳 2 光子态。

    Returns
    -------
    np.ndarray
        1517_A × 1517_B空间的36x36幺正矩阵（每个站点6×6）
    """
    # ------------------------------------------------------------------
    # 该 BS 只作用在 1517nm 子空间 (6D)：
    #   - H/V 各自按同一 theta 做 beamsplitter
    #   - 多光子态 (|2H>, |HV>, |2V>) 通过算符指数自然包含
    # ------------------------------------------------------------------
    return _bs_gate_1517(float(theta))


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
def _bs_gate_9d_dist_cached(theta: float = np.pi / 4) -> np.ndarray:
    return _bs_gate_9d_dist_from_6d(bs_gate_6d(theta))


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


_TWO_PORT_DETECTOR_ORDER = ("H1", "V1", "H2", "V2")


def _resolve_two_port_probability_map(value: float | dict, field_name: str) -> dict:
    """将标量或 detector->prob 映射统一解析为 H1/V1/H2/V2 四通道概率。"""
    if isinstance(value, dict):
        resolved = {}
        for raw_key, raw_value in value.items():
            detector = str(raw_key).strip().upper()
            if detector not in _TWO_PORT_DETECTOR_ORDER:
                raise ValueError(f"{field_name} 包含未知探测器: {raw_key}")
            prob = float(raw_value)
            if prob < 0.0 or prob > 1.0:
                raise ValueError(f"{field_name}[{detector}] 必须在 [0, 1] 内，得到 {prob}")
            resolved[detector] = prob
        missing = [det for det in _TWO_PORT_DETECTOR_ORDER if det not in resolved]
        if missing:
            raise ValueError(f"{field_name} 缺少探测器参数: {missing}")
        return resolved

    prob = float(value)
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"{field_name} 必须在 [0, 1] 内，得到 {prob}")
    return {detector: prob for detector in _TWO_PORT_DETECTOR_ORDER}


def _enumerate_two_port_subsets() -> List[Tuple[str, ...]]:
    """枚举双端口探测器全集的所有子集（含空集）。"""
    subsets: List[Tuple[str, ...]] = []
    n_det = len(_TWO_PORT_DETECTOR_ORDER)
    for mask in range(1 << n_det):
        subset = [
            _TWO_PORT_DETECTOR_ORDER[idx]
            for idx in range(n_det)
            if (mask >> idx) & 1
        ]
        subsets.append(order_two_port_detectors(subset))
    return subsets


_TWO_PORT_SUBSETS = _enumerate_two_port_subsets()


def apply_background_or_map(effects: dict, p_bg_click: float | dict) -> dict:
    """
    对单 bin 的 effect 集合施加背景点击 OR 后处理。

    记 A 为“信号+本征暗计数”点击集合，B 为独立背景点击集合，
    观测集合 S = A OR B。则

        E_obs[S] = Σ_A P(S | A) E_sig[A]

    其中 P(S | A) 由每个探测器以概率 p_bg_click 独立触发给出。
    """
    if not effects:
        return {}
    p_bg_map = _resolve_two_port_probability_map(p_bg_click, "p_bg_click")
    if max(p_bg_map.values()) <= 0.0:
        return effects

    mapped: dict = {}
    detector_set = set(_TWO_PORT_DETECTOR_ORDER)
    for src_key, src_effect in effects.items():
        src_set = set(src_key)
        if not src_set.issubset(detector_set):
            raise ValueError(f"未知探测器标签: {tuple(sorted(src_set - detector_set))}")
        for dst_key in _TWO_PORT_SUBSETS:
            dst_set = set(dst_key)
            if not src_set.issubset(dst_set):
                continue
            prob = 1.0
            for detector in _TWO_PORT_DETECTOR_ORDER:
                if detector in src_set:
                    continue
                p_bg_local = p_bg_map[detector]
                if detector in dst_set:
                    prob *= p_bg_local
                else:
                    prob *= (1.0 - p_bg_local)
            if prob <= 0.0:
                continue
            mapped[dst_key] = mapped.get(dst_key, 0) + prob * src_effect
    return mapped


def apply_background_or_map_masked(effects_by_mask: dict, p_bg_click: float | dict) -> dict:
    """
    对按 intrinsic-dark mask 分组的 effect 施加背景 OR 后处理。

    注意：该函数只做“观测点击集合”重映射，不引入新的背景来源标签；
    mask 仍表示本征暗计数来源。
    """
    if not effects_by_mask:
        return {}
    p_bg_map = _resolve_two_port_probability_map(p_bg_click, "p_bg_click")
    if max(p_bg_map.values()) <= 0.0:
        return effects_by_mask

    mapped: dict = {}
    detector_set = set(_TWO_PORT_DETECTOR_ORDER)
    for src_key, src_mask_map in effects_by_mask.items():
        src_set = set(src_key)
        if not src_set.issubset(detector_set):
            raise ValueError(f"未知探测器标签: {tuple(sorted(src_set - detector_set))}")
        for dst_key in _TWO_PORT_SUBSETS:
            dst_set = set(dst_key)
            if not src_set.issubset(dst_set):
                continue
            prob = 1.0
            for detector in _TWO_PORT_DETECTOR_ORDER:
                if detector in src_set:
                    continue
                p_bg_local = p_bg_map[detector]
                if detector in dst_set:
                    prob *= p_bg_local
                else:
                    prob *= (1.0 - p_bg_local)
            if prob <= 0.0:
                continue
            dst_mask_map = mapped.setdefault(dst_key, {})
            for dark_mask, src_effect in src_mask_map.items():
                dst_mask_map[dark_mask] = dst_mask_map.get(dark_mask, 0) + prob * src_effect
    return mapped


def build_detection_effects_6d(
    eta: float | dict,
    p_dark: float | dict = 0.0,
) -> Tuple[dict, dict, dict]:
    """构造 6D 端口空间的探测 POVM（含暗计数拆分）。"""

    eta_map = _resolve_two_port_probability_map(eta, "eta")
    p_dark_map = _resolve_two_port_probability_map(p_dark, "p_dark")

    def _order_detectors(detectors: List[str]) -> List[str]:
        order = {"H": 0, "V": 1}
        return sorted(detectors, key=lambda detector: order[detector])

    def _split_with_dark(
        kraus: np.ndarray,
        detectors: List[str],
        p_dark_local: dict,
    ) -> List[Tuple[np.ndarray, List[str], List[str]]]:
        for det_key in ("H", "V"):
            value = float(p_dark_local[det_key])
            if value < 0.0 or value > 1.0:
                raise ValueError(f"p_dark[{det_key}] 必须在 [0, 1] 内，得到 {value}")

        base_detectors = _order_detectors(detectors)
        if max(float(p_dark_local["H"]), float(p_dark_local["V"])) <= 0.0:
            return [(kraus, base_detectors, [])]

        off_detectors = [detector for detector in ("H", "V") if detector not in base_detectors]
        entries = []
        for mask in product([0, 1], repeat=len(off_detectors)):
            prob = 1.0
            dark_detectors = []
            for detector, use_dark in zip(off_detectors, mask):
                p_dark_detector = float(p_dark_local[detector])
                if use_dark:
                    prob *= p_dark_detector
                    dark_detectors.append(detector)
                else:
                    prob *= (1 - p_dark_detector)
            if prob <= 0:
                continue
            combined_detectors = _order_detectors(base_detectors + dark_detectors)
            entries.append((np.sqrt(prob) * kraus, combined_detectors, _order_detectors(dark_detectors)))
        return entries

    def _build_port_kraus_entries_6d(
        eta_h_local: float,
        eta_v_local: float,
        p_dark_h_local: float,
        p_dark_v_local: float,
    ) -> List[Tuple[np.ndarray, List[str], List[str]]]:
        p_dark_local = {"H": float(p_dark_h_local), "V": float(p_dark_v_local)}
        k00_6d = np.diag([
            1.0,
            np.sqrt(1 - eta_h_local),
            np.sqrt(1 - eta_v_local),
            (1 - eta_h_local),
            (1 - eta_v_local),
            np.sqrt((1 - eta_h_local) * (1 - eta_v_local)),
        ]).astype(complex)

        k10a_6d = np.zeros((6, 6), dtype=complex)
        k10a_6d[0, 1] = np.sqrt(eta_h_local)
        k10b_6d = np.zeros((6, 6), dtype=complex)
        k10b_6d[0, 3] = np.sqrt(1 - (1 - eta_h_local) ** 2)
        k10c_6d = np.zeros((6, 6), dtype=complex)
        k10c_6d[2, 5] = np.sqrt(eta_h_local * (1 - eta_v_local))

        k01a_6d = np.zeros((6, 6), dtype=complex)
        k01a_6d[0, 2] = np.sqrt(eta_v_local)
        k01b_6d = np.zeros((6, 6), dtype=complex)
        k01b_6d[0, 4] = np.sqrt(1 - (1 - eta_v_local) ** 2)
        k01c_6d = np.zeros((6, 6), dtype=complex)
        k01c_6d[1, 5] = np.sqrt((1 - eta_h_local) * eta_v_local)

        k11_6d = np.zeros((6, 6), dtype=complex)
        k11_6d[0, 5] = np.sqrt(eta_h_local * eta_v_local)

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

    port_entries_1 = _build_port_kraus_entries_6d(
        eta_map["H1"],
        eta_map["V1"],
        p_dark_map["H1"],
        p_dark_map["V1"],
    )
    port_entries_2 = _build_port_kraus_entries_6d(
        eta_map["H2"],
        eta_map["V2"],
        p_dark_map["H2"],
        p_dark_map["V2"],
    )
    kraus_list: List[np.ndarray] = []
    outcome_detectors: List[List[str]] = []
    outcome_dark: List[List[str]] = []

    for k1, det1, dark1 in port_entries_1:
        for k2, det2, dark2 in port_entries_2:
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
    eta: float | dict,
    p_dark: float | dict = 0.0,
) -> Tuple[dict, dict, dict]:
    """构造 9D 标签空间的探测 POVM（含暗计数拆分）。"""

    eta_map = _resolve_two_port_probability_map(eta, "eta")
    p_dark_map = _resolve_two_port_probability_map(p_dark, "p_dark")

    def _order_detectors(detectors: List[str]) -> List[str]:
        order = {"H": 0, "V": 1}
        return sorted(detectors, key=lambda detector: order[detector])

    basis = [(a, b) for a in (0, 1, 2) for b in (0, 1, 2)]
    dim = 9

    def _build_port_entries_9d(
        eta_h_local: float,
        eta_v_local: float,
        p_dark_h_local: float,
        p_dark_v_local: float,
    ) -> list:
        effects_by_mask_port = {}
        for idx, (a_pol, b_pol) in enumerate(basis):
            n_h = int(a_pol == 1) + int(b_pol == 1)
            n_v = int(a_pol == 2) + int(b_pol == 2)
            p_h_click = 1.0 - (1.0 - eta_h_local) ** n_h
            p_v_click = 1.0 - (1.0 - eta_v_local) ** n_v

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
                        p_dark_local = p_dark_h_local if detector == "H" else p_dark_v_local
                        if use_dark:
                            prob *= p_dark_local
                            dark_detectors.append(detector)
                        else:
                            prob *= (1.0 - p_dark_local)
                    if prob <= 0.0:
                        continue
                    dark_detectors = _order_detectors(dark_detectors)
                    final_detectors = _order_detectors(base_detectors + dark_detectors)
                    mask_map = effects_by_mask_port.setdefault(tuple(final_detectors), {})
                    effect = mask_map.get(tuple(dark_detectors))
                    if effect is None:
                        effect = np.zeros((dim, dim), dtype=complex)
                        mask_map[tuple(dark_detectors)] = effect
                    effect[idx, idx] += prob

        entries = []
        for key, mask_map in effects_by_mask_port.items():
            for mask, effect in mask_map.items():
                entries.append((effect, list(key), list(mask)))
        return entries

    port_entries_1 = _build_port_entries_9d(
        eta_map["H1"],
        eta_map["V1"],
        p_dark_map["H1"],
        p_dark_map["V1"],
    )
    port_entries_2 = _build_port_entries_9d(
        eta_map["H2"],
        eta_map["V2"],
        p_dark_map["H2"],
        p_dark_map["V2"],
    )

    effects_all = {}
    effects_true = {}
    effects_by_darkmask = {}
    for e1, det1, dark1 in port_entries_1:
        for e2, det2, dark2 in port_entries_2:
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
    eta_H_A: float,
    eta_V_A: float,
    eta_H_B: float,
    eta_V_B: float,
    U_A: Optional[np.ndarray] = None,
    U_B: Optional[np.ndarray] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """构造用于到达概率统计的 (pi0,pi1,pi2)（5D 账本）。"""
    pi0_3 = np.diag([1, 0, 0]).astype(complex)
    pi1_3 = np.diag([0, 1, 1]).astype(complex)
    pi2_3 = np.zeros((3, 3), dtype=complex)

    p_5_from_3 = embed_5_from_3()

    u_a_3 = jones_3d(np.eye(2, dtype=complex) if U_A is None else U_A)
    u_b_3 = jones_3d(np.eye(2, dtype=complex) if U_B is None else U_B)

    def _build_one_arm(eta_h: float, eta_v: float, u_arm_3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_loss = loss_channel_1517_single_photon(eta_h, eta_v)
        p0 = apply_channel_adjoint_single(pi0_3, k_loss)
        p1 = apply_channel_adjoint_single(pi1_3, k_loss)
        p2 = apply_channel_adjoint_single(pi2_3, k_loss)

        p0 = u_arm_3.conj().T @ p0 @ u_arm_3
        p1 = u_arm_3.conj().T @ p1 @ u_arm_3
        p2 = u_arm_3.conj().T @ p2 @ u_arm_3

        p0 = p_5_from_3 @ p0 @ p_5_from_3.conj().T
        p1 = p_5_from_3 @ p1 @ p_5_from_3.conj().T
        p2 = p_5_from_3 @ p2 @ p_5_from_3.conj().T

        return p0, p1, p2

    proj_a = _build_one_arm(eta_H_A, eta_V_A, u_a_3)
    proj_b = _build_one_arm(eta_H_B, eta_V_B, u_b_3)
    return proj_a, proj_b


def build_detection_effects_5d_by_bin(
    n_bins: int,
    eta_det: float | dict,
    p_dark: float | dict,
    bs_unitary_6d: Optional[np.ndarray],
    v_res: float,
    U_A: np.ndarray,
    U_B: np.ndarray,
    eta_H_A: float,
    eta_V_A: float,
    eta_H_B: float,
    eta_V_B: float,
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

        effects_all_by_bin.append(eff_all_5)
        effects_true_by_bin.append(eff_true_5)
        effects_mask_by_bin.append(eff_mask_5)

    return effects_all_by_bin, effects_true_by_bin, effects_mask_by_bin


@lru_cache(maxsize=32)
def _bs_gate_1517(theta: float = np.pi / 4) -> np.ndarray:
    """
    内部函数：1517_A × 1517_B上的可调分束器（36x36）。

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
        # 其中 sin^2(theta) 是跨端口透射概率。
        G = float(theta) * (c_dag_A @ c_B - c_A @ c_dag_B)
        return G

    # 为H和V偏振生成生成元
    G_H = make_generator(mode_id=0)  # H偏振
    G_V = make_generator(mode_id=1)  # V偏振

    # 总生成元（对两种偏振求和）
    G_total = G_H + G_V

    # 指数化得到幺正
    U_bs = expm(G_total)

    return U_bs


def _cavity_annihilation_ops_3d() -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 3D 腔基序 |vac>,|H>,|V> 上的湮灭算符 a_H/a_V。
    """
    a_h = np.zeros((3, 3), dtype=complex)
    a_h[0, 1] = 1.0
    a_v = np.zeros((3, 3), dtype=complex)
    a_v[0, 2] = 1.0
    return a_h, a_v


def build_emitter_operators_12d(
    alpha: np.ndarray,
    g: float,
    h_atom: np.ndarray,
    kappa_ex_H: float,
    kappa_ex_V: float,
    kappa_in_H: float,
    kappa_in_V: float,
    gamma_sigma_plus: float,
    gamma_sigma_minus: float,
    delta_c_H: float = 0.0,
    delta_c_V: float = 0.0,
) -> dict:
    """
    构造 12D emitter（atom4D ⊗ cavity3D）的哈密顿量与通道算符。

    返回：
      - h_emitter: (12,12)
      - l_out: (L_H, L_V)
      - collapse_ops: 不可见损耗 collapse 算符列表
    """
    alpha = np.asarray(alpha, dtype=complex)
    if alpha.shape != (2, 2):
        raise ValueError(f"alpha 形状应为 (2,2)，得到 {alpha.shape}")
    h_atom = np.asarray(h_atom, dtype=complex)
    if h_atom.shape != (4, 4):
        raise ValueError(f"h_atom 形状应为 (4,4)，得到 {h_atom.shape}")

    for name, value in (
        ("kappa_ex_H", kappa_ex_H),
        ("kappa_ex_V", kappa_ex_V),
        ("kappa_in_H", kappa_in_H),
        ("kappa_in_V", kappa_in_V),
        ("gamma_sigma_plus", gamma_sigma_plus),
        ("gamma_sigma_minus", gamma_sigma_minus),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} 必须 >= 0")

    a_h, a_v = _cavity_annihilation_ops_3d()
    a_h_dag = a_h.conj().T
    a_v_dag = a_v.conj().T

    i_atom = np.eye(4, dtype=complex)
    i_cavity = np.eye(3, dtype=complex)

    s_plus = atom_transition('+')
    s_minus = atom_transition('-')

    sigma_h = alpha[0, 0] * s_plus + alpha[0, 1] * s_minus
    sigma_v = alpha[1, 0] * s_plus + alpha[1, 1] * s_minus

    h_cavity = np.diag([0.0, float(delta_c_H), float(delta_c_V)]).astype(complex)
    h_int = float(g) * (
        np.kron(sigma_h, a_h_dag)
        + np.kron(sigma_h.conj().T, a_h)
        + np.kron(sigma_v, a_v_dag)
        + np.kron(sigma_v.conj().T, a_v)
    )
    h_emitter = np.kron(h_atom, i_cavity) + np.kron(i_atom, h_cavity) + h_int

    l_out_h = np.sqrt(float(kappa_ex_H)) * np.kron(i_atom, a_h)
    l_out_v = np.sqrt(float(kappa_ex_V)) * np.kron(i_atom, a_v)

    collapse_ops = []
    if float(kappa_in_H) > 0.0:
        collapse_ops.append(np.sqrt(float(kappa_in_H)) * np.kron(i_atom, a_h))
    if float(kappa_in_V) > 0.0:
        collapse_ops.append(np.sqrt(float(kappa_in_V)) * np.kron(i_atom, a_v))
    if float(gamma_sigma_plus) > 0.0:
        collapse_ops.append(np.sqrt(float(gamma_sigma_plus)) * np.kron(s_plus, i_cavity))
    if float(gamma_sigma_minus) > 0.0:
        collapse_ops.append(np.sqrt(float(gamma_sigma_minus)) * np.kron(s_minus, i_cavity))

    return {
        "h_emitter": h_emitter,
        "l_out": (l_out_h, l_out_v),
        "collapse_ops": collapse_ops,
    }


def emission_gate(
    dt: float,
    h_emitter: np.ndarray,
    l_out_h: np.ndarray,
    l_out_v: np.ndarray,
    phase: float = 0.0,
    bin_first: bool = False
) -> np.ndarray:
    """
    显式腔发射门 U_emit（12D emitter 嵌入 5D bin）。

    门嵌入5D bin空间：仅作用于 (vac, H_780, V_780) 子块，
    对 (H_1517, V_1517) 子块保持单位。

    发射门作用在 emitter×780 子空间（36D），再嵌入到 emitter×bin（60D）。

    Parameters
    ----------
    dt : float
        时间仓宽度
    h_emitter : np.ndarray
        显式 emitter 哈密顿量（12x12）
    l_out_h : np.ndarray
        H 偏振输出耦合算符（12x12）
    l_out_v : np.ndarray
        V 偏振输出耦合算符（12x12）
    phase : float
        发射波包的相位（会同时作用于H/V通道）
    bin_first : bool
        如果为 True，返回 I_1517 ⊗ U_36x36（作用于 bin × emitter）
        如果为 False，返回 U_36x36 ⊗ I_1517（作用于 emitter × bin）

    Returns
    -------
    np.ndarray
        60x60 幺正矩阵
        - bin_first=False: 作用在 emitter(12D) × bin(5D)
        - bin_first=True: 作用在 bin(5D) × emitter(12D)

    Examples
    --------
    >>> # 示例：显式 emitter
    >>> h = np.zeros((12, 12), dtype=complex)
    >>> l = np.zeros((12, 12), dtype=complex)
    >>> U = emission_gate(dt=1e-9, h_emitter=h, l_out_h=l, l_out_v=l)
    """
    if dt <= 0.0:
        raise ValueError("dt 必须 > 0")

    h_emitter = np.asarray(h_emitter, dtype=complex)
    l_out_h = np.asarray(l_out_h, dtype=complex)
    l_out_v = np.asarray(l_out_v, dtype=complex)

    if h_emitter.shape[0] != h_emitter.shape[1]:
        raise ValueError(f"h_emitter 必须为方阵，得到 {h_emitter.shape}")
    dim_emitter = int(h_emitter.shape[0])
    for name, op in (("l_out_h", l_out_h), ("l_out_v", l_out_v)):
        if op.shape != (dim_emitter, dim_emitter):
            raise ValueError(f"{name} 形状应为 ({dim_emitter},{dim_emitter})，得到 {op.shape}")

    phase_factor = np.exp(1j * phase) if phase != 0.0 else 1.0
    l_h = phase_factor * l_out_h
    l_v = phase_factor * l_out_v

    # 780上的光子算符（3D：vac, H, V）
    # b^†_H = |H><vac|
    bH_dag = np.zeros((3, 3), dtype=complex)
    bH_dag[1, 0] = 1.0
    bH = bH_dag.conj().T

    bV_dag = np.zeros((3, 3), dtype=complex)
    bV_dag[2, 0] = 1.0
    bV = bV_dag.conj().T

    # 生成元：G = √dt * (L_H ⊗ b_H^† + L_V ⊗ b_V^† - h.c.) - i dt (H_em ⊗ I)
    sqrt_dt = np.sqrt(dt)

    g_h = sqrt_dt * (np.kron(l_h, bH_dag) - np.kron(l_h.conj().T, bH))
    g_v = sqrt_dt * (np.kron(l_v, bV_dag) - np.kron(l_v.conj().T, bV))
    g_sys = -1j * dt * np.kron(h_emitter, np.eye(3, dtype=complex))
    g_36x36 = g_h + g_v + g_sys

    # 指数化得到 emitter×780 上的幺正
    u_36x36 = expm(g_36x36)

    # Reshape 为 (d_emitter, d_780, d_emitter, d_780)
    u_4d = u_36x36.reshape(dim_emitter, 3, dim_emitter, 3)

    if bin_first:
        # bin × emitter: (5D bin) × emitter
        # 在 bin-first 索引下嵌入 780 子块，其余 1517 分量保持单位
        u_60 = np.eye(5 * dim_emitter, dtype=complex)
        for iatom in range(dim_emitter):
            for i780 in range(3):
                row = i780 * dim_emitter + iatom
                for jatom in range(dim_emitter):
                    for j780 in range(3):
                        col = j780 * dim_emitter + jatom
                        u_60[row, col] = u_4d[iatom, i780, jatom, j780]
    else:
        # emitter × bin: emitter × (5D bin)
        # 在 bin-last 索引下嵌入 780 子块，其余 1517 分量保持单位
        u_60 = np.eye(dim_emitter * 5, dtype=complex)
        for iatom in range(dim_emitter):
            for i780 in range(3):
                row = iatom * 5 + i780
                for jatom in range(dim_emitter):
                    for j780 in range(3):
                        col = jatom * 5 + j780
                        u_60[row, col] = u_4d[iatom, i780, jatom, j780]

    return u_60
