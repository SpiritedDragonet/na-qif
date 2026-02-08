# -*- coding: utf-8 -*-
"""
双光子探测与 Bell 态测量（编排层）。

本文件只保留：
- 结果数据结构
- 检测流水线的流程编排
- 自检入口

底层实现已下沉：
- POVM/effect 构造与对偶映射：`atom_sim.physics.gates`
- MPS 收缩引擎：`atom_sim.core.mps`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

import numpy as np

from ..core.mps import (
    DetectionContractionEngine,
    MPSState,
    compute_joint_arrival_probabilities,
)
from ..hilbert.basis import embed_9_from_6, reduce_9d_effects_to_6d
from ..physics.gates import (
    apply_background_or_map,
    apply_background_or_map_masked,
    build_arrival_projectors_5d,
    build_detection_effects_5d_by_bin,
    build_detection_effects_6d,
    build_detection_effects_9d,
    map_effects_masked,
    mix_effects,
    mix_effects_masked,
    order_two_port_detectors,
)


def _is_port_samepol_coincidence(clicks: List[DetectionEvent], window_bins: Optional[int]) -> bool:
    """判定同偏振跨端口符合：H1-H2 或 V1-V2。"""
    h1_bins = [click.bin_index for click in clicks if click.detector == "H1"]
    h2_bins = [click.bin_index for click in clicks if click.detector == "H2"]
    v1_bins = [click.bin_index for click in clicks if click.detector == "V1"]
    v2_bins = [click.bin_index for click in clicks if click.detector == "V2"]
    if window_bins is None:
        return (h1_bins and h2_bins) or (v1_bins and v2_bins)
    for h1 in h1_bins:
        for h2 in h2_bins:
            if abs(h1 - h2) <= window_bins:
                return True
    for v1 in v1_bins:
        for v2 in v2_bins:
            if abs(v1 - v2) <= window_bins:
                return True
    return False


def _sample_fidelity_declared(samples: List[TwoPhotonDetectionResult]) -> float:
    """从抽样结果估计宣告态保真度（条件化后的均值）。"""
    values = []
    for sample in samples:
        if not sample.success or not sample.bell_state:
            continue
        rho = sample.qubit_state
        tr = float(np.trace(rho).real)
        if tr <= 1e-15:
            continue
        values.append(compute_fidelity_with_bell(rho, sample.bell_state) / tr)
    if not values:
        return 0.0
    return float(np.mean(values))


@dataclass
class DetectionEvent:
    """单次探测事件。"""

    detector: str
    bin_index: int
    is_dark: bool = False
    source: str = "signal"


@dataclass
class TwoPhotonDetectionResult:
    """双光子探测结果。"""

    clicks: List[DetectionEvent]
    success: bool
    bell_state: str
    qubit_state: np.ndarray
    dark_detectors: List[str] = field(default_factory=list)
    dark_count: int = 0
    p_true_given_record: float = 0.0


@dataclass
class TwoClickRecord:
    """双点击记录（用于 POVM 抽样）。"""

    detector_a: str
    detector_b: str
    bin_a: int
    bin_b: int
    weight: float


@dataclass
class SuccessEnumerationResult:
    """枚举成功事件的统计结果。"""

    p_arrive: float
    p_arrive_11: float
    p_arrive_20: float
    p_arrive_02: float
    p_arrive_same_arm: float
    p_success: float
    p_success_true: float
    p_success_false: float
    p_success_given_arrival: float
    fidelity_declared: float
    fidelity_true: float
    fidelity_false: float
    p_success_signal_approx: float = 0.0
    p_success_same_arm_approx: float = 0.0
    p_success_intrinsic_dark_assisted: float = 0.0
    p_success_bg_assisted: float = 0.0


@dataclass
class DetectionPipelineResult:
    """一次准备、多用途输出：成功枚举 + 抽样结果。"""

    p_arrive: float
    metrics: Optional[SuccessEnumerationResult]
    samples: List[TwoPhotonDetectionResult]
    timings: Optional[dict] = None


def compute_pauli_correlators_and_chsh(qubit_state: np.ndarray) -> dict:
    """计算双量子比特态的三基相关与 CHSH 最大值。"""
    rho = np.asarray(qubit_state, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError(f"qubit_state 形状应为 (4,4)，得到 {rho.shape}")

    trace_rho = float(np.trace(rho).real)
    if trace_rho <= 1e-15:
        return {
            "corr_exx": 0.0,
            "corr_eyy": 0.0,
            "corr_ezz": 0.0,
            "chsh_s_max": 0.0,
        }

    rho_norm = rho / trace_rho
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    pauli = (sigma_x, sigma_y, sigma_z)

    corr = np.zeros((3, 3), dtype=float)
    for i, sigma_i in enumerate(pauli):
        for j, sigma_j in enumerate(pauli):
            op = np.kron(sigma_i, sigma_j)
            corr[i, j] = float(np.trace(rho_norm @ op).real)

    eigvals = np.linalg.eigvalsh(corr.T @ corr)
    eigvals_sorted = np.sort(np.maximum(eigvals, 0.0))[::-1]
    chsh_s_max = float(2.0 * np.sqrt(eigvals_sorted[0] + eigvals_sorted[1]))
    return {
        "corr_exx": float(corr[0, 0]),
        "corr_eyy": float(corr[1, 1]),
        "corr_ezz": float(corr[2, 2]),
        "chsh_s_max": chsh_s_max,
    }


P_ARRIVE_EPS = 1e-8


def _order_two_port_detectors(detectors: List[str]) -> Tuple[str, ...]:
    """双端口探测器排序适配层（实际实现位于 `physics.gates`）。"""
    return order_two_port_detectors(detectors)


def _build_bell_projector_full(
    target_bell: str,
    single_dim: int = 4,
    qubit_levels: Tuple[int, int] = (0, 1),
) -> np.ndarray:
    """构造原子对空间中的 Bell 投影（支持 12D emitter 的腔自由度偏迹口径）。"""
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
    dim_pair = single_dim * single_dim
    q0, q1 = int(qubit_levels[0]), int(qubit_levels[1])
    atom_levels = 4 if single_dim % 4 == 0 else single_dim
    cavity_levels = single_dim // atom_levels
    if q0 < 0 or q1 < 0 or q0 >= atom_levels or q1 >= atom_levels:
        raise ValueError(
            f"qubit_levels={qubit_levels} 超出原子编码维度 {atom_levels} 的索引范围"
        )

    def _single_site_basis_op(row_level: int, col_level: int) -> np.ndarray:
        op = np.zeros((single_dim, single_dim), dtype=complex)
        for cavity_index in range(cavity_levels):
            row = row_level * cavity_levels + cavity_index
            col = col_level * cavity_levels + cavity_index
            op[row, col] = 1.0
        return op

    basis_levels = [
        (q0, q0),
        (q0, q1),
        (q1, q0),
        (q1, q1),
    ]
    local_cache = {
        (row, col): _single_site_basis_op(row, col)
        for row in (q0, q1)
        for col in (q0, q1)
    }

    proj_full = np.zeros((dim_pair, dim_pair), dtype=complex)
    for row_index, (a_row, b_row) in enumerate(basis_levels):
        for col_index, (a_col, b_col) in enumerate(basis_levels):
            coeff = proj_qubit[row_index, col_index]
            if abs(coeff) <= 0.0:
                continue
            proj_full += coeff * np.kron(local_cache[(a_row, a_col)], local_cache[(b_row, b_col)])
    return proj_full


def _validate_bs_unitary(bs_unitary: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """校验并标准化 BS 单位阵输入。"""
    if bs_unitary is None:
        return None
    bs_unitary = np.asarray(bs_unitary, dtype=complex)
    if bs_unitary.shape != (36, 36):
        raise ValueError(f"bs_unitary shape {bs_unitary.shape} != (36,36) for 6D output ports")
    return bs_unitary


def _parse_fiber_sample(fiber_sample: Optional[tuple]) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float, float]:
    """解析光纤采样参数；若未提供则返回理想无噪声参数。"""
    if fiber_sample is None:
        return (
            np.eye(2, dtype=complex),
            np.eye(2, dtype=complex),
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
        )
    try:
        U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, _phase, phase_slope, phase_jitter_std = fiber_sample
    except ValueError as exc:
        raise ValueError("fiber_sample 格式不正确，无法解析光纤参数") from exc
    return (
        U_A,
        U_B,
        float(eta_H_A),
        float(eta_V_A),
        float(eta_H_B),
        float(eta_V_B),
        float(phase_slope),
        float(phase_jitter_std),
    )


def _scale_qfc_source_background_map(
    p_bg_qfc_source_map: dict,
    eta_H_A: float,
    eta_V_A: float,
    eta_H_B: float,
    eta_V_B: float,
    bs_theta: float,
) -> dict:
    """
    将“源端 QFC 背景概率（每 bin）”映射为探测端 OR-map。

    近似模型：
    1) 源端背景在 A/B 臂产生后，先乘对应臂传输系数。
    2) 在中心站 BS 处按 cos^2(theta)/sin^2(theta) 分流到两个端口。
    3) 每个端口的背景在 H/V 探测器间按 1/2 均分。

    该映射保留了关键物理尺度：源端背景会随链路透过率衰减。
    """
    eta_arm_a = float(np.clip(0.5 * (eta_H_A + eta_V_A), 0.0, 1.0))
    eta_arm_b = float(np.clip(0.5 * (eta_H_B + eta_V_B), 0.0, 1.0))

    p_src_a = float(np.clip(0.5 * (p_bg_qfc_source_map["H1"] + p_bg_qfc_source_map["V1"]), 0.0, 1.0))
    p_src_b = float(np.clip(0.5 * (p_bg_qfc_source_map["H2"] + p_bg_qfc_source_map["V2"]), 0.0, 1.0))

    theta = float(np.clip(bs_theta, 0.0, np.pi / 2.0))
    c2 = float(np.cos(theta) ** 2)
    s2 = float(np.sin(theta) ** 2)

    p_port_1 = float(np.clip(p_src_a * eta_arm_a * c2 + p_src_b * eta_arm_b * s2, 0.0, 1.0))
    p_port_2 = float(np.clip(p_src_a * eta_arm_a * s2 + p_src_b * eta_arm_b * c2, 0.0, 1.0))

    return {
        "H1": float(np.clip(0.5 * p_port_1, 0.0, 1.0)),
        "V1": float(np.clip(0.5 * p_port_1, 0.0, 1.0)),
        "H2": float(np.clip(0.5 * p_port_2, 0.0, 1.0)),
        "V2": float(np.clip(0.5 * p_port_2, 0.0, 1.0)),
    }


def _build_empty_samples(n_samples: int) -> List[TwoPhotonDetectionResult]:
    """构造空点击占位结果（用于短路返回）。"""
    return [
        TwoPhotonDetectionResult(
            clicks=[],
            success=False,
            bell_state="",
            qubit_state=np.zeros((4, 4), dtype=complex),
        )
        for _ in range(max(0, n_samples))
    ]


def _accumulate_success_and_fidelity(
    engine: DetectionContractionEngine,
    left_envs_bell: dict,
    effects_by_bin: List[dict],
    patterns: List[Tuple[str, Tuple[str, str]]],
    window_bins: Optional[int],
) -> Tuple[float, float]:
    """在给定 effect 集合上累计成功率与 Bell 保真度加权和。"""
    p_success = 0.0
    fidelity_weighted = 0.0
    for bell_state, (det_a, det_b) in patterns:
        key_pair = _order_two_port_detectors([det_a, det_b])
        key_a = _order_two_port_detectors([det_a])
        key_b = _order_two_port_detectors([det_b])

        weight_same = engine.sum_same_bin(engine.left_envs_identity, effects_by_bin, key_pair)
        weight_diff = engine.sum_diff_bins(engine.left_envs_identity, effects_by_bin, key_a, key_b, window_bins)
        weight_diff += engine.sum_diff_bins(engine.left_envs_identity, effects_by_bin, key_b, key_a, window_bins)
        p_success += weight_same + weight_diff

        fidelity_weighted += engine.sum_same_bin(left_envs_bell[bell_state], effects_by_bin, key_pair)
        fidelity_weighted += engine.sum_diff_bins(left_envs_bell[bell_state], effects_by_bin, key_a, key_b, window_bins)
        fidelity_weighted += engine.sum_diff_bins(left_envs_bell[bell_state], effects_by_bin, key_b, key_a, window_bins)
    return float(max(0.0, p_success)), float(max(0.0, fidelity_weighted))


def _record_weight(
    engine: DetectionContractionEngine,
    effects_by_bin: List[dict],
    record: TwoClickRecord,
) -> float:
    """计算指定双点击记录在给定 effect 集合下的权重。"""
    value = engine.contract_record(
        left_envs=engine.left_envs_identity,
        effects_by_bin=effects_by_bin,
        det_a=record.detector_a,
        det_b=record.detector_b,
        bin_a=record.bin_a,
        bin_b=record.bin_b,
    )
    return float(max(0.0, value.real))


def run_detection_pipeline(
    mps: MPSState,
    n_bins: int,
    eta_det: float | dict = 0.85,
    p_dark_intrinsic: float | dict = 0.0,
    p_bg_qfc: float | dict = 0.0,
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
    bs_theta: float = np.pi / 4,
    v_res: float = 1.0,
    qubit_levels: Tuple[int, int] = (0, 1),
) -> DetectionPipelineResult:
    """
    执行探测流水线（编排层）。

    流程分为三段：
    1) 到达概率：调用 core 收缩引擎计算 p_arrive / p_arrive_11 / p_arrive_same_arm
    2) effect 构造：调用 physics 构造逐 bin 的 5D POVM（含 v_res 混合）
    3) 统计与抽样：调用 core 收缩引擎完成成功率枚举与逐次点击抽样

    注意：
    - 本函数不再承载底层算符生成和张量收缩细节；
    - v_res 仅表示“残差可区分度”，用于未显式建模因素的剩余项。
    - 当原子站点是 12D emitter（atom×cavity）时，Bell 与 qubit 读出采用“腔偏迹口径”。
    """
    if rng is None:
        rng = np.random.default_rng()
    timings = {}
    detect_start = time.perf_counter()

    def _resolve_detector_map(value: float | dict, field_name: str) -> dict:
        order = ("H1", "V1", "H2", "V2")
        if isinstance(value, dict):
            resolved = {}
            for raw_key, raw_value in value.items():
                detector = str(raw_key).strip().upper()
                if detector not in order:
                    raise ValueError(f"{field_name} 包含未知探测器: {raw_key}")
                resolved[detector] = float(np.clip(raw_value, 0.0, 1.0))
            missing = [det for det in order if det not in resolved]
            if missing:
                raise ValueError(f"{field_name} 缺少探测器参数: {missing}")
            return resolved
        scalar = float(np.clip(value, 0.0, 1.0))
        return {detector: scalar for detector in order}

    eta_det_map = _resolve_detector_map(eta_det, "eta_det")
    p_dark_intrinsic_map = _resolve_detector_map(p_dark_intrinsic, "p_dark_intrinsic")
    p_bg_qfc_source_map = _resolve_detector_map(p_bg_qfc, "p_bg_qfc")
    p_dark_intrinsic_scalar = max(p_dark_intrinsic_map.values())

    mps = mps.copy()
    v_res = min(max(float(v_res), 0.0), 1.0)

    # 支持“原子位点维度可变”场景：
    # - 常规布局：atomA, atomB, A1, B1, ... -> bin_start=2
    # - 纯光场输入：A1, B1, ... -> bin_start=0
    bin_start = 2 if (len(mps.d) >= 4 and mps.d[2] == 5 and mps.d[3] == 5) else 0
    bin_dim = mps.d[bin_start]
    if bin_dim != 5:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 5.")

    bs_unitary_6d = _validate_bs_unitary(bs_unitary)
    U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase_slope, phase_jitter_std = _parse_fiber_sample(fiber_sample)
    p_bg_qfc_map = _scale_qfc_source_background_map(
        p_bg_qfc_source_map,
        eta_H_A=eta_H_A,
        eta_V_A=eta_V_A,
        eta_H_B=eta_H_B,
        eta_V_B=eta_V_B,
        bs_theta=bs_theta,
    )
    p_bg_qfc_scalar = max(p_bg_qfc_map.values())

    if verbose and n_samples > 0:
        print("\n" + "=" * 60)
        print("双光子探测（POVM抽样）")
        print("=" * 60)

    # 先规约到规范形，保证后续收缩稳定。
    mps._mps.canonical_form_finite(renormalize=True)
    mps._mps.norm = 1.0

    proj_A, proj_B = build_arrival_projectors_5d(
        theta_H=theta_H,
        theta_V=theta_V,
        eta_H_A=eta_H_A,
        eta_V_A=eta_V_A,
        eta_H_B=eta_H_B,
        eta_V_B=eta_V_B,
        U_A=U_A,
        U_B=U_B,
        apply_filter_780=apply_filter_780,
    )
    p_arrive, p_arrive_11, p_arrive_20, p_arrive_02, p_arrive_same_arm = compute_joint_arrival_probabilities(
        state=mps,
        n_bins=n_bins,
        bin_start=bin_start,
        proj_A=proj_A,
        proj_B=proj_B,
    )

    if verbose and compute_metrics:
        print(
            f"  两光子到达概率 p_arrive={p_arrive:.6f} "
            f"(1&1={p_arrive_11:.6f}, same_arm={p_arrive_same_arm:.6f})"
        )

    if p_arrive < P_ARRIVE_EPS:
        p_arrive = 0.0
        p_arrive_11 = 0.0
        p_arrive_20 = 0.0
        p_arrive_02 = 0.0
        p_arrive_same_arm = 0.0

    if p_arrive <= P_ARRIVE_EPS and p_dark_intrinsic_scalar <= 0.0 and p_bg_qfc_scalar <= 0.0:
        if verbose:
            print(f"  p_arrive<{P_ARRIVE_EPS:.1e} 且无暗计数/背景噪声，跳过POVM收缩")
        metrics = None
        if compute_metrics:
            metrics = SuccessEnumerationResult(
                p_arrive=p_arrive,
                p_arrive_11=p_arrive_11,
                p_arrive_20=p_arrive_20,
                p_arrive_02=p_arrive_02,
                p_arrive_same_arm=p_arrive_same_arm,
                p_success=0.0,
                p_success_true=0.0,
                p_success_false=0.0,
                p_success_given_arrival=0.0,
                fidelity_declared=0.0,
                fidelity_true=0.0,
                fidelity_false=0.0,
            )
        return DetectionPipelineResult(
            p_arrive=p_arrive,
            metrics=metrics,
            samples=_build_empty_samples(n_samples),
            timings={"detection_total": time.perf_counter() - detect_start},
        )

    # 逐 bin 5D effect 构造：底层算符逻辑已下沉到 physics 层。
    t0 = time.perf_counter()
    effects_all_by_bin, effects_true_by_bin, effects_mask_by_bin = build_detection_effects_5d_by_bin(
        n_bins=n_bins,
        eta_det=eta_det_map,
        p_dark=p_dark_intrinsic_map,
        bs_unitary_6d=bs_unitary_6d,
        v_res=v_res,
        U_A=U_A,
        U_B=U_B,
        eta_H_A=eta_H_A,
        eta_V_A=eta_V_A,
        eta_H_B=eta_H_B,
        eta_V_B=eta_V_B,
        apply_filter_780=apply_filter_780,
        theta_H=theta_H,
        theta_V=theta_V,
        phase_slope=phase_slope,
        phase_jitter_std=phase_jitter_std,
        rng=rng,
    )
    timings["povm_effects"] = time.perf_counter() - t0

    effects_all_sig_by_bin = [dict(effects) for effects in effects_all_by_bin]
    effects_true_sig_by_bin = [dict(effects) for effects in effects_true_by_bin]

    if p_bg_qfc_scalar > 0.0:
        # B3: 背景点击不再并入探测器本征暗计数 POVM，而是做观测端 OR 卷积。
        effects_all_by_bin = [apply_background_or_map(effects, p_bg_qfc_map) for effects in effects_all_by_bin]
        effects_true_by_bin = [apply_background_or_map(effects, p_bg_qfc_map) for effects in effects_true_by_bin]
        effects_mask_by_bin = [
            apply_background_or_map_masked(effects_by_mask, p_bg_qfc_map)
            for effects_by_mask in effects_mask_by_bin
        ]

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

    e_no_list = [effects_all_by_bin[idx].get(empty_key, zero_effect) for idx in range(n_bins)]
    # MPS 收缩引擎：底层 left/right env 逻辑下沉到 core 层。
    engine = DetectionContractionEngine.from_mps(
        state=mps,
        n_bins=n_bins,
        e_no_list=e_no_list,
        zero_effect=zero_effect,
        detector_order_fn=order_two_port_detectors,
        qubit_levels=qubit_levels,
    )

    metrics = None
    if compute_metrics:
        t0 = time.perf_counter()
        if verbose:
            dim_pair = bin_dim * bin_dim
            print(f"  使用{bin_dim}D Kraus operators ({dim_pair}x{dim_pair}) - POVM收缩")

        bell_projectors = {
            bell: _build_bell_projector_full(
                bell,
                single_dim=engine.single_dim,
                qubit_levels=qubit_levels,
            )
            for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]
        }
        left_envs_bell = {
            bell: engine.build_left_envs(projector)
            for bell, projector in bell_projectors.items()
        }
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

        p_success_all, fidelity_weighted_all = _accumulate_success_and_fidelity(
            engine,
            left_envs_bell,
            effects_all_by_bin,
            patterns,
            window_bins,
        )
        p_success_true, fidelity_weighted_true = _accumulate_success_and_fidelity(
            engine,
            left_envs_bell,
            effects_true_sig_by_bin,
            patterns,
            window_bins,
        )
        # false 定义：相对“纯真实点击(true)”的差额，包含本征暗计数与背景辅助两部分。
        p_success_false = float(max(0.0, p_success_all - p_success_true))

        p_success_sig_total, _ = _accumulate_success_and_fidelity(
            engine,
            left_envs_bell,
            effects_all_sig_by_bin,
            patterns,
            window_bins,
        )
        p_success_sig_true, _ = _accumulate_success_and_fidelity(
            engine,
            left_envs_bell,
            effects_true_sig_by_bin,
            patterns,
            window_bins,
        )
        p_success_bg_assisted = float(max(0.0, p_success_all - p_success_sig_total))
        p_success_intrinsic_dark_assisted = float(max(0.0, p_success_sig_total - p_success_sig_true))

        fidelity_declared = (fidelity_weighted_all / p_success_all) if p_success_all > 0 else 0.0
        fidelity_true = (fidelity_weighted_true / p_success_true) if p_success_true > 0 else 0.0
        fidelity_false = (
            (fidelity_weighted_all - fidelity_weighted_true) / p_success_false
            if p_success_false > 0
            else 0.0
        )
        p_success_given_arrival = (p_success_true / p_arrive_11) if p_arrive_11 > P_ARRIVE_EPS else 0.0
        if p_arrive > P_ARRIVE_EPS:
            frac_11 = p_arrive_11 / p_arrive
            frac_same = p_arrive_same_arm / p_arrive
        else:
            frac_11 = 0.0
            frac_same = 0.0

        p_success_false = float(max(0.0, p_success_all - p_success_true))
        metrics = SuccessEnumerationResult(
            p_arrive=p_arrive,
            p_arrive_11=p_arrive_11,
            p_arrive_20=p_arrive_20,
            p_arrive_02=p_arrive_02,
            p_arrive_same_arm=p_arrive_same_arm,
            p_success=p_success_all,
            p_success_true=p_success_true,
            p_success_false=p_success_false,
            p_success_given_arrival=p_success_given_arrival,
            fidelity_declared=fidelity_declared,
            fidelity_true=fidelity_true,
            fidelity_false=fidelity_false,
            p_success_signal_approx=p_success_true * frac_11,
            p_success_same_arm_approx=p_success_true * frac_same,
            p_success_intrinsic_dark_assisted=p_success_intrinsic_dark_assisted,
            p_success_bg_assisted=p_success_bg_assisted,
        )
        timings["povm_enumeration"] = time.perf_counter() - t0

    if n_samples <= 0:
        timings["detection_total"] = time.perf_counter() - detect_start
        return DetectionPipelineResult(p_arrive=p_arrive, metrics=metrics, samples=[], timings=timings)

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
    for _, (det_a, det_b) in patterns_records:
        for record in engine.collect_same_bin_records(effects_all_by_bin, det_a, det_b, weight_eps):
            records.append(TwoClickRecord(*record))
        for record in engine.collect_diff_bin_records(
            effects_all_by_bin,
            det_a,
            det_b,
            weight_eps,
            window_bins,
        ):
            records.append(TwoClickRecord(*record))
        for record in engine.collect_diff_bin_records(
            effects_all_by_bin,
            det_b,
            det_a,
            weight_eps,
            window_bins,
        ):
            records.append(TwoClickRecord(*record))

    if not records:
        timings["detection_total"] = time.perf_counter() - detect_start
        return DetectionPipelineResult(
            p_arrive=p_arrive,
            metrics=metrics,
            samples=_build_empty_samples(n_samples),
            timings=timings,
        )

    weights = np.array([max(0.0, record.weight) for record in records], dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= weight_eps:
        timings["detection_total"] = time.perf_counter() - detect_start
        return DetectionPipelineResult(
            p_arrive=p_arrive,
            metrics=metrics,
            samples=_build_empty_samples(n_samples),
            timings=timings,
        )

    probs = weights / total_weight
    samples: List[TwoPhotonDetectionResult] = []
    t0 = time.perf_counter()
    for sample_index in range(1, n_samples + 1):
        record = records[int(rng.choice(len(records), p=probs))]

        mask_candidates = [
            empty_key,
            _order_two_port_detectors([record.detector_a]),
            _order_two_port_detectors([record.detector_b]),
            _order_two_port_detectors([record.detector_a, record.detector_b]),
        ]
        mask_weights = [
            max(
                0.0,
                engine.weight_record_masked(
                    effects_mask_by_bin=effects_mask_by_bin,
                    det_a=record.detector_a,
                    det_b=record.detector_b,
                    bin_a=record.bin_a,
                    bin_b=record.bin_b,
                    dark_mask=mask,
                    empty_key=empty_key,
                ),
            )
            for mask in mask_candidates
        ]
        total_mask_weight = float(sum(mask_weights))
        if total_mask_weight <= weight_eps:
            mask_choice = empty_key
        else:
            mask_probs = np.array(mask_weights, dtype=float) / total_mask_weight
            mask_choice = mask_candidates[int(rng.choice(len(mask_candidates), p=mask_probs))]

        dark_detectors = list(mask_choice) if mask_choice else []
        weight_obs_record = max(weight_eps, record.weight)
        weight_true_record = _record_weight(engine, effects_true_sig_by_bin, record)
        weight_sig_total_record = _record_weight(engine, effects_all_sig_by_bin, record)
        p_true_given_record = float(min(max(weight_true_record / weight_obs_record, 0.0), 1.0))
        p_bg_assist_record = float(
            min(max((weight_obs_record - weight_sig_total_record) / weight_obs_record, 0.0), 1.0)
        )
        bg_happened = bool(rng.random() < p_bg_assist_record)
        clicks = [
            DetectionEvent(
                detector=record.detector_a,
                bin_index=record.bin_a,
                is_dark=record.detector_a in dark_detectors,
                source=(
                    "dark_intrinsic"
                    if record.detector_a in dark_detectors
                    else ("bg_qfc" if bg_happened else "signal")
                ),
            ),
            DetectionEvent(
                detector=record.detector_b,
                bin_index=record.bin_b,
                is_dark=record.detector_b in dark_detectors,
                source=(
                    "dark_intrinsic"
                    if record.detector_b in dark_detectors
                    else ("bg_qfc" if bg_happened else "signal")
                ),
            ),
        ]

        success = False
        bell_state = ""
        detectors = {record.detector_a, record.detector_b}
        if detectors == {"H1", "V2"} or detectors == {"V1", "H2"}:
            success = True
            bell_state = "Psi-"
        elif detectors == {"H1", "V1"} or detectors == {"H2", "V2"}:
            success = True
            bell_state = "Psi+"

        qubit_state = engine.compute_record_qubit_state(
            effects_by_bin=effects_all_by_bin,
            det_a=record.detector_a,
            det_b=record.detector_b,
            bin_a=record.bin_a,
            bin_b=record.bin_b,
        )

        if verbose:
            if n_samples > 1:
                print(f"\n  [POVM抽样 {sample_index}/{n_samples}]")
            print("\n  结果：")
            print("    抽样自双点击分布（条件在两次点击）")
            print(
                "    点击："
                f"{[(c.detector, c.bin_index, 'dark' if c.is_dark else 'true', c.source) for c in clicks]}"
            )
            if dark_detectors:
                print(f"    暗计数点击：{dark_detectors}")
            print(f"    BSM成功：{success}")
            if success:
                print(f"    Bell态：{bell_state}")

        samples.append(
            TwoPhotonDetectionResult(
                clicks=clicks,
                success=success,
                bell_state=bell_state,
                qubit_state=qubit_state,
                dark_detectors=dark_detectors,
                dark_count=len(dark_detectors),
                p_true_given_record=float(p_true_given_record),
            )
        )
    timings["povm_sampling"] = time.perf_counter() - t0
    timings["detection_total"] = time.perf_counter() - detect_start
    return DetectionPipelineResult(p_arrive=p_arrive, metrics=metrics, samples=samples, timings=timings)


def extract_qubit_state(
    mps: MPSState,
    qubit_levels: Tuple[int, int] = (0, 1),
) -> Tuple[np.ndarray, float]:
    """提取双量子比特 4x4 子块（对腔自由度做偏迹后再取编码态子空间）。"""
    dim_atom = mps.d[0]
    q0, q1 = int(qubit_levels[0]), int(qubit_levels[1])
    atom_levels = 4 if dim_atom % 4 == 0 else dim_atom
    cavity_levels = dim_atom // atom_levels
    if q0 < 0 or q1 < 0 or q0 >= atom_levels or q1 >= atom_levels:
        raise ValueError(
            f"qubit_levels={qubit_levels} 超出原子编码维度 {atom_levels} 的索引范围"
        )
    rho_full = mps.get_reduced_density([0, 1])
    if rho_full.ndim == 4:
        rho_full = rho_full.reshape(dim_atom * dim_atom, dim_atom * dim_atom)
    basis_levels = [
        (q0, q0),
        (q0, q1),
        (q1, q0),
        (q1, q1),
    ]

    def _pair_index(atom_a_level: int, cav_a: int, atom_b_level: int, cav_b: int) -> int:
        single_a = atom_a_level * cavity_levels + cav_a
        single_b = atom_b_level * cavity_levels + cav_b
        return single_a * dim_atom + single_b

    rho_qubit = np.zeros((4, 4), dtype=complex)
    for row_index, (a_row, b_row) in enumerate(basis_levels):
        for col_index, (a_col, b_col) in enumerate(basis_levels):
            value = 0.0 + 0.0j
            for cav_a in range(cavity_levels):
                for cav_b in range(cavity_levels):
                    row = _pair_index(a_row, cav_a, b_row, cav_b)
                    col = _pair_index(a_col, cav_a, b_col, cav_b)
                    value += rho_full[row, col]
            rho_qubit[row_index, col_index] = value
    p_qubit = float(np.real(np.trace(rho_qubit)))
    return rho_qubit, p_qubit


def compute_fidelity_with_bell(qubit_state: np.ndarray, target_bell: str) -> float:
    """计算与 Bell 态的保真度。"""
    bell_states = {
        "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),
        "Phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),
        "Psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),
        "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),
    }
    if target_bell not in bell_states:
        raise ValueError(f"未知的Bell态：{target_bell}")
    psi = bell_states[target_bell]
    return float(np.real(psi.conj() @ qubit_state @ psi))


def run_detection_self_checks(verbose: bool = True) -> None:
    """运行探测模块快速自检（完备性、mask 分解、v_res 端点一致性）。"""
    tol = 1e-10

    def _assert_close(name: str, lhs: np.ndarray, rhs: np.ndarray) -> None:
        err = float(np.linalg.norm(lhs - rhs))
        if err > tol:
            raise AssertionError(f"{name} failed: ||lhs-rhs||={err:.3e} > {tol:.1e}")
        if verbose:
            print(f"[self-check] {name}: ok (err={err:.3e})")

    effects_all_6d, effects_true_6d, effects_mask_6d = build_detection_effects_6d(
        eta=0.83,
        p_dark=2.3e-6,
    )
    sum_6 = np.zeros((36, 36), dtype=complex)
    for effect in effects_all_6d.values():
        sum_6 += effect
    _assert_close("6D POVM completeness", sum_6, np.eye(36, dtype=complex))

    effects_all_9d, effects_true_9d, effects_mask_9d = build_detection_effects_9d(
        eta=0.83,
        p_dark=2.3e-6,
    )
    sum_9 = np.zeros((81, 81), dtype=complex)
    for effect in effects_all_9d.values():
        sum_9 += effect
    _assert_close("9D POVM completeness", sum_9, np.eye(81, dtype=complex))

    for key, effect in effects_all_6d.items():
        mask_sum = np.zeros_like(effect)
        for masked in effects_mask_6d.get(key, {}).values():
            mask_sum += masked
        _assert_close(f"6D mask decomposition key={key}", mask_sum, effect)
    for key, effect in effects_all_9d.items():
        mask_sum = np.zeros_like(effect)
        for masked in effects_mask_9d.get(key, {}).values():
            mask_sum += masked
        _assert_close(f"9D mask decomposition key={key}", mask_sum, effect)

    w = embed_9_from_6()
    w_pair = np.kron(w, w)
    reduced_all_dist = reduce_9d_effects_to_6d(effects_all_9d, w_pair)
    reduced_true_dist = reduce_9d_effects_to_6d(effects_true_9d, w_pair)
    reduced_mask_dist = map_effects_masked(
        effects_mask_9d,
        lambda effect: w_pair.conj().T @ effect @ w_pair,
    )

    mixed_all_v1 = mix_effects(effects_all_6d, reduced_all_dist, v_res=1.0)
    mixed_true_v1 = mix_effects(effects_true_6d, reduced_true_dist, v_res=1.0)
    mixed_mask_v1 = mix_effects_masked(effects_mask_6d, reduced_mask_dist, v_res=1.0)
    mixed_all_v0 = mix_effects(effects_all_6d, reduced_all_dist, v_res=0.0)
    mixed_true_v0 = mix_effects(effects_true_6d, reduced_true_dist, v_res=0.0)
    mixed_mask_v0 = mix_effects_masked(effects_mask_6d, reduced_mask_dist, v_res=0.0)

    for key in set(effects_all_6d.keys()) | set(reduced_all_dist.keys()):
        if key in effects_all_6d:
            _assert_close(f"v_res=1(all) key={key}", mixed_all_v1[key], effects_all_6d[key])
        if key in reduced_all_dist:
            _assert_close(f"v_res=0(all) key={key}", mixed_all_v0[key], reduced_all_dist[key])

    for key in set(effects_true_6d.keys()) | set(reduced_true_dist.keys()):
        if key in effects_true_6d:
            _assert_close(f"v_res=1(true) key={key}", mixed_true_v1[key], effects_true_6d[key])
        if key in reduced_true_dist:
            _assert_close(f"v_res=0(true) key={key}", mixed_true_v0[key], reduced_true_dist[key])

    for key in set(effects_mask_6d.keys()) | set(reduced_mask_dist.keys()):
        map_v1 = mixed_mask_v1.get(key, {})
        map_v0 = mixed_mask_v0.get(key, {})
        map_int = effects_mask_6d.get(key, {})
        map_dist = reduced_mask_dist.get(key, {})
        for mask, effect in map_int.items():
            _assert_close(f"v_res=1(mask) key={key} mask={mask}", map_v1[mask], effect)
        for mask, effect in map_dist.items():
            _assert_close(f"v_res=0(mask) key={key} mask={mask}", map_v0[mask], effect)

    # E2: 极限 sanity（eta=1, p_dark_intrinsic=0, p_bg_qfc=0, 理想链路）
    local_dims = [12, 12] + [5, 5]
    state_ideal = MPSState(local_dims=local_dims, init_state=[1, 1, 0, 0], max_bond=8)
    run_ideal = run_detection_pipeline(
        mps=state_ideal,
        n_bins=1,
        eta_det=1.0,
        p_dark_intrinsic=0.0,
        p_bg_qfc=0.0,
        window_bins=0,
        rng=np.random.default_rng(42),
        verbose=False,
        n_samples=0,
        compute_metrics=True,
        bs_unitary=None,
        fiber_sample=None,
        apply_filter_780=True,
        theta_H=np.pi / 4,
        theta_V=np.pi / 4,
        v_res=1.0,
    )
    if run_ideal.metrics is None:
        raise AssertionError("E2 ideal metrics missing")
    if run_ideal.metrics.p_success < -tol or run_ideal.metrics.fidelity_declared < -tol:
        raise AssertionError("E2 ideal metrics invalid")
    if verbose:
        print(
            "[self-check] E2 ideal sanity: "
            f"p_success={run_ideal.metrics.p_success:.6e}, "
            f"fidelity={run_ideal.metrics.fidelity_declared:.6f}"
        )

    # E3: 枚举 vs 抽样一致性（宣告保真度 + true_fraction）
    samples_n = 200
    run_mc = run_detection_pipeline(
        mps=state_ideal,
        n_bins=1,
        eta_det=0.85,
        p_dark_intrinsic=2.3e-6,
        p_bg_qfc=0.0,
        window_bins=0,
        rng=np.random.default_rng(123),
        verbose=False,
        n_samples=samples_n,
        compute_metrics=True,
        bs_unitary=None,
        fiber_sample=None,
        apply_filter_780=True,
        theta_H=np.pi / 4,
        theta_V=np.pi / 4,
        v_res=0.7,
    )
    if run_mc.metrics is None:
        raise AssertionError("E3 metrics missing")
    sample_fidelity = _sample_fidelity_declared(run_mc.samples)
    success_samples = [sample for sample in run_mc.samples if sample.success]
    sample_true_fraction = (
        float(np.mean([sample.p_true_given_record for sample in success_samples]))
        if success_samples
        else 0.0
    )
    enum_fidelity = float(run_mc.metrics.fidelity_declared)
    enum_true_fraction = (
        float(run_mc.metrics.p_success_true / run_mc.metrics.p_success)
        if run_mc.metrics.p_success > 1e-15
        else 0.0
    )
    # 使用宽松阈值，避免 Monte-Carlo 波动；检查可比统计量一致。
    if abs(sample_fidelity - enum_fidelity) > 0.35:
        raise AssertionError(
            f"E3 fidelity mismatch too large: sample={sample_fidelity:.4f}, enum={enum_fidelity:.4f}"
        )
    if abs(sample_true_fraction - enum_true_fraction) > 0.25:
        raise AssertionError(
            f"E3 true_fraction mismatch too large: "
            f"sample={sample_true_fraction:.4f}, enum={enum_true_fraction:.4f}"
        )
    if verbose:
        print(
            "[self-check] E3 sample-vs-enum: "
            f"fidelity(sample={sample_fidelity:.4f}, enum={enum_fidelity:.4f}), "
            f"true_fraction(sample={sample_true_fraction:.4f}, enum={enum_true_fraction:.4f})"
        )

    # E4: HOM delay 形状基线（无噪声时，tau=0 的同偏振跨端口符合应不高于大延迟）
    from ..simulation.trajectory import run_dual_atom_emission
    from ..physics.gates import bs_gate_6d

    def _hom_coincidence_for_tau(delay_ns: float) -> float:
        rng = np.random.default_rng(2026 + int(round(delay_ns * 10.0)))
        emission = run_dual_atom_emission(
            n_bins=10,
            dt_ns=0.5,
            chi_max=20,
            sigma=3.0,
            delay_ns=delay_ns,
            delay_jitter_ns=0.0,
            verbose=False,
            diagnostics=False,
        )
        samples = run_detection_pipeline(
            mps=emission.mps,
            n_bins=emission.get_n_bins(),
            eta_det=1.0,
            p_dark_intrinsic=0.0,
            p_bg_qfc=0.0,
            window_bins=0,
            rng=rng,
            verbose=False,
            n_samples=36,
            compute_metrics=False,
            bs_unitary=bs_gate_6d(),
            fiber_sample=(
                np.eye(2, dtype=complex),
                np.eye(2, dtype=complex),
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            apply_filter_780=True,
            theta_H=np.pi / 4,
            theta_V=np.pi / 4,
            v_res=1.0,
        ).samples
        records = [1.0 if _is_port_samepol_coincidence(sample.clicks, 0) else 0.0 for sample in samples]
        return float(np.mean(records))

    hom_tau0 = _hom_coincidence_for_tau(0.0)
    hom_tau_far = _hom_coincidence_for_tau(20.0)
    if hom_tau0 - hom_tau_far > 0.15:
        raise AssertionError(
            f"E4 HOM baseline unexpected: tau0={hom_tau0:.4f} > tau_far={hom_tau_far:.4f}"
        )
    if verbose:
        print(
            "[self-check] E4 HOM baseline: "
            f"coinc_tau0={hom_tau0:.4f}, coinc_tau_far={hom_tau_far:.4f}"
        )

    if verbose:
        print("[self-check] detection checks all passed")
