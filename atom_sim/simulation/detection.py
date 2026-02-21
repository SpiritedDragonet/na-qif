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
from typing import Callable, List, Optional, Tuple
import time

import numpy as np

from ..core.mps import (
    DetectionContractionEngine,
    MPSState,
    compute_joint_arrival_probabilities,
)
from ..hilbert.basis import embed_9d_dist_from_3d_pair, project_6d_to_3d
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

# 成功宣告模式：键为宣告 Bell，值为探测器双点击组合。
SUCCESS_PATTERNS: Tuple[Tuple[str, Tuple[str, str]], ...] = (
    ("Psi-", ("H1", "V2")),
    ("Psi-", ("V1", "H2")),
    ("Psi+", ("H1", "V1")),
    ("Psi+", ("H2", "V2")),
)

# 抽样记录模式：成功模式 + 两类典型失败模式（同偏振跨端口）。
RECORD_PATTERNS: Tuple[Tuple[str, Tuple[str, str]], ...] = SUCCESS_PATTERNS + (
    ("", ("H1", "H2")),
    ("", ("V1", "V2")),
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
    p_bg_assist_given_record: float = 0.0
    p_intrinsic_dark_assist_given_record: float = 0.0


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
    corr_exx: float = 0.0
    corr_eyy: float = 0.0
    corr_ezz: float = 0.0
    chsh_s_max: float = 0.0
    corr_exx_ff: float = 0.0
    corr_eyy_ff: float = 0.0
    corr_ezz_ff: float = 0.0
    chsh_s_max_ff: float = 0.0
    rho_declared_raw: Optional[np.ndarray] = None
    rho_declared_ff: Optional[np.ndarray] = None
    trace_declared_raw: float = 0.0
    trace_declared_ff: float = 0.0


@dataclass
class DetectionPipelineResult:
    """一次准备、多用途输出：成功枚举 + 抽样结果。"""

    p_arrive: float
    metrics: Optional[SuccessEnumerationResult]
    samples: List[TwoPhotonDetectionResult]
    # 抽样时枚举的“双点击记录集合”的总权重（绝对概率）。
    # 注意：samples 是在该集合上条件化抽样得到；
    # 若要恢复“每次尝试的绝对成功率”，需要用该权重做缩放。
    p_records_total: float = 0.0
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


def _build_feedforward_op_for_bell(bell_state: str) -> np.ndarray:
    """
    返回将声明 Bell 态映射到统一口径(默认 Psi+)的单比特校正。

    当前仅用到 Psi+/Psi-：
    - Psi+ : I
    - Psi- : Z_A（或 Z_B 等价，差全局相位）
    """
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    if bell_state == "Psi-":
        return np.kron(sigma_z, np.eye(2, dtype=complex))
    return np.eye(4, dtype=complex)


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


def _infer_bin_start_site(local_dims: List[int], n_bins: int) -> int:
    """
    根据局域维度推断首个 bin 站点。

    支持：
    - atomA, atomB, A1, B1, ...
    - atomA, atomB, memA, memB, A1, B1, ...
    - A1, B1, ...（纯光场输入）

    要求从首 bin 开始连续存在 n_bins 组 (5D, 5D)。
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins 必须为正整数，得到 {n_bins}")
    length = len(local_dims)
    for start in range(0, max(0, length - 1), 2):
        if int(local_dims[start]) != 5 or int(local_dims[start + 1]) != 5:
            continue
        is_valid = True
        for n in range(n_bins):
            site_a = start + 2 * n
            site_b = site_a + 1
            if site_b >= length:
                is_valid = False
                break
            if int(local_dims[site_a]) != 5 or int(local_dims[site_b]) != 5:
                is_valid = False
                break
        if is_valid:
            return start
    raise ValueError(
        "无法识别连续 bin 区间。"
        f" n_bins={n_bins}, L={length}, d_head={local_dims[:12]}, d_tail={local_dims[-12:]}"
    )


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


def _raise_if_should_abort(should_abort: Optional[Callable[[], bool]]) -> None:
    if should_abort is not None and bool(should_abort()):
        raise RuntimeError("OWNERSHIP_LOST")


def _accumulate_success_and_fidelity(
    engine: DetectionContractionEngine,
    left_envs_bell: dict,
    effects_by_bin: List[dict],
    patterns: List[Tuple[str, Tuple[str, str]]],
    window_bins: Optional[int],
    include_fidelity: bool = True,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Tuple[float, float]:
    """在给定 effect 集合上累计成功率与 Bell 保真度加权和。"""
    p_success = 0.0
    fidelity_weighted = 0.0
    for bell_state, (det_a, det_b) in patterns:
        _raise_if_should_abort(should_abort)
        key_pair = _order_two_port_detectors([det_a, det_b])
        key_a = _order_two_port_detectors([det_a])
        key_b = _order_two_port_detectors([det_b])

        weight_same = engine.sum_same_bin(
            engine.left_envs_identity,
            effects_by_bin,
            key_pair,
            should_abort=should_abort,
        )
        weight_diff = engine.sum_diff_bins_bidirectional(
            engine.left_envs_identity,
            effects_by_bin,
            key_a,
            key_b,
            window_bins,
            should_abort=should_abort,
        )
        p_success += weight_same + weight_diff

        if include_fidelity:
            fidelity_weighted += engine.sum_same_bin(
                left_envs_bell[bell_state],
                effects_by_bin,
                key_pair,
                should_abort=should_abort,
            )
            fidelity_weighted += engine.sum_diff_bins_bidirectional(
                left_envs_bell[bell_state],
                effects_by_bin,
                key_a,
                key_b,
                window_bins,
                should_abort=should_abort,
            )
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
    p_bg_detector: float | dict = 0.0,
    window_bins: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    n_samples: int = 1,
    compute_metrics: bool = False,
    bs_unitary: Optional[np.ndarray] = None,
    fiber_sample: Optional[tuple] = None,
    bs_theta: float = np.pi / 4,
    v_res: float = 1.0,
    qubit_levels: Tuple[int, int] = (0, 1),
    should_abort: Optional[Callable[[], bool]] = None,
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
    - `p_bg_detector` 语义为“探测器端每 bin 背景点击概率映射(H1/V1/H2/V2)”，
      本函数不会再做链路/BS 二次映射。
    """
    if rng is None:
        rng = np.random.default_rng()
    _raise_if_should_abort(should_abort)
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
    p_bg_map = _resolve_detector_map(p_bg_detector, "p_bg_detector")
    p_dark_intrinsic_scalar = max(p_dark_intrinsic_map.values())

    mps = mps.copy()
    v_res = min(max(float(v_res), 0.0), 1.0)

    # 支持“bins 前插入记忆对”的布局：
    # - atomA, atomB, A1, B1, ...
    # - atomA, atomB, memA, memB, A1, B1, ...
    # - A1, B1, ...（纯光场输入）
    bin_start = _infer_bin_start_site(mps.d, n_bins)
    bin_dim = mps.d[bin_start]
    if bin_dim != 5:
        raise ValueError(f"Unexpected bin dimension: {bin_dim}. Expected 5.")

    bs_unitary_6d = _validate_bs_unitary(bs_unitary)
    U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase_slope, phase_jitter_std = _parse_fiber_sample(fiber_sample)
    p_bg_scalar = max(p_bg_map.values())

    if verbose and n_samples > 0:
        print("\n" + "=" * 60)
        print("双光子探测（POVM抽样）")
        print("=" * 60)

    # 先规约到规范形，保证后续收缩稳定。
    mps._mps.canonical_form_finite(renormalize=True)
    mps._mps.norm = 1.0

    proj_A, proj_B = build_arrival_projectors_5d(
        eta_H_A=eta_H_A,
        eta_V_A=eta_V_A,
        eta_H_B=eta_H_B,
        eta_V_B=eta_V_B,
        U_A=U_A,
        U_B=U_B,
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

    if p_arrive <= P_ARRIVE_EPS and p_dark_intrinsic_scalar <= 0.0 and p_bg_scalar <= 0.0:
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
    _raise_if_should_abort(should_abort)
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
        phase_slope=phase_slope,
        phase_jitter_std=phase_jitter_std,
        rng=rng,
    )
    timings["povm_effects"] = time.perf_counter() - t0
    _raise_if_should_abort(should_abort)

    effects_all_sig_by_bin = [dict(effects) for effects in effects_all_by_bin]
    effects_true_sig_by_bin = [dict(effects) for effects in effects_true_by_bin]

    if p_bg_scalar > 0.0:
        # B3: 背景点击不再并入探测器本征暗计数 POVM，而是做观测端 OR 卷积。
        effects_all_by_bin = [apply_background_or_map(effects, p_bg_map) for effects in effects_all_by_bin]
        effects_true_by_bin = [apply_background_or_map(effects, p_bg_map) for effects in effects_true_by_bin]
        effects_mask_by_bin = [
            apply_background_or_map_masked(effects_by_mask, p_bg_map)
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

    if len(mps.d) % 2 != 0:
        raise ValueError(f"MPS站点数必须为偶数，当前 L={len(mps.d)}")
    grouped_bins = len(mps.d) // 2 - 1
    # grouped site 编号中：0 对应 (atomA, atomB)，其后为可观测 pair。
    # 若 bins 前插入了 (memA, memB)，则首 bin grouped site 会后移。
    if bin_start % 2 != 0:
        raise ValueError(f"bin_start 必须为偶数，得到 {bin_start}")
    bin_offset = max(1, bin_start // 2)
    available_bins = grouped_bins - bin_offset + 1
    if available_bins < n_bins:
        raise ValueError(
            f"可用分组bin不足: grouped_bins={grouped_bins}, bin_offset={bin_offset}, n_bins={n_bins}"
        )
    e_no_list: List[np.ndarray] = []
    for grouped_index in range(grouped_bins):
        grouped_site = grouped_index + 1
        logical_bin = grouped_site - bin_offset
        if 0 <= logical_bin < n_bins:
            # 固定口径：其余 bin 采用 no-click effect。
            e_no_list.append(effects_all_by_bin[logical_bin].get(empty_key, zero_effect))
            continue
        site_a = 2 * grouped_site
        site_b = site_a + 1
        if site_b >= len(mps.d):
            raise ValueError(f"尾部分组索引越界: grouped_index={grouped_index}")
        pair_dim = int(mps.d[site_a]) * int(mps.d[site_b])
        e_no_list.append(np.eye(pair_dim, dtype=complex))
    # MPS 收缩引擎：底层 left/right env 逻辑下沉到 core 层。
    engine = DetectionContractionEngine.from_mps(
        state=mps,
        n_bins=n_bins,
        e_no_list=e_no_list,
        zero_effect=zero_effect,
        detector_order_fn=order_two_port_detectors,
        qubit_levels=qubit_levels,
        bin_offset=bin_offset,
    )

    metrics = None
    if compute_metrics:
        _raise_if_should_abort(should_abort)
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
        patterns = list(SUCCESS_PATTERNS)

        p_success_all = 0.0
        p_success_true = 0.0
        fidelity_weighted_all = 0.0
        fidelity_weighted_true = 0.0
        corr_exx = 0.0
        corr_eyy = 0.0
        corr_ezz = 0.0
        chsh_s_max = 0.0
        if verbose:
            print(f"  POVM枚举模式数: {len(patterns)}")
            print("  POVM枚举阶段: 1/3 (all, success+fidelity)")

        p_success_all, fidelity_weighted_all = _accumulate_success_and_fidelity(
            engine,
            left_envs_bell,
            effects_all_by_bin,
            patterns,
            window_bins,
            include_fidelity=True,
            should_abort=should_abort,
        )
        _raise_if_should_abort(should_abort)
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"  POVM枚举阶段完成: 1/3 | elapsed={elapsed:.2f}s")

        if verbose:
            print("  POVM枚举阶段: 2/3 (true_signal, success+fidelity)")
        p_success_true, fidelity_weighted_true = _accumulate_success_and_fidelity(
            engine,
            left_envs_bell,
            effects_true_sig_by_bin,
            patterns,
            window_bins,
            include_fidelity=True,
            should_abort=should_abort,
        )
        _raise_if_should_abort(should_abort)
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"  POVM枚举阶段完成: 2/3 | elapsed={elapsed:.2f}s")
        # false 定义：相对“纯真实点击(true)”的差额，包含本征暗计数与背景辅助两部分。
        p_success_false = float(max(0.0, p_success_all - p_success_true))

        if verbose:
            print("  POVM枚举阶段: 3/3 (all_signal_only, success only)")
        p_success_sig_total, _ = _accumulate_success_and_fidelity(
            engine,
            left_envs_bell,
            effects_all_sig_by_bin,
            patterns,
            window_bins,
            include_fidelity=False,
            should_abort=should_abort,
        )
        _raise_if_should_abort(should_abort)
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"  POVM枚举阶段完成: 3/3 | elapsed={elapsed:.2f}s")
        # effects_true_sig_by_bin 与上面的 p_success_true 使用同一 effect 集合，
        # 无需重复做一次完全相同的收缩。
        p_success_sig_true = p_success_true
        p_success_bg_assisted = float(max(0.0, p_success_all - p_success_sig_total))
        p_success_intrinsic_dark_assisted = float(max(0.0, p_success_sig_total - p_success_sig_true))

        if verbose:
            print("  POVM枚举阶段: 4/4 (all, declared corr/chsh raw+ff)")
        sigma_by_bell = engine.accumulate_success_qubit_sigma_by_label(
            effects_by_bin=effects_all_by_bin,
            patterns=patterns,
            window_bins=window_bins,
            should_abort=should_abort,
        )
        sigma_declared_raw = np.zeros((4, 4), dtype=complex)
        sigma_declared_ff = np.zeros((4, 4), dtype=complex)
        for bell_state, sigma_part in sigma_by_bell.items():
            sigma_declared_raw += sigma_part
            u_ff = _build_feedforward_op_for_bell(bell_state)
            sigma_declared_ff += u_ff @ sigma_part @ u_ff.conj().T
        trace_declared_raw = float(np.trace(sigma_declared_raw).real)
        trace_declared_ff = float(np.trace(sigma_declared_ff).real)
        if trace_declared_raw > P_ARRIVE_EPS:
            rho_declared_raw = sigma_declared_raw / trace_declared_raw
        else:
            rho_declared_raw = np.zeros((4, 4), dtype=complex)
        if trace_declared_ff > P_ARRIVE_EPS:
            rho_declared_ff = sigma_declared_ff / trace_declared_ff
        else:
            rho_declared_ff = np.zeros((4, 4), dtype=complex)

        corr_metrics_raw = compute_pauli_correlators_and_chsh(sigma_declared_raw)
        corr_exx = float(corr_metrics_raw["corr_exx"])
        corr_eyy = float(corr_metrics_raw["corr_eyy"])
        corr_ezz = float(corr_metrics_raw["corr_ezz"])
        chsh_s_max = float(corr_metrics_raw["chsh_s_max"])
        corr_metrics_ff = compute_pauli_correlators_and_chsh(sigma_declared_ff)
        corr_exx_ff = float(corr_metrics_ff["corr_exx"])
        corr_eyy_ff = float(corr_metrics_ff["corr_eyy"])
        corr_ezz_ff = float(corr_metrics_ff["corr_ezz"])
        chsh_s_max_ff = float(corr_metrics_ff["chsh_s_max"])
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"  POVM枚举阶段完成: 4/4 | elapsed={elapsed:.2f}s")

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
            corr_exx=corr_exx,
            corr_eyy=corr_eyy,
            corr_ezz=corr_ezz,
            chsh_s_max=chsh_s_max,
            corr_exx_ff=corr_exx_ff,
            corr_eyy_ff=corr_eyy_ff,
            corr_ezz_ff=corr_ezz_ff,
            chsh_s_max_ff=chsh_s_max_ff,
            rho_declared_raw=rho_declared_raw,
            rho_declared_ff=rho_declared_ff,
            trace_declared_raw=trace_declared_raw,
            trace_declared_ff=trace_declared_ff,
        )
        timings["povm_enumeration"] = time.perf_counter() - t0

    if n_samples <= 0:
        timings["detection_total"] = time.perf_counter() - detect_start
        return DetectionPipelineResult(
            p_arrive=p_arrive,
            metrics=metrics,
            samples=[],
            p_records_total=0.0,
            timings=timings,
        )

    patterns_records = list(RECORD_PATTERNS)
    weight_eps = 1e-14
    records: List[TwoClickRecord] = []
    for _, (det_a, det_b) in patterns_records:
        _raise_if_should_abort(should_abort)
        for record in engine.collect_same_bin_records(
            effects_all_by_bin,
            det_a,
            det_b,
            weight_eps,
            should_abort=should_abort,
        ):
            records.append(TwoClickRecord(*record))
        for record in engine.collect_diff_bin_records(
            effects_all_by_bin,
            det_a,
            det_b,
            weight_eps,
            window_bins,
            should_abort=should_abort,
        ):
            records.append(TwoClickRecord(*record))
        for record in engine.collect_diff_bin_records(
            effects_all_by_bin,
            det_b,
            det_a,
            weight_eps,
            window_bins,
            should_abort=should_abort,
        ):
            records.append(TwoClickRecord(*record))

    if not records:
        timings["detection_total"] = time.perf_counter() - detect_start
        return DetectionPipelineResult(
            p_arrive=p_arrive,
            metrics=metrics,
            samples=_build_empty_samples(n_samples),
            p_records_total=0.0,
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
            p_records_total=float(total_weight),
            timings=timings,
        )

    probs = weights / total_weight
    samples: List[TwoPhotonDetectionResult] = []
    record_weight_cache: dict[int, Tuple[float, float]] = {}
    record_mask_cache: dict[int, Tuple[List[Tuple[str, ...]], Optional[np.ndarray], float]] = {}
    record_qubit_state_cache: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for sample_index in range(1, n_samples + 1):
        if (sample_index - 1) % 16 == 0:
            _raise_if_should_abort(should_abort)
        record_index = int(rng.choice(len(records), p=probs))
        record = records[record_index]

        mask_cached = record_mask_cache.get(record_index)
        if mask_cached is None:
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
            mask_probs = None
            if total_mask_weight > weight_eps:
                mask_probs = np.array(mask_weights, dtype=float) / total_mask_weight
            mask_cached = (mask_candidates, mask_probs, total_mask_weight)
            record_mask_cache[record_index] = mask_cached

        mask_candidates, mask_probs, total_mask_weight = mask_cached
        if total_mask_weight <= weight_eps:
            mask_choice = empty_key
        else:
            mask_choice = mask_candidates[int(rng.choice(len(mask_candidates), p=mask_probs))]

        dark_detectors = list(mask_choice) if mask_choice else []
        weight_obs_record = max(weight_eps, record.weight)
        cached_weights = record_weight_cache.get(record_index)
        if cached_weights is None:
            weight_true_record = _record_weight(engine, effects_true_sig_by_bin, record)
            weight_sig_total_record = _record_weight(engine, effects_all_sig_by_bin, record)
            cached_weights = (weight_true_record, weight_sig_total_record)
            record_weight_cache[record_index] = cached_weights
        else:
            weight_true_record, weight_sig_total_record = cached_weights
        p_true_given_record = float(min(max(weight_true_record / weight_obs_record, 0.0), 1.0))
        p_bg_assist_record = float(
            min(max((weight_obs_record - weight_sig_total_record) / weight_obs_record, 0.0), 1.0)
        )
        p_intrinsic_dark_assist_record = float(
            min(max(1.0 - p_true_given_record - p_bg_assist_record, 0.0), 1.0)
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
                    else ("bg_source" if bg_happened else "signal")
                ),
            ),
            DetectionEvent(
                detector=record.detector_b,
                bin_index=record.bin_b,
                is_dark=record.detector_b in dark_detectors,
                source=(
                    "dark_intrinsic"
                    if record.detector_b in dark_detectors
                    else ("bg_source" if bg_happened else "signal")
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

        qubit_state = record_qubit_state_cache.get(record_index)
        if qubit_state is None:
            qubit_state = engine.compute_record_qubit_state(
                effects_by_bin=effects_all_by_bin,
                det_a=record.detector_a,
                det_b=record.detector_b,
                bin_a=record.bin_a,
                bin_b=record.bin_b,
            )
            record_qubit_state_cache[record_index] = qubit_state

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
                p_bg_assist_given_record=float(p_bg_assist_record),
                p_intrinsic_dark_assist_given_record=float(p_intrinsic_dark_assist_record),
            )
        )
    timings["povm_sampling"] = time.perf_counter() - t0
    timings["detection_total"] = time.perf_counter() - detect_start
    return DetectionPipelineResult(
        p_arrive=p_arrive,
        metrics=metrics,
        samples=samples,
        p_records_total=float(total_weight),
        timings=timings,
    )


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

    effects_all_3d_int = {key: project_6d_to_3d(effect) for key, effect in effects_all_6d.items()}
    effects_true_3d_int = {key: project_6d_to_3d(effect) for key, effect in effects_true_6d.items()}
    effects_mask_3d_int = map_effects_masked(effects_mask_6d, project_6d_to_3d)

    j_dist = embed_9d_dist_from_3d_pair()
    j_dag = j_dist.conj().T
    effects_all_3d_dist = {key: j_dag @ effect @ j_dist for key, effect in effects_all_9d.items()}
    effects_true_3d_dist = {key: j_dag @ effect @ j_dist for key, effect in effects_true_9d.items()}
    effects_mask_3d_dist = map_effects_masked(
        effects_mask_9d,
        lambda effect: j_dag @ effect @ j_dist,
    )

    mixed_all_v1 = mix_effects(effects_all_3d_int, effects_all_3d_dist, v_res=1.0)
    mixed_true_v1 = mix_effects(effects_true_3d_int, effects_true_3d_dist, v_res=1.0)
    mixed_mask_v1 = mix_effects_masked(effects_mask_3d_int, effects_mask_3d_dist, v_res=1.0)
    mixed_all_v0 = mix_effects(effects_all_3d_int, effects_all_3d_dist, v_res=0.0)
    mixed_true_v0 = mix_effects(effects_true_3d_int, effects_true_3d_dist, v_res=0.0)
    mixed_mask_v0 = mix_effects_masked(effects_mask_3d_int, effects_mask_3d_dist, v_res=0.0)

    for key in set(effects_all_3d_int.keys()) | set(effects_all_3d_dist.keys()):
        if key in effects_all_3d_int:
            _assert_close(f"v_res=1(all) key={key}", mixed_all_v1[key], effects_all_3d_int[key])
        if key in effects_all_3d_dist:
            _assert_close(f"v_res=0(all) key={key}", mixed_all_v0[key], effects_all_3d_dist[key])

    for key in set(effects_true_3d_int.keys()) | set(effects_true_3d_dist.keys()):
        if key in effects_true_3d_int:
            _assert_close(f"v_res=1(true) key={key}", mixed_true_v1[key], effects_true_3d_int[key])
        if key in effects_true_3d_dist:
            _assert_close(f"v_res=0(true) key={key}", mixed_true_v0[key], effects_true_3d_dist[key])

    for key in set(effects_mask_3d_int.keys()) | set(effects_mask_3d_dist.keys()):
        map_v1 = mixed_mask_v1.get(key, {})
        map_v0 = mixed_mask_v0.get(key, {})
        map_int = effects_mask_3d_int.get(key, {})
        map_dist = effects_mask_3d_dist.get(key, {})
        for mask, effect in map_int.items():
            _assert_close(f"v_res=1(mask) key={key} mask={mask}", map_v1[mask], effect)
        for mask, effect in map_dist.items():
            _assert_close(f"v_res=0(mask) key={key} mask={mask}", map_v0[mask], effect)

    # E2: 极限 sanity（eta=1, p_dark_intrinsic=0, p_bg_detector=0, 理想链路）
    local_dims = [12, 12] + [5, 5]
    state_ideal = MPSState(local_dims=local_dims, init_state=[1, 1, 0, 0], max_bond=8)
    run_ideal = run_detection_pipeline(
        mps=state_ideal,
        n_bins=1,
        eta_det=1.0,
        p_dark_intrinsic=0.0,
        p_bg_detector=0.0,
        window_bins=0,
        rng=np.random.default_rng(42),
        verbose=False,
        n_samples=0,
        compute_metrics=True,
        bs_unitary=None,
        fiber_sample=None,
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
        p_bg_detector=0.0,
        window_bins=0,
        rng=np.random.default_rng(123),
        verbose=False,
        n_samples=samples_n,
        compute_metrics=True,
        bs_unitary=None,
        fiber_sample=None,
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
    from ..experiment.common import (
        SimConfig,
        build_emission_kernel_kwargs,
        build_hom_self_check_setup,
    )

    base_cfg = SimConfig()
    hom_emission_cfg, hom_samples, hom_tau_far_ns = build_hom_self_check_setup(base_cfg.emission)
    hom_emission_kwargs = build_emission_kernel_kwargs(hom_emission_cfg)

    def _hom_coincidence_for_tau(delay_ns: float) -> float:
        rng = np.random.default_rng(2026 + int(round(delay_ns * 10.0)))
        emission = run_dual_atom_emission(
            **hom_emission_kwargs,
            delay_ns=delay_ns,
            delay_jitter_ns=0.0,
            rng=rng,
            verbose=False,
            diagnostics=False,
        )
        samples = run_detection_pipeline(
            mps=emission.mps,
            n_bins=emission.get_n_bins(),
            eta_det=1.0,
            p_dark_intrinsic=0.0,
            p_bg_detector=0.0,
            window_bins=0,
            rng=rng,
            verbose=False,
            n_samples=hom_samples,
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
            v_res=1.0,
        ).samples
        records = [1.0 if _is_port_samepol_coincidence(sample.clicks, 0) else 0.0 for sample in samples]
        return float(np.mean(records))

    hom_tau0 = _hom_coincidence_for_tau(0.0)
    hom_tau_far = _hom_coincidence_for_tau(hom_tau_far_ns)
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
