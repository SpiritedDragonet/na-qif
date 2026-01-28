# -*- coding: utf-8 -*-
"""
实验流程中的通用配置与工具函数。
"""

from __future__ import annotations

from typing import Optional, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from ..physics import FiberChannelParams

if TYPE_CHECKING:
    from ..core.mps import MPSState

# 探测噪声默认参数（可用CLI覆盖）
DEFAULT_DARK_RATE_INTRINSIC_HZ = 65.0
DEFAULT_BG_RATE_MEAN_HZ = 165.0
DEFAULT_BG_RATE_STD_HZ = float(np.sqrt(5.0))

# 原子态极端判据
ATOM_EXTREME_EPS = 1e-2


def _get_emission_params(delay_ns: float) -> dict:
    # 发射参数集中管理，避免多处硬编码
    return {
        "n_bins": 100,
        "dt_ns": 0.5,
        "chi_max": 50,
        "gamma_peak_A": 2 * np.pi * 20e6,
        "gamma_peak_B": 2 * np.pi * 20e6,
        "sigma": 10.0,
        "delay_ns": delay_ns,
        "delay_jitter_ns": 0.5,
        "g": 2 * np.pi * 20e6,
        "kappa_ex": 2 * np.pi * 20e6,
        "kappa_in": 2 * np.pi * 1e6,
        "gamma_atom": 2 * np.pi * 3e6,
        "delta_u": 0.0,
        "delta_e": 0.0,
    }


def _make_fiber_params(noise_enabled: bool = True) -> FiberChannelParams:
    # 残余Jones旋转 + 小PDL + 相位斜率噪声
    compensation_sigma = 0.1  # 补偿后的残差旋转（弧度，可调）
    pdl_sigma = 0.02  # 小PDL：H/V透过率相对差异的标准差（线性）
    phase_slope_std = 0.05  # 相位斜率标准差（rad/bin）
    if not noise_enabled:
        # 关闭光纤噪声：保留平均透过率，移除随机偏振/相位/PDL扰动
        return FiberChannelParams(
            polarization_model="perturb",
            polarization_sigma=0.0,
            eta_std=0.0,
            pdl_sigma=0.0,
            phase_drift_std=0.0,
            phase_slope_std=0.0,
            phase_jitter_std=0.0,
        )
    return FiberChannelParams(
        polarization_model="perturb",
        polarization_sigma=compensation_sigma,
        pdl_sigma=pdl_sigma,
        phase_slope_std=phase_slope_std,
    )


def _compute_window_bins(window_ns: float, bin_dt_ns: float) -> int:
    if bin_dt_ns <= 0:
        return 0
    return int(round(window_ns / bin_dt_ns))


def _compute_noise_params(
    noise_cfg: Optional[dict],
    bin_dt_s: float,
    rng: np.random.Generator,
) -> dict:
    if noise_cfg is None:
        noise_cfg = {}
    dark_rate_intrinsic_hz = max(
        0.0, float(noise_cfg.get("dark_rate_intrinsic_hz", DEFAULT_DARK_RATE_INTRINSIC_HZ))
    )
    bg_rate_mean_hz = max(
        0.0, float(noise_cfg.get("bg_rate_mean_hz", DEFAULT_BG_RATE_MEAN_HZ))
    )
    bg_rate_std_hz = max(
        0.0, float(noise_cfg.get("bg_rate_std_hz", DEFAULT_BG_RATE_STD_HZ))
    )
    dark_rate_bg_hz = max(0.0, rng.normal(bg_rate_mean_hz, bg_rate_std_hz))
    p_dark_intrinsic = 1.0 - np.exp(-dark_rate_intrinsic_hz * bin_dt_s)
    p_bg = 1.0 - np.exp(-dark_rate_bg_hz * bin_dt_s)
    p_noise = 1.0 - (1.0 - p_dark_intrinsic) * (1.0 - p_bg)
    p_noise = min(max(p_noise, 0.0), 1.0)
    return {
        "dark_rate_intrinsic_hz": dark_rate_intrinsic_hz,
        "bg_rate_mean_hz": bg_rate_mean_hz,
        "bg_rate_std_hz": bg_rate_std_hz,
        "dark_rate_bg_hz": dark_rate_bg_hz,
        "p_dark_intrinsic": p_dark_intrinsic,
        "p_bg": p_bg,
        "p_noise": p_noise,
    }


def _atom_extreme_state(mps: MPSState, eps: float = ATOM_EXTREME_EPS) -> tuple:
    rho_A = mps.get_reduced_density([0])
    rho_B = mps.get_reduced_density([1])

    pA0 = float(np.real(rho_A[0, 0]))
    pA1 = float(np.real(rho_A[1, 1]))
    pAe = float(np.real(rho_A[2, 2]))
    pAu = float(np.real(rho_A[3, 3]))
    pB0 = float(np.real(rho_B[0, 0]))
    pB1 = float(np.real(rho_B[1, 1]))
    pBe = float(np.real(rho_B[2, 2]))
    pBu = float(np.real(rho_B[3, 3]))

    pA_qubit = pA0 + pA1
    pB_qubit = pB0 + pB1
    extreme_A = False
    extreme_B = False
    if pA_qubit > eps:
        pA0_rel = pA0 / pA_qubit
        pA1_rel = pA1 / pA_qubit
        extreme_A = (
            pA0_rel < eps or pA1_rel < eps or pA0_rel > 1.0 - eps or pA1_rel > 1.0 - eps
        )
    if pB_qubit > eps:
        pB0_rel = pB0 / pB_qubit
        pB1_rel = pB1 / pB_qubit
        extreme_B = (
            pB0_rel < eps or pB1_rel < eps or pB0_rel > 1.0 - eps or pB1_rel > 1.0 - eps
        )
    return (extreme_A or extreme_B), (pA0, pA1, pAe, pAu, pB0, pB1, pBe, pBu)


def _apply_atomic_dephasing(
    mps,
    p_dephase: float,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> None:
    """
    对双原子施加纯退相干通道（Z退相干）。
    """
    if p_dephase <= 0.0:
        if verbose:
            print("原子退相干：p_dephase=0，跳过。")
        return

    p_dephase = min(max(p_dephase, 0.0), 1.0)
    if rng is None:
        rng = np.random.default_rng()

    K0 = np.sqrt(1.0 - p_dephase) * np.eye(4, dtype=complex)
    Z = np.diag([1.0, -1.0, 1.0, 1.0]).astype(complex)
    K1 = np.sqrt(p_dephase) * Z
    kraus_list = [K0, K1]

    # 原子位于链最左端：atomA(0), atomB(1)
    for site in (0, 1):
        mps.apply_kraus_one_site(site, kraus_list, rng=rng)

    if verbose:
        print(f"原子退相干：已应用 p_dephase={p_dephase:.4e}")


@dataclass
class PipelineHooks:
    on_stage: Optional[Callable[[str], None]] = None
    after_emission: Optional[Callable[[Any], None]] = None
    after_qfc_filter: Optional[Callable[[Any], None]] = None
    after_fiber: Optional[Callable[[Any, tuple], None]] = None
    after_bs: Optional[Callable[[Any], None]] = None
    should_abort: Optional[Callable[[str, Any], Optional[str]]] = None


@dataclass
class PipelineResult:
    emission: Any
    mps: Any
    p_qubit_emit: float
    p_no_loss_780: float
    p_no_loss_fiber: float
    p_no_loss: float
    fiber_sample: Optional[tuple]
    t_wait_us: float
    t2_us: float
    p_dephase: float
    aborted: bool
    abort_stage: Optional[str]
    abort_reason: Optional[str]


def run_emission_to_bs(
    emission_cfg: dict,
    rng: np.random.Generator,
    fiber_params: Optional[FiberChannelParams] = None,
    verbose: bool = True,
    hooks: Optional[PipelineHooks] = None,
    t_wait_us: float = 80.0,
    t2_us: float = 1000.0,
) -> PipelineResult:
    """
    统一的发射->QFC->滤波->投影->光纤->退相干->BS 流水线。
    用于正常模式与HOM模式共用，避免重复逻辑。
    """
    from ..simulation import (
        run_dual_atom_emission,
        apply_qfc,
        apply_780_filter,
        apply_fiber_channel,
        project_to_1517,
        apply_bs,
        extract_spin_state,
    )

    if hooks is None:
        hooks = PipelineHooks()
    if fiber_params is None:
        fiber_params = _make_fiber_params()

    def _call_stage(label: str) -> None:
        if hooks.on_stage is not None:
            hooks.on_stage(label)

    def _maybe_abort(stage_label: str, mps) -> Optional[str]:
        if hooks.should_abort is None:
            return None
        return hooks.should_abort(stage_label, mps)

    _call_stage("发射")
    emission = run_dual_atom_emission(
        n_bins=emission_cfg["n_bins"],
        dt_ns=emission_cfg["dt_ns"],
        chi_max=emission_cfg["chi_max"],
        gamma_peak_A=emission_cfg["gamma_peak_A"],
        gamma_peak_B=emission_cfg["gamma_peak_B"],
        sigma=emission_cfg["sigma"],
        delay_ns=emission_cfg["delay_ns"],
        delay_jitter_ns=emission_cfg["delay_jitter_ns"],
        g=emission_cfg["g"],
        kappa_ex=emission_cfg["kappa_ex"],
        kappa_in=emission_cfg["kappa_in"],
        gamma_atom=emission_cfg["gamma_atom"],
        delta_u=emission_cfg["delta_u"],
        delta_e=emission_cfg["delta_e"],
        rng=rng,
        verbose=verbose,
    )
    mps = emission.mps
    _, p_qubit_emit = extract_spin_state(mps, emission.get_n_bins())
    if hooks.after_emission is not None:
        hooks.after_emission(emission)

    reason = _maybe_abort("After Emission", mps)
    if reason:
        return PipelineResult(
            emission=emission,
            mps=mps,
            p_qubit_emit=p_qubit_emit,
            p_no_loss_780=0.0,
            p_no_loss_fiber=0.0,
            p_no_loss=0.0,
            fiber_sample=None,
            t_wait_us=t_wait_us,
            t2_us=t2_us,
            p_dephase=0.0,
            aborted=True,
            abort_stage="After Emission",
            abort_reason=reason,
        )

    _call_stage("QFC + 780滤波 + 1517投影")
    apply_qfc(
        mps=mps,
        n_bins=emission.get_n_bins(),
        theta_H=np.pi / 4,
        theta_V=np.pi / 4,
        verbose=verbose,
    )
    mps, p_no_loss_780 = apply_780_filter(
        mps=mps,
        n_bins=emission.get_n_bins(),
        verbose=verbose,
        rng=rng,
    )
    if p_no_loss_780 <= 0.0:
        return PipelineResult(
            emission=emission,
            mps=mps,
            p_qubit_emit=p_qubit_emit,
            p_no_loss_780=0.0,
            p_no_loss_fiber=0.0,
            p_no_loss=0.0,
            fiber_sample=None,
            t_wait_us=t_wait_us,
            t2_us=t2_us,
            p_dephase=0.0,
            aborted=True,
            abort_stage="After QFC + Filter",
            abort_reason="780nm滤波后选概率为0，跳过后续计算",
        )
    project_to_1517(
        mps=mps,
        n_bins=emission.get_n_bins(),
        verbose=verbose,
    )
    if hooks.after_qfc_filter is not None:
        hooks.after_qfc_filter(emission)

    reason = _maybe_abort("After QFC + Filter", mps)
    if reason:
        return PipelineResult(
            emission=emission,
            mps=mps,
            p_qubit_emit=p_qubit_emit,
            p_no_loss_780=p_no_loss_780,
            p_no_loss_fiber=0.0,
            p_no_loss=0.0,
            fiber_sample=None,
            t_wait_us=t_wait_us,
            t2_us=t2_us,
            p_dephase=0.0,
            aborted=True,
            abort_stage="After QFC + Filter",
            abort_reason=reason,
        )

    _call_stage("光纤信道")
    mps, fiber_sample, p_no_loss_fiber = apply_fiber_channel(
        mps=mps,
        n_bins=emission.get_n_bins(),
        fiber_params=fiber_params,
        rng=rng,
        verbose=verbose,
    )
    p_no_loss = p_no_loss_780 * p_no_loss_fiber
    if p_no_loss <= 0.0:
        return PipelineResult(
            emission=emission,
            mps=mps,
            p_qubit_emit=p_qubit_emit,
            p_no_loss_780=p_no_loss_780,
            p_no_loss_fiber=p_no_loss_fiber,
            p_no_loss=0.0,
            fiber_sample=fiber_sample,
            t_wait_us=t_wait_us,
            t2_us=t2_us,
            p_dephase=0.0,
            aborted=True,
            abort_stage="After Fiber Channel",
            abort_reason="光纤无损耗后选概率为0，跳过后续计算",
        )
    if hooks.after_fiber is not None:
        hooks.after_fiber(emission, fiber_sample)

    reason = _maybe_abort("After Fiber Channel", mps)
    if reason:
        return PipelineResult(
            emission=emission,
            mps=mps,
            p_qubit_emit=p_qubit_emit,
            p_no_loss_780=p_no_loss_780,
            p_no_loss_fiber=p_no_loss_fiber,
            p_no_loss=p_no_loss,
            fiber_sample=fiber_sample,
            t_wait_us=t_wait_us,
            t2_us=t2_us,
            p_dephase=0.0,
            aborted=True,
            abort_stage="After Fiber Channel",
            abort_reason=reason,
        )

    if t2_us > 0.0:
        p_dephase = 0.5 * (1.0 - np.exp(-t_wait_us / t2_us))
    else:
        p_dephase = 0.0
    if verbose:
        print(f"\n原子等待退相干: T_wait={t_wait_us:.1f} us, T2={t2_us:.1f} us, p={p_dephase:.4e}")
    _apply_atomic_dephasing(mps, p_dephase, rng=rng, verbose=verbose)

    _call_stage("分束器 + 诊断/可视化")
    if verbose:
        print("\n应用分束器（BS）...")
    apply_bs(
        mps=mps,
        n_bins=emission.get_n_bins(),
        verbose=verbose,
    )
    if hooks.after_bs is not None:
        hooks.after_bs(emission)

    return PipelineResult(
        emission=emission,
        mps=mps,
        p_qubit_emit=p_qubit_emit,
        p_no_loss_780=p_no_loss_780,
        p_no_loss_fiber=p_no_loss_fiber,
        p_no_loss=p_no_loss,
        fiber_sample=fiber_sample,
        t_wait_us=t_wait_us,
        t2_us=t2_us,
        p_dephase=p_dephase,
        aborted=False,
        abort_stage=None,
        abort_reason=None,
    )


def run_task_queue(
    jobs: int,
    task_fn: Callable[..., Any],
    next_task: Callable[[], Optional[tuple]],
    on_result: Callable[[tuple, Any], None],
    focus_task: Optional[tuple] = None,
    focus_fn: Optional[Callable[..., Any]] = None,
) -> None:
    """
    统一的进程任务调度器：
    - next_task(): 返回一个任务参数元组，或 None 表示无任务
    - task_fn(*task): 在子进程执行
    - on_result(task, result): 主进程汇总
    - focus_task: 可选的前台任务（主进程执行一次）
    """
    if jobs <= 1:
        if focus_task is not None:
            if focus_fn is None:
                focus_fn = task_fn
            result = focus_fn(*focus_task)
            on_result(focus_task, result)
        while True:
            task = next_task()
            if task is None:
                break
            result = task_fn(*task)
            on_result(task, result)
        return

    worker_jobs = jobs - 1 if focus_task is not None and jobs > 1 else jobs
    if worker_jobs <= 0:
        worker_jobs = 1

    with ProcessPoolExecutor(max_workers=worker_jobs) as executor:
        pending = {}

        def _fill_pending() -> None:
            while len(pending) < worker_jobs:
                task = next_task()
                if task is None:
                    break
                pending[executor.submit(task_fn, *task)] = task

        _fill_pending()
        if focus_task is not None:
            if focus_fn is None:
                focus_fn = task_fn
            result = focus_fn(*focus_task)
            on_result(focus_task, result)
        while pending:
            future = next(as_completed(pending))
            task = pending.pop(future)
            on_result(task, future.result())
            _fill_pending()
