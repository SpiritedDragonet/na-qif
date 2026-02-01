# -*- coding: utf-8 -*-
"""
实验流程中的通用配置与工具函数。
"""

from __future__ import annotations

from typing import Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import os
import time
import numpy as np

from ..physics import FiberChannelParams

# 探测噪声默认参数（可用CLI覆盖）
DEFAULT_DARK_RATE_INTRINSIC_HZ = 65.0
DEFAULT_BG_RATE_MEAN_HZ = 165.0
DEFAULT_BG_RATE_STD_HZ = float(np.sqrt(5.0))


@dataclass
class EmissionParams:
    """发射阶段参数（可复用的最小物理输入）。"""
    n_bins: int = 100
    dt_ns: float = 0.5
    chi_max: int = 50
    gamma_peak_A: float = 2 * np.pi * 20e6
    gamma_peak_B: float = 2 * np.pi * 20e6
    sigma: float = 10.0
    g: float = 2 * np.pi * 20e6
    kappa_ex: float = 2 * np.pi * 20e6
    kappa_in: float = 2 * np.pi * 1e6
    gamma_atom: float = 2 * np.pi * 3e6
    delta_u: float = 0.0
    delta_e: float = 0.0
    delay_ns: Optional[float] = None
    delay_jitter_ns: float = 0.5
    delay_random_range: Tuple[float, float] = (-10.0, 10.0)


@dataclass
class NoiseParams:
    """探测噪声参数（暗计数 + 背景噪声）。"""
    dark_rate_intrinsic_hz: float = DEFAULT_DARK_RATE_INTRINSIC_HZ
    bg_rate_mean_hz: float = DEFAULT_BG_RATE_MEAN_HZ
    bg_rate_std_hz: float = DEFAULT_BG_RATE_STD_HZ


@dataclass
class DetectorParams:
    """探测器参数。"""
    eta_det: float = 0.85
    ideal_det: bool = False


@dataclass
class FiberParams:
    """光纤信道参数。"""
    noise_enabled: bool = True
    polarization_model: str = "perturb"
    polarization_sigma: float = 0.1
    eta_mean: float = 0.6
    eta_std: float = 0.02
    pdl_sigma: float = 0.02
    phase_drift_std: float = 0.2
    phase_slope_std: float = 0.05
    phase_jitter_std: float = 0.0


@dataclass
class RunConfig:
    """运行参数（次数、核预算、枚举模式等）。"""
    runs: int = 1
    shots_per_run: int = 1
    cores: int = 1
    enum_mode: str = "dark"
    plot_all: bool = False
    debug: bool = False


@dataclass
class HomConfig:
    """HOM 扫描参数。"""
    tau: Optional[float] = None
    tau_start: Optional[float] = None
    tau_end: Optional[float] = None
    tau_step: Optional[float] = None
    tau_points: Optional[int] = None
    tau_random: bool = False
    tau_random_range: Tuple[float, float] = (-10.0, 10.0)
    window_ns: float = 70.0
    max_attempts: Optional[int] = None


@dataclass
class SimConfig:
    """统一配置入口（所有实验参数挂在此处）。"""
    mode: str = "SIM"
    run: RunConfig = field(default_factory=RunConfig)
    emission: EmissionParams = field(default_factory=EmissionParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    detector: DetectorParams = field(default_factory=DetectorParams)
    fiber: FiberParams = field(default_factory=FiberParams)
    hom: Optional[HomConfig] = None


def _resolve_emission_delay(
    emission: EmissionParams,
    rng: np.random.Generator,
    delay_ns: Optional[float],
    delay_jitter_ns: Optional[float],
) -> tuple:
    if delay_ns is None:
        if emission.delay_ns is None:
            low, high = emission.delay_random_range
            delay_ns = float(rng.uniform(low, high))
        else:
            delay_ns = float(emission.delay_ns)
    if delay_jitter_ns is None:
        delay_jitter_ns = float(emission.delay_jitter_ns)
    return float(delay_ns), float(delay_jitter_ns)


def _build_fiber_params(cfg: FiberParams) -> FiberChannelParams:
    if not cfg.noise_enabled:
        return FiberChannelParams(
            polarization_model=cfg.polarization_model,
            polarization_sigma=0.0,
            eta_mean=cfg.eta_mean,
            eta_std=0.0,
            pdl_sigma=0.0,
            phase_drift_std=0.0,
            phase_slope_std=0.0,
            phase_jitter_std=0.0,
        )
    return FiberChannelParams(
        polarization_model=cfg.polarization_model,
        polarization_sigma=cfg.polarization_sigma,
        eta_mean=cfg.eta_mean,
        eta_std=cfg.eta_std,
        pdl_sigma=cfg.pdl_sigma,
        phase_drift_std=cfg.phase_drift_std,
        phase_slope_std=cfg.phase_slope_std,
        phase_jitter_std=cfg.phase_jitter_std,
    )


def _compute_window_bins(window_ns: float, bin_dt_ns: float) -> int:
    if bin_dt_ns <= 0:
        return 0
    return int(round(window_ns / bin_dt_ns))


def _compute_noise_params(
    noise_cfg: Optional[NoiseParams],
    bin_dt_s: float,
    rng: np.random.Generator,
) -> dict:
    if noise_cfg is None:
        noise_cfg = NoiseParams()
    dark_rate_intrinsic_hz = max(0.0, float(noise_cfg.dark_rate_intrinsic_hz))
    bg_rate_mean_hz = max(0.0, float(noise_cfg.bg_rate_mean_hz))
    bg_rate_std_hz = max(0.0, float(noise_cfg.bg_rate_std_hz))
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
    after_qfc: Optional[Callable[[Any], None]] = None
    after_fiber: Optional[Callable[[Any, tuple], None]] = None


@dataclass
class PipelineResult:
    emission: Any
    mps: Any
    p_qubit_emit: float
    fiber_sample: Optional[tuple]
    t_wait_us: float
    t2_us: float
    p_dephase: float
    aborted: bool
    abort_stage: Optional[str]
    abort_reason: Optional[str]
    timings: Optional[dict] = None


def run_emission_to_bs(
    emission: EmissionParams,
    rng: np.random.Generator,
    fiber: Optional[FiberParams] = None,
    delay_ns: Optional[float] = None,
    delay_jitter_ns: Optional[float] = None,
    verbose: bool = True,
    hooks: Optional[PipelineHooks] = None,
    t_wait_us: float = 80.0,
    t2_us: float = 1000.0,
    record_timings: bool = False,
) -> PipelineResult:
    """
    统一的发射->QFC->光纤(偏振/相位)->退相干 流水线。
    用于正常模式与HOM模式共用，避免重复逻辑。
    """
    from ..simulation import (
        run_dual_atom_emission,
        apply_qfc,
        apply_fiber_channel,
        extract_spin_state,
    )

    if hooks is None:
        hooks = PipelineHooks()
    if fiber is None:
        fiber = FiberParams()
    fiber_params = _build_fiber_params(fiber)
    delay_ns, delay_jitter_ns = _resolve_emission_delay(
        emission, rng, delay_ns, delay_jitter_ns
    )

    def _call_stage(label: str) -> None:
        if hooks.on_stage is not None:
            hooks.on_stage(label)

    timings = {} if record_timings else None

    _call_stage("发射")
    t0 = time.perf_counter() if timings is not None else None
    emission = run_dual_atom_emission(
        n_bins=emission.n_bins,
        dt_ns=emission.dt_ns,
        chi_max=emission.chi_max,
        gamma_peak_A=emission.gamma_peak_A,
        gamma_peak_B=emission.gamma_peak_B,
        sigma=emission.sigma,
        delay_ns=delay_ns,
        delay_jitter_ns=delay_jitter_ns,
        g=emission.g,
        kappa_ex=emission.kappa_ex,
        kappa_in=emission.kappa_in,
        gamma_atom=emission.gamma_atom,
        delta_u=emission.delta_u,
        delta_e=emission.delta_e,
        rng=rng,
        verbose=verbose,
    )
    if timings is not None and t0 is not None:
        timings["emission"] = time.perf_counter() - t0
    mps = emission.mps
    _, p_qubit_emit = extract_spin_state(mps, emission.get_n_bins())
    if hooks.after_emission is not None:
        hooks.after_emission(emission)

    _call_stage("QFC")
    t0 = time.perf_counter() if timings is not None else None
    apply_qfc(
        mps=mps,
        n_bins=emission.get_n_bins(),
        theta_H=np.pi / 4,
        theta_V=np.pi / 4,
        verbose=verbose,
    )
    if timings is not None and t0 is not None:
        timings["qfc"] = time.perf_counter() - t0
    if hooks.after_qfc is not None:
        hooks.after_qfc(emission)

    _call_stage("光纤信道")
    t0 = time.perf_counter() if timings is not None else None
    mps, fiber_sample = apply_fiber_channel(
        mps=mps,
        n_bins=emission.get_n_bins(),
        fiber_params=fiber_params,
        rng=rng,
        verbose=verbose,
    )
    if timings is not None and t0 is not None:
        timings["fiber"] = time.perf_counter() - t0
    if hooks.after_fiber is not None:
        hooks.after_fiber(emission, fiber_sample)

    if t2_us > 0.0:
        p_dephase = 0.5 * (1.0 - np.exp(-t_wait_us / t2_us))
    else:
        p_dephase = 0.0
    if verbose:
        print(f"\n原子等待退相干: T_wait={t_wait_us:.1f} us, T2={t2_us:.1f} us, p={p_dephase:.4e}")
    t0 = time.perf_counter() if timings is not None else None
    _apply_atomic_dephasing(mps, p_dephase, rng=rng, verbose=verbose)
    if timings is not None and t0 is not None:
        timings["dephase"] = time.perf_counter() - t0

    return PipelineResult(
        emission=emission,
        mps=mps,
        p_qubit_emit=p_qubit_emit,
        fiber_sample=fiber_sample,
        t_wait_us=t_wait_us,
        t2_us=t2_us,
        p_dephase=p_dephase,
        aborted=False,
        abort_stage=None,
        abort_reason=None,
        timings=timings,
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

    if worker_jobs > 1:
        # 避免多进程叠加 BLAS 线程导致过度并发。
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    def _run_task_silent(fn: Callable[..., Any], *task: Any) -> Any:
        # 只让前台任务输出，避免多进程 stdout 阻塞。
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull):
                return fn(*task)

    with ProcessPoolExecutor(max_workers=worker_jobs) as executor:
        pending = {}

        def _fill_pending() -> None:
            while len(pending) < worker_jobs:
                task = next_task()
                if task is None:
                    break
                pending[executor.submit(_run_task_silent, task_fn, *task)] = task

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
