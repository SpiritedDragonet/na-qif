# -*- coding: utf-8 -*-
"""
实验流程中的通用配置与工具函数。
"""

from __future__ import annotations

from typing import Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
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
    plot_enabled: bool = True
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
    # 统一解析 delay / delay_jitter：
    #   - 若调用方给了明确值，优先使用
    #   - 否则用 config 默认（或随机范围）
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
    # 将实验配置转换成 FiberChannelParams（用于采样每次光纤漂移）
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
    # 将物理时间窗口映射为 bin 数
    if bin_dt_ns <= 0:
        return 0
    return int(round(window_ns / bin_dt_ns))


def _compute_noise_params(
    noise_cfg: Optional[NoiseParams],
    bin_dt_s: float,
    rng: np.random.Generator,
) -> dict:
    # ------------------------------------------------------------------
    # 暗计数与背景噪声的合成：
    #   - 本底暗计数：dark_rate_intrinsic_hz
    #   - 背景噪声：在每次 run 采样一个 rate (高斯分布)
    #   - 合并：p_noise = 1 - (1-p_dark)(1-p_bg)
    # ------------------------------------------------------------------
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
    # 退相干通道：ρ -> (1-p)ρ + p ZρZ
    # 这里的 Z 只作用在 |0>/<1> 子空间，相当于相位噪声。
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


@dataclass
class PipelineResult:
    emission: Any
    mps: Any
    p_qubit_emit: float
    p_no_loss_780: Optional[float]
    p_no_loss_fiber: Optional[float]
    p_no_loss: Optional[float]
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
    统一的发射->QFC->滤波->投影->光纤->退相干->(BS并入测量) 流水线。
    用于正常模式与HOM模式共用，避免重复逻辑。
    """
    # ------------------------------------------------------------------
    # 这是“物理链路主流程”的统一入口：
    #   1) 发射 (TEBD on time bins)
    #   2) QFC + 780 滤波 + 投影到 1517
    #   3) 光纤漂移 + 损耗
    #   4) 原子退相干
    #
    # 注意：
    #   - BS 已经并入测量端 (Heisenberg side)；
    #   - 这里不再对 MPS 显式 apply_bs。
    # ------------------------------------------------------------------
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
    # 方案B：不再做 780 后选/1517 投影，损耗统一推到测量端 effect。
    p_no_loss_780 = None
    if hooks.after_qfc_filter is not None:
        hooks.after_qfc_filter(emission)

    _call_stage("光纤信道")
    t0 = time.perf_counter() if timings is not None else None
    mps, fiber_sample, p_no_loss_fiber = apply_fiber_channel(
        mps=mps,
        n_bins=emission.get_n_bins(),
        fiber_params=fiber_params,
        rng=rng,
        verbose=verbose,
        apply_loss=False,
    )
    if timings is not None and t0 is not None:
        timings["fiber"] = time.perf_counter() - t0
    p_no_loss = None
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

    # 这里仅触发“after_bs”可视化 hook；
    # 真正的 BS 已在测量端 effect 中处理。
    _call_stage("分束器(测量端) + 诊断/可视化")
    if verbose:
        print("\n分束器并入测量算符（Heisenberg 端口），不对态显式作用 BS。")
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
        timings=timings,
    )


