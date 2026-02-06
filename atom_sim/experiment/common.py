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
class AtomArmParams:
    """单臂发射参数（A/B 可独立设置）。"""
    omega_peak: float = 2 * np.pi * 20e6
    g: float = 2 * np.pi * 20e6
    kappa_ex: float = 2 * np.pi * 20e6
    kappa_in: float = 2 * np.pi * 1e6
    delta_u: float = 0.0
    delta_e: float = 0.0
    gamma_loss: float = 0.0


@dataclass
class EmissionParams:
    """发射阶段参数（可复用的最小物理输入）。"""
    n_bins: int = 100
    dt_ns: float = 0.5
    chi_max: int = 50
    sigma: float = 10.0
    arm_A: AtomArmParams = field(default_factory=AtomArmParams)
    arm_B: AtomArmParams = field(default_factory=AtomArmParams)
    delay_ns: Optional[float] = None
    delay_jitter_ns: float = 0.5
    delay_random_range: Tuple[float, float] = (-10.0, 10.0)


@dataclass
class NoiseParams:
    """探测噪声参数（暗计数 + 背景噪声）。"""
    dark_rate_intrinsic_hz: float = DEFAULT_DARK_RATE_INTRINSIC_HZ
    bg_rate_mean_hz: float = DEFAULT_BG_RATE_MEAN_HZ
    bg_rate_std_hz: float = DEFAULT_BG_RATE_STD_HZ
    detector_gate_ns: float = 1.0


@dataclass
class DetectorParams:
    """探测器参数。"""
    eta_det: float = 0.85
    ideal_det: bool = False
    # 残差可区分度：仅用于承载“当前模型未显式建模”的剩余退相干。
    # TODO(model-first-principles): 当以下因素全部显式建模后，删除该旋钮：
    #   1) QFC 场端噪声注入（替代探测端等效暗计数吸收）
    #   2) PMD/色散导致的跨bin时间模式混合
    #   3) 频域/时域模式重叠的显式建模（替代残差旋钮）
    v_res: float = 1.0


@dataclass
class NoiseBudget:
    """一次 run 采样得到的噪声预算（与参数表字段一一对应）。"""
    emission_bin_dt_s: float
    detection_gate_ns: float
    detection_gate_dt_s: float
    bins_per_gate: int
    dark_rate_intrinsic_hz: float
    bg_rate_mean_hz: float
    bg_rate_std_hz: float
    dark_rate_bg_hz: float
    p_dark_intrinsic_gate: float
    p_bg_gate: float
    p_noise_gate: float
    p_dark_intrinsic_bin: float
    p_bg_bin: float
    p_noise_bin: float


@dataclass
class RunParameterStore:
    """单次 run 的全局参数快照（用于统一调用与预算输出）。"""
    eta_det: float
    v_res: float
    window_ns: float
    window_bins: int
    noise_budget: NoiseBudget


@dataclass
class QfcParams:
    """QFC 参数（可扫）。"""
    theta_H: float = np.pi / 4
    theta_V: float = np.pi / 4
    apply_filter_780: bool = True


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
    window_ns: float = 70.0


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


@dataclass
class SimConfig:
    """统一配置入口（所有实验参数挂在此处）。"""
    mode: str = "SIM"
    run: RunConfig = field(default_factory=RunConfig)
    emission: EmissionParams = field(default_factory=EmissionParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    detector: DetectorParams = field(default_factory=DetectorParams)
    qfc: QfcParams = field(default_factory=QfcParams)
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
            polarization_model="fixed",
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


def _compute_window_bins(
    window_ns: float,
    bin_dt_ns: float,
    detection_gate_ns: Optional[float] = None,
) -> int:
    # 将物理时间窗口映射为“仿真细 bin 数”，
    # 并显式支持“探测门宽 != 数值 dt”的情形。
    if bin_dt_ns <= 0:
        return 0
    gate_ns = bin_dt_ns if detection_gate_ns is None else max(float(detection_gate_ns), bin_dt_ns)
    bins_per_gate = max(1, int(round(gate_ns / bin_dt_ns)))
    coarse_window_bins = int(round(window_ns / gate_ns))
    return max(0, coarse_window_bins * bins_per_gate)


def _sample_noise_budget(
    noise_cfg: Optional[NoiseParams],
    emission_bin_dt_s: float,
    rng: np.random.Generator,
    ideal_det: bool = False,
) -> NoiseBudget:
    # ------------------------------------------------------------------
    # 暗计数与背景噪声的合成：
    #   - 本底暗计数：dark_rate_intrinsic_hz
    #   - 背景噪声：在每次 run 采样一个 rate (高斯分布)
    #   - 合并：p_noise = 1 - (1-p_dark)(1-p_bg)
    # ------------------------------------------------------------------
    if noise_cfg is None:
        noise_cfg = NoiseParams()

    emission_bin_dt_s = float(emission_bin_dt_s)
    if emission_bin_dt_s <= 0.0:
        raise ValueError("emission_bin_dt_s 必须 > 0")

    gate_ns = max(float(noise_cfg.detector_gate_ns), emission_bin_dt_s * 1e9)
    gate_dt_s = gate_ns * 1e-9
    bins_per_gate = max(1, int(round(gate_dt_s / emission_bin_dt_s)))

    if ideal_det:
        dark_rate_intrinsic_hz = 0.0
        bg_rate_mean_hz = 0.0
        bg_rate_std_hz = 0.0
        dark_rate_bg_hz = 0.0
    else:
        dark_rate_intrinsic_hz = max(0.0, float(noise_cfg.dark_rate_intrinsic_hz))
        bg_rate_mean_hz = max(0.0, float(noise_cfg.bg_rate_mean_hz))
        bg_rate_std_hz = max(0.0, float(noise_cfg.bg_rate_std_hz))
        dark_rate_bg_hz = max(0.0, rng.normal(bg_rate_mean_hz, bg_rate_std_hz))

    p_dark_intrinsic_gate = 1.0 - np.exp(-dark_rate_intrinsic_hz * gate_dt_s)
    p_bg_gate = 1.0 - np.exp(-dark_rate_bg_hz * gate_dt_s)
    p_noise_gate = 1.0 - (1.0 - p_dark_intrinsic_gate) * (1.0 - p_bg_gate)

    ratio = emission_bin_dt_s / gate_dt_s
    p_dark_intrinsic_bin = 1.0 - (1.0 - p_dark_intrinsic_gate) ** ratio
    p_bg_bin = 1.0 - (1.0 - p_bg_gate) ** ratio
    p_noise_bin = 1.0 - (1.0 - p_dark_intrinsic_bin) * (1.0 - p_bg_bin)

    return NoiseBudget(
        emission_bin_dt_s=emission_bin_dt_s,
        detection_gate_ns=gate_ns,
        detection_gate_dt_s=gate_dt_s,
        bins_per_gate=bins_per_gate,
        dark_rate_intrinsic_hz=dark_rate_intrinsic_hz,
        bg_rate_mean_hz=bg_rate_mean_hz,
        bg_rate_std_hz=bg_rate_std_hz,
        dark_rate_bg_hz=dark_rate_bg_hz,
        p_dark_intrinsic_gate=min(max(p_dark_intrinsic_gate, 0.0), 1.0),
        p_bg_gate=min(max(p_bg_gate, 0.0), 1.0),
        p_noise_gate=min(max(p_noise_gate, 0.0), 1.0),
        p_dark_intrinsic_bin=min(max(p_dark_intrinsic_bin, 0.0), 1.0),
        p_bg_bin=min(max(p_bg_bin, 0.0), 1.0),
        p_noise_bin=min(max(p_noise_bin, 0.0), 1.0),
    )


def _build_run_parameter_store(
    config: SimConfig,
    emission_bin_dt_s: float,
    coincidence_window_ns: float,
    rng: np.random.Generator,
) -> RunParameterStore:
    noise_budget = _sample_noise_budget(
        noise_cfg=config.noise,
        emission_bin_dt_s=emission_bin_dt_s,
        rng=rng,
        ideal_det=config.detector.ideal_det,
    )
    eta_det = 1.0 if config.detector.ideal_det else float(config.detector.eta_det)
    window_bins = _compute_window_bins(
        coincidence_window_ns,
        emission_bin_dt_s * 1e9,
        detection_gate_ns=noise_budget.detection_gate_ns,
    )
    return RunParameterStore(
        eta_det=eta_det,
        v_res=float(config.detector.v_res),
        window_ns=float(coincidence_window_ns),
        window_bins=window_bins,
        noise_budget=noise_budget,
    )


def _build_parameter_snapshot(config: SimConfig, store: RunParameterStore) -> dict:
    emission = config.emission
    arm_a = emission.arm_A
    arm_b = emission.arm_B
    budget = store.noise_budget
    return {
        "n_bins": emission.n_bins,
        "dt_ns": emission.dt_ns,
        "sigma_ns": emission.sigma,
        "omega_peak_A": arm_a.omega_peak,
        "omega_peak_B": arm_b.omega_peak,
        "g_A": arm_a.g,
        "g_B": arm_b.g,
        "kappa_ex_A": arm_a.kappa_ex,
        "kappa_ex_B": arm_b.kappa_ex,
        "kappa_in_A": arm_a.kappa_in,
        "kappa_in_B": arm_b.kappa_in,
        "delta_u_A": arm_a.delta_u,
        "delta_u_B": arm_b.delta_u,
        "delta_e_A": arm_a.delta_e,
        "delta_e_B": arm_b.delta_e,
        "gamma_loss_A": arm_a.gamma_loss,
        "gamma_loss_B": arm_b.gamma_loss,
        "eta_det": store.eta_det,
        "v_res": store.v_res,
        "window_ns": store.window_ns,
        "window_bins": store.window_bins,
        "detector_gate_ns": budget.detection_gate_ns,
        "bins_per_gate": budget.bins_per_gate,
        "dark_rate_intrinsic_hz": budget.dark_rate_intrinsic_hz,
        "bg_rate_mean_hz": budget.bg_rate_mean_hz,
        "bg_rate_std_hz": budget.bg_rate_std_hz,
        "dark_rate_bg_hz": budget.dark_rate_bg_hz,
        "p_dark_intrinsic_gate": budget.p_dark_intrinsic_gate,
        "p_bg_gate": budget.p_bg_gate,
        "p_noise_gate": budget.p_noise_gate,
        "p_dark_intrinsic_bin": budget.p_dark_intrinsic_bin,
        "p_bg_bin": budget.p_bg_bin,
        "p_noise_bin": budget.p_noise_bin,
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
    after_qfc_filter: Optional[Callable[[Any, Tuple[float, float], bool], None]] = None
    after_fiber: Optional[Callable[[Any, tuple, Tuple[float, float], bool], None]] = None
    after_bs: Optional[Callable[[Any, tuple, Tuple[float, float], bool], None]] = None


@dataclass
class PipelineResult:
    emission: Any
    mps: Any
    p_qubit_emit: float
    fiber_sample: Optional[tuple]
    qfc_theta_H: float
    qfc_theta_V: float
    apply_filter_780: bool
    t_wait_us: float
    t2_us: float
    p_dephase: float
    timings: Optional[dict] = None


def _build_detection_kwargs(
    pipe: PipelineResult,
    *,
    param_store: RunParameterStore,
    rng: np.random.Generator,
    verbose: bool,
    bs_unitary: np.ndarray,
) -> dict:
    """
    统一拼装 run_detection_pipeline 的公共参数，避免多处重复与分叉。
    """
    return {
        "mps": pipe.mps,
        "n_bins": pipe.emission.get_n_bins(),
        "eta_det": float(param_store.eta_det),
        "window_bins": int(param_store.window_bins),
        "rng": rng,
        "verbose": verbose,
        "bs_unitary": bs_unitary,
        "fiber_sample": pipe.fiber_sample,
        "apply_filter_780": pipe.apply_filter_780,
        "theta_H": pipe.qfc_theta_H,
        "theta_V": pipe.qfc_theta_V,
        "v_res": float(param_store.v_res),
    }


def run_emission_to_bs(
    emission: EmissionParams,
    rng: np.random.Generator,
    fiber: Optional[FiberParams] = None,
    qfc: Optional[QfcParams] = None,
    delay_ns: Optional[float] = None,
    delay_jitter_ns: Optional[float] = None,
    verbose: bool = True,
    hooks: Optional[PipelineHooks] = None,
    t_wait_us: float = 80.0,
    t2_us: float = 1000.0,
    record_timings: bool = False,
    emission_diagnostics: bool = False,
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
    #   - 这里不再对 MPS 显式作用 BS 门。
    # ------------------------------------------------------------------
    from ..simulation import (
        run_dual_atom_emission,
        sample_fiber_realization,
        extract_qubit_state,
    )

    if hooks is None:
        hooks = PipelineHooks()
    if fiber is None:
        fiber = FiberParams()
    if qfc is None:
        qfc = QfcParams()
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
        omega_peak_A=emission.arm_A.omega_peak,
        omega_peak_B=emission.arm_B.omega_peak,
        sigma=emission.sigma,
        delay_ns=delay_ns,
        delay_jitter_ns=delay_jitter_ns,
        g_A=emission.arm_A.g,
        g_B=emission.arm_B.g,
        kappa_ex_A=emission.arm_A.kappa_ex,
        kappa_ex_B=emission.arm_B.kappa_ex,
        kappa_in_A=emission.arm_A.kappa_in,
        kappa_in_B=emission.arm_B.kappa_in,
        delta_u_A=emission.arm_A.delta_u,
        delta_u_B=emission.arm_B.delta_u,
        delta_e_A=emission.arm_A.delta_e,
        delta_e_B=emission.arm_B.delta_e,
        gamma_loss_A=emission.arm_A.gamma_loss,
        gamma_loss_B=emission.arm_B.gamma_loss,
        rng=rng,
        verbose=verbose,
        diagnostics=emission_diagnostics,
    )
    if timings is not None and t0 is not None:
        timings["emission"] = time.perf_counter() - t0
    mps = emission.mps
    _, p_qubit_emit = extract_qubit_state(mps)
    if hooks.after_emission is not None:
        hooks.after_emission(emission)

    _call_stage("QFC (Heisenberg 参数)")
    t0 = time.perf_counter() if timings is not None else None
    qfc_theta_H = float(qfc.theta_H)
    qfc_theta_V = float(qfc.theta_V)
    apply_filter_780 = bool(qfc.apply_filter_780)
    if timings is not None and t0 is not None:
        timings["qfc"] = time.perf_counter() - t0
    if hooks.after_qfc_filter is not None:
        hooks.after_qfc_filter(
            emission,
            qfc_params=(qfc_theta_H, qfc_theta_V),
            apply_filter_780=apply_filter_780,
        )
        if verbose:
            print("QFC/滤波仅作为测量端参数写入（Heisenberg），未对态显式作用。")

    _call_stage("光纤信道 (Heisenberg 参数)")
    t0 = time.perf_counter() if timings is not None else None
    mps, fiber_sample = sample_fiber_realization(
        mps=mps,
        n_bins=emission.get_n_bins(),
        fiber_params=fiber_params,
        rng=rng,
        verbose=verbose,
    )
    if timings is not None and t0 is not None:
        timings["fiber"] = time.perf_counter() - t0
    if hooks.after_fiber is not None:
        hooks.after_fiber(
            emission,
            fiber_sample=fiber_sample,
            qfc_params=(qfc_theta_H, qfc_theta_V),
            apply_filter_780=apply_filter_780,
        )
        if verbose:
            print("光纤噪声仅作为测量端参数写入（Heisenberg），未对态显式作用。")

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
        hooks.after_bs(
            emission,
            fiber_sample=fiber_sample,
            qfc_params=(qfc_theta_H, qfc_theta_V),
            apply_filter_780=apply_filter_780,
        )

    return PipelineResult(
        emission=emission,
        mps=mps,
        p_qubit_emit=p_qubit_emit,
        fiber_sample=fiber_sample,
        qfc_theta_H=qfc_theta_H,
        qfc_theta_V=qfc_theta_V,
        apply_filter_780=apply_filter_780,
        t_wait_us=t_wait_us,
        t2_us=t2_us,
        p_dephase=p_dephase,
        timings=timings,
    )


