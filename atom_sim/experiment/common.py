# -*- coding: utf-8 -*-
"""
实验流程中的通用配置与工具函数。
"""

from __future__ import annotations

from typing import Optional, Callable, Any, Tuple
from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import time
import numpy as np

from ..physics import FiberChannelParams

# 探测噪声默认参数（可用CLI覆盖）
DEFAULT_DARK_RATE_INTRINSIC_HZ = 65.0
DEFAULT_BG_RATE_MEAN_HZ = 165.0
DEFAULT_BG_RATE_STD_HZ = float(np.sqrt(5.0))
DEFAULT_QFC_EFFICIENCY = 0.57
DEFAULT_QFC_THETA_RAD = float(np.arcsin(np.sqrt(DEFAULT_QFC_EFFICIENCY)))
DEFAULT_QFC_NOISE_SD_CPS_PER_MHZ = 41.1
DEFAULT_EMISSION_SIGMA_NS = 8.9
DEFAULT_DELAY_JITTER_NS = 0.3
DEFAULT_T2_US = 330.0
DETECTOR_CHANNELS = ("H1", "V1", "H2", "V2")


def write_click_records(raw_dir: Path, click_records: Any) -> None:
    """
    写入 raw/clicks.json（task 级独占文件）。

    说明：
    - 每个 task 的 clicks 文件路径独立，无跨 worker 共享写入；
    - 仍使用原子替换，避免中断时读到半写入 JSON。
    """
    if click_records is None:
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "clicks.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"clicks": click_records}, f, ensure_ascii=False)
    tmp.replace(path)


@dataclass
class AtomArmParams:
    """单臂发射参数（A/B 可独立设置）。"""
    omega_peak: float = 2 * np.pi * 20e6
    g: float = 2 * np.pi * 20e6
    kappa_ex: float = 2 * np.pi * 20e6
    kappa_in: float = 2 * np.pi * 1e6
    kappa_ex_H: Optional[float] = None
    kappa_ex_V: Optional[float] = None
    kappa_in_H: Optional[float] = None
    kappa_in_V: Optional[float] = None
    gamma_sigma_plus: float = 0.0
    gamma_sigma_minus: float = 0.0
    delta_u: float = 0.0
    delta_e: float = 0.0
    delta_c_H: float = 0.0
    delta_c_V: float = 0.0
    # 偏振耦合矩阵 Alpha（2x2，默认单位阵）：
    #   [[alpha_h_plus, alpha_h_minus],
    #    [alpha_v_plus, alpha_v_minus]]
    alpha_h_plus: float = 1.0
    alpha_h_minus: float = 0.0
    alpha_v_plus: float = 0.0
    alpha_v_minus: float = 1.0


@dataclass
class EmissionParams:
    """发射阶段参数（可复用的最小物理输入）。"""
    n_bins: int = 100
    dt_ns: float = 0.5
    chi_max: int = 50
    sigma: float = DEFAULT_EMISSION_SIGMA_NS
    # 驱动包络类型（默认高斯）；支持: gaussian / sech / square
    drive_waveform_A: str = "gaussian"
    drive_waveform_B: str = "gaussian"
    arm_A: AtomArmParams = field(default_factory=AtomArmParams)
    arm_B: AtomArmParams = field(default_factory=AtomArmParams)
    delay_ns: Optional[float] = None
    delay_jitter_ns: float = DEFAULT_DELAY_JITTER_NS
    delay_random_range: Tuple[float, float] = (-10.0, 10.0)


@dataclass
class NoiseParams:
    """探测噪声参数（暗计数 + 背景噪声）。"""
    dark_rate_intrinsic_hz: float = DEFAULT_DARK_RATE_INTRINSIC_HZ
    bg_rate_mean_hz: float = DEFAULT_BG_RATE_MEAN_HZ
    bg_rate_std_hz: float = DEFAULT_BG_RATE_STD_HZ
    detector_gate_ns: float = 1.0
    dark_rate_intrinsic_hz_map: dict = field(default_factory=dict)
    bg_rate_mean_hz_map: dict = field(default_factory=dict)
    bg_rate_std_hz_map: dict = field(default_factory=dict)


@dataclass
class DetectorParams:
    """探测器参数。"""
    eta_det: float = 0.85
    ideal_det: bool = False
    eta_det_map: dict = field(default_factory=dict)
    # 残差可区分度：仅用于承载“当前模型未显式建模”的剩余退相干。
    # TODO(model-first-principles): 当以下因素全部显式建模后，删除该旋钮：
    #   1) QFC 场端噪声注入（替代探测端等效暗计数吸收）
    #   2) PMD/色散导致的跨bin时间模式混合
    #   3) 频域/时域模式重叠的显式建模（替代残差旋钮）
    v_res: float = 1.0
    # 中心站 BS 混合角：sin^2(theta)=跨端口透射概率。
    # theta=pi/4 对应理想 50/50。
    bs_theta: float = np.pi / 4


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
    eta_det_map: dict
    p_dark_intrinsic_bin_map: dict
    p_bg_bin_map: dict
    v_res: float
    window_ns: float
    window_bins: int
    noise_budget: NoiseBudget


@dataclass
class QfcFilterCavityParams:
    """QFC 后 1517nm 窄带滤波腔参数（用于显式记忆模型）。"""

    enabled: bool = True
    # 是否启用显式记忆动力学（会引入跨-bin关联并显著抬升张量复杂度）。
    dynamics_enabled: bool = False
    fwhm_mhz: float = 27.0
    detuning_mhz_A: float = 0.0
    detuning_mhz_B: float = 0.0
    eta_peak_A: float = 0.81
    eta_peak_B: float = 0.81


@dataclass
class QfcParams:
    """QFC 参数（可扫）。"""
    theta_H: float = DEFAULT_QFC_THETA_RAD
    theta_V: float = DEFAULT_QFC_THETA_RAD
    phi_H: float = 0.0
    phi_V: float = 0.0
    # QFC 背景噪声谱密度（cps/MHz），用于按“谱密度×带宽×链路η×探测η”估算默认背景率。
    # 参考量级：41.1 cps/MHz（docs/43, docs/45 讨论口径）。
    qfc_noise_sd_cps_per_mhz_A: float = DEFAULT_QFC_NOISE_SD_CPS_PER_MHZ
    qfc_noise_sd_cps_per_mhz_B: float = DEFAULT_QFC_NOISE_SD_CPS_PER_MHZ
    filter_cavity: QfcFilterCavityParams = field(default_factory=QfcFilterCavityParams)


@dataclass
class FiberParams:
    """光纤信道参数。"""
    noise_enabled: bool = True
    polarization_model: str = "perturb"
    polarization_sigma: float = 0.1
    length_km: float = 33.0
    attenuation_db_per_km: float = 0.2
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
    window_sweep_start_ns: Optional[float] = None
    window_sweep_end_ns: Optional[float] = None
    window_sweep_step_ns: Optional[float] = None
    bs_sweep_start_theta: Optional[float] = None
    bs_sweep_end_theta: Optional[float] = None
    bs_sweep_step_theta: Optional[float] = None
    length_sweep_start_km: Optional[float] = None
    length_sweep_end_km: Optional[float] = None
    length_sweep_step_km: Optional[float] = None
    attempt_rate_hz: float = 1.0
    attempt_overhead_us: float = 0.0
    # 原子等待时间模型：按单程光纤飞行时间自动绑定。
    fiber_group_velocity_mps: float = 2.0e8
    t_wait_overhead_us: float = 0.0
    # 等待时间线性系数：T_wait = scale * (L / v_g) + overhead。
    t_wait_length_scale: float = 1.0
    # 原子相干时间 T2（默认按文献基线设置）。
    t2_us: float = DEFAULT_T2_US


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


def _alpha_matrix(arm: AtomArmParams) -> np.ndarray:
    """将配置中的 Alpha 元素组装为 2x2 偏振耦合矩阵。"""
    return np.array(
        [
            [complex(arm.alpha_h_plus), complex(arm.alpha_h_minus)],
            [complex(arm.alpha_v_plus), complex(arm.alpha_v_minus)],
        ],
        dtype=complex,
    )


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
    eta_mean = 10.0 ** (-float(cfg.attenuation_db_per_km) * float(cfg.length_km) / 10.0)
    if not cfg.noise_enabled:
        return FiberChannelParams(
            polarization_model="fixed",
            polarization_sigma=0.0,
            eta_mean=eta_mean,
            eta_std=0.0,
            pdl_sigma=0.0,
            phase_drift_std=0.0,
            phase_slope_std=0.0,
            phase_jitter_std=0.0,
        )
    return FiberChannelParams(
        polarization_model=cfg.polarization_model,
        polarization_sigma=cfg.polarization_sigma,
        eta_mean=eta_mean,
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
    def _clip_value(value: float, min_value: float = 0.0, max_value: Optional[float] = None) -> float:
        clipped = max(min_value, float(value))
        if max_value is not None:
            clipped = min(clipped, max_value)
        return clipped

    def _build_detector_value_map(
        default_value: float,
        overrides: Optional[dict],
        field_name: str,
        min_value: float = 0.0,
        max_value: Optional[float] = None,
    ) -> dict:
        resolved = {
            detector: _clip_value(default_value, min_value=min_value, max_value=max_value)
            for detector in DETECTOR_CHANNELS
        }
        if not overrides:
            return resolved
        for raw_key, raw_value in overrides.items():
            detector = str(raw_key).strip().upper()
            if detector not in DETECTOR_CHANNELS:
                raise ValueError(f"{field_name} 包含未知探测器: {raw_key}")
            resolved[detector] = _clip_value(raw_value, min_value=min_value, max_value=max_value)
        return resolved

    def _rate_to_bin_probability(rate_hz: float, gate_dt_s: float, ratio: float) -> float:
        p_gate = 1.0 - np.exp(-max(0.0, float(rate_hz)) * gate_dt_s)
        p_bin = 1.0 - (1.0 - p_gate) ** ratio
        return float(np.clip(p_bin, 0.0, 1.0))

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
    ratio = emission_bin_dt_s / noise_budget.detection_gate_dt_s
    if config.detector.ideal_det:
        eta_det_map = {detector: 1.0 for detector in DETECTOR_CHANNELS}
        p_dark_intrinsic_bin_map = {detector: 0.0 for detector in DETECTOR_CHANNELS}
        p_bg_bin_map = {detector: 0.0 for detector in DETECTOR_CHANNELS}
    else:
        eta_det_map = _build_detector_value_map(
            default_value=eta_det,
            overrides=config.detector.eta_det_map,
            field_name="eta_det_map",
            min_value=0.0,
            max_value=1.0,
        )
        dark_rate_map = _build_detector_value_map(
            default_value=noise_budget.dark_rate_intrinsic_hz,
            overrides=config.noise.dark_rate_intrinsic_hz_map,
            field_name="dark_rate_intrinsic_hz_map",
            min_value=0.0,
        )
        use_qfc_sd_default_bg = (
            not config.noise.bg_rate_mean_hz_map
            and abs(float(config.noise.bg_rate_mean_hz) - DEFAULT_BG_RATE_MEAN_HZ) < 1e-12
        )
        if use_qfc_sd_default_bg:
            eta_link_mean = 10.0 ** (
                -float(config.fiber.attenuation_db_per_km) * float(config.fiber.length_km) / 10.0
            )
            filter_bw_mhz = max(float(config.qfc.filter_cavity.fwhm_mhz), 0.0)
            eta_filter_a = float(np.clip(config.qfc.filter_cavity.eta_peak_A, 0.0, 1.0))
            eta_filter_b = float(np.clip(config.qfc.filter_cavity.eta_peak_B, 0.0, 1.0))
            if not bool(config.qfc.filter_cavity.enabled):
                # 关闭窄带滤波腔时，仅保留“源谱密度×链路×探测”口径。
                eta_filter_a = 1.0
                eta_filter_b = 1.0

            sd_a = max(0.0, float(config.qfc.qfc_noise_sd_cps_per_mhz_A))
            sd_b = max(0.0, float(config.qfc.qfc_noise_sd_cps_per_mhz_B))
            bg_mean_map = {
                "H1": sd_a * filter_bw_mhz * eta_filter_a * eta_link_mean * eta_det_map["H1"],
                "V1": sd_a * filter_bw_mhz * eta_filter_a * eta_link_mean * eta_det_map["V1"],
                "H2": sd_b * filter_bw_mhz * eta_filter_b * eta_link_mean * eta_det_map["H2"],
                "V2": sd_b * filter_bw_mhz * eta_filter_b * eta_link_mean * eta_det_map["V2"],
            }
            if abs(float(config.noise.bg_rate_std_hz) - DEFAULT_BG_RATE_STD_HZ) < 1e-12 and not config.noise.bg_rate_std_hz_map:
                # 默认情况下采用 Poisson 口径的 sqrt(rate) 作为每通道波动尺度。
                bg_std_map = {
                    detector: float(np.sqrt(max(bg_mean_map[detector], 0.0)))
                    for detector in DETECTOR_CHANNELS
                }
            else:
                bg_std_map = _build_detector_value_map(
                    default_value=noise_budget.bg_rate_std_hz,
                    overrides=config.noise.bg_rate_std_hz_map,
                    field_name="bg_rate_std_hz_map",
                    min_value=0.0,
                )
        else:
            bg_mean_map = _build_detector_value_map(
                default_value=noise_budget.bg_rate_mean_hz,
                overrides=config.noise.bg_rate_mean_hz_map,
                field_name="bg_rate_mean_hz_map",
                min_value=0.0,
            )
            bg_std_map = _build_detector_value_map(
                default_value=noise_budget.bg_rate_std_hz,
                overrides=config.noise.bg_rate_std_hz_map,
                field_name="bg_rate_std_hz_map",
                min_value=0.0,
            )

        p_dark_intrinsic_bin_map = {}
        p_bg_bin_map = {}
        sampled_bg_rates = []
        for detector in DETECTOR_CHANNELS:
            p_dark_intrinsic_bin_map[detector] = _rate_to_bin_probability(
                dark_rate_map[detector],
                gate_dt_s=noise_budget.detection_gate_dt_s,
                ratio=ratio,
            )
            dark_rate_bg_local = max(0.0, rng.normal(bg_mean_map[detector], bg_std_map[detector]))
            sampled_bg_rates.append(dark_rate_bg_local)
            p_bg_bin_map[detector] = _rate_to_bin_probability(
                dark_rate_bg_local,
                gate_dt_s=noise_budget.detection_gate_dt_s,
                ratio=ratio,
            )

        if sampled_bg_rates:
            dark_rate_bg_hz = float(np.mean(sampled_bg_rates))
            bg_rate_mean_hz = float(np.mean(list(bg_mean_map.values())))
            bg_rate_std_hz = float(np.mean(list(bg_std_map.values())))
            p_bg_gate = 1.0 - np.exp(-dark_rate_bg_hz * noise_budget.detection_gate_dt_s)
            p_noise_gate = 1.0 - (1.0 - noise_budget.p_dark_intrinsic_gate) * (1.0 - p_bg_gate)
            p_bg_bin = 1.0 - (1.0 - p_bg_gate) ** ratio
            p_noise_bin = 1.0 - (1.0 - noise_budget.p_dark_intrinsic_bin) * (1.0 - p_bg_bin)
            noise_budget = replace(
                noise_budget,
                bg_rate_mean_hz=bg_rate_mean_hz,
                bg_rate_std_hz=bg_rate_std_hz,
                dark_rate_bg_hz=dark_rate_bg_hz,
                p_bg_gate=float(np.clip(p_bg_gate, 0.0, 1.0)),
                p_noise_gate=float(np.clip(p_noise_gate, 0.0, 1.0)),
                p_bg_bin=float(np.clip(p_bg_bin, 0.0, 1.0)),
                p_noise_bin=float(np.clip(p_noise_bin, 0.0, 1.0)),
            )

    return RunParameterStore(
        eta_det=eta_det,
        eta_det_map=eta_det_map,
        p_dark_intrinsic_bin_map=p_dark_intrinsic_bin_map,
        p_bg_bin_map=p_bg_bin_map,
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
        "drive_waveform_A": emission.drive_waveform_A,
        "drive_waveform_B": emission.drive_waveform_B,
        "omega_peak_A": arm_a.omega_peak,
        "omega_peak_B": arm_b.omega_peak,
        "g_A": arm_a.g,
        "g_B": arm_b.g,
        "kappa_ex_A": arm_a.kappa_ex,
        "kappa_ex_B": arm_b.kappa_ex,
        "kappa_in_A": arm_a.kappa_in,
        "kappa_in_B": arm_b.kappa_in,
        "kappa_ex_H_A": arm_a.kappa_ex_H,
        "kappa_ex_V_A": arm_a.kappa_ex_V,
        "kappa_in_H_A": arm_a.kappa_in_H,
        "kappa_in_V_A": arm_a.kappa_in_V,
        "kappa_ex_H_B": arm_b.kappa_ex_H,
        "kappa_ex_V_B": arm_b.kappa_ex_V,
        "kappa_in_H_B": arm_b.kappa_in_H,
        "kappa_in_V_B": arm_b.kappa_in_V,
        "gamma_sigma_plus_A": arm_a.gamma_sigma_plus,
        "gamma_sigma_minus_A": arm_a.gamma_sigma_minus,
        "gamma_sigma_plus_B": arm_b.gamma_sigma_plus,
        "gamma_sigma_minus_B": arm_b.gamma_sigma_minus,
        "delta_u_A": arm_a.delta_u,
        "delta_u_B": arm_b.delta_u,
        "delta_e_A": arm_a.delta_e,
        "delta_e_B": arm_b.delta_e,
        "delta_c_H_A": arm_a.delta_c_H,
        "delta_c_V_A": arm_a.delta_c_V,
        "delta_c_H_B": arm_b.delta_c_H,
        "delta_c_V_B": arm_b.delta_c_V,
        "alpha_A": [
            [float(arm_a.alpha_h_plus), float(arm_a.alpha_h_minus)],
            [float(arm_a.alpha_v_plus), float(arm_a.alpha_v_minus)],
        ],
        "alpha_B": [
            [float(arm_b.alpha_h_plus), float(arm_b.alpha_h_minus)],
            [float(arm_b.alpha_v_plus), float(arm_b.alpha_v_minus)],
        ],
        "qfc_theta_H": config.qfc.theta_H,
        "qfc_theta_V": config.qfc.theta_V,
        "qfc_phi_H": config.qfc.phi_H,
        "qfc_phi_V": config.qfc.phi_V,
        "qfc_noise_sd_cps_per_mhz_A": config.qfc.qfc_noise_sd_cps_per_mhz_A,
        "qfc_noise_sd_cps_per_mhz_B": config.qfc.qfc_noise_sd_cps_per_mhz_B,
        "filter_cavity_enabled": config.qfc.filter_cavity.enabled,
        "filter_cavity_dynamics_enabled": config.qfc.filter_cavity.dynamics_enabled,
        "filter_cavity_fwhm_mhz": config.qfc.filter_cavity.fwhm_mhz,
        "filter_cavity_detuning_mhz_A": config.qfc.filter_cavity.detuning_mhz_A,
        "filter_cavity_detuning_mhz_B": config.qfc.filter_cavity.detuning_mhz_B,
        "filter_cavity_eta_peak_A": config.qfc.filter_cavity.eta_peak_A,
        "filter_cavity_eta_peak_B": config.qfc.filter_cavity.eta_peak_B,
        "eta_det": store.eta_det,
        "eta_det_map": dict(store.eta_det_map),
        "v_res": store.v_res,
        "window_ns": store.window_ns,
        "window_bins": store.window_bins,
        "fiber_group_velocity_mps": config.run.fiber_group_velocity_mps,
        "t_wait_overhead_us": config.run.t_wait_overhead_us,
        "t_wait_length_scale": config.run.t_wait_length_scale,
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
        "p_dark_intrinsic_bin_map": dict(store.p_dark_intrinsic_bin_map),
        "p_bg_bin_map": dict(store.p_bg_bin_map),
    }


def _compute_effective_attempt_rate_hz(attempt_rate_hz: float, attempt_overhead_us: float = 0.0) -> float:
    """
    由“裸尝试率 + 单次额外开销”得到有效尝试率。

    设裸周期为 T0=1/attempt_rate_hz，附加开销为 Toverhead，
    则有效尝试率为 1/(T0+Toverhead)。
    """
    base_rate = max(0.0, float(attempt_rate_hz))
    if base_rate <= 0.0:
        return 0.0
    overhead_s = max(0.0, float(attempt_overhead_us)) * 1e-6
    cycle_s = (1.0 / base_rate) + overhead_s
    if cycle_s <= 0.0:
        return 0.0
    return 1.0 / cycle_s


def _compute_t_wait_us_from_length(
    length_km: float,
    fiber_group_velocity_mps: float = 2.0e8,
    t_wait_overhead_us: float = 0.0,
    t_wait_length_scale: float = 1.0,
) -> float:
    """
    由线性长度模型计算原子等待时间。

    模型：
      T_wait = scale * (L / v_g) + T_overhead

    其中：
      - L：用于等待估计的等效长度（km -> m）
      - v_g：光纤群速度（m/s）
      - scale：线性系数（可表达单程/往返/协议附加等待）
    """
    length_m = max(0.0, float(length_km)) * 1e3
    vg = max(1.0, float(fiber_group_velocity_mps))
    scale = max(0.0, float(t_wait_length_scale))
    overhead = max(0.0, float(t_wait_overhead_us))
    flight_us = (length_m / vg) * 1e6
    return float(max(0.0, scale * flight_us + overhead))


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

    # 原子位于链最左端：atomA(0), atomB(1)
    # 当前主路径中站点是 emitter 维度（12D = atom4D ⊗ cavity3D），
    # 因此退相干算符需在 atom 子空间定义后扩展为 atom⊗I_cavity。
    for site in (0, 1):
        site_dim = int(mps.d[site])
        if site_dim % 4 != 0:
            raise ValueError(f"原子站点维度应可被4整除，得到 site={site}, dim={site_dim}")
        cavity_dim = site_dim // 4

        z_atom = np.diag([1.0, -1.0, 1.0, 1.0]).astype(complex)
        z_op = np.kron(z_atom, np.eye(cavity_dim, dtype=complex))
        k0 = np.sqrt(1.0 - p_dephase) * np.eye(site_dim, dtype=complex)
        k1 = np.sqrt(p_dephase) * z_op
        mps.apply_kraus_one_site(site, [k0, k1], rng=rng)
    mps.canonicalize(renormalize=True)

    if verbose:
        print(f"原子退相干：已应用 p_dephase={p_dephase:.4e}")


@dataclass
class PipelineHooks:
    on_stage: Optional[Callable[[str], None]] = None
    after_emission: Optional[Callable[[Any], None]] = None
    after_qfc_filter: Optional[Callable[[Any, Tuple[float, float]], None]] = None
    after_fiber: Optional[Callable[[Any, tuple, Tuple[float, float]], None]] = None
    after_bs: Optional[Callable[[Any, tuple, Tuple[float, float]], None]] = None


@dataclass
class PipelineResult:
    emission: Any
    mps: Any
    p_qubit_emit: float
    fiber_sample: Optional[tuple]
    qfc_theta_H: float
    qfc_theta_V: float
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
    bs_theta: float = np.pi / 4,
) -> dict:
    """
    统一拼装 run_detection_pipeline 的公共参数，避免多处重复与分叉。
    """
    return {
        "mps": pipe.mps,
        "n_bins": pipe.emission.get_n_bins(),
        "eta_det": dict(param_store.eta_det_map),
        "window_bins": int(param_store.window_bins),
        "rng": rng,
        "verbose": verbose,
        "bs_unitary": bs_unitary,
        "bs_theta": float(bs_theta),
        "fiber_sample": pipe.fiber_sample,
        "v_res": float(param_store.v_res),
    }


_WINDOW_BINS_UNSET = object()


def run_trial_physics_core(
    *,
    rng: Optional[np.random.Generator],
    config: SimConfig,
    delay_ns: Optional[float],
    delay_jitter_ns: Optional[float],
    verbose: bool,
    debug: bool,
    hooks: Optional[PipelineHooks],
    emission_diagnostics: bool,
) -> PipelineResult:
    """
    统一单次物理链路核心（发射->QFC/滤波->光纤->退相干->BS并入测量）。

    说明：
    - 该函数不做探测统计；
    - 所有 task 应复用该入口，避免重复造轮子。
    """
    run_rng = rng or np.random.default_rng()
    t_wait_us = _compute_t_wait_us_from_length(
        length_km=config.fiber.length_km,
        fiber_group_velocity_mps=config.run.fiber_group_velocity_mps,
        t_wait_overhead_us=config.run.t_wait_overhead_us,
        t_wait_length_scale=config.run.t_wait_length_scale,
    )
    return run_emission_to_bs(
        emission=config.emission,
        rng=run_rng,
        fiber=config.fiber,
        qfc=config.qfc,
        delay_ns=delay_ns,
        delay_jitter_ns=delay_jitter_ns,
        verbose=verbose,
        hooks=hooks,
        t_wait_us=t_wait_us,
        t2_us=float(config.run.t2_us),
        record_timings=debug,
        emission_diagnostics=emission_diagnostics,
    )


def run_detection_core_from_pipe(
    *,
    pipe: PipelineResult,
    config: SimConfig,
    rng: np.random.Generator,
    coincidence_window_ns: float,
    shots_per_run: int,
    compute_metrics: bool,
    verbose: bool,
    bs_theta: Optional[float] = None,
    window_bins=_WINDOW_BINS_UNSET,
    param_store: Optional[RunParameterStore] = None,
    p_dark_intrinsic_map: Optional[dict] = None,
    p_bg_source_map: Optional[dict] = None,
):
    """
    基于已生成的单次物理链路态，执行统一探测 core。

    参数说明：
    - 若传入 param_store，将复用同一噪声预算（适用于 BSM/LENGTH 等扫描同一 run 内对比）；
    - window_bins 可显式覆盖：
      - 传入整数：使用该窗口；
      - 传入 None：禁用窗口限制（保留全记录）；
      - 不传：使用 param_store.window_bins。
    """
    from ..physics.gates import bs_gate_6d
    from ..simulation import run_detection_pipeline

    if param_store is None:
        param_store = _build_run_parameter_store(
            config=config,
            emission_bin_dt_s=pipe.emission.dt_s,
            coincidence_window_ns=float(coincidence_window_ns),
            rng=rng,
        )

    bs_theta_value = float(config.detector.bs_theta if bs_theta is None else bs_theta)
    detect_common = _build_detection_kwargs(
        pipe=pipe,
        param_store=param_store,
        rng=rng,
        verbose=verbose,
        bs_unitary=bs_gate_6d(bs_theta_value),
        bs_theta=bs_theta_value,
    )
    if window_bins is not _WINDOW_BINS_UNSET:
        detect_common["window_bins"] = None if window_bins is None else int(window_bins)

    dark_map = (
        dict(param_store.p_dark_intrinsic_bin_map)
        if p_dark_intrinsic_map is None
        else dict(p_dark_intrinsic_map)
    )
    bg_map = (
        dict(param_store.p_bg_bin_map)
        if p_bg_source_map is None
        else dict(p_bg_source_map)
    )
    pipeline = run_detection_pipeline(
        **detect_common,
        p_dark_intrinsic=dark_map,
        p_bg_source=bg_map,
        n_samples=int(shots_per_run),
        compute_metrics=bool(compute_metrics),
    )
    return param_store, pipeline


def run_trial_detection_core(
    *,
    rng: Optional[np.random.Generator],
    config: SimConfig,
    delay_ns: Optional[float],
    delay_jitter_ns: Optional[float],
    coincidence_window_ns: float,
    shots_per_run: int,
    compute_metrics: bool,
    verbose: bool,
    debug: bool,
    hooks: Optional[PipelineHooks],
    emission_diagnostics: bool,
    bs_theta: Optional[float] = None,
    window_bins=_WINDOW_BINS_UNSET,
    p_dark_intrinsic_map: Optional[dict] = None,
    p_bg_source_map: Optional[dict] = None,
):
    """
    统一 trial+detect 入口：先跑物理链路，再跑探测统计。

    返回：(pipe, param_store, pipeline)
    """
    run_rng = rng or np.random.default_rng()
    pipe = run_trial_physics_core(
        rng=run_rng,
        config=config,
        delay_ns=delay_ns,
        delay_jitter_ns=delay_jitter_ns,
        verbose=verbose,
        debug=debug,
        hooks=hooks,
        emission_diagnostics=emission_diagnostics,
    )
    param_store, pipeline = run_detection_core_from_pipe(
        pipe=pipe,
        config=config,
        rng=run_rng,
        coincidence_window_ns=float(coincidence_window_ns),
        shots_per_run=int(shots_per_run),
        compute_metrics=bool(compute_metrics),
        verbose=verbose,
        bs_theta=bs_theta,
        window_bins=window_bins,
        p_dark_intrinsic_map=p_dark_intrinsic_map,
        p_bg_source_map=p_bg_source_map,
    )
    return pipe, param_store, pipeline


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
    t2_us: float = DEFAULT_T2_US,
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
        apply_qfc_filter_memory_chain,
    )

    if hooks is None:
        hooks = PipelineHooks()
    if fiber is None:
        fiber = FiberParams()
    if qfc is None:
        qfc = QfcParams()
    emission_chi_max = int(emission.chi_max)
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
        Alpha_A=_alpha_matrix(emission.arm_A),
        Alpha_B=_alpha_matrix(emission.arm_B),
        omega_peak_A=emission.arm_A.omega_peak,
        omega_peak_B=emission.arm_B.omega_peak,
        drive_waveform_A=emission.drive_waveform_A,
        drive_waveform_B=emission.drive_waveform_B,
        sigma=emission.sigma,
        delay_ns=delay_ns,
        delay_jitter_ns=delay_jitter_ns,
        g_A=emission.arm_A.g,
        g_B=emission.arm_B.g,
        kappa_ex_A=emission.arm_A.kappa_ex,
        kappa_ex_B=emission.arm_B.kappa_ex,
        kappa_in_A=emission.arm_A.kappa_in,
        kappa_in_B=emission.arm_B.kappa_in,
        kappa_ex_H_A=emission.arm_A.kappa_ex_H,
        kappa_ex_V_A=emission.arm_A.kappa_ex_V,
        kappa_in_H_A=emission.arm_A.kappa_in_H,
        kappa_in_V_A=emission.arm_A.kappa_in_V,
        kappa_ex_H_B=emission.arm_B.kappa_ex_H,
        kappa_ex_V_B=emission.arm_B.kappa_ex_V,
        kappa_in_H_B=emission.arm_B.kappa_in_H,
        kappa_in_V_B=emission.arm_B.kappa_in_V,
        gamma_sigma_plus_A=emission.arm_A.gamma_sigma_plus,
        gamma_sigma_minus_A=emission.arm_A.gamma_sigma_minus,
        gamma_sigma_plus_B=emission.arm_B.gamma_sigma_plus,
        gamma_sigma_minus_B=emission.arm_B.gamma_sigma_minus,
        delta_u_A=emission.arm_A.delta_u,
        delta_u_B=emission.arm_B.delta_u,
        delta_e_A=emission.arm_A.delta_e,
        delta_e_B=emission.arm_B.delta_e,
        delta_c_H_A=emission.arm_A.delta_c_H,
        delta_c_V_A=emission.arm_A.delta_c_V,
        delta_c_H_B=emission.arm_B.delta_c_H,
        delta_c_V_B=emission.arm_B.delta_c_V,
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

    _call_stage("QFC + 滤波记忆(态端)")
    t0 = time.perf_counter() if timings is not None else None
    qfc_theta_H = float(qfc.theta_H)
    qfc_theta_V = float(qfc.theta_V)
    qfc_phi_H = float(qfc.phi_H)
    qfc_phi_V = float(qfc.phi_V)
    mps = apply_qfc_filter_memory_chain(
        mps=mps,
        n_bins=emission.get_n_bins(),
        dt_s=emission.dt_s,
        rng=rng,
        theta_H=qfc_theta_H,
        theta_V=qfc_theta_V,
        phi_H=qfc_phi_H,
        phi_V=qfc_phi_V,
        filter_enabled=bool(qfc.filter_cavity.enabled),
        filter_dynamics_enabled=bool(qfc.filter_cavity.enabled and qfc.filter_cavity.dynamics_enabled),
        filter_fwhm_mhz=float(qfc.filter_cavity.fwhm_mhz),
        filter_detuning_mhz_A=float(qfc.filter_cavity.detuning_mhz_A),
        filter_detuning_mhz_B=float(qfc.filter_cavity.detuning_mhz_B),
        filter_eta_peak_A=float(qfc.filter_cavity.eta_peak_A),
        filter_eta_peak_B=float(qfc.filter_cavity.eta_peak_B),
        chi_max=emission_chi_max,
        verbose=verbose,
    )
    emission.mps = mps
    if timings is not None and t0 is not None:
        timings["qfc_filter_memory"] = time.perf_counter() - t0
    if hooks.after_qfc_filter is not None:
        hooks.after_qfc_filter(
            emission,
            qfc_params=(qfc_theta_H, qfc_theta_V),
        )
        if verbose:
            print("QFC/滤波记忆已在态端显式作用。")

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
        )

    return PipelineResult(
        emission=emission,
        mps=mps,
        p_qubit_emit=p_qubit_emit,
        fiber_sample=fiber_sample,
        qfc_theta_H=qfc_theta_H,
        qfc_theta_V=qfc_theta_V,
        t_wait_us=t_wait_us,
        t2_us=t2_us,
        p_dephase=p_dephase,
        timings=timings,
    )


