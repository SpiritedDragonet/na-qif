# -*- coding: utf-8 -*-
"""
统一参数扫描引擎：
- 统一解析 scan_<param>_{start,end,step} 三元组；
- 闭合性校验（要么全无，要么全有）；
- 生成 1D/2D/N 维笛卡尔积；
- 为旧任务提供 builder 级 alias；
- 提供 PARAM_SCAN 通用单点 runner。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ..simulation import compute_pauli_correlators_and_chsh
from . import hom
from .common import (
    SimConfig,
    _compute_effective_attempt_rate_hz,
    _compute_t_wait_us_from_length,
    run_trial_detection_core,
    write_click_records,
    write_declared_density_matrix,
)


FAMILY_SIM_CORE = "SIM_CORE"
FAMILY_HOM_CORE = "HOM_CORE"


@dataclass(frozen=True)
class ScanAxisSpec:
    key: str
    family: str
    start_attr: str
    end_attr: str
    step_attr: str
    legacy_start_attr: Optional[str] = None
    legacy_end_attr: Optional[str] = None
    legacy_step_attr: Optional[str] = None
    min_value: Optional[float] = None
    min_open: bool = False
    max_value: Optional[float] = None
    max_open: bool = False

    @property
    def legacy_triplet(self) -> tuple[Optional[str], Optional[str], Optional[str]]:
        return (self.legacy_start_attr, self.legacy_end_attr, self.legacy_step_attr)

    @property
    def canonical_triplet(self) -> tuple[str, str, str]:
        return (self.start_attr, self.end_attr, self.step_attr)


_SCAN_AXIS_ORDER = (
    "window_ns",
    "tau_ns",
    "qfc_noise_sd_cps_per_mhz",
    "qfc_eta",
    "eta_det",
    "bg_rate_mean_hz",
    "bs_theta",
    "length_km",
)


SCAN_AXIS_SPECS: dict[str, ScanAxisSpec] = {
    "window_ns": ScanAxisSpec(
        key="window_ns",
        family=FAMILY_SIM_CORE,
        start_attr="scan_window_ns_start",
        end_attr="scan_window_ns_end",
        step_attr="scan_window_ns_step",
        legacy_start_attr="window_sweep_start_ns",
        legacy_end_attr="window_sweep_end_ns",
        legacy_step_attr="window_sweep_step_ns",
        min_value=0.0,
    ),
    "tau_ns": ScanAxisSpec(
        key="tau_ns",
        family=FAMILY_HOM_CORE,
        start_attr="scan_tau_ns_start",
        end_attr="scan_tau_ns_end",
        step_attr="scan_tau_ns_step",
    ),
    "qfc_noise_sd_cps_per_mhz": ScanAxisSpec(
        key="qfc_noise_sd_cps_per_mhz",
        family=FAMILY_SIM_CORE,
        start_attr="scan_qfc_noise_sd_cps_per_mhz_start",
        end_attr="scan_qfc_noise_sd_cps_per_mhz_end",
        step_attr="scan_qfc_noise_sd_cps_per_mhz_step",
        legacy_start_attr="qfc_noise_sweep_start_cps_per_mhz",
        legacy_end_attr="qfc_noise_sweep_end_cps_per_mhz",
        legacy_step_attr="qfc_noise_sweep_step_cps_per_mhz",
        min_value=0.0,
    ),
    "qfc_eta": ScanAxisSpec(
        key="qfc_eta",
        family=FAMILY_SIM_CORE,
        start_attr="scan_qfc_eta_start",
        end_attr="scan_qfc_eta_end",
        step_attr="scan_qfc_eta_step",
        legacy_start_attr="qfc_eta_sweep_start",
        legacy_end_attr="qfc_eta_sweep_end",
        legacy_step_attr="qfc_eta_sweep_step",
        min_value=0.0,
        max_value=1.0,
    ),
    "eta_det": ScanAxisSpec(
        key="eta_det",
        family=FAMILY_SIM_CORE,
        start_attr="scan_eta_det_start",
        end_attr="scan_eta_det_end",
        step_attr="scan_eta_det_step",
        legacy_start_attr="eta_det_sweep_start",
        legacy_end_attr="eta_det_sweep_end",
        legacy_step_attr="eta_det_sweep_step",
        min_value=0.0,
        min_open=True,
        max_value=1.0,
    ),
    "bg_rate_mean_hz": ScanAxisSpec(
        key="bg_rate_mean_hz",
        family=FAMILY_SIM_CORE,
        start_attr="scan_bg_rate_mean_hz_start",
        end_attr="scan_bg_rate_mean_hz_end",
        step_attr="scan_bg_rate_mean_hz_step",
        legacy_start_attr="bg_mean_sweep_start_hz",
        legacy_end_attr="bg_mean_sweep_end_hz",
        legacy_step_attr="bg_mean_sweep_step_hz",
        min_value=0.0,
    ),
    "bs_theta": ScanAxisSpec(
        key="bs_theta",
        family=FAMILY_SIM_CORE,
        start_attr="scan_bs_theta_start",
        end_attr="scan_bs_theta_end",
        step_attr="scan_bs_theta_step",
        legacy_start_attr="bs_sweep_start_theta",
        legacy_end_attr="bs_sweep_end_theta",
        legacy_step_attr="bs_sweep_step_theta",
        min_value=0.0,
        max_value=float(np.pi / 2.0),
    ),
    "length_km": ScanAxisSpec(
        key="length_km",
        family=FAMILY_SIM_CORE,
        start_attr="scan_length_km_start",
        end_attr="scan_length_km_end",
        step_attr="scan_length_km_step",
        legacy_start_attr="length_sweep_start_km",
        legacy_end_attr="length_sweep_end_km",
        legacy_step_attr="length_sweep_step_km",
        min_value=0.0,
    ),
}


ALIAS_REQUIRED_AXES: dict[str, tuple[str, ...]] = {
    "WINDOW_SCAN": ("window_ns",),
    "HOM": ("tau_ns",),
    "BSM_SCAN": ("bs_theta",),
    "LENGTH_SCAN": ("length_km",),
    "QFC_NOISE_SCAN": ("qfc_noise_sd_cps_per_mhz",),
    "QFC_EFF_NOISE_SCAN": ("qfc_eta", "qfc_noise_sd_cps_per_mhz"),
    "DETECTOR_BG_SCAN": ("eta_det", "bg_rate_mean_hz"),
}


def _triplet_state(values: tuple[object, object, object]) -> str:
    present = [value is not None for value in values]
    if all(present):
        return "full"
    if not any(present):
        return "empty"
    return "partial"


def _triplet_values(obj, attrs: tuple[str, str, str]) -> tuple[object, object, object]:
    return (getattr(obj, attrs[0], None), getattr(obj, attrs[1], None), getattr(obj, attrs[2], None))


def _float_value(value: object, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 需要可解析为浮点数，当前={value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{field_name} 必须是有限数值，当前={value!r}")
    return out


def _validate_axis_value(spec: ScanAxisSpec, value: float, field_name: str) -> None:
    if spec.min_value is not None:
        if spec.min_open:
            if not (value > spec.min_value):
                raise ValueError(f"{field_name} 必须 > {spec.min_value}")
        elif value < spec.min_value:
            raise ValueError(f"{field_name} 必须 >= {spec.min_value}")
    if spec.max_value is not None:
        if spec.max_open:
            if not (value < spec.max_value):
                raise ValueError(f"{field_name} 必须 < {spec.max_value}")
        elif value > spec.max_value:
            raise ValueError(f"{field_name} 必须 <= {spec.max_value}")


def _build_scan_values(start: float, end: float, step: float) -> list[float]:
    values = []
    cursor = float(start)
    while cursor <= end + 1e-12:
        values.append(round(float(cursor), 9))
        cursor += float(step)
    if not values:
        values = [round(float(start), 9)]
    return values


def _qfc_eta_to_theta(qfc_eta: float) -> float:
    eta = float(np.clip(qfc_eta, 0.0, 1.0))
    return float(np.arcsin(np.sqrt(eta)))


def sync_scan_alias_fields(run_config) -> None:
    """
    同步 canonical <-> legacy 扫描字段：
    - canonical 全无 + legacy 全有：legacy -> canonical
    - canonical 全有 + legacy 全无：canonical -> legacy
    - 两边都全有：canonical 覆盖 legacy
    - 任一侧部分给定：立即报错（闭合性不满足）
    """
    for key in _SCAN_AXIS_ORDER:
        spec = SCAN_AXIS_SPECS[key]
        canonical_attrs = spec.canonical_triplet
        canonical_values = _triplet_values(run_config, canonical_attrs)
        canonical_state = _triplet_state(canonical_values)
        if canonical_state == "partial":
            raise ValueError(
                f"{spec.start_attr}/{spec.end_attr}/{spec.step_attr} 必须同时给定或同时省略"
            )

        legacy_attrs = spec.legacy_triplet
        if any(name is None for name in legacy_attrs):
            continue
        if not all(hasattr(run_config, name) for name in legacy_attrs):
            continue
        legacy_attr_triplet = (legacy_attrs[0], legacy_attrs[1], legacy_attrs[2])  # type: ignore[index]
        legacy_values = _triplet_values(run_config, legacy_attr_triplet)
        legacy_state = _triplet_state(legacy_values)
        if legacy_state == "partial":
            raise ValueError(
                f"{legacy_attr_triplet[0]}/{legacy_attr_triplet[1]}/{legacy_attr_triplet[2]} 必须同时给定或同时省略"
            )

        if canonical_state == "empty" and legacy_state == "full":
            for attr, value in zip(canonical_attrs, legacy_values):
                setattr(run_config, attr, value)
            continue
        if canonical_state == "full" and legacy_state == "empty":
            for attr, value in zip(legacy_attr_triplet, canonical_values):
                setattr(run_config, attr, value)
            continue
        if canonical_state == "full" and legacy_state == "full":
            for attr, value in zip(legacy_attr_triplet, canonical_values):
                setattr(run_config, attr, value)


def collect_active_scan_triplets(run_config) -> dict[str, tuple[float, float, float]]:
    active: dict[str, tuple[float, float, float]] = {}
    for key in _SCAN_AXIS_ORDER:
        spec = SCAN_AXIS_SPECS[key]
        start_raw, end_raw, step_raw = _triplet_values(run_config, spec.canonical_triplet)
        state = _triplet_state((start_raw, end_raw, step_raw))
        if state == "empty":
            continue
        if state == "partial":
            raise ValueError(
                f"{spec.start_attr}/{spec.end_attr}/{spec.step_attr} 必须同时给定或同时省略"
            )
        start = _float_value(start_raw, spec.start_attr)
        end = _float_value(end_raw, spec.end_attr)
        step = _float_value(step_raw, spec.step_attr)
        if step <= 0.0:
            raise ValueError(f"{spec.step_attr} 必须 > 0")
        if end < start:
            raise ValueError(f"{spec.end_attr} 必须 >= {spec.start_attr}")
        _validate_axis_value(spec, start, spec.start_attr)
        _validate_axis_value(spec, end, spec.end_attr)
        active[key] = (start, end, step)
    return active


def validate_param_scan_config(config: SimConfig) -> None:
    active = collect_active_scan_triplets(config.run)
    if not active:
        raise ValueError(
            "PARAM_SCAN 需要至少一个完整维度：scan_<param>_{start,end,step}"
        )
    families = {SCAN_AXIS_SPECS[key].family for key in active}
    if len(families) > 1:
        raise ValueError("PARAM_SCAN 不支持同时混用 HOM 与 SIM 维度")
    if FAMILY_SIM_CORE in families and float(config.run.attempt_rate_hz) <= 0.0:
        raise ValueError("attempt_rate_hz 必须 > 0")
    if FAMILY_SIM_CORE in families and float(config.run.attempt_overhead_us) < 0.0:
        raise ValueError("attempt_overhead_us 必须 >= 0")


def _resolve_alias_axes(config: SimConfig, task_type: str) -> tuple[list[str], dict[str, list[float]]]:
    task_type = str(task_type).upper()
    active = collect_active_scan_triplets(config.run)
    required = ALIAS_REQUIRED_AXES.get(task_type, ())
    if task_type == "WINDOW_SCAN":
        extras = [key for key in active if key not in {"window_ns"}]
        if extras:
            raise ValueError(f"WINDOW_SCAN 不接受额外维度：{', '.join(extras)}")
        if "window_ns" not in active:
            raise ValueError("WINDOW_SCAN 需要完整维度 scan_window_ns_{start,end,step}")
        start, end, step = active["window_ns"]
        return ["window_ns"], {"window_ns": _build_scan_values(start, end, step)}

    if task_type == "HOM":
        extras = [key for key in active if key not in {"tau_ns"}]
        if extras:
            raise ValueError(f"HOM 不接受额外维度：{', '.join(extras)}")
        if "tau_ns" in active:
            start, end, step = active["tau_ns"]
            return ["tau_ns"], {"tau_ns": _build_scan_values(start, end, step)}
        if config.hom is None:
            raise ValueError("HOM 缺少 tau 维度配置")
        tau_values = [float(v) for v in hom._build_hom_tau_values(config.hom)]
        return ["tau_ns"], {"tau_ns": tau_values}

    missing = [axis_key for axis_key in required if axis_key not in active]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{task_type} 缺少维度：{joined}")
    extras = [key for key in active if key not in required]
    if extras:
        raise ValueError(f"{task_type} 不接受额外维度：{', '.join(extras)}")
    axis_values: dict[str, list[float]] = {}
    for axis_key in required:
        start, end, step = active[axis_key]
        axis_values[axis_key] = _build_scan_values(start, end, step)
    return list(required), axis_values


def resolve_alias_axis_values(
    config: SimConfig,
    task_type: str,
) -> tuple[list[str], dict[str, list[float]]]:
    return _resolve_alias_axes(config, task_type)


def validate_alias_scan_task(config: SimConfig, task_type: str) -> None:
    axis_keys, _axis_values = _resolve_alias_axes(config, task_type)
    if any(SCAN_AXIS_SPECS[key].family == FAMILY_SIM_CORE for key in axis_keys):
        if float(config.run.attempt_rate_hz) <= 0.0:
            raise ValueError("attempt_rate_hz 必须 > 0")
        if float(config.run.attempt_overhead_us) < 0.0:
            raise ValueError("attempt_overhead_us 必须 >= 0")


def _cartesian_points(axis_keys: list[str], axis_values: dict[str, list[float]]) -> Iterator[tuple[int, dict[str, float]]]:
    if not axis_keys:
        return
    pools = [axis_values[key] for key in axis_keys]
    for point_index, values in enumerate(itertools.product(*pools)):
        point = {axis_key: float(value) for axis_key, value in zip(axis_keys, values)}
        yield point_index, point


def iter_window_scan_alias_core_tasks(config: SimConfig) -> Iterator[dict]:
    _resolve_alias_axes(config, "WINDOW_SCAN")
    for run_index in range(config.run.runs):
        yield {
            "id": f"wscan_run_{run_index:06d}",
            "experiment": "WINDOW_SCAN",
            "run_index": run_index,
            "payload": {},
        }


def iter_hom_scan_alias_core_tasks(config: SimConfig) -> Iterator[dict]:
    axis_keys, axis_values = _resolve_alias_axes(config, "HOM")
    _ = axis_keys
    tau_values = axis_values["tau_ns"]
    window_ns = (
        float(config.hom.window_ns)
        if config.hom is not None
        else float(config.run.window_ns)
    )
    for tau in tau_values:
        for run_index in range(config.run.runs):
            yield {
                "id": f"hom_tau_{float(tau):+.3f}_run_{run_index:06d}",
                "experiment": "HOM",
                "run_index": run_index,
                "payload": {
                    "tau_ns": float(tau),
                    "window_ns": window_ns,
                },
            }


def iter_bsm_scan_alias_core_tasks(config: SimConfig) -> Iterator[dict]:
    axis_keys, axis_values = _resolve_alias_axes(config, "BSM_SCAN")
    _ = axis_keys
    values = axis_values["bs_theta"]
    for theta_index, bs_theta in enumerate(values):
        for run_index in range(config.run.runs):
            yield {
                "id": f"bscan_theta_{theta_index:04d}_run_{run_index:06d}",
                "experiment": "BSM_SCAN",
                "run_index": run_index,
                "payload": {"bs_theta": float(bs_theta)},
            }


def iter_length_scan_alias_core_tasks(config: SimConfig) -> Iterator[dict]:
    axis_keys, axis_values = _resolve_alias_axes(config, "LENGTH_SCAN")
    _ = axis_keys
    values = axis_values["length_km"]
    for length_index, length_km in enumerate(values):
        for run_index in range(config.run.runs):
            yield {
                "id": f"lscan_len_{length_index:04d}_run_{run_index:06d}",
                "experiment": "LENGTH_SCAN",
                "run_index": run_index,
                "payload": {"length_km": float(length_km)},
            }


def iter_qfc_noise_scan_alias_core_tasks(config: SimConfig) -> Iterator[dict]:
    axis_keys, axis_values = _resolve_alias_axes(config, "QFC_NOISE_SCAN")
    _ = axis_keys
    values = axis_values["qfc_noise_sd_cps_per_mhz"]
    for noise_index, noise_sd in enumerate(values):
        for run_index in range(config.run.runs):
            yield {
                "id": f"qscan_noise_{noise_index:04d}_run_{run_index:06d}",
                "experiment": "QFC_NOISE_SCAN",
                "run_index": run_index,
                "payload": {"qfc_noise_sd_cps_per_mhz": float(noise_sd)},
            }


def iter_qfc_eff_noise_scan_alias_core_tasks(config: SimConfig) -> Iterator[dict]:
    axis_keys, axis_values = _resolve_alias_axes(config, "QFC_EFF_NOISE_SCAN")
    _ = axis_keys
    eta_values = axis_values["qfc_eta"]
    noise_values = axis_values["qfc_noise_sd_cps_per_mhz"]
    for eta_index, qfc_eta in enumerate(eta_values):
        for noise_index, noise_sd in enumerate(noise_values):
            for run_index in range(config.run.runs):
                yield {
                    "id": f"qescan_eta_{eta_index:04d}_noise_{noise_index:04d}_run_{run_index:06d}",
                    "experiment": "QFC_EFF_NOISE_SCAN",
                    "run_index": run_index,
                    "payload": {
                        "qfc_eta": float(qfc_eta),
                        "qfc_noise_sd_cps_per_mhz": float(noise_sd),
                    },
                }


def iter_detector_bg_scan_alias_core_tasks(config: SimConfig) -> Iterator[dict]:
    axis_keys, axis_values = _resolve_alias_axes(config, "DETECTOR_BG_SCAN")
    _ = axis_keys
    eta_values = axis_values["eta_det"]
    bg_values = axis_values["bg_rate_mean_hz"]
    for eta_index, eta_det in enumerate(eta_values):
        for bg_index, bg_rate_mean_hz in enumerate(bg_values):
            for run_index in range(config.run.runs):
                yield {
                    "id": f"dscan_eta_{eta_index:04d}_bg_{bg_index:04d}_run_{run_index:06d}",
                    "experiment": "DETECTOR_BG_SCAN",
                    "run_index": run_index,
                    "payload": {
                        "eta_det": float(eta_det),
                        "bg_rate_mean_hz": float(bg_rate_mean_hz),
                    },
                }


def iter_param_scan_core_tasks(config: SimConfig) -> Iterator[dict]:
    validate_param_scan_config(config)
    active = collect_active_scan_triplets(config.run)
    axis_keys = [key for key in _SCAN_AXIS_ORDER if key in active]
    axis_values = {
        key: _build_scan_values(*active[key])  # type: ignore[arg-type]
        for key in axis_keys
    }
    family = SCAN_AXIS_SPECS[axis_keys[0]].family
    for point_index, point in _cartesian_points(axis_keys, axis_values):
        for run_index in range(config.run.runs):
            yield {
                "id": f"pscan_point_{point_index:06d}_run_{run_index:06d}",
                "experiment": "PARAM_SCAN",
                "run_index": run_index,
                "payload": {
                    "scan_axes": list(axis_keys),
                    "scan_family": family,
                    "scan_point": point,
                },
            }


def _point_key(point: dict[str, float]) -> str:
    parts = []
    for key in sorted(point):
        parts.append(f"{key}={float(point[key]):.9f}")
    return "|".join(parts) if parts else "point=default"


def _run_param_scan_hom_point(
    *,
    task: dict,
    config: SimConfig,
    raw_dir: Path,
    point: dict[str, float],
    shots_per_run: int,
    run_index: int,
    seed: Optional[int],
    should_abort=None,
) -> dict:
    tau_ns = float(point.get("tau_ns", 0.0))
    window_ns = float(point.get("window_ns", config.run.window_ns))
    coinc, p_arrive, click_records = hom._run_hom_run(
        tau_ns=tau_ns,
        shots_per_run=shots_per_run,
        config=config,
        window_ns=window_ns,
        delay_jitter_ns=0.0,
        verbose=False,
        debug=config.run.debug,
        rng_seed=seed,
        should_abort=should_abort,
    )
    write_click_records(raw_dir, click_records)
    return {
        "run_index": run_index,
        "scan_family": FAMILY_HOM_CORE,
        "scan_point": point,
        "window_ns": window_ns,
        "shots": int(shots_per_run),
        "success": int(coinc),
        "p_arrive": float(p_arrive),
        "coinc": int(coinc),
        "tau_ns": tau_ns,
    }


def _run_param_scan_sim_point(
    *,
    task: dict,
    config: SimConfig,
    raw_dir: Path,
    point: dict[str, float],
    shots_per_run: int,
    run_index: int,
    seed: Optional[int],
    should_abort=None,
) -> dict:
    _ = task
    run_rng = np.random.default_rng(seed)
    base_state = {
        "window_ns": float(config.run.window_ns),
        "length_km": float(config.fiber.length_km),
        "qfc_theta_h": float(config.qfc.theta_H),
        "qfc_theta_v": float(config.qfc.theta_V),
        "qfc_noise_a": float(config.qfc.qfc_noise_sd_cps_per_mhz_A),
        "qfc_noise_b": float(config.qfc.qfc_noise_sd_cps_per_mhz_B),
        "eta_det": float(config.detector.eta_det),
        "eta_det_map": dict(config.detector.eta_det_map),
        "bg_rate_mean_hz": float(config.noise.bg_rate_mean_hz),
        "bg_rate_mean_hz_map": dict(config.noise.bg_rate_mean_hz_map),
    }
    bs_theta_override = None
    try:
        for key, raw_value in point.items():
            value = float(raw_value)
            if key == "window_ns":
                config.run.window_ns = value
            elif key == "length_km":
                config.fiber.length_km = value
            elif key == "qfc_noise_sd_cps_per_mhz":
                config.qfc.qfc_noise_sd_cps_per_mhz_A = value
                config.qfc.qfc_noise_sd_cps_per_mhz_B = value
            elif key == "qfc_eta":
                theta = _qfc_eta_to_theta(value)
                config.qfc.theta_H = theta
                config.qfc.theta_V = theta
            elif key == "eta_det":
                config.detector.eta_det = value
                config.detector.eta_det_map = {}
            elif key == "bg_rate_mean_hz":
                config.noise.bg_rate_mean_hz = value
                config.noise.bg_rate_mean_hz_map = {}
            elif key == "bs_theta":
                bs_theta_override = value

        t_wait_us = _compute_t_wait_us_from_length(
            length_km=float(config.fiber.length_km),
            fiber_group_velocity_mps=float(config.run.fiber_group_velocity_mps),
            t_wait_overhead_us=float(config.run.t_wait_overhead_us),
            t_wait_length_scale=float(config.run.t_wait_length_scale),
        )
        attempt_rate_hz_eff = _compute_effective_attempt_rate_hz(
            float(config.run.attempt_rate_hz),
            float(config.run.attempt_overhead_us),
            wait_time_us=t_wait_us,
        )

        _pipe, _param_store, pipeline = run_trial_detection_core(
            rng=run_rng,
            config=config,
            delay_ns=None,
            delay_jitter_ns=None,
            coincidence_window_ns=float(config.run.window_ns),
            shots_per_run=shots_per_run,
            compute_metrics=True,
            verbose=False,
            debug=config.run.debug,
            hooks=None,
            emission_diagnostics=False,
            bs_theta=bs_theta_override,
            should_abort=should_abort,
        )
        enum_main = pipeline.metrics
        if enum_main is None:
            raise RuntimeError("PARAM_SCAN 需要 compute_metrics=True 且返回有效枚举结果")

        corr_exx_vals = []
        corr_eyy_vals = []
        corr_ezz_vals = []
        chsh_vals = []
        click_records = []
        for shot_index, sample in enumerate(pipeline.samples):
            p_true_given_record = float(
                np.clip(getattr(sample, "p_true_given_record", 0.0), 0.0, 1.0)
            )
            p_bg_assist_given_record = float(
                np.clip(getattr(sample, "p_bg_assist_given_record", 0.0), 0.0, 1.0)
            )
            p_intrinsic_dark_assist_given_record = float(
                np.clip(
                    getattr(sample, "p_intrinsic_dark_assist_given_record", 0.0),
                    0.0,
                    1.0,
                )
            )
            click_pairs = [
                (
                    c.detector,
                    c.bin_index,
                    bool(getattr(c, "is_dark", False)),
                    str(getattr(c, "source", "signal")),
                )
                for c in sample.clicks
            ]
            click_records.append(
                {
                    "shot_index": shot_index,
                    "success": bool(sample.success),
                    "bell": sample.bell_state,
                    "clicks": click_pairs,
                    "p_true_given_record": p_true_given_record,
                    "p_bg_assist_given_record": p_bg_assist_given_record,
                    "p_intrinsic_dark_assist_given_record": p_intrinsic_dark_assist_given_record,
                }
            )
            if sample.success:
                corr = compute_pauli_correlators_and_chsh(sample.qubit_state)
                corr_exx_vals.append(float(corr["corr_exx"]))
                corr_eyy_vals.append(float(corr["corr_eyy"]))
                corr_ezz_vals.append(float(corr["corr_ezz"]))
                chsh_vals.append(float(corr["chsh_s_max"]))

        write_click_records(raw_dir, {_point_key(point): click_records})
        write_declared_density_matrix(
            raw_dir,
            rho_raw=getattr(enum_main, "rho_declared_raw", None),
            rho_ff=getattr(enum_main, "rho_declared_ff", None),
            trace_raw=float(getattr(enum_main, "trace_declared_raw", 0.0)),
            trace_ff=float(getattr(enum_main, "trace_declared_ff", 0.0)),
            rho_raw_by_bell=getattr(enum_main, "rho_declared_raw_by_bell", None),
            rho_ff_by_bell=getattr(enum_main, "rho_declared_ff_by_bell", None),
            trace_raw_by_bell=getattr(enum_main, "trace_declared_raw_by_bell", None),
            trace_ff_by_bell=getattr(enum_main, "trace_declared_ff_by_bell", None),
        )

        p_success_abs = float(enum_main.p_success)
        event_rate_hz = float(p_success_abs * attempt_rate_hz_eff)
        return {
            "run_index": run_index,
            "scan_family": FAMILY_SIM_CORE,
            "scan_point": point,
            "window_ns": float(config.run.window_ns),
            "shots": int(shots_per_run),
            "success": int(sum(1 for sample in pipeline.samples if sample.success)),
            "attempt_rate_hz": float(attempt_rate_hz_eff),
            "event_rate_hz": event_rate_hz,
            "p_two_click_abs": float(np.clip(pipeline.p_records_total, 0.0, 1.0)),
            "p_arrive": float(enum_main.p_arrive),
            "p_arrive_11": float(enum_main.p_arrive_11),
            "p_arrive_same_arm": float(enum_main.p_arrive_same_arm),
            "p_arrive_20": float(enum_main.p_arrive_20),
            "p_arrive_02": float(enum_main.p_arrive_02),
            "p_success_abs": p_success_abs,
            "p_success_true_abs": float(enum_main.p_success_true),
            "p_success_false_abs": float(enum_main.p_success_false),
            "p_success_true_given_arrival": float(enum_main.p_success_given_arrival),
            "fidelity_all": float(enum_main.fidelity_declared),
            "fidelity_true": float(enum_main.fidelity_true),
            "fidelity_false": float(enum_main.fidelity_false),
            "false_fraction": (
                float(enum_main.p_success_false / enum_main.p_success)
                if enum_main.p_success > 0
                else 0.0
            ),
            "corr_exx": float(np.mean(corr_exx_vals)) if corr_exx_vals else 0.0,
            "corr_eyy": float(np.mean(corr_eyy_vals)) if corr_eyy_vals else 0.0,
            "corr_ezz": float(np.mean(corr_ezz_vals)) if corr_ezz_vals else 0.0,
            "chsh_s_max": float(np.mean(chsh_vals)) if chsh_vals else 0.0,
            "coinc": None,
        }
    finally:
        config.run.window_ns = base_state["window_ns"]
        config.fiber.length_km = base_state["length_km"]
        config.qfc.theta_H = base_state["qfc_theta_h"]
        config.qfc.theta_V = base_state["qfc_theta_v"]
        config.qfc.qfc_noise_sd_cps_per_mhz_A = base_state["qfc_noise_a"]
        config.qfc.qfc_noise_sd_cps_per_mhz_B = base_state["qfc_noise_b"]
        config.detector.eta_det = base_state["eta_det"]
        config.detector.eta_det_map = base_state["eta_det_map"]
        config.noise.bg_rate_mean_hz = base_state["bg_rate_mean_hz"]
        config.noise.bg_rate_mean_hz_map = base_state["bg_rate_mean_hz_map"]


def run_param_scan_task(
    task: dict,
    config: SimConfig,
    raw_dir: Path,
    plots_dir: Path,
    task_id: str,
    should_abort=None,
) -> dict:
    _ = plots_dir, task_id
    seed_raw = task.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None
    run_index = int(task.get("run_index", 0))
    shots_per_run = int(task.get("shots", config.run.shots_per_run))
    payload = task.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("SCHEMA_ERROR: PARAM_SCAN task 缺少 payload")
    point = payload.get("scan_point")
    if not isinstance(point, dict):
        raise ValueError("SCHEMA_ERROR: PARAM_SCAN 缺少 payload.scan_point")
    point = {str(k): float(v) for k, v in point.items()}
    family = str(payload.get("scan_family", "")).upper()
    if family not in {FAMILY_SIM_CORE, FAMILY_HOM_CORE}:
        family = FAMILY_HOM_CORE if "tau_ns" in point else FAMILY_SIM_CORE
    if family == FAMILY_HOM_CORE:
        return _run_param_scan_hom_point(
            task=task,
            config=config,
            raw_dir=raw_dir,
            point=point,
            shots_per_run=shots_per_run,
            run_index=run_index,
            seed=seed,
            should_abort=should_abort,
        )
    return _run_param_scan_sim_point(
        task=task,
        config=config,
        raw_dir=raw_dir,
        point=point,
        shots_per_run=shots_per_run,
        run_index=run_index,
        seed=seed,
        should_abort=should_abort,
    )
