# -*- coding: utf-8 -*-
"""
QFC_EFF_NOISE_SCAN 任务：按 QFC 转换效率与噪声谱密度二维网格扫描运行 SIM 核心。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np

from ..simulation import compute_pauli_correlators_and_chsh
from .common import (
    SimConfig,
    run_trial_detection_core,
    _compute_effective_attempt_rate_hz,
    write_click_records,
    write_declared_density_matrix,
)


def _qfc_eta_to_theta(qfc_eta: float) -> float:
    eta = float(np.clip(qfc_eta, 0.0, 1.0))
    return float(np.arcsin(np.sqrt(eta)))


def validate_qfc_eff_noise_scan_config(config: SimConfig) -> None:
    from . import param_scan

    param_scan.validate_alias_scan_task(config, "QFC_EFF_NOISE_SCAN")


def build_qfc_eff_noise_scan_values(config: SimConfig) -> tuple[list[float], list[float]]:
    from . import param_scan

    _axis_keys, axis_values = param_scan.resolve_alias_axis_values(config, "QFC_EFF_NOISE_SCAN")
    return list(axis_values["qfc_eta"]), list(axis_values["qfc_noise_sd_cps_per_mhz"])


def run_qfc_eff_noise_scan_task(
    task: dict,
    config: SimConfig,
    raw_dir: Path,
    plots_dir: Path,
    task_id: str,
    should_abort=None,
) -> dict:
    _ = plots_dir, task_id

    seed = task.get("seed")
    seed = int(seed) if seed is not None else None
    run_index = int(task.get("run_index", 0))
    shots_per_run = int(task.get("shots", config.run.shots_per_run))

    payload = task.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("SCHEMA_ERROR: QFC_EFF_NOISE_SCAN task 缺少 payload")
    if "qfc_eta" not in payload or "qfc_noise_sd_cps_per_mhz" not in payload:
        raise ValueError(
            "SCHEMA_ERROR: QFC_EFF_NOISE_SCAN 缺少 payload.qfc_eta / payload.qfc_noise_sd_cps_per_mhz"
        )

    qfc_eta = float(payload["qfc_eta"])
    qfc_noise_sd_cps_per_mhz = float(payload["qfc_noise_sd_cps_per_mhz"])
    if not (0.0 <= qfc_eta <= 1.0):
        raise ValueError("QFC_EFF_NOISE_SCAN 的 qfc_eta 必须在 [0,1] 内")
    if qfc_noise_sd_cps_per_mhz < 0.0:
        raise ValueError("QFC_EFF_NOISE_SCAN 的 qfc_noise_sd_cps_per_mhz 必须 >= 0")

    run_rng = np.random.default_rng(seed)
    attempt_rate_hz_eff = _compute_effective_attempt_rate_hz(
        config.run.attempt_rate_hz,
        config.run.attempt_overhead_us,
    )

    base_theta_h = float(config.qfc.theta_H)
    base_theta_v = float(config.qfc.theta_V)
    base_noise_a = float(config.qfc.qfc_noise_sd_cps_per_mhz_A)
    base_noise_b = float(config.qfc.qfc_noise_sd_cps_per_mhz_B)
    qfc_theta = _qfc_eta_to_theta(qfc_eta)

    wall_start = time.perf_counter()
    try:
        config.qfc.theta_H = qfc_theta
        config.qfc.theta_V = qfc_theta
        config.qfc.qfc_noise_sd_cps_per_mhz_A = float(qfc_noise_sd_cps_per_mhz)
        config.qfc.qfc_noise_sd_cps_per_mhz_B = float(qfc_noise_sd_cps_per_mhz)
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
            should_abort=should_abort,
        )
    finally:
        config.qfc.theta_H = base_theta_h
        config.qfc.theta_V = base_theta_v
        config.qfc.qfc_noise_sd_cps_per_mhz_A = base_noise_a
        config.qfc.qfc_noise_sd_cps_per_mhz_B = base_noise_b
    runtime_wall_s = float(time.perf_counter() - wall_start)

    enum_main = pipeline.metrics
    if enum_main is None:
        raise RuntimeError("QFC_EFF_NOISE_SCAN 需要 compute_metrics=True 且返回有效枚举结果")

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

    p_success_abs = float(enum_main.p_success)
    event_rate_hz = float(p_success_abs * attempt_rate_hz_eff)
    entry = {
        "qfc_eta": float(qfc_eta),
        "qfc_theta": float(qfc_theta),
        "qfc_noise_sd_cps_per_mhz": float(qfc_noise_sd_cps_per_mhz),
        "run_index": run_index,
        "shots": shots_per_run,
        "success": int(sum(1 for sample in pipeline.samples if sample.success)),
        "p_two_click_abs": float(np.clip(pipeline.p_records_total, 0.0, 1.0)),
        "window_ns": float(config.run.window_ns),
        "attempt_rate_hz": float(attempt_rate_hz_eff),
        "event_rate_hz": event_rate_hz,
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
        "p_success_intrinsic_dark_assisted": float(
            enum_main.p_success_intrinsic_dark_assisted
        ),
        "p_success_bg_assisted": float(enum_main.p_success_bg_assisted),
        "runtime_wall_s": runtime_wall_s,
    }

    metrics = {
        "run_index": run_index,
        "qfc_eff_noise_points": [entry],
    }
    click_key = (
        f"eta_{float(qfc_eta):.9f}_noise_{float(qfc_noise_sd_cps_per_mhz):.9f}"
    )
    write_click_records(raw_dir, {click_key: click_records})
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
    return metrics


def iter_qfc_eff_noise_scan_core_tasks(config: SimConfig) -> Iterator[dict]:
    from . import param_scan

    yield from param_scan.iter_qfc_eff_noise_scan_alias_core_tasks(config)
