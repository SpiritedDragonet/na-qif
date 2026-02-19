# -*- coding: utf-8 -*-
"""
DETECTOR_BG_SCAN 任务：按探测效率与背景噪声二维网格扫描运行 SIM 核心。
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterator

from ..simulation import compute_pauli_correlators_and_chsh
from .common import (
    SimConfig,
    run_trial_detection_core,
    _compute_effective_attempt_rate_hz,
    write_click_records,
    write_declared_density_matrix,
)


def validate_detector_bg_scan_config(config: SimConfig) -> None:
    required = (
        config.run.eta_det_sweep_start,
        config.run.eta_det_sweep_end,
        config.run.eta_det_sweep_step,
        config.run.bg_mean_sweep_start_hz,
        config.run.bg_mean_sweep_end_hz,
        config.run.bg_mean_sweep_step_hz,
    )
    if any(value is None for value in required):
        raise ValueError(
            "DETECTOR_BG_SCAN 需要 --eta-det-sweep-start/--eta-det-sweep-end/--eta-det-sweep-step "
            "和 --bg-mean-sweep-start-hz/--bg-mean-sweep-end-hz/--bg-mean-sweep-step-hz"
        )
    if config.run.eta_det_sweep_step <= 0.0:
        raise ValueError("eta_det_sweep_step 必须 > 0")
    if config.run.bg_mean_sweep_step_hz <= 0.0:
        raise ValueError("bg_mean_sweep_step_hz 必须 > 0")
    if config.run.eta_det_sweep_end < config.run.eta_det_sweep_start:
        raise ValueError("eta_det_sweep_end 必须 >= eta_det_sweep_start")
    if config.run.bg_mean_sweep_end_hz < config.run.bg_mean_sweep_start_hz:
        raise ValueError("bg_mean_sweep_end_hz 必须 >= bg_mean_sweep_start_hz")
    for field_name, value in (
        ("eta_det_sweep_start", config.run.eta_det_sweep_start),
        ("eta_det_sweep_end", config.run.eta_det_sweep_end),
    ):
        if not (0.0 < float(value) <= 1.0):
            raise ValueError(f"{field_name} 必须在 (0,1] 内")
    for field_name, value in (
        ("bg_mean_sweep_start_hz", config.run.bg_mean_sweep_start_hz),
        ("bg_mean_sweep_end_hz", config.run.bg_mean_sweep_end_hz),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{field_name} 必须 >= 0")


def build_detector_bg_scan_values(config: SimConfig) -> tuple[list[float], list[float]]:
    validate_detector_bg_scan_config(config)
    eta_values = []
    eta = float(config.run.eta_det_sweep_start)
    eta_end = float(config.run.eta_det_sweep_end)
    eta_step = float(config.run.eta_det_sweep_step)
    while eta <= eta_end + 1e-12:
        eta_values.append(round(eta, 9))
        eta += eta_step
    if not eta_values:
        eta_values = [float(config.run.eta_det_sweep_start)]

    bg_values = []
    bg = float(config.run.bg_mean_sweep_start_hz)
    bg_end = float(config.run.bg_mean_sweep_end_hz)
    bg_step = float(config.run.bg_mean_sweep_step_hz)
    while bg <= bg_end + 1e-12:
        bg_values.append(round(bg, 9))
        bg += bg_step
    if not bg_values:
        bg_values = [float(config.run.bg_mean_sweep_start_hz)]
    return eta_values, bg_values


def run_detector_bg_scan_task(
    task: dict,
    config: SimConfig,
    raw_dir: Path,
    plots_dir: Path,
    task_id: str,
) -> dict:
    _ = plots_dir, task_id

    seed = task.get("seed")
    seed = int(seed) if seed is not None else None
    run_index = int(task.get("run_index", 0))
    shots_per_run = int(task.get("shots", config.run.shots_per_run))

    payload = task.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("SCHEMA_ERROR: DETECTOR_BG_SCAN task 缺少 payload")
    if "eta_det" not in payload or "bg_rate_mean_hz" not in payload:
        raise ValueError("SCHEMA_ERROR: DETECTOR_BG_SCAN task 缺少 payload.eta_det / payload.bg_rate_mean_hz")

    eta_det = float(payload["eta_det"])
    bg_rate_mean_hz = float(payload["bg_rate_mean_hz"])
    if not (0.0 < eta_det <= 1.0):
        raise ValueError("DETECTOR_BG_SCAN 的 eta_det 必须在 (0,1] 内")
    if bg_rate_mean_hz < 0.0:
        raise ValueError("DETECTOR_BG_SCAN 的 bg_rate_mean_hz 必须 >= 0")

    run_rng = np.random.default_rng(seed)
    attempt_rate_hz_eff = _compute_effective_attempt_rate_hz(
        config.run.attempt_rate_hz,
        config.run.attempt_overhead_us,
    )

    base_eta = float(config.detector.eta_det)
    base_eta_map = dict(config.detector.eta_det_map)
    base_bg_mean = float(config.noise.bg_rate_mean_hz)
    base_bg_mean_map = dict(config.noise.bg_rate_mean_hz_map)

    try:
        # 统一二维网格口径：四路探测器使用同一 eta 与同一背景均值。
        config.detector.eta_det = float(eta_det)
        config.detector.eta_det_map = {}
        config.noise.bg_rate_mean_hz = float(bg_rate_mean_hz)
        config.noise.bg_rate_mean_hz_map = {}
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
        )
    finally:
        config.detector.eta_det = base_eta
        config.detector.eta_det_map = base_eta_map
        config.noise.bg_rate_mean_hz = base_bg_mean
        config.noise.bg_rate_mean_hz_map = base_bg_mean_map

    enum_main = pipeline.metrics
    if enum_main is None:
        raise RuntimeError("DETECTOR_BG_SCAN 需要 compute_metrics=True 且返回有效枚举结果")

    corr_exx_vals = []
    corr_eyy_vals = []
    corr_ezz_vals = []
    chsh_vals = []
    click_records = []
    for shot_index, sample in enumerate(pipeline.samples):
        p_true_given_record = float(np.clip(getattr(sample, "p_true_given_record", 0.0), 0.0, 1.0))
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
        "eta_det": float(eta_det),
        "bg_rate_mean_hz": float(bg_rate_mean_hz),
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
        "p_success_intrinsic_dark_assisted": float(enum_main.p_success_intrinsic_dark_assisted),
        "p_success_bg_assisted": float(enum_main.p_success_bg_assisted),
    }

    metrics = {
        "run_index": run_index,
        "detector_bg_points": [entry],
    }
    click_key = f"eta_{float(eta_det):.9f}_bg_{float(bg_rate_mean_hz):.9f}"
    write_click_records(raw_dir, {click_key: click_records})
    write_declared_density_matrix(
        raw_dir,
        rho_raw=getattr(enum_main, "rho_declared_raw", None),
        rho_ff=getattr(enum_main, "rho_declared_ff", None),
        trace_raw=float(getattr(enum_main, "trace_declared_raw", 0.0)),
        trace_ff=float(getattr(enum_main, "trace_declared_ff", 0.0)),
    )
    return metrics


def iter_detector_bg_scan_core_tasks(config: SimConfig) -> Iterator[dict]:
    eta_values, bg_values = build_detector_bg_scan_values(config)
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
