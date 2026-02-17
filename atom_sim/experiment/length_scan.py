# -*- coding: utf-8 -*-
"""
LENGTH_SCAN 任务：按光纤长度扫描运行 SIM 核心。
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


def validate_length_scan_config(config: SimConfig) -> None:
    if (
        config.run.length_sweep_start_km is None
        or config.run.length_sweep_end_km is None
        or config.run.length_sweep_step_km is None
    ):
        raise ValueError("LENGTH_SCAN 需要 --length-sweep-start-km/--length-sweep-end-km/--length-sweep-step-km")
    if config.run.length_sweep_step_km <= 0.0:
        raise ValueError("length_sweep_step_km 必须 > 0")
    if config.run.length_sweep_end_km < config.run.length_sweep_start_km:
        raise ValueError("length_sweep_end_km 必须 >= length_sweep_start_km")
    if config.run.attempt_rate_hz <= 0.0:
        raise ValueError("attempt_rate_hz 必须 > 0")
    if config.run.attempt_overhead_us < 0.0:
        raise ValueError("attempt_overhead_us 必须 >= 0")


def build_length_scan_values(config: SimConfig) -> list[float]:
    validate_length_scan_config(config)
    start = float(config.run.length_sweep_start_km)
    end = float(config.run.length_sweep_end_km)
    step = float(config.run.length_sweep_step_km)
    values = []
    value = start
    while value <= end + 1e-12:
        values.append(round(value, 9))
        value += step
    if not values:
        values = [start]
    return values


def run_length_scan_task(
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
    if not isinstance(payload, dict) or "length_km" not in payload:
        raise ValueError("SCHEMA_ERROR: LENGTH_SCAN task 缺少 payload.length_km")
    length_km = float(payload["length_km"])

    run_rng = np.random.default_rng(seed)
    attempt_rate_hz_eff = _compute_effective_attempt_rate_hz(
        config.run.attempt_rate_hz,
        config.run.attempt_overhead_us,
    )

    base_length_km = float(config.fiber.length_km)

    try:
        config.fiber.length_km = float(length_km)
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
        config.fiber.length_km = base_length_km

    enum_main = pipeline.metrics
    if enum_main is None:
        raise RuntimeError("LENGTH_SCAN 需要 compute_metrics=True 且返回有效枚举结果")

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
        "length_km": float(length_km),
        "run_index": run_index,
        "shots": shots_per_run,
        "success": int(sum(1 for sample in pipeline.samples if sample.success)),
        "p_two_click_abs": float(np.clip(pipeline.p_records_total, 0.0, 1.0)),
        "window_ns": float(config.run.window_ns),
        "attempt_rate_hz": attempt_rate_hz_eff,
        "event_rate_hz": event_rate_hz,
        "p_arrive": enum_main.p_arrive,
        "p_arrive_11": enum_main.p_arrive_11,
        "p_arrive_same_arm": enum_main.p_arrive_same_arm,
        "p_arrive_20": enum_main.p_arrive_20,
        "p_arrive_02": enum_main.p_arrive_02,
        "p_success_abs": enum_main.p_success,
        "p_success_true_abs": enum_main.p_success_true,
        "p_success_false_abs": enum_main.p_success_false,
        "p_success_true_given_arrival": enum_main.p_success_given_arrival,
        "fidelity_all": enum_main.fidelity_declared,
        "fidelity_true": enum_main.fidelity_true,
        "fidelity_false": enum_main.fidelity_false,
        "false_fraction": (
            enum_main.p_success_false / enum_main.p_success
            if enum_main.p_success > 0
            else 0.0
        ),
        "corr_exx": float(np.mean(corr_exx_vals)) if corr_exx_vals else 0.0,
        "corr_eyy": float(np.mean(corr_eyy_vals)) if corr_eyy_vals else 0.0,
        "corr_ezz": float(np.mean(corr_ezz_vals)) if corr_ezz_vals else 0.0,
        "chsh_s_max": float(np.mean(chsh_vals)) if chsh_vals else 0.0,
    }

    metrics = {
        "run_index": run_index,
        "lengths": [entry],
    }
    write_click_records(raw_dir, {f"{float(length_km):.9f}": click_records})
    write_declared_density_matrix(
        raw_dir,
        rho_raw=getattr(enum_main, "rho_declared_raw", None),
        rho_ff=getattr(enum_main, "rho_declared_ff", None),
        trace_raw=float(getattr(enum_main, "trace_declared_raw", 0.0)),
        trace_ff=float(getattr(enum_main, "trace_declared_ff", 0.0)),
    )
    return metrics


def iter_length_scan_core_tasks(config: SimConfig) -> Iterator[dict]:
    length_values = build_length_scan_values(config)
    for length_idx, length_km in enumerate(length_values):
        for run_index in range(config.run.runs):
            yield {
                "id": f"lscan_len_{length_idx:04d}_run_{run_index:06d}",
                "experiment": "LENGTH_SCAN",
                "run_index": run_index,
                "payload": {"length_km": float(length_km)},
            }
