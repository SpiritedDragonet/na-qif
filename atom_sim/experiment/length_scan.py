# -*- coding: utf-8 -*-
"""
LENGTH_SCAN 任务：按光纤长度扫描运行 SIM 核心。
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterator
from .common import (
    SimConfig,
    run_trial_detection_core,
    _compute_effective_attempt_rate_hz,
    _compute_t_wait_us_from_length,
    write_click_records,
    write_declared_density_matrix,
)


def validate_length_scan_config(config: SimConfig) -> None:
    from . import param_scan

    param_scan.validate_alias_scan_task(config, "LENGTH_SCAN")


def build_length_scan_values(config: SimConfig) -> list[float]:
    from . import param_scan

    _axis_keys, axis_values = param_scan.resolve_alias_axis_values(config, "LENGTH_SCAN")
    return list(axis_values["length_km"])


def run_length_scan_task(
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
    if not isinstance(payload, dict) or "length_km" not in payload:
        raise ValueError("SCHEMA_ERROR: LENGTH_SCAN task 缺少 payload.length_km")
    length_km = float(payload["length_km"])

    run_rng = np.random.default_rng(seed)
    t_wait_us = _compute_t_wait_us_from_length(
        length_km=length_km,
        fiber_group_velocity_mps=config.run.fiber_group_velocity_mps,
        t_wait_overhead_us=config.run.t_wait_overhead_us,
        t_wait_length_scale=config.run.t_wait_length_scale,
    )
    attempt_rate_hz_eff = _compute_effective_attempt_rate_hz(
        config.run.attempt_rate_hz,
        config.run.attempt_overhead_us,
        wait_time_us=t_wait_us,
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
            should_abort=should_abort,
        )
    finally:
        config.fiber.length_km = base_length_km

    enum_main = pipeline.metrics
    if enum_main is None:
        raise RuntimeError("LENGTH_SCAN 需要 compute_metrics=True 且返回有效枚举结果")

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
        # 使用枚举态口径（非单次抽样口径），避免 shots_per_run=1 时
        # "无成功抽样 -> CHSH=0" 对 run 级统计产生系统性悲观偏置。
        "corr_exx": float(enum_main.corr_exx),
        "corr_eyy": float(enum_main.corr_eyy),
        "corr_ezz": float(enum_main.corr_ezz),
        "chsh_s_max": float(enum_main.chsh_s_max),
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
        rho_raw_by_bell=getattr(enum_main, "rho_declared_raw_by_bell", None),
        rho_ff_by_bell=getattr(enum_main, "rho_declared_ff_by_bell", None),
        trace_raw_by_bell=getattr(enum_main, "trace_declared_raw_by_bell", None),
        trace_ff_by_bell=getattr(enum_main, "trace_declared_ff_by_bell", None),
    )
    return metrics


def iter_length_scan_core_tasks(config: SimConfig) -> Iterator[dict]:
    from . import param_scan

    yield from param_scan.iter_length_scan_alias_core_tasks(config)
