# -*- coding: utf-8 -*-
"""
BSM_SCAN 任务：按中心站 BS 混合角扫描运行 SIM 核心。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..simulation import compute_pauli_correlators_and_chsh
from .common import (
    SimConfig,
    run_trial_detection_core,
    write_click_records,
    write_declared_density_matrix,
)


BSM_PATTERN_KEYS = (
    "pattern_h1v2",
    "pattern_v1h2",
    "pattern_h1v1",
    "pattern_h2v2",
    "pattern_h1h2",
    "pattern_v1v2",
    "pattern_other",
)


_BSM_PATTERN_MAP = {
    frozenset(("H1", "V2")): "pattern_h1v2",
    frozenset(("V1", "H2")): "pattern_v1h2",
    frozenset(("H1", "V1")): "pattern_h1v1",
    frozenset(("H2", "V2")): "pattern_h2v2",
    frozenset(("H1", "H2")): "pattern_h1h2",
    frozenset(("V1", "V2")): "pattern_v1v2",
}


def _classify_bsm_pattern(sample) -> str:
    detectors = [click.detector for click in sample.clicks]
    if len(detectors) != 2:
        return "pattern_other"
    if detectors[0] == detectors[1]:
        return "pattern_other"
    return _BSM_PATTERN_MAP.get(frozenset(detectors), "pattern_other")


def validate_bsm_scan_config(config: SimConfig) -> None:
    if (
        config.run.bs_sweep_start_theta is None
        or config.run.bs_sweep_end_theta is None
        or config.run.bs_sweep_step_theta is None
    ):
        raise ValueError(
            "BSM_SCAN 需要 --bs-sweep-start-theta/--bs-sweep-end-theta/--bs-sweep-step-theta"
        )
    if config.run.bs_sweep_step_theta <= 0.0:
        raise ValueError("bs_sweep_step_theta 必须 > 0")
    if config.run.bs_sweep_end_theta < config.run.bs_sweep_start_theta:
        raise ValueError("bs_sweep_end_theta 必须 >= bs_sweep_start_theta")
    if config.run.bs_sweep_start_theta < 0.0 or config.run.bs_sweep_end_theta > float(np.pi / 2.0):
        raise ValueError("BSM 扫描 theta 必须在 [0, pi/2] 内")


def build_bsm_scan_values(config: SimConfig) -> list[float]:
    validate_bsm_scan_config(config)
    start = float(config.run.bs_sweep_start_theta)
    end = float(config.run.bs_sweep_end_theta)
    step = float(config.run.bs_sweep_step_theta)
    values = []
    value = start
    while value <= end + 1e-12:
        values.append(round(value, 9))
        value += step
    if not values:
        values = [start]
    return values


def run_bsm_scan_task(
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
    if not isinstance(payload, dict) or "bs_theta" not in payload:
        raise ValueError("SCHEMA_ERROR: BSM_SCAN task 缺少 payload.bs_theta")
    bs_theta = float(payload["bs_theta"])

    run_rng = np.random.default_rng(seed)
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
        bs_theta=bs_theta,
        hooks=None,
        emission_diagnostics=False,
        should_abort=should_abort,
    )

    enum_main = pipeline.metrics
    if enum_main is None:
        raise RuntimeError("BSM_SCAN 需要 compute_metrics=True 且返回有效枚举结果")

    corr_exx_vals = []
    corr_eyy_vals = []
    corr_ezz_vals = []
    chsh_vals = []
    pattern_counter = {pattern_key: 0 for pattern_key in BSM_PATTERN_KEYS}
    pattern_true_mass = {pattern_key: 0.0 for pattern_key in BSM_PATTERN_KEYS}
    pattern_false_mass = {pattern_key: 0.0 for pattern_key in BSM_PATTERN_KEYS}
    click_records = []

    for shot_index, sample in enumerate(pipeline.samples):
        pattern_key = _classify_bsm_pattern(sample)
        pattern_counter[pattern_key] += 1
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
        pattern_true_mass[pattern_key] += p_true_given_record
        pattern_false_mass[pattern_key] += (1.0 - p_true_given_record)

        click_pairs = [
            (
                click.detector,
                click.bin_index,
                bool(getattr(click, "is_dark", False)),
                str(getattr(click, "source", "signal")),
            )
            for click in sample.clicks
        ]
        click_records.append(
            {
                "shot_index": shot_index,
                "success": bool(sample.success),
                "bell": sample.bell_state,
                "pattern": pattern_key,
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

    shots_total = len(pipeline.samples)
    success_total = int(sum(1 for sample in pipeline.samples if sample.success))
    p_two_click_abs = float(np.clip(pipeline.p_records_total, 0.0, 1.0))
    entry = {
        "bs_theta": float(bs_theta),
        "bs_split_ratio": float(np.sin(float(bs_theta)) ** 2),
        "run_index": run_index,
        "shots": shots_total,
        "success": success_total,
        "p_two_click_abs": p_two_click_abs,
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
            enum_main.p_success_false / enum_main.p_success if enum_main.p_success > 0 else 0.0
        ),
        "corr_exx": float(np.mean(corr_exx_vals)) if corr_exx_vals else 0.0,
        "corr_eyy": float(np.mean(corr_eyy_vals)) if corr_eyy_vals else 0.0,
        "corr_ezz": float(np.mean(corr_ezz_vals)) if corr_ezz_vals else 0.0,
        "chsh_s_max": float(np.mean(chsh_vals)) if chsh_vals else 0.0,
    }
    for pattern_key in BSM_PATTERN_KEYS:
        count = int(pattern_counter[pattern_key])
        entry[pattern_key] = count
        entry[f"{pattern_key}_rate"] = (float(count) / float(shots_total)) if shots_total > 0 else 0.0
        if shots_total > 0:
            entry[f"{pattern_key}_true_abs"] = p_two_click_abs * (pattern_true_mass[pattern_key] / float(shots_total))
            entry[f"{pattern_key}_false_abs"] = p_two_click_abs * (pattern_false_mass[pattern_key] / float(shots_total))
        else:
            entry[f"{pattern_key}_true_abs"] = 0.0
            entry[f"{pattern_key}_false_abs"] = 0.0

    metrics = {
        "run_index": run_index,
        "bs_thetas": [entry],
    }
    write_click_records(raw_dir, {f"{float(bs_theta):.9f}": click_records})
    write_declared_density_matrix(
        raw_dir,
        rho_raw=getattr(enum_main, "rho_declared_raw", None),
        rho_ff=getattr(enum_main, "rho_declared_ff", None),
        trace_raw=float(getattr(enum_main, "trace_declared_raw", 0.0)),
        trace_ff=float(getattr(enum_main, "trace_declared_ff", 0.0)),
    )
    return metrics


def iter_bsm_scan_core_tasks(config: SimConfig) -> Iterator[dict]:
    bs_values = build_bsm_scan_values(config)
    for bs_idx, bs_theta in enumerate(bs_values):
        for run_index in range(config.run.runs):
            yield {
                "id": f"bscan_theta_{bs_idx:04d}_run_{run_index:06d}",
                "experiment": "BSM_SCAN",
                "run_index": run_index,
                "payload": {"bs_theta": float(bs_theta)},
            }

