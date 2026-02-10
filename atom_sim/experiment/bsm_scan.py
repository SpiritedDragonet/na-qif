# -*- coding: utf-8 -*-
"""
BSM_SCAN 任务：按中心站 BS 混合角扫描运行 SIM 核心。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..simulation import compute_pauli_correlators_and_chsh
from .common import (
    SimConfig,
    run_trial_physics_core,
    run_detection_core_from_pipe,
    _build_run_parameter_store,
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
) -> tuple[dict, dict]:
    _ = raw_dir, plots_dir, task_id

    seed = task.get("seed")
    seed = int(seed) if seed is not None else None
    run_index = int(task.get("run_index", 0))
    shots_per_run = int(task.get("shots", config.run.shots_per_run))

    bs_thetas = [float(value) for value in task.get("bs_thetas", [])]
    if not bs_thetas:
        bs_thetas = [float(config.detector.bs_theta)]

    run_rng = np.random.default_rng(seed)
    pipe = run_trial_physics_core(
        rng=run_rng,
        config=config,
        delay_ns=None,
        delay_jitter_ns=None,
        verbose=False,
        debug=config.run.debug,
        hooks=None,
        emission_diagnostics=False,
    )
    emission = pipe.emission

    # 对同一个 run 的 BSM 角扫描复用同一发射态和同一噪声预算，
    # 使扫描结果主要反映 BS 混合角本身的影响。
    param_store = _build_run_parameter_store(
        config=config,
        emission_bin_dt_s=emission.dt_s,
        coincidence_window_ns=float(config.run.window_ns),
        rng=run_rng,
    )

    metrics_by_theta = []
    clicks_by_theta = {}

    for bs_theta in bs_thetas:
        _reuse_store, pipeline = run_detection_core_from_pipe(
            pipe=pipe,
            config=config,
            rng=run_rng,
            coincidence_window_ns=float(config.run.window_ns),
            shots_per_run=shots_per_run,
            compute_metrics=True,
            verbose=False,
            bs_theta=float(bs_theta),
            param_store=param_store,
        )

        enum_main = pipeline.metrics
        if enum_main is None:
            raise RuntimeError("BSM_SCAN 需要 compute_metrics=True 且返回有效枚举结果")

        corr_exx_vals = []
        corr_eyy_vals = []
        corr_ezz_vals = []
        chsh_vals = []
        pattern_counter = {pattern_key: 0 for pattern_key in BSM_PATTERN_KEYS}
        click_records = []

        for shot_index, sample in enumerate(pipeline.samples):
            pattern_key = _classify_bsm_pattern(sample)
            pattern_counter[pattern_key] += 1

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
        entry = {
            "bs_theta": float(bs_theta),
            "bs_split_ratio": float(np.sin(float(bs_theta)) ** 2),
            "run_index": run_index,
            "shots": shots_total,
            "success": success_total,
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

        metrics_by_theta.append(entry)
        clicks_by_theta[f"{float(bs_theta):.9f}"] = click_records

    return {
        "run_index": run_index,
        "bs_thetas": metrics_by_theta,
    }, clicks_by_theta

