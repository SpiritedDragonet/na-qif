# -*- coding: utf-8 -*-
"""
WINDOW_SCAN 任务：按 coincidence window 扫描运行 SIM 核心。
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ..physics.gates import bs_gate_6d
from ..simulation import run_detection_pipeline, compute_pauli_correlators_and_chsh
from .single_run import _run_single_trial
from .common import SimConfig, _build_run_parameter_store, _build_detection_kwargs

WINDOW_SCAN_METRIC_KEYS = (
    "p_arrive",
    "p_arrive_11",
    "p_arrive_same_arm",
    "p_arrive_20",
    "p_arrive_02",
    "p_success_abs",
    "p_success_true_abs",
    "p_success_false_abs",
    "p_success_true_given_arrival",
    "fidelity_all",
    "fidelity_true",
    "fidelity_false",
    "false_fraction",
    "corr_exx",
    "corr_eyy",
    "corr_ezz",
    "chsh_s_max",
)


def validate_window_scan_config(config: SimConfig) -> None:
    if (
        config.run.window_sweep_start_ns is None
        or config.run.window_sweep_end_ns is None
        or config.run.window_sweep_step_ns is None
    ):
        raise ValueError("WINDOW_SCAN 需要 --window-sweep-start-ns/--window-sweep-end-ns/--window-sweep-step-ns")
    if config.run.window_sweep_step_ns <= 0.0:
        raise ValueError("window_sweep_step_ns 必须 > 0")
    if config.run.window_sweep_end_ns < config.run.window_sweep_start_ns:
        raise ValueError("window_sweep_end_ns 必须 >= window_sweep_start_ns")


def build_window_scan_values(config: SimConfig) -> list[float]:
    validate_window_scan_config(config)
    start = float(config.run.window_sweep_start_ns)
    end = float(config.run.window_sweep_end_ns)
    step = float(config.run.window_sweep_step_ns)
    values = []
    value = start
    while value <= end + 1e-12:
        values.append(round(value, 9))
        value += step
    if not values:
        values = [start]
    return values


def run_window_scan_task(
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

    windows_ns = [float(v) for v in task.get("windows_ns", [])]
    if not windows_ns:
        # 防御式兜底：若任务未写入窗口列表，则退化为单窗口。
        windows_ns = [float(task.get("window_ns", config.run.window_ns))]

    run_rng = np.random.default_rng(seed)
    pipe = _run_single_trial(
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
    bs_unitary = bs_gate_6d(config.detector.bs_theta)

    windows_metrics = []
    clicks_by_window = {}
    for window_ns in windows_ns:
        param_store = _build_run_parameter_store(
            config=config,
            emission_bin_dt_s=emission.dt_s,
            coincidence_window_ns=float(window_ns),
            rng=run_rng,
        )
        detect_common = _build_detection_kwargs(
            pipe=pipe,
            param_store=param_store,
            rng=run_rng,
            verbose=False,
            bs_unitary=bs_unitary,
            bs_theta=config.detector.bs_theta,
        )
        pipeline = run_detection_pipeline(
            **detect_common,
            p_dark_intrinsic=param_store.p_dark_intrinsic_bin_map,
            p_bg_source=param_store.p_bg_bin_map,
            n_samples=shots_per_run,
            compute_metrics=True,
        )

        enum_main = pipeline.metrics
        if enum_main is None:
            raise RuntimeError("WINDOW_SCAN 需要 compute_metrics=True 且返回有效枚举结果")

        corr_exx_vals = []
        corr_eyy_vals = []
        corr_ezz_vals = []
        chsh_vals = []
        click_records = []
        for shot_index, sample in enumerate(pipeline.samples):
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
                }
            )
            if sample.success:
                corr = compute_pauli_correlators_and_chsh(sample.qubit_state)
                corr_exx_vals.append(float(corr["corr_exx"]))
                corr_eyy_vals.append(float(corr["corr_eyy"]))
                corr_ezz_vals.append(float(corr["corr_ezz"]))
                chsh_vals.append(float(corr["chsh_s_max"]))

        entry = {
            "window_ns": float(window_ns),
            "run_index": run_index,
            "shots": shots_per_run,
            "success": int(sum(1 for sample in pipeline.samples if sample.success)),
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

        windows_metrics.append(entry)
        clicks_by_window[f"{float(window_ns):.9f}"] = click_records

    return {
        "run_index": run_index,
        "windows": windows_metrics,
    }, clicks_by_window
