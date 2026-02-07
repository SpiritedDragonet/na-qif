# -*- coding: utf-8 -*-
"""
WINDOW_SCAN 任务：按 coincidence window 扫描运行 SIM 核心。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import single_run
from .common import SimConfig

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
) -> tuple[dict, list[dict] | None]:
    seed = task.get("seed")
    seed = int(seed) if seed is not None else None
    run_index = int(task.get("run_index", 0))
    window_ns = float(task.get("window_ns", config.run.window_ns))

    run_config = deepcopy(config)
    run_config.run.window_ns = window_ns

    run_stats, success_metrics, click_records = single_run._run_single_simulation_core(
        output_dir=raw_dir,
        run_index=run_index,
        config=run_config,
        show_plots=False,
        plot_dir=plots_dir,
        run_tag=task_id,
        seed=seed,
    )

    metrics = {
        "shots": run_stats["shots"],
        "success": run_stats["success"],
        "window_ns": window_ns,
        "run_index": run_index,
    }
    if success_metrics:
        for key in WINDOW_SCAN_METRIC_KEYS:
            metrics[key] = success_metrics.get(key)
    return metrics, click_records
