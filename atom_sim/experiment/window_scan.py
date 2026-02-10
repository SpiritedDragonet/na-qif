# -*- coding: utf-8 -*-
"""
WINDOW_SCAN 任务：按 coincidence window 扫描运行 SIM 核心。
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ..physics.gates import bs_gate_6d
from ..simulation import (
    run_detection_pipeline,
    compute_pauli_correlators_and_chsh,
    compute_fidelity_with_bell,
)
from .single_run import _run_single_trial
from .common import (
    SimConfig,
    _build_run_parameter_store,
    _build_detection_kwargs,
    _compute_window_bins,
)

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
        # 新语义：窗口列表来自运行配置，而不是逐 task 重复携带。
        windows_ns = build_window_scan_values(config)

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

    # WINDOW_SCAN 新路径：
    # 1) 每个 run 仅做一次探测采样（window_bins=None，保留全部双点击记录）；
    # 2) 各窗口结果通过“对同一批 shot 点击记录做窗口判定”得到。
    max_window_ns = max(windows_ns) if windows_ns else float(config.run.window_ns)
    param_store = _build_run_parameter_store(
        config=config,
        emission_bin_dt_s=emission.dt_s,
        coincidence_window_ns=max_window_ns,
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
    detect_common["window_bins"] = None
    pipeline = run_detection_pipeline(
        **detect_common,
        p_dark_intrinsic=param_store.p_dark_intrinsic_bin_map,
        p_bg_source=param_store.p_bg_bin_map,
        n_samples=shots_per_run,
        compute_metrics=False,
    )

    def _is_click_record_within_window(clicks: list, window_bins: int) -> bool:
        bins = [int(click[1]) for click in clicks if len(click) >= 2]
        if len(bins) < 2:
            return False
        for i in range(len(bins)):
            for j in range(i + 1, len(bins)):
                if abs(bins[i] - bins[j]) <= window_bins:
                    return True
        return False

    # 先把每个 shot 的原始信息缓存下来，避免在窗口循环里重复算保真度/关联量。
    shot_records = []
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
        p_true_given_record = float(np.clip(getattr(sample, "p_true_given_record", 0.0), 0.0, 1.0))
        fidelity_declared = 0.0
        corr_exx = 0.0
        corr_eyy = 0.0
        corr_ezz = 0.0
        chsh_s_max = 0.0
        if sample.success and sample.bell_state:
            tr_rho = float(np.trace(sample.qubit_state).real)
            if tr_rho > 1e-15:
                fidelity_declared = float(
                    compute_fidelity_with_bell(sample.qubit_state, sample.bell_state) / tr_rho
                )
            corr = compute_pauli_correlators_and_chsh(sample.qubit_state)
            corr_exx = float(corr["corr_exx"])
            corr_eyy = float(corr["corr_eyy"])
            corr_ezz = float(corr["corr_ezz"])
            chsh_s_max = float(corr["chsh_s_max"])
        shot_records.append(
            {
                "shot_index": shot_index,
                "raw_success": bool(sample.success),
                "raw_bell": sample.bell_state,
                "clicks": click_pairs,
                "p_true_given_record": p_true_given_record,
                "fidelity_declared": fidelity_declared,
                "corr_exx": corr_exx,
                "corr_eyy": corr_eyy,
                "corr_ezz": corr_ezz,
                "chsh_s_max": chsh_s_max,
            }
        )

    windows_metrics = []
    clicks_by_window = {}
    for window_ns in windows_ns:
        window_bins = _compute_window_bins(
            float(window_ns),
            float(emission.dt_s) * 1e9,
            detection_gate_ns=param_store.noise_budget.detection_gate_ns,
        )

        corr_exx_vals = []
        corr_eyy_vals = []
        corr_ezz_vals = []
        chsh_vals = []
        fidelity_all_vals = []
        fidelity_true_num = 0.0
        fidelity_false_num = 0.0
        success_count = 0
        success_true_count = 0.0
        success_false_count = 0.0
        accepted_count = 0
        click_records = []
        for shot in shot_records:
            accepted = _is_click_record_within_window(shot["clicks"], window_bins)
            if accepted:
                accepted_count += 1
            shot_success = bool(shot["raw_success"] and accepted)
            shot_bell = shot["raw_bell"] if shot_success else ""
            p_true_given_record = float(shot["p_true_given_record"])
            click_records.append(
                {
                    "shot_index": int(shot["shot_index"]),
                    "success": shot_success,
                    "bell": shot_bell,
                    "clicks": list(shot["clicks"]),
                    "accepted_by_window": accepted,
                    "p_true_given_record": p_true_given_record,
                }
            )
            if not shot_success:
                continue
            success_count += 1
            success_true_count += p_true_given_record
            success_false_count += 1.0 - p_true_given_record
            fidelity_declared = float(shot["fidelity_declared"])
            fidelity_all_vals.append(fidelity_declared)
            fidelity_true_num += fidelity_declared * p_true_given_record
            fidelity_false_num += fidelity_declared * (1.0 - p_true_given_record)
            corr_exx_vals.append(float(shot["corr_exx"]))
            corr_eyy_vals.append(float(shot["corr_eyy"]))
            corr_ezz_vals.append(float(shot["corr_ezz"]))
            chsh_vals.append(float(shot["chsh_s_max"]))

        shots_total = float(max(shots_per_run, 1))
        p_success_abs = float(success_count / shots_total)
        p_success_true_abs = float(success_true_count / shots_total)
        p_success_false_abs = float(success_false_count / shots_total)

        entry = {
            "window_ns": float(window_ns),
            "window_bins": int(window_bins),
            "run_index": run_index,
            "shots": shots_per_run,
            "accepted": int(accepted_count),
            "success": int(success_count),
            "p_arrive": float(pipeline.p_arrive),
            "p_arrive_11": None,
            "p_arrive_same_arm": None,
            "p_arrive_20": None,
            "p_arrive_02": None,
            "p_success_abs": p_success_abs,
            "p_success_true_abs": p_success_true_abs,
            "p_success_false_abs": p_success_false_abs,
            "p_success_true_given_arrival": None,
            "fidelity_all": float(np.mean(fidelity_all_vals)) if fidelity_all_vals else 0.0,
            "fidelity_true": (
                float(fidelity_true_num / success_true_count)
                if success_true_count > 0.0
                else 0.0
            ),
            "fidelity_false": (
                float(fidelity_false_num / success_false_count)
                if success_false_count > 0.0
                else 0.0
            ),
            "p_success_intrinsic_dark_assisted": None,
            "p_success_bg_assisted": None,
            "false_fraction": (
                p_success_false_abs / p_success_abs
                if p_success_abs > 0.0
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
