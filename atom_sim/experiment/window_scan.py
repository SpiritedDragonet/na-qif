# -*- coding: utf-8 -*-
"""
WINDOW_SCAN 任务：按 coincidence window 扫描运行 SIM 核心。
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterator

from ..simulation import (
    compute_pauli_correlators_and_chsh,
    compute_fidelity_with_bell,
)
from .common import (
    SimConfig,
    run_trial_detection_core,
    write_click_records,
    write_declared_density_matrix,
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
    from . import param_scan

    param_scan.validate_alias_scan_task(config, "WINDOW_SCAN")


def build_window_scan_values(config: SimConfig) -> list[float]:
    from . import param_scan

    _axis_keys, axis_values = param_scan.resolve_alias_axis_values(config, "WINDOW_SCAN")
    return list(axis_values["window_ns"])


def run_window_scan_task(
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

    # 严格模式：WINDOW_SCAN 的窗口定义必须来自配置。
    # 不再兼容旧任务里的 windows_ns，避免任务负载冗余与语义分叉。
    if "windows_ns" in task:
        raise ValueError("WINDOW_SCAN task 不再支持 windows_ns；请使用配置的 window_sweep_* 参数")
    _ = build_window_scan_values(config)

    run_rng = np.random.default_rng(seed)
    pipe, _param_store, pipeline = run_trial_detection_core(
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
        window_bins=None,
        should_abort=should_abort,
    )

    # WINDOW_SCAN 口径统一：
    # - run 端只产出“无窗口限制”的统一点击记录与基础统计；
    # - window 判定全部在 summary 阶段进行，避免 task 侧重复扫描窗口。
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
                "p_bg_assist_given_record": p_bg_assist_given_record,
                "p_intrinsic_dark_assist_given_record": p_intrinsic_dark_assist_given_record,
                "fidelity_declared": fidelity_declared,
                "corr_exx": corr_exx,
                "corr_eyy": corr_eyy,
                "corr_ezz": corr_ezz,
                "chsh_s_max": chsh_s_max,
            }
        )

    enum_main = pipeline.metrics
    if enum_main is None:
        raise RuntimeError("WINDOW_SCAN 需要 compute_metrics=True 且返回有效枚举结果")

    accepted_count = int(len(shot_records))
    success_count = int(sum(1 for shot in shot_records if shot["raw_success"]))
    success_true_count = float(sum(float(shot["p_true_given_record"]) for shot in shot_records if shot["raw_success"]))
    success_false_count = float(success_count) - success_true_count
    shots_total = float(max(shots_per_run, 1))
    p_two_click_abs = float(np.clip(pipeline.p_records_total, 0.0, 1.0))
    accepted_cond_given_two_click = float(accepted_count / shots_total)
    success_cond_given_two_click = float(success_count / shots_total)
    success_true_cond_given_two_click = float(success_true_count / shots_total)
    success_false_cond_given_two_click = float(success_false_count / shots_total)
    success_records = [shot for shot in shot_records if shot["raw_success"]]
    fidelity_all_vals = [float(shot["fidelity_declared"]) for shot in success_records]
    corr_exx_vals = [float(shot["corr_exx"]) for shot in success_records]
    corr_eyy_vals = [float(shot["corr_eyy"]) for shot in success_records]
    corr_ezz_vals = [float(shot["corr_ezz"]) for shot in success_records]
    chsh_vals = [float(shot["chsh_s_max"]) for shot in success_records]
    fidelity_true_num = float(
        sum(float(shot["fidelity_declared"]) * float(shot["p_true_given_record"]) for shot in success_records)
    )
    fidelity_false_num = float(
        sum(float(shot["fidelity_declared"]) * (1.0 - float(shot["p_true_given_record"])) for shot in success_records)
    )

    entry = {
        "run_index": run_index,
        "shots": shots_per_run,
        "accepted": accepted_count,
        "success": success_count,
        "p_two_click_abs": p_two_click_abs,
        "accepted_cond_given_two_click": accepted_cond_given_two_click,
        "success_cond_given_two_click": success_cond_given_two_click,
        "success_true_cond_given_two_click": success_true_cond_given_two_click,
        "success_false_cond_given_two_click": success_false_cond_given_two_click,
        "window_ns": float(config.run.window_ns),
        "window_bins": None,
        "p_arrive": float(enum_main.p_arrive),
        "p_arrive_11": float(enum_main.p_arrive_11),
        "p_arrive_same_arm": float(enum_main.p_arrive_same_arm),
        "p_arrive_20": float(enum_main.p_arrive_20),
        "p_arrive_02": float(enum_main.p_arrive_02),
        "p_success_abs": float(enum_main.p_success),
        "p_success_true_abs": float(enum_main.p_success_true),
        "p_success_false_abs": float(enum_main.p_success_false),
        "p_success_true_given_arrival": float(enum_main.p_success_given_arrival),
        "fidelity_all": float(np.mean(fidelity_all_vals)) if fidelity_all_vals else 0.0,
        "fidelity_true": (float(fidelity_true_num / success_true_count) if success_true_count > 0.0 else 0.0),
        "fidelity_false": (float(fidelity_false_num / success_false_count) if success_false_count > 0.0 else 0.0),
        "p_success_intrinsic_dark_assisted": float(enum_main.p_success_intrinsic_dark_assisted),
        "p_success_bg_assisted": float(enum_main.p_success_bg_assisted),
        "false_fraction": (float(enum_main.p_success_false / enum_main.p_success) if enum_main.p_success > 0 else 0.0),
        "corr_exx": float(np.mean(corr_exx_vals)) if corr_exx_vals else 0.0,
        "corr_eyy": float(np.mean(corr_eyy_vals)) if corr_eyy_vals else 0.0,
        "corr_ezz": float(np.mean(corr_ezz_vals)) if corr_ezz_vals else 0.0,
        "chsh_s_max": float(np.mean(chsh_vals)) if chsh_vals else 0.0,
    }

    click_records = [
        {
            "shot_index": int(shot["shot_index"]),
            "success": bool(shot["raw_success"]),
            "bell": shot["raw_bell"],
            "clicks": list(shot["clicks"]),
            "p_true_given_record": float(shot["p_true_given_record"]),
            "p_bg_assist_given_record": float(shot["p_bg_assist_given_record"]),
            "p_intrinsic_dark_assist_given_record": float(
                shot["p_intrinsic_dark_assist_given_record"]
            ),
            "fidelity_declared": float(shot["fidelity_declared"]),
            "corr_exx": float(shot["corr_exx"]),
            "corr_eyy": float(shot["corr_eyy"]),
            "corr_ezz": float(shot["corr_ezz"]),
            "chsh_s_max": float(shot["chsh_s_max"]),
        }
        for shot in shot_records
    ]

    metrics = {
        "run_index": run_index,
        "window_scan": entry,
    }
    write_click_records(raw_dir, click_records)
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


def iter_window_scan_core_tasks(config: SimConfig) -> Iterator[dict]:
    from . import param_scan

    yield from param_scan.iter_window_scan_alias_core_tasks(config)
