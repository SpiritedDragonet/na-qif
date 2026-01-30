# -*- coding: utf-8 -*-
"""
HOM 实验仿真：统计符合率随延迟 tau 的变化。
"""

import csv
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from ..simulation import run_detection_pipeline
from .common import (
    HomConfig,
    SimConfig,
    _compute_window_bins,
    _compute_noise_params,
    run_task_queue,
)
from .single_run import _run_single_trial

DEFAULT_TAU_RANDOM_RANGE_NS = (-10.0, 10.0)


def _is_port_samepol_coincidence(clicks, window_bins: Optional[int]) -> bool:
    # 仅统计同偏振跨端口符合：H1-H2 或 V1-V2
    h1_bins = [c.bin_index for c in clicks if c.detector == "H1"]
    h2_bins = [c.bin_index for c in clicks if c.detector == "H2"]
    v1_bins = [c.bin_index for c in clicks if c.detector == "V1"]
    v2_bins = [c.bin_index for c in clicks if c.detector == "V2"]
    if window_bins is None:
        return (h1_bins and h2_bins) or (v1_bins and v2_bins)
    for b1 in h1_bins:
        for b2 in h2_bins:
            if abs(b1 - b2) <= window_bins:
                return True
    for b1 in v1_bins:
        for b2 in v2_bins:
            if abs(b1 - b2) <= window_bins:
                return True
    return False


def _progress_every(total: int) -> int:
    if total <= 0:
        return 1
    return max(1, min(5, total // 10))


def _build_hom_tau_values(hom_cfg: HomConfig) -> list:
    if hom_cfg.tau_random:
        tau_min, tau_max = hom_cfg.tau_random_range
        rng = np.random.default_rng()
        return [float(rng.uniform(tau_min, tau_max))]
    tau = hom_cfg.tau
    if tau is not None:
        return [float(tau)]
    tau_start = float(hom_cfg.tau_start)
    tau_end = float(hom_cfg.tau_end)
    tau_step = hom_cfg.tau_step
    tau_points = hom_cfg.tau_points
    if tau_end < tau_start:
        raise ValueError("tau_end 必须 >= tau_start")
    if tau_points is not None:
        tau_points = int(tau_points)
        if tau_points < 1:
            raise ValueError("tau_points 必须 >= 1")
        if tau_points == 1:
            if tau_start != tau_end:
                raise ValueError("tau_points=1 时要求 tau_start == tau_end")
            return [float(tau_start)]
        values = list(np.linspace(tau_start, tau_end, tau_points))
        return [float(v) for v in values]
    tau_step = float(tau_step)
    if tau_step <= 0:
        raise ValueError("tau_step 必须 > 0")
    values = list(np.arange(tau_start, tau_end + tau_step * 0.5, tau_step))
    return [float(v) for v in values]


def parse_hom_cli(args, parser) -> HomConfig:
    tau = args.tau
    tau_start = args.tau_start
    tau_end = args.tau_end
    tau_step = args.tau_step
    tau_points = args.tau_points
    if (
        tau is None
        and tau_start is None
        and tau_end is None
        and tau_step is None
        and tau_points is None
    ):
        window_ns = args.window_ns if args.window_ns is not None else 70.0
        return HomConfig(
            tau=None,
            tau_start=None,
            tau_end=None,
            tau_step=None,
            tau_points=None,
            tau_random=True,
            tau_random_range=DEFAULT_TAU_RANDOM_RANGE_NS,
            window_ns=window_ns,
            max_attempts=args.max_attempts,
        )
    if tau is not None:
        if (
            tau_start is not None
            or tau_end is not None
            or tau_step is not None
            or tau_points is not None
        ):
            parser.error("HOM 使用 --tau 时不能再提供 tau-start/tau-end/tau-step/tau-points")
    else:
        if tau_start is None or tau_end is None:
            parser.error("HOM 需要 tau-start 与 tau-end")
        if tau_step is None and tau_points is None:
            parser.error("HOM 需要 tau-step 或 tau-points")
        if tau_step is not None and tau_points is not None:
            parser.error("HOM 的 tau-step 与 tau-points 只能选其一")

    window_ns = args.window_ns if args.window_ns is not None else 70.0
    return HomConfig(
        tau=tau,
        tau_start=tau_start,
        tau_end=tau_end,
        tau_step=tau_step,
        tau_points=tau_points,
        tau_random=False,
        tau_random_range=DEFAULT_TAU_RANDOM_RANGE_NS,
        window_ns=window_ns,
        max_attempts=args.max_attempts,
    )

def validate_no_hom_args(args, parser) -> None:
    if (
        args.tau is not None
        or args.tau_start is not None
        or args.tau_end is not None
        or args.tau_step is not None
        or args.tau_points is not None
        or args.window_ns is not None
        or args.max_attempts is not None
    ):
        parser.error("非 HOM 模式不接受 HOM 参数")


def _run_hom_run(
    tau_ns: float,
    shots_per_run: int,
    config: SimConfig,
    window_ns: float,
    delay_jitter_ns: float = 0.0,
    verbose: bool = False,
    debug: bool = False,
    rng_seed: Optional[int] = None,
) -> tuple:
    run_rng = np.random.default_rng(rng_seed)
    timings = {} if debug else None
    pipe = _run_single_trial(
        rng=run_rng,
        config=config,
        delay_ns=tau_ns,
        delay_jitter_ns=delay_jitter_ns,
        verbose=verbose,
        debug=debug,
        hooks=None,
    )
    if debug and pipe.timings:
        timings.update(pipe.timings)
    if pipe.aborted:
        return 0, True, 0.0, 0.0, []

    result = pipe.emission
    p_no_loss = pipe.p_no_loss

    bin_dt_s = result.dt_s
    bin_dt_ns = bin_dt_s * 1e9
    window_bins = _compute_window_bins(window_ns, bin_dt_ns)

    if config.detector.ideal_det:
        p_noise = 0.0
        eta_det = 1.0
    else:
        noise = _compute_noise_params(config.noise, bin_dt_s, run_rng)
        p_noise = noise["p_noise"]
        eta_det = config.detector.eta_det

    coincidences = 0
    click_records = []
    detect_start = time.perf_counter() if debug else None
    pipeline = run_detection_pipeline(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        eta_det=eta_det,
        p_dark=p_noise,
        window_bins=window_bins,
        rng=run_rng,
        verbose=verbose,
        n_samples=shots_per_run,
        compute_metrics=False,
    )
    p_arrive = pipeline.p_arrive
    for det_result in pipeline.samples:
        click_records.append([(c.detector, c.bin_index) for c in det_result.clicks])
        if _is_port_samepol_coincidence(det_result.clicks, window_bins):
            coincidences += 1

    if debug and timings is not None and detect_start is not None:
        timings["detection_total"] = time.perf_counter() - detect_start
        if shots_per_run > 0:
            timings["detection_per_shot"] = timings["detection_total"] / shots_per_run
        timing_order = [
            ("emission", "发射"),
            ("qfc", "QFC"),
            ("filter_780", "780滤波"),
            ("project_1517", "1517投影"),
            ("fiber", "光纤"),
            ("dephase", "退相干"),
            ("bs", "BS"),
            ("detection_total", "探测抽样"),
        ]
        parts = []
        for key, label in timing_order:
            if key in timings:
                parts.append(f"{label}={timings[key]:.2f}s")
        if parts:
            print(f"[HOM][调试耗时] tau={tau_ns:.3f} ns | " + " | ".join(parts))

    return coincidences, False, p_arrive, p_no_loss, click_records


def _run_hom_task_indexed(
    tau_idx: int,
    tau_ns: float,
    shots_per_run: int,
    config: SimConfig,
    window_ns: float,
    delay_jitter_ns: float = 0.0,
    verbose: bool = False,
    debug: bool = False,
    rng_seed: Optional[int] = None,
) -> tuple:
    return _run_hom_run(
        tau_ns,
        shots_per_run,
        config,
        window_ns,
        delay_jitter_ns=delay_jitter_ns,
        verbose=verbose,
        debug=debug,
        rng_seed=rng_seed,
    )


def run_hom_experiment(
    output_dir: Path,
    config: SimConfig,
) -> None:
    if config.hom is None:
        raise ValueError("HOM 模式需要提供 hom 配置")
    hom_cfg = config.hom
    run_cfg = config.run
    n_runs = run_cfg.runs
    shots_per_run = run_cfg.shots_per_run
    core_budget = run_cfg.cores
    debug = run_cfg.debug

    tau_values = _build_hom_tau_values(hom_cfg)
    window_ns = hom_cfg.window_ns
    max_attempts = hom_cfg.max_attempts

    core_budget = max(1, min(core_budget, os.cpu_count() or 1))
    run_cfg.cores = core_budget

    tau_desc = ""
    if hom_cfg.tau is not None:
        tau_desc = f"tau={hom_cfg.tau:.3f} ns"
    elif hom_cfg.tau_random:
        tau_desc = (
            f"tau=random in [{hom_cfg.tau_random_range[0]:.1f}, "
            f"{hom_cfg.tau_random_range[1]:.1f}] ns"
        )
    elif hom_cfg.tau_points is not None:
        tau_desc = (
            f"tau_start={hom_cfg.tau_start:.3f} ns, "
            f"tau_end={hom_cfg.tau_end:.3f} ns, "
            f"tau_points={hom_cfg.tau_points}"
        )
    else:
        tau_desc = (
            f"tau_start={hom_cfg.tau_start:.3f} ns, "
            f"tau_end={hom_cfg.tau_end:.3f} ns, "
            f"tau_step={hom_cfg.tau_step:.3f} ns"
        )
    print(
        f"[HOM] {tau_desc} | window_ns={window_ns:.1f} | "
        f"runs={n_runs} | shots_per_run={shots_per_run} | cores={core_budget}"
    )
    print(f"[HOM] 光纤噪声: {'开启' if config.fiber.noise_enabled else '关闭'}")
    print(
        f"[HOM] 探测参数: eta_det={config.detector.eta_det:.3f} | "
        f"理想探测={'是' if config.detector.ideal_det else '否'}"
    )
    if max_attempts is not None:
        if max_attempts < n_runs:
            max_attempts = n_runs
        limit = max_attempts
    else:
        limit = n_runs * 20
    print(f"[HOM] valid-only=ON | max_attempts={limit}")

    summary_path = output_dir / "hom_summary.csv"
    with open(summary_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "tau_ns",
            "trials_target",
            "trials_total",
            "valid_runs",
            "arrive_trials",
            "p_no_loss_avg",
            "early_abort_runs",
            "coinc_counts",
            "coinc_rate",
            "p_arrive_avg",
            "window_ns",
            "runs_target",
            "runs_attempted",
        ])

    if not tau_values:
        return

    total_taus = len(tau_values)
    progress_every = _progress_every(limit)
    trials_target = n_runs * shots_per_run
    delay_jitter_ns = 0.5 if hom_cfg.tau_random else 0.0

    states = []
    for _ in tau_values:
        states.append({
            "attempted": 0,
            "completed": 0,
            "valid": 0,
            "early_abort": 0,
            "coincidences": 0,
            "p_arrive_sum": 0.0,
            "p_no_loss_sum": 0.0,
            "arrive_trials": 0.0,
            "progress_next": progress_every,
        })

    focus_idx = 0

    def _needs_more(idx: int) -> bool:
        s = states[idx]
        return s["valid"] < n_runs and s["attempted"] < limit

    def _apply_result(idx: int, result: tuple) -> None:
        s = states[idx]
        s["completed"] += 1
        coincid_run, early_abort, p_arrive, p_no_loss, _clicks = result
        if early_abort:
            s["early_abort"] += 1
        else:
            if s["valid"] < n_runs:
                s["valid"] += 1
                s["coincidences"] += coincid_run
                s["p_arrive_sum"] += p_arrive
                s["p_no_loss_sum"] += p_no_loss
                s["arrive_trials"] += p_arrive * shots_per_run
        if idx == focus_idx and s["completed"] >= s["progress_next"]:
            print(
                f"[HOM] tau={tau_values[idx]:.3f} ns 进度: "
                f"completed {s['completed']}/{limit}, "
                f"valid {s['valid']}/{n_runs}, "
                f"early_abort {s['early_abort']}"
            )
            s["progress_next"] += progress_every

    jobs = max(1, min(core_budget, limit))
    print(f"[HOM] 并行: 核数预算={core_budget} | 实际并发进程={jobs}")
    tau_queue = deque(range(total_taus))

    def _pick_tau() -> Optional[int]:
        for _ in range(total_taus):
            idx = tau_queue[0]
            tau_queue.rotate(-1)
            if _needs_more(idx):
                return idx
        return None

    def _next_task() -> Optional[tuple]:
        idx = _pick_tau()
        if idx is None:
            return None
        states[idx]["attempted"] += 1
        return (
            idx,
            tau_values[idx],
            shots_per_run,
            config,
            window_ns,
            delay_jitter_ns,
            False,
            debug,
        )

    def _on_result(task: tuple, result: tuple) -> None:
        idx = int(task[0])
        _apply_result(idx, result)

    focus_task = None
    if _needs_more(focus_idx):
        print(f"[HOM] 详细日志: tau={tau_values[focus_idx]:.3f} ns, attempt=1")
        states[focus_idx]["attempted"] += 1
        focus_task = (
            focus_idx,
            tau_values[focus_idx],
            shots_per_run,
            config,
            window_ns,
            delay_jitter_ns,
            True,
            debug,
        )

    run_task_queue(
        jobs=jobs,
        task_fn=_run_hom_task_indexed,
        next_task=_next_task,
        on_result=_on_result,
        focus_task=focus_task,
    )

    for idx, tau_ns in enumerate(tau_values):
        s = states[idx]
        if s["valid"] < n_runs:
            print(
                f"[HOM] tau={tau_ns:.3f} ns 未达到目标有效运行数: "
                f"{s['valid']}/{n_runs} (max_attempts={limit})"
            )
        trials_total = s["attempted"] * shots_per_run
        arrive_trials = s["arrive_trials"]
        coinc_rate = (s["coincidences"] / arrive_trials) if arrive_trials > 0 else 0.0
        p_arrive_avg = (s["p_arrive_sum"] / s["valid"]) if s["valid"] > 0 else 0.0
        p_no_loss_avg = (s["p_no_loss_sum"] / s["valid"]) if s["valid"] > 0 else 0.0
        with open(summary_path, 'a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                f"{tau_ns:.6f}",
                trials_target,
                trials_total,
                s["valid"],
                f"{arrive_trials:.6f}",
                f"{p_no_loss_avg:.8f}",
                s["early_abort"],
                s["coincidences"],
                f"{coinc_rate:.8f}",
                f"{p_arrive_avg:.6f}",
                f"{window_ns:.3f}",
                n_runs,
                s["attempted"],
            ])
        print(
            f"[HOM] {idx + 1:02d}/{total_taus:02d} "
            f"tau={tau_ns:.3f} ns | "
            f"coinc={s['coincidences']}/{arrive_trials:.1f} "
            f"rate={coinc_rate:.4f} | early_abort={s['early_abort']}"
        )
