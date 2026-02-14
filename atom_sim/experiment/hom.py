# -*- coding: utf-8 -*-
"""
HOM 实验仿真：统计符合率随延迟 tau 的变化。
"""

import time
from pathlib import Path
from typing import Optional, Iterator

import numpy as np

from .common import (
    HomConfig,
    SimConfig,
    run_trial_physics_core,
    run_detection_core_from_pipe,
    write_click_records,
)

DEFAULT_TAU_RANDOM_RANGE_NS = (-10.0, 10.0)


def _is_port_samepol_coincidence(clicks, window_bins: Optional[int]) -> bool:
    # 仅统计“同偏振跨端口符合”：H1-H2 或 V1-V2
    # 物理含义：HOM 在同偏振时发生干涉，跨端口同时点击反映可见度。
    h1_bins = [c.bin_index for c in clicks if c.detector == "H1"]
    h2_bins = [c.bin_index for c in clicks if c.detector == "H2"]
    v1_bins = [c.bin_index for c in clicks if c.detector == "V1"]
    v2_bins = [c.bin_index for c in clicks if c.detector == "V2"]
    if window_bins is None:
        # 不限定时间窗：只要同偏振跨端口各出现一次就算符合
        return (h1_bins and h2_bins) or (v1_bins and v2_bins)
    # 限定时间窗：只统计 |bin_i - bin_j| <= window_bins 的符合
    for b1 in h1_bins:
        for b2 in h2_bins:
            if abs(b1 - b2) <= window_bins:
                return True
    for b1 in v1_bins:
        for b2 in v2_bins:
            if abs(b1 - b2) <= window_bins:
                return True
    return False


def _build_hom_tau_values(hom_cfg: HomConfig) -> list:
    # 构造 τ 列表的优先级：
    #   1) random 模式：每次生成一个随机 τ
    #   2) 单点 τ：直接返回 [τ]
    #   3) 扫描：使用 (start, end, step) 或 (start, end, points)
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
    # 约束：起止必须递增
    if tau_end < tau_start:
        raise ValueError("tau_end 必须 >= tau_start")
    if tau_points is not None:
        # 以“点数”定义扫描：等间隔 linspace
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
    # 以“步长”定义扫描：包含终点（允许浮点误差）
    if tau_step <= 0:
        raise ValueError("tau_step 必须 > 0")
    values = list(np.arange(tau_start, tau_end + tau_step * 0.5, tau_step))
    return [float(v) for v in values]


def parse_hom_cli(args, parser) -> HomConfig:
    # 解析 HOM 参数并做一致性校验：
    #   - 只能在 (tau) 与 (tau_start/tau_end/tau_step|points) 之间二选一
    #   - 若完全未给 tau 参数，默认随机 tau 进行 smoke
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
    )

def validate_no_hom_args(args, parser) -> None:
    # 非 HOM 模式下禁止传入 HOM 相关参数，避免 silent misuse
    if (
        args.tau is not None
        or args.tau_start is not None
        or args.tau_end is not None
        or args.tau_step is not None
        or args.tau_points is not None
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
    # ------------------------------------------------------------------
    # 单个 τ 的 HOM 统计：
    #   - 复用 single_run 的发射/链路流程
    #   - 对每次 shot 抽样点击记录
    #   - 只统计“同偏振跨端口符合”(H1-H2 或 V1-V2)
    # ------------------------------------------------------------------
    # 固定随机种子：保证同一 τ 的重复性（便于比较）
    run_wall_start = time.perf_counter()
    run_rng = np.random.default_rng(rng_seed)
    timings = {} if debug else None
    pipe = run_trial_physics_core(
        rng=run_rng,
        config=config,
        delay_ns=tau_ns,
        delay_jitter_ns=delay_jitter_ns,
        verbose=verbose,
        debug=debug,
        hooks=None,
        emission_diagnostics=False,
    )
    if debug and pipe.timings:
        timings.update(pipe.timings)

    param_store, pipeline = run_detection_core_from_pipe(
        pipe=pipe,
        config=config,
        rng=run_rng,
        coincidence_window_ns=window_ns,
        shots_per_run=shots_per_run,
        compute_metrics=False,
        verbose=verbose,
        bs_theta=float(config.detector.bs_theta),
    )
    window_bins = param_store.window_bins
    coincidences = 0
    click_records = []
    detect_start = time.perf_counter() if debug else None
    # 抽样双点击记录（POVM）；BS 已并入测量端
    if debug and timings is not None and pipeline.timings:
        timings["povm_effects"] = pipeline.timings.get("povm_effects", 0.0)
        timings["povm_sampling"] = pipeline.timings.get("povm_sampling", 0.0)
        timings["detection_total"] = pipeline.timings.get("detection_total", 0.0)
    p_arrive = pipeline.p_arrive
    # 逐 shot 统计符合与点击记录
    for det_result in pipeline.samples:
        click_records.append(
            [
                (
                    c.detector,
                    c.bin_index,
                    bool(getattr(c, "is_dark", False)),
                    str(getattr(c, "source", "signal")),
                )
                for c in det_result.clicks
            ]
        )
        if _is_port_samepol_coincidence(det_result.clicks, window_bins):
            coincidences += 1

    if debug and timings is not None and detect_start is not None:
        if "detection_total" not in timings:
            timings["detection_total"] = time.perf_counter() - detect_start
        if shots_per_run > 0:
            timings["detection_per_shot"] = timings["detection_total"] / shots_per_run
        timings["run_wall_total"] = time.perf_counter() - run_wall_start
        timing_order = [
            ("emission", "发射"),
            ("qfc_filter_memory", "QFC+滤波记忆"),
            ("fiber", "光纤"),
            ("dephase", "退相干"),
            ("povm_effects", "POVM构建"),
            ("povm_sampling", "POVM抽样"),
            ("detection_total", "探测总计"),
        ]
        parts = []
        for key, label in timing_order:
            if key in timings:
                value = float(timings[key])
                parts.append(f"{label}={value:.2f}s")
        if parts:
            print(f"[HOM][调试耗时] tau={tau_ns:.3f} ns | " + " | ".join(parts))
        core_base_keys = ("emission", "qfc_filter_memory", "fiber", "dephase")
        core_sum = sum(float(timings[k]) for k in core_base_keys if k in timings)
        if "detection_total" in timings:
            core_sum += float(timings["detection_total"])
        else:
            core_sum += sum(float(timings[k]) for k in ("povm_effects", "povm_sampling") if k in timings)
        wall = float(timings.get("run_wall_total", 0.0))
        overhead = max(0.0, wall - core_sum)
        print(
            f"[HOM][调试总览] tau={tau_ns:.3f} ns | "
            f"核心阶段(去重)={core_sum:.2f}s | run墙钟={wall:.2f}s | 额外开销={overhead:.2f}s"
        )

    return coincidences, p_arrive, click_records


def iter_hom_core_tasks(config: SimConfig) -> Iterator[dict]:
    if config.hom is None:
        raise ValueError("HOM 任务需要 --mode HOM 并提供 tau 参数")
    tau_values = [float(v) for v in _build_hom_tau_values(config.hom)]
    for tau in tau_values:
        for run_index in range(config.run.runs):
            yield {
                "id": f"hom_tau_{tau:+.3f}_run_{run_index:06d}",
                "experiment": "HOM",
                "run_index": run_index,
                "payload": {
                    "tau_ns": float(tau),
                    "window_ns": float(config.hom.window_ns),
                },
            }


def run_hom_task(
    task: dict,
    config: SimConfig,
    raw_dir: Path,
    plots_dir: Path,
    task_id: str,
) -> dict:
    _ = plots_dir, task_id
    seed_raw = task.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None
    run_index = int(task.get("run_index", 0) or 0)
    shots = int(task.get("shots", config.run.shots_per_run))
    payload = task.get("payload", {})
    tau_ns = float(payload["tau_ns"])
    default_window = config.hom.window_ns if config.hom is not None else config.run.window_ns
    window_ns = float(payload.get("window_ns", default_window))
    coincid, p_arrive, click_records = _run_hom_run(
        tau_ns,
        shots,
        config,
        window_ns,
        delay_jitter_ns=0.0,
        verbose=False,
        debug=config.run.debug,
        rng_seed=seed,
    )
    metrics = {
        "run_index": run_index,
        "tau_ns": tau_ns,
        "window_ns": window_ns,
        "shots": shots,
        "p_arrive": p_arrive,
        "coinc": coincid,
    }
    write_click_records(raw_dir, click_records)
    return metrics
