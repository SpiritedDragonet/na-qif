# -*- coding: utf-8 -*-
"""
HOM 实验仿真：统计符合率随延迟 tau 的变化。
"""

import time
from typing import Optional

import numpy as np

from ..simulation import run_detection_pipeline
from ..physics.gates import bs_gate_6d
from .common import (
    HomConfig,
    SimConfig,
    _compute_window_bins,
    _compute_noise_params,
)
from .single_run import _run_single_trial

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
    # 非 HOM 模式下禁止传入 HOM 相关参数，避免 silent misuse
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
    # ------------------------------------------------------------------
    # 单个 τ 的 HOM 统计：
    #   - 复用 single_run 的发射/链路流程
    #   - 对每次 shot 抽样点击记录
    #   - 只统计“同偏振跨端口符合”(H1-H2 或 V1-V2)
    # ------------------------------------------------------------------
    # 固定随机种子：保证同一 τ 的重复性（便于比较）
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
        # 发射/光纤等阶段提前中止：视为无效 run
        return 0, True, 0.0, 0.0, []

    result = pipe.emission
    p_no_loss = pipe.p_no_loss

    bin_dt_s = result.dt_s
    bin_dt_ns = bin_dt_s * 1e9
    # 将时间窗映射到 bin 数（用于符合判定）
    window_bins = _compute_window_bins(window_ns, bin_dt_ns)

    if config.detector.ideal_det:
        # 理想探测：无暗计数、探测效率=1
        p_noise = 0.0
        eta_det = 1.0
    else:
        # 现实探测：每个 run 采样一次噪声率（背景 + 本底暗计数）
        noise = _compute_noise_params(config.noise, bin_dt_s, run_rng)
        p_noise = noise["p_noise"]
        eta_det = config.detector.eta_det

    # BS 并入测量端 (U^† E U)
    bs_unitary = bs_gate_6d()
    coincidences = 0
    click_records = []
    detect_start = time.perf_counter() if debug else None
    # 抽样双点击记录（POVM）；bs_unitary 将 BS 并入测量端
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
        bs_unitary=bs_unitary,
        fiber_sample=pipe.fiber_sample,
        apply_filter_780=pipe.apply_filter_780,
        theta_H=pipe.qfc_theta_H,
        theta_V=pipe.qfc_theta_V,
    )
    p_arrive = pipeline.p_arrive
    # 逐 shot 统计符合与点击记录
    for det_result in pipeline.samples:
        click_records.append(
            [(c.detector, c.bin_index, bool(getattr(c, "is_dark", False))) for c in det_result.clicks]
        )
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
