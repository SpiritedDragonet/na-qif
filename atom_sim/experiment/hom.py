# -*- coding: utf-8 -*-
"""
HOM 实验仿真：统计符合率随延迟 tau 的变化。
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from ..simulation import (
    run_dual_atom_emission,
    apply_qfc,
    apply_780_filter,
    apply_fiber_channel,
    apply_bs,
    project_to_1517,
    run_two_photon_detection,
    compute_two_photon_arrival_prob,
)
from .common import (
    ATOM_EXTREME_EPS,
    _get_emission_params,
    _make_fiber_params,
    _compute_window_bins,
    _compute_noise_params,
    _apply_atomic_dephasing,
    _atom_extreme_state,
)


def _is_port_coincidence(clicks, window_bins: Optional[int]) -> bool:
    port1_bins = [c.bin_index for c in clicks if c.detector in ("H1", "V1")]
    port2_bins = [c.bin_index for c in clicks if c.detector in ("H2", "V2")]
    if not port1_bins or not port2_bins:
        return False
    if window_bins is None:
        return True
    for b1 in port1_bins:
        for b2 in port2_bins:
            if abs(b1 - b2) <= window_bins:
                return True
    return False


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


def _build_hom_tau_values(hom_cfg: dict) -> list:
    tau = hom_cfg.get("tau")
    if tau is not None:
        return [float(tau)]
    tau_start = float(hom_cfg["tau_start"])
    tau_end = float(hom_cfg["tau_end"])
    tau_step = hom_cfg.get("tau_step")
    tau_points = hom_cfg.get("tau_points")
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


def _normalize_hom_criterion(raw_value: str, parser) -> str:
    criterion_raw = (raw_value or "port").lower()
    criterion_map = {
        "port": "port",
        "any": "port",
        "bsm": "bsm",
        "port_same": "port_same",
        "port-same": "port_same",
        "same": "port_same",
        "samepol": "port_same",
        "port_samepol": "port_same",
        "port-samepol": "port_same",
    }
    if criterion_raw not in criterion_map:
        parser.error("criterion 仅支持 port / port_same / bsm")
    return criterion_map[criterion_raw]


def parse_hom_cli(args, positional, parser, consume) -> dict:
    if args.tau is None:
        consume("tau_start", float)
        consume("tau_end", float)
        if args.tau_step is not None and args.tau_points is not None:
            parser.error("HOM 的 tau-step 与 tau-points 只能选其一")
        if args.tau_step is None and args.tau_points is None:
            consume("tau_step", float)
    consume("window_ns", float)
    consume("criterion", str)

    if positional:
        parser.error(f"未识别的参数: {' '.join(positional)}")

    tau = args.tau
    tau_start = args.tau_start
    tau_end = args.tau_end
    tau_step = args.tau_step
    tau_points = args.tau_points
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
    criterion = _normalize_hom_criterion(args.criterion, parser)
    return {
        "tau": tau,
        "tau_start": tau_start,
        "tau_end": tau_end,
        "tau_step": tau_step,
        "tau_points": tau_points,
        "window_ns": window_ns,
        "criterion": criterion,
        "valid_only": bool(args.valid_only),
        "max_attempts": args.max_attempts,
    }


def validate_no_hom_args(args, positional, parser) -> None:
    if positional:
        parser.error(f"未识别的参数: {' '.join(positional)}")
    if (
        args.tau is not None
        or args.tau_start is not None
        or args.tau_end is not None
        or args.tau_step is not None
        or args.tau_points is not None
        or args.window_ns is not None
        or args.criterion is not None
        or args.valid_only
        or args.max_attempts is not None
    ):
        parser.error("非 HOM 模式不接受 HOM 参数")


def _run_hom_run(
    tau_ns: float,
    shots_per_run: int,
    noise_cfg: Optional[dict],
    window_ns: float,
    criterion: str,
    verbose: bool = False,
) -> tuple:
    run_rng = np.random.default_rng()
    emission_cfg = _get_emission_params(delay_ns=tau_ns)
    result = run_dual_atom_emission(
        n_bins=emission_cfg["n_bins"],
        dt_ns=emission_cfg["dt_ns"],
        chi_max=emission_cfg["chi_max"],
        gamma_peak_A=emission_cfg["gamma_peak_A"],
        gamma_peak_B=emission_cfg["gamma_peak_B"],
        sigma=emission_cfg["sigma"],
        delay_ns=emission_cfg["delay_ns"],
        delay_jitter_ns=emission_cfg["delay_jitter_ns"],
        rng=run_rng,
        verbose=verbose,
    )

    apply_qfc(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        theta_H=np.pi/4,
        theta_V=np.pi/4,
        verbose=verbose,
    )
    apply_780_filter(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=verbose,
        rng=run_rng,
    )
    project_to_1517(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=verbose,
    )
    result.mps, _ = apply_fiber_channel(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        fiber_params=_make_fiber_params(),
        rng=run_rng,
        verbose=verbose,
    )

    t_wait_us = 80.0
    t2_us = 1000.0
    if t2_us > 0.0:
        p_dephase = 0.5 * (1.0 - np.exp(-t_wait_us / t2_us))
    else:
        p_dephase = 0.0
    _apply_atomic_dephasing(result.mps, p_dephase, rng=run_rng, verbose=verbose)

    apply_bs(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=verbose,
    )

    extreme, _ = _atom_extreme_state(result.mps, eps=ATOM_EXTREME_EPS)
    if extreme:
        return 0, True, 0.0

    # 有效样本：两光子都到达探测器（不含探测效率与暗计数）
    if verbose:
        print("计算两光子到达概率...")
    p_arrive = compute_two_photon_arrival_prob(
        result.mps,
        result.get_n_bins(),
        verbose=verbose,
    )

    bin_dt_s = result.time_grid.dt
    bin_dt_ns = bin_dt_s * 1e9
    window_bins = _compute_window_bins(window_ns, bin_dt_ns)

    noise = _compute_noise_params(noise_cfg, bin_dt_s, run_rng)
    p_noise = noise["p_noise"]

    coincidences = 0
    for _ in range(shots_per_run):
        det_result = run_two_photon_detection(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=0.85,
            window_bins=window_bins,
            p_dark=p_noise,
            rng=run_rng,
            verbose=verbose,
        )
        if criterion == "bsm":
            if det_result.success:
                coincidences += 1
        elif criterion == "port_same":
            if _is_port_samepol_coincidence(det_result.clicks, window_bins):
                coincidences += 1
        else:
            if _is_port_coincidence(det_result.clicks, window_bins):
                coincidences += 1

    return coincidences, False, p_arrive


def _run_hom_run_task(args):
    tau_ns, shots_per_run, noise_cfg, window_ns, criterion = args
    coincidences, early_abort, p_arrive = _run_hom_run(
        tau_ns,
        shots_per_run,
        noise_cfg,
        window_ns,
        criterion,
    )
    return coincidences, early_abort, p_arrive


def run_hom_experiment(
    output_dir: Path,
    n_runs: int,
    shots_per_run: int,
    jobs: int,
    hom_cfg: dict,
    noise_cfg: Optional[dict],
) -> None:
    tau_values = _build_hom_tau_values(hom_cfg)
    window_ns = hom_cfg["window_ns"]
    criterion = hom_cfg["criterion"]
    valid_only = hom_cfg["valid_only"]
    max_attempts = hom_cfg["max_attempts"]

    jobs = max(1, min(jobs, n_runs, os.cpu_count() or 1))
    if valid_only and jobs > 1:
        print("HOM valid-only 模式下强制使用单进程，避免统计偏差。")
        jobs = 1

    tau_desc = ""
    if hom_cfg.get("tau") is not None:
        tau_desc = f"tau={hom_cfg['tau']:.3f} ns"
    elif hom_cfg.get("tau_points") is not None:
        tau_desc = (
            f"tau_start={hom_cfg['tau_start']:.3f} ns, "
            f"tau_end={hom_cfg['tau_end']:.3f} ns, "
            f"tau_points={hom_cfg['tau_points']}"
        )
    else:
        tau_desc = (
            f"tau_start={hom_cfg['tau_start']:.3f} ns, "
            f"tau_end={hom_cfg['tau_end']:.3f} ns, "
            f"tau_step={hom_cfg['tau_step']:.3f} ns"
        )
    print(
        f"[HOM] {tau_desc} | window_ns={window_ns:.1f} | "
        f"criterion={criterion} | runs={n_runs} | shots_per_run={shots_per_run} | jobs={jobs}"
    )
    if valid_only:
        limit = max_attempts if max_attempts is not None else n_runs * 3
        print(f"[HOM] valid-only=ON | max_attempts={limit}")

    def _progress_every(total: int) -> int:
        if total <= 0:
            return 1
        return max(1, min(50, total // 10))

    summary_path = output_dir / "hom_summary.csv"
    with open(summary_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "tau_ns",
            "trials_target",
            "trials_total",
            "valid_runs",
            "arrive_trials",
            "early_abort_runs",
            "coinc_counts",
            "coinc_rate",
            "p_arrive_avg",
            "window_ns",
            "criterion",
            "runs_target",
            "runs_attempted",
        ])

    trials_target = n_runs * shots_per_run
    total_taus = len(tau_values)
    for tau_idx, tau_ns in enumerate(tau_values, start=1):
        coincidences = 0
        trials_total = 0
        valid_runs = 0
        arrive_trials = 0.0
        p_arrive_sum = 0.0
        early_abort_runs = 0
        runs_attempted = 0
        runs_done = 0
        focus_used = False
        progress_every = _progress_every(n_runs if not valid_only else (max_attempts or n_runs * 3))

        if valid_only:
            limit = max_attempts if max_attempts is not None else n_runs * 3
            while valid_runs < n_runs and runs_attempted < limit:
                runs_attempted += 1
                verbose = not focus_used
                if verbose:
                    print(f"[HOM] 详细日志: tau={tau_ns:.3f} ns, attempt={runs_attempted}")
                coincid_run, early_abort, p_arrive = _run_hom_run(
                    tau_ns, shots_per_run, noise_cfg, window_ns, criterion, verbose=verbose
                )
                focus_used = True
                trials_total += shots_per_run
                if early_abort:
                    early_abort_runs += 1
                    continue
                valid_runs += 1
                arrive_trials += p_arrive * shots_per_run
                p_arrive_sum += p_arrive
                coincidences += coincid_run
                if runs_attempted % progress_every == 0 or runs_attempted == limit:
                    print(
                        f"[HOM] tau={tau_ns:.3f} ns 进度: "
                        f"attempted {runs_attempted}/{limit}, "
                        f"valid {valid_runs}/{n_runs}, "
                        f"early_abort {early_abort_runs}"
                    )
            if valid_runs < n_runs:
                print(
                    f"[HOM] tau={tau_ns:.3f} ns 未达到目标有效运行数: "
                    f"{valid_runs}/{n_runs} (max_attempts={limit})"
                )
        else:
            runs_attempted = n_runs
            if jobs <= 1:
                for _ in range(n_runs):
                    verbose = not focus_used
                    if verbose:
                        print(f"[HOM] 详细日志: tau={tau_ns:.3f} ns, run={runs_done + 1}")
                    coincid_run, early_abort, p_arrive = _run_hom_run(
                        tau_ns, shots_per_run, noise_cfg, window_ns, criterion, verbose=verbose
                    )
                    focus_used = True
                    trials_total += shots_per_run
                    runs_done += 1
                    if early_abort:
                        early_abort_runs += 1
                        continue
                    valid_runs += 1
                    arrive_trials += p_arrive * shots_per_run
                    p_arrive_sum += p_arrive
                    coincidences += coincid_run
                    if runs_done % progress_every == 0 or runs_done == n_runs:
                        print(
                            f"[HOM] tau={tau_ns:.3f} ns 进度: "
                            f"{runs_done}/{n_runs}, valid {valid_runs}, "
                            f"early_abort {early_abort_runs}"
                        )
            else:
                def _accumulate_run(coincid_run, early_abort, p_arrive):
                    nonlocal trials_total, runs_done, early_abort_runs
                    nonlocal valid_runs, arrive_trials, p_arrive_sum, coincidences
                    trials_total += shots_per_run
                    runs_done += 1
                    if early_abort:
                        early_abort_runs += 1
                        return
                    valid_runs += 1
                    arrive_trials += p_arrive * shots_per_run
                    p_arrive_sum += p_arrive
                    coincidences += coincid_run

                remaining = n_runs - 1
                max_workers = max(1, jobs - 1)
                futures = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    if remaining > 0:
                        tasks = [
                            (tau_ns, shots_per_run, noise_cfg, window_ns, criterion)
                            for _ in range(remaining)
                        ]
                        for task in tasks:
                            futures.append(executor.submit(_run_hom_run_task, task))

                    print(f"[HOM] 详细日志: tau={tau_ns:.3f} ns, run=1")
                    coincid_run, early_abort, p_arrive = _run_hom_run(
                        tau_ns, shots_per_run, noise_cfg, window_ns, criterion, verbose=True
                    )
                    focus_used = True
                    _accumulate_run(coincid_run, early_abort, p_arrive)
                    if runs_done % progress_every == 0 or runs_done == n_runs:
                        print(
                            f"[HOM] tau={tau_ns:.3f} ns 进度: "
                            f"{runs_done}/{n_runs}, valid {valid_runs}, "
                            f"early_abort {early_abort_runs}"
                        )

                    for future in as_completed(futures):
                        coincid_run, early_abort, p_arrive = future.result()
                        _accumulate_run(coincid_run, early_abort, p_arrive)
                        if runs_done % progress_every == 0 or runs_done == n_runs:
                            print(
                                f"[HOM] tau={tau_ns:.3f} ns 进度: "
                                f"{runs_done}/{n_runs}, valid {valid_runs}, "
                                f"early_abort {early_abort_runs}"
                            )

        coinc_rate = (coincidences / arrive_trials) if arrive_trials > 0 else 0.0
        p_arrive_avg = (p_arrive_sum / valid_runs) if valid_runs > 0 else 0.0
        with open(summary_path, 'a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                f"{tau_ns:.6f}",
                trials_target,
                trials_total,
                valid_runs,
                f"{arrive_trials:.6f}",
                early_abort_runs,
                coincidences,
                f"{coinc_rate:.8f}",
                f"{p_arrive_avg:.6f}",
                f"{window_ns:.3f}",
                criterion,
                n_runs,
                runs_attempted,
            ])
        print(
            f"[HOM] {tau_idx:02d}/{total_taus:02d} "
            f"tau={tau_ns:.3f} ns | "
            f"coinc={coincidences}/{arrive_trials:.1f} "
            f"rate={coinc_rate:.4f} | early_abort={early_abort_runs}"
        )
