# -*- coding: utf-8 -*-
"""
HOM 实验仿真：统计符合率随延迟 tau 的变化。
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor
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


def _run_hom_run(
    tau_ns: float,
    shots_per_run: int,
    noise_cfg: Optional[dict],
    window_ns: float,
    criterion: str,
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
        verbose=False,
    )

    apply_qfc(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        theta_H=np.pi/4,
        theta_V=np.pi/4,
        verbose=False,
    )
    apply_780_filter(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=False,
        rng=run_rng,
    )
    project_to_1517(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=False,
    )
    result.mps, _ = apply_fiber_channel(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        fiber_params=_make_fiber_params(),
        rng=run_rng,
        verbose=False,
    )

    t_wait_us = 80.0
    t2_us = 1000.0
    if t2_us > 0.0:
        p_dephase = 0.5 * (1.0 - np.exp(-t_wait_us / t2_us))
    else:
        p_dephase = 0.0
    _apply_atomic_dephasing(result.mps, p_dephase, rng=run_rng, verbose=False)

    apply_bs(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=False,
    )

    extreme, _ = _atom_extreme_state(result.mps, eps=ATOM_EXTREME_EPS)
    if extreme:
        return 0, True

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
            verbose=False,
        )
        if criterion == "bsm":
            if det_result.success:
                coincidences += 1
        else:
            if _is_port_coincidence(det_result.clicks, window_bins):
                coincidences += 1

    return coincidences, False


def _run_hom_run_task(args):
    tau_ns, shots_per_run, noise_cfg, window_ns, criterion = args
    coincidences, early_abort = _run_hom_run(
        tau_ns,
        shots_per_run,
        noise_cfg,
        window_ns,
        criterion,
    )
    return coincidences, early_abort


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

    summary_path = output_dir / "hom_summary.csv"
    with open(summary_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "tau_ns",
            "trials_target",
            "trials_total",
            "valid_trials",
            "early_abort_runs",
            "coinc_counts",
            "coinc_rate",
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
        valid_trials = 0
        early_abort_runs = 0
        runs_attempted = 0

        if valid_only:
            limit = max_attempts if max_attempts is not None else n_runs * 3
            while valid_trials < trials_target and runs_attempted < limit:
                runs_attempted += 1
                coincid_run, early_abort = _run_hom_run(
                    tau_ns, shots_per_run, noise_cfg, window_ns, criterion
                )
                if early_abort:
                    early_abort_runs += 1
                    continue
                valid_trials += shots_per_run
                trials_total += shots_per_run
                coincidences += coincid_run
            if valid_trials < trials_target:
                print(
                    f"[HOM] tau={tau_ns:.3f} ns 未达到目标有效样本数: "
                    f"{valid_trials}/{trials_target} (max_attempts={limit})"
                )
        else:
            runs_attempted = n_runs
            if jobs <= 1:
                for _ in range(n_runs):
                    coincid_run, early_abort = _run_hom_run(
                        tau_ns, shots_per_run, noise_cfg, window_ns, criterion
                    )
                    if early_abort:
                        early_abort_runs += 1
                    valid_trials += shots_per_run
                    trials_total += shots_per_run
                    coincidences += coincid_run
            else:
                tasks = [
                    (tau_ns, shots_per_run, noise_cfg, window_ns, criterion)
                    for _ in range(n_runs)
                ]
                with ProcessPoolExecutor(max_workers=jobs) as executor:
                    for coincid_run, early_abort in executor.map(_run_hom_run_task, tasks):
                        if early_abort:
                            early_abort_runs += 1
                        valid_trials += shots_per_run
                        trials_total += shots_per_run
                        coincidences += coincid_run

        coinc_rate = (coincidences / trials_total) if trials_total > 0 else 0.0
        with open(summary_path, 'a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                f"{tau_ns:.6f}",
                trials_target,
                trials_total,
                valid_trials,
                early_abort_runs,
                coincidences,
                f"{coinc_rate:.8f}",
                f"{window_ns:.3f}",
                criterion,
                n_runs,
                runs_attempted,
            ])
        print(
            f"[HOM] {tau_idx:02d}/{total_taus:02d} "
            f"tau={tau_ns:.3f} ns | "
            f"coinc={coincidences}/{trials_total} "
            f"rate={coinc_rate:.4f} | early_abort={early_abort_runs}"
        )
