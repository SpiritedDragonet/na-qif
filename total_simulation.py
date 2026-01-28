# -*- coding: utf-8 -*-
"""
完整仿真流程：双原子发射 -> 时间仓波包 -> QFC -> 分束器 -> 探测

仿真阶段：
1. 双原子发射（780nm光子到时间仓）
2. QFC频率转换（780nm -> 1517nm）
3. 780nm滤波器（滤除未转换的光子）
4. 分束器（A_n与B_n干涉）
5. 双光子探测与Bell态测量

链结构（新架构）：
    初始：[A1, B1, ..., AN, BN, atomA, atomB]  (仓在前，原子在后)
    发射后：[atomA, atomB, A1, B1, ..., AN, BN]  (原子向左移动到最前)

原子向左移动，依次与每个仓对相互作用，不再需要SWAP conveyor belt。
"""

import sys
import os
import csv
import time
import argparse
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Optional, Tuple
from collections import Counter, deque
from types import SimpleNamespace

# Add project root to path (for running as standalone script)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 调试开关：默认 False（非调试模式）
DEBUG_MODE = False

from atom_sim.simulation import (  # noqa: E402
    run_two_photon_detection,
    enumerate_success_events,
    build_detection_effects_6d,
    compute_fidelity_with_bell,
)
from atom_sim.visualization import plot_dual_arm_heatmap  # noqa: E402
from atom_sim.visualization.wavepacket import plot_cross_bin_joint_heatmap  # noqa: E402
from atom_sim.experiment.common import (  # noqa: E402
    DEFAULT_DARK_RATE_INTRINSIC_HZ,
    DEFAULT_BG_RATE_MEAN_HZ,
    DEFAULT_BG_RATE_STD_HZ,
    ATOM_EXTREME_EPS,
    _get_emission_params,
    _make_fiber_params,
    _compute_window_bins,
    _compute_noise_params,
    _atom_extreme_state,
    PipelineHooks,
    run_emission_to_bs,
    run_task_queue,
)
from atom_sim.experiment.hom import (  # noqa: E402
    run_hom_experiment,
    parse_hom_cli,
    validate_no_hom_args,
)

SUMMARY_HEADER = [
    "run",
    "shot",
    "success",
    "bell",
    "click_count",
    "events",
    "dark_rate_intrinsic_hz",
    "dark_rate_bg_hz",
    "p_dark_intrinsic",
    "p_bg",
    "p_noise",
    "p_no_loss",
    "p_arrive",
    "p_success_given_arrival",
    "p_success_all",
    "p_success_true",
    "p_success_false",
    "p_success_no_dark",
    "p_false_approx",
    "false_fraction",
    "false_fraction_approx",
    "fidelity_true",
    "fidelity_false",
    "fidelity_all",
    "fidelity_no_dark",
    "p_qubit_emit",
    "fidelity_shot_full",
    "runs",
    "shots_per_run",
    "total_shots",
    "success",
    "success_rate",
    "bell_counts",
    "click_count_dist",
    "dark_rate_intrinsic_hz",
    "dark_rate_bg_hz",
    "p_dark_intrinsic",
    "p_bg",
    "p_noise",
    "p_no_loss",
    "p_arrive",
    "p_success_given_arrival",
    "p_success_all",
    "p_success_true",
    "p_success_false",
    "p_success_no_dark",
    "p_false_approx",
    "false_fraction",
    "false_fraction_approx",
    "avg_p_qubit_emit",
    "avg_fidelity_true",
    "avg_fidelity_false",
    "avg_fidelity_all",
    "avg_fidelity_no_dark",
]


class Tee:
    """同时输出到多个流（用于日志与控制台同步）。"""
    def __init__(self, *streams):
        self._streams = streams
        self._tty_streams = [
            s for s in streams if getattr(s, "isatty", lambda: False)()
        ]

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        for stream in self._tty_streams:
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return bool(self._tty_streams)


def _parse_run_params(
    argv,
) -> Tuple[int, int, int, dict, str, str, dict, bool, bool, float, bool]:
    parser = argparse.ArgumentParser(
        prog="python total_simulation.py",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--runs", "--n-runs", dest="n_runs", type=int)
    parser.add_argument("--shots", "--shots-per-run", dest="shots_per_run", type=int)
    parser.add_argument("--jobs", dest="jobs", type=int)
    parser.add_argument("--mode", "--trial-type", dest="mode")
    parser.add_argument("--no-fiber-noise", dest="no_fiber_noise", action="store_true")

    parser.add_argument("--tau", dest="tau", type=float)
    parser.add_argument("--tau-start", dest="tau_start", type=float)
    parser.add_argument("--tau-end", dest="tau_end", type=float)
    parser.add_argument("--tau-step", dest="tau_step", type=float)
    parser.add_argument("--tau-points", dest="tau_points", type=int)
    parser.add_argument("--window-ns", dest="window_ns", type=float)
    parser.add_argument("--max-attempts", dest="max_attempts", type=int)

    parser.add_argument("--dark-hz", dest="dark_rate_intrinsic_hz", type=float)
    parser.add_argument("--bg-mean-hz", dest="bg_rate_mean_hz", type=float)
    parser.add_argument("--bg-std-hz", dest="bg_rate_std_hz", type=float)
    parser.add_argument("--enum-mode", dest="enum_mode", type=str)
    parser.add_argument("--plot-all", dest="plot_all", action="store_true")
    parser.add_argument("--eta-det", dest="eta_det", type=float)
    parser.add_argument("--ideal-det", dest="ideal_det", action="store_true")
    args = parser.parse_args(argv[1:])

    noise_cfg = {
        "dark_rate_intrinsic_hz": DEFAULT_DARK_RATE_INTRINSIC_HZ,
        "bg_rate_mean_hz": DEFAULT_BG_RATE_MEAN_HZ,
        "bg_rate_std_hz": DEFAULT_BG_RATE_STD_HZ,
    }
    if args.dark_rate_intrinsic_hz is not None:
        noise_cfg["dark_rate_intrinsic_hz"] = args.dark_rate_intrinsic_hz
    if args.bg_rate_mean_hz is not None:
        noise_cfg["bg_rate_mean_hz"] = args.bg_rate_mean_hz
    if args.bg_rate_std_hz is not None:
        noise_cfg["bg_rate_std_hz"] = args.bg_rate_std_hz

    n_runs = args.n_runs if args.n_runs is not None else 1
    shots_per_run = args.shots_per_run if args.shots_per_run is not None else 1
    jobs = args.jobs if args.jobs is not None else 1
    mode = (args.mode or "SIM").upper()
    fiber_noise = not args.no_fiber_noise

    if n_runs < 1:
        parser.error("N_runs 必须 >= 1")
    if shots_per_run < 1:
        parser.error("shots_per_run 必须 >= 1")
    if jobs < 1:
        parser.error("jobs 必须 >= 1")

    enum_mode = (args.enum_mode or "dark").strip().lower()
    if enum_mode not in ("dark", "no-dark"):
        parser.error("enum-mode 仅支持 dark / no-dark")

    eta_det = 0.85
    if args.eta_det is not None:
        eta_det = float(args.eta_det)
    ideal_det = bool(args.ideal_det)
    if ideal_det:
        eta_det = 1.0
    if not (0.0 < eta_det <= 1.0):
        parser.error("eta_det 必须在 (0, 1] 内")

    hom_cfg = {}
    if mode == "HOM":
        hom_cfg = parse_hom_cli(args, parser)
    else:
        validate_no_hom_args(args, parser)

    plot_all = bool(args.plot_all)
    return (
        n_runs,
        shots_per_run,
        jobs,
        noise_cfg,
        enum_mode,
        mode,
        hom_cfg,
        plot_all,
        fiber_noise,
        eta_det,
        ideal_det,
    )


def save_debug_info(
    mps,
    n_bins: int,
    stage: str,
    output_dir: Path,
    step_index: int,
    run_tag: Optional[str] = None,
):
    """
    保存调试信息到文件。

    Parameters
    ----------
    mps : MPSState
        当前MPS态
    n_bins : int
        时间仓数量
    stage : str
        当前阶段名称
    output_dir : Path
        输出目录
    step_index : int
        步骤索引
    """
    from atom_sim.simulation.detection import (
        compute_photon_statistics,
        extract_spin_state, compute_fidelity_with_bell
    )

    info = {}
    info['stage'] = stage
    info['step'] = step_index

    # MPS维度信息
    chi_list = mps._mps.chi
    d_list = mps.d
    info['n_sites'] = len(d_list)
    info['n_bins'] = n_bins
    info['bond_dimensions'] = f'chi_min={min(chi_list)}, chi_max={max(chi_list)}, chi_mean={np.mean(chi_list):.1f}'
    info['local_dimensions'] = f'first_5={d_list[:5]}, last_5={d_list[-5:]}'

    # 光子统计
    stats = compute_photon_statistics(mps, n_bins, verbose=False)
    info['photon_stats'] = stats

    # 原子态信息
    spin_state, p_qubit = extract_spin_state(mps, n_bins)
    info['spin_state_diag'] = np.diag(spin_state).real.tolist()
    info['p_qubit'] = p_qubit
    if p_qubit > 0:
        spin_state_cond = spin_state / p_qubit
        info['spin_purity'] = float(np.real(np.trace(spin_state_cond @ spin_state_cond)))
    else:
        info['spin_purity'] = 0.0

    # Bell态保真度
    for bell in ['Psi+', 'Psi-', 'Phi+', 'Phi-']:
        f_full = compute_fidelity_with_bell(spin_state, bell)
        f_cond = f_full / p_qubit if p_qubit > 0 else 0.0
        info[f'fidelity_{bell.replace("+", "p").replace("-", "m")}_full'] = f_full
        info[f'fidelity_{bell.replace("+", "p").replace("-", "m")}_cond'] = f_cond

    # 保存到文件
    prefix = f"{run_tag}_" if run_tag else ""
    info_file = output_dir / f'{prefix}debug_step_{step_index:02d}_{stage.replace(" ", "_").lower()}.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f'调试信息 - {stage}\n')
        f.write('='*60 + '\n\n')
        f.write('MPS维度信息:\n')
        f.write(f'  n_sites = {info["n_sites"]}\n')
        f.write(f'  n_bins = {info["n_bins"]}\n')
        f.write(f'  {info["bond_dimensions"]}\n')
        f.write(f'  {info["local_dimensions"]}\n\n')
        f.write('光子统计:\n')
        f.write(f'  总期望光子数 = {stats["n_total"]:.4f}\n')
        f.write(f'  780nm: H={stats.get("n_780_H", 0):.4f}, V={stats.get("n_780_V", 0):.4f}, total={stats.get("n_780_total", 0):.4f}\n')
        f.write(f'  1517nm: H={stats.get("n_1517_H", 0):.4f}, V={stats.get("n_1517_V", 0):.4f}, total={stats.get("n_1517_total", 0):.4f}\n')
        f.write(f'  期望损耗光子数 = {stats["loss_expected"]:.4f}\n\n')
        f.write('原子态信息:\n')
        f.write(f'  对角元: {info["spin_state_diag"]}\n')
        f.write(f'  p_qubit: {info["p_qubit"]:.4f}\n')
        f.write(f'  纯度(条件化): {info["spin_purity"]:.4f}\n\n')
        f.write('Bell态保真度:\n')
        f.write(f'  Psi+ (full/cond) = {info["fidelity_Psip_full"]:.4f} / {info["fidelity_Psip_cond"]:.4f}\n')
        f.write(f'  Psi- (full/cond) = {info["fidelity_Psim_full"]:.4f} / {info["fidelity_Psim_cond"]:.4f}\n')
        f.write(f'  Phi+ (full/cond) = {info["fidelity_Phip_full"]:.4f} / {info["fidelity_Phip_cond"]:.4f}\n')
        f.write(f'  Phi- (full/cond) = {info["fidelity_Phim_full"]:.4f} / {info["fidelity_Phim_cond"]:.4f}\n')

    print(f'  调试信息已保存: {info_file.name}')


def _append_click_summary(
    summary_path: Path,
    lock_path: Path,
    run_index: int,
    shot_index: int,
    det_result,
    metrics: Optional[dict],
):
    clicks = [(c.detector, c.bin_index) for c in det_result.clicks]
    fidelity_shot_full = ""
    if det_result.success and det_result.bell_state:
        fidelity_full = compute_fidelity_with_bell(det_result.spin_state, det_result.bell_state)
        fidelity_shot_full = format(fidelity_full, ".6f")
    row = [
        run_index,
        shot_index,
        det_result.success,
        det_result.bell_state,
        len(clicks),
        clicks,
        _format_metric(metrics, "dark_rate_intrinsic_hz", ".3f"),
        _format_metric(metrics, "dark_rate_bg_hz", ".3f"),
        _format_metric(metrics, "p_dark_intrinsic", ".8f"),
        _format_metric(metrics, "p_bg", ".8f"),
        _format_metric(metrics, "p_noise", ".8f"),
        _format_metric(metrics, "p_no_loss", ".8f"),
        _format_metric(metrics, "p_arrive", ".8f"),
        _format_metric(metrics, "p_success_given_arrival", ".8f"),
        _format_metric(metrics, "p_success_all", ".8f"),
        _format_metric(metrics, "p_success_true", ".8f"),
        _format_metric(metrics, "p_success_false", ".8f"),
        _format_metric(metrics, "p_success_no_dark", ".8f"),
        _format_metric(metrics, "p_false_approx", ".8f"),
        _format_metric(metrics, "false_fraction", ".6f"),
        _format_metric(metrics, "false_fraction_approx", ".6f"),
        _format_metric(metrics, "fidelity_true", ".6f"),
        _format_metric(metrics, "fidelity_false", ".6f"),
        _format_metric(metrics, "fidelity_all", ".6f"),
        _format_metric(metrics, "fidelity_no_dark", ".6f"),
        _format_metric(metrics, "p_qubit_emit", ".6f"),
        fidelity_shot_full,
    ]
    if len(row) < len(SUMMARY_HEADER):
        row += [""] * (len(SUMMARY_HEADER) - len(row))
    with _file_lock(lock_path):
        _append_csv_row(summary_path, row)


def _make_zero_success_metrics(include_no_dark: bool = True) -> dict:
    metrics = {
        "p_no_loss": 0.0,
        "p_arrive": 0.0,
        "p_success_all": 0.0,
        "p_success_true": 0.0,
        "p_success_false": 0.0,
        "p_success_given_arrival": 0.0,
        "fidelity_all": 0.0,
        "fidelity_true": 0.0,
        "fidelity_false": 0.0,
        "p_success_no_dark": 0.0 if include_no_dark else None,
        "fidelity_no_dark": 0.0 if include_no_dark else None,
        "p_qubit_emit": 0.0,
        "p_false_approx": 0.0 if include_no_dark else None,
        "false_fraction": 0.0,
        "false_fraction_approx": 0.0 if include_no_dark else None,
    }
    return metrics


def _make_empty_det_result() -> SimpleNamespace:
    zero_spin = np.zeros((4, 4), dtype=complex)
    return SimpleNamespace(
        clicks=[],
        success=False,
        bell_state="",
        spin_state=zero_spin,
    )


def _early_abort_run(
    summary_path: Path,
    lock_path: Path,
    run_index: int,
    shots_per_run: int,
    success_metrics: Optional[dict],
    reason: str,
) -> tuple:
    print(f"\n[早停] {reason}")
    run_stats = _init_stats()
    det_result = _make_empty_det_result()
    for shot_index in range(1, shots_per_run + 1):
        _append_click_summary(
            summary_path,
            lock_path,
            run_index,
            shot_index,
            det_result,
            success_metrics,
        )
        run_stats["shots"] += 1
        run_stats["clicks"][0] += 1
    return run_stats, success_metrics


def _init_stats() -> dict:
    return {
        "shots": 0,
        "success": 0,
        "bell": Counter(),
        "clicks": Counter(),
    }


def _merge_stats(dst: dict, src: dict) -> None:
    dst["shots"] += src["shots"]
    dst["success"] += src["success"]
    dst["bell"].update(src["bell"])
    dst["clicks"].update(src["clicks"])


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "-"
    parts = []
    for key in sorted(counter.keys()):
        parts.append(f"{key}:{counter[key]}")
    return ",".join(parts)


def _write_csv_header(path: Path, header: list) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


def _append_csv_row(path: Path, row: list) -> None:
    with open(path, 'a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def _format_metric(metrics: Optional[dict], key: str, fmt: str) -> str:
    if not metrics or key not in metrics:
        return ""
    value = metrics.get(key)
    if value is None:
        return ""
    return format(value, fmt)


@contextmanager
def _file_lock(lock_path: Path, stale_s: float = 120.0) -> None:
    lock_fd = None
    while True:
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, f"{os.getpid()} {time.time()}".encode("utf-8"))
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_s:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _claim_plot_slot(output_dir: Path, run_tag: str) -> bool:
    marker_path = output_dir / ".plot_run_claimed"
    lock_path = output_dir / ".plot_run_claimed.lock"
    with _file_lock(lock_path):
        if marker_path.exists():
            return False
        marker_path.write_text(run_tag, encoding="utf-8")
    return True


def _release_plot_slot(output_dir: Path, run_tag: str) -> None:
    marker_path = output_dir / ".plot_run_claimed"
    lock_path = output_dir / ".plot_run_claimed.lock"
    with _file_lock(lock_path):
        if not marker_path.exists():
            return
        current = marker_path.read_text(encoding="utf-8").strip()
        if current == run_tag:
            marker_path.unlink()


def _init_combined_summary(summary_path: Path) -> None:
    _write_csv_header(summary_path, SUMMARY_HEADER)
    _append_csv_row(summary_path, [""] * len(SUMMARY_HEADER))


def _finalize_combined_summary(
    summary_path: Path,
    lock_path: Path,
    stats: dict,
    n_runs: int,
    shots_per_run: int,
    metrics: Optional[dict],
) -> None:
    shots = stats["shots"]
    success = stats["success"]
    success_rate = (success / shots) if shots > 0 else 0.0
    bell_str = _format_counter(stats["bell"])
    click_str = _format_counter(stats["clicks"])
    summary_values = [
        n_runs,
        shots_per_run,
        shots,
        success,
        f"{success_rate:.4f}",
        bell_str,
        click_str,
        _format_metric(metrics, "dark_rate_intrinsic_hz", ".3f"),
        _format_metric(metrics, "dark_rate_bg_hz", ".3f"),
        _format_metric(metrics, "p_dark_intrinsic", ".8f"),
        _format_metric(metrics, "p_bg", ".8f"),
        _format_metric(metrics, "p_noise", ".8f"),
        _format_metric(metrics, "p_no_loss", ".8f"),
        _format_metric(metrics, "p_arrive", ".8f"),
        _format_metric(metrics, "p_success_given_arrival", ".8f"),
        _format_metric(metrics, "p_success_all", ".8f"),
        _format_metric(metrics, "p_success_true", ".8f"),
        _format_metric(metrics, "p_success_false", ".8f"),
        _format_metric(metrics, "p_success_no_dark", ".8f"),
        _format_metric(metrics, "p_false_approx", ".8f"),
        _format_metric(metrics, "false_fraction", ".6f"),
        _format_metric(metrics, "false_fraction_approx", ".6f"),
        _format_metric(metrics, "p_qubit_emit", ".6f"),
        _format_metric(metrics, "fidelity_true", ".6f"),
        _format_metric(metrics, "fidelity_false", ".6f"),
        _format_metric(metrics, "fidelity_all", ".6f"),
        _format_metric(metrics, "fidelity_no_dark", ".6f"),
    ]
    with _file_lock(lock_path):
        with open(summary_path, 'r', encoding='utf-8', newline='') as file:
            rows = list(csv.reader(file))
        if not rows:
            _init_combined_summary(summary_path)
            with open(summary_path, 'r', encoding='utf-8', newline='') as file:
                rows = list(csv.reader(file))
        header = rows[0] if rows else SUMMARY_HEADER
        if len(rows) < 2:
            rows = [header, [""] * len(header)]
        summary_row = [""] * len(header)
        try:
            start_idx = header.index("runs")
        except ValueError:
            start_idx = SUMMARY_HEADER.index("runs")
        for offset, value in enumerate(summary_values):
            if start_idx + offset < len(summary_row):
                summary_row[start_idx + offset] = value
        data_rows = rows[2:] if len(rows) > 2 else []
        def _sort_key(row):
            try:
                run_id = int(row[0])
            except (ValueError, TypeError, IndexError):
                run_id = 10**9
            try:
                shot_id = int(row[1])
            except (ValueError, TypeError, IndexError):
                shot_id = 10**9
            return (run_id, shot_id)
        data_rows.sort(key=_sort_key)
        final_rows = [header, summary_row] + data_rows
        with open(summary_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(final_rows)


def _write_extra_data(
    output_path: Path,
    fiber_sample,
    pre_bs_arm_a: np.ndarray,
    pre_bs_arm_b: np.ndarray,
    cross_bin_joint: np.ndarray,
) -> None:
    def _format_matrix(mat: np.ndarray) -> str:
        return np.array2string(
            mat,
            precision=6,
            suppress_small=True,
            separator=", ",
        )

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("fiber_sample\n")
        if fiber_sample is not None:
            if len(fiber_sample) >= 9:
                U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase, phase_slope, phase_jitter_std = fiber_sample[:9]
            else:
                U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase = fiber_sample
                phase_slope = 0.0
                phase_jitter_std = 0.0
            eta_mean_A = 0.5 * (eta_H_A + eta_V_A)
            eta_mean_B = 0.5 * (eta_H_B + eta_V_B)
            file.write(f"eta_mean_A = {eta_mean_A:.6f}\n")
            file.write(f"eta_mean_B = {eta_mean_B:.6f}\n")
            file.write(f"eta_H_A = {eta_H_A:.6f}\n")
            file.write(f"eta_V_A = {eta_V_A:.6f}\n")
            file.write(f"eta_H_B = {eta_H_B:.6f}\n")
            file.write(f"eta_V_B = {eta_V_B:.6f}\n")
            file.write(f"pdl_A = {eta_H_A - eta_V_A:+.6f}\n")
            file.write(f"pdl_B = {eta_H_B - eta_V_B:+.6f}\n")
            file.write(f"phase = {phase:.6f}\n")
            file.write(f"phase_slope = {phase_slope:.6e}\n")
            file.write(f"phase_jitter_std = {phase_jitter_std:.6e}\n")
            file.write(f"U_A = {_format_matrix(U_A)}\n")
            file.write(f"U_B = {_format_matrix(U_B)}\n")
        else:
            file.write("fiber_sample = None\n")

        file.write("\npre_bs_photon_per_bin\n")
        file.write("bin\tarmA\tarmB\n")
        for idx in range(len(pre_bs_arm_a)):
            file.write(f"{idx}\t{pre_bs_arm_a[idx]:.6f}\t{pre_bs_arm_b[idx]:.6f}\n")

        file.write("\ncross_bin_joint_after_bs\n")
        file.write(f"shape = {cross_bin_joint.shape}\n")
        np.savetxt(file, cross_bin_joint, fmt="%.6e")


def _write_success_metrics_detail(
    output_dir: Path,
    run_tag: str,
    metrics: dict,
) -> Path:
    output_path = output_dir / f"{run_tag}_success_metrics.txt"
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("success_metrics\n")
        file.write("=" * 60 + "\n")
        if "eta_det" in metrics:
            file.write(f"eta_det = {metrics['eta_det']:.6f}\n")
        if "window_bins" in metrics:
            file.write(f"window_bins = {metrics['window_bins']}\n")
        if "dark_rate_intrinsic_hz" in metrics:
            file.write(f"dark_rate_intrinsic_hz = {metrics['dark_rate_intrinsic_hz']:.3f}\n")
        if "dark_rate_bg_hz" in metrics:
            file.write(f"dark_rate_bg_hz = {metrics['dark_rate_bg_hz']:.3f}\n")
        if "p_dark_intrinsic" in metrics:
            file.write(f"p_dark_intrinsic = {metrics['p_dark_intrinsic']:.8f}\n")
        if "p_bg" in metrics:
            file.write(f"p_bg = {metrics['p_bg']:.8f}\n")
        if "p_noise" in metrics:
            file.write(f"p_noise = {metrics['p_noise']:.8f}\n")
        if "t_wait_us" in metrics:
            file.write(f"t_wait_us = {metrics['t_wait_us']:.3f}\n")
        if "t2_us" in metrics:
            file.write(f"t2_us = {metrics['t2_us']:.3f}\n")
        if "p_dephase" in metrics:
            file.write(f"p_dephase = {metrics['p_dephase']:.6f}\n")
        if "p_qubit_emit" in metrics:
            file.write(f"p_qubit_emit = {metrics['p_qubit_emit']:.6f}\n")
        if "p_no_loss" in metrics:
            file.write(f"p_no_loss = {metrics['p_no_loss']:.8f}\n")

        file.write(f"p_arrive = {metrics['p_arrive']:.8f}\n")
        if metrics.get("p_success_no_dark") is not None:
            file.write(f"p_success_no_dark = {metrics['p_success_no_dark']:.8f}\n")
            file.write(f"fidelity_no_dark = {metrics['fidelity_no_dark']:.6f}\n")
        else:
            file.write("p_success_no_dark = N/A\n")
            file.write("fidelity_no_dark = N/A\n")
        file.write(f"p_success_all = {metrics['p_success_all']:.8f}\n")
        file.write(f"fidelity_all = {metrics['fidelity_all']:.6f}\n")
        if metrics.get("p_false_approx") is not None:
            file.write(f"p_false_approx = {metrics['p_false_approx']:.8f}\n")
            file.write(f"false_fraction_approx = {metrics['false_fraction_approx']:.6f}\n")
        else:
            file.write("p_false_approx = N/A\n")
            file.write("false_fraction_approx = N/A\n")

        file.write(f"p_success_true = {metrics['p_success_true']:.8f}\n")
        file.write(f"p_success_false = {metrics['p_success_false']:.8f}\n")
        file.write(f"p_success_given_arrival = {metrics['p_success_given_arrival']:.8f}\n")
        file.write(f"false_fraction = {metrics['false_fraction']:.6f}\n")
        file.write(f"fidelity_true = {metrics['fidelity_true']:.6f}\n")
        file.write(f"fidelity_false = {metrics['fidelity_false']:.6f}\n")
    return output_path


def _init_success_metrics_accumulator() -> dict:
    return {
        "runs": 0,
        "no_dark_runs": 0,
        "p_no_loss_sum": 0.0,
        "p_arrive_sum": 0.0,
        "p_success_sum": 0.0,
        "p_success_true_sum": 0.0,
        "p_success_false_sum": 0.0,
        "p_success_no_dark_sum": 0.0,
        "dark_rate_intrinsic_sum": 0.0,
        "dark_rate_bg_sum": 0.0,
        "p_dark_intrinsic_sum": 0.0,
        "p_bg_sum": 0.0,
        "p_noise_sum": 0.0,
        "p_qubit_emit_sum": 0.0,
        "fidelity_weighted_sum": 0.0,
        "fidelity_true_weighted_sum": 0.0,
        "fidelity_false_weighted_sum": 0.0,
        "fidelity_no_dark_weighted_sum": 0.0,
    }


def _accumulate_success_metrics(acc: dict, metrics: dict) -> None:
    acc["runs"] += 1
    acc["p_no_loss_sum"] += metrics.get("p_no_loss", 0.0)
    acc["p_arrive_sum"] += metrics["p_arrive"]
    acc["p_success_sum"] += metrics["p_success_all"]
    acc["p_success_true_sum"] += metrics["p_success_true"]
    acc["p_success_false_sum"] += metrics["p_success_false"]
    p_success_no_dark = metrics.get("p_success_no_dark")
    fidelity_no_dark = metrics.get("fidelity_no_dark")
    if p_success_no_dark is not None and fidelity_no_dark is not None:
        acc["no_dark_runs"] += 1
        acc["p_success_no_dark_sum"] += p_success_no_dark
        acc["fidelity_no_dark_weighted_sum"] += p_success_no_dark * fidelity_no_dark
    acc["dark_rate_intrinsic_sum"] += metrics.get("dark_rate_intrinsic_hz", 0.0)
    acc["dark_rate_bg_sum"] += metrics.get("dark_rate_bg_hz", 0.0)
    acc["p_dark_intrinsic_sum"] += metrics.get("p_dark_intrinsic", 0.0)
    acc["p_bg_sum"] += metrics.get("p_bg", 0.0)
    acc["p_noise_sum"] += metrics.get("p_noise", 0.0)
    acc["p_qubit_emit_sum"] += metrics.get("p_qubit_emit", 0.0)
    acc["fidelity_weighted_sum"] += metrics["p_success_all"] * metrics["fidelity_all"]
    acc["fidelity_true_weighted_sum"] += metrics["p_success_true"] * metrics["fidelity_true"]
    acc["fidelity_false_weighted_sum"] += metrics["p_success_false"] * metrics["fidelity_false"]


def _finalize_success_metrics(acc: dict) -> dict:
    runs = max(acc["runs"], 1)
    no_dark_runs = acc.get("no_dark_runs", 0)
    p_no_loss = acc["p_no_loss_sum"] / runs
    p_arrive = acc["p_arrive_sum"] / runs
    p_success_all = acc["p_success_sum"] / runs
    p_success_true = acc["p_success_true_sum"] / runs
    p_success_false = acc["p_success_false_sum"] / runs
    p_success_no_dark = (
        acc["p_success_no_dark_sum"] / no_dark_runs
        if no_dark_runs > 0
        else None
    )
    p_success_given_arrival = (acc["p_success_true_sum"] / acc["p_arrive_sum"]) if acc["p_arrive_sum"] > 0 else 0.0
    dark_rate_intrinsic_hz = acc["dark_rate_intrinsic_sum"] / runs
    dark_rate_bg_hz = acc["dark_rate_bg_sum"] / runs
    p_dark_intrinsic = acc["p_dark_intrinsic_sum"] / runs
    p_bg = acc["p_bg_sum"] / runs
    p_noise = acc["p_noise_sum"] / runs
    p_qubit_emit = acc["p_qubit_emit_sum"] / runs

    fidelity_all = acc["fidelity_weighted_sum"] / acc["p_success_sum"] if acc["p_success_sum"] > 0 else 0.0
    fidelity_true = acc["fidelity_true_weighted_sum"] / acc["p_success_true_sum"] if acc["p_success_true_sum"] > 0 else 0.0
    fidelity_false = acc["fidelity_false_weighted_sum"] / acc["p_success_false_sum"] if acc["p_success_false_sum"] > 0 else 0.0
    fidelity_no_dark = None
    if no_dark_runs > 0 and acc["p_success_no_dark_sum"] > 0:
        fidelity_no_dark = acc["fidelity_no_dark_weighted_sum"] / acc["p_success_no_dark_sum"]

    p_false_approx = None
    if p_success_no_dark is not None:
        p_false_approx = max(0.0, p_success_all - p_success_no_dark)
    false_fraction = (p_success_false / p_success_all) if p_success_all > 0 else 0.0
    false_fraction_approx = None
    if p_false_approx is not None:
        false_fraction_approx = (p_false_approx / p_success_all) if p_success_all > 0 else 0.0

    return {
        "p_no_loss": p_no_loss,
        "p_arrive": p_arrive,
        "p_success_all": p_success_all,
        "p_success_true": p_success_true,
        "p_success_false": p_success_false,
        "p_success_given_arrival": p_success_given_arrival,
        "dark_rate_intrinsic_hz": dark_rate_intrinsic_hz,
        "dark_rate_bg_hz": dark_rate_bg_hz,
        "p_dark_intrinsic": p_dark_intrinsic,
        "p_bg": p_bg,
        "p_noise": p_noise,
        "p_qubit_emit": p_qubit_emit,
        "fidelity_all": fidelity_all,
        "fidelity_true": fidelity_true,
        "fidelity_false": fidelity_false,
        "p_success_no_dark": p_success_no_dark,
        "fidelity_no_dark": fidelity_no_dark,
        "p_false_approx": p_false_approx,
        "false_fraction": false_fraction,
        "false_fraction_approx": false_fraction_approx,
    }

def _run_single_simulation_core(
    output_dir: Path,
    run_index: int,
    n_runs: int,
    summary_path: Path,
    summary_lock_path: Path,
    shots_per_run: int,
    show_plots: bool,
    plot_all: bool,
    noise_cfg: Optional[dict],
    enum_mode: str,
    fiber_noise: bool,
    eta_det: float,
    ideal_det: bool,
):
    run_tag = f"run{run_index:03d}"
    success_metrics = None
    stage_total = 6
    plot_paths = []
    plot_claimed = False

    def _stage(idx: int, label: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{run_tag} {ts}] [阶段 {idx}/{stage_total}] {label}")

    def _should_plot() -> bool:
        nonlocal plot_claimed
        if plot_all:
            return True
        if plot_claimed:
            return True
        plot_claimed = _claim_plot_slot(output_dir, run_tag)
        return plot_claimed

    def _record_plot(path: Path) -> None:
        if plot_all or not plot_claimed:
            return
        plot_paths.append(path)

    def _cleanup_plots() -> None:
        if plot_all or not plot_claimed:
            return
        for path in plot_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
        _release_plot_slot(output_dir, run_tag)

    print("\n" + "=" * 80)
    print(f"Run {run_index}/{n_runs} ({run_tag})")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("运行发射 + QFC + 分束器 + 探测仿真...")

    run_rng = np.random.default_rng()
    p_no_loss = 0.0

    pre_bs_arm_a = None
    pre_bs_arm_b = None
    joint_after_bs = None
    fiber_sample = None

    stage_map = {
        "发射": 1,
        "QFC + 780滤波 + 1517投影": 2,
        "光纤信道": 3,
        "分束器 + 诊断/可视化": 4,
    }

    def _on_stage(label: str) -> None:
        _stage(stage_map[label], label)

    def _atom_extreme_reason(stage_label: str, mps) -> Optional[str]:
        if stage_label not in ("After QFC + Filter", "After Fiber Channel"):
            return None
        extreme, probs = _atom_extreme_state(mps, eps=ATOM_EXTREME_EPS)
        if not extreme:
            return None
        pA0, pA1, pAe, pAu, pB0, pB1, pBe, pBu = probs
        return (
            f"{stage_label} 原子态接近基态 |0>/<1| 极端值，"
            f"A(p0={pA0:.3f}, p1={pA1:.3f}, pe={pAe:.3f}, pu={pAu:.3f}) "
            f"B(p0={pB0:.3f}, p1={pB1:.3f}, pe={pBe:.3f}, pu={pBu:.3f})"
        )

    def _after_emission(emission) -> None:
        if _should_plot():
            print("\n生成发射后的可视化图...")
            emission_plot_path = output_dir / f"{run_tag}_1_after_emission.png"
            plot_dual_arm_heatmap(
                emission,
                save_path=str(emission_plot_path),
                show_atomic=True,
                stage_name="After Emission",
                show=show_plots,
            )
            _record_plot(emission_plot_path)
        if DEBUG_MODE:
            print("\n保存调试信息...")
            save_debug_info(
                mps=emission.mps,
                n_bins=emission.get_n_bins(),
                stage='After Emission',
                output_dir=output_dir,
                step_index=1,
                run_tag=run_tag,
            )

    def _after_qfc_filter(emission) -> None:
        if _should_plot():
            print("\n生成QFC+滤波后的可视化图...")
            qfc_plot_path = output_dir / f"{run_tag}_2_after_qfc.png"
            plot_dual_arm_heatmap(
                emission.mps,
                save_path=str(qfc_plot_path),
                show_atomic=False,
                stage_name="After QFC + 780nm Filter",
                time_grid=emission.time_grid,
                show=show_plots,
            )
            _record_plot(qfc_plot_path)
        if DEBUG_MODE:
            save_debug_info(
                mps=emission.mps,
                n_bins=emission.get_n_bins(),
                stage='After QFC + Filter',
                output_dir=output_dir,
                step_index=2,
                run_tag=run_tag,
            )

    def _after_fiber(emission, fiber_sample_in: tuple) -> None:
        nonlocal pre_bs_arm_a, pre_bs_arm_b, fiber_sample
        fiber_sample = fiber_sample_in
        if _should_plot():
            print("\n生成光纤传输后的可视化...")
            fiber_plot_path = output_dir / f"{run_tag}_3_after_fiber.png"
            plot_dual_arm_heatmap(
                emission.mps,
                save_path=str(fiber_plot_path),
                show_atomic=False,
                stage_name="After Fiber Channel",
                time_grid=emission.time_grid,
                show=show_plots,
            )
            _record_plot(fiber_plot_path)
        if DEBUG_MODE:
            save_debug_info(
                mps=emission.mps,
                n_bins=emission.get_n_bins(),
                stage='After Fiber Channel',
                output_dir=output_dir,
                step_index=3,
                run_tag=run_tag,
            )
        if DEBUG_MODE:
            print("\n诊断：检查BS前（含光纤）每个arm的光子分布...")
            n_bins = emission.get_n_bins()
            total_A = 0.0
            total_B = 0.0
            pre_bs_arm_a = np.zeros(n_bins)
            pre_bs_arm_b = np.zeros(n_bins)
            for n in range(n_bins):
                site_A = 2 + 2 * n  # Arm A的bin n
                site_B = 2 + 2 * n + 1  # Arm B的bin n

                rho_A = emission.mps.get_reduced_density([site_A])
                rho_B = emission.mps.get_reduced_density([site_B])

                photon_count = [0, 1, 1, 2, 2, 2]
                p_A = sum(rho_A[i, i].real * photon_count[i] for i in range(6))
                p_B = sum(rho_B[i, i].real * photon_count[i] for i in range(6))
                pre_bs_arm_a[n] = p_A
                pre_bs_arm_b[n] = p_B
                total_A += p_A
                total_B += p_B

                if p_A > 0.01 or p_B > 0.01:
                    print(f"  bin {n}: Arm_A={p_A:.4f}, Arm_B={p_B:.4f}")

            print(f"  总计: Arm_A={total_A:.4f}, Arm_B={total_B:.4f}")

    def _after_bs(emission) -> None:
        nonlocal joint_after_bs
        if _should_plot():
            print("\n生成BS后的可视化...")
            bs_plot_path = output_dir / f"{run_tag}_4_after_bs.png"
            plot_dual_arm_heatmap(
                emission.mps,
                save_path=str(bs_plot_path),
                show_atomic=False,
                stage_name="After Beam Splitter",
                time_grid=emission.time_grid,
                show=show_plots,
            )
            _record_plot(bs_plot_path)

        joint_after_bs = None
        if DEBUG_MODE:
            print("\n生成BS后的跨bin联合分布热图...")
            if _should_plot():
                joint_plot_path = output_dir / f"{run_tag}_4b_after_bs_cross_bin_joint.png"
                joint_after_bs = plot_cross_bin_joint_heatmap(
                    emission.mps,
                    n_bins=emission.get_n_bins(),
                    save_path=str(joint_plot_path),
                    arm_pair=("A", "B"),
                    normalize=False,
                    show=show_plots,
                )
                _record_plot(joint_plot_path)

        if DEBUG_MODE:
            save_debug_info(
                mps=emission.mps,
                n_bins=emission.get_n_bins(),
                stage='After BS',
                output_dir=output_dir,
                step_index=4,
                run_tag=run_tag,
            )

        if DEBUG_MODE:
            extra_data_path = output_dir / f"{run_tag}_extra_data.txt"
            _write_extra_data(
                extra_data_path,
                fiber_sample=fiber_sample,
                pre_bs_arm_a=pre_bs_arm_a,
                pre_bs_arm_b=pre_bs_arm_b,
                cross_bin_joint=joint_after_bs,
            )
            print(f"  Extra data saved: {extra_data_path.name}")

        if DEBUG_MODE:
            print("\n" + "="*80)
            print("【深入分析After BS】检查双光子态的分布")
            print("="*80)

            n_bins = emission.get_n_bins()

            total_photons_global = 0.0
            total_two_photon_states = 0.0

            state_names = ['vac', 'H', 'V', '2H', '2V', 'HV']

            print("\n【逐bin分析】")
            for n in range(n_bins):
                site_A = 2 + 2 * n
                site_B = 2 + 2 * n + 1
                rho_AB = emission.mps.get_reduced_density([site_A, site_B])

                bin_photons = 0.0
                bin_two_photon = 0.0

                for i_A in range(6):
                    for i_B in range(6):
                        prob = rho_AB[i_A, i_B, i_A, i_B].real

                        n_A = 0 if i_A == 0 else (1 if i_A in [1, 2] else 2)
                        n_B = 0 if i_B == 0 else (1 if i_B in [1, 2] else 2)

                        bin_photons += prob * (n_A + n_B)

                        if n_A + n_B == 2:
                            bin_two_photon += prob

                total_photons_global += bin_photons
                total_two_photon_states += bin_two_photon

                if bin_photons > 0.01:
                    print(f"\nBin {n}:")
                    print(f"  总光子数: {bin_photons:.6f}")
                    print(f"  双光子态概率: {bin_two_photon:.6f}")

                    print("  主要态分量:")
                    for i_A in range(6):
                        for i_B in range(6):
                            prob = rho_AB[i_A, i_B, i_A, i_B].real
                            if prob > 0.001:
                                print(f"    |{state_names[i_A]},{state_names[i_B]}>: {prob:.6f}")

            print("\n【全局统计】")
            print(f"  总光子数（所有bin）: {total_photons_global:.6f}")
            print(f"  双光子态总概率: {total_two_photon_states:.6f}")
            print(f"  非双光子态概率: {1.0 - total_two_photon_states:.6f}")

            if total_two_photon_states < 0.95:
                print("警告：双光子态概率 < 95%，存在单光子或多光子分量")

            print("="*80)

        if DEBUG_MODE:
            print("\n诊断：检查BS后的两端口关联...")
            n_bins = emission.get_n_bins()
            for n in range(n_bins):
                site_A = 2 + 2 * n
                site_B = 2 + 2 * n + 1
                rho_AB = emission.mps.get_reduced_density([site_A, site_B])

                p_vac_vac = rho_AB[0, 0, 0, 0].real
                p_H_H = rho_AB[1, 1, 1, 1].real
                p_V_V = rho_AB[2, 2, 2, 2].real
                p_H_V = rho_AB[1, 2, 1, 2].real
                p_V_H = rho_AB[2, 1, 2, 1].real
                p_2H_vac = rho_AB[3, 0, 3, 0].real
                p_2V_vac = rho_AB[4, 0, 4, 0].real
                p_HV_vac = rho_AB[5, 0, 5, 0].real
                p_vac_2H = rho_AB[0, 3, 0, 3].real
                p_vac_2V = rho_AB[0, 4, 0, 4].real
                p_vac_HV = rho_AB[0, 5, 0, 5].real

                total_nonvac = 1 - p_vac_vac
                if total_nonvac > 0.01:
                    print(f"  bin {n}: P(non-vac)={total_nonvac:.4f}")
                    if p_H_V + p_V_H > 1e-6:
                        print(f"    BSM成功态: P(H,V)={p_H_V:.6f}, P(V,H)={p_V_H:.6f}")
                    if p_H_H + p_V_V > 1e-6:
                        print(f"    同极化: P(H,H)={p_H_H:.6f}, P(V,V)={p_V_V:.6f}")
                    if p_2H_vac + p_2V_vac + p_HV_vac > 1e-6:
                        print(f"    bunching port1: P(2H,0)={p_2H_vac:.6f}, P(2V,0)={p_2V_vac:.6f}, P(HV,0)={p_HV_vac:.6f}")
                    if p_vac_2H + p_vac_2V + p_vac_HV > 1e-6:
                        print(f"    bunching port2: P(0,2H)={p_vac_2H:.6f}, P(0,2V)={p_vac_2V:.6f}, P(0,HV)={p_vac_HV:.6f}")

    emission_cfg = _get_emission_params(delay_ns=5.0)
    pipe = run_emission_to_bs(
        emission_cfg=emission_cfg,
        rng=run_rng,
        fiber_params=_make_fiber_params(noise_enabled=fiber_noise),
        verbose=True,
        hooks=PipelineHooks(
            on_stage=_on_stage,
            after_emission=_after_emission,
            after_qfc_filter=_after_qfc_filter,
            after_fiber=_after_fiber,
            after_bs=_after_bs,
            should_abort=_atom_extreme_reason,
        ),
    )
    if pipe.aborted:
        metrics = _make_zero_success_metrics(include_no_dark=(enum_mode == "no-dark"))
        metrics["p_qubit_emit"] = pipe.p_qubit_emit
        metrics["p_no_loss"] = pipe.p_no_loss
        _cleanup_plots()
        return _early_abort_run(
            summary_path,
            summary_lock_path,
            run_index,
            shots_per_run,
            metrics,
            reason=pipe.abort_reason or "流水线提前终止",
        )

    result = pipe.emission
    p_qubit_emit = pipe.p_qubit_emit
    p_no_loss = pipe.p_no_loss
    fiber_sample = pipe.fiber_sample
    t_wait_us = pipe.t_wait_us
    t2_us = pipe.t2_us
    p_dephase = pipe.p_dephase

    # =========================================================================
    # 探测
    # =========================================================================
    # 探测参数（基于Nature 2022实验）
    if ideal_det:
        eta_det = 1.0
    # 符合窗口：默认采用论文中数据分析窗口 70 ns
    coincidence_window_ns = 70.0
    bin_dt_s = result.time_grid.dt
    bin_dt_ns = bin_dt_s * 1e9
    window_bins = _compute_window_bins(coincidence_window_ns, bin_dt_ns)

    # QFC 背景噪声 + 探测器本底暗计数（两者独立）
    if ideal_det:
        noise = {
            "dark_rate_intrinsic_hz": 0.0,
            "bg_rate_mean_hz": 0.0,
            "bg_rate_std_hz": 0.0,
            "dark_rate_bg_hz": 0.0,
            "p_dark_intrinsic": 0.0,
            "p_bg": 0.0,
            "p_noise": 0.0,
        }
    else:
        noise = _compute_noise_params(noise_cfg, bin_dt_s, run_rng)
    dark_rate_intrinsic_hz = noise["dark_rate_intrinsic_hz"]
    bg_rate_mean_hz = noise["bg_rate_mean_hz"]
    bg_rate_std_hz = noise["bg_rate_std_hz"]
    dark_rate_bg_hz = noise["dark_rate_bg_hz"]
    p_dark_intrinsic = noise["p_dark_intrinsic"]
    p_bg = noise["p_bg"]
    p_noise = noise["p_noise"]
    print(f"\n探测器本底暗计数率: {dark_rate_intrinsic_hz:.3f} Hz -> p_dark={p_dark_intrinsic:.3e}")
    print(f"背景噪声参数: mean={bg_rate_mean_hz:.3f} Hz, std={bg_rate_std_hz:.3f} Hz")
    print(f"QFC 背景噪声率: {dark_rate_bg_hz:.3f} Hz -> p_bg={p_bg:.3e}")
    print(f"合并噪声概率 p_noise={p_noise:.3e}")
    print(f"点击时间窗 window_bins = {window_bins} (~{window_bins * bin_dt_ns:.1f} ns)")

    if DEBUG_MODE:
        # 诊断：检查每个bin的光子分布
        print("\n诊断：检查每个bin的光子分布...")
        # 先检查第一个bin的rho形状
        site_A = 2
        site_B = 3
        rho_AB = result.mps.get_reduced_density([site_A, site_B])
        print(f"  rho_AB shape: {rho_AB.shape}")
    _stage(5, "成功事件统计 (POVM)")
    print(f"\n成功事件枚举模式: {enum_mode}")
    enum_no_dark = None
    enum_main = None

    if enum_mode == "no-dark":
        print("\n枚举成功事件（无暗计数）...")
        enum_no_dark = enumerate_success_events(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            p_dark=0.0,
            window_bins=window_bins,
            verbose=True,
        )
        if p_noise > 0.0:
            print("  注意：枚举模式为 no-dark，将忽略暗计数，仅用于基线对比。")
        enum_main = enum_no_dark
    else:
        print("\n枚举成功事件（含暗计数）...")
        enum_main = enumerate_success_events(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            p_dark=p_noise,
            window_bins=window_bins,
            verbose=True,
        )

    success_metrics = {
        "eta_det": eta_det,
        "window_bins": window_bins,
        "dark_rate_intrinsic_hz": dark_rate_intrinsic_hz,
        "dark_rate_bg_hz": dark_rate_bg_hz,
        "p_dark_intrinsic": p_dark_intrinsic,
        "p_bg": p_bg,
        "p_noise": p_noise,
        "t_wait_us": t_wait_us,
        "t2_us": t2_us,
        "p_dephase": p_dephase,
        "p_qubit_emit": p_qubit_emit,
        "p_no_loss": p_no_loss,
        "p_arrive": enum_main.p_arrive,
        "p_success_all": enum_main.p_success,
        "p_success_true": enum_main.p_success_true,
        "p_success_false": enum_main.p_success_false,
        "p_success_given_arrival": enum_main.p_success_given_arrival,
        "fidelity_all": enum_main.fidelity_declared,
        "fidelity_true": enum_main.fidelity_true,
        "fidelity_false": enum_main.fidelity_false,
        "p_success_no_dark": enum_no_dark.p_success if enum_no_dark is not None else None,
        "fidelity_no_dark": enum_no_dark.fidelity_declared if enum_no_dark is not None else None,
    }
    success_metrics["false_fraction"] = (
        success_metrics["p_success_false"] / success_metrics["p_success_all"]
        if success_metrics["p_success_all"] > 0
        else 0.0
    )
    if enum_no_dark is not None:
        success_metrics["p_false_approx"] = max(
            0.0, success_metrics["p_success_all"] - success_metrics["p_success_no_dark"]
        )
        success_metrics["false_fraction_approx"] = (
            success_metrics["p_false_approx"] / success_metrics["p_success_all"]
            if success_metrics["p_success_all"] > 0
            else 0.0
        )
    else:
        success_metrics["p_false_approx"] = None
        success_metrics["false_fraction_approx"] = None

    success_path = _write_success_metrics_detail(output_dir, run_tag, success_metrics)
    print(f"  Success metrics saved: {success_path.name}")

    # 使用POVM抽样运行探测和BSM（可多次采样）
    _stage(6, "POVM抽样")
    print("\n运行探测和BSM（POVM抽样）...")
    run_stats = _init_stats()
    effects_cache = build_detection_effects_6d(eta_det, p_noise)
    for shot_index in range(1, shots_per_run + 1):
        print(f"\n[shot {shot_index}/{shots_per_run}]")
        det_result = run_two_photon_detection(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            window_bins=window_bins,
            p_dark=p_noise,
            effects_cache=effects_cache,
            #rng=np.random.default_rng(seed=19),
            rng=run_rng,
            verbose=True,
        )

        # 打印结果
        if det_result.success:
            print("\n  BSM成功!")
            print(f"  宣告的Bell态: {det_result.bell_state}")

            # 计算与期望Bell态的保真度（未归一化）
            fidelity_full = compute_fidelity_with_bell(det_result.spin_state, det_result.bell_state)
            rho = det_result.spin_state
            trace_rho = float(np.trace(rho).real)
            fidelity_cond = (fidelity_full / trace_rho) if trace_rho > 0 else 0.0
            print(f"  F_full(|{det_result.bell_state}>): {fidelity_full:.4e}")
            print(f"  F_cond(|{det_result.bell_state}>): {fidelity_cond:.4f}")

            # 计算与所有Bell态的保真度以供参考
            print("\n  与所有Bell态的保真度:")
            for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
                f_full = compute_fidelity_with_bell(det_result.spin_state, bell)
                f_cond = (f_full / trace_rho) if trace_rho > 0 else 0.0
                marker = " <-- 宣告的" if bell == det_result.bell_state else ""
                print(f"    F_full(|{bell}>): {f_full:.4e}, F_cond: {f_cond:.4f}{marker}")

            # 打印自旋态
            print("\n  自旋密度矩阵（量子比特子空间）:")
            print(f"    Tr(rho) = {trace_rho:.4e}")
            print(f"    纯度(未归一化) = {np.trace(rho @ rho).real:.4f}")
        else:
            print("\n  BSM失败 - 未找到成功模式")
            print(f"  点击数量: {len(det_result.clicks)}")

        # 保存探测后的调试信息
        if DEBUG_MODE:
            print("\n保存探测后调试信息...")
            if shots_per_run == 1:
                det_file = output_dir / f'{run_tag}_debug_detection_result.txt'
            else:
                det_file = output_dir / f'{run_tag}_shot{shot_index:03d}_debug_detection_result.txt'
            with open(det_file, 'w', encoding='utf-8') as file:
                file.write('探测结果\n')
                file.write('='*60 + '\n\n')
                file.write(f'成功: {det_result.success}\n')
                file.write(f'Bell态: {det_result.bell_state}\n')
                file.write(f'点击次数: {len(det_result.clicks)}\n')
                if det_result.clicks:
                    file.write(f'点击详情: {[(c.detector, c.bin_index) for c in det_result.clicks]}\n')

                    file.write('\n自旋密度矩阵:\n')
                    rho = det_result.spin_state
                    file.write('  基: |00>, |01>, |10>, |11>\n')
                    for i in range(4):
                        for j in range(4):
                            val = rho[i, j]
                            if abs(val) > 1e-10:
                                file.write(f'  rho[{i},{j}] = {val:.4f}\n')

                    file.write(f'\n纯度: {np.trace(rho @ rho).real:.4f}\n')

                    file.write('\nBell态保真度:\n')
                    for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
                        fid = compute_fidelity_with_bell(rho, bell)
                        marker = " <-- 探测到的" if bell == det_result.bell_state else ""
                        file.write(f'  F({bell}) = {fid:.4f}{marker}\n')

            print(f"  调试信息已保存: {det_file.name}")

        click_pairs = [(c.detector, c.bin_index) for c in det_result.clicks]
        declared = det_result.bell_state if det_result.success else "失败"
        if click_pairs:
            print(f"  宣告结果: {declared} | 点击记录: {click_pairs}")
        else:
            print(f"  宣告结果: {declared} | 点击记录: 无")

        run_stats["shots"] += 1
        run_stats["clicks"][len(det_result.clicks)] += 1
        if det_result.success:
            run_stats["success"] += 1
            if det_result.bell_state:
                run_stats["bell"][det_result.bell_state] += 1

        _append_click_summary(
            summary_path,
            summary_lock_path,
            run_index,
            shot_index,
            det_result,
            success_metrics,
        )

    print(f"\n完成! 文件已保存至: {output_dir}/")
    return run_stats, success_metrics


def _run_single_simulation(
    output_dir: Path,
    run_index: int,
    n_runs: int,
    summary_path: Path,
    summary_lock_path: Path,
    shots_per_run: int,
    noise_cfg: Optional[dict],
    enum_mode: str,
    fiber_noise: bool,
    eta_det: float,
    ideal_det: bool,
    log_path: Optional[Path] = None,
    mirror_console: bool = True,
    show_plots: bool = True,
    plot_all: bool = False,
):
    run_tag = f"run{run_index:03d}"
    if log_path is None:
        print(f"正在处理: {run_tag} | summary: {summary_path.name}")
        return _run_single_simulation_core(
            output_dir,
            run_index,
            n_runs,
            summary_path,
            summary_lock_path,
            shots_per_run,
            show_plots,
            plot_all,
            noise_cfg,
            enum_mode,
            fiber_noise,
            eta_det,
            ideal_det,
        )
    with open(log_path, 'w', encoding='utf-8') as log_file:
        if mirror_console:
            tee_out = Tee(sys.stdout, log_file)
            tee_err = Tee(sys.stderr, log_file)
        else:
            tee_out = Tee(log_file)
            tee_err = Tee(log_file)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_out, tee_err
        try:
            print(f"正在处理: {run_tag} | log: {log_path.name} | summary: {summary_path.name}")
            return _run_single_simulation_core(
                output_dir,
                run_index,
                n_runs,
                summary_path,
                summary_lock_path,
                shots_per_run,
                show_plots,
                plot_all,
                noise_cfg,
                enum_mode,
                fiber_noise,
                eta_det,
                ideal_det,
            )
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _run_single_simulation_task(
    output_dir,
    run_index,
    n_runs,
    summary_path,
    summary_lock_path,
    shots_per_run,
    noise_cfg,
    enum_mode,
    fiber_noise,
    eta_det,
    ideal_det,
    mirror_console,
    show_plots,
    plot_all,
):
    run_tag = f"run{run_index:03d}"
    log_path = output_dir / f"{run_tag}_console.log"
    run_stats, success_metrics = _run_single_simulation(
        output_dir,
        run_index,
        n_runs,
        summary_path,
        summary_lock_path,
        shots_per_run,
        noise_cfg,
        enum_mode,
        fiber_noise,
        eta_det,
        ideal_det,
        log_path=log_path,
        mirror_console=mirror_console,
        show_plots=show_plots,
        plot_all=plot_all,
    )
    return run_index, run_stats, success_metrics


def main():
    """主函数：运行发射 + QFC + 分束器 + 探测仿真。"""
    (
        n_runs,
        shots_per_run,
        jobs,
        noise_cfg,
        enum_mode,
        mode,
        hom_cfg,
        plot_all,
        fiber_noise,
        eta_det,
        ideal_det,
    ) = _parse_run_params(sys.argv)

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "HOM":
        print(f"Output directory: {output_dir}")
        print(
            f"HOM 模式: runs={n_runs}, shots_per_run={shots_per_run}, jobs={jobs}, "
            f"window_ns={hom_cfg.get('window_ns')}"
        )
        run_hom_experiment(
            output_dir=output_dir,
            n_runs=n_runs,
            shots_per_run=shots_per_run,
            jobs=jobs,
            hom_cfg=hom_cfg,
            noise_cfg=noise_cfg,
            fiber_noise=fiber_noise,
            eta_det=eta_det,
            ideal_det=ideal_det,
        )
        return

    # 统一的点击结果汇总文件
    clicks_summary_path = output_dir / "all_clicks_summary.csv"
    clicks_lock_path = output_dir / ".all_clicks_summary.lock"
    _init_combined_summary(clicks_summary_path)

    print(f"Output directory: {output_dir}")
    print(f"将运行 {n_runs} 次仿真，每次 {shots_per_run} 次探测采样...")
    jobs = max(1, min(jobs, n_runs, os.cpu_count() or 1))
    print(f"并行进程数: {jobs}")
    print(f"成功事件枚举模式: {enum_mode}")
    print(f"绘图模式: {'全部run' if plot_all else '仅单run'}")
    print(f"光纤噪声: {'开启' if fiber_noise else '关闭'}")
    print(f"探测参数: eta_det={eta_det:.3f} | 理想探测={'是' if ideal_det else '否'}")

    rng = np.random.default_rng()
    run_indices = list(range(1, n_runs + 1))

    def _build_task(run_index: int, mirror_console: bool, show_plots: bool) -> tuple:
        return (
            output_dir,
            run_index,
            n_runs,
            clicks_summary_path,
            clicks_lock_path,
            shots_per_run,
            noise_cfg,
            enum_mode,
            fiber_noise,
            eta_det,
            ideal_det,
            mirror_console,
            show_plots,
            plot_all,
        )

    focus_task = None
    if jobs > 1 and run_indices:
        focus_run = int(rng.choice(run_indices))
        run_indices = [r for r in run_indices if r != focus_run]
        focus_task = _build_task(focus_run, True, True)
        print(f"并行模式: 前台输出绑定 run{focus_run:03d}")

    overall_stats = _init_stats()
    overall_success = _init_success_metrics_accumulator()

    if jobs <= 1:
        tasks = deque(_build_task(r, True, True) for r in run_indices)
        focus_task = None
    else:
        tasks = deque(_build_task(r, False, False) for r in run_indices)

    completed = 0
    progress_every = max(1, n_runs // 10)

    def _next_task() -> Optional[tuple]:
        if not tasks:
            return None
        return tasks.popleft()

    def _on_result(task: tuple, result: tuple) -> None:
        nonlocal completed
        run_index, run_stats, success_metrics = result
        completed += 1
        if run_stats is not None:
            _merge_stats(overall_stats, run_stats)
        if success_metrics is not None:
            _accumulate_success_metrics(overall_success, success_metrics)
        print(f"[完成] run{run_index:03d}", flush=True)
        if completed % progress_every == 0 or completed == n_runs:
            print(f"[进度] {completed}/{n_runs}", flush=True)

    run_task_queue(
        jobs=jobs,
        task_fn=_run_single_simulation_task,
        next_task=_next_task,
        on_result=_on_result,
        focus_task=focus_task,
    )

    total_success_metrics = _finalize_success_metrics(overall_success)
    _finalize_combined_summary(
        clicks_summary_path,
        clicks_lock_path,
        overall_stats,
        n_runs,
        shots_per_run,
        total_success_metrics,
    )

if __name__ == "__main__":
    main()
