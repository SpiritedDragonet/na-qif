# -*- coding: utf-8 -*-
"""
CLI 入口与任务调度（单次实验逻辑见 atom_sim.experiment.single_run）。
"""

import sys
import os
import json
import argparse
import threading
import multiprocessing as mp
import time
import re
import shutil

# 必须在数值库导入前设置线程上限，避免 worker 多进程下触发线程风暴。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from dataclasses import fields, is_dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, get_origin, get_args, Union, get_type_hints
from concurrent.futures import ProcessPoolExecutor

# Add project root to path (for running as standalone script)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from atom_sim.experiment.common import (  # noqa: E402
    SimConfig,
)
from atom_sim.experiment.hom import (  # noqa: E402
    parse_hom_cli,
    validate_no_hom_args,
)
from atom_sim.experiment import (  # noqa: E402
    single_run,
    hom,
    window_scan,
    length_scan,
    bsm_scan,
    qfc_noise_scan,
    detector_bg_scan,
    summary,
)
from atom_sim.simulation import run_detection_self_checks  # noqa: E402


TASK_PROTOCOL_VERSION = "v2_core_trial"
RUN_MANIFEST_FILENAME = "run_manifest.json"
CORE_TASK_MODE = "CORE_TRIAL"
SUMMARY_TASK_MODE = "SUMMARY"
SUMMARY_TASK_FILENAME = "task_summary.json"
WORKER_HEARTBEAT_INTERVAL_SECS = 40
WORKER_STALE_RECOVERY_SECS = 240
SUPPORTED_EXPERIMENTS = {
    "SIM",
    "HOM",
    "WINDOW_SCAN",
    "BSM_SCAN",
    "LENGTH_SCAN",
    "QFC_NOISE_SCAN",
    "DETECTOR_BG_SCAN",
}
SERVER_CAPABILITY_CLI_DESTS = {"run_id", "rebuild_run"}
RESUME_PASSTHROUGH_CLI_DESTS = {
    "role",
    "queue_root",
    "run_id",
    "rebuild_run",
    "server_progress",
    "server_progress_quiet_secs",
    "server_progress_inline",
    "self_check",
}
CORE_TASK_BUILDERS = {
    "SIM": single_run.iter_sim_core_tasks,
    "HOM": hom.iter_hom_core_tasks,
    "WINDOW_SCAN": window_scan.iter_window_scan_core_tasks,
    "BSM_SCAN": bsm_scan.iter_bsm_scan_core_tasks,
    "LENGTH_SCAN": length_scan.iter_length_scan_core_tasks,
    "QFC_NOISE_SCAN": qfc_noise_scan.iter_qfc_noise_scan_core_tasks,
    "DETECTOR_BG_SCAN": detector_bg_scan.iter_detector_bg_scan_core_tasks,
}
CORE_TRIAL_TASK_RUNNERS = {
    "SIM": single_run.run_sim_task,
    "HOM": hom.run_hom_task,
    "WINDOW_SCAN": window_scan.run_window_scan_task,
    "BSM_SCAN": bsm_scan.run_bsm_scan_task,
    "LENGTH_SCAN": length_scan.run_length_scan_task,
    "QFC_NOISE_SCAN": qfc_noise_scan.run_qfc_noise_scan_task,
    "DETECTOR_BG_SCAN": detector_bg_scan.run_detector_bg_scan_task,
}


def _install_output_tracker(marker_path: Optional[Path]) -> None:
    """安装输出追踪器：每次 print 更新 marker_path 的 mtime。"""
    if marker_path is None:
        return
    import builtins

    if getattr(builtins, "_qsim_print_wrapped", False):
        return
    original_print = builtins.print

    def _tracked_print(*args, **kwargs):
        try:
            is_main_thread = threading.current_thread() is threading.main_thread()
            # 只在 marker 目录已存在时更新心跳，
            # 避免 run 已归档后再次 print 把旧 queue/<run>/heartbeat 重建出来。
            # 且仅统计主线程输出，避免 both 模式下 monitor 线程自己的进度打印
            # 反向触发 quiet window，导致“看起来每 quiet_secs 才刷一次进度”。
            if is_main_thread and marker_path.parent.exists():
                marker_path.touch()
        except Exception:
            pass
        return original_print(*args, **kwargs)

    builtins.print = _tracked_print
    builtins._qsim_print_wrapped = True


def _parse_run_params(argv):
    # 目的：解析 CLI 参数并构造统一的 SimConfig。
    # 复杂点：mode/task_type/role 三套概念并存——
    #   - role: server/worker/both（调度角色）
    #   - mode/task_type: SIM/HOM/WINDOW_SCAN/LENGTH_SCAN/BSM_SCAN（物理任务类型）
    #   - queue_root/run_id: run_id 仅 server/both 使用，用于任务目录隔离与续算
    parser = argparse.ArgumentParser(
        prog="python total_simulation.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "量子仿真入口：统一为 server/worker 任务队列架构。"
        ),
        epilog=(
            "用法示例:\n"
            "  # 主节点（服务端）\n"
            "  python total_simulation.py --role server --queue-root /mnt/quantum_sim/queue --run-id homA "
            "--task-type HOM --runs 120 --tau-start -40 --tau-end 40 --tau-step 2 --shots 1\n"
            "  # 抢占式节点（worker）：不指定 run-id，自动抢当前 queue 下可执行任务\n"
            "  python total_simulation.py --role worker --queue-root /mnt/quantum_sim/queue --cores 64\n"
            "  # 本地运行（server + worker）\n"
            "  python total_simulation.py --role both --queue-root /mnt/quantum_sim/queue --run-id simA --task-type SIM --runs 10\n"
            "  # 窗口扫描任务（WINDOW_SCAN）\n"
            "  python total_simulation.py --role both --queue-root /mnt/quantum_sim/queue --run-id simwA --task-type WINDOW_SCAN --runs 5 "
            "--window-sweep-start-ns 40 --window-sweep-end-ns 120 --window-sweep-step-ns 10 --shots 1\n"
            "  # 光纤长度扫描任务（LENGTH_SCAN）\n"
            "  python total_simulation.py --role both --queue-root /mnt/quantum_sim/queue --run-id simlA --task-type LENGTH_SCAN --runs 5 "
            "--length-sweep-start-km 10 --length-sweep-end-km 50 --length-sweep-step-km 10 --shots 1\n"
            "  # BSM误差扫描任务（BSM_SCAN）\n"
            "  python total_simulation.py --role both --queue-root /mnt/quantum_sim/queue --run-id simbsA --task-type BSM_SCAN --runs 5 "
            "--bs-sweep-start-theta 0.55 --bs-sweep-end-theta 1.02 --bs-sweep-step-theta 0.05 --shots 1\n"
            "  # QFC噪声扫描任务（QFC_NOISE_SCAN）\n"
            "  python total_simulation.py --role both --queue-root /mnt/quantum_sim/queue --run-id simqA --task-type QFC_NOISE_SCAN --runs 5 "
            "--qfc-noise-sweep-start-cps-per-mhz 10 --qfc-noise-sweep-end-cps-per-mhz 70 --qfc-noise-sweep-step-cps-per-mhz 5 --shots 1\n"
            "  # 探测效率-背景二维扫描任务（DETECTOR_BG_SCAN）\n"
            "  python total_simulation.py --role both --queue-root /mnt/quantum_sim/queue --run-id simdA --task-type DETECTOR_BG_SCAN --runs 5 "
            "--eta-det-sweep-start 0.60 --eta-det-sweep-end 0.95 --eta-det-sweep-step 0.05 "
            "--bg-mean-sweep-start-hz 80 --bg-mean-sweep-end-hz 240 --bg-mean-sweep-step-hz 20 --shots 1\n"
        ),
    )
    parser.add_argument("--role", dest="role", choices=["server", "worker", "both"], default="both", help="运行角色：server/worker/both（默认 both）")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--server-progress",
        dest="server_progress",
        action="store_true",
        help="server 定期输出进度（默认 server-only 开启，both 关闭）",
    )
    progress_group.add_argument(
        "--no-server-progress",
        dest="server_progress",
        action="store_false",
        help="禁用 server 进度输出（避免与 worker 输出混行）",
    )
    parser.set_defaults(server_progress=None)
    parser.add_argument(
        "--server-progress-quiet-secs",
        dest="server_progress_quiet_secs",
        type=float,
        help="server 进度输出的静默窗口（秒）；在此期间检测到 worker 输出则跳过显示",
    )
    parser.add_argument("--queue-root", dest="queue_root", type=str, default="queue", help="任务队列根目录（默认项目根目录下的 queue）")
    parser.add_argument("--run-id", dest="run_id", type=str, help="运行ID（仅 server/both 使用；用于隔离任务与断点续算）")
    parser.add_argument("--rebuild-run", dest="rebuild_run", action="store_true", help="仅 server 能力可用（role=server/both）；当 run-id 已存在时先归档为 _u 再按当前参数重建任务")
    parser.add_argument(
        "--task-type",
        dest="task_type",
        type=str,
        choices=[
            "SIM",
            "HOM",
            "WINDOW_SCAN",
            "LENGTH_SCAN",
            "BSM_SCAN",
            "QFC_NOISE_SCAN",
            "DETECTOR_BG_SCAN",
        ],
        help=(
            "任务类型：SIM / HOM / WINDOW_SCAN / LENGTH_SCAN / BSM_SCAN / "
            "QFC_NOISE_SCAN / DETECTOR_BG_SCAN（默认随 --mode）"
        ),
    )
    parser.add_argument("--runs", "--n-runs", dest="n_runs", type=int, help="仿真 run 次数（默认 1）")
    parser.add_argument("--shots", "--shots-per-run", dest="shots_per_run", type=int, help="每个 run 的探测采样次数（默认 1）")
    parser.add_argument(
        "--cores",
        dest="cores",
        type=int,
        help="可用 CPU 核数预算（默认 1，程序会自动计算实际并发进程数）",
    )
    parser.add_argument(
        "--mode",
        "--trial-type",
        dest="mode",
        help=(
            "运行模式：SIM / HOM / WINDOW_SCAN / LENGTH_SCAN / BSM_SCAN / "
            "QFC_NOISE_SCAN / DETECTOR_BG_SCAN（默认 SIM）"
        ),
    )
    parser.add_argument("--no-fiber-noise", dest="no_fiber_noise", action="store_true", help="关闭光纤噪声")
    for option, dest, cast, help_text in (
        ("--fiber-length-km", "fiber_length_km", float, "单臂到中继站的光纤长度 (km)，用于计算平均透过率"),
        ("--fiber-atten-db-per-km", "fiber_atten_db_per_km", float, "光纤衰减 (dB/km)"),
        ("--fiber-eta-std", "fiber_eta_std", float, "透过率随机波动标准差"),
        ("--fiber-pdl-sigma", "fiber_pdl_sigma", float, "PDL 相对差异标准差"),
        ("--fiber-phase-drift-std", "fiber_phase_drift_std", float, "两臂相位漂移标准差 (rad)"),
        ("--fiber-phase-slope-std", "fiber_phase_slope_std", float, "相位斜率标准差（基准口径: rad/bin@1ns，内部按 dt 线性缩放）"),
        ("--fiber-phase-jitter-std", "fiber_phase_jitter_std", float, "单bin相位抖动标准差（基准口径: rad@1ns，内部按 sqrt(dt) 缩放）"),
        ("--fiber-polarization-sigma", "fiber_polarization_sigma", float, "偏振小扰动模型标准差 (rad)"),
        ("--fiber-group-velocity-mps", "fiber_group_velocity_mps", float, "光纤群速度 (m/s)，用于自动计算 t_wait_us"),
        ("--t-wait-overhead-us", "t_wait_overhead_us", float, "等待时间固定开销 (us)，会加到 L/v_g 上"),
        ("--t-wait-length-scale", "t_wait_length_scale", float, "等待时间线性系数：T_wait = scale*L/v_g + overhead（默认 2，单臂往返口径）"),
        ("--t2-us", "t2_us", float, "原子退相干时间 T2 (us)"),
        ("--tau", "tau", float, "(HOM) 单一延迟 τ (ns)"),
        ("--tau-start", "tau_start", float, "(HOM) τ 起点 (ns)"),
        ("--tau-end", "tau_end", float, "(HOM) τ 终点 (ns)"),
        ("--tau-step", "tau_step", float, "(HOM) τ 步长 (ns)"),
        ("--tau-points", "tau_points", int, "(HOM) τ 采样点数"),
        ("--window-ns", "window_ns", float, "(HOM) 符合窗口 (ns)"),
        ("--window-sweep-start-ns", "window_sweep_start_ns", float, "(WINDOW_SCAN) 扫描起点窗口 (ns)"),
        ("--window-sweep-end-ns", "window_sweep_end_ns", float, "(WINDOW_SCAN) 扫描终点窗口 (ns)"),
        ("--window-sweep-step-ns", "window_sweep_step_ns", float, "(WINDOW_SCAN) 扫描步长窗口 (ns)"),
        ("--qfc-noise-sweep-start-cps-per-mhz", "qfc_noise_sweep_start_cps_per_mhz", float, "(QFC_NOISE_SCAN) 扫描起点QFC噪声谱密度 (cps/MHz)"),
        ("--qfc-noise-sweep-end-cps-per-mhz", "qfc_noise_sweep_end_cps_per_mhz", float, "(QFC_NOISE_SCAN) 扫描终点QFC噪声谱密度 (cps/MHz)"),
        ("--qfc-noise-sweep-step-cps-per-mhz", "qfc_noise_sweep_step_cps_per_mhz", float, "(QFC_NOISE_SCAN) 扫描步长QFC噪声谱密度 (cps/MHz)"),
        ("--eta-det-sweep-start", "eta_det_sweep_start", float, "(DETECTOR_BG_SCAN) 扫描起点探测效率 eta"),
        ("--eta-det-sweep-end", "eta_det_sweep_end", float, "(DETECTOR_BG_SCAN) 扫描终点探测效率 eta"),
        ("--eta-det-sweep-step", "eta_det_sweep_step", float, "(DETECTOR_BG_SCAN) 扫描步长探测效率 eta"),
        ("--bg-mean-sweep-start-hz", "bg_mean_sweep_start_hz", float, "(DETECTOR_BG_SCAN) 扫描起点背景均值 (Hz)"),
        ("--bg-mean-sweep-end-hz", "bg_mean_sweep_end_hz", float, "(DETECTOR_BG_SCAN) 扫描终点背景均值 (Hz)"),
        ("--bg-mean-sweep-step-hz", "bg_mean_sweep_step_hz", float, "(DETECTOR_BG_SCAN) 扫描步长背景均值 (Hz)"),
        ("--bs-sweep-start-theta", "bs_sweep_start_theta", float, "(BSM_SCAN) 扫描起点 BS theta (rad)"),
        ("--bs-sweep-end-theta", "bs_sweep_end_theta", float, "(BSM_SCAN) 扫描终点 BS theta (rad)"),
        ("--bs-sweep-step-theta", "bs_sweep_step_theta", float, "(BSM_SCAN) 扫描步长 BS theta (rad)"),
        ("--length-sweep-start-km", "length_sweep_start_km", float, "(LENGTH_SCAN) 扫描起点长度 (km)"),
        ("--length-sweep-end-km", "length_sweep_end_km", float, "(LENGTH_SCAN) 扫描终点长度 (km)"),
        ("--length-sweep-step-km", "length_sweep_step_km", float, "(LENGTH_SCAN) 扫描步长长度 (km)"),
        ("--attempt-rate-hz", "attempt_rate_hz", float, "(LENGTH_SCAN) 基础尝试率 (Hz)"),
        ("--attempt-overhead-us", "attempt_overhead_us", float, "(LENGTH_SCAN) 单次额外时延 (us)"),
    ):
        parser.add_argument(option, dest=dest, type=cast, help=help_text)
    parser.add_argument(
        "--fiber-polarization-model",
        dest="fiber_polarization_model",
        choices=["fixed", "haar", "perturb", "euler"],
        help="光纤偏振模型",
    )

    parser.add_argument("--dark-hz", dest="dark_rate_intrinsic_hz", type=float, help="探测器本底暗计数率 (Hz)")
    parser.add_argument("--bg-mean-hz", dest="bg_rate_mean_hz", type=float, help="背景噪声均值 (Hz)")
    parser.add_argument("--bg-std-hz", dest="bg_rate_std_hz", type=float, help="背景噪声标准差 (Hz)")
    for channel in ("h1", "v1", "h2", "v2"):
        channel_tag = channel.upper()
        parser.add_argument(
            f"--dark-hz-{channel}",
            dest=f"dark_hz_{channel}",
            type=float,
            help=f"{channel_tag} 通道本底暗计数率 (Hz)；未指定则用 --dark-hz",
        )
        for opt_prefix, dest_prefix, label in (
            ("bg-mean-hz", "bg_mean_hz", "背景噪声均值"),
            ("bg-std-hz", "bg_std_hz", "背景噪声标准差"),
        ):
            parser.add_argument(
                f"--{opt_prefix}-{channel}",
                dest=f"{dest_prefix}_{channel}",
                type=float,
                help=f"{channel_tag} 通道{label} (Hz)；未指定则用 --{opt_prefix}",
            )
    parser.add_argument("--detector-gate-ns", dest="detector_gate_ns", type=float, help="探测门宽 (ns)，用于将噪声概率从门宽映射到仿真 bin")
    parser.add_argument("--omega-peak-a", dest="omega_peak_a", type=float, help="A 臂驱动脉冲峰值 Ω_peak_A（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）")
    parser.add_argument("--omega-peak-b", dest="omega_peak_b", type=float, help="B 臂驱动脉冲峰值 Ω_peak_B（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）")
    parser.add_argument("--drive-waveform-a", dest="drive_waveform_a", choices=["gaussian", "sech", "square"], help="A 臂驱动包络类型")
    parser.add_argument("--drive-waveform-b", dest="drive_waveform_b", choices=["gaussian", "sech", "square"], help="B 臂驱动包络类型")
    parser.add_argument(
        "--hamiltonian-rate-unit",
        dest="hamiltonian_rate_unit",
        choices=["rad_s", "hz"],
        help="发射哈密顿量参数单位（omega/g/delta_*）：rad_s 或 hz（默认 rad_s）",
    )
    parser.add_argument(
        "--dissipation-rate-unit",
        dest="dissipation_rate_unit",
        choices=["hz", "rad_s"],
        help="发射耗散参数单位（kappa_*/gamma_*）：hz(=1/s) 或 rad_s（默认 hz）",
    )
    for option, dest, help_text in (
        ("--g-a", "g_a", "A 臂原子-腔耦合强度 g_A（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
        ("--g-b", "g_b", "B 臂原子-腔耦合强度 g_B（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
        ("--kappa-ex-a", "kappa_ex_a", "A 臂腔外耦合衰减率 kappa_ex_A（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-ex-b", "kappa_ex_b", "B 臂腔外耦合衰减率 kappa_ex_B（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-in-a", "kappa_in_a", "A 臂腔内损耗衰减率 kappa_in_A（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-in-b", "kappa_in_b", "B 臂腔内损耗衰减率 kappa_in_B（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-ex-h-a", "kappa_ex_h_a", "A 臂 H 偏振外耦合衰减率 kappa_ex_H_A（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-ex-v-a", "kappa_ex_v_a", "A 臂 V 偏振外耦合衰减率 kappa_ex_V_A（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-in-h-a", "kappa_in_h_a", "A 臂 H 偏振内损耗衰减率 kappa_in_H_A（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-in-v-a", "kappa_in_v_a", "A 臂 V 偏振内损耗衰减率 kappa_in_V_A（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-ex-h-b", "kappa_ex_h_b", "B 臂 H 偏振外耦合衰减率 kappa_ex_H_B（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-ex-v-b", "kappa_ex_v_b", "B 臂 V 偏振外耦合衰减率 kappa_ex_V_B（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-in-h-b", "kappa_in_h_b", "B 臂 H 偏振内损耗衰减率 kappa_in_H_B（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--kappa-in-v-b", "kappa_in_v_b", "B 臂 V 偏振内损耗衰减率 kappa_in_V_B（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--delta-u-a", "delta_u_a", "A 臂 |u> 态失谐 delta_u_A（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
        ("--delta-u-b", "delta_u_b", "B 臂 |u> 态失谐 delta_u_B（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
        ("--delta-e-a", "delta_e_a", "A 臂 |e> 态失谐 delta_e_A（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
        ("--delta-e-b", "delta_e_b", "B 臂 |e> 态失谐 delta_e_B（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
        ("--gamma-sigma-plus-a", "gamma_sigma_plus_a", "A 臂 |e>->|0> 自发辐射率 gamma_sigma_plus_A（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--gamma-sigma-minus-a", "gamma_sigma_minus_a", "A 臂 |e>->|1> 自发辐射率 gamma_sigma_minus_A（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--gamma-sigma-plus-b", "gamma_sigma_plus_b", "B 臂 |e>->|0> 自发辐射率 gamma_sigma_plus_B（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--gamma-sigma-minus-b", "gamma_sigma_minus_b", "B 臂 |e>->|1> 自发辐射率 gamma_sigma_minus_B（单位由 --dissipation-rate-unit 决定，默认 1/s）"),
        ("--delta-c-h-a", "delta_c_h_a", "A 臂 H 腔模失谐 delta_c_H_A（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
        ("--delta-c-v-a", "delta_c_v_a", "A 臂 V 腔模失谐 delta_c_V_A（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
        ("--delta-c-h-b", "delta_c_h_b", "B 臂 H 腔模失谐 delta_c_H_B（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
        ("--delta-c-v-b", "delta_c_v_b", "B 臂 V 腔模失谐 delta_c_V_B（单位由 --hamiltonian-rate-unit 决定，默认 rad/s）"),
    ):
        parser.add_argument(option, dest=dest, type=float, help=help_text)
    parser.add_argument("--qfc-theta-h", dest="qfc_theta_h", type=float, help="QFC H转换角 theta_H (rad)")
    parser.add_argument("--qfc-theta-v", dest="qfc_theta_v", type=float, help="QFC V转换角 theta_V (rad)")
    parser.add_argument("--qfc-phi-h", dest="qfc_phi_h", type=float, help="QFC H通道相位 phi_H (rad)")
    parser.add_argument("--qfc-phi-v", dest="qfc_phi_v", type=float, help="QFC V通道相位 phi_V (rad)")
    parser.add_argument("--alpha-h-plus-a", dest="alpha_h_plus_a", type=float, help="A 臂 Alpha[H,+] 偏振耦合系数")
    parser.add_argument("--alpha-h-minus-a", dest="alpha_h_minus_a", type=float, help="A 臂 Alpha[H,-] 偏振耦合系数")
    parser.add_argument("--alpha-v-plus-a", dest="alpha_v_plus_a", type=float, help="A 臂 Alpha[V,+] 偏振耦合系数")
    parser.add_argument("--alpha-v-minus-a", dest="alpha_v_minus_a", type=float, help="A 臂 Alpha[V,-] 偏振耦合系数")
    parser.add_argument("--alpha-h-plus-b", dest="alpha_h_plus_b", type=float, help="B 臂 Alpha[H,+] 偏振耦合系数")
    parser.add_argument("--alpha-h-minus-b", dest="alpha_h_minus_b", type=float, help="B 臂 Alpha[H,-] 偏振耦合系数")
    parser.add_argument("--alpha-v-plus-b", dest="alpha_v_plus_b", type=float, help="B 臂 Alpha[V,+] 偏振耦合系数")
    parser.add_argument("--alpha-v-minus-b", dest="alpha_v_minus_b", type=float, help="B 臂 Alpha[V,-] 偏振耦合系数")
    parser.add_argument("--filter-cavity-fwhm-mhz", dest="filter_cavity_fwhm_mhz", type=float, help="QFC后1517滤波腔线宽 FWHM (MHz)")
    parser.add_argument("--filter-cavity-detuning-mhz-a", dest="filter_cavity_detuning_mhz_a", type=float, help="A 臂滤波腔失谐 (MHz)")
    parser.add_argument("--filter-cavity-detuning-mhz-b", dest="filter_cavity_detuning_mhz_b", type=float, help="B 臂滤波腔失谐 (MHz)")
    parser.add_argument("--filter-cavity-eta-peak-a", dest="filter_cavity_eta_peak_a", type=float, help="A 臂滤波腔峰值透过率")
    parser.add_argument("--filter-cavity-eta-peak-b", dest="filter_cavity_eta_peak_b", type=float, help="B 臂滤波腔峰值透过率")
    parser.add_argument("--qfc-noise-sd-cps-per-mhz-a", dest="qfc_noise_sd_cps_per_mhz_a", type=float, help="A 臂QFC背景噪声谱密度 (cps/MHz)")
    parser.add_argument("--qfc-noise-sd-cps-per-mhz-b", dest="qfc_noise_sd_cps_per_mhz_b", type=float, help="B 臂QFC背景噪声谱密度 (cps/MHz)")
    parser.add_argument("--enum-mode", dest="enum_mode", type=str, help="成功事件枚举模式：dark/no-dark/both")
    parser.add_argument("--plot-all", dest="plot_all", action="store_true", help="所有 run 都绘图（默认仅保留一个）")
    parser.add_argument("--no-plot", dest="no_plot", action="store_true", help="完全禁止绘图（覆盖 plot-all）")
    parser.add_argument("--eta-det", dest="eta_det", type=float, help="探测效率 η (0~1)")
    for channel in ("h1", "v1", "h2", "v2"):
        parser.add_argument(
            f"--eta-det-{channel}",
            dest=f"eta_det_{channel}",
            type=float,
            help=f"{channel.upper()} 通道探测效率 η (0~1)；未指定则用 --eta-det",
        )
    parser.add_argument("--ideal-det", dest="ideal_det", action="store_true", help="理想探测（eta_det=1, 无噪声）")
    parser.add_argument("--bs-theta", dest="bs_theta", type=float, help="中心站 BS 混合角 theta (rad)，sin^2(theta) 为跨端口透射概率")
    parser.add_argument("--v-res", dest="v_res", type=float, help="残差可区分度 V_res (0~1)")
    parser.add_argument("--debug", dest="debug", action="store_true", help="开启调试模式（输出耗时等）")
    parser.add_argument("--self-check", dest="self_check", action="store_true", help="仅运行探测端自检并退出")
    args = parser.parse_args(argv[1:])
    has_server_capability = args.role in ("server", "both")
    has_worker_capability = args.role in ("worker", "both")
    explicit_cli_dests = set()
    option_actions = parser._option_string_actions
    for token in argv[1:]:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None:
            explicit_cli_dests.add(action.dest)

    # run-id 仅用于目录命名，因此要严格限制为“非路径字符串”
    run_id = args.run_id.strip() if args.run_id else None
    if run_id:
        if "/" in run_id or "\\" in run_id:
            parser.error("run-id 不能包含路径分隔符")
        if run_id in (".", ".."):
            parser.error("run-id 不能为 . 或 ..")
    if not has_server_capability:
        invalid = sorted(dst for dst in explicit_cli_dests if dst in SERVER_CAPABILITY_CLI_DESTS)
        if invalid:
            parser.error(
                "当前 role 不具备 server 能力，不能使用: "
                + ", ".join(invalid)
                + "（server/both 可用）"
            )
    if bool(args.rebuild_run) and run_id is None:
        parser.error("--rebuild-run 需要与 --run-id 一起使用")

    # 构造默认配置，然后用 CLI 覆盖。注意：SimConfig 是“可序列化的仿真输入快照”。
    config = SimConfig()

    def _set_arg_if_present(arg_name: str, target, attr_name: str, cast=None) -> None:
        raw = getattr(args, arg_name)
        if raw is None:
            return
        setattr(target, attr_name, cast(raw) if cast is not None else raw)

    for arg_name, target, attr_name, cast in (
        ("dark_rate_intrinsic_hz", config.noise, "dark_rate_intrinsic_hz", float),
        ("bg_rate_mean_hz", config.noise, "bg_rate_mean_hz", float),
        ("bg_rate_std_hz", config.noise, "bg_rate_std_hz", float),
        ("detector_gate_ns", config.noise, "detector_gate_ns", float),
        ("omega_peak_a", config.emission.arm_A, "omega_peak", float),
        ("omega_peak_b", config.emission.arm_B, "omega_peak", float),
        ("drive_waveform_a", config.emission, "drive_waveform_A", lambda value: str(value).strip().lower()),
        ("drive_waveform_b", config.emission, "drive_waveform_B", lambda value: str(value).strip().lower()),
        ("hamiltonian_rate_unit", config.emission, "hamiltonian_rate_unit", lambda value: str(value).strip().lower()),
        ("dissipation_rate_unit", config.emission, "dissipation_rate_unit", lambda value: str(value).strip().lower()),
        ("g_a", config.emission.arm_A, "g", float),
        ("g_b", config.emission.arm_B, "g", float),
        ("kappa_ex_a", config.emission.arm_A, "kappa_ex", float),
        ("kappa_ex_b", config.emission.arm_B, "kappa_ex", float),
        ("kappa_in_a", config.emission.arm_A, "kappa_in", float),
        ("kappa_in_b", config.emission.arm_B, "kappa_in", float),
        ("kappa_ex_h_a", config.emission.arm_A, "kappa_ex_H", float),
        ("kappa_ex_v_a", config.emission.arm_A, "kappa_ex_V", float),
        ("kappa_in_h_a", config.emission.arm_A, "kappa_in_H", float),
        ("kappa_in_v_a", config.emission.arm_A, "kappa_in_V", float),
        ("kappa_ex_h_b", config.emission.arm_B, "kappa_ex_H", float),
        ("kappa_ex_v_b", config.emission.arm_B, "kappa_ex_V", float),
        ("kappa_in_h_b", config.emission.arm_B, "kappa_in_H", float),
        ("kappa_in_v_b", config.emission.arm_B, "kappa_in_V", float),
        ("delta_u_a", config.emission.arm_A, "delta_u", float),
        ("delta_u_b", config.emission.arm_B, "delta_u", float),
        ("delta_e_a", config.emission.arm_A, "delta_e", float),
        ("delta_e_b", config.emission.arm_B, "delta_e", float),
        ("gamma_sigma_plus_a", config.emission.arm_A, "gamma_sigma_plus", float),
        ("gamma_sigma_minus_a", config.emission.arm_A, "gamma_sigma_minus", float),
        ("gamma_sigma_plus_b", config.emission.arm_B, "gamma_sigma_plus", float),
        ("gamma_sigma_minus_b", config.emission.arm_B, "gamma_sigma_minus", float),
        ("delta_c_h_a", config.emission.arm_A, "delta_c_H", float),
        ("delta_c_v_a", config.emission.arm_A, "delta_c_V", float),
        ("delta_c_h_b", config.emission.arm_B, "delta_c_H", float),
        ("delta_c_v_b", config.emission.arm_B, "delta_c_V", float),
        ("alpha_h_plus_a", config.emission.arm_A, "alpha_h_plus", float),
        ("alpha_h_minus_a", config.emission.arm_A, "alpha_h_minus", float),
        ("alpha_v_plus_a", config.emission.arm_A, "alpha_v_plus", float),
        ("alpha_v_minus_a", config.emission.arm_A, "alpha_v_minus", float),
        ("alpha_h_plus_b", config.emission.arm_B, "alpha_h_plus", float),
        ("alpha_h_minus_b", config.emission.arm_B, "alpha_h_minus", float),
        ("alpha_v_plus_b", config.emission.arm_B, "alpha_v_plus", float),
        ("alpha_v_minus_b", config.emission.arm_B, "alpha_v_minus", float),
        ("qfc_theta_h", config.qfc, "theta_H", float),
        ("qfc_theta_v", config.qfc, "theta_V", float),
        ("qfc_phi_h", config.qfc, "phi_H", float),
        ("qfc_phi_v", config.qfc, "phi_V", float),
        ("filter_cavity_fwhm_mhz", config.qfc.filter_cavity, "fwhm_mhz", float),
        ("filter_cavity_detuning_mhz_a", config.qfc.filter_cavity, "detuning_mhz_A", float),
        ("filter_cavity_detuning_mhz_b", config.qfc.filter_cavity, "detuning_mhz_B", float),
        ("filter_cavity_eta_peak_a", config.qfc.filter_cavity, "eta_peak_A", float),
        ("filter_cavity_eta_peak_b", config.qfc.filter_cavity, "eta_peak_B", float),
        ("qfc_noise_sd_cps_per_mhz_a", config.qfc, "qfc_noise_sd_cps_per_mhz_A", float),
        ("qfc_noise_sd_cps_per_mhz_b", config.qfc, "qfc_noise_sd_cps_per_mhz_B", float),
    ):
        _set_arg_if_present(arg_name, target, attr_name, cast)

    for detector in ("h1", "v1", "h2", "v2"):
        dark_value = getattr(args, f"dark_hz_{detector}", None)
        if dark_value is not None:
            config.noise.dark_rate_intrinsic_hz_map[detector.upper()] = float(dark_value)
        bg_mean_value = getattr(args, f"bg_mean_hz_{detector}", None)
        if bg_mean_value is not None:
            config.noise.bg_rate_mean_hz_map[detector.upper()] = float(bg_mean_value)
        bg_std_value = getattr(args, f"bg_std_hz_{detector}", None)
        if bg_std_value is not None:
            config.noise.bg_rate_std_hz_map[detector.upper()] = float(bg_std_value)
    # runs/shots/cores 是“任务粒度 + 并发预算”的核心参数
    for arg_name, attr_name in (
        ("n_runs", "runs"),
        ("shots_per_run", "shots_per_run"),
        ("cores", "cores"),
    ):
        _set_arg_if_present(arg_name, config.run, attr_name, int)
    # mode 与 task_type 的一致性约束：
    #   - mode 代表“当前命令意图”，task_type 代表“写入任务的类型”
    #   - 若两者同时给出，必须完全一致，否则会造成任务与执行逻辑错配
    mode = (args.mode or config.mode).upper()
    task_type = (args.task_type or mode).upper()
    if args.task_type is not None and args.mode is not None and task_type != mode:
        parser.error("task-type 与 mode 冲突，请保持一致")
    config.mode = task_type
    task_scan_attr_groups = {
        "WINDOW_SCAN": (
            "window_sweep_start_ns",
            "window_sweep_end_ns",
            "window_sweep_step_ns",
        ),
        "QFC_NOISE_SCAN": (
            "qfc_noise_sweep_start_cps_per_mhz",
            "qfc_noise_sweep_end_cps_per_mhz",
            "qfc_noise_sweep_step_cps_per_mhz",
        ),
        "DETECTOR_BG_SCAN": (
            "eta_det_sweep_start",
            "eta_det_sweep_end",
            "eta_det_sweep_step",
            "bg_mean_sweep_start_hz",
            "bg_mean_sweep_end_hz",
            "bg_mean_sweep_step_hz",
        ),
        "BSM_SCAN": (
            "bs_sweep_start_theta",
            "bs_sweep_end_theta",
            "bs_sweep_step_theta",
        ),
        "LENGTH_SCAN": (
            "length_sweep_start_km",
            "length_sweep_end_km",
            "length_sweep_step_km",
        ),
    }
    for attr_name in task_scan_attr_groups.get(task_type, ()):
        setattr(config.run, attr_name, getattr(args, attr_name))

    if task_type == "WINDOW_SCAN":
        pass
    if task_type == "LENGTH_SCAN":
        _set_arg_if_present("attempt_rate_hz", config.run, "attempt_rate_hz", float)
        _set_arg_if_present("attempt_overhead_us", config.run, "attempt_overhead_us", float)

    # 光纤噪声开关（注意：这会影响到统计与物理可解释性）
    config.fiber.noise_enabled = not args.no_fiber_noise
    for arg_name, target, attr_name, cast in (
        ("fiber_length_km", config.fiber, "length_km", float),
        ("fiber_atten_db_per_km", config.fiber, "attenuation_db_per_km", float),
        ("fiber_eta_std", config.fiber, "eta_std", float),
        ("fiber_pdl_sigma", config.fiber, "pdl_sigma", float),
        ("fiber_phase_drift_std", config.fiber, "phase_drift_std", float),
        ("fiber_phase_slope_std", config.fiber, "phase_slope_std", float),
        ("fiber_phase_jitter_std", config.fiber, "phase_jitter_std", float),
        ("fiber_polarization_model", config.fiber, "polarization_model", str),
        ("fiber_polarization_sigma", config.fiber, "polarization_sigma", float),
        ("fiber_group_velocity_mps", config.run, "fiber_group_velocity_mps", float),
        ("t_wait_overhead_us", config.run, "t_wait_overhead_us", float),
        ("t_wait_length_scale", config.run, "t_wait_length_scale", float),
        ("t2_us", config.run, "t2_us", float),
    ):
        _set_arg_if_present(arg_name, target, attr_name, cast)

    if config.run.runs < 1:
        parser.error("N_runs 必须 >= 1")
    if config.run.shots_per_run < 1:
        parser.error("shots_per_run 必须 >= 1")
    if config.run.cores < 1:
        parser.error("cores 必须 >= 1")
    if config.noise.detector_gate_ns <= 0.0:
        parser.error("detector_gate_ns 必须 > 0")
    if config.fiber.length_km < 0.0:
        parser.error("fiber_length_km 必须 >= 0")
    if config.fiber.attenuation_db_per_km < 0.0:
        parser.error("fiber_atten_db_per_km 必须 >= 0")
    if config.fiber.eta_std < 0.0:
        parser.error("fiber_eta_std 必须 >= 0")
    if config.fiber.pdl_sigma < 0.0:
        parser.error("fiber_pdl_sigma 必须 >= 0")
    if config.fiber.phase_drift_std < 0.0:
        parser.error("fiber_phase_drift_std 必须 >= 0")
    if config.fiber.phase_slope_std < 0.0:
        parser.error("fiber_phase_slope_std 必须 >= 0")
    if config.fiber.phase_jitter_std < 0.0:
        parser.error("fiber_phase_jitter_std 必须 >= 0")
    if config.run.fiber_group_velocity_mps <= 0.0:
        parser.error("fiber_group_velocity_mps 必须 > 0")
    if config.run.t_wait_overhead_us < 0.0:
        parser.error("t_wait_overhead_us 必须 >= 0")
    if config.run.t_wait_length_scale < 0.0:
        parser.error("t_wait_length_scale 必须 >= 0")
    if config.run.t2_us <= 0.0:
        parser.error("t2_us 必须 > 0")
    for field_name, value in (
        ("kappa_ex_A", config.emission.arm_A.kappa_ex),
        ("kappa_ex_B", config.emission.arm_B.kappa_ex),
        ("kappa_in_A", config.emission.arm_A.kappa_in),
        ("kappa_in_B", config.emission.arm_B.kappa_in),
        ("gamma_sigma_plus_A", config.emission.arm_A.gamma_sigma_plus),
        ("gamma_sigma_minus_A", config.emission.arm_A.gamma_sigma_minus),
        ("gamma_sigma_plus_B", config.emission.arm_B.gamma_sigma_plus),
        ("gamma_sigma_minus_B", config.emission.arm_B.gamma_sigma_minus),
    ):
        if float(value) < 0.0:
            parser.error(f"{field_name} 必须 >= 0")
    for field_name, value in (
        ("kappa_ex_H_A", config.emission.arm_A.kappa_ex_H),
        ("kappa_ex_V_A", config.emission.arm_A.kappa_ex_V),
        ("kappa_in_H_A", config.emission.arm_A.kappa_in_H),
        ("kappa_in_V_A", config.emission.arm_A.kappa_in_V),
        ("kappa_ex_H_B", config.emission.arm_B.kappa_ex_H),
        ("kappa_ex_V_B", config.emission.arm_B.kappa_ex_V),
        ("kappa_in_H_B", config.emission.arm_B.kappa_in_H),
        ("kappa_in_V_B", config.emission.arm_B.kappa_in_V),
    ):
        if value is not None and float(value) < 0.0:
            parser.error(f"{field_name} 必须 >= 0")

    # 成功事件枚举模式：
    #   - dark: 含暗计数
    #   - no-dark: 无暗计数
    #   - both: 同时输出 dark/no-dark 基线
    config.run.enum_mode = (args.enum_mode or config.run.enum_mode).strip().lower()
    if config.run.enum_mode not in ("dark", "no-dark", "both"):
        parser.error("enum-mode 仅支持 dark / no-dark / both")

    # 探测效率与理想探测的互斥/覆盖逻辑
    _set_arg_if_present("eta_det", config.detector, "eta_det", float)
    for detector in ("h1", "v1", "h2", "v2"):
        eta_value = getattr(args, f"eta_det_{detector}", None)
        if eta_value is not None:
            config.detector.eta_det_map[detector.upper()] = float(eta_value)
    config.detector.ideal_det = bool(args.ideal_det)
    if config.detector.ideal_det:
        config.detector.eta_det = 1.0
    _set_arg_if_present("bs_theta", config.detector, "bs_theta", float)
    _set_arg_if_present("v_res", config.detector, "v_res", float)
    if not (0.0 < config.detector.eta_det <= 1.0):
        parser.error("eta_det 必须在 (0, 1] 内")
    if not (0.0 <= config.detector.v_res <= 1.0):
        parser.error("v_res 必须在 [0, 1] 内")
    if not (0.0 <= config.detector.bs_theta <= float(np.pi / 2.0)):
        parser.error("bs_theta 必须在 [0, pi/2] 内")
    for field_name, value in (
        ("filter_cavity_fwhm_mhz", config.qfc.filter_cavity.fwhm_mhz),
        ("qfc_noise_sd_cps_per_mhz_A", config.qfc.qfc_noise_sd_cps_per_mhz_A),
        ("qfc_noise_sd_cps_per_mhz_B", config.qfc.qfc_noise_sd_cps_per_mhz_B),
    ):
        if float(value) < 0.0:
            parser.error(f"{field_name} 必须 >= 0")
    for field_name, value in (
        ("filter_cavity_eta_peak_A", config.qfc.filter_cavity.eta_peak_A),
        ("filter_cavity_eta_peak_B", config.qfc.filter_cavity.eta_peak_B),
    ):
        if not (0.0 <= float(value) <= 1.0):
            parser.error(f"{field_name} 必须在 [0,1] 内")

    if task_type == "HOM":
        config.hom = parse_hom_cli(args, parser)
    else:
        validate_no_hom_args(args, parser)
        config.hom = None
        if args.window_ns is not None:
            config.run.window_ns = float(args.window_ns)

    scan_validators = {
        "WINDOW_SCAN": window_scan.validate_window_scan_config,
        "QFC_NOISE_SCAN": qfc_noise_scan.validate_qfc_noise_scan_config,
        "DETECTOR_BG_SCAN": detector_bg_scan.validate_detector_bg_scan_config,
        "BSM_SCAN": bsm_scan.validate_bsm_scan_config,
        "LENGTH_SCAN": length_scan.validate_length_scan_config,
    }
    validator = scan_validators.get(task_type)
    if validator is not None:
        try:
            validator(config)
        except ValueError as exc:
            parser.error(str(exc))

    config.run.plot_all = bool(args.plot_all)
    config.run.plot_enabled = not bool(args.no_plot)
    config.run.debug = bool(args.debug)
    server_progress = True if args.server_progress is None else bool(args.server_progress)
    progress_quiet_secs = (
        20.0 if (has_server_capability and has_worker_capability) else 0.0
    ) if args.server_progress_quiet_secs is None else max(0.0, float(args.server_progress_quiet_secs))
    progress_inline = has_server_capability and not has_worker_capability
    return (
        config,
        args.role,
        args.queue_root,
        run_id,
        bool(args.rebuild_run),
        sorted(explicit_cli_dests),
        task_type,
        server_progress,
        progress_quiet_secs,
        progress_inline,
        bool(args.self_check),
    )


def _queue_paths(queue_root: Path) -> dict:
    # 约定目录结构（server/worker 通过这些目录交换任务与结果）
    root = Path(queue_root)
    return {
        "root": root,
        "tasks": root / "tasks",
        "pending": root / "tasks" / "pending",
        "inprogress": root / "tasks" / "inprogress",
        "done": root / "tasks" / "done",
        "error": root / "tasks" / "error",
        "results": root / "results",
        "summary": root / "summary",
        "heartbeat": root / "heartbeat",
    }


def _ensure_queue_dirs(paths: dict) -> None:
    # 任务目录必须全部存在，避免 worker race condition
    paths["pending"].mkdir(parents=True, exist_ok=True)
    paths["inprogress"].mkdir(parents=True, exist_ok=True)
    paths["done"].mkdir(parents=True, exist_ok=True)
    paths["error"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["summary"].mkdir(parents=True, exist_ok=True)
    paths["heartbeat"].mkdir(parents=True, exist_ok=True)


def _discover_run_roots(base_root: Path) -> list:
    # 扫描所有可能的 run 目录（以 tasks/pending 为识别特征）
    if not base_root.exists():
        return []
    roots = []
    for child in base_root.iterdir():
        if not child.is_dir():
            continue
        pending_dir = child / "tasks" / "pending"
        if pending_dir.exists():
            roots.append(child)
    return sorted(
        roots,
        key=lambda p: (
            (0, int(m.group(1)), p.name)
            if (m := re.match(r"(\d+)", p.name))
            else (1, p.name)
        ),
    )


def _is_run_active(run_root: Path, stale_seconds: int = 120, startup_grace_seconds: int = 20) -> bool:
    # 活跃判定：
    #   - 新建 run 的短暂保护窗口（避免并发 server 启动瞬间误归档）
    #   - server 心跳新鲜
    #   - 任一 worker 心跳新鲜
    #   - inprogress 任务近期被 touch
    now = time.time()
    try:
        if now - run_root.stat().st_mtime <= startup_grace_seconds:
            return True
    except FileNotFoundError:
        return False

    heartbeat = run_root / "summary" / "server_heartbeat.txt"
    if heartbeat.exists():
        try:
            if now - heartbeat.stat().st_mtime <= stale_seconds:
                return True
        except FileNotFoundError:
            pass

    worker_heartbeat_dir = run_root / "heartbeat"
    if worker_heartbeat_dir.exists():
        for hb in worker_heartbeat_dir.glob("worker_*.txt"):
            try:
                if now - hb.stat().st_mtime <= stale_seconds:
                    return True
            except FileNotFoundError:
                continue

    inprogress_dir = run_root / "tasks" / "inprogress"
    if inprogress_dir.exists():
        for task_path in inprogress_dir.glob("task_*.json"):
            try:
                if now - task_path.stat().st_mtime <= stale_seconds:
                    return True
            except FileNotFoundError:
                continue
    return False


def _archive_run(run_root: Path, outputs_root: Path, unfinished: bool = False) -> Optional[Path]:
    # 将一个 run_root 目录整体移动到 outputs/<timestamp>（未完成则加 _u）
    if not run_root.exists():
        return None
    outputs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{stamp}_u" if unfinished else stamp
    dest = outputs_root / name
    if dest.exists():
        suffix = 1
        while (outputs_root / f"{name}_{suffix}").exists():
            suffix += 1
        dest = outputs_root / f"{name}_{suffix}"
    try:
        shutil.move(str(run_root), str(dest))
    except Exception as exc:
        print(f"[server] 归档失败: {exc}")
        return None
    label = "未完成任务" if unfinished else "完成任务"
    print(f"[server] 已归档{label}到: {dest}")
    return dest


def _archive_existing_runs(
    base_root: Path,
    outputs_root: Path,
    exclude_run_id: Optional[str] = None,
    stale_seconds: int = 120,
) -> None:
    # 扫描所有 run 根目录：
    #   - 活跃 -> 保留
    #   - 非活跃 -> 一律归档为未完成（_u）
    for run_root in _discover_run_roots(base_root):
        if exclude_run_id and run_root.name == exclude_run_id:
            continue
        if _is_run_active(run_root, stale_seconds=stale_seconds):
            continue
        _archive_run(run_root, outputs_root, unfinished=True)


def _reserve_next_run_id(base_root: Path) -> str:
    # 原子分配 run-id：并发 server 同时启动时，确保不会领到同一个 id。
    candidate = 1
    while True:
        run_id = str(candidate)
        run_root = base_root / run_id
        try:
            run_root.mkdir(parents=False, exist_ok=False)
            return run_id
        except FileExistsError:
            candidate += 1


def _write_json_atomic(path: Path, payload: dict) -> None:
    # 原子写入：先写临时文件，再 replace，避免读到半写入文件
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)


def _to_plain_jsonable(value):
    if is_dataclass(value):
        return {field.name: _to_plain_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _to_plain_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_jsonable(item) for item in value]
    return value


def _apply_dict_to_dataclass(target, payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("SCHEMA_ERROR: manifest.config 必须是对象")
    type_hints = get_type_hints(type(target))
    for field in fields(target):
        if field.name not in payload:
            continue
        incoming = payload[field.name]
        current = getattr(target, field.name)
        annotation = type_hints.get(field.name, field.type)
        nested_type = None
        if isinstance(annotation, type) and hasattr(annotation, "__dataclass_fields__"):
            nested_type = annotation
        else:
            origin = get_origin(annotation)
            if origin is Union:
                for arg in get_args(annotation):
                    if arg is type(None):
                        continue
                    if isinstance(arg, type) and hasattr(arg, "__dataclass_fields__"):
                        nested_type = arg
                        break

        if incoming is None:
            setattr(target, field.name, None)
            continue

        if isinstance(incoming, dict):
            if is_dataclass(current):
                _apply_dict_to_dataclass(current, incoming)
                continue
            if nested_type is not None:
                nested_obj = nested_type()
                _apply_dict_to_dataclass(nested_obj, incoming)
                setattr(target, field.name, nested_obj)
                continue

        setattr(target, field.name, incoming)


def _write_run_manifest(paths: dict, task_type: str, config: SimConfig) -> None:
    manifest = {
        "protocol_version": TASK_PROTOCOL_VERSION,
        "task_type": task_type,
        "config": _to_plain_jsonable(config),
    }
    _write_json_atomic(paths["summary"] / RUN_MANIFEST_FILENAME, manifest)


def _load_run_manifest(paths: dict) -> dict:
    manifest_path = paths["summary"] / RUN_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(f"SCHEMA_ERROR: 缺少 {RUN_MANIFEST_FILENAME}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"SCHEMA_ERROR: 无法读取 {RUN_MANIFEST_FILENAME}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("SCHEMA_ERROR: run_manifest 顶层必须是对象")
    return data


def _apply_manifest_to_config(config: SimConfig, manifest: dict) -> None:
    protocol = str(manifest.get("protocol_version", "")).strip()
    if protocol != TASK_PROTOCOL_VERSION:
        raise ValueError(
            f"SCHEMA_ERROR: protocol_version 不匹配，期望 {TASK_PROTOCOL_VERSION}，实际 {protocol or '缺失'}"
        )
    cfg_payload = manifest.get("config")
    if not isinstance(cfg_payload, dict):
        raise ValueError("SCHEMA_ERROR: run_manifest.config 缺失或类型错误")
    _apply_dict_to_dataclass(config, cfg_payload)


def _validate_task_schema(task: dict, manifest: dict) -> None:
    mode = str(task.get("mode", "")).upper()
    if mode not in {CORE_TASK_MODE, SUMMARY_TASK_MODE}:
        raise ValueError(f"SCHEMA_ERROR: 不支持的 task.mode={mode}")

    if mode == CORE_TASK_MODE:
        experiment = str(task.get("experiment", "")).upper()
        if experiment not in SUPPORTED_EXPERIMENTS:
            raise ValueError(f"SCHEMA_ERROR: 不支持的 task.experiment={experiment}")
        payload = task.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("SCHEMA_ERROR: CORE_TRIAL 的 payload 必须是对象")
        if experiment == "HOM" and "tau_ns" not in payload:
            raise ValueError("SCHEMA_ERROR: HOM 缺少 payload.tau_ns")
        if experiment == "BSM_SCAN" and "bs_theta" not in payload:
            raise ValueError("SCHEMA_ERROR: BSM_SCAN 缺少 payload.bs_theta")
        if experiment == "LENGTH_SCAN" and "length_km" not in payload:
            raise ValueError("SCHEMA_ERROR: LENGTH_SCAN 缺少 payload.length_km")
        if experiment == "QFC_NOISE_SCAN" and "qfc_noise_sd_cps_per_mhz" not in payload:
            raise ValueError("SCHEMA_ERROR: QFC_NOISE_SCAN 缺少 payload.qfc_noise_sd_cps_per_mhz")
        if experiment == "DETECTOR_BG_SCAN":
            if "eta_det" not in payload or "bg_rate_mean_hz" not in payload:
                raise ValueError("SCHEMA_ERROR: DETECTOR_BG_SCAN 缺少 payload.eta_det / payload.bg_rate_mean_hz")

    if mode == SUMMARY_TASK_MODE:
        summary_for = str(task.get("summary_for", "")).upper()
        if summary_for not in SUPPORTED_EXPERIMENTS:
            raise ValueError(f"SCHEMA_ERROR: SUMMARY 缺少合法 summary_for，当前={summary_for}")
        manifest_task_type = str(manifest.get("task_type", "")).upper()
        if manifest_task_type and summary_for != manifest_task_type:
            raise RuntimeError(
                f"CONFIG_MISMATCH: summary_for={summary_for} 与 manifest.task_type={manifest_task_type} 不一致"
            )


def _build_summary_task(summary_for: str) -> dict:
    summary_for = str(summary_for).upper()
    return {
        "id": "summary",
        "mode": SUMMARY_TASK_MODE,
        "experiment": summary_for,
        "summary_for": summary_for,
    }


def _count_total_and_core_tasks(task_dir: Path) -> tuple[int, int]:
    total = 0
    core = 0
    for task_path in task_dir.glob("task_*.json"):
        total += 1
        if task_path.name != SUMMARY_TASK_FILENAME:
            core += 1
    return total, core


def _summary_task_exists(paths: dict) -> bool:
    for section in ("pending", "inprogress", "done", "error"):
        if (paths[section] / SUMMARY_TASK_FILENAME).exists():
            return True
    return False


def _enqueue_summary_task_if_needed(paths: dict, summary_for: Optional[str]) -> bool:
    if not summary_for or summary_for not in SUPPORTED_EXPERIMENTS:
        return False
    if _summary_task_exists(paths):
        return False
    try:
        _write_json_atomic(paths["pending"] / SUMMARY_TASK_FILENAME, _build_summary_task(summary_for))
        return True
    except Exception:
        return False


def _resolve_summary_for(paths: dict) -> Optional[str]:
    try:
        manifest = _load_run_manifest(paths)
    except Exception:
        return None
    summary_for = str(manifest.get("task_type", "")).upper()
    if summary_for in SUPPORTED_EXPERIMENTS:
        return summary_for
    return None


def _build_task_list(
    task_type: str,
    config: SimConfig,
    pending_dir: Path,
) -> int:
    # ------------------------------------------------------------------
    # 任务生成规则：
    #   - 所有物理子任务统一为 mode=CORE_TRIAL
    #   - 实验语义放在 experiment 字段
    #   - 参数分为 run_manifest(全局) + task.payload(局部)
    #   - SUMMARY 任务由 server 在 core 任务清空后下发
    #
    # 所有任务只写入 pending/task_*.json，执行由 worker 完成。
    # ------------------------------------------------------------------
    shots_per_run = config.run.shots_per_run
    task_count = 0

    def _emit_core_task(task_id: str, experiment: str, run_index: int, payload: Optional[dict] = None) -> None:
        nonlocal task_count
        task = {
            "id": task_id,
            "mode": CORE_TASK_MODE,
            "experiment": experiment,
            "run_index": run_index,
            "shots": shots_per_run,
            "seed": 100000 + task_count + 1,
            "payload": payload or {},
        }
        task_path = pending_dir / f"task_{task_id}.json"
        if not task_path.exists():
            _write_json_atomic(task_path, task)
        task_count += 1

    builder = CORE_TASK_BUILDERS.get(task_type)
    if builder is None:
        raise ValueError(f"不支持的 task_type: {task_type}")
    for entry in builder(config):
        _emit_core_task(
            task_id=str(entry["id"]),
            experiment=str(entry["experiment"]),
            run_index=int(entry["run_index"]),
            payload=entry.get("payload") or {},
        )
    return task_count


def _recover_stale_tasks(paths: dict, stale_seconds: int = WORKER_STALE_RECOVERY_SECS) -> int:
    # ------------------------------------------------------------------
    # 任务回收：
    #   - inprogress 中若超过 stale_seconds 未更新，视为失联
    #   - 回滚到 pending 以便其他 worker 重新领取
    # ------------------------------------------------------------------
    now = time.time()
    recovered = 0
    for task_path in paths["inprogress"].glob("task_*.json"):
        try:
            # 以 mtime 作为“心跳”，超时则回收
            if now - task_path.stat().st_mtime > stale_seconds:
                task_path.replace(paths["pending"] / task_path.name)
                recovered += 1
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return recovered


def _requeue_inprogress_to_pending(paths: dict) -> int:
    moved = 0
    for task_path in sorted(paths["inprogress"].glob("task_*.json")):
        pending_path = paths["pending"] / task_path.name
        try:
            if pending_path.exists():
                task_path.unlink()
            else:
                task_path.replace(pending_path)
            moved += 1
        except FileNotFoundError:
            continue
    return moved


def _worker_heartbeat_process(
    heartbeat_path_str: str,
    task_path_str: str,
    pending_path_str: str,
    stop_event: "mp.synchronize.Event",
    ownership_lost_event: "mp.synchronize.Event",
    interval_secs: int,
) -> None:
    heartbeat_path = Path(heartbeat_path_str)
    task_path = Path(task_path_str)
    pending_path = Path(pending_path_str)
    interval = max(1, int(interval_secs))
    while not stop_event.is_set():
        now = int(time.time())
        try:
            if heartbeat_path.parent.exists():
                heartbeat_path.write_text(str(now), encoding="utf-8")
        except Exception:
            pass
        if pending_path.exists() or (not task_path.exists()):
            ownership_lost_event.set()
            break
        try:
            task_path.touch()
        except FileNotFoundError:
            ownership_lost_event.set()
            break
        except Exception:
            pass
        if stop_event.wait(timeout=float(interval)):
            break


def _run_server_monitor(
    paths: dict,
    done_flag_path: Optional[Path] = None,
    show_progress: bool = True,
    quiet_output_path: Optional[Path] = None,
    quiet_secs: float = 0.0,
    inline: bool = True,
) -> None:
    # ------------------------------------------------------------------
    # server 监控：
    #   - 维护 server_heartbeat
    #   - 回收 stale inprogress
    #   - 定期打印进度：done / inprogress / pending
    # ------------------------------------------------------------------
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    last_report = 0.0
    last_heartbeat = 0.0
    last_done_error_refresh = -1.0
    done_count_cached = 0
    error_count_cached = 0
    stale_recovered_total = 0
    heartbeat_path = paths["summary"] / "server_heartbeat.txt"
    summary_for = _resolve_summary_for(paths)
    start_ts = time.time()
    while True:
        now = time.time()
        if now - last_heartbeat >= 30:
            try:
                # server 心跳：用于“活动判断”和过期回收
                heartbeat_path.write_text(str(int(now)), encoding="utf-8")
            except Exception:
                pass
            last_heartbeat = now
        stale_recovered_total += _recover_stale_tasks(paths)
        pending_count, pending_core_count = _count_total_and_core_tasks(paths["pending"])
        inprogress_count, inprogress_core_count = _count_total_and_core_tasks(paths["inprogress"])
        if pending_core_count == 0 and inprogress_core_count == 0:
            if _enqueue_summary_task_if_needed(paths, summary_for):
                pending_count, pending_core_count = _count_total_and_core_tasks(paths["pending"])

        if (now - last_done_error_refresh >= 30.0) or (pending_count == 0 and inprogress_count == 0):
            done_count_cached = sum(1 for _ in paths["done"].glob("task_*.json"))
            error_count_cached = sum(1 for _ in paths["error"].glob("task_*.json"))
            last_done_error_refresh = now
        done_count = done_count_cached
        error_count = error_count_cached
        quiet_recent = False
        if quiet_output_path is not None and quiet_secs > 0:
            try:
                quiet_recent = (now - quiet_output_path.stat().st_mtime) < quiet_secs
            except FileNotFoundError:
                quiet_recent = False
        if show_progress and (not quiet_recent) and now - last_report >= 5:
            # 总量与 ETA 统一使用实时队列总数，
            # 便于运行中手工删减/新增 pending 任务时进度与 ETA 同步更新。
            total = done_count + pending_count + inprogress_count + error_count
            elapsed = now - start_ts
            eta = "--:--:--"
            if done_count > 0 and total > done_count:
                rate = done_count / max(elapsed, 1e-9)
                eta = _format_duration((total - done_count) / max(rate, 1e-9))
            msg = (
                f"[server] 进度: 已完成 {done_count}/{total} | "
                f"进行中 {inprogress_count} | 待完成 {pending_count} | 失败 {error_count} | "
                f"stale回收 {stale_recovered_total} | 用时 {_format_duration(elapsed)} | ETA {eta}"
            )
            if inline:
                print(f"\r{msg}", end="", flush=True)
            else:
                print(msg, flush=True)
            last_report = now
        if pending_count == 0 and inprogress_count == 0:
            if show_progress:
                if inline:
                    print()
                print(
                    f"[server] 队列已空，结束监控（已完成 {done_count} | 失败 {error_count} | stale回收 {stale_recovered_total}）",
                    flush=True,
                )
            if done_flag_path is not None:
                try:
                    done_flag_path.write_text("done", encoding="utf-8")
                except Exception:
                    pass
            break
        time.sleep(5)


def _run_worker_loop(
    worker_id: int,
    queue_root: str,
    config: SimConfig,
    exit_when_done: bool = False,
    done_flag_path: Optional[str] = None,
    auto_pick: bool = False,
    output_tracker_path: Optional[Path] = None,
) -> None:
    # ------------------------------------------------------------------
    # worker 执行循环：
    #   - 从 pending 抢任务 -> inprogress
    #   - 执行 SIM/HOM/SUMMARY
    #   - 生成 meta.json / raw 输出
    #   - 完成后移入 done
    #
    # auto_pick=True:
    #   - 不指定 run_id 时，自动从最小可用 run_root 中挑任务
    # ------------------------------------------------------------------
    base_root = Path(queue_root)
    paths = None
    host = os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "worker"
    heartbeat_path = None
    done_flag = Path(done_flag_path) if done_flag_path else None
    tracker_installed = False
    backoff = [5, 15, 30, 60, 120]
    backoff_idx = 0
    last_heartbeat = 0.0
    heartbeat_interval_secs = WORKER_HEARTBEAT_INTERVAL_SECS
    seen_task = False
    empty_rounds = 0

    def _sleep_backoff() -> None:
        delay = float(backoff[backoff_idx])
        jitter = float(np.random.random() * min(3.0, max(0.5, 0.1 * delay)))
        time.sleep(delay + jitter)

    while True:
        if not auto_pick and paths is None and exit_when_done:
            root_path = Path(queue_root)
            if not root_path.exists():
                break
        if auto_pick:
            # auto_pick：在多个 run_root 之间挑选可执行的任务
            picked = None
            for run_root in _discover_run_roots(base_root):
                run_paths = _queue_paths(run_root)
                pending_count, _ = _count_total_and_core_tasks(run_paths["pending"])
                if pending_count <= 0:
                    continue
                picked = run_paths
                break
            if picked is None:
                _sleep_backoff()
                backoff_idx = min(backoff_idx + 1, len(backoff) - 1)
                continue
            paths = picked
            _ensure_queue_dirs(paths)
            heartbeat_path = paths["heartbeat"] / f"worker_{host}_{worker_id}.txt"
            if not tracker_installed:
                if output_tracker_path is None:
                    output_tracker_path = paths["heartbeat"] / "worker_output.txt"
                _install_output_tracker(output_tracker_path)
                tracker_installed = True
        elif paths is None:
            # 非 auto_pick：固定使用指定 run_root
            paths = _queue_paths(base_root)
            _ensure_queue_dirs(paths)
            heartbeat_path = paths["heartbeat"] / f"worker_{host}_{worker_id}.txt"
            if not tracker_installed:
                if output_tracker_path is None:
                    output_tracker_path = paths["heartbeat"] / "worker_output.txt"
                _install_output_tracker(output_tracker_path)
                tracker_installed = True

        now = time.time()
        if now - last_heartbeat > heartbeat_interval_secs:
            if heartbeat_path is not None and heartbeat_path.parent.exists():
                heartbeat_path.write_text(str(int(now)), encoding="utf-8")
            last_heartbeat = now
        pending_all = list(paths["pending"].glob("task_*.json"))
        if not pending_all:
            inprogress = list(paths["inprogress"].glob("task_*.json"))
            if exit_when_done:
                # 退出条件：done_flag + 无 inprogress
                if done_flag and done_flag.exists() and not inprogress:
                    break
                # 若曾经领过任务，连续空转 N 轮后退出
                if seen_task and not inprogress:
                    empty_rounds += 1
                    if empty_rounds >= 5:
                        break
                else:
                    empty_rounds = 0
            _sleep_backoff()
            backoff_idx = min(backoff_idx + 1, len(backoff) - 1)
            continue
        pending = pending_all
        if len(pending) > 1:
            start = int(np.random.randint(0, len(pending)))
            pending = pending[start:] + pending[:start]
        empty_rounds = 0
        task_path = None
        for cand in pending:
            dest = paths["inprogress"] / cand.name
            try:
                # 原子“抢占”：rename 成功即视为领取
                cand.replace(dest)
                # pending 任务可能已存在较久；抢占后若不刷新 mtime，
                # 监控线程会按“过期 inprogress”立刻回收，导致任务反复丢失所有权。
                try:
                    dest.touch()
                except Exception:
                    pass
                task_path = dest
                break
            except FileNotFoundError:
                continue
            except PermissionError:
                continue
        if task_path is None:
            _sleep_backoff()
            backoff_idx = min(backoff_idx + 1, len(backoff) - 1)
            continue
        seen_task = True
        backoff_idx = 0

        stop_flag = mp.Event()
        ownership_lost = mp.Event()
        pending_task_path = paths["pending"] / task_path.name

        def _has_task_ownership() -> bool:
            if ownership_lost.is_set():
                return False
            if pending_task_path.exists() or not task_path.exists():
                ownership_lost.set()
                return False
            return True

        def _should_abort_task() -> bool:
            if ownership_lost.is_set():
                return True
            if pending_task_path.exists() or (not task_path.exists()):
                ownership_lost.set()
                return True
            return False

        heartbeat_target = (
            str(heartbeat_path)
            if heartbeat_path is not None
            else str(paths["heartbeat"] / f"worker_{host}_{worker_id}.txt")
        )
        heartbeat_proc = mp.Process(
            target=_worker_heartbeat_process,
            args=(
                heartbeat_target,
                str(task_path),
                str(pending_task_path),
                stop_flag,
                ownership_lost,
                heartbeat_interval_secs,
            ),
            daemon=True,
        )
        heartbeat_proc.start()
        task = {}
        result_dir = None
        status = "ok"
        error_type = ""
        err_msg = ""
        metrics = {}
        task_mode = ""
        task_experiment = ""
        task_id = task_path.stem.replace("task_", "")
        try:
            if not _has_task_ownership():
                raise RuntimeError("OWNERSHIP_LOST")
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task_id = task.get("id", task_path.stem.replace("task_", ""))
            task_mode = str(task.get("mode", "")).upper()
            task_experiment = str(task.get("experiment", "")).upper()

            manifest = _load_run_manifest(paths)
            runtime_cores = int(config.run.cores)
            _apply_manifest_to_config(config, manifest)
            config.run.cores = runtime_cores
            _validate_task_schema(task, manifest)
            if not _has_task_ownership():
                raise RuntimeError("OWNERSHIP_LOST")

            result_dir = paths["results"] / f"result_{task_id}"
            result_dir.mkdir(parents=True, exist_ok=True)
            if task_mode == SUMMARY_TASK_MODE:
                # SUMMARY 任务：集中汇总 CSV
                summary_for = str(task.get("summary_for", "SIM")).upper()
                summary.write_summary(task_type=summary_for, paths=paths, config=config)
                metrics = {"summary_for": summary_for}
                task_experiment = summary_for
                if not _has_task_ownership():
                    raise RuntimeError("OWNERSHIP_LOST")
            elif task_mode == CORE_TASK_MODE:
                plots_dir = result_dir / "plots"
                raw_dir = result_dir / "raw"
                plots_dir.mkdir(parents=True, exist_ok=True)
                raw_dir.mkdir(parents=True, exist_ok=True)
                runner = CORE_TRIAL_TASK_RUNNERS.get(task_experiment)
                if runner is None:
                    raise ValueError(
                        f"SCHEMA_ERROR: 无法分发 CORE_TRIAL，experiment={task_experiment or '缺失'}"
                    )
                metrics = runner(
                    task=task,
                    config=config,
                    raw_dir=raw_dir,
                    plots_dir=plots_dir,
                    task_id=task_id,
                    should_abort=_should_abort_task,
                )
                if not _has_task_ownership():
                    raise RuntimeError("OWNERSHIP_LOST")
            else:
                raise ValueError(
                    f"SCHEMA_ERROR: 无法分发 task，mode={task_mode or '缺失'} "
                    f"experiment={task_experiment or '缺失'}"
                )
        except Exception as exc:
            status = "error"
            err_msg = str(exc)
            if err_msg.startswith("SCHEMA_ERROR:"):
                error_type = "SCHEMA_ERROR"
            elif err_msg.startswith("CONFIG_MISMATCH:"):
                error_type = "CONFIG_MISMATCH"
            else:
                error_type = "RUNTIME_ERROR"
        finally:
            stop_flag.set()
            heartbeat_proc.join(timeout=3)
            if heartbeat_proc.is_alive():
                heartbeat_proc.terminate()
        if ownership_lost.is_set():
            print(
                f"[worker-{worker_id}] 任务 {task_id} 丢失所有权，已放弃本次结果",
                flush=True,
            )
            continue
        try:
            task_id = task.get("id", task_path.stem.replace("task_", ""))
            result_dir = result_dir or (paths["results"] / f"result_{task_id}")
            result_dir.mkdir(parents=True, exist_ok=True)
            meta_path = result_dir / "meta.json"
            meta = {
                "id": task_id,
                "mode": task_mode or task.get("mode", CORE_TASK_MODE),
                "experiment": task_experiment or task.get("experiment", "SIM"),
                "status": status,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "metrics": metrics,
            }
            if err_msg:
                meta["error"] = err_msg
                meta["error_type"] = error_type or "RUNTIME_ERROR"
                meta["error_detail"] = err_msg
            _write_json_atomic(meta_path, meta)
        except Exception as exc:
            print(
                f"[worker-{worker_id}] 写入 meta 失败: task={task_id}, err={exc}",
                file=sys.stderr,
                flush=True,
            )
        try:
            if status == "error":
                error_path = paths["error"] / task_path.name
                task_path.replace(error_path)
            else:
                done_path = paths["done"] / task_path.name
                task_path.replace(done_path)
        except Exception as exc:
            target = "error" if status == "error" else "done"
            print(
                f"[worker-{worker_id}] 任务状态迁移失败: task={task_id}, target={target}, err={exc}",
                file=sys.stderr,
                flush=True,
            )


def main():
    """
    主函数：基于共享目录的 server/worker 调度。
    """
    # ------------------------------------------------------------------
    # 运行角色说明：
    #   server：建队列 + 监控 + 归档
    #   worker：执行任务
    #   both  ：本机同时承担 server+worker
    # ------------------------------------------------------------------
    (
        config,
        role,
        queue_root,
        run_id,
        rebuild_run,
        explicit_cli_dests,
        task_type,
        server_progress,
        progress_quiet_secs,
        progress_inline,
        self_check,
    ) = _parse_run_params(sys.argv)
    # 本机 worker 并发预算（cores）允许在续算时作为调度参数透传。
    cli_worker_cores = int(config.run.cores)
    has_server_capability = role in ("server", "both")
    has_worker_capability = role in ("worker", "both")
    run_monitor_in_background = has_server_capability and has_worker_capability
    if self_check:
        print("[self-check] 运行探测端一致性检查...")
        run_detection_self_checks(verbose=False)
        print("[self-check] 完成")
        return
    task_type = task_type.upper()
    # queue_root 支持相对路径（相对项目根目录）
    base_root = Path(queue_root)
    if not base_root.is_absolute():
        base_root = (PROJECT_ROOT / base_root).resolve()
    outputs_root = PROJECT_ROOT / "outputs"
    resume_existing_run = False
    if has_server_capability:
        base_root.mkdir(parents=True, exist_ok=True)
        outputs_root.mkdir(parents=True, exist_ok=True)
        # 启动前归档旧 run（未完成则加 _u）
        _archive_existing_runs(base_root, outputs_root, exclude_run_id=run_id)
        if run_id is None:
            # 未指定 run-id 则自动分配一个唯一 id（并发安全）
            run_id = _reserve_next_run_id(base_root)
            print(f"[server] 未指定 run-id，自动选择: {run_id}")
    run_root = base_root / run_id if run_id else base_root
    paths = _queue_paths(run_root)
    if has_server_capability:
        if run_root.exists() and any(run_root.iterdir()):
            if rebuild_run:
                archived = _archive_run(run_root, outputs_root, unfinished=True)
                if archived is None:
                    raise SystemExit(f"重建 run 失败，无法先归档已有 run-id: {run_root}")
                resume_existing_run = False
                print(f"[server] 检测到已有 run-id，已归档为未完成并重建: {run_root}")
            else:
                resume_existing_run = True
        _ensure_queue_dirs(paths)
        heartbeat_path = paths["summary"] / "server_heartbeat.txt"
        try:
            heartbeat_path.write_text(str(int(time.time())), encoding="utf-8")
        except Exception:
            pass
        if resume_existing_run:
            print(f"[server] 检测到已有 run-id，进入断点续算: {run_root}")
            recovered = _requeue_inprogress_to_pending(paths)
            if recovered > 0:
                print(f"[server] 已回收 inprogress -> pending: {recovered} 个任务")
            manifest_path = paths["summary"] / RUN_MANIFEST_FILENAME
            if not manifest_path.exists():
                raise SystemExit(f"断点续算失败：缺少 {RUN_MANIFEST_FILENAME}，路径={paths['summary']}")
    elif run_id:
        _ensure_queue_dirs(paths)
    single_run.DEBUG_MODE = config.run.debug

    expected_total = 0
    done_flag = paths["summary"] / "server_done.flag"
    if has_server_capability:
        if done_flag.exists():
            try:
                done_flag.unlink()
            except Exception:
                pass
        if resume_existing_run:
            manifest = _load_run_manifest(paths)
            _apply_manifest_to_config(config, manifest)
            if has_worker_capability and "cores" in explicit_cli_dests:
                config.run.cores = cli_worker_cores
            task_type = str(manifest.get("task_type", task_type)).upper()
            resume_passthrough = set(RESUME_PASSTHROUGH_CLI_DESTS)
            if has_worker_capability:
                resume_passthrough.add("cores")
            ignored_cli = sorted(
                dst
                for dst in explicit_cli_dests
                if dst not in resume_passthrough
            )
            if ignored_cli:
                msg = (
                    "[server] 警告：续算模式下以下任务/物理参数将被自动忽略，"
                    f"统一以 run_manifest 为准: {', '.join(ignored_cli)}；"
                    "如需应用新参数，请使用 --rebuild-run。"
                )
                print(msg)
                try:
                    (paths["summary"] / "resume_ignored_cli_args.txt").write_text(msg + "\n", encoding="utf-8")
                except Exception:
                    pass
            expected_total = sum(
                len(list(paths[section].glob("task_*.json")))
                for section in ("pending", "inprogress", "done", "error")
            )
        else:
            # 先写 manifest，再生成任务列表（避免中断时缺少 run_manifest）
            _write_run_manifest(paths, task_type, config)
            core_task_total = _build_task_list(task_type, config, paths["pending"])
            expected_total = core_task_total + 1
        print(f"[server] 任务总数: {expected_total} | queue: {paths['root']}")
        if not has_worker_capability:
            _run_server_monitor(
                paths,
                done_flag,
                show_progress=server_progress,
                quiet_output_path=None,
                quiet_secs=0.0,
                inline=progress_inline,
            )

    if has_worker_capability:
        # 核数预算：留 1 核给系统
        core_budget = max(1, min(config.run.cores, os.cpu_count() or 1))
        reserve = 1
        effective_budget = max(1, core_budget - reserve)
        if run_id:
            pending_total = len(list(paths["pending"].glob("task_*.json")))
        else:
            pending_total = 0
            for run_root in _discover_run_roots(base_root):
                run_paths = _queue_paths(run_root)
                pending_total += len(list(run_paths["pending"].glob("task_*.json")))
                if pending_total >= effective_budget:
                    break
        if pending_total > 0:
            worker_count = min(effective_budget, pending_total)
        else:
            worker_count = effective_budget
        if worker_count > 1:
            # 防止 BLAS 内部线程叠加导致过度并发
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
        queue_exec_root = run_root if run_id else base_root
        queue_hint = str(paths["root"]) if run_id else str(base_root)
        print(f"[worker] cores={core_budget} | workers={worker_count} | queue={queue_hint}")

        if run_monitor_in_background:
            output_tracker_path = paths["heartbeat"] / "worker_output.txt"
            monitor_thread = threading.Thread(
                target=_run_server_monitor,
                args=(
                    paths,
                    done_flag,
                    server_progress,
                    output_tracker_path,
                    progress_quiet_secs,
                    progress_inline,
                ),
            )
            monitor_thread.start()

        exit_when_done = has_worker_capability
        done_flag_arg = str(done_flag) if run_id is not None else None
        auto_pick = run_id is None
        tracker_path = None if run_id is None else (paths["heartbeat"] / "worker_output.txt")
        if worker_count == 1:
            _run_worker_loop(
                1,
                str(queue_exec_root),
                config,
                exit_when_done,
                done_flag_arg,
                auto_pick,
                tracker_path,
            )
        else:
            # 多进程并行：每个进程独立抢任务
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = []
                for idx in range(worker_count):
                    futures.append(
                        executor.submit(
                            _run_worker_loop,
                            idx + 1,
                            str(queue_exec_root),
                            config,
                            exit_when_done,
                            done_flag_arg,
                            auto_pick,
                            tracker_path,
                        )
                    )
                for future in futures:
                    future.result()
        if run_monitor_in_background:
            monitor_thread.join()

    if has_server_capability:
        _archive_run(
            run_root,
            outputs_root,
            unfinished=(
                any(paths["pending"].glob("task_*.json"))
                or any(paths["inprogress"].glob("task_*.json"))
            ),
        )


if __name__ == "__main__":
    main()
