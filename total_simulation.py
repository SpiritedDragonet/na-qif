# -*- coding: utf-8 -*-
"""
CLI 入口与任务调度（单次实验逻辑见 atom_sim.experiment.single_run）。
"""

import sys
import os
import json
import argparse
import threading
import time
import re
import shutil
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
    _build_hom_tau_values,
    _run_hom_run,
)
from atom_sim.experiment import single_run, window_scan, length_scan, bsm_scan, summary  # noqa: E402
from atom_sim.simulation import run_detection_self_checks  # noqa: E402


TASK_PROTOCOL_VERSION = "v2_core_trial"
RUN_MANIFEST_FILENAME = "run_manifest.json"
CORE_TASK_MODE = "CORE_TRIAL"
SUMMARY_TASK_MODE = "SUMMARY"
SUPPORTED_EXPERIMENTS = {"SIM", "HOM", "WINDOW_SCAN", "BSM_SCAN", "LENGTH_SCAN"}


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
            # 只在 marker 目录已存在时更新心跳，
            # 避免 run 已归档后再次 print 把旧 queue/<run>/heartbeat 重建出来。
            if marker_path.parent.exists():
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
    #   - queue_root/run_id: 决定任务目录隔离与任务回收策略
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
            "  # 抢占式节点（worker）\n"
            "  python total_simulation.py --role worker --queue-root /mnt/quantum_sim/queue --run-id homA --cores 64\n"
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
    parser.add_argument("--run-id", dest="run_id", type=str, help="运行ID（用于隔离多次任务；未提供则自动选择最小可用ID）")
    parser.add_argument("--task-type", dest="task_type", type=str, choices=["SIM", "HOM", "WINDOW_SCAN", "LENGTH_SCAN", "BSM_SCAN"], help="任务类型：SIM / HOM / WINDOW_SCAN / LENGTH_SCAN / BSM_SCAN（默认随 --mode）")
    parser.add_argument("--config-hash", dest="config_hash", type=str, help="任务配置版本标识（默认自动读取 git）")
    parser.add_argument("--runs", "--n-runs", dest="n_runs", type=int, help="仿真 run 次数（默认 1）")
    parser.add_argument("--shots", "--shots-per-run", dest="shots_per_run", type=int, help="每个 run 的探测采样次数（默认 1）")
    parser.add_argument(
        "--cores",
        dest="cores",
        type=int,
        help="可用 CPU 核数预算（默认 1，程序会自动计算实际并发进程数）",
    )
    parser.add_argument("--mode", "--trial-type", dest="mode", help="运行模式：SIM 或 HOM（默认 SIM）")
    parser.add_argument("--no-fiber-noise", dest="no_fiber_noise", action="store_true", help="关闭光纤噪声")
    parser.add_argument("--fiber-length-km", dest="fiber_length_km", type=float, help="光纤长度 (km)，用于计算平均透过率")
    parser.add_argument("--fiber-atten-db-per-km", dest="fiber_atten_db_per_km", type=float, help="光纤衰减 (dB/km)")
    parser.add_argument("--fiber-eta-std", dest="fiber_eta_std", type=float, help="透过率随机波动标准差")
    parser.add_argument("--fiber-pdl-sigma", dest="fiber_pdl_sigma", type=float, help="PDL 相对差异标准差")
    parser.add_argument("--fiber-phase-drift-std", dest="fiber_phase_drift_std", type=float, help="两臂相位漂移标准差 (rad)")
    parser.add_argument("--fiber-phase-slope-std", dest="fiber_phase_slope_std", type=float, help="相位斜率标准差 (rad/bin)")
    parser.add_argument("--fiber-phase-jitter-std", dest="fiber_phase_jitter_std", type=float, help="单bin相位抖动标准差 (rad)")
    parser.add_argument("--fiber-polarization-model", dest="fiber_polarization_model", choices=["fixed", "haar", "perturb", "euler"], help="光纤偏振模型")
    parser.add_argument("--fiber-polarization-sigma", dest="fiber_polarization_sigma", type=float, help="偏振小扰动模型标准差 (rad)")
    parser.add_argument("--fiber-group-velocity-mps", dest="fiber_group_velocity_mps", type=float, help="光纤群速度 (m/s)，用于自动计算 t_wait_us")
    parser.add_argument("--t-wait-overhead-us", dest="t_wait_overhead_us", type=float, help="等待时间固定开销 (us)，会加到 L/v_g 上")
    parser.add_argument("--t-wait-length-scale", dest="t_wait_length_scale", type=float, help="等待时间线性系数：T_wait = scale*L/v_g + overhead")
    parser.add_argument("--t2-us", dest="t2_us", type=float, help="原子退相干时间 T2 (us)")

    parser.add_argument("--tau", dest="tau", type=float, help="(HOM) 单一延迟 τ (ns)")
    parser.add_argument("--tau-start", dest="tau_start", type=float, help="(HOM) τ 起点 (ns)")
    parser.add_argument("--tau-end", dest="tau_end", type=float, help="(HOM) τ 终点 (ns)")
    parser.add_argument("--tau-step", dest="tau_step", type=float, help="(HOM) τ 步长 (ns)")
    parser.add_argument("--tau-points", dest="tau_points", type=int, help="(HOM) τ 采样点数")
    parser.add_argument("--window-ns", dest="window_ns", type=float, help="(HOM) 符合窗口 (ns)")
    parser.add_argument("--window-sweep-start-ns", dest="window_sweep_start_ns", type=float, help="(WINDOW_SCAN) 扫描起点窗口 (ns)")
    parser.add_argument("--window-sweep-end-ns", dest="window_sweep_end_ns", type=float, help="(WINDOW_SCAN) 扫描终点窗口 (ns)")
    parser.add_argument("--window-sweep-step-ns", dest="window_sweep_step_ns", type=float, help="(WINDOW_SCAN) 扫描步长窗口 (ns)")
    parser.add_argument("--bs-sweep-start-theta", dest="bs_sweep_start_theta", type=float, help="(BSM_SCAN) 扫描起点 BS theta (rad)")
    parser.add_argument("--bs-sweep-end-theta", dest="bs_sweep_end_theta", type=float, help="(BSM_SCAN) 扫描终点 BS theta (rad)")
    parser.add_argument("--bs-sweep-step-theta", dest="bs_sweep_step_theta", type=float, help="(BSM_SCAN) 扫描步长 BS theta (rad)")
    parser.add_argument("--length-sweep-start-km", dest="length_sweep_start_km", type=float, help="(LENGTH_SCAN) 扫描起点长度 (km)")
    parser.add_argument("--length-sweep-end-km", dest="length_sweep_end_km", type=float, help="(LENGTH_SCAN) 扫描终点长度 (km)")
    parser.add_argument("--length-sweep-step-km", dest="length_sweep_step_km", type=float, help="(LENGTH_SCAN) 扫描步长长度 (km)")
    parser.add_argument("--attempt-rate-hz", dest="attempt_rate_hz", type=float, help="(LENGTH_SCAN) 基础尝试率 (Hz)")
    parser.add_argument("--attempt-overhead-us", dest="attempt_overhead_us", type=float, help="(LENGTH_SCAN) 单次额外时延 (us)")

    parser.add_argument("--dark-hz", dest="dark_rate_intrinsic_hz", type=float, help="探测器本底暗计数率 (Hz)")
    parser.add_argument("--dark-hz-h1", dest="dark_hz_h1", type=float, help="H1 通道本底暗计数率 (Hz)；未指定则用 --dark-hz")
    parser.add_argument("--dark-hz-v1", dest="dark_hz_v1", type=float, help="V1 通道本底暗计数率 (Hz)；未指定则用 --dark-hz")
    parser.add_argument("--dark-hz-h2", dest="dark_hz_h2", type=float, help="H2 通道本底暗计数率 (Hz)；未指定则用 --dark-hz")
    parser.add_argument("--dark-hz-v2", dest="dark_hz_v2", type=float, help="V2 通道本底暗计数率 (Hz)；未指定则用 --dark-hz")
    parser.add_argument("--bg-mean-hz", dest="bg_rate_mean_hz", type=float, help="背景噪声均值 (Hz)")
    parser.add_argument("--bg-mean-hz-h1", dest="bg_mean_hz_h1", type=float, help="H1 通道背景噪声均值 (Hz)；未指定则用 --bg-mean-hz")
    parser.add_argument("--bg-mean-hz-v1", dest="bg_mean_hz_v1", type=float, help="V1 通道背景噪声均值 (Hz)；未指定则用 --bg-mean-hz")
    parser.add_argument("--bg-mean-hz-h2", dest="bg_mean_hz_h2", type=float, help="H2 通道背景噪声均值 (Hz)；未指定则用 --bg-mean-hz")
    parser.add_argument("--bg-mean-hz-v2", dest="bg_mean_hz_v2", type=float, help="V2 通道背景噪声均值 (Hz)；未指定则用 --bg-mean-hz")
    parser.add_argument("--bg-std-hz", dest="bg_rate_std_hz", type=float, help="背景噪声标准差 (Hz)")
    parser.add_argument("--bg-std-hz-h1", dest="bg_std_hz_h1", type=float, help="H1 通道背景噪声标准差 (Hz)；未指定则用 --bg-std-hz")
    parser.add_argument("--bg-std-hz-v1", dest="bg_std_hz_v1", type=float, help="V1 通道背景噪声标准差 (Hz)；未指定则用 --bg-std-hz")
    parser.add_argument("--bg-std-hz-h2", dest="bg_std_hz_h2", type=float, help="H2 通道背景噪声标准差 (Hz)；未指定则用 --bg-std-hz")
    parser.add_argument("--bg-std-hz-v2", dest="bg_std_hz_v2", type=float, help="V2 通道背景噪声标准差 (Hz)；未指定则用 --bg-std-hz")
    parser.add_argument("--detector-gate-ns", dest="detector_gate_ns", type=float, help="探测门宽 (ns)，用于将噪声概率从门宽映射到仿真 bin")
    parser.add_argument("--omega-peak-a", dest="omega_peak_a", type=float, help="A 臂驱动脉冲峰值 Ω_peak_A (rad/s)")
    parser.add_argument("--omega-peak-b", dest="omega_peak_b", type=float, help="B 臂驱动脉冲峰值 Ω_peak_B (rad/s)")
    parser.add_argument("--drive-waveform-a", dest="drive_waveform_a", choices=["gaussian", "sech", "square"], help="A 臂驱动包络类型")
    parser.add_argument("--drive-waveform-b", dest="drive_waveform_b", choices=["gaussian", "sech", "square"], help="B 臂驱动包络类型")
    parser.add_argument("--g-a", dest="g_a", type=float, help="A 臂原子-腔耦合强度 g_A (rad/s)")
    parser.add_argument("--g-b", dest="g_b", type=float, help="B 臂原子-腔耦合强度 g_B (rad/s)")
    parser.add_argument("--kappa-ex-a", dest="kappa_ex_a", type=float, help="A 臂腔外耦合衰减率 kappa_ex_A (rad/s)")
    parser.add_argument("--kappa-ex-b", dest="kappa_ex_b", type=float, help="B 臂腔外耦合衰减率 kappa_ex_B (rad/s)")
    parser.add_argument("--kappa-in-a", dest="kappa_in_a", type=float, help="A 臂腔内损耗衰减率 kappa_in_A (rad/s)")
    parser.add_argument("--kappa-in-b", dest="kappa_in_b", type=float, help="B 臂腔内损耗衰减率 kappa_in_B (rad/s)")
    parser.add_argument("--kappa-ex-h-a", dest="kappa_ex_h_a", type=float, help="A 臂 H 偏振外耦合衰减率 kappa_ex_H_A (rad/s)")
    parser.add_argument("--kappa-ex-v-a", dest="kappa_ex_v_a", type=float, help="A 臂 V 偏振外耦合衰减率 kappa_ex_V_A (rad/s)")
    parser.add_argument("--kappa-in-h-a", dest="kappa_in_h_a", type=float, help="A 臂 H 偏振内损耗衰减率 kappa_in_H_A (rad/s)")
    parser.add_argument("--kappa-in-v-a", dest="kappa_in_v_a", type=float, help="A 臂 V 偏振内损耗衰减率 kappa_in_V_A (rad/s)")
    parser.add_argument("--kappa-ex-h-b", dest="kappa_ex_h_b", type=float, help="B 臂 H 偏振外耦合衰减率 kappa_ex_H_B (rad/s)")
    parser.add_argument("--kappa-ex-v-b", dest="kappa_ex_v_b", type=float, help="B 臂 V 偏振外耦合衰减率 kappa_ex_V_B (rad/s)")
    parser.add_argument("--kappa-in-h-b", dest="kappa_in_h_b", type=float, help="B 臂 H 偏振内损耗衰减率 kappa_in_H_B (rad/s)")
    parser.add_argument("--kappa-in-v-b", dest="kappa_in_v_b", type=float, help="B 臂 V 偏振内损耗衰减率 kappa_in_V_B (rad/s)")
    parser.add_argument("--delta-u-a", dest="delta_u_a", type=float, help="A 臂 |u> 态失谐 delta_u_A (rad/s)")
    parser.add_argument("--delta-u-b", dest="delta_u_b", type=float, help="B 臂 |u> 态失谐 delta_u_B (rad/s)")
    parser.add_argument("--delta-e-a", dest="delta_e_a", type=float, help="A 臂 |e> 态失谐 delta_e_A (rad/s)")
    parser.add_argument("--delta-e-b", dest="delta_e_b", type=float, help="B 臂 |e> 态失谐 delta_e_B (rad/s)")
    parser.add_argument("--gamma-sigma-plus-a", dest="gamma_sigma_plus_a", type=float, help="A 臂 |e>->|0> 自发辐射率 gamma_sigma_plus_A (1/s)")
    parser.add_argument("--gamma-sigma-minus-a", dest="gamma_sigma_minus_a", type=float, help="A 臂 |e>->|1> 自发辐射率 gamma_sigma_minus_A (1/s)")
    parser.add_argument("--gamma-sigma-plus-b", dest="gamma_sigma_plus_b", type=float, help="B 臂 |e>->|0> 自发辐射率 gamma_sigma_plus_B (1/s)")
    parser.add_argument("--gamma-sigma-minus-b", dest="gamma_sigma_minus_b", type=float, help="B 臂 |e>->|1> 自发辐射率 gamma_sigma_minus_B (1/s)")
    parser.add_argument("--delta-c-h-a", dest="delta_c_h_a", type=float, help="A 臂 H 腔模失谐 delta_c_H_A (rad/s)")
    parser.add_argument("--delta-c-v-a", dest="delta_c_v_a", type=float, help="A 臂 V 腔模失谐 delta_c_V_A (rad/s)")
    parser.add_argument("--delta-c-h-b", dest="delta_c_h_b", type=float, help="B 臂 H 腔模失谐 delta_c_H_B (rad/s)")
    parser.add_argument("--delta-c-v-b", dest="delta_c_v_b", type=float, help="B 臂 V 腔模失谐 delta_c_V_B (rad/s)")
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
    parser.add_argument("--no-filter-cavity", dest="no_filter_cavity", action="store_true", help="关闭QFC后1517滤波腔显式记忆")
    parser.add_argument("--enum-mode", dest="enum_mode", type=str, help="成功事件枚举模式：dark/no-dark/both")
    parser.add_argument("--plot-all", dest="plot_all", action="store_true", help="所有 run 都绘图（默认仅保留一个）")
    parser.add_argument("--no-plot", dest="no_plot", action="store_true", help="完全禁止绘图（覆盖 plot-all）")
    parser.add_argument("--eta-det", dest="eta_det", type=float, help="探测效率 η (0~1)")
    parser.add_argument("--eta-det-h1", dest="eta_det_h1", type=float, help="H1 通道探测效率 η (0~1)；未指定则用 --eta-det")
    parser.add_argument("--eta-det-v1", dest="eta_det_v1", type=float, help="V1 通道探测效率 η (0~1)；未指定则用 --eta-det")
    parser.add_argument("--eta-det-h2", dest="eta_det_h2", type=float, help="H2 通道探测效率 η (0~1)；未指定则用 --eta-det")
    parser.add_argument("--eta-det-v2", dest="eta_det_v2", type=float, help="V2 通道探测效率 η (0~1)；未指定则用 --eta-det")
    parser.add_argument("--ideal-det", dest="ideal_det", action="store_true", help="理想探测（eta_det=1, 无噪声）")
    parser.add_argument("--bs-theta", dest="bs_theta", type=float, help="中心站 BS 混合角 theta (rad)，sin^2(theta) 为跨端口透射概率")
    parser.add_argument("--v-res", dest="v_res", type=float, help="残差可区分度 V_res (0~1)")
    parser.add_argument("--debug", dest="debug", action="store_true", help="开启调试模式（输出耗时等）")
    parser.add_argument("--self-check", dest="self_check", action="store_true", help="仅运行探测端自检并退出")
    args = parser.parse_args(argv[1:])

    # run-id 仅用于目录命名，因此要严格限制为“非路径字符串”
    run_id = args.run_id.strip() if args.run_id else None
    if run_id:
        if "/" in run_id or "\\" in run_id:
            parser.error("run-id 不能包含路径分隔符")
        if run_id in (".", ".."):
            parser.error("run-id 不能为 . 或 ..")

    # 构造默认配置，然后用 CLI 覆盖。注意：SimConfig 是“可序列化的仿真输入快照”。
    config = SimConfig()

    if args.dark_rate_intrinsic_hz is not None:
        config.noise.dark_rate_intrinsic_hz = args.dark_rate_intrinsic_hz
    if args.bg_rate_mean_hz is not None:
        config.noise.bg_rate_mean_hz = args.bg_rate_mean_hz
    if args.bg_rate_std_hz is not None:
        config.noise.bg_rate_std_hz = args.bg_rate_std_hz
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
    if args.detector_gate_ns is not None:
        config.noise.detector_gate_ns = float(args.detector_gate_ns)
    if args.omega_peak_a is not None:
        config.emission.arm_A.omega_peak = float(args.omega_peak_a)
    if args.omega_peak_b is not None:
        config.emission.arm_B.omega_peak = float(args.omega_peak_b)
    if args.drive_waveform_a is not None:
        config.emission.drive_waveform_A = str(args.drive_waveform_a).strip().lower()
    if args.drive_waveform_b is not None:
        config.emission.drive_waveform_B = str(args.drive_waveform_b).strip().lower()
    if args.g_a is not None:
        config.emission.arm_A.g = float(args.g_a)
    if args.g_b is not None:
        config.emission.arm_B.g = float(args.g_b)
    if args.kappa_ex_a is not None:
        config.emission.arm_A.kappa_ex = float(args.kappa_ex_a)
    if args.kappa_ex_b is not None:
        config.emission.arm_B.kappa_ex = float(args.kappa_ex_b)
    if args.kappa_in_a is not None:
        config.emission.arm_A.kappa_in = float(args.kappa_in_a)
    if args.kappa_in_b is not None:
        config.emission.arm_B.kappa_in = float(args.kappa_in_b)
    if args.kappa_ex_h_a is not None:
        config.emission.arm_A.kappa_ex_H = float(args.kappa_ex_h_a)
    if args.kappa_ex_v_a is not None:
        config.emission.arm_A.kappa_ex_V = float(args.kappa_ex_v_a)
    if args.kappa_in_h_a is not None:
        config.emission.arm_A.kappa_in_H = float(args.kappa_in_h_a)
    if args.kappa_in_v_a is not None:
        config.emission.arm_A.kappa_in_V = float(args.kappa_in_v_a)
    if args.kappa_ex_h_b is not None:
        config.emission.arm_B.kappa_ex_H = float(args.kappa_ex_h_b)
    if args.kappa_ex_v_b is not None:
        config.emission.arm_B.kappa_ex_V = float(args.kappa_ex_v_b)
    if args.kappa_in_h_b is not None:
        config.emission.arm_B.kappa_in_H = float(args.kappa_in_h_b)
    if args.kappa_in_v_b is not None:
        config.emission.arm_B.kappa_in_V = float(args.kappa_in_v_b)
    if args.delta_u_a is not None:
        config.emission.arm_A.delta_u = float(args.delta_u_a)
    if args.delta_u_b is not None:
        config.emission.arm_B.delta_u = float(args.delta_u_b)
    if args.delta_e_a is not None:
        config.emission.arm_A.delta_e = float(args.delta_e_a)
    if args.delta_e_b is not None:
        config.emission.arm_B.delta_e = float(args.delta_e_b)
    if args.gamma_sigma_plus_a is not None:
        config.emission.arm_A.gamma_sigma_plus = float(args.gamma_sigma_plus_a)
    if args.gamma_sigma_minus_a is not None:
        config.emission.arm_A.gamma_sigma_minus = float(args.gamma_sigma_minus_a)
    if args.gamma_sigma_plus_b is not None:
        config.emission.arm_B.gamma_sigma_plus = float(args.gamma_sigma_plus_b)
    if args.gamma_sigma_minus_b is not None:
        config.emission.arm_B.gamma_sigma_minus = float(args.gamma_sigma_minus_b)
    if args.delta_c_h_a is not None:
        config.emission.arm_A.delta_c_H = float(args.delta_c_h_a)
    if args.delta_c_v_a is not None:
        config.emission.arm_A.delta_c_V = float(args.delta_c_v_a)
    if args.delta_c_h_b is not None:
        config.emission.arm_B.delta_c_H = float(args.delta_c_h_b)
    if args.delta_c_v_b is not None:
        config.emission.arm_B.delta_c_V = float(args.delta_c_v_b)
    if args.alpha_h_plus_a is not None:
        config.emission.arm_A.alpha_h_plus = float(args.alpha_h_plus_a)
    if args.alpha_h_minus_a is not None:
        config.emission.arm_A.alpha_h_minus = float(args.alpha_h_minus_a)
    if args.alpha_v_plus_a is not None:
        config.emission.arm_A.alpha_v_plus = float(args.alpha_v_plus_a)
    if args.alpha_v_minus_a is not None:
        config.emission.arm_A.alpha_v_minus = float(args.alpha_v_minus_a)
    if args.alpha_h_plus_b is not None:
        config.emission.arm_B.alpha_h_plus = float(args.alpha_h_plus_b)
    if args.alpha_h_minus_b is not None:
        config.emission.arm_B.alpha_h_minus = float(args.alpha_h_minus_b)
    if args.alpha_v_plus_b is not None:
        config.emission.arm_B.alpha_v_plus = float(args.alpha_v_plus_b)
    if args.alpha_v_minus_b is not None:
        config.emission.arm_B.alpha_v_minus = float(args.alpha_v_minus_b)
    if args.qfc_theta_h is not None:
        config.qfc.theta_H = float(args.qfc_theta_h)
    if args.qfc_theta_v is not None:
        config.qfc.theta_V = float(args.qfc_theta_v)
    if args.qfc_phi_h is not None:
        config.qfc.phi_H = float(args.qfc_phi_h)
    if args.qfc_phi_v is not None:
        config.qfc.phi_V = float(args.qfc_phi_v)
    if args.filter_cavity_fwhm_mhz is not None:
        config.qfc.filter_cavity.fwhm_mhz = float(args.filter_cavity_fwhm_mhz)
    if args.filter_cavity_detuning_mhz_a is not None:
        config.qfc.filter_cavity.detuning_mhz_A = float(args.filter_cavity_detuning_mhz_a)
    if args.filter_cavity_detuning_mhz_b is not None:
        config.qfc.filter_cavity.detuning_mhz_B = float(args.filter_cavity_detuning_mhz_b)
    if args.filter_cavity_eta_peak_a is not None:
        config.qfc.filter_cavity.eta_peak_A = float(args.filter_cavity_eta_peak_a)
    if args.filter_cavity_eta_peak_b is not None:
        config.qfc.filter_cavity.eta_peak_B = float(args.filter_cavity_eta_peak_b)
    if args.qfc_noise_sd_cps_per_mhz_a is not None:
        config.qfc.qfc_noise_sd_cps_per_mhz_A = float(args.qfc_noise_sd_cps_per_mhz_a)
    if args.qfc_noise_sd_cps_per_mhz_b is not None:
        config.qfc.qfc_noise_sd_cps_per_mhz_B = float(args.qfc_noise_sd_cps_per_mhz_b)
    if args.no_filter_cavity:
        config.qfc.filter_cavity.enabled = False

    # runs/shots/cores 是“任务粒度 + 并发预算”的核心参数
    config.run.runs = args.n_runs if args.n_runs is not None else config.run.runs
    config.run.shots_per_run = (
        args.shots_per_run if args.shots_per_run is not None else config.run.shots_per_run
    )
    config.run.cores = args.cores if args.cores is not None else config.run.cores
    # mode 与 task_type 的一致性约束：
    #   - mode 代表“当前命令意图”，task_type 代表“写入任务的类型”
    #   - 若两者同时给出，必须完全一致，否则会造成任务与执行逻辑错配
    mode = (args.mode or config.mode).upper()
    task_type = (args.task_type or mode).upper()
    if args.task_type is not None and args.mode is not None and task_type != mode:
        parser.error("task-type 与 mode 冲突，请保持一致")
    config.mode = task_type
    if task_type == "WINDOW_SCAN":
        config.run.window_sweep_start_ns = args.window_sweep_start_ns
        config.run.window_sweep_end_ns = args.window_sweep_end_ns
        config.run.window_sweep_step_ns = args.window_sweep_step_ns
    if task_type == "BSM_SCAN":
        config.run.bs_sweep_start_theta = args.bs_sweep_start_theta
        config.run.bs_sweep_end_theta = args.bs_sweep_end_theta
        config.run.bs_sweep_step_theta = args.bs_sweep_step_theta
    if task_type == "LENGTH_SCAN":
        config.run.length_sweep_start_km = args.length_sweep_start_km
        config.run.length_sweep_end_km = args.length_sweep_end_km
        config.run.length_sweep_step_km = args.length_sweep_step_km
        if args.attempt_rate_hz is not None:
            config.run.attempt_rate_hz = float(args.attempt_rate_hz)
        if args.attempt_overhead_us is not None:
            config.run.attempt_overhead_us = float(args.attempt_overhead_us)
    # 光纤噪声开关（注意：这会影响到统计与物理可解释性）
    config.fiber.noise_enabled = not args.no_fiber_noise
    if args.fiber_length_km is not None:
        config.fiber.length_km = float(args.fiber_length_km)
    if args.fiber_atten_db_per_km is not None:
        config.fiber.attenuation_db_per_km = float(args.fiber_atten_db_per_km)
    if args.fiber_eta_std is not None:
        config.fiber.eta_std = float(args.fiber_eta_std)
    if args.fiber_pdl_sigma is not None:
        config.fiber.pdl_sigma = float(args.fiber_pdl_sigma)
    if args.fiber_phase_drift_std is not None:
        config.fiber.phase_drift_std = float(args.fiber_phase_drift_std)
    if args.fiber_phase_slope_std is not None:
        config.fiber.phase_slope_std = float(args.fiber_phase_slope_std)
    if args.fiber_phase_jitter_std is not None:
        config.fiber.phase_jitter_std = float(args.fiber_phase_jitter_std)
    if args.fiber_polarization_model is not None:
        config.fiber.polarization_model = str(args.fiber_polarization_model)
    if args.fiber_polarization_sigma is not None:
        config.fiber.polarization_sigma = float(args.fiber_polarization_sigma)
    if args.fiber_group_velocity_mps is not None:
        config.run.fiber_group_velocity_mps = float(args.fiber_group_velocity_mps)
    if args.t_wait_overhead_us is not None:
        config.run.t_wait_overhead_us = float(args.t_wait_overhead_us)
    if args.t_wait_length_scale is not None:
        config.run.t_wait_length_scale = float(args.t_wait_length_scale)
    if args.t2_us is not None:
        config.run.t2_us = float(args.t2_us)

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
    if args.eta_det is not None:
        config.detector.eta_det = float(args.eta_det)
    for detector in ("h1", "v1", "h2", "v2"):
        eta_value = getattr(args, f"eta_det_{detector}", None)
        if eta_value is not None:
            config.detector.eta_det_map[detector.upper()] = float(eta_value)
    config.detector.ideal_det = bool(args.ideal_det)
    if config.detector.ideal_det:
        config.detector.eta_det = 1.0
    if args.bs_theta is not None:
        config.detector.bs_theta = float(args.bs_theta)
    if args.v_res is not None:
        config.detector.v_res = float(args.v_res)
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

    if task_type == "WINDOW_SCAN":
        try:
            window_scan.validate_window_scan_config(config)
        except ValueError as exc:
            parser.error(str(exc))
    if task_type == "BSM_SCAN":
        try:
            bsm_scan.validate_bsm_scan_config(config)
        except ValueError as exc:
            parser.error(str(exc))
    if task_type == "LENGTH_SCAN":
        try:
            length_scan.validate_length_scan_config(config)
        except ValueError as exc:
            parser.error(str(exc))

    config.run.plot_all = bool(args.plot_all)
    config.run.plot_enabled = not bool(args.no_plot)
    config.run.debug = bool(args.debug)
    if args.server_progress is None:
        server_progress = True
    else:
        server_progress = bool(args.server_progress)
    if args.server_progress_quiet_secs is None:
        progress_quiet_secs = 20.0 if args.role == "both" else 0.0
    else:
        progress_quiet_secs = max(0.0, float(args.server_progress_quiet_secs))
    progress_inline = args.role == "server"
    return (
        config,
        args.role,
        args.queue_root,
        run_id,
        task_type,
        args.config_hash,
        server_progress,
        progress_quiet_secs,
        progress_inline,
        bool(args.self_check),
    )


def _resolve_config_hash(explicit: Optional[str]) -> str:
    # 目的：给任务打一个“配置版本号”标签，便于结果可追溯。
    # 优先级：显式传入 > git HEAD > 简化字符串。
    if explicit:
        return explicit
    git_dir = PROJECT_ROOT / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return "git:unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split("ref:", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            return f"git:{ref_path.read_text(encoding='utf-8').strip()}"
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                sha, name = line.split(" ", 1)
                if name.strip() == ref:
                    return f"git:{sha.strip()}"
    return f"git:{head[:12]}"


def _resolve_queue_root(path_str: str) -> Path:
    # 规则：相对路径一律解释为“项目根目录下的 queue 子树”
    path = Path(path_str)
    if not path.is_absolute():
        return (PROJECT_ROOT / path).resolve()
    return path


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


def _run_id_sort_key(run_id: str) -> tuple:
    # 规则：纯数字 run_id 优先排序，其次按前缀数字排序，再按字典序
    if run_id.isdigit():
        return (0, int(run_id), run_id)
    m = re.match(r"(\d+)", run_id)
    if m:
        return (0, int(m.group(1)), run_id)
    return (1, run_id)


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
    return sorted(roots, key=lambda p: _run_id_sort_key(p.name))


def _is_summary_task_path(path: Path) -> bool:
    # summary 任务是整个 run 的“最终汇总步骤”
    return path.name == "task_summary.json"


def _is_run_complete(run_root: Path) -> bool:
    # 完成判定：pending/inprogress 均空 + server_done.flag 存在
    pending = run_root / "tasks" / "pending"
    inprogress = run_root / "tasks" / "inprogress"
    done_flag = run_root / "summary" / "server_done.flag"
    if not pending.exists() or not inprogress.exists():
        return False
    if not done_flag.exists():
        return False
    return (not any(pending.glob("task_*.json"))) and (not any(inprogress.glob("task_*.json")))


def _is_run_active(run_root: Path, stale_seconds: int = 600) -> bool:
    # 活跃判定：server heartbeat 或 inprogress 文件近期被 touch
    now = time.time()
    heartbeat = run_root / "summary" / "server_heartbeat.txt"
    if heartbeat.exists():
        try:
            if now - heartbeat.stat().st_mtime <= stale_seconds:
                return True
        except FileNotFoundError:
            pass
    inprogress_dir = run_root / "tasks" / "inprogress"
    if inprogress_dir.exists():
        for task_path in inprogress_dir.glob("task_*.json"):
            try:
                if now - task_path.stat().st_mtime <= stale_seconds:
                    return True
            except FileNotFoundError:
                continue
    return False


def _pick_output_root(outputs_root: Path, name: str) -> Path:
    # 归档命名冲突处理：若同名存在，则追加 _1, _2, ...
    dest = outputs_root / name
    if not dest.exists():
        return dest
    suffix = 1
    while (outputs_root / f"{name}_{suffix}").exists():
        suffix += 1
    return outputs_root / f"{name}_{suffix}"


def _archive_run(run_root: Path, outputs_root: Path, unfinished: bool = False) -> Optional[Path]:
    # 将一个 run_root 目录整体移动到 outputs/<timestamp>（未完成则加 _u）
    if not run_root.exists():
        return None
    outputs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{stamp}_u" if unfinished else stamp
    dest = _pick_output_root(outputs_root, name)
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
    stale_seconds: int = 600,
) -> None:
    # 扫描所有 run 根目录：
    #   - 已完成 -> 直接归档
    #   - 活跃 -> 保留
    #   - 非活跃 -> 归档为未完成（_u）
    for run_root in _discover_run_roots(base_root):
        if exclude_run_id and run_root.name == exclude_run_id:
            continue
        if _is_run_complete(run_root):
            _archive_run(run_root, outputs_root, unfinished=False)
            continue
        if _is_run_active(run_root, stale_seconds=stale_seconds):
            continue
        _archive_run(run_root, outputs_root, unfinished=True)


def _pick_next_run_id(base_root: Path) -> str:
    # 规则：从 1 开始找最小未使用的纯数字 run_id
    used = set()
    for child in base_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.isdigit():
            used.add(int(child.name))
    candidate = 1
    while candidate in used:
        candidate += 1
    return str(candidate)


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


def _resolve_dataclass_type(annotation):
    if isinstance(annotation, type) and hasattr(annotation, "__dataclass_fields__"):
        return annotation
    origin = get_origin(annotation)
    if origin is Union:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            if isinstance(arg, type) and hasattr(arg, "__dataclass_fields__"):
                return arg
    return None


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
        nested_type = _resolve_dataclass_type(annotation)

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


def _build_run_manifest(task_type: str, config_hash: str, config: SimConfig, scan: dict) -> dict:
    return {
        "protocol_version": TASK_PROTOCOL_VERSION,
        "config_hash": config_hash,
        "task_type": task_type,
        "config": _to_plain_jsonable(config),
        "scan": scan,
    }


def _write_run_manifest(paths: dict, task_type: str, config_hash: str, config: SimConfig, scan: dict) -> None:
    manifest = _build_run_manifest(task_type=task_type, config_hash=config_hash, config=config, scan=scan)
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
    task_hash = str(task.get("config_hash", "")).strip()
    manifest_hash = str(manifest.get("config_hash", "")).strip()
    if task_hash and manifest_hash and task_hash != manifest_hash:
        raise RuntimeError(f"CONFIG_MISMATCH: task={task_hash} manifest={manifest_hash}")

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

    if mode == SUMMARY_TASK_MODE:
        summary_for = str(task.get("summary_for", "")).upper()
        if summary_for not in SUPPORTED_EXPERIMENTS:
            raise ValueError(f"SCHEMA_ERROR: SUMMARY 缺少合法 summary_for，当前={summary_for}")
        manifest_task_type = str(manifest.get("task_type", "")).upper()
        if manifest_task_type and summary_for != manifest_task_type:
            raise RuntimeError(
                f"CONFIG_MISMATCH: summary_for={summary_for} 与 manifest.task_type={manifest_task_type} 不一致"
            )


def _build_task_list(
    task_type: str,
    config: SimConfig,
    config_hash: str,
    pending_dir: Path,
) -> tuple[int, dict]:
    # ------------------------------------------------------------------
    # 任务生成规则：
    #   - 所有物理子任务统一为 mode=CORE_TRIAL
    #   - 实验语义放在 experiment 字段
    #   - 参数分为 run_manifest(全局) + task.payload(局部)
    #   - SUMMARY 作为唯一汇总任务
    #
    # 所有任务只写入 pending/task_*.json，执行由 worker 完成。
    # ------------------------------------------------------------------
    n_runs = config.run.runs
    shots_per_run = config.run.shots_per_run
    task_count = 0
    scan = {
        "window_sweep_start_ns": config.run.window_sweep_start_ns,
        "window_sweep_end_ns": config.run.window_sweep_end_ns,
        "window_sweep_step_ns": config.run.window_sweep_step_ns,
        "tau_values_ns": None,
        "bs_theta_values": None,
        "length_values_km": None,
    }
    if task_type == "HOM":
        if config.hom is None:
            raise ValueError("HOM 任务需要 --mode HOM 并提供 tau 参数")
        tau_values = [float(v) for v in _build_hom_tau_values(config.hom)]
        scan["tau_values_ns"] = tau_values
        for tau in tau_values:
            for run_index in range(n_runs):
                tid = f"hom_tau_{tau:+.3f}_run_{run_index:06d}"
                task = {
                    "id": tid,
                    "mode": CORE_TASK_MODE,
                    "experiment": "HOM",
                    "run_index": run_index,
                    "shots": shots_per_run,
                    "seed": 100000 + task_count + 1,
                    "config_hash": config_hash,
                    "payload": {
                        "tau_ns": float(tau),
                        "window_ns": float(config.hom.window_ns),
                    },
                }
                path = pending_dir / f"task_{tid}.json"
                if not path.exists():
                    _write_json_atomic(path, task)
                task_count += 1
    elif task_type == "WINDOW_SCAN":
        for run_index in range(n_runs):
            tid = f"wscan_run_{run_index:06d}"
            task = {
                "id": tid,
                "mode": CORE_TASK_MODE,
                "experiment": "WINDOW_SCAN",
                "run_index": run_index,
                "shots": shots_per_run,
                "seed": 100000 + task_count + 1,
                "config_hash": config_hash,
                "payload": {},
            }
            path = pending_dir / f"task_{tid}.json"
            if not path.exists():
                _write_json_atomic(path, task)
            task_count += 1
    elif task_type == "BSM_SCAN":
        bs_values = bsm_scan.build_bsm_scan_values(config)
        scan["bs_theta_values"] = [float(v) for v in bs_values]
        for bs_idx, bs_theta in enumerate(bs_values):
            for run_index in range(n_runs):
                tid = f"bscan_theta_{bs_idx:04d}_run_{run_index:06d}"
                task = {
                    "id": tid,
                    "mode": CORE_TASK_MODE,
                    "experiment": "BSM_SCAN",
                    "run_index": run_index,
                    "shots": shots_per_run,
                    "seed": 100000 + task_count + 1,
                    "config_hash": config_hash,
                    "payload": {
                        "bs_theta": float(bs_theta),
                    },
                }
                path = pending_dir / f"task_{tid}.json"
                if not path.exists():
                    _write_json_atomic(path, task)
                task_count += 1
    elif task_type == "LENGTH_SCAN":
        length_values = length_scan.build_length_scan_values(config)
        scan["length_values_km"] = [float(v) for v in length_values]
        for length_idx, length_km in enumerate(length_values):
            for run_index in range(n_runs):
                tid = f"lscan_len_{length_idx:04d}_run_{run_index:06d}"
                task = {
                    "id": tid,
                    "mode": CORE_TASK_MODE,
                    "experiment": "LENGTH_SCAN",
                    "run_index": run_index,
                    "shots": shots_per_run,
                    "seed": 100000 + task_count + 1,
                    "config_hash": config_hash,
                    "payload": {
                        "length_km": float(length_km),
                    },
                }
                path = pending_dir / f"task_{tid}.json"
                if not path.exists():
                    _write_json_atomic(path, task)
                task_count += 1
    else:
        for run_index in range(n_runs):
            tid = f"sim_run_{run_index:06d}"
            task = {
                "id": tid,
                "mode": CORE_TASK_MODE,
                "experiment": "SIM",
                "run_index": run_index,
                "shots": shots_per_run,
                "seed": 100000 + task_count + 1,
                "config_hash": config_hash,
                "payload": {},
            }
            path = pending_dir / f"task_{tid}.json"
            if not path.exists():
                _write_json_atomic(path, task)
            task_count += 1
    summary_task = {
        "id": "summary",
        "mode": SUMMARY_TASK_MODE,
        "experiment": task_type,
        "summary_for": task_type,
        "config_hash": config_hash,
    }
    summary_path = pending_dir / "task_summary.json"
    if not summary_path.exists():
        _write_json_atomic(summary_path, summary_task)
    task_count += 1
    return task_count, scan


def _write_summary(task_type: str, paths: dict, config: SimConfig) -> None:
    summary.write_summary(task_type=task_type, paths=paths, config=config)


def _recover_stale_tasks(paths: dict, stale_seconds: int = 600) -> None:
    # ------------------------------------------------------------------
    # 任务回收：
    #   - inprogress 中若超过 stale_seconds 未更新，视为失联
    #   - 回滚到 pending 以便其他 worker 重新领取
    # ------------------------------------------------------------------
    now = time.time()
    for task_path in paths["inprogress"].glob("task_*.json"):
        try:
            # 以 mtime 作为“心跳”，超时则回收
            if now - task_path.stat().st_mtime > stale_seconds:
                task_path.replace(paths["pending"] / task_path.name)
        except FileNotFoundError:
            continue


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _run_server_monitor(
    paths: dict,
    expected_total: int,
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
    last_report = 0.0
    last_heartbeat = 0.0
    heartbeat_path = paths["summary"] / "server_heartbeat.txt"
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
        _recover_stale_tasks(paths)
        pending = list(paths["pending"].glob("task_*.json"))
        inprogress = list(paths["inprogress"].glob("task_*.json"))
        done_count = len(list(paths["done"].glob("task_*.json")))
        error_count = len(list(paths["error"].glob("task_*.json")))
        quiet_recent = False
        if quiet_output_path is not None and quiet_secs > 0:
            try:
                quiet_recent = (now - quiet_output_path.stat().st_mtime) < quiet_secs
            except FileNotFoundError:
                quiet_recent = False
        if show_progress and (not quiet_recent) and now - last_report >= 5:
            # total 以 expected_total 为优先（避免被回收/新增波动）
            total = (
                expected_total
                if expected_total > 0
                else done_count + len(pending) + len(inprogress) + error_count
            )
            elapsed = now - start_ts
            eta = "--:--:--"
            if done_count > 0 and total > done_count:
                rate = done_count / max(elapsed, 1e-9)
                eta = _format_duration((total - done_count) / max(rate, 1e-9))
            msg = (
                f"[server] 进度: 已完成 {done_count}/{total} | "
                f"进行中 {len(inprogress)} | 待完成 {len(pending)} | 失败 {error_count} | "
                f"用时 {_format_duration(elapsed)} | ETA {eta}"
            )
            if inline:
                print(f"\r{msg}", end="", flush=True)
            else:
                print(msg, flush=True)
            last_report = now
        if done_flag_path is not None and done_flag_path.exists():
            print()
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
    backoff = [5, 10, 30]
    backoff_idx = 0
    last_heartbeat = 0.0
    seen_task = False
    empty_rounds = 0
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
                _recover_stale_tasks(run_paths)
                pending_all = list(run_paths["pending"].glob("task_*.json"))
                if not pending_all:
                    continue
                # 优先抢“非 summary”任务；summary 只有在无人执行时才领
                non_summary_pending = [p for p in pending_all if not _is_summary_task_path(p)]
                non_summary_inprogress = [
                    p for p in run_paths["inprogress"].glob("task_*.json") if not _is_summary_task_path(p)
                ]
                if non_summary_pending or not non_summary_inprogress:
                    picked = run_paths
                    break
            if picked is None:
                time.sleep(backoff[backoff_idx])
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
        if now - last_heartbeat > 60:
            if heartbeat_path is not None:
                heartbeat_path.write_text(str(int(now)), encoding="utf-8")
            last_heartbeat = now
        _recover_stale_tasks(paths)
        pending_all = sorted(paths["pending"].glob("task_*.json"))
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
            time.sleep(backoff[backoff_idx])
            backoff_idx = min(backoff_idx + 1, len(backoff) - 1)
            continue
        # pending 优先非 summary；若只剩 summary 且无非 summary 在跑则领
        non_summary_pending = [p for p in pending_all if not _is_summary_task_path(p)]
        non_summary_inprogress = [
            p for p in paths["inprogress"].glob("task_*.json") if not _is_summary_task_path(p)
        ]
        if non_summary_pending:
            pending = non_summary_pending
        elif not non_summary_inprogress:
            pending = pending_all
        else:
            time.sleep(backoff[backoff_idx])
            backoff_idx = min(backoff_idx + 1, len(backoff) - 1)
            continue
        seen_task = True
        empty_rounds = 0
        task_path = None
        for cand in pending:
            dest = paths["inprogress"] / cand.name
            try:
                # 原子“抢占”：rename 成功即视为领取
                cand.replace(dest)
                task_path = dest
                break
            except FileNotFoundError:
                continue
            except PermissionError:
                continue
        if task_path is None:
            time.sleep(backoff[backoff_idx])
            backoff_idx = min(backoff_idx + 1, len(backoff) - 1)
            continue
        backoff_idx = 0

        stop_flag = threading.Event()

        def _heartbeat_loop() -> None:
            while not stop_flag.is_set():
                ts = int(time.time())
                heartbeat_path.write_text(str(ts), encoding="utf-8")
                # touch task_path：避免被回收为 stale
                if task_path.exists():
                    task_path.touch()
                time.sleep(60)

        t = threading.Thread(target=_heartbeat_loop, daemon=True)
        t.start()
        task = {}
        result_dir = None
        status = "ok"
        error_type = ""
        err_msg = ""
        metrics = {}
        task_mode = ""
        task_experiment = ""
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task_id = task.get("id", task_path.stem.replace("task_", ""))
            task_mode = str(task.get("mode", "")).upper()
            task_experiment = str(task.get("experiment", "")).upper()

            manifest = _load_run_manifest(paths)
            _apply_manifest_to_config(config, manifest)
            _validate_task_schema(task, manifest)

            result_dir = paths["results"] / f"result_{task_id}"
            plots_dir = result_dir / "plots"
            raw_dir = result_dir / "raw"
            plots_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            if task_mode == SUMMARY_TASK_MODE or _is_summary_task_path(task_path):
                # SUMMARY 任务：集中汇总 CSV
                summary_for = str(task.get("summary_for", "SIM")).upper()
                _write_summary(summary_for, paths, config)
                done_flag = paths["summary"] / "server_done.flag"
                try:
                    done_flag.write_text("done", encoding="utf-8")
                except Exception:
                    pass
                metrics = {"summary_for": summary_for}
                task_experiment = summary_for
            elif task_mode == CORE_TASK_MODE and task_experiment == "HOM":
                # HOM 任务：单 τ × run 的统计
                seed = task.get("seed")
                seed = int(seed) if seed is not None else None
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
                if click_records is not None:
                    # 每个 shot 的点击记录写入 raw/clicks.json
                    _write_json_atomic(raw_dir / "clicks.json", {"clicks": click_records})
            elif task_mode == CORE_TASK_MODE and task_experiment == "WINDOW_SCAN":
                metrics, click_records = window_scan.run_window_scan_task(
                    task=task,
                    config=config,
                    raw_dir=raw_dir,
                    plots_dir=plots_dir,
                    task_id=task_id,
                )
                if click_records is not None:
                    _write_json_atomic(raw_dir / "clicks.json", {"clicks": click_records})
            elif task_mode == CORE_TASK_MODE and task_experiment == "BSM_SCAN":
                metrics, click_records = bsm_scan.run_bsm_scan_task(
                    task=task,
                    config=config,
                    raw_dir=raw_dir,
                    plots_dir=plots_dir,
                    task_id=task_id,
                )
                if click_records is not None:
                    _write_json_atomic(raw_dir / "clicks.json", {"clicks": click_records})
            elif task_mode == CORE_TASK_MODE and task_experiment == "LENGTH_SCAN":
                metrics, click_records = length_scan.run_length_scan_task(
                    task=task,
                    config=config,
                    raw_dir=raw_dir,
                    plots_dir=plots_dir,
                    task_id=task_id,
                )
                if click_records is not None:
                    _write_json_atomic(raw_dir / "clicks.json", {"clicks": click_records})
            elif task_mode == CORE_TASK_MODE and task_experiment == "SIM":
                # SIM 任务：单 run 的成功统计与点击抽样
                seed = task.get("seed")
                seed = int(seed) if seed is not None else None
                run_index = int(task.get("run_index", 1))
                run_stats, success_metrics, click_records = single_run._run_single_simulation_core(
                    output_dir=raw_dir,
                    run_index=run_index,
                    config=config,
                    show_plots=config.run.plot_all,
                    plot_dir=plots_dir,
                    run_tag=task_id,
                    seed=seed,
                )
                metrics = {
                    "shots": run_stats["shots"],
                    "success": run_stats["success"],
                    "run_index": run_index,
                }
                if success_metrics:
                    metrics["p_arrive"] = success_metrics.get("p_arrive")
                    metrics["p_arrive_11"] = success_metrics.get("p_arrive_11")
                    metrics["p_arrive_same_arm"] = success_metrics.get("p_arrive_same_arm")
                    metrics["parameter_snapshot"] = success_metrics.get("parameter_snapshot")
                    metrics["p_success_abs"] = success_metrics.get("p_success_abs")
                    metrics["p_success_true_abs"] = success_metrics.get("p_success_true_abs")
                    metrics["p_success_false_abs"] = success_metrics.get("p_success_false_abs")
                    metrics["p_success_true_given_arrival"] = success_metrics.get("p_success_true_given_arrival")
                    metrics["fidelity_all"] = success_metrics.get("fidelity_all")
                    metrics["fidelity_true"] = success_metrics.get("fidelity_true")
                    metrics["fidelity_false"] = success_metrics.get("fidelity_false")
                    metrics["false_fraction"] = success_metrics.get("false_fraction")
                    metrics["corr_exx"] = success_metrics.get("corr_exx")
                    metrics["corr_eyy"] = success_metrics.get("corr_eyy")
                    metrics["corr_ezz"] = success_metrics.get("corr_ezz")
                    metrics["chsh_s_max"] = success_metrics.get("chsh_s_max")
                    metrics["p_success_signal_approx"] = success_metrics.get("p_success_signal_approx")
                    metrics["p_success_same_arm_approx"] = success_metrics.get("p_success_same_arm_approx")
                    metrics["p_success_intrinsic_dark_assisted"] = success_metrics.get(
                        "p_success_intrinsic_dark_assisted"
                    )
                    metrics["p_success_bg_assisted"] = success_metrics.get("p_success_bg_assisted")
                if click_records is not None:
                    _write_json_atomic(raw_dir / "clicks.json", {"clicks": click_records})
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
            t.join(timeout=1)
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
        except Exception:
            pass
        try:
            if status == "error":
                error_path = paths["error"] / task_path.name
                task_path.replace(error_path)
            else:
                done_path = paths["done"] / task_path.name
                task_path.replace(done_path)
        except Exception:
            pass


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
        task_type,
        config_hash,
        server_progress,
        progress_quiet_secs,
        progress_inline,
        self_check,
    ) = _parse_run_params(sys.argv)
    if self_check:
        print("[self-check] 运行探测端一致性检查...")
        run_detection_self_checks(verbose=False)
        print("[self-check] 完成")
        return
    config_hash = _resolve_config_hash(config_hash)
    task_type = task_type.upper()
    # queue_root 支持相对路径（相对项目根目录）
    base_root = _resolve_queue_root(queue_root)
    outputs_root = PROJECT_ROOT / "outputs"
    if role in ("server", "both"):
        base_root.mkdir(parents=True, exist_ok=True)
        outputs_root.mkdir(parents=True, exist_ok=True)
        # 启动前归档旧 run（未完成则加 _u）
        _archive_existing_runs(base_root, outputs_root, exclude_run_id=run_id)
        if run_id is None:
            # 未指定 run-id 则自动找最小可用数字
            run_id = _pick_next_run_id(base_root)
            print(f"[server] 未指定 run-id，自动选择: {run_id}")
    run_root = base_root / run_id if run_id else base_root
    paths = _queue_paths(run_root)
    if role in ("server", "both"):
        if run_root.exists() and any(run_root.iterdir()):
            raise SystemExit(f"run-id 已存在且非空: {run_root}")
        _ensure_queue_dirs(paths)
    elif run_id:
        _ensure_queue_dirs(paths)
    single_run.DEBUG_MODE = config.run.debug

    expected_total = 0
    done_flag = paths["summary"] / "server_done.flag"
    if role in ("server", "both"):
        if done_flag.exists():
            try:
                done_flag.unlink()
            except Exception:
                pass
        # 生成任务列表（含 SUMMARY）并写入 manifest
        expected_total, scan = _build_task_list(task_type, config, config_hash, paths["pending"])
        _write_run_manifest(paths, task_type, config_hash, config, scan)
        print(f"[server] 任务总数: {expected_total} | queue: {paths['root']}")
        if role == "server":
            _run_server_monitor(
                paths,
                expected_total,
                done_flag,
                show_progress=server_progress,
                quiet_output_path=None,
                quiet_secs=0.0,
                inline=progress_inline,
            )
            _archive_run(run_root, outputs_root)
            return

    if role in ("worker", "both"):
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
        queue_hint = str(paths["root"]) if run_id else str(base_root)
        print(f"[worker] cores={core_budget} | workers={worker_count} | queue={queue_hint}")

        if role == "both":
            output_tracker_path = paths["heartbeat"] / "worker_output.txt"
            monitor_thread = threading.Thread(
                target=_run_server_monitor,
                args=(
                    paths,
                    expected_total,
                    done_flag,
                    server_progress,
                    output_tracker_path,
                    progress_quiet_secs,
                    progress_inline,
                ),
            )
            monitor_thread.start()

        exit_when_done = role == "both" or (role == "worker" and run_id is not None)
        done_flag_arg = str(done_flag) if run_id is not None else None
        auto_pick = run_id is None
        tracker_path = None if run_id is None else (paths["heartbeat"] / "worker_output.txt")
        if worker_count == 1:
            _run_worker_loop(
                1,
                str(run_root if run_id else base_root),
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
                            str(run_root if run_id else base_root),
                            config,
                            exit_when_done,
                            done_flag_arg,
                            auto_pick,
                            tracker_path,
                        )
                    )
                for future in futures:
                    future.result()
        if role == "both":
            monitor_thread.join()
            _archive_run(run_root, outputs_root)


if __name__ == "__main__":
    main()
