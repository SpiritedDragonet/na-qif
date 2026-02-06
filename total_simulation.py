# -*- coding: utf-8 -*-
"""
CLI 入口与任务调度（单次实验逻辑见 atom_sim.experiment.single_run）。
"""

import sys
import os
import csv
import json
import argparse
import threading
import time
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor

# Add project root to path (for running as standalone script)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from atom_sim.experiment.common import (  # noqa: E402
    SimConfig,
    _compute_window_bins,
)
from atom_sim.experiment.hom import (  # noqa: E402
    parse_hom_cli,
    validate_no_hom_args,
    _build_hom_tau_values,
    _run_hom_run,
    _is_port_samepol_coincidence,
)
from atom_sim.experiment import single_run  # noqa: E402
from atom_sim.simulation import run_detection_self_checks  # noqa: E402


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
    #   - mode/task_type: SIM/HOM（物理任务类型）
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
    parser.add_argument("--task-type", dest="task_type", type=str, choices=["SIM", "HOM"], help="任务类型：SIM 或 HOM（默认随 --mode）")
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

    parser.add_argument("--tau", dest="tau", type=float, help="(HOM) 单一延迟 τ (ns)")
    parser.add_argument("--tau-start", dest="tau_start", type=float, help="(HOM) τ 起点 (ns)")
    parser.add_argument("--tau-end", dest="tau_end", type=float, help="(HOM) τ 终点 (ns)")
    parser.add_argument("--tau-step", dest="tau_step", type=float, help="(HOM) τ 步长 (ns)")
    parser.add_argument("--tau-points", dest="tau_points", type=int, help="(HOM) τ 采样点数")
    parser.add_argument("--window-ns", dest="window_ns", type=float, help="(HOM) 符合窗口 (ns)")

    parser.add_argument("--dark-hz", dest="dark_rate_intrinsic_hz", type=float, help="探测器本底暗计数率 (Hz)")
    parser.add_argument("--bg-mean-hz", dest="bg_rate_mean_hz", type=float, help="背景噪声均值 (Hz)")
    parser.add_argument("--bg-std-hz", dest="bg_rate_std_hz", type=float, help="背景噪声标准差 (Hz)")
    parser.add_argument("--detector-gate-ns", dest="detector_gate_ns", type=float, help="探测门宽 (ns)，用于将噪声概率从门宽映射到仿真 bin")
    parser.add_argument("--omega-peak-a", dest="omega_peak_a", type=float, help="A 臂驱动脉冲峰值 Ω_peak_A (rad/s)")
    parser.add_argument("--omega-peak-b", dest="omega_peak_b", type=float, help="B 臂驱动脉冲峰值 Ω_peak_B (rad/s)")
    parser.add_argument("--g-a", dest="g_a", type=float, help="A 臂原子-腔耦合强度 g_A (rad/s)")
    parser.add_argument("--g-b", dest="g_b", type=float, help="B 臂原子-腔耦合强度 g_B (rad/s)")
    parser.add_argument("--kappa-ex-a", dest="kappa_ex_a", type=float, help="A 臂腔外耦合衰减率 kappa_ex_A (rad/s)")
    parser.add_argument("--kappa-ex-b", dest="kappa_ex_b", type=float, help="B 臂腔外耦合衰减率 kappa_ex_B (rad/s)")
    parser.add_argument("--kappa-in-a", dest="kappa_in_a", type=float, help="A 臂腔内损耗衰减率 kappa_in_A (rad/s)")
    parser.add_argument("--kappa-in-b", dest="kappa_in_b", type=float, help="B 臂腔内损耗衰减率 kappa_in_B (rad/s)")
    parser.add_argument("--delta-u-a", dest="delta_u_a", type=float, help="A 臂 |u> 态失谐 delta_u_A (rad/s)")
    parser.add_argument("--delta-u-b", dest="delta_u_b", type=float, help="B 臂 |u> 态失谐 delta_u_B (rad/s)")
    parser.add_argument("--delta-e-a", dest="delta_e_a", type=float, help="A 臂 |e> 态失谐 delta_e_A (rad/s)")
    parser.add_argument("--delta-e-b", dest="delta_e_b", type=float, help="B 臂 |e> 态失谐 delta_e_B (rad/s)")
    parser.add_argument("--gamma-loss-a", dest="gamma_loss_a", type=float, help="A 臂不可收集通道等效损耗率 gamma_loss_A (1/s)")
    parser.add_argument("--gamma-loss-b", dest="gamma_loss_b", type=float, help="B 臂不可收集通道等效损耗率 gamma_loss_B (1/s)")
    parser.add_argument("--qfc-theta-h", dest="qfc_theta_h", type=float, help="QFC H转换角 theta_H (rad)")
    parser.add_argument("--qfc-theta-v", dest="qfc_theta_v", type=float, help="QFC V转换角 theta_V (rad)")
    parser.add_argument("--no-filter-780", dest="no_filter_780", action="store_true", help="关闭 780 滤波（保留 780 分量）")
    parser.add_argument("--enum-mode", dest="enum_mode", type=str, help="成功事件枚举模式：dark/no-dark/both")
    parser.add_argument("--plot-all", dest="plot_all", action="store_true", help="所有 run 都绘图（默认仅保留一个）")
    parser.add_argument("--no-plot", dest="no_plot", action="store_true", help="完全禁止绘图（覆盖 plot-all）")
    parser.add_argument("--eta-det", dest="eta_det", type=float, help="探测效率 η (0~1)")
    parser.add_argument("--ideal-det", dest="ideal_det", action="store_true", help="理想探测（eta_det=1, 无噪声）")
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
    if args.detector_gate_ns is not None:
        config.noise.detector_gate_ns = float(args.detector_gate_ns)
    if args.omega_peak_a is not None:
        config.emission.arm_A.omega_peak = float(args.omega_peak_a)
    if args.omega_peak_b is not None:
        config.emission.arm_B.omega_peak = float(args.omega_peak_b)
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
    if args.delta_u_a is not None:
        config.emission.arm_A.delta_u = float(args.delta_u_a)
    if args.delta_u_b is not None:
        config.emission.arm_B.delta_u = float(args.delta_u_b)
    if args.delta_e_a is not None:
        config.emission.arm_A.delta_e = float(args.delta_e_a)
    if args.delta_e_b is not None:
        config.emission.arm_B.delta_e = float(args.delta_e_b)
    if args.gamma_loss_a is not None:
        config.emission.arm_A.gamma_loss = float(args.gamma_loss_a)
    if args.gamma_loss_b is not None:
        config.emission.arm_B.gamma_loss = float(args.gamma_loss_b)
    if args.qfc_theta_h is not None:
        config.qfc.theta_H = float(args.qfc_theta_h)
    if args.qfc_theta_v is not None:
        config.qfc.theta_V = float(args.qfc_theta_v)
    if args.no_filter_780:
        config.qfc.apply_filter_780 = False

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
    # 光纤噪声开关（注意：这会影响到统计与物理可解释性）
    config.fiber.noise_enabled = not args.no_fiber_noise

    if config.run.runs < 1:
        parser.error("N_runs 必须 >= 1")
    if config.run.shots_per_run < 1:
        parser.error("shots_per_run 必须 >= 1")
    if config.run.cores < 1:
        parser.error("cores 必须 >= 1")
    if config.noise.detector_gate_ns <= 0.0:
        parser.error("detector_gate_ns 必须 > 0")
    if config.emission.arm_A.gamma_loss < 0.0 or config.emission.arm_B.gamma_loss < 0.0:
        parser.error("gamma_loss_a / gamma_loss_b 必须 >= 0")

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
    config.detector.ideal_det = bool(args.ideal_det)
    if config.detector.ideal_det:
        config.detector.eta_det = 1.0
    if args.v_res is not None:
        config.detector.v_res = float(args.v_res)
    if not (0.0 < config.detector.eta_det <= 1.0):
        parser.error("eta_det 必须在 (0, 1] 内")
    if not (0.0 <= config.detector.v_res <= 1.0):
        parser.error("v_res 必须在 [0, 1] 内")

    if task_type == "HOM":
        config.hom = parse_hom_cli(args, parser)
    else:
        validate_no_hom_args(args, parser)
        config.hom = None
        if args.window_ns is not None:
            config.run.window_ns = float(args.window_ns)

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


def _build_task_list(
    task_type: str,
    config: SimConfig,
    config_hash: str,
    pending_dir: Path,
) -> int:
    # ------------------------------------------------------------------
    # 任务生成规则：
    #   - SIM：每个 run 一个 task
    #   - HOM：每个 τ × run 一个 task
    #   - SUMMARY：最后追加一个汇总任务（由 worker 执行）
    #
    # 所有任务只写入 pending/task_*.json，执行由 worker 完成。
    # ------------------------------------------------------------------
    n_runs = config.run.runs
    shots_per_run = config.run.shots_per_run
    task_count = 0
    if task_type == "HOM":
        if config.hom is None:
            raise ValueError("HOM 任务需要 --mode HOM 并提供 tau 参数")
        # τ 列表由 hom 配置决定（可能是随机或扫描）
        tau_values = _build_hom_tau_values(config.hom)
        for tau in tau_values:
            for run_index in range(n_runs):
                # id 编码 tau 与 run_index，保证唯一性
                tid = f"hom_tau_{tau:+.3f}_run_{run_index:06d}"
                task = {
                    "id": tid,
                    "mode": "HOM",
                    "tau_ns": float(tau),
                    "shots": shots_per_run,
                    # seed 用于保证可重现；与 task_count 绑定
                    "seed": 100000 + task_count + 1,
                    "config_hash": config_hash,
                    "window_ns": config.hom.window_ns,
                }
                path = pending_dir / f"task_{tid}.json"
                if not path.exists():
                    # 不覆盖已有任务，便于断点续算
                    _write_json_atomic(path, task)
                task_count += 1
    else:
        for run_index in range(n_runs):
            # SIM：仅按 run_index 划分
            tid = f"sim_run_{run_index:06d}"
            task = {
                "id": tid,
                "mode": "SIM",
                "run_index": run_index,
                "shots": shots_per_run,
                "seed": 100000 + task_count + 1,
                "config_hash": config_hash,
            }
            path = pending_dir / f"task_{tid}.json"
            if not path.exists():
                _write_json_atomic(path, task)
            task_count += 1
    # SUMMARY 任务：由 worker 在最后执行汇总
    summary_task = {
        "id": "summary",
        "mode": "SUMMARY",
        "summary_for": task_type,
        "config_hash": config_hash,
    }
    summary_path = pending_dir / "task_summary.json"
    if not summary_path.exists():
        _write_json_atomic(summary_path, summary_task)
    task_count += 1
    return task_count


def _write_summary(task_type: str, paths: dict, config: SimConfig) -> None:
    # ------------------------------------------------------------------
    # 汇总任务（SUMMARY）：
    #   - 遍历 results/result_*/meta.json
    #   - HOM：额外读取 raw/clicks.json 生成 hom_trials.csv / hom_summary.csv
    #   - SIM：生成 sim_summary.csv
    # ------------------------------------------------------------------
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    if task_type == "HOM":
        window_bins = None
        if config.hom is not None:
            window_bins = _compute_window_bins(
                config.hom.window_ns,
                config.emission.dt_ns,
                detection_gate_ns=config.noise.detector_gate_ns,
            )
        trials_path = summary_dir / "hom_trials.csv"
        tau_path = summary_dir / "hom_summary.csv"
        # hom_trials：逐 run × shot 的明细（含点击 bin）
        with open(trials_path, "w", encoding="utf-8", newline="") as trials_file:
            trials_writer = csv.writer(trials_file)
            trials_writer.writerow([
                "tau_ns",
                "run_index",
                "shot_index",
                "p_arrive",
                "H1_bin",
                "V1_bin",
                "H2_bin",
                "V2_bin",
                "H1_dark",
                "V1_dark",
                "H2_dark",
                "V2_dark",
            ])
            tau_states = {}
            for meta_path in sorted(results_dir.glob("result_*/meta.json")):
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if data.get("mode") != "HOM":
                    continue
                tid = data.get("id", "")
                m = re.match(r"hom_tau_([+-]?\d+\.\d+)_run_(\d+)", tid)
                if not m:
                    continue
                tau_ns = float(m.group(1))
                run_index = int(m.group(2))
                metrics = data.get("metrics", {})
                p_arrive = metrics.get("p_arrive")
                tau_key = f"{tau_ns:.6f}"
                # tau_states 用于汇总每个 τ 的统计
                state = tau_states.setdefault(
                    tau_key,
                    {
                        "tau_ns": tau_ns,
                        "runs_total": 0,
                        "coinc": 0,
                        "p_arrive_sum": 0.0,
                        "arrive_trials": 0.0,
                        "shots_total": 0,
                        "coinc_true": 0,
                        "coinc_dark_any": 0,
                        "coinc_dark_single": 0,
                        "coinc_dark_double": 0,
                        "dark_clicks_total": 0,
                        "clicks_total": 0,
                    },
                )
                state["runs_total"] += 1
                if data.get("status") != "ok":
                    continue
                state["coinc"] += int(metrics.get("coinc", 0) or 0)
                if p_arrive is not None:
                    state["p_arrive_sum"] += float(p_arrive)
                    # arrive_trials：按 p_arrive 估算有效试验数
                    state["arrive_trials"] += float(p_arrive) * config.run.shots_per_run
                clicks_path = meta_path.parent / "raw" / "clicks.json"
                clicks = []
                if clicks_path.exists():
                    try:
                        clicks = json.loads(clicks_path.read_text(encoding="utf-8")).get("clicks", [])
                    except Exception:
                        clicks = []
                shots_in_run = len(clicks) if clicks else config.run.shots_per_run
                state["shots_total"] += shots_in_run
                # 无点击记录也写一行占位，便于对齐 run_index
                if not clicks:
                    trials_writer.writerow([
                        f"{tau_ns:.6f}",
                        run_index,
                        -1,
                        p_arrive,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ])
                else:
                    # 每个 shot 一行，点击 bin 以分号拼接
                    for shot_idx, shot_clicks in enumerate(clicks):
                        bins = {"H1": "", "V1": "", "H2": "", "V2": ""}
                        darks = {"H1": "", "V1": "", "H2": "", "V2": ""}
                        events = []
                        for click in shot_clicks:
                            if len(click) < 3:
                                raise ValueError("HOM clicks 至少包含 (det, bin, is_dark)")
                            det = click[0]
                            bin_idx = click[1]
                            is_dark = bool(click[2])
                            events.append(SimpleNamespace(detector=det, bin_index=bin_idx, is_dark=is_dark))
                            if det in bins:
                                bins[det] = f"{bin_idx}" if bins[det] == "" else f"{bins[det]};{bin_idx}"
                                flag = "1" if is_dark else "0"
                                darks[det] = flag if darks[det] == "" else f"{darks[det]};{flag}"

                        dark_clicks = sum(1 for e in events if e.is_dark)
                        state["dark_clicks_total"] += dark_clicks
                        state["clicks_total"] += len(events)
                        if events and _is_port_samepol_coincidence(events, window_bins):
                            if dark_clicks == 0:
                                state["coinc_true"] += 1
                            else:
                                state["coinc_dark_any"] += 1
                                if dark_clicks == 1:
                                    state["coinc_dark_single"] += 1
                                else:
                                    state["coinc_dark_double"] += 1
                        trials_writer.writerow([
                            f"{tau_ns:.6f}",
                            run_index,
                            shot_idx,
                            p_arrive,
                            bins["H1"],
                            bins["V1"],
                            bins["H2"],
                            bins["V2"],
                            darks["H1"],
                            darks["V1"],
                            darks["H2"],
                            darks["V2"],
                        ])
        # hom_summary：按 τ 汇总统计
        with open(tau_path, "w", encoding="utf-8", newline="") as tau_file:
            tau_writer = csv.writer(tau_file)
            tau_writer.writerow([
                "tau_ns",
                "runs_target",
                "runs_total",
                "coinc_counts",
                "coinc_rate",
                "p_arrive_avg",
                "arrive_trials",
                "window_ns",
                "shots_per_run",
                "shots_total",
                "coinc_true",
                "coinc_dark_any",
                "coinc_dark_single",
                "coinc_dark_double",
                "dark_clicks_total",
                "clicks_total",
                "dark_click_rate",
                "dark_click_rate_per_det",
            ])
            for tau_key in sorted(tau_states, key=lambda x: float(x)):
                s = tau_states[tau_key]
                runs_total = s["runs_total"]
                # 平均值对全部已完成 run 取均值
                p_arrive_avg = (s["p_arrive_sum"] / runs_total) if runs_total > 0 else 0.0
                # coinc_rate：符合数 / 预计到达试验数
                coinc_rate = (s["coinc"] / s["arrive_trials"]) if s["arrive_trials"] > 0 else 0.0
                dark_click_rate = (s["dark_clicks_total"] / s["clicks_total"]) if s["clicks_total"] > 0 else 0.0
                dark_click_rate_per_det = (
                    s["dark_clicks_total"] / (s["shots_total"] * 4)
                    if s["shots_total"] > 0
                    else 0.0
                )
                tau_writer.writerow([
                    f"{s['tau_ns']:.6f}",
                    config.run.runs,
                    s["runs_total"],
                    s["coinc"],
                    f"{coinc_rate:.8f}",
                    f"{p_arrive_avg:.6f}",
                    f"{s['arrive_trials']:.6f}",
                    f"{config.hom.window_ns if config.hom else 0.0:.3f}",
                    config.run.shots_per_run,
                    s["shots_total"],
                    s["coinc_true"],
                    s["coinc_dark_any"],
                    s["coinc_dark_single"],
                    s["coinc_dark_double"],
                    s["dark_clicks_total"],
                    s["clicks_total"],
                    f"{dark_click_rate:.8f}",
                    f"{dark_click_rate_per_det:.8f}",
                ])
        return
    if task_type == "SIM":
        trials_path = summary_dir / "sim_trials.csv"
        with open(trials_path, "w", encoding="utf-8", newline="") as trials_file:
            trials_writer = csv.writer(trials_file)
            trials_writer.writerow([
                "run_index",
                "shot_index",
                "success",
                "bell",
                "p_arrive",
                "p_arrive_11",
                "p_arrive_same_arm",
                "p_arrive_20",
                "p_arrive_02",
                "H1_bin",
                "V1_bin",
                "H2_bin",
                "V2_bin",
                "H1_dark",
                "V1_dark",
                "H2_dark",
                "V2_dark",
            ])
            for meta_path in sorted(results_dir.glob("result_*/meta.json")):
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if data.get("mode") != "SIM":
                    continue
                tid = data.get("id", "")
                m = re.match(r"sim_run_(\d+)", tid)
                if not m:
                    continue
                run_index = int(m.group(1))
                metrics = data.get("metrics", {})
                p_arrive = metrics.get("p_arrive")
                p_arrive_11 = metrics.get("p_arrive_11")
                p_arrive_same_arm = metrics.get("p_arrive_same_arm")
                p_arrive_20 = metrics.get("p_arrive_20")
                p_arrive_02 = metrics.get("p_arrive_02")
                clicks_path = meta_path.parent / "raw" / "clicks.json"
                clicks = []
                if clicks_path.exists():
                    try:
                        clicks = json.loads(clicks_path.read_text(encoding="utf-8")).get("clicks", [])
                    except Exception:
                        clicks = []
                if not clicks:
                    trials_writer.writerow([
                        run_index,
                        -1,
                        "",
                        "",
                        p_arrive,
                        p_arrive_11,
                        p_arrive_same_arm,
                        p_arrive_20,
                        p_arrive_02,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ])
                    continue
                for record in clicks:
                    shot_idx = record.get("shot_index")
                    success = record.get("success")
                    bell = record.get("bell")
                    shot_clicks = record.get("clicks", [])
                    bins = {"H1": "", "V1": "", "H2": "", "V2": ""}
                    darks = {"H1": "", "V1": "", "H2": "", "V2": ""}
                    for click in shot_clicks:
                        if len(click) < 3:
                            raise ValueError("SIM clicks 至少包含 (det, bin, is_dark)")
                        det = click[0]
                        bin_idx = click[1]
                        is_dark = bool(click[2])
                        if det in bins:
                            bins[det] = f"{bin_idx}" if bins[det] == "" else f"{bins[det]};{bin_idx}"
                            flag = "1" if is_dark else "0"
                            darks[det] = flag if darks[det] == "" else f"{darks[det]};{flag}"
                    trials_writer.writerow([
                        run_index,
                        shot_idx,
                        success,
                        bell,
                        p_arrive,
                        p_arrive_11,
                        p_arrive_same_arm,
                        p_arrive_20,
                        p_arrive_02,
                        bins["H1"],
                        bins["V1"],
                        bins["H2"],
                        bins["V2"],
                        darks["H1"],
                        darks["V1"],
                        darks["H2"],
                        darks["V2"],
                    ])
    summary_path = summary_dir / f"{task_type.lower()}_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "mode",
            "p_arrive",
            "p_arrive_11",
            "p_arrive_same_arm",
            "p_arrive_20",
            "p_arrive_02",
            "coinc",
            "timestamp",
        ])
        for meta_path in sorted(results_dir.glob("result_*/meta.json")):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tid = data.get("id")
            if not tid:
                continue
            m = data.get("metrics", {})
            writer.writerow([
                tid,
                data.get("mode", task_type),
                m.get("p_arrive"),
                m.get("p_arrive_11"),
                m.get("p_arrive_same_arm"),
                m.get("p_arrive_20"),
                m.get("p_arrive_02"),
                m.get("coinc"),
                data.get("timestamp"),
            ])


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
        err_msg = ""
        metrics = {}
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task_id = task.get("id", task_path.stem.replace("task_", ""))
            result_dir = paths["results"] / f"result_{task_id}"
            plots_dir = result_dir / "plots"
            raw_dir = result_dir / "raw"
            plots_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            if task.get("mode") == "SUMMARY" or _is_summary_task_path(task_path):
                # SUMMARY 任务：集中汇总 CSV
                summary_for = str(task.get("summary_for", "SIM")).upper()
                _write_summary(summary_for, paths, config)
                done_flag = paths["summary"] / "server_done.flag"
                try:
                    done_flag.write_text("done", encoding="utf-8")
                except Exception:
                    pass
                metrics = {"summary_for": summary_for}
            elif task.get("mode") == "HOM":
                # HOM 任务：单 τ × run 的统计
                seed = task.get("seed")
                seed = int(seed) if seed is not None else None
                tau_ns = float(task.get("tau_ns", 0.0))
                shots = int(task.get("shots", config.run.shots_per_run))
                window_ns = float(task.get("window_ns", config.hom.window_ns if config.hom else 70.0))
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
                    "p_arrive": p_arrive,
                    "coinc": coincid,
                }
                if click_records is not None:
                    # 每个 shot 的点击记录写入 raw/clicks.json
                    _write_json_atomic(raw_dir / "clicks.json", {"clicks": click_records})
            else:
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
                }
                if success_metrics:
                    metrics["p_arrive"] = success_metrics.get("p_arrive")
                    metrics["parameter_snapshot"] = success_metrics.get("parameter_snapshot")
                    metrics["p_success_abs"] = success_metrics.get("p_success_abs")
                    metrics["p_success_true_abs"] = success_metrics.get("p_success_true_abs")
                    metrics["p_success_false_abs"] = success_metrics.get("p_success_false_abs")
                    metrics["p_success_signal_approx"] = success_metrics.get("p_success_signal_approx")
                    metrics["p_success_same_arm_approx"] = success_metrics.get("p_success_same_arm_approx")
                    metrics["p_success_intrinsic_dark_assisted"] = success_metrics.get(
                        "p_success_intrinsic_dark_assisted"
                    )
                    metrics["p_success_bg_assisted"] = success_metrics.get("p_success_bg_assisted")
                if click_records is not None:
                    _write_json_atomic(raw_dir / "clicks.json", {"clicks": click_records})
        except Exception as exc:
            status = "error"
            err_msg = str(exc)
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
                "mode": task.get("mode", "SIM"),
                "status": status,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "metrics": metrics,
            }
            if err_msg:
                meta["error"] = err_msg
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
        # 生成任务列表（含 SUMMARY）
        expected_total = _build_task_list(task_type, config, config_hash, paths["pending"])
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
