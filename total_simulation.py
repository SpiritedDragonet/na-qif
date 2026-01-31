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
from atom_sim.experiment import single_run  # noqa: E402


def _parse_run_params(argv):
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
    parser.add_argument("--max-attempts", dest="max_attempts", type=int, help="(HOM) 每个 τ 的最大尝试次数")

    parser.add_argument("--dark-hz", dest="dark_rate_intrinsic_hz", type=float, help="探测器本底暗计数率 (Hz)")
    parser.add_argument("--bg-mean-hz", dest="bg_rate_mean_hz", type=float, help="背景噪声均值 (Hz)")
    parser.add_argument("--bg-std-hz", dest="bg_rate_std_hz", type=float, help="背景噪声标准差 (Hz)")
    parser.add_argument("--enum-mode", dest="enum_mode", type=str, help="成功事件枚举模式：dark/no-dark")
    parser.add_argument("--plot-all", dest="plot_all", action="store_true", help="所有 run 都绘图（默认仅保留一个）")
    parser.add_argument("--eta-det", dest="eta_det", type=float, help="探测效率 η (0~1)")
    parser.add_argument("--ideal-det", dest="ideal_det", action="store_true", help="理想探测（eta_det=1, 无噪声）")
    parser.add_argument("--debug", dest="debug", action="store_true", help="开启调试模式（输出耗时等）")
    args = parser.parse_args(argv[1:])

    run_id = args.run_id.strip() if args.run_id else None
    if run_id:
        if "/" in run_id or "\\" in run_id:
            parser.error("run-id 不能包含路径分隔符")
        if run_id in (".", ".."):
            parser.error("run-id 不能为 . 或 ..")

    config = SimConfig()

    if args.dark_rate_intrinsic_hz is not None:
        config.noise.dark_rate_intrinsic_hz = args.dark_rate_intrinsic_hz
    if args.bg_rate_mean_hz is not None:
        config.noise.bg_rate_mean_hz = args.bg_rate_mean_hz
    if args.bg_rate_std_hz is not None:
        config.noise.bg_rate_std_hz = args.bg_rate_std_hz

    config.run.runs = args.n_runs if args.n_runs is not None else config.run.runs
    config.run.shots_per_run = (
        args.shots_per_run if args.shots_per_run is not None else config.run.shots_per_run
    )
    config.run.cores = args.cores if args.cores is not None else config.run.cores
    mode = (args.mode or config.mode).upper()
    task_type = (args.task_type or mode).upper()
    if args.task_type is not None and args.mode is not None and task_type != mode:
        parser.error("task-type 与 mode 冲突，请保持一致")
    config.mode = task_type
    config.fiber.noise_enabled = not args.no_fiber_noise

    if config.run.runs < 1:
        parser.error("N_runs 必须 >= 1")
    if config.run.shots_per_run < 1:
        parser.error("shots_per_run 必须 >= 1")
    if config.run.cores < 1:
        parser.error("cores 必须 >= 1")

    config.run.enum_mode = (args.enum_mode or config.run.enum_mode).strip().lower()
    if config.run.enum_mode not in ("dark", "no-dark"):
        parser.error("enum-mode 仅支持 dark / no-dark")

    if args.eta_det is not None:
        config.detector.eta_det = float(args.eta_det)
    config.detector.ideal_det = bool(args.ideal_det)
    if config.detector.ideal_det:
        config.detector.eta_det = 1.0
    if not (0.0 < config.detector.eta_det <= 1.0):
        parser.error("eta_det 必须在 (0, 1] 内")

    if task_type == "HOM":
        config.hom = parse_hom_cli(args, parser)
    else:
        validate_no_hom_args(args, parser)
        config.hom = None

    config.run.plot_all = bool(args.plot_all)
    config.run.debug = bool(args.debug)
    return config, args.role, args.queue_root, run_id, task_type, args.config_hash


def _resolve_config_hash(explicit: Optional[str]) -> str:
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
    path = Path(path_str)
    if not path.is_absolute():
        return (PROJECT_ROOT / path).resolve()
    return path


def _queue_paths(queue_root: Path) -> dict:
    root = Path(queue_root)
    return {
        "root": root,
        "tasks": root / "tasks",
        "pending": root / "tasks" / "pending",
        "inprogress": root / "tasks" / "inprogress",
        "done": root / "tasks" / "done",
        "results": root / "results",
        "summary": root / "summary",
        "heartbeat": root / "heartbeat",
    }


def _ensure_queue_dirs(paths: dict) -> None:
    paths["pending"].mkdir(parents=True, exist_ok=True)
    paths["inprogress"].mkdir(parents=True, exist_ok=True)
    paths["done"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["summary"].mkdir(parents=True, exist_ok=True)
    paths["heartbeat"].mkdir(parents=True, exist_ok=True)


def _run_id_sort_key(run_id: str) -> tuple:
    if run_id.isdigit():
        return (0, int(run_id), run_id)
    m = re.match(r"(\d+)", run_id)
    if m:
        return (0, int(m.group(1)), run_id)
    return (1, run_id)


def _discover_run_roots(base_root: Path) -> list:
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


def _pick_output_root(outputs_root: Path, stamp: str) -> Path:
    dest = outputs_root / stamp
    if not dest.exists():
        return dest
    suffix = 1
    while (outputs_root / f"{stamp}_{suffix}").exists():
        suffix += 1
    return outputs_root / f"{stamp}_{suffix}"


def _archive_run(run_root: Path, outputs_root: Path) -> Optional[Path]:
    if not run_root.exists():
        return None
    outputs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dest = _pick_output_root(outputs_root, stamp)
    try:
        shutil.move(str(run_root), str(dest))
    except Exception as exc:
        print(f"[server] 归档失败: {exc}")
        return None
    print(f"[server] 已归档完成任务到: {dest}")
    return dest


def _pick_next_run_id(base_root: Path) -> str:
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
    n_runs = config.run.runs
    shots_per_run = config.run.shots_per_run
    task_count = 0
    if task_type == "HOM":
        if config.hom is None:
            raise ValueError("HOM 任务需要 --mode HOM 并提供 tau 参数")
        tau_values = _build_hom_tau_values(config.hom)
        for tau in tau_values:
            for run_index in range(n_runs):
                tid = f"hom_tau_{tau:+.3f}_run_{run_index:06d}"
                task = {
                    "id": tid,
                    "mode": "HOM",
                    "tau_ns": float(tau),
                    "shots": shots_per_run,
                    "seed": 100000 + task_count + 1,
                    "config_hash": config_hash,
                    "window_ns": config.hom.window_ns,
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
    return task_count


def _write_summary(task_type: str, paths: dict) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{task_type.lower()}_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "mode", "p_arrive", "coinc", "valid", "timestamp"])
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
                m.get("coinc"),
                m.get("valid"),
                data.get("timestamp"),
            ])


def _run_summary_task(
    task_type: str,
    paths: dict,
    done_flag_path: Optional[Path] = None,
) -> None:
    while True:
        now = time.time()
        for task_path in paths["inprogress"].glob("task_*.json"):
            try:
                if now - task_path.stat().st_mtime > 600:
                    task_path.replace(paths["pending"] / task_path.name)
            except FileNotFoundError:
                continue
        pending = list(paths["pending"].glob("task_*.json"))
        inprogress = list(paths["inprogress"].glob("task_*.json"))
        if not pending and not inprogress:
            break
        time.sleep(10)
    _write_summary(task_type, paths)
    if done_flag_path is not None:
        try:
            done_flag_path.write_text("done", encoding="utf-8")
        except Exception:
            pass


def _run_worker_loop(
    worker_id: int,
    queue_root: str,
    config: SimConfig,
    exit_when_done: bool = False,
    done_flag_path: Optional[str] = None,
    auto_pick: bool = False,
) -> None:
    base_root = Path(queue_root)
    paths = None
    host = os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "worker"
    heartbeat_path = None
    done_flag = Path(done_flag_path) if done_flag_path else None
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
            picked = None
            for run_root in _discover_run_roots(base_root):
                run_paths = _queue_paths(run_root)
                pending = list(run_paths["pending"].glob("task_*.json"))
                if pending:
                    picked = run_paths
                    break
            if picked is None:
                time.sleep(backoff[backoff_idx])
                backoff_idx = min(backoff_idx + 1, len(backoff) - 1)
                continue
            paths = picked
            _ensure_queue_dirs(paths)
            heartbeat_path = paths["heartbeat"] / f"worker_{host}_{worker_id}.txt"
        elif paths is None:
            paths = _queue_paths(base_root)
            _ensure_queue_dirs(paths)
            heartbeat_path = paths["heartbeat"] / f"worker_{host}_{worker_id}.txt"

        now = time.time()
        if now - last_heartbeat > 60:
            if heartbeat_path is not None:
                heartbeat_path.write_text(str(int(now)), encoding="utf-8")
            last_heartbeat = now
        pending = sorted(paths["pending"].glob("task_*.json"))
        if not pending:
            inprogress = list(paths["inprogress"].glob("task_*.json"))
            if exit_when_done:
                if done_flag and done_flag.exists() and not inprogress:
                    break
                if seen_task and not inprogress:
                    empty_rounds += 1
                    if empty_rounds >= 5:
                        break
                else:
                    empty_rounds = 0
            time.sleep(backoff[backoff_idx])
            backoff_idx = min(backoff_idx + 1, len(backoff) - 1)
            continue
        seen_task = True
        empty_rounds = 0
        task_path = None
        for cand in pending:
            dest = paths["inprogress"] / cand.name
            try:
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
            seed = task.get("seed")
            seed = int(seed) if seed is not None else None
            result_dir = paths["results"] / f"result_{task_id}"
            plots_dir = result_dir / "plots"
            raw_dir = result_dir / "raw"
            plots_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            if task.get("mode") == "HOM":
                tau_ns = float(task.get("tau_ns", 0.0))
                shots = int(task.get("shots", config.run.shots_per_run))
                window_ns = float(task.get("window_ns", config.hom.window_ns if config.hom else 70.0))
                coincid, early_abort, p_arrive, p_no_loss, click_records = _run_hom_run(
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
                    "valid": 0 if early_abort else 1,
                    "p_no_loss": p_no_loss,
                }
                if click_records is not None:
                    _write_json_atomic(raw_dir / "clicks.json", {"clicks": click_records})
            else:
                run_index = int(task.get("run_index", 1))
                run_stats, success_metrics = single_run._run_single_simulation_core(
                    output_dir=raw_dir,
                    run_index=run_index,
                    config=config,
                    summary_path=None,
                    summary_lock_path=None,
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
                    metrics["p_success_all"] = success_metrics.get("p_success_all")
                    metrics["p_success_true"] = success_metrics.get("p_success_true")
                    metrics["p_success_false"] = success_metrics.get("p_success_false")
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
            done_path = paths["done"] / task_path.name
            task_path.replace(done_path)
        except Exception:
            pass


def main():
    """
    主函数：基于共享目录的 server/worker 调度。
    """
    config, role, queue_root, run_id, task_type, config_hash = _parse_run_params(sys.argv)
    config_hash = _resolve_config_hash(config_hash)
    task_type = task_type.upper()
    base_root = _resolve_queue_root(queue_root)
    if role in ("server", "both") and run_id is None:
        base_root.mkdir(parents=True, exist_ok=True)
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
        expected_total = _build_task_list(task_type, config, config_hash, paths["pending"])
        print(f"[server] 任务总数: {expected_total} | queue: {paths['root']}")
        if role == "server":
            _run_summary_task(task_type, paths, done_flag)
            _archive_run(run_root, PROJECT_ROOT / "outputs")
            return

    if role in ("worker", "both"):
        core_budget = max(1, min(config.run.cores, os.cpu_count() or 1))
        reserve = 1
        pending_count = len(list(paths["pending"].glob("task_*.json"))) if run_id else 1
        target_tasks = pending_count if pending_count > 0 else 1
        worker_count = max(1, min(core_budget - reserve, target_tasks))
        if worker_count < 1:
            worker_count = 1
        if worker_count > 1:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
        queue_hint = str(paths["root"]) if run_id else str(base_root)
        print(f"[worker] cores={core_budget} | workers={worker_count} | queue={queue_hint}")

        if role == "both":
            summary_thread = threading.Thread(
                target=_run_summary_task,
                args=(task_type, paths, done_flag),
            )
            summary_thread.start()

        exit_when_done = role == "both" or (role == "worker" and run_id is not None)
        done_flag_arg = str(done_flag) if run_id is not None else None
        auto_pick = run_id is None
        if worker_count == 1:
            _run_worker_loop(
                1,
                str(run_root if run_id else base_root),
                config,
                exit_when_done,
                done_flag_arg,
                auto_pick,
            )
        else:
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
                        )
                    )
                for future in futures:
                    future.result()
        if role == "both":
            summary_thread.join()
            _archive_run(run_root, PROJECT_ROOT / "outputs")


if __name__ == "__main__":
    main()
