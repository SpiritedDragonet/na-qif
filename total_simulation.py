# -*- coding: utf-8 -*-
"""
CLI 入口与任务调度（单次实验逻辑见 atom_sim.experiment.single_run）。
"""

import sys
import os
import csv
import argparse
from contextlib import contextmanager
import time
from pathlib import Path
from datetime import datetime
from collections import deque, Counter
from typing import Optional

import numpy as np

# Add project root to path (for running as standalone script)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from atom_sim.experiment.common import (  # noqa: E402
    SimConfig,
    run_task_queue,
)
from atom_sim.experiment.hom import (  # noqa: E402
    run_hom_experiment,
    parse_hom_cli,
    validate_no_hom_args,
)
from atom_sim.experiment import single_run  # noqa: E402


def _parse_run_params(argv) -> SimConfig:
    parser = argparse.ArgumentParser(
        prog="python total_simulation.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "量子仿真入口：支持单次/多次仿真与 HOM 扫描。"
        ),
        epilog=(
            "用法示例:\n"
            "  python total_simulation.py --runs 1 --shots 1\n"
            "  python total_simulation.py --runs 30 --jobs 8 --plot-all\n"
            "  python total_simulation.py --mode HOM --runs 100 --tau-start -10 --tau-end 10 "
            "--tau-step 0.5 --window-ns 70\n"
        ),
    )
    parser.add_argument("--runs", "--n-runs", dest="n_runs", type=int, help="仿真 run 次数（默认 1）")
    parser.add_argument("--shots", "--shots-per-run", dest="shots_per_run", type=int, help="每个 run 的探测采样次数（默认 1）")
    parser.add_argument("--jobs", dest="jobs", type=int, help="并行进程数（默认 1）")
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
    config.run.jobs = args.jobs if args.jobs is not None else config.run.jobs
    config.mode = (args.mode or config.mode).upper()
    config.fiber.noise_enabled = not args.no_fiber_noise

    if config.run.runs < 1:
        parser.error("N_runs 必须 >= 1")
    if config.run.shots_per_run < 1:
        parser.error("shots_per_run 必须 >= 1")
    if config.run.jobs < 1:
        parser.error("jobs 必须 >= 1")

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

    if config.mode == "HOM":
        config.hom = parse_hom_cli(args, parser)
    else:
        validate_no_hom_args(args, parser)
        config.hom = None

    config.run.plot_all = bool(args.plot_all)
    config.run.debug = bool(args.debug)
    return config


def main():
    """
    主函数：CLI 参数解析与任务调度。

    用法概览（与 --help 保持一致）：
    - 单次仿真：
      python total_simulation.py --runs 1 --shots 1
    - 多次仿真 + 并行：
      python total_simulation.py --runs 30 --jobs 8 --plot-all
    - HOM 扫描：
      python total_simulation.py --mode HOM --runs 100 --tau-start -10 --tau-end 10 --tau-step 0.5 --window-ns 70

    常用参数说明：
    - --runs / --shots / --jobs：控制仿真次数、采样次数与并行进程数
    - --mode HOM：进入 HOM 扫描模式（需提供 τ 参数）
    - --no-fiber-noise / --ideal-det：关闭光纤噪声或使用理想探测
    - --enum-mode：成功事件枚举（dark/no-dark）
    """
    # 目的：写入汇总行并排序；公式：success_rate = success / shots。
    def _finalize_combined_summary(
        summary_path: Path,
        lock_path: Path,
        stats: dict,
        n_runs: int,
        shots_per_run: int,
        metrics: Optional[dict],
    ) -> None:
        # 目的：格式化 Counter 输出，便于 CSV/日志阅读。
        def _format_counter(counter) -> str:
            if not counter:
                return "-"
            parts = []
            for key in sorted(counter.keys()):
                parts.append(f"{key}:{counter[key]}")
            return ",".join(parts)

        # 目的：格式化可选统计字段；规则：缺失/None 输出空。
        def _format_metric(key: str, fmt: str) -> str:
            if not metrics or key not in metrics:
                return ""
            value = metrics.get(key)
            if value is None:
                return ""
            return format(value, fmt)

        @contextmanager
        # 目的：并发写汇总表时互斥；规则：锁超时 stale_s 则回收。
        def _file_lock(stale_s: float = 120.0):
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
            _format_metric("dark_rate_intrinsic_hz", ".3f"),
            _format_metric("dark_rate_bg_hz", ".3f"),
            _format_metric("p_dark_intrinsic", ".8f"),
            _format_metric("p_bg", ".8f"),
            _format_metric("p_noise", ".8f"),
            _format_metric("p_no_loss", ".8f"),
            _format_metric("p_arrive", ".8f"),
            _format_metric("p_success_given_arrival", ".8f"),
            _format_metric("p_success_all", ".8f"),
            _format_metric("p_success_true", ".8f"),
            _format_metric("p_success_false", ".8f"),
            _format_metric("p_success_no_dark", ".8f"),
            _format_metric("p_false_approx", ".8f"),
            _format_metric("false_fraction", ".6f"),
            _format_metric("false_fraction_approx", ".6f"),
            _format_metric("p_qubit_emit", ".6f"),
            _format_metric("fidelity_true", ".6f"),
            _format_metric("fidelity_false", ".6f"),
            _format_metric("fidelity_all", ".6f"),
            _format_metric("fidelity_no_dark", ".6f"),
        ]
        with _file_lock():
            with open(summary_path, 'r', encoding='utf-8', newline='') as file:
                rows = list(csv.reader(file))
            if not rows:
                with open(summary_path, 'w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(single_run.SUMMARY_HEADER)
                    writer.writerow([""] * len(single_run.SUMMARY_HEADER))
                with open(summary_path, 'r', encoding='utf-8', newline='') as file:
                    rows = list(csv.reader(file))
            header = rows[0] if rows else single_run.SUMMARY_HEADER
            if len(rows) < 2:
                rows = [header, [""] * len(header)]
            summary_row = [""] * len(header)
            try:
                start_idx = header.index("runs")
            except ValueError:
                start_idx = single_run.SUMMARY_HEADER.index("runs")
            for offset, value in enumerate(summary_values):
                if start_idx + offset < len(summary_row):
                    summary_row[start_idx + offset] = value
            data_rows = rows[2:] if len(rows) > 2 else []

            # 目的：按 run/shot 排序（保证输出稳定）。
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

    # 目的：成功率/保真度累计器（update + finalize），减少分散函数。
    acc = {
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

    # 目的：累计每个 run 的成功率/保真度。
    def acc_update(metrics: dict) -> None:
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

    # 目的：由累计量生成平均指标；公式：F = Σ(p_i F_i) / Σ p_i。
    def acc_finalize() -> dict:
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
        p_success_given_arrival = (
            acc["p_success_true_sum"] / acc["p_arrive_sum"]
            if acc["p_arrive_sum"] > 0
            else 0.0
        )
        dark_rate_intrinsic_hz = acc["dark_rate_intrinsic_sum"] / runs
        dark_rate_bg_hz = acc["dark_rate_bg_sum"] / runs
        p_dark_intrinsic = acc["p_dark_intrinsic_sum"] / runs
        p_bg = acc["p_bg_sum"] / runs
        p_noise = acc["p_noise_sum"] / runs
        p_qubit_emit = acc["p_qubit_emit_sum"] / runs

        fidelity_all = (
            acc["fidelity_weighted_sum"] / acc["p_success_sum"]
            if acc["p_success_sum"] > 0
            else 0.0
        )
        fidelity_true = (
            acc["fidelity_true_weighted_sum"] / acc["p_success_true_sum"]
            if acc["p_success_true_sum"] > 0
            else 0.0
        )
        fidelity_false = (
            acc["fidelity_false_weighted_sum"] / acc["p_success_false_sum"]
            if acc["p_success_false_sum"] > 0
            else 0.0
        )
        fidelity_no_dark = None
        if no_dark_runs > 0 and acc["p_success_no_dark_sum"] > 0:
            fidelity_no_dark = (
                acc["fidelity_no_dark_weighted_sum"] / acc["p_success_no_dark_sum"]
            )

        p_false_approx = None
        if p_success_no_dark is not None:
            p_false_approx = max(0.0, p_success_all - p_success_no_dark)
        false_fraction = (p_success_false / p_success_all) if p_success_all > 0 else 0.0
        false_fraction_approx = None
        if p_false_approx is not None:
            false_fraction_approx = (
                p_false_approx / p_success_all if p_success_all > 0 else 0.0
            )

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

    config = _parse_run_params(sys.argv)
    run_cfg = config.run
    n_runs = run_cfg.runs
    shots_per_run = run_cfg.shots_per_run
    jobs = run_cfg.jobs

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.mode == "HOM":
        print(f"Output directory: {output_dir}")
        print(
            f"HOM 模式: runs={n_runs}, shots_per_run={shots_per_run}, jobs={jobs}, "
            f"window_ns={config.hom.window_ns if config.hom else None}"
        )
        run_hom_experiment(
            output_dir=output_dir,
            config=config,
        )
        return

    # 调试开关交给 single_run
    single_run.DEBUG_MODE = run_cfg.debug

    # 统一的点击结果汇总文件
    clicks_summary_path = output_dir / "all_clicks_summary.csv"
    clicks_lock_path = output_dir / ".all_clicks_summary.lock"
    with open(clicks_summary_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(single_run.SUMMARY_HEADER)
        writer.writerow([""] * len(single_run.SUMMARY_HEADER))

    print(f"Output directory: {output_dir}")
    print(f"将运行 {n_runs} 次仿真，每次 {shots_per_run} 次探测采样...")
    jobs = max(1, min(jobs, n_runs, os.cpu_count() or 1))
    run_cfg.jobs = jobs
    print(f"并行进程数: {jobs}")
    print(f"成功事件枚举模式: {run_cfg.enum_mode}")
    print(f"绘图模式: {'全部run' if run_cfg.plot_all else '仅单run'}")
    print(f"光纤噪声: {'开启' if config.fiber.noise_enabled else '关闭'}")
    print(
        f"探测参数: eta_det={config.detector.eta_det:.3f} | "
        f"理想探测={'是' if config.detector.ideal_det else '否'}"
    )

    rng = np.random.default_rng()
    run_indices = list(range(1, n_runs + 1))

    def _build_task(run_index: int, mirror_console: bool, show_plots: bool) -> tuple:
        # 目的：统一子任务参数打包，减少重复拼装。
        return (
            output_dir,
            run_index,
            config,
            clicks_summary_path,
            clicks_lock_path,
            mirror_console,
            show_plots,
        )

    focus_task = None
    if jobs > 1 and run_indices:
        focus_run = int(rng.choice(run_indices))
        run_indices = [r for r in run_indices if r != focus_run]
        focus_task = _build_task(focus_run, True, True)
        print(f"并行模式: 前台输出绑定 run{focus_run:03d}")

    overall_stats = {
        "shots": 0,
        "success": 0,
        "bell": Counter(),
        "clicks": Counter(),
    }

    if jobs <= 1:
        tasks = deque(_build_task(r, True, True) for r in run_indices)
        focus_task = None
    else:
        tasks = deque(_build_task(r, False, False) for r in run_indices)

    completed = 0
    progress_every = max(1, n_runs // 10)

    def _next_task() -> Optional[tuple]:
        # 目的：从队列取下一个任务。
        if not tasks:
            return None
        return tasks.popleft()

    def _on_result(task: tuple, result: tuple) -> None:
        # 目的：汇总子任务输出；公式：success_rate 在 finalize 中计算。
        nonlocal completed
        run_index, run_stats, success_metrics = result
        completed += 1
        if run_stats is not None:
            overall_stats["shots"] += run_stats["shots"]
            overall_stats["success"] += run_stats["success"]
            overall_stats["bell"].update(run_stats["bell"])
            overall_stats["clicks"].update(run_stats["clicks"])
        if success_metrics is not None:
            acc_update(success_metrics)
        print(f"[完成] run{run_index:03d}", flush=True)
        if completed % progress_every == 0 or completed == n_runs:
            print(f"[进度] {completed}/{n_runs}", flush=True)

    run_task_queue(
        jobs=jobs,
        task_fn=single_run._run_single_simulation_task,
        next_task=_next_task,
        on_result=_on_result,
        focus_task=focus_task,
    )

    total_success_metrics = acc_finalize()
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
