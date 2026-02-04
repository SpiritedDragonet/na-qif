# -*- coding: utf-8 -*-
"""
Single-run simulation workflow and summary helpers.
"""

from __future__ import annotations

import sys
import os
import csv
import time
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter
from types import SimpleNamespace

import numpy as np

from ..simulation import (
    run_detection_pipeline,
    compute_fidelity_with_bell,
)
from ..visualization import plot_dual_arm_heatmap
from ..physics.gates import bs_gate_6d
from .common import (
    SimConfig,
    PipelineHooks,
    run_emission_to_bs,
    _compute_window_bins,
    _compute_noise_params,
)

# Debug toggle (default False)
DEBUG_MODE = False

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

def _run_single_trial(
    rng: Optional[np.random.Generator],
    config: SimConfig,
    delay_ns: Optional[float],
    delay_jitter_ns: Optional[float],
    verbose: bool,
    debug: bool,
    hooks: Optional[PipelineHooks],
):
    """
    目的：抽出最小可复用的物理流程（发射->QFC->滤波->光纤->退相干->BS并入测量）。
    规则：delay_ns/delay_jitter_ns 优先采用显式传入，否则取配置默认。
    """
    # 该函数只负责“物理链路”部分，不包含探测统计；
    # 便于 SIM/HOM 复用并减少重复代码。
    run_rng = rng or np.random.default_rng()
    pipe = run_emission_to_bs(
        emission=config.emission,
        rng=run_rng,
        fiber=config.fiber,
        delay_ns=delay_ns,
        delay_jitter_ns=delay_jitter_ns,
        verbose=verbose,
        hooks=hooks,
        record_timings=debug,
    )
    return pipe

def _run_single_simulation_core(
    output_dir: Path,
    run_index: int,
    config: SimConfig,
    summary_path: Optional[Path],
    summary_lock_path: Optional[Path],
    show_plots: bool,
    plot_dir: Optional[Path] = None,
    run_tag: Optional[str] = None,
    seed: Optional[int] = None,
):
    run_cfg = config.run
    n_runs = run_cfg.runs
    shots_per_run = run_cfg.shots_per_run
    plot_all = run_cfg.plot_all
    plot_enabled = run_cfg.plot_enabled
    enum_mode = run_cfg.enum_mode
    noise_cfg = config.noise
    eta_det = config.detector.eta_det
    ideal_det = config.detector.ideal_det
    run_tag = f"run{run_index:03d}"
    success_metrics = None
    stage_total = 6
    plot_gate = {"claimed": False, "paths": []}

    @contextmanager
    # 目的：并发写 CSV/占位时的互斥锁；规则：锁文件超时 stale_s 视为僵死并回收。
    def _file_lock(lock_path: Path, stale_s: float = 120.0) -> None:
        # --------------------------------------------------------------
        # 简易文件锁：
        #   - 创建 lock 文件即视为占用
        #   - 超过 stale_s 未更新视为僵死锁
        # 目的是保证多进程写 CSV 时不互相覆盖。
        # --------------------------------------------------------------
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

    def _append_click_summary(
        summary_path: Path,
        lock_path: Path,
        run_index: int,
        shot_index: int,
        det_result,
        metrics: Optional[dict],
    ) -> None:
        if summary_path is None or lock_path is None:
            return
        # 目的：记录单次点击结果。
        # 公式：F_full = <Bell|ρ|Bell>（未归一化保真度）
        # 这里的 ρ 为“点击记录 r”条件化后的原子态（未归一化）。
        def _fmt(key: str, fmt: str) -> str:
            if not metrics or key not in metrics:
                return ""
            value = metrics.get(key)
            if value is None:
                return ""
            return format(value, fmt)

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
            _fmt("dark_rate_intrinsic_hz", ".3f"),
            _fmt("dark_rate_bg_hz", ".3f"),
            _fmt("p_dark_intrinsic", ".8f"),
            _fmt("p_bg", ".8f"),
            _fmt("p_noise", ".8f"),
            _fmt("p_no_loss", ".8f"),
            _fmt("p_arrive", ".8f"),
            _fmt("p_success_given_arrival", ".8f"),
            _fmt("p_success_all", ".8f"),
            _fmt("p_success_true", ".8f"),
            _fmt("p_success_false", ".8f"),
            _fmt("p_success_no_dark", ".8f"),
            _fmt("p_false_approx", ".8f"),
            _fmt("false_fraction", ".6f"),
            _fmt("false_fraction_approx", ".6f"),
            _fmt("fidelity_true", ".6f"),
            _fmt("fidelity_false", ".6f"),
            _fmt("fidelity_all", ".6f"),
            _fmt("fidelity_no_dark", ".6f"),
            _fmt("p_qubit_emit", ".6f"),
            fidelity_shot_full,
        ]
        if len(row) < len(SUMMARY_HEADER):
            row += [""] * (len(SUMMARY_HEADER) - len(row))
        with _file_lock(lock_path):
            with open(summary_path, 'a', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
    # 目的：落盘保存成功率/保真度等关键指标，便于复现与后处理。
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
            if metrics.get("p_no_loss") is not None:
                file.write(f"p_no_loss = {metrics['p_no_loss']:.8f}\n")
            else:
                file.write("p_no_loss = N/A\n")

            if metrics.get("p_arrive") is not None:
                file.write(f"p_arrive = {metrics['p_arrive']:.8f}\n")
            else:
                file.write("p_arrive = N/A\n")
            if metrics.get("p_success_no_dark") is not None:
                file.write(f"p_success_no_dark = {metrics['p_success_no_dark']:.8f}\n")
                file.write(f"fidelity_no_dark = {metrics['fidelity_no_dark']:.6f}\n")
            else:
                file.write("p_success_no_dark = N/A\n")
                file.write("fidelity_no_dark = N/A\n")
            if metrics.get("p_success_all") is not None:
                file.write(f"p_success_all = {metrics['p_success_all']:.8f}\n")
            else:
                file.write("p_success_all = N/A\n")
            if metrics.get("fidelity_all") is not None:
                file.write(f"fidelity_all = {metrics['fidelity_all']:.6f}\n")
            else:
                file.write("fidelity_all = N/A\n")
            if metrics.get("p_false_approx") is not None:
                file.write(f"p_false_approx = {metrics['p_false_approx']:.8f}\n")
                file.write(f"false_fraction_approx = {metrics['false_fraction_approx']:.6f}\n")
            else:
                file.write("p_false_approx = N/A\n")
                file.write("false_fraction_approx = N/A\n")

            if metrics.get("p_success_true") is not None:
                file.write(f"p_success_true = {metrics['p_success_true']:.8f}\n")
            else:
                file.write("p_success_true = N/A\n")
            if metrics.get("p_success_false") is not None:
                file.write(f"p_success_false = {metrics['p_success_false']:.8f}\n")
            else:
                file.write("p_success_false = N/A\n")
            if metrics.get("p_success_given_arrival") is not None:
                file.write(f"p_success_given_arrival = {metrics['p_success_given_arrival']:.8f}\n")
            else:
                file.write("p_success_given_arrival = N/A\n")
            if metrics.get("false_fraction") is not None:
                file.write(f"false_fraction = {metrics['false_fraction']:.6f}\n")
            else:
                file.write("false_fraction = N/A\n")
            if metrics.get("fidelity_true") is not None:
                file.write(f"fidelity_true = {metrics['fidelity_true']:.6f}\n")
            else:
                file.write("fidelity_true = N/A\n")
            if metrics.get("fidelity_false") is not None:
                file.write(f"fidelity_false = {metrics['fidelity_false']:.6f}\n")
            else:
                file.write("fidelity_false = N/A\n")
        return output_path

    # 目的：绘图占位与清理逻辑集中，避免散落多个函数。
    # 目的：占用/判断是否允许绘图；规则：plot_all 或首次抢占成功。
    def _plot_gate_allow() -> bool:
        # --------------------------------------------------------------
        # 逻辑：默认只保留一个 run 的图，避免多进程 I/O 爆炸。
        # plot_all=True 时不做限制。
        # --------------------------------------------------------------
        if not plot_enabled:
            return False
        if plot_all:
            return True
        if plot_gate["claimed"]:
            return True
        marker_path = output_dir / ".plot_run_claimed"
        lock_path = output_dir / ".plot_run_claimed.lock"
        with _file_lock(lock_path):
            if marker_path.exists():
                return False
            marker_path.write_text(run_tag, encoding="utf-8")
        plot_gate["claimed"] = True
        return True

    # 目的：记录本 run 的图路径（仅占位 run 保留）。
    def _plot_gate_register(path: Path) -> None:
        # 记录当前 run 产出的图路径，便于后续清理。
        if plot_all:
            return
        if plot_gate["claimed"]:
            plot_gate["paths"].append(path)

    # 目的：清理非占位 run 图 + 释放占位锁。
    def _plot_gate_finalize() -> None:
        # 清理非占位 run 产生的图，避免磁盘膨胀。
        if plot_all or not plot_gate["claimed"]:
            return
        for path in plot_gate["paths"]:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
        marker_path = output_dir / ".plot_run_claimed"
        lock_path = output_dir / ".plot_run_claimed.lock"
        with _file_lock(lock_path):
            if not marker_path.exists():
                return
            current = marker_path.read_text(encoding="utf-8").strip()
        if current == run_tag:
            marker_path.unlink()

    print("\n" + "=" * 80)
    print(f"Run {run_index}/{n_runs} ({run_tag})")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("运行发射 + QFC + 分束器 + 探测仿真...")

    run_rng = np.random.default_rng(seed)
    p_no_loss = 0.0

    stage_map = {
        "发射": 1,
        "QFC": 2,
        "光纤信道": 3,
        "分束器(测量端) + 诊断/可视化": 4,
        "成功事件统计 (POVM)": 5,
        "POVM抽样": 6,
    }

    # 目的：输出阶段日志；公式：显示 "阶段 idx / 总阶段数"。
    def _on_stage(label: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        idx = stage_map[label]
        print(f"[{run_tag} {ts}] [阶段 {idx}/{stage_total}] {label}")

    # 目的：统一生成阶段可视化与调试快照，减少重复代码。
    def _make_plot_hook(
        stage_name: str,
        file_suffix: str,
        show_atomic: bool,
        use_emission_obj: bool,
        use_time_grid: bool,
        debug_stage: str,
        step_index: int,
        bs_unitary: Optional[np.ndarray] = None,
    ):
        # --------------------------------------------------------------
        # 统一封装可视化 hook：
        #   - 只在“允许绘图”的 run 上执行
        #   - 输出命名规则统一，便于汇总
        #   - bs_unitary 用于 Heisenberg 端口“after BS”热图
        # --------------------------------------------------------------
        def _hook(
            emission,
            fiber_sample=None,
            qfc_params=None,
            apply_filter_780: bool = True,
            *_args,
        ):
            if _plot_gate_allow():
                print(f"\n生成{stage_name}的可视化图...")
                plot_path = plot_dir / f"{run_tag}_{file_suffix}.png"
                target = emission if use_emission_obj else emission.mps
                kwargs = dict(
                    save_path=str(plot_path),
                    show_atomic=show_atomic,
                    stage_name=stage_name,
                    show=show_plots,
                )
                if use_time_grid:
                    kwargs["time_grid"] = {"dt_s": emission.dt_s}
                if bs_unitary is not None:
                    kwargs["bs_unitary"] = bs_unitary
                kwargs["qfc_params"] = qfc_params
                kwargs["fiber_sample"] = fiber_sample
                kwargs["apply_filter_780"] = apply_filter_780
                plot_dual_arm_heatmap(target, **kwargs)
                _plot_gate_register(plot_path)
            if DEBUG_MODE:
                save_debug_info(
                    mps=emission.mps,
                    n_bins=emission.get_n_bins(),
                    stage=debug_stage,
                    output_dir=output_dir,
                    step_index=step_index,
                    run_tag=run_tag,
                )
        return _hook

    _after_emission = _make_plot_hook(
        stage_name="After Emission",
        file_suffix="1_after_emission",
        show_atomic=True,
        use_emission_obj=True,
        use_time_grid=False,
        debug_stage="After Emission",
        step_index=1,
    )
    _after_qfc_filter = _make_plot_hook(
        stage_name="After QFC",
        file_suffix="2_after_qfc",
        show_atomic=False,
        use_emission_obj=False,
        use_time_grid=True,
        debug_stage="After QFC",
        step_index=2,
    )
    _after_fiber = _make_plot_hook(
        stage_name="After Fiber",
        file_suffix="3_after_fiber",
        show_atomic=False,
        use_emission_obj=False,
        use_time_grid=True,
        debug_stage="After Fiber Channel",
        step_index=3,
    )
    _after_bs = _make_plot_hook(
        stage_name="After BS",
        file_suffix="4_after_bs",
        show_atomic=False,
        use_emission_obj=False,
        use_time_grid=True,
        debug_stage="After BS (measurement pre-state)",
        step_index=4,
        bs_unitary=bs_gate_6d(),
    )

    timings = {} if DEBUG_MODE else None
    pipe = _run_single_trial(
        rng=run_rng,
        config=config,
        delay_ns=None,
        delay_jitter_ns=None,
        verbose=True,
        debug=DEBUG_MODE,
        hooks=PipelineHooks(
            on_stage=_on_stage,
            after_emission=_after_emission,
            after_qfc_filter=_after_qfc_filter,
            after_fiber=_after_fiber,
            after_bs=_after_bs,
        ),
    )
    if DEBUG_MODE and pipe.timings:
        timings.update(pipe.timings)
    if pipe.aborted:
        include_no_dark = (enum_mode == "no-dark")
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
            "p_qubit_emit": pipe.p_qubit_emit,
            "p_false_approx": 0.0 if include_no_dark else None,
            "false_fraction": 0.0,
            "false_fraction_approx": 0.0 if include_no_dark else None,
        }
        metrics["p_no_loss"] = pipe.p_no_loss
        _plot_gate_finalize()
        reason = pipe.abort_reason or "流水线提前终止"
        print(f"\n[早停] {reason}")
        run_stats = {
            "shots": 0,
            "success": 0,
            "bell": Counter(),
            "clicks": Counter(),
        }
        det_result = SimpleNamespace(
            clicks=[],
            success=False,
            bell_state="",
            spin_state=np.zeros((4, 4), dtype=complex),
        )
        for shot_index in range(1, shots_per_run + 1):
            _append_click_summary(
                summary_path,
                summary_lock_path,
                run_index,
                shot_index,
                det_result,
                metrics,
            )
            run_stats["shots"] += 1
            run_stats["clicks"][0] += 1
        return run_stats, metrics

    result = pipe.emission
    p_qubit_emit = pipe.p_qubit_emit
    p_no_loss = pipe.p_no_loss
    t_wait_us = pipe.t_wait_us
    t2_us = pipe.t2_us
    p_dephase = pipe.p_dephase

    # =========================================================================
    # 探测
    # =========================================================================
    # 探测参数（基于 Nature 2022 实验设置）
    if ideal_det:
        eta_det = 1.0
    # 符合窗口：默认采用论文中数据分析窗口 70 ns
    coincidence_window_ns = 70.0
    bin_dt_s = result.dt_s
    bin_dt_ns = bin_dt_s * 1e9
    # 将时间窗映射到 bin 数，决定“符合”判定的最大 bin 差
    window_bins = _compute_window_bins(coincidence_window_ns, bin_dt_ns)

    # QFC 背景噪声 + 探测器本底暗计数（两者独立）
    if ideal_det:
        # 理想探测：噪声全关
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
        # 现实探测：每次 run 采样背景噪声率（正态扰动）
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
    # 重要：BS 已并入测量端。这里传入 U_BS，用 U^† E U 计算点击分布。
    bs_unitary = bs_gate_6d()

    _on_stage("成功事件统计 (POVM)")
    print(f"\n成功事件枚举模式: {enum_mode}")
    enum_no_dark = None
    enum_main = None
    samples = []
    run_stats = {
        "shots": 0,
        "success": 0,
        "bell": Counter(),
        "clicks": Counter(),
    }

    enum_start = time.perf_counter() if DEBUG_MODE else None
    if enum_mode == "no-dark":
        print("\n枚举成功事件（无暗计数）...")
        # 枚举阶段：只计算统计量，不做抽样
        enum_pipeline = run_detection_pipeline(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            p_dark=0.0,
            window_bins=window_bins,
            rng=run_rng,
            verbose=True,
            n_samples=0,
            compute_metrics=True,
            bs_unitary=bs_unitary,
            fiber_sample=pipe.fiber_sample,
            apply_filter_780=pipe.apply_filter_780,
            theta_H=pipe.qfc_theta_H,
            theta_V=pipe.qfc_theta_V,
        )
        enum_no_dark = enum_pipeline.metrics
        if p_noise > 0.0:
            print("  注意：枚举模式为 no-dark，将忽略暗计数，仅用于基线对比。")
        enum_main = enum_no_dark
        if DEBUG_MODE and timings is not None and enum_start is not None:
            timings["povm_enum"] = time.perf_counter() - enum_start

        # 使用POVM抽样运行探测和BSM（可多次采样）
        _on_stage("POVM抽样")
        print("\n运行探测和BSM（POVM抽样）...")
        detect_start = time.perf_counter() if DEBUG_MODE else None
        # 抽样阶段：按噪声概率 p_noise 生成点击记录
        sample_pipeline = run_detection_pipeline(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            p_dark=p_noise,
            window_bins=window_bins,
            rng=run_rng,
            verbose=True,
            n_samples=shots_per_run,
            compute_metrics=False,
            bs_unitary=bs_unitary,
            fiber_sample=pipe.fiber_sample,
            apply_filter_780=pipe.apply_filter_780,
            theta_H=pipe.qfc_theta_H,
            theta_V=pipe.qfc_theta_V,
        )
        samples = sample_pipeline.samples
    else:
        print("\n枚举成功事件（含暗计数）...")
        # 使用POVM抽样运行探测和BSM（可多次采样）
        _on_stage("POVM抽样")
        print("\n运行探测和BSM（POVM抽样）...")
        detect_start = time.perf_counter() if DEBUG_MODE else None
        # 直接在“含暗计数”的统计上枚举并抽样
        pipeline = run_detection_pipeline(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            p_dark=p_noise,
            window_bins=window_bins,
            rng=run_rng,
            verbose=True,
            n_samples=shots_per_run,
            compute_metrics=True,
            bs_unitary=bs_unitary,
            fiber_sample=pipe.fiber_sample,
            apply_filter_780=pipe.apply_filter_780,
            theta_H=pipe.qfc_theta_H,
            theta_V=pipe.qfc_theta_V,
        )
        enum_main = pipeline.metrics
        samples = pipeline.samples

    # 汇总统计量（跨 shots）
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
    # 误判占比：false / all
    success_metrics["false_fraction"] = (
        success_metrics["p_success_false"] / success_metrics["p_success_all"]
        if success_metrics["p_success_all"] > 0
        else 0.0
    )
    if enum_no_dark is not None:
        # 粗略估计：将暗计数引入的“额外成功”视为 false
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

    for shot_index, det_result in enumerate(samples, start=1):
        print(f"\n[shot {shot_index}/{shots_per_run}]")

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

    if DEBUG_MODE and timings is not None and detect_start is not None:
        timings["detection_total"] = time.perf_counter() - detect_start
        if shots_per_run > 0:
            timings["detection_per_shot"] = timings["detection_total"] / shots_per_run

    print(f"\n完成! 文件已保存至: {output_dir}/")
    if DEBUG_MODE and timings:
        timing_order = [
            ("emission", "发射"),
            ("qfc", "QFC"),
            ("filter_780", "780滤波"),
            ("project_1517", "1517投影"),
            ("fiber", "光纤"),
            ("dephase", "退相干"),
            ("bs", "BS"),
            ("povm_enum", "成功事件枚举"),
            ("povm_effects", "POVM构建"),
            ("detection_total", "探测抽样"),
        ]
        parts = []
        for key, label in timing_order:
            if key in timings:
                parts.append(f"{label}={timings[key]:.2f}s")
        if parts:
            print("\n[调试耗时] " + " | ".join(parts))
    return run_stats, success_metrics


def _run_single_simulation_task(
    output_dir,
    run_index,
    config,
    summary_path,
    summary_lock_path,
    mirror_console,
    show_plots,
):
    # 外部调度器动态调用入口（ProcessPool 需要顶层函数以支持序列化）。
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

    run_tag = f"run{run_index:03d}"
    log_path = output_dir / f"{run_tag}_console.log"
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
            run_stats, success_metrics = _run_single_simulation_core(
                output_dir,
                run_index,
                config,
                summary_path,
                summary_lock_path,
                show_plots,
            )
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    return run_index, run_stats, success_metrics
