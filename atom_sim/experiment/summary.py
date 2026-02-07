# -*- coding: utf-8 -*-
"""
SUMMARY 任务：汇总 results 目录到 CSV。
"""

import csv
import json
import re
from types import SimpleNamespace

from .common import SimConfig, _compute_window_bins
from .hom import _is_port_samepol_coincidence


def _safe_num(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finalize_group_summary(group: dict) -> dict:
    runs_total = int(group["runs_total"])
    shots_total = int(group["shots_total"])

    p_arrive_avg = (group["p_arrive_sum"] / runs_total) if runs_total > 0 else 0.0
    p_success_abs_avg = (group["p_success_abs_sum"] / runs_total) if runs_total > 0 else 0.0
    p_success_true_abs_avg = (group["p_success_true_abs_sum"] / runs_total) if runs_total > 0 else 0.0
    p_success_false_abs_avg = (group["p_success_false_abs_sum"] / runs_total) if runs_total > 0 else 0.0

    p_success_true_given_arrival_global = (
        group["p_success_true_abs_sum"] / group["p_arrive_sum"]
        if group["p_arrive_sum"] > 0
        else 0.0
    )
    false_fraction_global = (
        group["p_success_false_abs_sum"] / group["p_success_abs_sum"]
        if group["p_success_abs_sum"] > 0
        else 0.0
    )

    fidelity_all_avg = (group["fidelity_all_sum"] / runs_total) if runs_total > 0 else 0.0
    fidelity_true_avg = (group["fidelity_true_sum"] / runs_total) if runs_total > 0 else 0.0
    fidelity_false_avg = (group["fidelity_false_sum"] / runs_total) if runs_total > 0 else 0.0
    corr_exx_avg = (group["corr_exx_sum"] / runs_total) if runs_total > 0 else 0.0
    corr_eyy_avg = (group["corr_eyy_sum"] / runs_total) if runs_total > 0 else 0.0
    corr_ezz_avg = (group["corr_ezz_sum"] / runs_total) if runs_total > 0 else 0.0
    chsh_s_max_avg = (group["chsh_s_max_sum"] / runs_total) if runs_total > 0 else 0.0

    return {
        "window_ns": group["window_ns"],
        "runs_target": group["runs_target"],
        "runs_total": runs_total,
        "shots_total": shots_total,
        "p_arrive_avg": p_arrive_avg,
        "p_success_abs_avg": p_success_abs_avg,
        "p_success_true_abs_avg": p_success_true_abs_avg,
        "p_success_false_abs_avg": p_success_false_abs_avg,
        "p_success_true_given_arrival_global": p_success_true_given_arrival_global,
        "false_fraction_global": false_fraction_global,
        "fidelity_all_avg": fidelity_all_avg,
        "fidelity_true_avg": fidelity_true_avg,
        "fidelity_false_avg": fidelity_false_avg,
        "corr_exx_avg": corr_exx_avg,
        "corr_eyy_avg": corr_eyy_avg,
        "corr_ezz_avg": corr_ezz_avg,
        "chsh_s_max_avg": chsh_s_max_avg,
    }


def _write_window_scan_summary(paths: dict, config: SimConfig) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    trials_path = summary_dir / "window_scan_trials.csv"
    runs_path = summary_dir / "window_scan_runs.csv"
    summary_path = summary_dir / "window_scan_summary.csv"

    groups = {}

    with open(trials_path, "w", encoding="utf-8", newline="") as trials_file, open(
        runs_path, "w", encoding="utf-8", newline=""
    ) as runs_file:
        trials_writer = csv.writer(trials_file)
        runs_writer = csv.writer(runs_file)

        trials_writer.writerow([
            "window_ns",
            "run_index",
            "shot_index",
            "success",
            "bell",
            "p_arrive",
            "p_arrive_11",
            "p_arrive_same_arm",
            "p_arrive_20",
            "p_arrive_02",
            "p_success_abs",
            "p_success_true_abs",
            "p_success_false_abs",
            "p_success_true_given_arrival",
            "fidelity_all",
            "fidelity_true",
            "fidelity_false",
            "false_fraction",
            "corr_exx",
            "corr_eyy",
            "corr_ezz",
            "chsh_s_max",
            "H1_bin",
            "V1_bin",
            "H2_bin",
            "V2_bin",
            "H1_dark",
            "V1_dark",
            "H2_dark",
            "V2_dark",
        ])

        runs_writer.writerow([
            "id",
            "window_ns",
            "run_index",
            "shots",
            "success",
            "p_arrive",
            "p_arrive_11",
            "p_arrive_same_arm",
            "p_arrive_20",
            "p_arrive_02",
            "p_success_abs",
            "p_success_true_abs",
            "p_success_false_abs",
            "p_success_true_given_arrival",
            "fidelity_all",
            "fidelity_true",
            "fidelity_false",
            "false_fraction",
            "corr_exx",
            "corr_eyy",
            "corr_ezz",
            "chsh_s_max",
            "timestamp",
        ])

        for meta_path in sorted(results_dir.glob("result_*/meta.json")):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("mode") != "WINDOW_SCAN":
                continue

            tid = data.get("id", "")
            metrics = data.get("metrics", {})
            run_index = int(metrics.get("run_index", 0) or 0)
            match = re.match(r"wscan_w_([+-]?\d+\.\d+)_run_(\d+)", tid)
            if match:
                window_ns = float(match.group(1))
                run_index = int(match.group(2))
            else:
                window_ns = float(metrics.get("window_ns", 0.0) or 0.0)

            p_arrive = _safe_num(metrics.get("p_arrive"))
            p_arrive_11 = _safe_num(metrics.get("p_arrive_11"))
            p_arrive_same_arm = _safe_num(metrics.get("p_arrive_same_arm"))
            p_arrive_20 = _safe_num(metrics.get("p_arrive_20"))
            p_arrive_02 = _safe_num(metrics.get("p_arrive_02"))
            p_success_abs = _safe_num(metrics.get("p_success_abs"))
            p_success_true_abs = _safe_num(metrics.get("p_success_true_abs"))
            p_success_false_abs = _safe_num(metrics.get("p_success_false_abs"))
            p_success_true_given_arrival = _safe_num(metrics.get("p_success_true_given_arrival"))
            fidelity_all = _safe_num(metrics.get("fidelity_all"))
            fidelity_true = _safe_num(metrics.get("fidelity_true"))
            fidelity_false = _safe_num(metrics.get("fidelity_false"))
            false_fraction = _safe_num(metrics.get("false_fraction"))
            corr_exx = _safe_num(metrics.get("corr_exx"))
            corr_eyy = _safe_num(metrics.get("corr_eyy"))
            corr_ezz = _safe_num(metrics.get("corr_ezz"))
            chsh_s_max = _safe_num(metrics.get("chsh_s_max"))
            shots = int(metrics.get("shots", 0) or 0)
            success = int(metrics.get("success", 0) or 0)

            runs_writer.writerow([
                tid,
                window_ns,
                run_index,
                shots,
                success,
                p_arrive,
                p_arrive_11,
                p_arrive_same_arm,
                p_arrive_20,
                p_arrive_02,
                p_success_abs,
                p_success_true_abs,
                p_success_false_abs,
                p_success_true_given_arrival,
                fidelity_all,
                fidelity_true,
                fidelity_false,
                false_fraction,
                corr_exx,
                corr_eyy,
                corr_ezz,
                chsh_s_max,
                data.get("timestamp"),
            ])

            group_key = f"{window_ns:.9f}"
            group = groups.setdefault(
                group_key,
                {
                    "window_ns": window_ns,
                    "runs_target": config.run.runs,
                    "runs_total": 0,
                    "shots_total": 0,
                    "p_arrive_sum": 0.0,
                    "p_success_abs_sum": 0.0,
                    "p_success_true_abs_sum": 0.0,
                    "p_success_false_abs_sum": 0.0,
                    "fidelity_all_sum": 0.0,
                    "fidelity_true_sum": 0.0,
                    "fidelity_false_sum": 0.0,
                    "corr_exx_sum": 0.0,
                    "corr_eyy_sum": 0.0,
                    "corr_ezz_sum": 0.0,
                    "chsh_s_max_sum": 0.0,
                },
            )

            group["runs_total"] += 1
            group["shots_total"] += shots
            group["p_arrive_sum"] += p_arrive or 0.0
            group["p_success_abs_sum"] += p_success_abs or 0.0
            group["p_success_true_abs_sum"] += p_success_true_abs or 0.0
            group["p_success_false_abs_sum"] += p_success_false_abs or 0.0
            group["fidelity_all_sum"] += fidelity_all or 0.0
            group["fidelity_true_sum"] += fidelity_true or 0.0
            group["fidelity_false_sum"] += fidelity_false or 0.0
            group["corr_exx_sum"] += corr_exx or 0.0
            group["corr_eyy_sum"] += corr_eyy or 0.0
            group["corr_ezz_sum"] += corr_ezz or 0.0
            group["chsh_s_max_sum"] += chsh_s_max or 0.0

            clicks_path = meta_path.parent / "raw" / "clicks.json"
            clicks = []
            if clicks_path.exists():
                try:
                    clicks = json.loads(clicks_path.read_text(encoding="utf-8")).get("clicks", [])
                except Exception:
                    clicks = []

            if not clicks:
                trials_writer.writerow([
                    window_ns,
                    run_index,
                    -1,
                    "",
                    "",
                    p_arrive,
                    p_arrive_11,
                    p_arrive_same_arm,
                    p_arrive_20,
                    p_arrive_02,
                    p_success_abs,
                    p_success_true_abs,
                    p_success_false_abs,
                    p_success_true_given_arrival,
                    fidelity_all,
                    fidelity_true,
                    fidelity_false,
                    false_fraction,
                    corr_exx,
                    corr_eyy,
                    corr_ezz,
                    chsh_s_max,
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
                shot_success = record.get("success")
                bell = record.get("bell")
                shot_clicks = record.get("clicks", [])
                bins = {"H1": "", "V1": "", "H2": "", "V2": ""}
                darks = {"H1": "", "V1": "", "H2": "", "V2": ""}
                for click in shot_clicks:
                    if len(click) < 3:
                        raise ValueError("WINDOW_SCAN clicks 至少包含 (det, bin, is_dark)")
                    det = click[0]
                    bin_idx = click[1]
                    is_dark = bool(click[2])
                    if det in bins:
                        bins[det] = f"{bin_idx}" if bins[det] == "" else f"{bins[det]};{bin_idx}"
                        flag = "1" if is_dark else "0"
                        darks[det] = flag if darks[det] == "" else f"{darks[det]};{flag}"

                trials_writer.writerow([
                    window_ns,
                    run_index,
                    shot_idx,
                    shot_success,
                    bell,
                    p_arrive,
                    p_arrive_11,
                    p_arrive_same_arm,
                    p_arrive_20,
                    p_arrive_02,
                    p_success_abs,
                    p_success_true_abs,
                    p_success_false_abs,
                    p_success_true_given_arrival,
                    fidelity_all,
                    fidelity_true,
                    fidelity_false,
                    false_fraction,
                    corr_exx,
                    corr_eyy,
                    corr_ezz,
                    chsh_s_max,
                    bins["H1"],
                    bins["V1"],
                    bins["H2"],
                    bins["V2"],
                    darks["H1"],
                    darks["V1"],
                    darks["H2"],
                    darks["V2"],
                ])

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow([
            "window_ns",
            "runs_target",
            "runs_total",
            "shots_total",
            "p_arrive_avg",
            "p_success_abs_avg",
            "p_success_true_abs_avg",
            "p_success_false_abs_avg",
            "p_success_true_given_arrival_global",
            "false_fraction_global",
            "fidelity_all_avg",
            "fidelity_true_avg",
            "fidelity_false_avg",
            "corr_exx_avg",
            "corr_eyy_avg",
            "corr_ezz_avg",
            "chsh_s_max_avg",
        ])
        for key in sorted(groups.keys(), key=lambda item: float(item)):
            row = _finalize_group_summary(groups[key])
            summary_writer.writerow([
                row["window_ns"],
                row["runs_target"],
                row["runs_total"],
                row["shots_total"],
                row["p_arrive_avg"],
                row["p_success_abs_avg"],
                row["p_success_true_abs_avg"],
                row["p_success_false_abs_avg"],
                row["p_success_true_given_arrival_global"],
                row["false_fraction_global"],
                row["fidelity_all_avg"],
                row["fidelity_true_avg"],
                row["fidelity_false_avg"],
                row["corr_exx_avg"],
                row["corr_eyy_avg"],
                row["corr_ezz_avg"],
                row["chsh_s_max_avg"],
            ])


def write_summary(task_type: str, paths: dict, config: SimConfig) -> None:
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
    if task_type == "WINDOW_SCAN":
        _write_window_scan_summary(paths=paths, config=config)
        return

    if task_type == "SIM":
        trials_path = summary_dir / "sim_trials.csv"
        with open(trials_path, "w", encoding="utf-8", newline="") as trials_file:
            trials_writer = csv.writer(trials_file)
            trials_writer.writerow([
                "task_mode",
                "window_ns",
                "run_index",
                "shot_index",
                "success",
                "bell",
                "p_arrive",
                "p_arrive_11",
                "p_arrive_same_arm",
                "p_arrive_20",
                "p_arrive_02",
                "p_success_abs",
                "p_success_true_abs",
                "p_success_false_abs",
                "p_success_true_given_arrival",
                "fidelity_all",
                "fidelity_true",
                "fidelity_false",
                "false_fraction",
                "corr_exx",
                "corr_eyy",
                "corr_ezz",
                "chsh_s_max",
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
                metrics = data.get("metrics", {})
                run_index = int(metrics.get("run_index", 0) or 0)
                m = re.match(r"sim_run_(\d+)", tid)
                if m:
                    run_index = int(m.group(1))
                task_mode = data.get("mode")
                window_ns = metrics.get("window_ns")
                p_arrive = metrics.get("p_arrive")
                p_arrive_11 = metrics.get("p_arrive_11")
                p_arrive_same_arm = metrics.get("p_arrive_same_arm")
                p_arrive_20 = metrics.get("p_arrive_20")
                p_arrive_02 = metrics.get("p_arrive_02")
                p_success_abs = metrics.get("p_success_abs")
                p_success_true_abs = metrics.get("p_success_true_abs")
                p_success_false_abs = metrics.get("p_success_false_abs")
                p_success_true_given_arrival = metrics.get("p_success_true_given_arrival")
                fidelity_all = metrics.get("fidelity_all")
                fidelity_true = metrics.get("fidelity_true")
                fidelity_false = metrics.get("fidelity_false")
                false_fraction = metrics.get("false_fraction")
                corr_exx = metrics.get("corr_exx")
                corr_eyy = metrics.get("corr_eyy")
                corr_ezz = metrics.get("corr_ezz")
                chsh_s_max = metrics.get("chsh_s_max")
                clicks_path = meta_path.parent / "raw" / "clicks.json"
                clicks = []
                if clicks_path.exists():
                    try:
                        clicks = json.loads(clicks_path.read_text(encoding="utf-8")).get("clicks", [])
                    except Exception:
                        clicks = []
                if not clicks:
                    trials_writer.writerow([
                        task_mode,
                        window_ns,
                        run_index,
                        -1,
                        "",
                        "",
                        p_arrive,
                        p_arrive_11,
                        p_arrive_same_arm,
                        p_arrive_20,
                        p_arrive_02,
                        p_success_abs,
                        p_success_true_abs,
                        p_success_false_abs,
                        p_success_true_given_arrival,
                        fidelity_all,
                        fidelity_true,
                        fidelity_false,
                        false_fraction,
                        corr_exx,
                        corr_eyy,
                        corr_ezz,
                        chsh_s_max,
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
                        task_mode,
                        window_ns,
                        run_index,
                        shot_idx,
                        success,
                        bell,
                        p_arrive,
                        p_arrive_11,
                        p_arrive_same_arm,
                        p_arrive_20,
                        p_arrive_02,
                        p_success_abs,
                        p_success_true_abs,
                        p_success_false_abs,
                        p_success_true_given_arrival,
                        fidelity_all,
                        fidelity_true,
                        fidelity_false,
                        false_fraction,
                        corr_exx,
                        corr_eyy,
                        corr_ezz,
                        chsh_s_max,
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
            "window_ns",
            "p_arrive",
            "p_arrive_11",
            "p_arrive_same_arm",
            "p_arrive_20",
            "p_arrive_02",
            "coinc",
            "p_success_abs",
            "p_success_true_abs",
            "p_success_false_abs",
            "p_success_true_given_arrival",
            "fidelity_all",
            "fidelity_true",
            "fidelity_false",
            "false_fraction",
            "corr_exx",
            "corr_eyy",
            "corr_ezz",
            "chsh_s_max",
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
                m.get("window_ns"),
                m.get("p_arrive"),
                m.get("p_arrive_11"),
                m.get("p_arrive_same_arm"),
                m.get("p_arrive_20"),
                m.get("p_arrive_02"),
                m.get("coinc"),
                m.get("p_success_abs"),
                m.get("p_success_true_abs"),
                m.get("p_success_false_abs"),
                m.get("p_success_true_given_arrival"),
                m.get("fidelity_all"),
                m.get("fidelity_true"),
                m.get("fidelity_false"),
                m.get("false_fraction"),
                m.get("corr_exx"),
                m.get("corr_eyy"),
                m.get("corr_ezz"),
                m.get("chsh_s_max"),
                data.get("timestamp"),
            ])
