# -*- coding: utf-8 -*-
"""
SUMMARY 任务：汇总 results 目录到 CSV。
"""

import csv
import json
import re
from types import SimpleNamespace

import numpy as np

from .common import SimConfig, _compute_window_bins, _compute_effective_attempt_rate_hz
from .hom import _is_port_samepol_coincidence
from .bsm_scan import BSM_PATTERN_KEYS


CORE_TASK_MODE = "CORE_TRIAL"


def _is_core_experiment(data: dict, experiment: str) -> bool:
    return str(data.get("mode", "")).upper() == CORE_TASK_MODE and str(data.get("experiment", "")).upper() == experiment.upper()


def _extract_run_index(metrics: dict, tid: str, patterns: tuple[str, ...]) -> int:
    run_index_raw = metrics.get("run_index", 0)
    try:
        return int(run_index_raw or 0)
    except Exception:
        pass
    for pattern in patterns:
        m = re.match(pattern, tid)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return 0


def _safe_num(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_DETECTOR_ORDER = ("H1", "V1", "H2", "V2")
_DETECTOR_ORDER_INDEX = {detector: idx for idx, detector in enumerate(_DETECTOR_ORDER)}


def _sort_group_value_key(value):
    if isinstance(value, (int, float, np.floating)):
        return (0, float(value))
    if value is None:
        return (2, "")
    return (1, str(value))


def _analysis_group_key(value) -> str:
    if isinstance(value, (int, float, np.floating)):
        return f"{float(value):.9f}"
    if value is None:
        return "none"
    return str(value)


def _ensure_click_analysis_group(groups: dict, group_value):
    key = _analysis_group_key(group_value)
    state = groups.get(key)
    if state is not None:
        return state
    state = {
        "group_value": group_value,
        "success_delta": {},
        "true_delta": {},
        "false_delta": {},
        "heatmap": {},
    }
    groups[key] = state
    return state


def _accumulate_bucket_weight(bucket: dict, delta_bin: int, weight: float) -> None:
    key = int(delta_bin)
    bucket[key] = float(bucket.get(key, 0.0) + float(weight))


def _parse_click_events(raw_clicks, task_name: str) -> list[SimpleNamespace]:
    if not isinstance(raw_clicks, list):
        raise ValueError(f"{task_name} clicks 必须为 list")
    events = []
    for click in raw_clicks:
        if not isinstance(click, (list, tuple)) or len(click) < 4:
            raise ValueError(f"{task_name} clicks 必须为 (det, bin, is_dark, source) 四元组")
        detector = click[0]
        if detector is None:
            raise ValueError(f"{task_name} clicks 的 detector 不能为空")
        bin_index = click[1]
        is_dark = bool(click[2])
        source = str(click[3])
        try:
            bin_value = int(bin_index)
        except Exception as exc:
            raise ValueError(f"{task_name} clicks 的 bin 必须可转为 int: {bin_index}") from exc
        events.append(
            SimpleNamespace(
                detector=str(detector),
                bin_index=bin_value,
                is_dark=bool(is_dark),
                source=source,
            )
        )
    return events


def _build_click_channel_maps(events: list[SimpleNamespace]):
    bins = {detector: "" for detector in _DETECTOR_ORDER}
    darks = {detector: "" for detector in _DETECTOR_ORDER}
    sources = {detector: "" for detector in _DETECTOR_ORDER}
    for event in events:
        detector = str(event.detector)
        if detector not in bins:
            continue
        bin_text = str(int(event.bin_index))
        bins[detector] = bin_text if bins[detector] == "" else f"{bins[detector]};{bin_text}"
        dark_text = "1" if bool(event.is_dark) else "0"
        darks[detector] = dark_text if darks[detector] == "" else f"{darks[detector]};{dark_text}"
        source_text = str(event.source)
        sources[detector] = source_text if sources[detector] == "" else f"{sources[detector]};{source_text}"
    return bins, darks, sources


def _format_pair_key(detector_a: str, detector_b: str) -> str:
    ordered = sorted(
        (str(detector_a), str(detector_b)),
        key=lambda item: (_DETECTOR_ORDER_INDEX.get(item, 99), item),
    )
    return f"{ordered[0]}+{ordered[1]}"


def _extract_record_pair_delta(events: list[SimpleNamespace]):
    if len(events) < 2:
        return "", None
    best = None
    for idx_a in range(len(events)):
        event_a = events[idx_a]
        for idx_b in range(idx_a + 1, len(events)):
            event_b = events[idx_b]
            delta_bin = abs(int(event_a.bin_index) - int(event_b.bin_index))
            pair_key = _format_pair_key(event_a.detector, event_b.detector)
            candidate = (int(delta_bin), pair_key, idx_a, idx_b)
            if best is None or candidate < best[0]:
                best = (candidate, pair_key, int(delta_bin))
    if best is None:
        return "", None
    return best[1], best[2]


def _resolve_record_true_false_weights(record: dict, events: list[SimpleNamespace]):
    p_true = None
    if isinstance(record, dict):
        p_true = _safe_num(record.get("p_true_given_record"))
    if p_true is None:
        p_true = 0.0 if any(bool(event.is_dark) for event in events) else 1.0
    p_true = float(np.clip(float(p_true), 0.0, 1.0))
    return p_true, float(1.0 - p_true)


def _accumulate_click_analysis(
    groups: dict,
    group_value,
    record: dict,
    events: list[SimpleNamespace],
    success: bool,
) -> None:
    if not bool(success):
        return
    pair_key, delta_bin = _extract_record_pair_delta(events)
    if delta_bin is None or pair_key == "":
        return
    group = _ensure_click_analysis_group(groups, group_value)
    _accumulate_bucket_weight(group["success_delta"], int(delta_bin), 1.0)
    true_weight, false_weight = _resolve_record_true_false_weights(record, events)
    _accumulate_bucket_weight(group["true_delta"], int(delta_bin), true_weight)
    _accumulate_bucket_weight(group["false_delta"], int(delta_bin), false_weight)
    cell_key = (pair_key, int(delta_bin))
    cell = group["heatmap"].setdefault(
        cell_key,
        {"records": 0, "true_weight": 0.0, "false_weight": 0.0},
    )
    cell["records"] = int(cell["records"]) + 1
    cell["true_weight"] = float(cell["true_weight"]) + float(true_weight)
    cell["false_weight"] = float(cell["false_weight"]) + float(false_weight)


def _write_click_analysis_outputs(summary_dir, prefix: str, group_column: str, groups: dict) -> None:
    delta_path = summary_dir / f"{prefix}_delta_bin_distribution.csv"
    heatmap_path = summary_dir / f"{prefix}_record_reliability_heatmap.csv"
    sorted_groups = sorted(groups.values(), key=lambda state: _sort_group_value_key(state.get("group_value")))

    with open(delta_path, "w", encoding="utf-8", newline="") as delta_file:
        delta_writer = csv.writer(delta_file)
        delta_writer.writerow([
            group_column,
            "bucket",
            "delta_bin",
            "weight",
            "probability",
            "total_weight",
        ])
        for state in sorted_groups:
            group_value = state.get("group_value")
            for bucket_name, bucket_data in (
                ("success", state.get("success_delta", {})),
                ("true", state.get("true_delta", {})),
                ("false", state.get("false_delta", {})),
            ):
                total_weight = float(sum(float(weight) for weight in bucket_data.values()))
                for delta_bin in sorted(bucket_data.keys()):
                    weight = float(bucket_data[delta_bin])
                    probability = (weight / total_weight) if total_weight > 0.0 else 0.0
                    delta_writer.writerow([
                        group_value,
                        bucket_name,
                        int(delta_bin),
                        weight,
                        probability,
                        total_weight,
                    ])

    with open(heatmap_path, "w", encoding="utf-8", newline="") as heatmap_file:
        heatmap_writer = csv.writer(heatmap_file)
        heatmap_writer.writerow([
            group_column,
            "pair",
            "delta_bin",
            "p_true_given_record_avg",
            "records",
            "true_weight",
            "false_weight",
        ])
        for state in sorted_groups:
            group_value = state.get("group_value")
            heatmap = state.get("heatmap", {})
            for (pair_key, delta_bin), cell in sorted(heatmap.items(), key=lambda item: (item[0][0], item[0][1])):
                records = int(cell.get("records", 0) or 0)
                true_weight = float(cell.get("true_weight", 0.0) or 0.0)
                false_weight = float(cell.get("false_weight", 0.0) or 0.0)
                p_true_avg = (true_weight / float(records)) if records > 0 else 0.0
                heatmap_writer.writerow([
                    group_value,
                    pair_key,
                    int(delta_bin),
                    p_true_avg,
                    records,
                    true_weight,
                    false_weight,
                ])


def _finalize_group_summary(group: dict) -> dict:
    runs_total = int(group["runs_total"])
    shots_total = int(group["shots_total"])
    accepted_total = int(group.get("accepted_total", 0))
    p_two_click_abs_avg = (group["p_two_click_abs_sum"] / runs_total) if runs_total > 0 else 0.0
    accepted_cond_given_two_click_avg = (
        (group["accepted_cond_given_two_click_sum"] / runs_total)
        if runs_total > 0
        else 0.0
    )
    success_cond_given_two_click_avg = (
        (group["success_cond_given_two_click_sum"] / runs_total)
        if runs_total > 0
        else 0.0
    )
    success_true_cond_given_two_click_avg = (
        (group["success_true_cond_given_two_click_sum"] / runs_total)
        if runs_total > 0
        else 0.0
    )
    success_false_cond_given_two_click_avg = (
        (group["success_false_cond_given_two_click_sum"] / runs_total)
        if runs_total > 0
        else 0.0
    )

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
    herald_rate_abs = p_success_abs_avg
    sbr_true_false = (
        (p_success_true_abs_avg / p_success_false_abs_avg)
        if p_success_false_abs_avg > 0
        else None
    )
    acceptance_fraction_abs = (
        (float(accepted_total) / float(shots_total))
        if shots_total > 0
        else 0.0
    )

    return {
        "window_ns": group["window_ns"],
        "runs_target": group["runs_target"],
        "runs_total": runs_total,
        "shots_total": shots_total,
        "accepted_total": accepted_total,
        "acceptance_fraction_abs": acceptance_fraction_abs,
        "p_two_click_abs_avg": p_two_click_abs_avg,
        "accepted_cond_given_two_click_avg": accepted_cond_given_two_click_avg,
        "success_cond_given_two_click_avg": success_cond_given_two_click_avg,
        "success_true_cond_given_two_click_avg": success_true_cond_given_two_click_avg,
        "success_false_cond_given_two_click_avg": success_false_cond_given_two_click_avg,
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
        "herald_rate_abs": herald_rate_abs,
        "sbr_true_false": sbr_true_false,
    }


def _write_window_scan_summary(paths: dict, config: SimConfig) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    trials_path = summary_dir / "window_scan_trials.csv"
    runs_path = summary_dir / "window_scan_runs.csv"
    summary_path = summary_dir / "window_scan_summary.csv"

    groups = {}
    click_analysis_groups = {}

    with open(trials_path, "w", encoding="utf-8", newline="") as trials_file, open(
        runs_path, "w", encoding="utf-8", newline=""
    ) as runs_file:
        trials_writer = csv.writer(trials_file)
        runs_writer = csv.writer(runs_file)

        trials_writer.writerow([
            "window_ns",
            "window_bins",
            "run_index",
            "shot_index",
            "success",
            "bell",
            "accepted_by_window",
            "p_true_given_record",
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
            "window_bins",
            "run_index",
            "shots",
            "accepted",
            "success",
            "p_two_click_abs",
            "accepted_cond_given_two_click",
            "success_cond_given_two_click",
            "success_true_cond_given_two_click",
            "success_false_cond_given_two_click",
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
            "p_success_intrinsic_dark_assisted",
            "p_success_bg_assisted",
            "timestamp",
        ])

        for meta_path in sorted(results_dir.glob("result_*/meta.json")):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not _is_core_experiment(data, "WINDOW_SCAN"):
                continue

            tid = str(data.get("id", ""))
            metrics = data.get("metrics", {})
            run_index = _extract_run_index(metrics, tid, (r"wscan_run_(\d+)",))

            base_entry = metrics.get("window_scan")
            if not isinstance(base_entry, dict):
                raise ValueError("WINDOW_SCAN summary 需要 metrics.window_scan")

            clicks_path = meta_path.parent / "raw" / "clicks.json"
            clicks_shared = []
            if clicks_path.exists():
                try:
                    raw_clicks = json.loads(clicks_path.read_text(encoding="utf-8")).get("clicks", [])
                except Exception as exc:
                    raise ValueError(f"WINDOW_SCAN clicks.json 读取失败: {exc}") from exc
                if not isinstance(raw_clicks, list):
                    raise ValueError("WINDOW_SCAN clicks 必须为 list")
                clicks_shared = raw_clicks

            window_values = []
            if (
                config.run.window_sweep_start_ns is not None
                and config.run.window_sweep_end_ns is not None
                and config.run.window_sweep_step_ns is not None
            ):
                start = float(config.run.window_sweep_start_ns)
                end = float(config.run.window_sweep_end_ns)
                step = float(config.run.window_sweep_step_ns)
                value = start
                while value <= end + 1e-12:
                    window_values.append(round(value, 9))
                    value += step
            if not window_values:
                window_values = [float(config.run.window_ns)]

            def _record_within_window_bins(record: dict, window_bins: int) -> bool:
                shot_clicks = record.get("clicks", [])
                bins = []
                for click in shot_clicks:
                    if isinstance(click, (list, tuple)) and len(click) >= 2:
                        try:
                            bins.append(int(click[1]))
                        except Exception:
                            continue
                for i in range(len(bins)):
                    for j in range(i + 1, len(bins)):
                        if abs(int(bins[i]) - int(bins[j])) <= int(window_bins):
                            return True
                return False

            def _record_success_by_window(record: dict, window_bins: int) -> bool:
                if not bool(record.get("success", False)):
                    return False
                return _record_within_window_bins(record, window_bins)

            expanded_windows = []
            for window_ns in window_values:
                window_bins = _compute_window_bins(
                    float(window_ns),
                    float(config.emission.dt_ns),
                    detection_gate_ns=config.noise.detector_gate_ns,
                )
                accepted = 0
                success = 0
                success_true_sum = 0.0
                success_false_sum = 0.0
                fidelity_all_vals = []
                fidelity_true_num = 0.0
                fidelity_false_num = 0.0
                corr_exx_vals = []
                corr_eyy_vals = []
                corr_ezz_vals = []
                chsh_vals = []
                for record in clicks_shared:
                    in_window = _record_within_window_bins(record, window_bins)
                    if in_window:
                        accepted += 1
                    if not _record_success_by_window(record, window_bins):
                        continue
                    success += 1
                    p_true = _safe_num(record.get("p_true_given_record")) or 0.0
                    success_true_sum += p_true
                    success_false_sum += 1.0 - p_true
                    fidelity_declared = _safe_num(record.get("fidelity_declared")) or 0.0
                    corr_exx = _safe_num(record.get("corr_exx")) or 0.0
                    corr_eyy = _safe_num(record.get("corr_eyy")) or 0.0
                    corr_ezz = _safe_num(record.get("corr_ezz")) or 0.0
                    chsh = _safe_num(record.get("chsh_s_max")) or 0.0
                    fidelity_all_vals.append(fidelity_declared)
                    fidelity_true_num += fidelity_declared * p_true
                    fidelity_false_num += fidelity_declared * (1.0 - p_true)
                    corr_exx_vals.append(corr_exx)
                    corr_eyy_vals.append(corr_eyy)
                    corr_ezz_vals.append(corr_ezz)
                    chsh_vals.append(chsh)
                shots = int(base_entry.get("shots", len(clicks_shared)) or len(clicks_shared))
                shots_total = float(max(shots, 1))
                p_two_click_abs = float(_safe_num(base_entry.get("p_two_click_abs")) or 0.0)
                accepted_cond_given_two_click = float(accepted / shots_total)
                success_cond_given_two_click = float(success / shots_total)
                success_true_cond_given_two_click = float(success_true_sum / shots_total)
                success_false_cond_given_two_click = float(success_false_sum / shots_total)
                p_success_abs = float(p_two_click_abs * success_cond_given_two_click)
                p_success_true_abs = float(p_two_click_abs * success_true_cond_given_two_click)
                p_success_false_abs = float(p_two_click_abs * success_false_cond_given_two_click)
                expanded_windows.append(
                    {
                        "window_ns": float(window_ns),
                        "window_bins": int(window_bins),
                        "run_index": int(base_entry.get("run_index", run_index) or run_index),
                        "shots": shots,
                        "accepted": int(accepted),
                        "success": int(success),
                        "p_two_click_abs": p_two_click_abs,
                        "accepted_cond_given_two_click": accepted_cond_given_two_click,
                        "success_cond_given_two_click": success_cond_given_two_click,
                        "success_true_cond_given_two_click": success_true_cond_given_two_click,
                        "success_false_cond_given_two_click": success_false_cond_given_two_click,
                        "p_arrive": _safe_num(base_entry.get("p_arrive")),
                        "p_arrive_11": _safe_num(base_entry.get("p_arrive_11")),
                        "p_arrive_same_arm": _safe_num(base_entry.get("p_arrive_same_arm")),
                        "p_arrive_20": _safe_num(base_entry.get("p_arrive_20")),
                        "p_arrive_02": _safe_num(base_entry.get("p_arrive_02")),
                        "p_success_abs": p_success_abs,
                        "p_success_true_abs": p_success_true_abs,
                        "p_success_false_abs": p_success_false_abs,
                        "p_success_true_given_arrival": (
                            p_success_true_abs / _safe_num(base_entry.get("p_arrive_11"))
                            if (_safe_num(base_entry.get("p_arrive_11")) or 0.0) > 0.0
                            else 0.0
                        ),
                        "fidelity_all": float(np.mean(fidelity_all_vals)) if fidelity_all_vals else 0.0,
                        "fidelity_true": (fidelity_true_num / success_true_sum) if success_true_sum > 0.0 else 0.0,
                        "fidelity_false": (fidelity_false_num / success_false_sum) if success_false_sum > 0.0 else 0.0,
                        "false_fraction": (p_success_false_abs / p_success_abs) if p_success_abs > 0.0 else 0.0,
                        "corr_exx": float(np.mean(corr_exx_vals)) if corr_exx_vals else 0.0,
                        "corr_eyy": float(np.mean(corr_eyy_vals)) if corr_eyy_vals else 0.0,
                        "corr_ezz": float(np.mean(corr_ezz_vals)) if corr_ezz_vals else 0.0,
                        "chsh_s_max": float(np.mean(chsh_vals)) if chsh_vals else 0.0,
                        "p_success_intrinsic_dark_assisted": _safe_num(base_entry.get("p_success_intrinsic_dark_assisted")),
                        "p_success_bg_assisted": _safe_num(base_entry.get("p_success_bg_assisted")),
                    }
                )

            for entry in expanded_windows:
                window_ns = float(entry.get("window_ns", 0.0) or 0.0)
                window_key = f"{window_ns:.9f}"

                p_arrive = _safe_num(entry.get("p_arrive"))
                p_arrive_11 = _safe_num(entry.get("p_arrive_11"))
                p_arrive_same_arm = _safe_num(entry.get("p_arrive_same_arm"))
                p_arrive_20 = _safe_num(entry.get("p_arrive_20"))
                p_arrive_02 = _safe_num(entry.get("p_arrive_02"))
                p_success_abs = _safe_num(entry.get("p_success_abs"))
                p_success_true_abs = _safe_num(entry.get("p_success_true_abs"))
                p_success_false_abs = _safe_num(entry.get("p_success_false_abs"))
                p_success_true_given_arrival = _safe_num(entry.get("p_success_true_given_arrival"))
                p_success_intrinsic_dark_assisted = _safe_num(entry.get("p_success_intrinsic_dark_assisted"))
                p_success_bg_assisted = _safe_num(entry.get("p_success_bg_assisted"))
                fidelity_all = _safe_num(entry.get("fidelity_all"))
                fidelity_true = _safe_num(entry.get("fidelity_true"))
                fidelity_false = _safe_num(entry.get("fidelity_false"))
                false_fraction = _safe_num(entry.get("false_fraction"))
                corr_exx = _safe_num(entry.get("corr_exx"))
                corr_eyy = _safe_num(entry.get("corr_eyy"))
                corr_ezz = _safe_num(entry.get("corr_ezz"))
                chsh_s_max = _safe_num(entry.get("chsh_s_max"))
                window_bins = int(entry.get("window_bins", 0) or 0)
                shots = int(entry.get("shots", 0) or 0)
                accepted = int(entry.get("accepted", 0) or 0)
                success = int(entry.get("success", 0) or 0)
                p_two_click_abs = _safe_num(entry.get("p_two_click_abs"))
                accepted_cond_given_two_click = _safe_num(entry.get("accepted_cond_given_two_click"))
                success_cond_given_two_click = _safe_num(entry.get("success_cond_given_two_click"))
                success_true_cond_given_two_click = _safe_num(entry.get("success_true_cond_given_two_click"))
                success_false_cond_given_two_click = _safe_num(entry.get("success_false_cond_given_two_click"))

                runs_writer.writerow([
                    tid,
                    window_ns,
                    window_bins,
                    run_index,
                    shots,
                    accepted,
                    success,
                    p_two_click_abs,
                    accepted_cond_given_two_click,
                    success_cond_given_two_click,
                    success_true_cond_given_two_click,
                    success_false_cond_given_two_click,
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
                    p_success_intrinsic_dark_assisted,
                    p_success_bg_assisted,
                    data.get("timestamp"),
                ])

                group = groups.setdefault(
                    window_key,
                    {
                        "window_ns": window_ns,
                        "runs_target": config.run.runs,
                        "runs_total": 0,
                        "shots_total": 0,
                        "accepted_total": 0,
                        "p_two_click_abs_sum": 0.0,
                        "accepted_cond_given_two_click_sum": 0.0,
                        "success_cond_given_two_click_sum": 0.0,
                        "success_true_cond_given_two_click_sum": 0.0,
                        "success_false_cond_given_two_click_sum": 0.0,
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
                group["accepted_total"] += accepted
                group["p_two_click_abs_sum"] += p_two_click_abs or 0.0
                group["accepted_cond_given_two_click_sum"] += accepted_cond_given_two_click or 0.0
                group["success_cond_given_two_click_sum"] += success_cond_given_two_click or 0.0
                group["success_true_cond_given_two_click_sum"] += success_true_cond_given_two_click or 0.0
                group["success_false_cond_given_two_click_sum"] += success_false_cond_given_two_click or 0.0
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

                records = []
                for record in clicks_shared:
                    shot_copy = dict(record)
                    in_window = _record_within_window_bins(shot_copy, int(window_bins))
                    shot_copy["accepted_by_window"] = bool(in_window)
                    shot_copy["success"] = bool(_record_success_by_window(shot_copy, int(window_bins)))
                    shot_copy["bell"] = shot_copy.get("bell") if shot_copy.get("success") else ""
                    records.append(shot_copy)
                if not records:
                    trials_writer.writerow([
                        window_ns,
                        window_bins,
                        run_index,
                        -1,
                        "",
                        "",
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

                for record in records:
                    shot_idx = record.get("shot_index")
                    shot_success = record.get("success")
                    bell = record.get("bell")
                    accepted_by_window = record.get("accepted_by_window")
                    p_true_given_record = _safe_num(record.get("p_true_given_record"))
                    shot_clicks = record.get("clicks", [])
                    events = _parse_click_events(shot_clicks, "WINDOW_SCAN")
                    bins, darks, _ = _build_click_channel_maps(events)
                    _accumulate_click_analysis(
                        click_analysis_groups,
                        window_ns,
                        record,
                        events,
                        bool(shot_success),
                    )

                    trials_writer.writerow([
                        window_ns,
                        window_bins,
                        run_index,
                        shot_idx,
                        shot_success,
                        bell,
                        accepted_by_window,
                        p_true_given_record,
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

    rows = [_finalize_group_summary(groups[key]) for key in sorted(groups.keys(), key=lambda item: float(item))]
    max_herald = max((row["herald_rate_abs"] for row in rows), default=0.0)

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow([
            "window_ns",
            "runs_target",
            "runs_total",
            "shots_total",
            "accepted_total",
            "acceptance_fraction_abs",
            "p_two_click_abs_avg",
            "accepted_cond_given_two_click_avg",
            "success_cond_given_two_click_avg",
            "success_true_cond_given_two_click_avg",
            "success_false_cond_given_two_click_avg",
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
            "herald_rate_abs",
            "sbr_true_false",
            "acceptance_fraction_vs_max_window",
        ])
        for row in rows:
            acceptance_fraction = (row["herald_rate_abs"] / max_herald) if max_herald > 0 else 0.0
            summary_writer.writerow([
                row["window_ns"],
                row["runs_target"],
                row["runs_total"],
                row["shots_total"],
                row["accepted_total"],
                row["acceptance_fraction_abs"],
                row["p_two_click_abs_avg"],
                row["accepted_cond_given_two_click_avg"],
                row["success_cond_given_two_click_avg"],
                row["success_true_cond_given_two_click_avg"],
                row["success_false_cond_given_two_click_avg"],
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
                row["herald_rate_abs"],
                row["sbr_true_false"],
                acceptance_fraction,
            ])

    _write_click_analysis_outputs(summary_dir, "window_scan", "window_ns", click_analysis_groups)


def _write_length_scan_summary(paths: dict, config: SimConfig) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    trials_path = summary_dir / "length_scan_trials.csv"
    runs_path = summary_dir / "length_scan_runs.csv"
    summary_path = summary_dir / "length_scan_summary.csv"

    groups = {}
    click_analysis_groups = {}

    with open(trials_path, "w", encoding="utf-8", newline="") as trials_file, open(
        runs_path, "w", encoding="utf-8", newline=""
    ) as runs_file:
        trials_writer = csv.writer(trials_file)
        runs_writer = csv.writer(runs_file)

        trials_writer.writerow([
            "length_km",
            "run_index",
            "shot_index",
            "success",
            "bell",
            "window_ns",
            "attempt_rate_hz",
            "event_rate_hz",
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
            "length_km",
            "run_index",
            "shots",
            "success",
            "window_ns",
            "attempt_rate_hz",
            "event_rate_hz",
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
            if not _is_core_experiment(data, "LENGTH_SCAN"):
                continue

            tid = str(data.get("id", ""))
            metrics = data.get("metrics", {})
            run_index = _extract_run_index(metrics, tid, (r"lscan_len_\d+_run_(\d+)", r"lscan_run_(\d+)"))

            lengths = metrics.get("lengths", [])
            if not isinstance(lengths, list):
                lengths = []

            clicks_path = meta_path.parent / "raw" / "clicks.json"
            clicks_by_length = {}
            if clicks_path.exists():
                try:
                    raw_clicks = json.loads(clicks_path.read_text(encoding="utf-8")).get("clicks", {})
                    if isinstance(raw_clicks, dict):
                        clicks_by_length = raw_clicks
                except Exception:
                    clicks_by_length = {}

            for entry in lengths:
                length_km = float(entry.get("length_km", 0.0) or 0.0)
                length_key = f"{length_km:.9f}"

                p_arrive = _safe_num(entry.get("p_arrive"))
                p_arrive_11 = _safe_num(entry.get("p_arrive_11"))
                p_arrive_same_arm = _safe_num(entry.get("p_arrive_same_arm"))
                p_arrive_20 = _safe_num(entry.get("p_arrive_20"))
                p_arrive_02 = _safe_num(entry.get("p_arrive_02"))
                p_success_abs = _safe_num(entry.get("p_success_abs"))
                p_success_true_abs = _safe_num(entry.get("p_success_true_abs"))
                p_success_false_abs = _safe_num(entry.get("p_success_false_abs"))
                p_success_true_given_arrival = _safe_num(entry.get("p_success_true_given_arrival"))
                fidelity_all = _safe_num(entry.get("fidelity_all"))
                fidelity_true = _safe_num(entry.get("fidelity_true"))
                fidelity_false = _safe_num(entry.get("fidelity_false"))
                false_fraction = _safe_num(entry.get("false_fraction"))
                corr_exx = _safe_num(entry.get("corr_exx"))
                corr_eyy = _safe_num(entry.get("corr_eyy"))
                corr_ezz = _safe_num(entry.get("corr_ezz"))
                chsh_s_max = _safe_num(entry.get("chsh_s_max"))
                window_ns = _safe_num(entry.get("window_ns"))
                attempt_rate_hz = _safe_num(entry.get("attempt_rate_hz"))
                event_rate_hz = _safe_num(entry.get("event_rate_hz"))
                shots = int(entry.get("shots", 0) or 0)
                success = int(entry.get("success", 0) or 0)

                runs_writer.writerow([
                    tid,
                    length_km,
                    run_index,
                    shots,
                    success,
                    window_ns,
                    attempt_rate_hz,
                    event_rate_hz,
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

                group = groups.setdefault(
                    length_key,
                    {
                        "length_km": length_km,
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
                        "event_rate_hz_sum": 0.0,
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
                group["event_rate_hz_sum"] += event_rate_hz or 0.0

                records = clicks_by_length.get(length_key, [])
                if not isinstance(records, list) or not records:
                    trials_writer.writerow([
                        length_km,
                        run_index,
                        -1,
                        "",
                        "",
                        window_ns,
                        attempt_rate_hz,
                        event_rate_hz,
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

                for record in records:
                    shot_idx = record.get("shot_index")
                    shot_success = record.get("success")
                    bell = record.get("bell")
                    shot_clicks = record.get("clicks", [])
                    events = _parse_click_events(shot_clicks, "LENGTH_SCAN")
                    bins, darks, _ = _build_click_channel_maps(events)
                    _accumulate_click_analysis(
                        click_analysis_groups,
                        length_km,
                        record,
                        events,
                        bool(shot_success),
                    )

                    trials_writer.writerow([
                        length_km,
                        run_index,
                        shot_idx,
                        shot_success,
                        bell,
                        window_ns,
                        attempt_rate_hz,
                        event_rate_hz,
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

    rows = []
    for key in sorted(groups.keys(), key=lambda item: float(item)):
        group = groups[key]
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
        herald_rate_abs = p_success_abs_avg
        sbr_true_false = (
            (p_success_true_abs_avg / p_success_false_abs_avg)
            if p_success_false_abs_avg > 0
            else None
        )
        attempt_rate_hz_eff = _compute_effective_attempt_rate_hz(
            config.run.attempt_rate_hz,
            config.run.attempt_overhead_us,
        )
        event_rate_hz_avg = herald_rate_abs * attempt_rate_hz_eff
        rows.append(
            {
                "length_km": group["length_km"],
                "runs_target": group["runs_target"],
                "runs_total": runs_total,
                "shots_total": shots_total,
                "window_ns": config.run.window_ns,
                "attempt_rate_hz": attempt_rate_hz_eff,
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
                "herald_rate_abs": herald_rate_abs,
                "event_rate_hz_avg": event_rate_hz_avg,
                "sbr_true_false": sbr_true_false,
            }
        )

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow([
            "length_km",
            "runs_target",
            "runs_total",
            "shots_total",
            "window_ns",
            "attempt_rate_hz",
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
            "herald_rate_abs",
            "event_rate_hz_avg",
            "sbr_true_false",
        ])
        for row in rows:
            summary_writer.writerow([
                row["length_km"],
                row["runs_target"],
                row["runs_total"],
                row["shots_total"],
                row["window_ns"],
                row["attempt_rate_hz"],
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
                row["herald_rate_abs"],
                row["event_rate_hz_avg"],
                row["sbr_true_false"],
            ])

    _write_click_analysis_outputs(summary_dir, "length_scan", "length_km", click_analysis_groups)


def _write_bsm_scan_summary(paths: dict, config: SimConfig) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    trials_path = summary_dir / "bsm_scan_trials.csv"
    runs_path = summary_dir / "bsm_scan_runs.csv"
    summary_path = summary_dir / "bsm_scan_summary.csv"

    groups = {}
    click_analysis_groups = {}

    with open(trials_path, "w", encoding="utf-8", newline="") as trials_file, open(
        runs_path, "w", encoding="utf-8", newline=""
    ) as runs_file:
        trials_writer = csv.writer(trials_file)
        runs_writer = csv.writer(runs_file)

        trial_header = [
            "bs_theta",
            "bs_split_ratio",
            "run_index",
            "shot_index",
            "success",
            "bell",
            "pattern",
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
            "H1_src",
            "V1_src",
            "H2_src",
            "V2_src",
        ]
        for pattern_key in BSM_PATTERN_KEYS:
            trial_header.append(pattern_key)
            trial_header.append(f"{pattern_key}_rate")
        trials_writer.writerow(trial_header)

        run_header = [
            "id",
            "bs_theta",
            "bs_split_ratio",
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
        ]
        for pattern_key in BSM_PATTERN_KEYS:
            run_header.append(pattern_key)
            run_header.append(f"{pattern_key}_rate")
        run_header.append("timestamp")
        runs_writer.writerow(run_header)

        for meta_path in sorted(results_dir.glob("result_*/meta.json")):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not _is_core_experiment(data, "BSM_SCAN"):
                continue

            tid = str(data.get("id", ""))
            metrics = data.get("metrics", {})
            run_index = _extract_run_index(metrics, tid, (r"bscan_theta_\d+_run_(\d+)", r"bscan_run_(\d+)"))

            entries = metrics.get("bs_thetas", [])
            if not isinstance(entries, list):
                entries = []

            clicks_path = meta_path.parent / "raw" / "clicks.json"
            clicks_by_theta = {}
            if clicks_path.exists():
                try:
                    raw_clicks = json.loads(clicks_path.read_text(encoding="utf-8")).get("clicks", {})
                    if isinstance(raw_clicks, dict):
                        clicks_by_theta = raw_clicks
                except Exception:
                    clicks_by_theta = {}

            for entry in entries:
                bs_theta = float(entry.get("bs_theta", 0.0) or 0.0)
                bs_key = f"{bs_theta:.9f}"
                bs_split_ratio = _safe_num(entry.get("bs_split_ratio"))

                p_arrive = _safe_num(entry.get("p_arrive"))
                p_arrive_11 = _safe_num(entry.get("p_arrive_11"))
                p_arrive_same_arm = _safe_num(entry.get("p_arrive_same_arm"))
                p_arrive_20 = _safe_num(entry.get("p_arrive_20"))
                p_arrive_02 = _safe_num(entry.get("p_arrive_02"))
                p_success_abs = _safe_num(entry.get("p_success_abs"))
                p_success_true_abs = _safe_num(entry.get("p_success_true_abs"))
                p_success_false_abs = _safe_num(entry.get("p_success_false_abs"))
                p_success_true_given_arrival = _safe_num(entry.get("p_success_true_given_arrival"))
                fidelity_all = _safe_num(entry.get("fidelity_all"))
                fidelity_true = _safe_num(entry.get("fidelity_true"))
                fidelity_false = _safe_num(entry.get("fidelity_false"))
                false_fraction = _safe_num(entry.get("false_fraction"))
                corr_exx = _safe_num(entry.get("corr_exx"))
                corr_eyy = _safe_num(entry.get("corr_eyy"))
                corr_ezz = _safe_num(entry.get("corr_ezz"))
                chsh_s_max = _safe_num(entry.get("chsh_s_max"))
                shots = int(entry.get("shots", 0) or 0)
                success = int(entry.get("success", 0) or 0)

                run_row = [
                    tid,
                    bs_theta,
                    bs_split_ratio,
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
                ]
                for pattern_key in BSM_PATTERN_KEYS:
                    run_row.append(int(entry.get(pattern_key, 0) or 0))
                    run_row.append(_safe_num(entry.get(f"{pattern_key}_rate")) or 0.0)
                run_row.append(data.get("timestamp"))
                runs_writer.writerow(run_row)

                group = groups.setdefault(
                    bs_key,
                    {
                        "bs_theta": bs_theta,
                        "bs_split_ratio": float(bs_split_ratio or 0.0),
                        "runs_target": config.run.runs,
                        "runs_total": 0,
                        "shots_total": 0,
                        "success_total": 0,
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
                        "pattern_sums": {pattern_key: 0 for pattern_key in BSM_PATTERN_KEYS},
                    },
                )

                group["runs_total"] += 1
                group["shots_total"] += shots
                group["success_total"] += success
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
                for pattern_key in BSM_PATTERN_KEYS:
                    group["pattern_sums"][pattern_key] += int(entry.get(pattern_key, 0) or 0)

                records = clicks_by_theta.get(bs_key, [])
                if not isinstance(records, list) or not records:
                    row = [
                        bs_theta,
                        bs_split_ratio,
                        run_index,
                        -1,
                        "",
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
                        "",
                        "",
                        "",
                        "",
                    ]
                    for pattern_key in BSM_PATTERN_KEYS:
                        row.append(int(entry.get(pattern_key, 0) or 0))
                        row.append(_safe_num(entry.get(f"{pattern_key}_rate")) or 0.0)
                    trials_writer.writerow(row)
                    continue

                for record in records:
                    shot_idx = record.get("shot_index")
                    shot_success = record.get("success")
                    bell = record.get("bell")
                    pattern = record.get("pattern")
                    shot_clicks = record.get("clicks", [])
                    events = _parse_click_events(shot_clicks, "BSM_SCAN")
                    bins, darks, sources = _build_click_channel_maps(events)
                    _accumulate_click_analysis(
                        click_analysis_groups,
                        bs_theta,
                        record,
                        events,
                        bool(shot_success),
                    )

                    row = [
                        bs_theta,
                        bs_split_ratio,
                        run_index,
                        shot_idx,
                        shot_success,
                        bell,
                        pattern,
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
                        sources["H1"],
                        sources["V1"],
                        sources["H2"],
                        sources["V2"],
                    ]
                    for pattern_key in BSM_PATTERN_KEYS:
                        row.append(int(entry.get(pattern_key, 0) or 0))
                        row.append(_safe_num(entry.get(f"{pattern_key}_rate")) or 0.0)
                    trials_writer.writerow(row)

    rows = []
    for key in sorted(groups.keys(), key=lambda value: float(value)):
        group = groups[key]
        runs_total = int(group["runs_total"])
        shots_total = int(group["shots_total"])

        p_arrive_avg = (group["p_arrive_sum"] / runs_total) if runs_total > 0 else 0.0
        p_success_abs_avg = (group["p_success_abs_sum"] / runs_total) if runs_total > 0 else 0.0
        p_success_true_abs_avg = (
            (group["p_success_true_abs_sum"] / runs_total) if runs_total > 0 else 0.0
        )
        p_success_false_abs_avg = (
            (group["p_success_false_abs_sum"] / runs_total) if runs_total > 0 else 0.0
        )
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

        herald_rate_abs = p_success_abs_avg
        sbr_true_false = (
            (p_success_true_abs_avg / p_success_false_abs_avg)
            if p_success_false_abs_avg > 0
            else None
        )

        row = {
            "bs_theta": group["bs_theta"],
            "bs_split_ratio": group["bs_split_ratio"],
            "runs_target": group["runs_target"],
            "runs_total": runs_total,
            "shots_total": shots_total,
            "success_total": int(group["success_total"]),
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
            "herald_rate_abs": herald_rate_abs,
            "sbr_true_false": sbr_true_false,
        }
        for pattern_key in BSM_PATTERN_KEYS:
            count = int(group["pattern_sums"][pattern_key])
            row[pattern_key] = count
            row[f"{pattern_key}_rate"] = (float(count) / float(shots_total)) if shots_total > 0 else 0.0
        rows.append(row)

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_header = [
            "bs_theta",
            "bs_split_ratio",
            "runs_target",
            "runs_total",
            "shots_total",
            "success_total",
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
            "herald_rate_abs",
            "sbr_true_false",
        ]
        for pattern_key in BSM_PATTERN_KEYS:
            summary_header.append(pattern_key)
            summary_header.append(f"{pattern_key}_rate")
        summary_writer.writerow(summary_header)

        for row in rows:
            output_row = [
                row["bs_theta"],
                row["bs_split_ratio"],
                row["runs_target"],
                row["runs_total"],
                row["shots_total"],
                row["success_total"],
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
                row["herald_rate_abs"],
                row["sbr_true_false"],
            ]
            for pattern_key in BSM_PATTERN_KEYS:
                output_row.append(row[pattern_key])
                output_row.append(row[f"{pattern_key}_rate"])
            summary_writer.writerow(output_row)

    _write_click_analysis_outputs(summary_dir, "bsm_scan", "bs_theta", click_analysis_groups)


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
        click_analysis_groups = {}
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
                if not _is_core_experiment(data, "HOM"):
                    continue
                tid = data.get("id", "")
                metrics = data.get("metrics", {})
                tau_ns = float(metrics.get("tau_ns", 0.0) or 0.0)
                run_index = _extract_run_index(metrics, tid, (r"hom_tau_[+-]?\d+\.\d+_run_(\d+)",))
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
                        events = _parse_click_events(shot_clicks, "HOM")
                        bins, darks, _ = _build_click_channel_maps(events)

                        dark_clicks = sum(1 for e in events if e.is_dark)
                        state["dark_clicks_total"] += dark_clicks
                        state["clicks_total"] += len(events)
                        is_success = bool(events and _is_port_samepol_coincidence(events, window_bins))
                        _accumulate_click_analysis(
                            click_analysis_groups,
                            tau_ns,
                            {},
                            events,
                            is_success,
                        )
                        if is_success:
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
        _write_click_analysis_outputs(summary_dir, "hom", "tau_ns", click_analysis_groups)
        return
    if task_type == "WINDOW_SCAN":
        _write_window_scan_summary(paths=paths, config=config)
        return
    if task_type == "BSM_SCAN":
        _write_bsm_scan_summary(paths=paths, config=config)
        return
    if task_type == "LENGTH_SCAN":
        _write_length_scan_summary(paths=paths, config=config)
        return

    if task_type == "SIM":
        trials_path = summary_dir / "sim_trials.csv"
        click_analysis_groups = {}
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
                if not _is_core_experiment(data, "SIM"):
                    continue
                tid = data.get("id", "")
                metrics = data.get("metrics", {})
                run_index = _extract_run_index(metrics, tid, (r"sim_run_(\d+)",))
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
                    events = _parse_click_events(shot_clicks, "SIM")
                    bins, darks, _ = _build_click_channel_maps(events)
                    group_window_ns = _safe_num(window_ns)
                    if group_window_ns is None:
                        raise ValueError("SIM summary 需要 metrics.window_ns")
                    _accumulate_click_analysis(
                        click_analysis_groups,
                        group_window_ns,
                        record,
                        events,
                        bool(success),
                    )
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
        _write_click_analysis_outputs(summary_dir, "sim", "window_ns", click_analysis_groups)
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
            "p_success_intrinsic_dark_assisted",
            "p_success_bg_assisted",
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
                m.get("p_success_intrinsic_dark_assisted"),
                m.get("p_success_bg_assisted"),
                data.get("timestamp"),
            ])
