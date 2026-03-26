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


def _require_record_probability(record: dict, key: str) -> float:
    if not isinstance(record, dict):
        raise ValueError(f"点击记录必须为 dict，缺少字段 {key}")
    value = _safe_num(record.get(key))
    if value is None:
        shot_index = record.get("shot_index")
        raise ValueError(f"点击记录缺少 {key} (shot_index={shot_index})")
    return float(np.clip(value, 0.0, 1.0))


def _resolve_record_component_weights(record: dict) -> tuple[float, float, float]:
    p_true = _require_record_probability(record, "p_true_given_record")
    p_bg = _require_record_probability(record, "p_bg_assist_given_record")
    p_intrinsic = _require_record_probability(record, "p_intrinsic_dark_assist_given_record")
    return p_true, p_bg, p_intrinsic


_DETECTOR_ORDER = ("H1", "V1", "H2", "V2")
_DETECTOR_ORDER_INDEX = {detector: idx for idx, detector in enumerate(_DETECTOR_ORDER)}
_CLICK_BIN_FIELDS = tuple(f"{detector}_bin" for detector in _DETECTOR_ORDER)
_CLICK_DARK_FIELDS = tuple(f"{detector}_dark" for detector in _DETECTOR_ORDER)
_CLICK_SOURCE_FIELDS = tuple(f"{detector}_src" for detector in _DETECTOR_ORDER)
_RECORD_WEIGHT_FIELDS = ("p_true_given_record", "p_bg_assist_given_record", "p_intrinsic_dark_assist_given_record")
_EMPTY_RECORD_WEIGHTS = ("",) * len(_RECORD_WEIGHT_FIELDS)
_CORE_METRIC_FIELDS = (
    "p_arrive", "p_arrive_11", "p_arrive_same_arm", "p_arrive_20", "p_arrive_02",
    "p_success_abs", "p_success_true_abs", "p_success_false_abs", "p_success_true_given_arrival",
    "fidelity_all", "fidelity_true", "fidelity_false", "false_fraction", "corr_exx", "corr_eyy", "corr_ezz",
    "chsh_s_max",
)
_GENERIC_SUMMARY_METRIC_FIELDS = (
    "window_ns", "p_arrive", "p_arrive_11", "p_arrive_same_arm", "p_arrive_20", "p_arrive_02", "coinc",
    "p_success_abs", "p_success_true_abs", "p_success_false_abs", "p_success_true_given_arrival",
    "fidelity_all", "fidelity_true", "fidelity_false", "false_fraction", "corr_exx", "corr_eyy", "corr_ezz",
    "chsh_s_max", "p_success_intrinsic_dark_assisted", "p_success_bg_assisted",
)
_GENERIC_SUMMARY_HEADER = ("id", "mode", *_GENERIC_SUMMARY_METRIC_FIELDS, "timestamp")
_HOM_TRIAL_HEADER = (
    "tau_ns", "run_index", "shot_index", *_RECORD_WEIGHT_FIELDS, "p_arrive", *_CLICK_BIN_FIELDS, *_CLICK_DARK_FIELDS
)
_HOM_SUMMARY_HEADER = (
    "tau_ns", "runs_target", "runs_total", "coinc_counts", "coinc_rate", "p_arrive_avg", "arrive_trials",
    "window_ns", "shots_per_run", "shots_total", "coinc_true", "coinc_dark_any", "coinc_dark_single",
    "coinc_dark_double", "dark_clicks_total", "clicks_total", "dark_click_rate", "dark_click_rate_per_det",
)
_SIM_TRIAL_HEADER = (
    "task_mode", "window_ns", "run_index", "shot_index", "success", "bell",
    *_RECORD_WEIGHT_FIELDS, *_CORE_METRIC_FIELDS, *_CLICK_BIN_FIELDS, *_CLICK_DARK_FIELDS,
)
_COMMON_SCAN_SUMMARY_FIELDS = (
    "p_arrive_avg", "p_arrive_11_avg", "p_arrive_same_arm_avg", "p_arrive_20_avg", "p_arrive_02_avg",
    "p_success_abs_avg", "p_success_true_abs_avg", "p_success_false_abs_avg",
    "p_success_true_given_arrival11_global", "false_fraction_global",
    "fidelity_all_avg", "fidelity_true_avg", "fidelity_false_avg", "corr_exx_avg", "corr_eyy_avg", "corr_ezz_avg",
    "chsh_s_max_avg", "herald_rate_abs", "sbr_true_false",
)
_LENGTH_SCAN_TRIAL_HEADER = (
    "length_km", "run_index", "shot_index", "success", "bell", *_RECORD_WEIGHT_FIELDS,
    "window_ns", "attempt_rate_hz", "event_rate_hz", *_CORE_METRIC_FIELDS, *_CLICK_BIN_FIELDS, *_CLICK_DARK_FIELDS,
)
_LENGTH_SCAN_RUN_HEADER = (
    "id", "length_km", "run_index", "shots", "success", "p_two_click_abs",
    "window_ns", "attempt_rate_hz", "event_rate_hz", *_CORE_METRIC_FIELDS, "timestamp",
)
_LENGTH_SCAN_SUMMARY_FIELDS = (
    *_COMMON_SCAN_SUMMARY_FIELDS[:-1],
    "event_rate_hz_avg",
    _COMMON_SCAN_SUMMARY_FIELDS[-1],
)
_NOISE_SCAN_RUN_FIELDS = (
    "window_ns", "attempt_rate_hz", "event_rate_hz", "p_two_click_abs",
    *_CORE_METRIC_FIELDS, "p_success_intrinsic_dark_assisted", "p_success_bg_assisted",
)
_NOISE_SCAN_SUMMARY_FIELDS = (
    "window_ns_avg", "attempt_rate_hz_avg", "event_rate_hz_avg", "p_two_click_abs_avg",
    *_COMMON_SCAN_SUMMARY_FIELDS, "p_success_intrinsic_dark_assisted_avg", "p_success_bg_assisted_avg",
)
_BSM_SCAN_BASE_SUMMARY_HEADER = (
    "bs_theta", "bs_split_ratio", "runs_target", "runs_total", "shots_total", "success_total",
    *_COMMON_SCAN_SUMMARY_FIELDS,
)
_BSM_PATTERN_COLUMN_FIELDS = tuple(
    field for pattern_key in BSM_PATTERN_KEYS for field in (
        pattern_key, f"{pattern_key}_rate", f"{pattern_key}_true_abs", f"{pattern_key}_false_abs"
    )
)
_BSM_SCAN_TRIAL_HEADER = (
    "bs_theta", "bs_split_ratio", "run_index", "shot_index", "success", "bell", "pattern",
    *_RECORD_WEIGHT_FIELDS, *_CORE_METRIC_FIELDS, *_CLICK_BIN_FIELDS, *_CLICK_DARK_FIELDS, *_CLICK_SOURCE_FIELDS,
)
_BSM_SCAN_RUN_HEADER = (
    "id", "bs_theta", "bs_split_ratio", "run_index", "shots", "success", "p_two_click_abs",
    *_CORE_METRIC_FIELDS, *_BSM_PATTERN_COLUMN_FIELDS, "timestamp",
)
_BSM_SCAN_SUMMARY_HEADER = (*_BSM_SCAN_BASE_SUMMARY_HEADER, *_BSM_PATTERN_COLUMN_FIELDS)
_WINDOW_SCAN_TRIAL_HEADER = (
    "window_ns", "window_bins", "run_index", "shot_index", "success", "bell", "accepted_by_window",
    *_RECORD_WEIGHT_FIELDS, *_CORE_METRIC_FIELDS, *_CLICK_BIN_FIELDS, *_CLICK_DARK_FIELDS,
)
_WINDOW_SCAN_RUN_FIELDS = (
    "p_two_click_abs",
    "accepted_cond_given_two_click",
    "success_cond_given_two_click",
    "success_true_cond_given_two_click",
    "success_false_cond_given_two_click",
    *_CORE_METRIC_FIELDS,
    "p_success_intrinsic_dark_assisted_abs",
    "p_success_bg_assisted_abs",
)
_WINDOW_SCAN_RUN_HEADER = (
    "id", "window_ns", "window_bins", "run_index", "shots", "accepted", "success", *_WINDOW_SCAN_RUN_FIELDS, "timestamp"
)
_WINDOW_SCAN_SUMMARY_METRIC_FIELDS = (
    "acceptance_fraction_abs",
    "p_two_click_abs_avg",
    "accepted_cond_given_two_click_avg",
    "success_cond_given_two_click_avg",
    "success_true_cond_given_two_click_avg",
    "success_false_cond_given_two_click_avg",
    "p_arrive_avg",
    "p_arrive_11_avg",
    "p_arrive_same_arm_avg",
    "p_arrive_20_avg",
    "p_arrive_02_avg",
    "p_success_abs_avg",
    "p_success_true_abs_avg",
    "p_success_false_abs_avg",
    "p_success_intrinsic_dark_assisted_abs_avg",
    "p_success_bg_assisted_abs_avg",
    "p_success_true_given_arrival11_global",
    "false_fraction_global",
    "fidelity_all_avg",
    "fidelity_true_avg",
    "fidelity_false_avg",
    "corr_exx_avg",
    "corr_eyy_avg",
    "corr_ezz_avg",
    "chsh_s_max_avg",
    "herald_rate_abs",
    "attempt_rate_hz_eff",
    "event_rate_hz_avg",
    "sbr_true_false",
)
_WINDOW_SCAN_SUMMARY_HEADER = (
    "window_ns",
    "runs_target",
    "runs_total",
    "shots_total",
    "accepted_total",
    *_WINDOW_SCAN_SUMMARY_METRIC_FIELDS,
    "acceptance_fraction_vs_max_window",
)


def _row_values(mapping: dict, fields: tuple[str, ...]):
    return [mapping.get(field) for field in fields]


def _click_values(
    bins: dict | None = None,
    darks: dict | None = None,
    sources: dict | None = None,
    *,
    include_sources: bool = False,
) -> list[str]:
    bins = bins or {}
    darks = darks or {}
    values = [str(bins.get(detector, "")) for detector in _DETECTOR_ORDER]
    values.extend(str(darks.get(detector, "")) for detector in _DETECTOR_ORDER)
    if include_sources:
        src = sources or {}
        values.extend(str(src.get(detector, "")) for detector in _DETECTOR_ORDER)
    return values


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


def _extract_record_click_bins(record: dict) -> list[int]:
    bins = []
    shot_clicks = record.get("clicks", [])
    for click in shot_clicks:
        if isinstance(click, (list, tuple)) and len(click) >= 2:
            try:
                bins.append(int(click[1]))
            except Exception:
                continue
    return bins


def _record_within_window_bins(record: dict, window_bins: int) -> bool:
    bins = _extract_record_click_bins(record)
    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            if abs(int(bins[i]) - int(bins[j])) <= int(window_bins):
                return True
    return False


def _record_success_by_window(record: dict, window_bins: int) -> bool:
    return bool(record.get("success", False)) and _record_within_window_bins(record, window_bins)


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


def _resolve_record_true_false_weights(record: dict):
    p_true = _require_record_probability(record, "p_true_given_record")
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
    true_weight, false_weight = _resolve_record_true_false_weights(record)
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


def _finalize_group_summary(group: dict, attempt_rate_hz_eff: float) -> dict:
    runs_total = int(group["runs_total"])
    shots_total = int(group["shots_total"])
    accepted_total = int(group.get("accepted_total", 0))
    avg_pairs = {
        "p_two_click_abs_avg": "p_two_click_abs_sum",
        "accepted_cond_given_two_click_avg": "accepted_cond_given_two_click_sum",
        "success_cond_given_two_click_avg": "success_cond_given_two_click_sum",
        "success_true_cond_given_two_click_avg": "success_true_cond_given_two_click_sum",
        "success_false_cond_given_two_click_avg": "success_false_cond_given_two_click_sum",
        "p_arrive_avg": "p_arrive_sum",
        "p_arrive_11_avg": "p_arrive_11_sum",
        "p_arrive_same_arm_avg": "p_arrive_same_arm_sum",
        "p_arrive_20_avg": "p_arrive_20_sum",
        "p_arrive_02_avg": "p_arrive_02_sum",
        "p_success_abs_avg": "p_success_abs_sum",
        "p_success_true_abs_avg": "p_success_true_abs_sum",
        "p_success_false_abs_avg": "p_success_false_abs_sum",
        "p_success_intrinsic_dark_assisted_abs_avg": "p_success_intrinsic_dark_assisted_abs_sum",
        "p_success_bg_assisted_abs_avg": "p_success_bg_assisted_abs_sum",
        "fidelity_all_avg": "fidelity_all_sum",
        "fidelity_false_avg": "fidelity_false_sum",
        "corr_exx_avg": "corr_exx_sum",
        "corr_eyy_avg": "corr_eyy_sum",
        "corr_ezz_avg": "corr_ezz_sum",
        "chsh_s_max_avg": "chsh_s_max_sum",
    }
    averages = {
        out_key: (group[sum_key] / runs_total if runs_total > 0 else 0.0)
        for out_key, sum_key in avg_pairs.items()
    }
    # WINDOW_SCAN 的 true-fidelity 使用全局条件口径：
    # fidelity_true_avg = sum_r[fidelity_true(r) * p_success_true_abs(r)] / sum_r[p_success_true_abs(r)]
    # 避免 shots_per_run 很小时，按 run 等权平均导致的系统性低估。
    p_success_true_abs_sum = float(group["p_success_true_abs_sum"])
    fidelity_true_weighted_abs_sum = float(group.get("fidelity_true_weighted_abs_sum", 0.0))
    averages["fidelity_true_avg"] = (
        fidelity_true_weighted_abs_sum / p_success_true_abs_sum
        if p_success_true_abs_sum > 0.0
        else 0.0
    )
    averages["p_success_true_given_arrival11_global"] = (
        group["p_success_true_abs_sum"] / group["p_arrive_11_sum"] if group["p_arrive_11_sum"] > 0 else 0.0
    )
    averages["false_fraction_global"] = (
        group["p_success_false_abs_sum"] / group["p_success_abs_sum"] if group["p_success_abs_sum"] > 0 else 0.0
    )
    averages["herald_rate_abs"] = averages["p_success_abs_avg"]
    averages["sbr_true_false"] = (
        averages["p_success_true_abs_avg"] / averages["p_success_false_abs_avg"]
        if averages["p_success_false_abs_avg"] > 0
        else None
    )
    acceptance_fraction_abs = (float(accepted_total) / float(shots_total)) if shots_total > 0 else 0.0
    result = {
        "window_ns": group["window_ns"],
        "runs_target": group["runs_target"],
        "runs_total": runs_total,
        "shots_total": shots_total,
        "accepted_total": accepted_total,
        "acceptance_fraction_abs": acceptance_fraction_abs,
        "attempt_rate_hz_eff": float(attempt_rate_hz_eff),
        "event_rate_hz_avg": averages["p_success_abs_avg"] * float(attempt_rate_hz_eff),
    }
    result.update(averages)
    return result


def _init_window_scan_group(window_ns: float, runs_target: int) -> dict:
    return {
        "window_ns": float(window_ns),
        "runs_target": int(runs_target),
        "runs_total": 0,
        "shots_total": 0,
        "accepted_total": 0,
        "p_two_click_abs_sum": 0.0,
        "accepted_cond_given_two_click_sum": 0.0,
        "success_cond_given_two_click_sum": 0.0,
        "success_true_cond_given_two_click_sum": 0.0,
        "success_false_cond_given_two_click_sum": 0.0,
        "p_arrive_sum": 0.0,
        "p_arrive_11_sum": 0.0,
        "p_arrive_same_arm_sum": 0.0,
        "p_arrive_20_sum": 0.0,
        "p_arrive_02_sum": 0.0,
        "p_success_abs_sum": 0.0,
        "p_success_true_abs_sum": 0.0,
        "p_success_false_abs_sum": 0.0,
        "p_success_intrinsic_dark_assisted_abs_sum": 0.0,
        "p_success_bg_assisted_abs_sum": 0.0,
        "fidelity_all_sum": 0.0,
        "fidelity_true_sum": 0.0,
        "fidelity_true_weighted_abs_sum": 0.0,
        "fidelity_false_sum": 0.0,
        "false_fraction_sum": 0.0,
        "corr_exx_sum": 0.0,
        "corr_eyy_sum": 0.0,
        "corr_ezz_sum": 0.0,
        "chsh_s_max_sum": 0.0,
    }


def _accumulate_window_scan_group(group: dict, entry: dict) -> None:
    group["runs_total"] += 1
    group["shots_total"] += int(entry.get("shots", 0) or 0)
    group["accepted_total"] += int(entry.get("accepted", 0) or 0)
    group["p_two_click_abs_sum"] += _safe_num(entry.get("p_two_click_abs")) or 0.0
    group["accepted_cond_given_two_click_sum"] += _safe_num(entry.get("accepted_cond_given_two_click")) or 0.0
    group["success_cond_given_two_click_sum"] += _safe_num(entry.get("success_cond_given_two_click")) or 0.0
    group["success_true_cond_given_two_click_sum"] += _safe_num(entry.get("success_true_cond_given_two_click")) or 0.0
    group["success_false_cond_given_two_click_sum"] += _safe_num(entry.get("success_false_cond_given_two_click")) or 0.0
    for key in _CORE_METRIC_FIELDS:
        if key == "p_success_true_given_arrival":
            continue
        group[f"{key}_sum"] += _safe_num(entry.get(key)) or 0.0
    p_success_true_abs = _safe_num(entry.get("p_success_true_abs")) or 0.0
    fidelity_true = _safe_num(entry.get("fidelity_true")) or 0.0
    group["fidelity_true_weighted_abs_sum"] += p_success_true_abs * fidelity_true
    group["p_success_intrinsic_dark_assisted_abs_sum"] += (
        _safe_num(entry.get("p_success_intrinsic_dark_assisted_abs")) or 0.0
    )
    group["p_success_bg_assisted_abs_sum"] += _safe_num(entry.get("p_success_bg_assisted_abs")) or 0.0


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

        trials_writer.writerow(_WINDOW_SCAN_TRIAL_HEADER)
        runs_writer.writerow(_WINDOW_SCAN_RUN_HEADER)

        for meta_path, data, metrics, tid in _iter_meta_entries(results_dir):
            if not _is_core_experiment(data, "WINDOW_SCAN"):
                continue

            run_index = _extract_run_index(metrics, tid, (r"wscan_run_(\d+)",))

            base_entry = metrics.get("window_scan")
            if not isinstance(base_entry, dict):
                raise ValueError("WINDOW_SCAN summary 需要 metrics.window_scan")

            clicks_shared = _read_click_payload(meta_path, [])
            if not isinstance(clicks_shared, list):
                raise ValueError("WINDOW_SCAN clicks 必须为 list")

            window_values = []
            if all(
                value is not None
                for value in (
                    config.run.window_sweep_start_ns,
                    config.run.window_sweep_end_ns,
                    config.run.window_sweep_step_ns,
                )
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
                success_bg_sum = 0.0
                success_intrinsic_sum = 0.0
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
                    p_true, p_bg, p_intrinsic = _resolve_record_component_weights(record)
                    success_true_sum += p_true
                    success_false_sum += 1.0 - p_true
                    success_bg_sum += p_bg
                    success_intrinsic_sum += p_intrinsic
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
                p_success_bg_assisted_abs = float(p_two_click_abs * (success_bg_sum / shots_total))
                p_success_intrinsic_dark_assisted_abs = float(
                    p_two_click_abs * (success_intrinsic_sum / shots_total)
                )
                base_arrive_metrics = {
                    field: _safe_num(base_entry.get(field))
                    for field in ("p_arrive", "p_arrive_11", "p_arrive_same_arm", "p_arrive_20", "p_arrive_02")
                }
                p_arrive_11 = base_arrive_metrics["p_arrive_11"]
                p_success_true_given_arrival = (
                    (p_success_true_abs / p_arrive_11) if (p_arrive_11 or 0.0) > 0.0 else 0.0
                )
                expanded_windows.append(
                    {
                        "window_ns": float(window_ns),
                        "window_bins": int(window_bins),
                        "shots": shots,
                        "accepted": int(accepted),
                        "success": int(success),
                        "p_two_click_abs": p_two_click_abs,
                        "accepted_cond_given_two_click": accepted_cond_given_two_click,
                        "success_cond_given_two_click": success_cond_given_two_click,
                        "success_true_cond_given_two_click": success_true_cond_given_two_click,
                        "success_false_cond_given_two_click": success_false_cond_given_two_click,
                        **base_arrive_metrics,
                        "p_success_abs": p_success_abs,
                        "p_success_true_abs": p_success_true_abs,
                        "p_success_false_abs": p_success_false_abs,
                        "p_success_true_given_arrival": p_success_true_given_arrival,
                        "fidelity_all": float(np.mean(fidelity_all_vals)) if fidelity_all_vals else 0.0,
                        "fidelity_true": (fidelity_true_num / success_true_sum) if success_true_sum > 0.0 else 0.0,
                        "fidelity_false": (fidelity_false_num / success_false_sum) if success_false_sum > 0.0 else 0.0,
                        "false_fraction": (p_success_false_abs / p_success_abs) if p_success_abs > 0.0 else 0.0,
                        "corr_exx": float(np.mean(corr_exx_vals)) if corr_exx_vals else 0.0,
                        "corr_eyy": float(np.mean(corr_eyy_vals)) if corr_eyy_vals else 0.0,
                        "corr_ezz": float(np.mean(corr_ezz_vals)) if corr_ezz_vals else 0.0,
                        "chsh_s_max": float(np.mean(chsh_vals)) if chsh_vals else 0.0,
                        "p_success_intrinsic_dark_assisted_abs": p_success_intrinsic_dark_assisted_abs,
                        "p_success_bg_assisted_abs": p_success_bg_assisted_abs,
                    }
                )

            for entry in expanded_windows:
                window_ns = float(entry.get("window_ns", 0.0) or 0.0)
                window_key = f"{window_ns:.9f}"
                window_bins = int(entry.get("window_bins", 0) or 0)
                shots = int(entry.get("shots", 0) or 0)
                accepted = int(entry.get("accepted", 0) or 0)
                success = int(entry.get("success", 0) or 0)
                metric_values = _row_values(entry, _CORE_METRIC_FIELDS)

                runs_writer.writerow([
                    tid,
                    window_ns,
                    window_bins,
                    run_index,
                    shots,
                    accepted,
                    success,
                    *_row_values(entry, _WINDOW_SCAN_RUN_FIELDS),
                    data.get("timestamp"),
                ])

                group = groups.setdefault(
                    window_key,
                    _init_window_scan_group(window_ns, config.run.runs),
                )
                _accumulate_window_scan_group(group, entry)

                if not clicks_shared:
                    trials_writer.writerow([
                        window_ns,
                        window_bins,
                        run_index,
                        -1,
                        "",
                        "",
                        "",
                        *_EMPTY_RECORD_WEIGHTS,
                        *metric_values,
                        *_click_values(),
                    ])
                    continue

                for record in clicks_shared:
                    shot_copy = dict(record)
                    shot_copy["accepted_by_window"] = bool(_record_within_window_bins(shot_copy, int(window_bins)))
                    shot_copy["success"] = bool(_record_success_by_window(shot_copy, int(window_bins)))
                    shot_copy["bell"] = shot_copy.get("bell") if shot_copy.get("success") else ""
                    shot_idx = shot_copy.get("shot_index")
                    shot_success = shot_copy.get("success")
                    bell = shot_copy.get("bell")
                    accepted_by_window = shot_copy.get("accepted_by_window")
                    (
                        p_true_given_record,
                        p_bg_assist_given_record,
                        p_intrinsic_dark_assist_given_record,
                    ) = _resolve_record_component_weights(shot_copy)
                    shot_clicks = shot_copy.get("clicks", [])
                    events = _parse_click_events(shot_clicks, "WINDOW_SCAN")
                    bins, darks, _ = _build_click_channel_maps(events)
                    _accumulate_click_analysis(
                        click_analysis_groups,
                        window_ns,
                        shot_copy,
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
                        p_bg_assist_given_record,
                        p_intrinsic_dark_assist_given_record,
                        *metric_values,
                        *_click_values(bins=bins, darks=darks),
                    ])

    attempt_rate_hz_eff = _compute_effective_attempt_rate_hz(
        config.run.attempt_rate_hz,
        config.run.attempt_overhead_us,
    )
    rows = [
        _finalize_group_summary(groups[key], attempt_rate_hz_eff)
        for key in sorted(groups.keys(), key=lambda item: float(item))
    ]
    max_herald = max((row["herald_rate_abs"] for row in rows), default=0.0)

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(_WINDOW_SCAN_SUMMARY_HEADER)
        for row in rows:
            acceptance_fraction = (row["herald_rate_abs"] / max_herald) if max_herald > 0 else 0.0
            summary_writer.writerow([
                row["window_ns"],
                row["runs_target"],
                row["runs_total"],
                row["shots_total"],
                row["accepted_total"],
                *_row_values(row, _WINDOW_SCAN_SUMMARY_METRIC_FIELDS),
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

        trials_writer.writerow(_LENGTH_SCAN_TRIAL_HEADER)
        runs_writer.writerow(_LENGTH_SCAN_RUN_HEADER)

        for meta_path, data, metrics, tid in _iter_meta_entries(results_dir):
            if not _is_core_experiment(data, "LENGTH_SCAN"):
                continue

            run_index = _extract_run_index(metrics, tid, (r"lscan_len_\d+_run_(\d+)", r"lscan_run_(\d+)"))

            lengths = metrics.get("lengths", [])
            if not isinstance(lengths, list):
                lengths = []

            clicks_by_length = _read_click_mapping(meta_path)

            for entry in lengths:
                length_km = float(entry.get("length_km", 0.0) or 0.0)
                length_key = f"{length_km:.9f}"
                point = _extract_common_scan_point(entry)
                metric_values = _row_values(point, _CORE_METRIC_FIELDS)
                shots = int(point["shots"])
                success = int(point["success"])

                runs_writer.writerow([
                    tid,
                    length_km,
                    run_index,
                    shots,
                    success,
                    point["p_two_click_abs"],
                    point["window_ns"],
                    point["attempt_rate_hz"],
                    point["event_rate_hz"],
                    *metric_values,
                    data.get("timestamp"),
                ])

                group = groups.setdefault(
                    length_key,
                    _init_common_scan_group(config.run.runs, ("length_km",), (length_km,)),
                )
                _accumulate_common_scan_group(group, point)

                records = clicks_by_length.get(length_key, [])
                if not isinstance(records, list) or not records:
                    trials_writer.writerow([
                        length_km,
                        run_index,
                        -1,
                        "",
                        "",
                        "",
                        "",
                        "",
                        point["window_ns"],
                        point["attempt_rate_hz"],
                        point["event_rate_hz"],
                        *metric_values,
                        *_click_values(),
                    ])
                    continue

                for record in records:
                    shot_idx = record.get("shot_index")
                    shot_success = record.get("success")
                    bell = record.get("bell")
                    (
                        p_true_given_record,
                        p_bg_assist_given_record,
                        p_intrinsic_dark_assist_given_record,
                    ) = _resolve_record_component_weights(record)
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
                        p_true_given_record,
                        p_bg_assist_given_record,
                        p_intrinsic_dark_assist_given_record,
                        point["window_ns"],
                        point["attempt_rate_hz"],
                        point["event_rate_hz"],
                        *metric_values,
                        *_click_values(bins=bins, darks=darks),
                    ])

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow([
            "length_km",
            "runs_target",
            "runs_total",
            "shots_total",
            "window_ns",
            "attempt_rate_hz",
            *_LENGTH_SCAN_SUMMARY_FIELDS,
        ])
        for key in sorted(groups.keys(), key=lambda item: float(item)):
            group = groups[key]
            summary = _finalize_common_scan_group(group)
            length_summary_values = [
                *_row_values(summary, _COMMON_SCAN_SUMMARY_FIELDS[:-1]),
                summary["event_rate_hz_avg"],
                summary["sbr_true_false"],
            ]
            summary_writer.writerow([
                group["length_km"],
                group["runs_target"],
                summary["runs_total"],
                summary["shots_total"],
                config.run.window_ns,
                summary["attempt_rate_hz_avg"],
                *length_summary_values,
            ])

    _write_click_analysis_outputs(summary_dir, "length_scan", "length_km", click_analysis_groups)


def _extract_common_scan_point(entry: dict) -> dict:
    point = {
        "shots": int(entry.get("shots", 0) or 0),
        "success": int(entry.get("success", 0) or 0),
        "window_ns": _safe_num(entry.get("window_ns")),
        "attempt_rate_hz": _safe_num(entry.get("attempt_rate_hz")),
        "event_rate_hz": _safe_num(entry.get("event_rate_hz")),
        "p_two_click_abs": _safe_num(entry.get("p_two_click_abs")),
        "p_arrive": _safe_num(entry.get("p_arrive")),
        "p_arrive_11": _safe_num(entry.get("p_arrive_11")),
        "p_arrive_same_arm": _safe_num(entry.get("p_arrive_same_arm")),
        "p_arrive_20": _safe_num(entry.get("p_arrive_20")),
        "p_arrive_02": _safe_num(entry.get("p_arrive_02")),
        "p_success_abs": _safe_num(entry.get("p_success_abs")),
        "p_success_true_abs": _safe_num(entry.get("p_success_true_abs")),
        "p_success_false_abs": _safe_num(entry.get("p_success_false_abs")),
        "p_success_true_given_arrival": _safe_num(entry.get("p_success_true_given_arrival")),
        "fidelity_all": _safe_num(entry.get("fidelity_all")),
        "fidelity_true": _safe_num(entry.get("fidelity_true")),
        "fidelity_false": _safe_num(entry.get("fidelity_false")),
        "false_fraction": _safe_num(entry.get("false_fraction")),
        "corr_exx": _safe_num(entry.get("corr_exx")),
        "corr_eyy": _safe_num(entry.get("corr_eyy")),
        "corr_ezz": _safe_num(entry.get("corr_ezz")),
        "chsh_s_max": _safe_num(entry.get("chsh_s_max")),
        "p_success_intrinsic_dark_assisted": _safe_num(
            entry.get("p_success_intrinsic_dark_assisted")
        ),
        "p_success_bg_assisted": _safe_num(entry.get("p_success_bg_assisted")),
    }
    if (
        point["event_rate_hz"] is None
        and point["p_success_abs"] is not None
        and point["attempt_rate_hz"] is not None
    ):
        point["event_rate_hz"] = float(point["p_success_abs"]) * float(point["attempt_rate_hz"])
    return point


def _init_common_scan_group(
    runs_target: int,
    group_columns: tuple[str, ...],
    group_values: tuple[float, ...],
) -> dict:
    group = {
        "runs_target": int(runs_target),
        "runs_total": 0,
        "shots_total": 0,
        "success_total": 0,
        "window_ns_sum": 0.0,
        "attempt_rate_hz_sum": 0.0,
        "event_rate_hz_sum": 0.0,
        "p_two_click_abs_sum": 0.0,
        "p_arrive_sum": 0.0,
        "p_arrive_11_sum": 0.0,
        "p_arrive_same_arm_sum": 0.0,
        "p_arrive_20_sum": 0.0,
        "p_arrive_02_sum": 0.0,
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
        "p_success_intrinsic_dark_assisted_sum": 0.0,
        "p_success_bg_assisted_sum": 0.0,
        "__sort_key": tuple(float(value) for value in group_values),
    }
    for column, value in zip(group_columns, group_values):
        group[column] = float(value)
    return group


def _accumulate_common_scan_group(group: dict, point: dict) -> None:
    group["runs_total"] += 1
    group["shots_total"] += int(point["shots"])
    group["success_total"] += int(point["success"])
    group["window_ns_sum"] += point["window_ns"] or 0.0
    group["attempt_rate_hz_sum"] += point["attempt_rate_hz"] or 0.0
    group["event_rate_hz_sum"] += point["event_rate_hz"] or 0.0
    group["p_two_click_abs_sum"] += point["p_two_click_abs"] or 0.0
    group["p_arrive_sum"] += point["p_arrive"] or 0.0
    group["p_arrive_11_sum"] += point["p_arrive_11"] or 0.0
    group["p_arrive_same_arm_sum"] += point["p_arrive_same_arm"] or 0.0
    group["p_arrive_20_sum"] += point["p_arrive_20"] or 0.0
    group["p_arrive_02_sum"] += point["p_arrive_02"] or 0.0
    group["p_success_abs_sum"] += point["p_success_abs"] or 0.0
    group["p_success_true_abs_sum"] += point["p_success_true_abs"] or 0.0
    group["p_success_false_abs_sum"] += point["p_success_false_abs"] or 0.0
    group["fidelity_all_sum"] += point["fidelity_all"] or 0.0
    group["fidelity_true_sum"] += point["fidelity_true"] or 0.0
    group["fidelity_false_sum"] += point["fidelity_false"] or 0.0
    group["corr_exx_sum"] += point["corr_exx"] or 0.0
    group["corr_eyy_sum"] += point["corr_eyy"] or 0.0
    group["corr_ezz_sum"] += point["corr_ezz"] or 0.0
    group["chsh_s_max_sum"] += point["chsh_s_max"] or 0.0
    group["p_success_intrinsic_dark_assisted_sum"] += point["p_success_intrinsic_dark_assisted"] or 0.0
    group["p_success_bg_assisted_sum"] += point["p_success_bg_assisted"] or 0.0


def _finalize_common_scan_group(group: dict) -> dict:
    runs_total = int(group["runs_total"])
    p_success_abs_avg = (group["p_success_abs_sum"] / runs_total) if runs_total > 0 else 0.0
    p_success_true_abs_avg = (group["p_success_true_abs_sum"] / runs_total) if runs_total > 0 else 0.0
    p_success_false_abs_avg = (group["p_success_false_abs_sum"] / runs_total) if runs_total > 0 else 0.0
    p_success_true_given_arrival11_global = (
        group["p_success_true_abs_sum"] / group["p_arrive_11_sum"] if group["p_arrive_11_sum"] > 0 else 0.0
    )
    false_fraction_global = (
        group["p_success_false_abs_sum"] / group["p_success_abs_sum"] if group["p_success_abs_sum"] > 0 else 0.0
    )
    herald_rate_abs = p_success_abs_avg
    sbr_true_false = (p_success_true_abs_avg / p_success_false_abs_avg) if p_success_false_abs_avg > 0 else None
    return {
        "runs_total": runs_total,
        "shots_total": int(group["shots_total"]),
        "success_total": int(group["success_total"]),
        "window_ns_avg": (group["window_ns_sum"] / runs_total) if runs_total > 0 else 0.0,
        "attempt_rate_hz_avg": (group["attempt_rate_hz_sum"] / runs_total) if runs_total > 0 else 0.0,
        "event_rate_hz_avg": (group["event_rate_hz_sum"] / runs_total) if runs_total > 0 else 0.0,
        "p_two_click_abs_avg": (group["p_two_click_abs_sum"] / runs_total) if runs_total > 0 else 0.0,
        "p_arrive_avg": (group["p_arrive_sum"] / runs_total) if runs_total > 0 else 0.0,
        "p_arrive_11_avg": (group["p_arrive_11_sum"] / runs_total) if runs_total > 0 else 0.0,
        "p_arrive_same_arm_avg": (group["p_arrive_same_arm_sum"] / runs_total) if runs_total > 0 else 0.0,
        "p_arrive_20_avg": (group["p_arrive_20_sum"] / runs_total) if runs_total > 0 else 0.0,
        "p_arrive_02_avg": (group["p_arrive_02_sum"] / runs_total) if runs_total > 0 else 0.0,
        "p_success_abs_avg": p_success_abs_avg,
        "p_success_true_abs_avg": p_success_true_abs_avg,
        "p_success_false_abs_avg": p_success_false_abs_avg,
        "p_success_true_given_arrival11_global": p_success_true_given_arrival11_global,
        "false_fraction_global": false_fraction_global,
        "fidelity_all_avg": (group["fidelity_all_sum"] / runs_total) if runs_total > 0 else 0.0,
        "fidelity_true_avg": (group["fidelity_true_sum"] / runs_total) if runs_total > 0 else 0.0,
        "fidelity_false_avg": (group["fidelity_false_sum"] / runs_total) if runs_total > 0 else 0.0,
        "corr_exx_avg": (group["corr_exx_sum"] / runs_total) if runs_total > 0 else 0.0,
        "corr_eyy_avg": (group["corr_eyy_sum"] / runs_total) if runs_total > 0 else 0.0,
        "corr_ezz_avg": (group["corr_ezz_sum"] / runs_total) if runs_total > 0 else 0.0,
        "chsh_s_max_avg": (group["chsh_s_max_sum"] / runs_total) if runs_total > 0 else 0.0,
        "herald_rate_abs": herald_rate_abs,
        "sbr_true_false": sbr_true_false,
        "p_success_intrinsic_dark_assisted_avg": (
            (group["p_success_intrinsic_dark_assisted_sum"] / runs_total) if runs_total > 0 else 0.0
        ),
        "p_success_bg_assisted_avg": (
            (group["p_success_bg_assisted_sum"] / runs_total) if runs_total > 0 else 0.0
        ),
    }


def _write_generic_noise_scan_summary(
    paths: dict,
    config: SimConfig,
    *,
    experiment_name: str,
    metrics_key: str,
    run_index_patterns: tuple[str, ...],
    runs_filename: str,
    summary_filename: str,
    group_columns: tuple[str, ...],
    group_value_reader,
) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    runs_path = summary_dir / runs_filename
    summary_path = summary_dir / summary_filename
    groups = {}

    with open(runs_path, "w", encoding="utf-8", newline="") as runs_file:
        runs_writer = csv.writer(runs_file)
        runs_writer.writerow(["id", *group_columns, "run_index", "shots", "success", *_NOISE_SCAN_RUN_FIELDS, "timestamp"])

        for _, data, metrics, tid in _iter_meta_entries(results_dir):
            if not _is_core_experiment(data, experiment_name):
                continue

            run_index = _extract_run_index(metrics, tid, run_index_patterns)
            entries = metrics.get(metrics_key, [])
            if not isinstance(entries, list):
                entries = []

            for entry in entries:
                group_values = tuple(float(value) for value in group_value_reader(entry))
                group_key = "|".join(f"{value:.9f}" for value in group_values)
                point = _extract_common_scan_point(entry)

                runs_writer.writerow([
                    tid,
                    *group_values,
                    run_index,
                    point["shots"],
                    point["success"],
                    *_row_values(point, _NOISE_SCAN_RUN_FIELDS),
                    data.get("timestamp"),
                ])

                group = groups.setdefault(
                    group_key,
                    _init_common_scan_group(config.run.runs, group_columns, group_values),
                )
                _accumulate_common_scan_group(group, point)

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow([
            *group_columns,
            "runs_target",
            "runs_total",
            "shots_total",
            "success_total",
            *_NOISE_SCAN_SUMMARY_FIELDS,
        ])
        for group in sorted(groups.values(), key=lambda item: item["__sort_key"]):
            summary = _finalize_common_scan_group(group)
            summary_writer.writerow([
                *(group[column] for column in group_columns),
                group["runs_target"],
                summary["runs_total"],
                summary["shots_total"],
                summary["success_total"],
                *_row_values(summary, _NOISE_SCAN_SUMMARY_FIELDS),
            ])


def _write_qfc_noise_scan_summary(paths: dict, config: SimConfig) -> None:
    _write_generic_noise_scan_summary(
        paths=paths,
        config=config,
        experiment_name="QFC_NOISE_SCAN",
        metrics_key="qfc_noise_levels",
        run_index_patterns=(r"qscan_noise_\d+_run_(\d+)", r"qscan_run_(\d+)"),
        runs_filename="qfc_noise_scan_runs.csv",
        summary_filename="qfc_noise_scan_summary.csv",
        group_columns=("qfc_noise_sd_cps_per_mhz",),
        group_value_reader=lambda entry: (entry.get("qfc_noise_sd_cps_per_mhz", 0.0),),
    )


def _write_qfc_eff_noise_scan_summary(paths: dict, config: SimConfig) -> None:
    _write_generic_noise_scan_summary(
        paths=paths,
        config=config,
        experiment_name="QFC_EFF_NOISE_SCAN",
        metrics_key="qfc_eff_noise_points",
        run_index_patterns=(r"qescan_eta_\d+_noise_\d+_run_(\d+)", r"qescan_run_(\d+)"),
        runs_filename="qfc_eff_noise_scan_runs.csv",
        summary_filename="qfc_eff_noise_scan_summary.csv",
        group_columns=("qfc_eta", "qfc_noise_sd_cps_per_mhz"),
        group_value_reader=lambda entry: (
            entry.get("qfc_eta", 0.0),
            entry.get("qfc_noise_sd_cps_per_mhz", 0.0),
        ),
    )


def _write_detector_bg_scan_summary(paths: dict, config: SimConfig) -> None:
    _write_generic_noise_scan_summary(
        paths=paths,
        config=config,
        experiment_name="DETECTOR_BG_SCAN",
        metrics_key="detector_bg_points",
        run_index_patterns=(r"dscan_eta_\d+_bg_\d+_run_(\d+)", r"dscan_run_(\d+)"),
        runs_filename="detector_bg_scan_runs.csv",
        summary_filename="detector_bg_scan_summary.csv",
        group_columns=("eta_det", "bg_rate_mean_hz"),
        group_value_reader=lambda entry: (
            entry.get("eta_det", 0.0),
            entry.get("bg_rate_mean_hz", 0.0),
        ),
    )


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

        trials_writer.writerow(_BSM_SCAN_TRIAL_HEADER)
        runs_writer.writerow(_BSM_SCAN_RUN_HEADER)

        for meta_path, data, metrics, tid in _iter_meta_entries(results_dir):
            if not _is_core_experiment(data, "BSM_SCAN"):
                continue

            run_index = _extract_run_index(metrics, tid, (r"bscan_theta_\d+_run_(\d+)", r"bscan_run_(\d+)"))

            entries = metrics.get("bs_thetas", [])
            if not isinstance(entries, list):
                entries = []

            clicks_by_theta = _read_click_mapping(meta_path)

            for entry in entries:
                bs_theta = float(entry.get("bs_theta", 0.0) or 0.0)
                bs_key = f"{bs_theta:.9f}"
                bs_split_ratio = _safe_num(entry.get("bs_split_ratio"))
                point = _extract_common_scan_point(entry)
                metric_values = _row_values(point, _CORE_METRIC_FIELDS)

                run_row = [
                    tid,
                    bs_theta,
                    bs_split_ratio,
                    run_index,
                    int(point["shots"]),
                    int(point["success"]),
                    point["p_two_click_abs"],
                    *metric_values,
                ]
                run_row.extend(
                    value
                    for pattern_key in BSM_PATTERN_KEYS
                    for value in (
                        int(entry.get(pattern_key, 0) or 0),
                        _safe_num(entry.get(f"{pattern_key}_rate")) or 0.0,
                        _safe_num(entry.get(f"{pattern_key}_true_abs")) or 0.0,
                        _safe_num(entry.get(f"{pattern_key}_false_abs")) or 0.0,
                    )
                )
                run_row.append(data.get("timestamp"))
                runs_writer.writerow(run_row)

                group = groups.setdefault(
                    bs_key,
                    {
                        **_init_common_scan_group(
                            config.run.runs,
                            ("bs_theta", "bs_split_ratio"),
                            (bs_theta, float(bs_split_ratio or 0.0)),
                        ),
                        "pattern_sums": {pattern_key: 0 for pattern_key in BSM_PATTERN_KEYS},
                        "pattern_true_abs_sums": {pattern_key: 0.0 for pattern_key in BSM_PATTERN_KEYS},
                        "pattern_false_abs_sums": {pattern_key: 0.0 for pattern_key in BSM_PATTERN_KEYS},
                    },
                )
                _accumulate_common_scan_group(group, point)
                for pattern_key in BSM_PATTERN_KEYS:
                    group["pattern_sums"][pattern_key] += int(entry.get(pattern_key, 0) or 0)
                    group["pattern_true_abs_sums"][pattern_key] += (
                        _safe_num(entry.get(f"{pattern_key}_true_abs")) or 0.0
                    )
                    group["pattern_false_abs_sums"][pattern_key] += (
                        _safe_num(entry.get(f"{pattern_key}_false_abs")) or 0.0
                    )

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
                        *_EMPTY_RECORD_WEIGHTS,
                        *metric_values,
                        *_click_values(include_sources=True),
                    ]
                    trials_writer.writerow(row)
                    continue

                for record in records:
                    shot_idx = record.get("shot_index")
                    shot_success = record.get("success")
                    bell = record.get("bell")
                    pattern = record.get("pattern")
                    (
                        p_true_given_record,
                        p_bg_assist_given_record,
                        p_intrinsic_dark_assist_given_record,
                    ) = _resolve_record_component_weights(record)
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
                        p_true_given_record,
                        p_bg_assist_given_record,
                        p_intrinsic_dark_assist_given_record,
                        *metric_values,
                        *_click_values(bins=bins, darks=darks, sources=sources, include_sources=True),
                    ]
                    trials_writer.writerow(row)

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(_BSM_SCAN_SUMMARY_HEADER)

        for key in sorted(groups.keys(), key=lambda value: float(value)):
            group = groups[key]
            summary = _finalize_common_scan_group(group)
            output_row = [
                group["bs_theta"],
                group["bs_split_ratio"],
                group["runs_target"],
                summary["runs_total"],
                summary["shots_total"],
                summary["success_total"],
                *_row_values(summary, _COMMON_SCAN_SUMMARY_FIELDS),
            ]
            runs_total = summary["runs_total"]
            shots_total = summary["shots_total"]
            output_row.extend(
                value
                for pattern_key in BSM_PATTERN_KEYS
                for value in (
                    int(group["pattern_sums"][pattern_key]),
                    (
                        float(group["pattern_sums"][pattern_key]) / float(shots_total)
                        if shots_total > 0
                        else 0.0
                    ),
                    (
                        group["pattern_true_abs_sums"][pattern_key] / float(runs_total)
                        if runs_total > 0
                        else 0.0
                    ),
                    (
                        group["pattern_false_abs_sums"][pattern_key] / float(runs_total)
                        if runs_total > 0
                        else 0.0
                    ),
                )
            )
            summary_writer.writerow(output_row)

    _write_click_analysis_outputs(summary_dir, "bsm_scan", "bs_theta", click_analysis_groups)


def _iter_meta_entries(results_dir):
    for meta_path in sorted(results_dir.glob("result_*/meta.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = data.get("metrics", {})
        tid = str(data.get("id", ""))
        yield meta_path, data, metrics, tid


def _read_click_payload(meta_path, default):
    clicks_path = meta_path.parent / "raw" / "clicks.json"
    if not clicks_path.exists():
        return default
    try:
        payload = json.loads(clicks_path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if not isinstance(payload, dict):
        return default
    return payload.get("clicks", default)


def _read_click_mapping(meta_path) -> dict:
    payload = _read_click_payload(meta_path, {})
    return payload if isinstance(payload, dict) else {}


def _init_hom_tau_state(tau_ns: float) -> dict:
    return {
        "tau_ns": float(tau_ns),
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
    }


def _write_hom_summary(paths: dict, config: SimConfig) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    window_bins = None
    if config.hom is not None:
        window_bins = _compute_window_bins(
            config.hom.window_ns,
            config.emission.dt_ns,
            detection_gate_ns=config.noise.detector_gate_ns,
        )

    trials_path = summary_dir / "hom_trials.csv"
    tau_path = summary_dir / "hom_summary.csv"
    tau_states = {}
    click_analysis_groups = {}

    with open(trials_path, "w", encoding="utf-8", newline="") as trials_file:
        trials_writer = csv.writer(trials_file)
        trials_writer.writerow(_HOM_TRIAL_HEADER)

        for meta_path, data, metrics, tid in _iter_meta_entries(results_dir):
            if not _is_core_experiment(data, "HOM"):
                continue

            tau_ns = float(metrics.get("tau_ns", 0.0) or 0.0)
            run_index = _extract_run_index(metrics, tid, (r"hom_tau_[+-]?\d+\.\d+_run_(\d+)",))
            p_arrive = metrics.get("p_arrive")
            tau_key = f"{tau_ns:.6f}"
            state = tau_states.setdefault(tau_key, _init_hom_tau_state(tau_ns))
            state["runs_total"] += 1
            if data.get("status") != "ok":
                continue

            state["coinc"] += int(metrics.get("coinc", 0) or 0)
            if p_arrive is not None:
                p_arrive_float = float(p_arrive)
                state["p_arrive_sum"] += p_arrive_float
                state["arrive_trials"] += p_arrive_float * config.run.shots_per_run

            clicks = _read_click_payload(meta_path, [])
            shots_in_run = len(clicks) if clicks else config.run.shots_per_run
            state["shots_total"] += shots_in_run
            if not clicks:
                trials_writer.writerow([
                    f"{tau_ns:.6f}",
                    run_index,
                    -1,
                    "",
                    "",
                    "",
                    p_arrive,
                    *_click_values(),
                ])
                continue

            for shot_idx, record in enumerate(clicks):
                if not isinstance(record, dict):
                    raise ValueError("HOM clicks 记录必须为 dict")
                p_true, p_bg, p_intrinsic = _resolve_record_component_weights(record)
                events = _parse_click_events(record.get("clicks", []), "HOM")
                bins, darks, _ = _build_click_channel_maps(events)
                dark_clicks = sum(1 for event in events if event.is_dark)
                state["dark_clicks_total"] += dark_clicks
                state["clicks_total"] += len(events)

                is_success = bool(events and _is_port_samepol_coincidence(events, window_bins))
                _accumulate_click_analysis(click_analysis_groups, tau_ns, record, events, is_success)
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
                    p_true,
                    p_bg,
                    p_intrinsic,
                    p_arrive,
                    *_click_values(bins=bins, darks=darks),
                ])

    with open(tau_path, "w", encoding="utf-8", newline="") as tau_file:
        tau_writer = csv.writer(tau_file)
        tau_writer.writerow(_HOM_SUMMARY_HEADER)
        for tau_key in sorted(tau_states, key=lambda key: float(key)):
            state = tau_states[tau_key]
            runs_total = int(state["runs_total"])
            p_arrive_avg = (state["p_arrive_sum"] / runs_total) if runs_total > 0 else 0.0
            coinc_rate = (state["coinc"] / state["arrive_trials"]) if state["arrive_trials"] > 0 else 0.0
            dark_click_rate = (state["dark_clicks_total"] / state["clicks_total"]) if state["clicks_total"] > 0 else 0.0
            dark_click_rate_per_det = (
                state["dark_clicks_total"] / (state["shots_total"] * len(_DETECTOR_ORDER))
                if state["shots_total"] > 0
                else 0.0
            )
            tau_writer.writerow([
                f"{state['tau_ns']:.6f}",
                config.run.runs,
                runs_total,
                state["coinc"],
                f"{coinc_rate:.8f}",
                f"{p_arrive_avg:.6f}",
                f"{state['arrive_trials']:.6f}",
                f"{config.hom.window_ns if config.hom else 0.0:.3f}",
                config.run.shots_per_run,
                state["shots_total"],
                state["coinc_true"],
                state["coinc_dark_any"],
                state["coinc_dark_single"],
                state["coinc_dark_double"],
                state["dark_clicks_total"],
                state["clicks_total"],
                f"{dark_click_rate:.8f}",
                f"{dark_click_rate_per_det:.8f}",
            ])

    _write_click_analysis_outputs(summary_dir, "hom", "tau_ns", click_analysis_groups)


def _write_sim_trials(paths: dict) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    trials_path = summary_dir / "sim_trials.csv"
    click_analysis_groups = {}
    with open(trials_path, "w", encoding="utf-8", newline="") as trials_file:
        trials_writer = csv.writer(trials_file)
        trials_writer.writerow(_SIM_TRIAL_HEADER)

        for meta_path, data, metrics, tid in _iter_meta_entries(results_dir):
            if not _is_core_experiment(data, "SIM"):
                continue

            run_index = _extract_run_index(metrics, tid, (r"sim_run_(\d+)",))
            task_mode = data.get("mode")
            window_ns = metrics.get("window_ns")
            metric_values = _row_values(metrics, _CORE_METRIC_FIELDS)
            clicks = _read_click_payload(meta_path, [])

            if not clicks:
                trials_writer.writerow([
                    task_mode,
                    window_ns,
                    run_index,
                    -1,
                    "",
                    "",
                    "",
                    "",
                    "",
                    *metric_values,
                    *_click_values(),
                ])
                continue

            for record in clicks:
                shot_idx = record.get("shot_index")
                success = record.get("success")
                bell = record.get("bell")
                p_true, p_bg, p_intrinsic = _resolve_record_component_weights(record)
                events = _parse_click_events(record.get("clicks", []), "SIM")
                bins, darks, _ = _build_click_channel_maps(events)
                group_window_ns = _safe_num(window_ns)
                if group_window_ns is None:
                    raise ValueError("SIM summary 需要 metrics.window_ns")
                _accumulate_click_analysis(click_analysis_groups, group_window_ns, record, events, bool(success))

                trials_writer.writerow([
                    task_mode,
                    window_ns,
                    run_index,
                    shot_idx,
                    success,
                    bell,
                    p_true,
                    p_bg,
                    p_intrinsic,
                    *metric_values,
                    *_click_values(bins=bins, darks=darks),
                ])

    _write_click_analysis_outputs(summary_dir, "sim", "window_ns", click_analysis_groups)


def _write_generic_task_summary(task_type: str, paths: dict) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / f"{task_type.lower()}_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(_GENERIC_SUMMARY_HEADER)
        for _, data, metrics, _ in _iter_meta_entries(results_dir):
            tid = data.get("id")
            if not tid:
                continue
            writer.writerow([
                tid,
                data.get("mode", task_type),
                *[metrics.get(field) for field in _GENERIC_SUMMARY_METRIC_FIELDS],
                data.get("timestamp"),
            ])


def _write_param_scan_summary(paths: dict) -> None:
    results_dir = paths["results"]
    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    records = []
    axis_keys = set()
    metric_fields = (
        "window_ns",
        "attempt_rate_hz",
        "event_rate_hz",
        "runtime_wall_s",
        "p_two_click_abs",
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
    )
    for _, data, metrics, tid in _iter_meta_entries(results_dir):
        if not _is_core_experiment(data, "PARAM_SCAN"):
            continue
        if not isinstance(metrics, dict):
            continue
        point = metrics.get("scan_point", {})
        if not isinstance(point, dict):
            point = {}
        point = {str(k): _safe_num(v) for k, v in point.items()}
        axis_keys.update(point.keys())
        run_index = _extract_run_index(metrics, tid, patterns=())
        row = {
            "id": tid,
            "scan_family": str(metrics.get("scan_family", "")).upper(),
            "run_index": run_index,
            "shots": int(_safe_num(metrics.get("shots")) or 0),
            "success": int(_safe_num(metrics.get("success")) or 0),
            "timestamp": data.get("timestamp"),
            "point": point,
        }
        for field in metric_fields:
            row[field] = _safe_num(metrics.get(field))
        records.append(row)

    axis_columns = sorted(axis_keys)
    runs_path = summary_dir / "param_scan_runs.csv"
    summary_path = summary_dir / "param_scan_summary.csv"

    with open(runs_path, "w", encoding="utf-8", newline="") as runs_file:
        writer = csv.writer(runs_file)
        writer.writerow([
            "id",
            "scan_family",
            *axis_columns,
            "run_index",
            "shots",
            "success",
            *metric_fields,
            "timestamp",
        ])
        for row in records:
            point = row["point"]
            writer.writerow([
                row["id"],
                row["scan_family"],
                *[point.get(column) for column in axis_columns],
                row["run_index"],
                row["shots"],
                row["success"],
                *[row.get(field) for field in metric_fields],
                row["timestamp"],
            ])

    groups = {}
    for row in records:
        point = row["point"]
        key_values = tuple(point.get(column) for column in axis_columns)
        key = (row["scan_family"], key_values)
        state = groups.setdefault(
            key,
            {
                "scan_family": row["scan_family"],
                "point": {column: point.get(column) for column in axis_columns},
                "runs_total": 0,
                "shots_total": 0,
                "success_total": 0,
                "metric_sums": {field: 0.0 for field in metric_fields},
                "metric_counts": {field: 0 for field in metric_fields},
            },
        )
        state["runs_total"] += 1
        state["shots_total"] += int(row["shots"])
        state["success_total"] += int(row["success"])
        for field in metric_fields:
            value = row.get(field)
            if value is None:
                continue
            state["metric_sums"][field] += float(value)
            state["metric_counts"][field] += 1

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow([
            "scan_family",
            *axis_columns,
            "runs_total",
            "shots_total",
            "success_total",
            *[f"{field}_avg" for field in metric_fields],
        ])
        for key in sorted(groups.keys(), key=lambda item: (item[0], item[1])):
            state = groups[key]
            writer.writerow([
                state["scan_family"],
                *[state["point"].get(column) for column in axis_columns],
                state["runs_total"],
                state["shots_total"],
                state["success_total"],
                *[
                    (
                        state["metric_sums"][field] / state["metric_counts"][field]
                        if state["metric_counts"][field] > 0
                        else None
                    )
                    for field in metric_fields
                ],
            ])


def write_summary(task_type: str, paths: dict, config: SimConfig) -> None:
    if task_type == "HOM":
        _write_hom_summary(paths=paths, config=config)
        return

    specialized = {
        "PARAM_SCAN": lambda paths, config: _write_param_scan_summary(paths),
        "WINDOW_SCAN": _write_window_scan_summary,
        "BSM_SCAN": _write_bsm_scan_summary,
        "LENGTH_SCAN": _write_length_scan_summary,
        "QFC_NOISE_SCAN": _write_qfc_noise_scan_summary,
        "QFC_EFF_NOISE_SCAN": _write_qfc_eff_noise_scan_summary,
        "DETECTOR_BG_SCAN": _write_detector_bg_scan_summary,
    }
    writer = specialized.get(task_type)
    if writer is not None:
        writer(paths=paths, config=config)
        return

    if task_type == "SIM":
        _write_sim_trials(paths=paths)
    _write_generic_task_summary(task_type=task_type, paths=paths)
