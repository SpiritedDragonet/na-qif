from plot_style import frame_all_axes
import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

EXPORT_PNG = False


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _default_summary_csv() -> pathlib.Path:
    data_root = pathlib.Path(__file__).resolve().parents[1] / "data"
    candidates = sorted(
        data_root.glob("*/summary/length_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    queue_root = _repo_root() / "queue"
    candidates = sorted(
        queue_root.glob("*/summary/length_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    outputs_root = _repo_root() / "outputs"
    candidates = sorted(
        outputs_root.glob("*/summary/length_scan_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    return data_root / "length_scan_server_output_latest" / "summary" / "length_scan_summary.csv"


def _default_runs_csv(summary_csv: pathlib.Path) -> pathlib.Path:
    return summary_csv.with_name("length_scan_runs.csv")


def _parse_args() -> argparse.Namespace:
    default_summary = _default_summary_csv()
    parser = argparse.ArgumentParser(description="Plot CHSH distance scaling from LENGTH_SCAN summary/runs CSV.")
    parser.add_argument("--summary-csv", type=pathlib.Path, default=default_summary)
    parser.add_argument("--runs-csv", type=pathlib.Path, default=_default_runs_csv(default_summary))
    return parser.parse_args()


def _parse_float(row: dict[str, str], key: str, line_no: int) -> float:
    raw = (row.get(key) or "").strip()
    if raw == "":
        raise ValueError(f"CSV line {line_no}: missing value for '{key}'")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"CSV line {line_no}: invalid float '{raw}' for '{key}'") from exc
    if not np.isfinite(value):
        raise ValueError(f"CSV line {line_no}: non-finite '{raw}' for '{key}'")
    return value


def _load_summary(summary_csv: pathlib.Path) -> dict[str, np.ndarray]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = ("length_km", "chsh_s_max_avg", "fidelity_true_avg", "p_success_true_abs_avg")
    missing = [key for key in required if key not in fieldnames]
    if missing:
        raise ValueError(f"Summary CSV missing columns: {', '.join(missing)}")
    if not rows:
        raise ValueError(f"Summary CSV has no rows: {summary_csv}")

    records = []
    for line_no, row in enumerate(rows, start=2):
        records.append(
            (
                _parse_float(row, "length_km", line_no),
                _parse_float(row, "chsh_s_max_avg", line_no),
                _parse_float(row, "fidelity_true_avg", line_no),
                _parse_float(row, "p_success_true_abs_avg", line_no),
            )
        )
    arr = np.asarray(records, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return {
        "length_km": arr[:, 0],
        "s_mean": arr[:, 1],
        "f_t": arr[:, 2],
        "p_t": arr[:, 3],
    }


def _load_run_stats(runs_csv: pathlib.Path, lengths: np.ndarray) -> dict[str, np.ndarray]:
    n = lengths.size
    result = {
        "s_sem": np.zeros(n, dtype=float),
        "s_violate_prob": np.zeros(n, dtype=float),
        "counts": np.zeros(n, dtype=int),
        "s_cond_mean": np.full(n, np.nan, dtype=float),
        "s_cond_sem": np.zeros(n, dtype=float),
        "s_cond_violate_prob": np.zeros(n, dtype=float),
        "s_cond_counts": np.zeros(n, dtype=int),
        "success_rate": np.zeros(n, dtype=float),
    }
    if not runs_csv.exists():
        return result

    with runs_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        if not {"length_km", "chsh_s_max"}.issubset(fieldnames):
            return result
        has_success_col = "success" in fieldnames
        groups_all: dict[float, list[float]] = {}
        groups_success: dict[float, list[float]] = {}
        groups_success_flag: dict[float, list[int]] = {}
        for row in reader:
            try:
                length_km = float((row.get("length_km") or "").strip())
                s_val = float((row.get("chsh_s_max") or "").strip())
            except ValueError:
                continue
            if not (np.isfinite(length_km) and np.isfinite(s_val)):
                continue
            key = round(length_km, 9)
            groups_all.setdefault(key, []).append(float(s_val))
            if has_success_col:
                success_raw = (row.get("success") or "").strip()
                try:
                    success_flag = int(success_raw)
                except ValueError:
                    continue
                groups_success_flag.setdefault(key, []).append(1 if success_flag > 0 else 0)
                if success_flag > 0:
                    groups_success.setdefault(key, []).append(float(s_val))
            elif s_val > 0.0:
                groups_success.setdefault(key, []).append(float(s_val))

    for i, length_km in enumerate(lengths):
        key = round(float(length_km), 9)
        vals = groups_all.get(key, [])
        if len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=float)
        result["counts"][i] = int(arr.size)
        if arr.size > 1:
            result["s_sem"][i] = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        result["s_violate_prob"][i] = float(np.mean(arr > 2.0))
        if has_success_col:
            flags = groups_success_flag.get(key, [])
            if len(flags) == len(vals) and len(flags) > 0:
                result["success_rate"][i] = float(np.mean(np.asarray(flags, dtype=float)))
            elif len(vals) > 0:
                result["success_rate"][i] = float(len(groups_success.get(key, [])) / len(vals))
        elif len(vals) > 0:
            result["success_rate"][i] = float(len(groups_success.get(key, [])) / len(vals))

        vals_success = groups_success.get(key, [])
        if len(vals_success) == 0:
            continue
        arr_success = np.asarray(vals_success, dtype=float)
        result["s_cond_counts"][i] = int(arr_success.size)
        result["s_cond_mean"][i] = float(np.mean(arr_success))
        if arr_success.size > 1:
            result["s_cond_sem"][i] = float(np.std(arr_success, ddof=1) / np.sqrt(arr_success.size))
        result["s_cond_violate_prob"][i] = float(np.mean(arr_success > 2.0))
    return result


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=12.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> None:
    args = _parse_args()
    summary = _load_summary(args.summary_csv)
    run_stats = _load_run_stats(args.runs_csv, summary["length_km"])

    length_km = summary["length_km"]
    s_mean_summary = summary["s_mean"]
    f_t = np.clip(summary["f_t"], 0.0, 1.0)
    p_t = np.clip(summary["p_t"], np.finfo(float).tiny, None)

    s_mean_plot = np.asarray(s_mean_summary, dtype=float).copy()
    s_sem_plot = np.asarray(run_stats["s_sem"], dtype=float).copy()
    violate_prob_plot = np.asarray(run_stats["s_violate_prob"], dtype=float).copy()
    cond_mask = np.asarray(run_stats["s_cond_counts"], dtype=int) > 0
    use_cond = bool(np.any(cond_mask))
    if use_cond:
        s_mean_plot[cond_mask] = np.asarray(run_stats["s_cond_mean"], dtype=float)[cond_mask]
        s_sem_plot[cond_mask] = np.asarray(run_stats["s_cond_sem"], dtype=float)[cond_mask]
        violate_prob_plot[cond_mask] = np.asarray(run_stats["s_cond_violate_prob"], dtype=float)[cond_mask]

    s_ci95 = 1.96 * s_sem_plot
    violate_pct = 100.0 * violate_prob_plot
    if np.all(np.asarray(run_stats["counts"], dtype=int) <= 0):
        violate_pct = 100.0 * (s_mean_plot > 2.0).astype(float)

    valid_nonlocal = np.where(s_mean_plot > 2.0)[0]
    max_nonlocal_length = float(length_km[valid_nonlocal[-1]]) if valid_nonlocal.size > 0 else float("nan")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.8,
        }
    )

    fig = plt.figure(figsize=(10.2, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    classical_bound = 2.0

    if use_cond:
        ax0.plot(
            length_km,
            s_mean_summary,
            color="#94a3b8",
            lw=1.5,
            ls="--",
            marker="o",
            ms=2.9,
            alpha=0.85,
            label=r"$\langle S_{\max}\rangle$（全部运行）",
        )
    ax0.axhline(
        classical_bound,
        color="#7f1d1d",
        lw=2.2,
        ls=(0, (6, 3)),
        label="经典界限 $S=2$",
        zorder=4,
    )
    s_label = (
        r"$\langle S_{\max}\rangle$（成功运行）"
        if use_cond
        else r"$\langle S_{\max}\rangle$"
    )
    ci_label = "95% 置信区间（成功运行）" if use_cond else "95% 置信区间（运行级）"
    ax0.plot(length_km, s_mean_plot, color="#0f766e", lw=2.4, marker="o", ms=3.8, label=s_label)
    ax0.fill_between(
        length_km,
        s_mean_plot - s_ci95,
        s_mean_plot + s_ci95,
        color="#14b8a6",
        alpha=0.18,
        linewidth=0.0,
        label=ci_label,
    )
    ax0.axvline(33.0, lw=1.1, color="#9ca3af", ls=":")
    ax0.set_xlabel("总光纤长度 (km)")
    ax0.set_ylabel("CHSH $S$")
    y_min = float(np.min(s_mean_plot - s_ci95))
    y_max = float(np.max(s_mean_plot + s_ci95))
    pad = max(0.06 * (y_max - y_min), 0.06)
    # 强制把经典界 S=2 保持在可见范围内，避免成功条件口径下曲线整体>2时把参考线裁掉。
    classical_margin = max(0.10 * max(y_max - y_min, 0.3), 0.06)
    y_lower = min(y_min - pad, classical_bound - classical_margin)
    y_upper = max(y_max + pad, classical_bound + classical_margin)
    ax0.set_ylim(y_lower, y_upper)
    ylim_top = float(ax0.get_ylim()[1])
    if ylim_top > classical_bound:
        ax0.axhspan(classical_bound, ylim_top, color="#ecfdf5", alpha=0.35, zorder=0)
    ax0.set_title("CHSH 随距离变化（宣告态口径）")
    ax0.legend(frameon=False, fontsize=8.4, loc="upper right")
    _panel_label(ax0, "(a)")
    ax0.text(
        0.02,
        0.93,
        r"非经典区：$S>2$",
        transform=ax0.transAxes,
        fontsize=8.3,
        color="#065f46",
    )

    if np.isfinite(max_nonlocal_length):
        ax0.text(
            0.02,
            0.08,
            rf"平均 $S>2$ 可达 {max_nonlocal_length:.1f} km",
            transform=ax0.transAxes,
            fontsize=8.5,
            color="#0f766e",
        )

    violate_label = r"$\Pr(S>2\mid \mathrm{succ})$" if use_cond else r"$\Pr(S>2)$"
    ax1.plot(length_km, violate_pct, color="#1d4ed8", lw=2.2, marker="s", ms=3.5, label=violate_label)
    ax1.set_xlabel("总光纤长度 (km)")
    ax1.set_ylabel("违背概率 (%)", color="#1e40af")
    ax1.tick_params(axis="y", colors="#1e40af")
    ax1.set_ylim(0.0, 103.0)
    ax1.axvline(33.0, lw=1.1, color="#9ca3af", ls=":")
    if use_cond:
        success_pct = 100.0 * np.asarray(run_stats["success_rate"], dtype=float)
        ax1.plot(
            length_km,
            success_pct,
            color="#475569",
            lw=1.4,
            marker="x",
            ms=3.0,
            ls=":",
            label="运行成功比例 (%)",
        )

    ax1_r = ax1.twinx()
    ax1_r.plot(length_km, f_t, color="#059669", lw=2.0, marker="D", ms=3.0, label=r"$F_t$")
    ax1_r.set_ylabel(r"条件保真度 $F_t$", color="#047857")
    ax1_r.tick_params(axis="y", colors="#047857")
    ax1_r.set_ylim(max(0.0, float(np.min(f_t)) - 0.03), min(1.0, float(np.max(f_t)) + 0.03))

    # Add normalized true-component context as a visible trend line.
    p_t_norm = p_t / float(np.max(p_t))
    ax1.plot(
        length_km,
        100.0 * p_t_norm,
        color="#f59e0b",
        lw=1.9,
        marker="^",
        ms=3.3,
        ls="-.",
        label=r"归一化真成功率",
    )

    lines = []
    labels = []
    for axis in (ax1, ax1_r):
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    ax1.legend(lines, labels, frameon=False, fontsize=8.2, loc="upper right")
    ax1.set_title("违背概率与态质量对照")
    _panel_label(ax1, "(b)")

    fig.suptitle(
        "CHSH 随距离趋势（长度扫描数据）",
        fontsize=12.4,
        fontweight="bold",
    )

    out_base = pathlib.Path(__file__).with_suffix("")
    frame_all_axes(fig)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=260)
    if EXPORT_PNG:
        fig.savefig(out_base.with_suffix(".png"), dpi=230)
    plt.close(fig)


if __name__ == "__main__":
    main()
