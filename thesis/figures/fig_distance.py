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
    parser = argparse.ArgumentParser(description="Plot distance scaling from LENGTH_SCAN summary/runs CSV.")
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

    required = (
        "length_km",
        "herald_rate_abs",
        "p_success_true_abs_avg",
        "p_success_false_abs_avg",
        "fidelity_true_avg",
        "false_fraction_global",
    )
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
                _parse_float(row, "herald_rate_abs", line_no),
                _parse_float(row, "p_success_true_abs_avg", line_no),
                _parse_float(row, "p_success_false_abs_avg", line_no),
                _parse_float(row, "fidelity_true_avg", line_no),
                _parse_float(row, "false_fraction_global", line_no),
            )
        )
    arr = np.asarray(records, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return {
        "length_km": arr[:, 0],
        "p_s": arr[:, 1],
        "p_t": arr[:, 2],
        "p_f": arr[:, 3],
        "f_t": arr[:, 4],
        "false_fraction": arr[:, 5],
    }


def _load_run_uncertainty(runs_csv: pathlib.Path, lengths: np.ndarray) -> dict[str, np.ndarray]:
    n = lengths.size
    zeros = np.zeros(n, dtype=float)
    result = {
        "fidelity_sem": zeros.copy(),
        "false_fraction_sem": zeros.copy(),
        "counts": np.zeros(n, dtype=int),
    }
    if not runs_csv.exists():
        return result

    with runs_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        if not {"length_km", "fidelity_true", "false_fraction"}.issubset(fieldnames):
            return result
        groups: dict[float, dict[str, list[float]]] = {}
        for row in reader:
            try:
                length_km = float((row.get("length_km") or "").strip())
                fidelity_true = float((row.get("fidelity_true") or "").strip())
                false_fraction = float((row.get("false_fraction") or "").strip())
            except ValueError:
                continue
            if not (np.isfinite(length_km) and np.isfinite(fidelity_true) and np.isfinite(false_fraction)):
                continue
            key = round(length_km, 9)
            group = groups.setdefault(key, {"fidelity": [], "false_fraction": []})
            group["fidelity"].append(float(fidelity_true))
            group["false_fraction"].append(float(false_fraction))

    for i, length_km in enumerate(lengths):
        key = round(float(length_km), 9)
        group = groups.get(key)
        if not group:
            continue
        f_vals = np.asarray(group["fidelity"], dtype=float)
        ff_vals = np.asarray(group["false_fraction"], dtype=float)
        result["counts"][i] = int(f_vals.size)
        if f_vals.size > 1:
            result["fidelity_sem"][i] = float(np.std(f_vals, ddof=1) / np.sqrt(f_vals.size))
        if ff_vals.size > 1:
            result["false_fraction_sem"][i] = float(np.std(ff_vals, ddof=1) / np.sqrt(ff_vals.size))
    return result


def _fit_true_component(length_km: np.ndarray, p_t: np.ndarray) -> tuple[np.ndarray, float]:
    mask = p_t > np.finfo(float).tiny
    if int(np.sum(mask)) < 2:
        return np.full_like(length_km, np.nan, dtype=float), float("nan")
    coeff = np.polyfit(length_km[mask], np.log10(p_t[mask]), deg=1)
    slope = float(coeff[0])
    pred = np.power(10.0, coeff[1] + slope * length_km)
    return pred, slope


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
    unc = _load_run_uncertainty(args.runs_csv, summary["length_km"])

    length_km = summary["length_km"]
    p_s = np.clip(summary["p_s"], np.finfo(float).tiny, None)
    p_t = np.clip(summary["p_t"], np.finfo(float).tiny, None)
    p_f = np.clip(summary["p_f"], np.finfo(float).tiny, None)
    f_t = np.clip(summary["f_t"], 0.0, 1.0)
    false_frac_pct = 100.0 * np.clip(summary["false_fraction"], 0.0, 1.0)

    ci95_f_t = 1.96 * np.asarray(unc["fidelity_sem"], dtype=float)
    ci95_ff_pct = 100.0 * 1.96 * np.asarray(unc["false_fraction_sem"], dtype=float)

    p_t_fit, slope = _fit_true_component(length_km, p_t)

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

    fig = plt.figure(figsize=(10.4, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.semilogy(length_km, p_s, color="#111827", lw=2.3, marker="o", ms=3.6, label=r"$p_s$（全部宣告）")
    ax0.semilogy(length_km, p_t, color="#1f77b4", lw=2.3, marker="s", ms=3.2, label=r"$p_t$（真成功）")
    ax0.semilogy(length_km, p_f, color="#d62728", lw=2.0, marker="^", ms=3.0, label=r"$p_f$（假成功）")
    if np.any(np.isfinite(p_t_fit)):
        fit_label = r"$p_t$ 的对数线性拟合"
        ax0.semilogy(length_km, p_t_fit, color="#0ea5e9", lw=1.5, ls="--", alpha=0.95, label=fit_label)
    ax0.axvline(33.0, lw=1.1, color="#9ca3af", ls=":")
    ax0.set_xlabel("总光纤长度 (km)")
    ax0.set_ylabel("每次尝试概率")
    ax0.set_title("宣告分量随距离变化")
    ax0.legend(frameon=False, fontsize=8.5, loc="upper right")
    _panel_label(ax0, "(a)")
    if np.isfinite(slope):
        ax0.text(
            0.02,
            0.04,
            rf"log$_{{10}} p_t \sim {slope:.4f}L$",
            transform=ax0.transAxes,
            fontsize=8.4,
            color="#0369a1",
        )

    ax1.plot(length_km, f_t, color="#059669", lw=2.4, marker="o", ms=3.6, label=r"$F_t$")
    ax1.fill_between(
        length_km,
        np.clip(f_t - ci95_f_t, 0.0, 1.0),
        np.clip(f_t + ci95_f_t, 0.0, 1.0),
        color="#10b981",
        alpha=0.16,
        linewidth=0.0,
    )
    ax1.set_xlabel("总光纤长度 (km)")
    ax1.set_ylabel(r"条件保真度 $F_t$", color="#047857")
    ax1.tick_params(axis="y", colors="#047857")
    ax1.set_ylim(max(0.0, float(np.min(f_t - ci95_f_t)) - 0.03), min(1.0, float(np.max(f_t + ci95_f_t)) + 0.03))

    ax1_r = ax1.twinx()
    ax1_r.plot(length_km, false_frac_pct, color="#b91c1c", lw=2.0, marker="D", ms=3.0, label="假成功占比")
    ax1_r.fill_between(
        length_km,
        np.clip(false_frac_pct - ci95_ff_pct, 0.0, 100.0),
        np.clip(false_frac_pct + ci95_ff_pct, 0.0, 100.0),
        color="#ef4444",
        alpha=0.15,
        linewidth=0.0,
    )
    ax1_r.set_ylabel("假成功占比 (%)", color="#991b1b")
    ax1_r.tick_params(axis="y", colors="#991b1b")
    ff_top = float(np.max(false_frac_pct + ci95_ff_pct))
    ax1_r.set_ylim(0.0, max(2.0, ff_top * 1.18))
    ax1.axvline(33.0, lw=1.1, color="#9ca3af", ls=":")

    lines = []
    labels = []
    for axis in (ax1, ax1_r):
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    ax1.legend(lines, labels, frameon=False, fontsize=8.5, loc="upper left")
    ax1.set_title("质量-可靠性权衡")
    _panel_label(ax1, "(b)")

    fig.suptitle(
        "长度扫描性能包络（汇总数据 + 运行级不确定度）",
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
