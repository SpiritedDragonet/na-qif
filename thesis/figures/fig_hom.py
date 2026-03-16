import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

EXPORT_PNG = False


def _default_summary_csv() -> pathlib.Path:
    data_root = pathlib.Path(__file__).resolve().parents[1] / "data"
    candidates = sorted(
        data_root.glob("hom_summary_output_*/summary/hom_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    return data_root / "hom_summary" / "hom_summary.csv"


def _load_g2_from_double_clicks(summary_csv: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"HOM summary CSV not found: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    tau_vals: list[float] = []
    p2_vals: list[float] = []
    sem_vals: list[float] = []

    # Preferred schema: hom_summary.csv aggregated by tau.
    if {"tau_ns", "coinc_counts", "shots_total"}.issubset(fieldnames):
        for row in rows:
            try:
                tau = float((row.get("tau_ns") or "").strip())
                coinc_counts = float((row.get("coinc_counts") or "").strip())
                shots_total = float((row.get("shots_total") or "").strip())
            except ValueError:
                continue
            if not np.isfinite(tau) or not np.isfinite(coinc_counts) or shots_total <= 0.0:
                continue
            p2 = coinc_counts / shots_total
            p2 = float(np.clip(p2, 0.0, 1.0))
            sem = float(np.sqrt(max(p2 * (1.0 - p2), 0.0) / shots_total))
            tau_vals.append(tau)
            p2_vals.append(p2)
            sem_vals.append(sem)
    # Fallback schema: per-shot tau/coinc rows.
    elif {"tau", "coinc"}.issubset(fieldnames):
        by_tau: dict[float, list[float]] = {}
        for row in rows:
            try:
                tau = float((row.get("tau") or "").strip())
                coinc = float((row.get("coinc") or "").strip())
                valid_raw = row.get("valid")
                valid = 1.0 if valid_raw is None or str(valid_raw).strip() == "" else float(str(valid_raw).strip())
            except ValueError:
                continue
            if valid <= 0.0 or not np.isfinite(tau) or not np.isfinite(coinc):
                continue
            by_tau.setdefault(tau, []).append(float(np.clip(coinc, 0.0, 1.0)))
        for tau in sorted(by_tau):
            values = np.asarray(by_tau[tau], dtype=float)
            p2 = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            tau_vals.append(float(tau))
            p2_vals.append(p2)
            sem_vals.append(sem)
    else:
        raise ValueError(f"Unsupported HOM summary format: {summary_csv}")

    if not tau_vals:
        raise ValueError(f"No valid HOM points in: {summary_csv}")

    tau = np.asarray(tau_vals, dtype=float)
    p2 = np.asarray(p2_vals, dtype=float)
    sem = np.asarray(sem_vals, dtype=float)
    order = np.argsort(tau)
    return tau[order], p2[order], sem[order]


def _normalize_to_far_delay(tau: np.ndarray, p2: np.ndarray, sem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    abs_tau = np.abs(tau)
    edge_cut = float(np.quantile(abs_tau, 0.85))
    far_mask = abs_tau >= edge_cut
    if not np.any(far_mask):
        return p2, sem
    ref = float(np.mean(p2[far_mask]))
    if ref <= 0.0:
        return p2, sem
    return p2 / ref, sem / ref


def _fit_hom_curve_least_squares(tau: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Model: y = b - a * exp(-((tau - tau0) / sigma)^2)
    tau0_grid = np.linspace(-10.0, 10.0, 161)
    sigma_grid = np.linspace(0.5, 40.0, 260)

    best_sse = np.inf
    best_params = None
    for tau0 in tau0_grid:
        shifted = tau - tau0
        for sigma in sigma_grid:
            basis = np.exp(-((shifted / sigma) ** 2))
            design = np.column_stack([np.ones_like(basis), -basis])
            coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
            b, a = float(coeff[0]), float(coeff[1])
            pred = b - a * basis
            sse = float(np.sum((y - pred) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_params = (tau0, sigma, b, a)

    if best_params is None:
        raise RuntimeError("Least-squares fit failed")

    tau0, sigma, b, a = best_params
    tau_dense = np.linspace(float(np.min(tau)), float(np.max(tau)), 800)
    y_dense = b - a * np.exp(-((tau_dense - tau0) / sigma) ** 2)
    return tau_dense, y_dense


def main() -> None:
    summary_csv = _default_summary_csv()
    tau, p2, p2_sem = _load_g2_from_double_clicks(summary_csv)
    g2, g2_sem = _normalize_to_far_delay(tau, p2, p2_sem)
    tau_dense, g2_fit = _fit_hom_curve_least_squares(tau, g2)

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
            "grid.alpha": 0.18,
        }
    )

    plt.figure(figsize=(6.2, 3.9))
    if float(np.max(g2_sem)) > 0.0:
        plt.errorbar(
            tau,
            g2,
            yerr=g2_sem,
            fmt="o",
            ms=4.0,
            color="#1f77b4",
            ecolor="#6baed6",
            elinewidth=0.8,
            capsize=1.8,
            label="双点击统计",
        )
    else:
        plt.scatter(tau, g2, s=18, color="#1f77b4", label="双点击统计")
    plt.plot(tau_dense, g2_fit, color="#d62728", lw=2.0, label="最小二乘拟合")

    y_all = np.concatenate([g2, g2_fit])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    y_pad = max(0.04 * (y_max - y_min), 0.04)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.xlabel(r"相对时延 $\tau$ (ns)")
    plt.ylabel(r"二阶关联 $g^{(2)}_{HH,VV}(\tau)$")
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()

    out_base = pathlib.Path(__file__).with_suffix("")
    plt.savefig(out_base.with_suffix(".pdf"), dpi=220)
    if EXPORT_PNG:
        plt.savefig(out_base.with_suffix(".png"), dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
