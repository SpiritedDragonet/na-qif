import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _simulate_working_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    win = np.linspace(35.0, 130.0, 28)
    qbg = np.linspace(0.7, 1.7, 24)
    ww, bb = np.meshgrid(win, qbg, indexing="ij")

    p_true = 1.25e-6 * (1.0 - np.exp(-(ww - 18.0) / 36.0)) * np.exp(-0.24 * (bb - 1.0))
    p_false = (0.16e-6 + 0.78e-6 * (1.0 - np.exp(-(ww - 15.0) / 62.0))) * bb
    p_all = np.clip(p_true + p_false, 1e-10, None)
    fidelity = np.clip(
        0.965 - 0.15 * (1.0 - np.exp(-(ww - 20.0) / 55.0)) - 0.11 * (bb - 1.0),
        0.50,
        0.98,
    )
    false_frac = np.clip(p_false / p_all, 0.0, 0.95)
    return p_all.ravel(), fidelity.ravel(), false_frac.ravel(), ww.ravel()


def _pareto_front(rate: np.ndarray, fidelity: np.ndarray) -> np.ndarray:
    order = np.argsort(rate)[::-1]
    keep = []
    f_best = -np.inf
    for idx in order:
        fi = fidelity[idx]
        if fi > f_best:
            keep.append(idx)
            f_best = fi
    keep = np.array(keep, dtype=int)
    keep = keep[np.argsort(rate[keep])]
    return keep


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
        }
    )

    rate, fidelity, false_frac, win = _simulate_working_points()
    idx_front = _pareto_front(rate, fidelity)

    fig = plt.figure(figsize=(9.0, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    sc0 = ax0.scatter(
        1e6 * rate,
        fidelity,
        c=100.0 * false_frac,
        cmap="magma_r",
        s=28,
        alpha=0.85,
        edgecolors="none",
    )
    ax0.plot(1e6 * rate[idx_front], fidelity[idx_front], color="#1f1f1f", lw=2.0, label="Pareto front")
    ax0.set_xscale("log")
    ax0.set_xlabel(r"Herald probability ($\times 10^{-6}$)")
    ax0.set_ylabel(r"Conditional fidelity $F_t$")
    ax0.set_ylim(0.5, 0.98)
    ax0.set_title("Rate-fidelity operating cloud")
    ax0.legend(frameon=False, fontsize=8.8, loc="lower left")
    cbar0 = fig.colorbar(sc0, ax=ax0, fraction=0.048, pad=0.03)
    cbar0.set_label("False fraction (%)")

    sc1 = ax1.scatter(
        1e6 * rate,
        fidelity,
        c=win,
        cmap="viridis",
        s=28,
        alpha=0.85,
        edgecolors="none",
    )
    ax1.plot(1e6 * rate[idx_front], fidelity[idx_front], color="#0f172a", lw=2.0, ls="--")
    ax1.axhline(0.8, color="#d62728", lw=1.4, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Herald probability ($\times 10^{-6}$)")
    ax1.set_ylabel(r"Conditional fidelity $F_t$")
    ax1.set_ylim(0.5, 0.98)
    ax1.set_title("Window-coded operating points")
    cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.048, pad=0.03)
    cbar1.set_label("Window (ns)")

    fig.suptitle("Rate-Fidelity frontier with explicit false-herald burden", fontsize=12.3, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
