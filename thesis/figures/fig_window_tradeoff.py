import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _model_curves(window_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_true = 1.25e-6 * (1.0 - np.exp(-(window_ns - 15.0) / 38.0))
    s_true = np.clip(s_true, 0.0, None)
    s_false = 0.12e-6 + 0.95e-6 * (1.0 - np.exp(-(window_ns - 15.0) / 65.0))
    s_false = np.clip(s_false, 0.0, None)
    s_all = s_true + s_false

    f_cond = 0.947 - 0.165 * (1.0 - np.exp(-(window_ns - 12.0) / 60.0))
    f_cond = np.clip(f_cond, 0.5, 0.98)
    return s_true, s_false, s_all, f_cond


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
        }
    )

    w = np.linspace(20.0, 140.0, 260)
    s_true, s_false, s_all, f_cond = _model_curves(w)
    false_frac = np.clip(s_false / np.maximum(s_all, 1e-15), 0.0, 0.98)
    p_t11 = np.clip(0.965 - 0.18 * (w - 20.0) / 120.0, 0.72, 0.99)
    knee = 70.0

    fig = plt.figure(figsize=(9.6, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    c_true = "#1f77b4"
    c_false = "#c44e52"
    c_all = "#2f2f2f"

    ax0.plot(w, 1e6 * s_all, color=c_all, lw=2.2, label=r"$p_s$")
    ax0.plot(w, 1e6 * s_true, color=c_true, lw=2.2, label=r"$p_t$")
    ax0.plot(w, 1e6 * s_false, color=c_false, lw=2.0, ls="--", label=r"$p_f$")
    ax0.fill_between(w, 1e6 * s_true, 1e6 * s_all, color=c_false, alpha=0.12)
    ax0.axvline(knee, color="#f2c14e", lw=1.6, ls="--")
    ax0.set_xlabel("Acceptance window (ns)")
    ax0.set_ylabel(r"Probability per attempt ($\times10^{-6}$)")
    ax0.set_title("Herald-rate decomposition")
    ax0.legend(frameon=False, fontsize=9, loc="upper left")

    ax1.plot(w, f_cond, color="#2a9d8f", lw=2.3, label=r"$F_t$")
    ax1.plot(w, false_frac, color="#d62728", lw=2.1, label=r"$p_f/p_s$")
    ax1.plot(w, p_t11, color="#9467bd", lw=2.1, ls="-.", label=r"$p_{t|11}$")
    ax1.axvline(knee, color="#f2c14e", lw=1.6, ls="--", label="70 ns")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Acceptance window (ns)")
    ax1.set_ylabel("Conditional metric")
    ax1.set_title("Quality metrics")
    ax1.legend(frameon=False, fontsize=8.8, loc="center right")

    ax1.annotate(
        "recommended working point",
        xy=(knee, float(np.interp(knee, w, f_cond))),
        xytext=(84.0, 0.88),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
        fontsize=8.8,
    )

    fig.suptitle("Window scan: rate-quality tradeoff under true/false decomposition", fontsize=12.3, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
