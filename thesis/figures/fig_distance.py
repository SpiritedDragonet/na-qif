import pathlib

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )

    dist = np.linspace(0.0, 120.0, 241)
    alpha_db_km = 0.2
    trans = 10 ** (-alpha_db_km * dist / 10.0)

    p_true = 1.45e-6 * trans
    p_false = 0.20e-6 + 0.35e-6 * (1.0 - np.exp(-dist / 70.0))
    p_all = p_true + p_false

    f_base = 0.936 - 0.00245 * dist
    f_base -= 0.055 * (1.0 - np.exp(-dist / 40.0))
    f_base = np.clip(f_base, 0.52, 0.95)
    chsh = 2.0 + 1.85 * np.maximum(f_base - 0.5, 0.0)
    chsh = np.clip(chsh, 1.95, 2.45)

    fig = plt.figure(figsize=(9.3, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.semilogy(dist, 1e6 * p_all, lw=2.2, color="#1f1f1f", label=r"$p_s$")
    ax0.semilogy(dist, 1e6 * p_true, lw=2.2, color="#1f77b4", label=r"$p_t$")
    ax0.semilogy(dist, 1e6 * p_false, lw=2.0, color="#d62728", ls="--", label=r"$p_f$")
    ax0.axvline(33.0, lw=1.4, color="#f2c14e", ls="--")
    ax0.set_xlabel("Total fiber length (km)")
    ax0.set_ylabel(r"Probability per attempt ($\times10^{-6}$)")
    ax0.set_title("Distance scaling of herald components")
    ax0.legend(frameon=False, fontsize=8.8, loc="upper right")

    ax1.plot(dist, f_base, lw=2.3, color="#2a9d8f", label=r"$F_t$")
    ax1.fill_between(dist, np.maximum(f_base - 0.03, 0.0), np.minimum(f_base + 0.03, 1.0), color="#2a9d8f", alpha=0.16)
    ax1_t = ax1.twinx()
    ax1_t.plot(dist, chsh, lw=2.1, color="#9467bd", label=r"$S$")
    ax1_t.axhline(2.0, lw=1.4, color="#8b0000", ls=":")

    ax1.set_xlabel("Total fiber length (km)")
    ax1.set_ylabel(r"Conditional fidelity $F_t$")
    ax1_t.set_ylabel("CHSH S")
    ax1.set_ylim(0.5, 0.96)
    ax1_t.set_ylim(1.9, 2.5)
    ax1.set_title("Entanglement quality vs distance")

    lines = [l for l in (ax1.get_lines() + ax1_t.get_lines()) if not l.get_label().startswith("_")]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=8.8, loc="upper right")

    ax1.annotate(
        "33 km benchmark",
        xy=(33.0, float(np.interp(33.0, dist, f_base))),
        xytext=(52.0, 0.84),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444444"},
        fontsize=8.6,
    )

    fig.suptitle("End-to-end scaling over metropolitan fiber links", fontsize=12.4, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
