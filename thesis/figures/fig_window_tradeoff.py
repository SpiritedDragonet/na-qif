import pathlib

from math import erf, sqrt

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    win_ns = np.linspace(20.0, 140.0, 200)
    sigma = 25.0
    erf_vec = np.vectorize(erf)
    accept = 0.5 * (1.0 + erf_vec(win_ns / (sqrt(2.0) * sigma)))
    accept = np.clip(accept, 0.0, 1.0)

    sbr = 8.0 / (1.0 + 0.03 * win_ns)
    fidelity = 0.95 - 0.15 * (1.0 - np.exp(-win_ns / 90.0))
    false_frac = 0.02 + 0.4 * (1.0 - np.exp(-win_ns / 80.0))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(win_ns, accept, color="tab:blue", label="acceptance")
    ax.plot(win_ns, fidelity, color="tab:green", label="fidelity")
    ax.plot(win_ns, false_frac, color="tab:red", label="false fraction")
    ax.set_xlabel("window width (ns)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0.0, 1.0)

    ax2 = ax.twinx()
    ax2.plot(win_ns, sbr, color="tab:purple", linestyle="--", label="SBR")
    ax2.set_ylabel("SBR (arb.)")

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, frameon=False, fontsize=8, loc="center right")

    fig.tight_layout()
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
