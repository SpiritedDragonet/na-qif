import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    t_ns = np.linspace(-150.0, 150.0, 600)
    sigma = 25.0
    signal = np.exp(-0.5 * (t_ns / sigma) ** 2)
    background = 0.08 * np.ones_like(t_ns)
    total = signal + background

    win_half = 35.0
    win_mask = (t_ns >= -win_half) & (t_ns <= win_half)

    plt.figure(figsize=(6.0, 3.6))
    plt.plot(t_ns, total, color="tab:blue", label="signal + background")
    plt.plot(t_ns, background, color="tab:gray", linestyle="--", label="background")
    plt.fill_between(
        t_ns,
        0.0,
        total,
        where=win_mask,
        color="tab:orange",
        alpha=0.2,
        label="acceptance window",
    )
    plt.xlabel("arrival time (ns)")
    plt.ylabel("counts (arb.)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
