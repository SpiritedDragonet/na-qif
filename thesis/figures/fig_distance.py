import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    distance = np.linspace(0.0, 100.0, 200)
    success = 1e-2 * 10 ** (-0.2 * distance / 10.0)
    fidelity = 0.95 - 0.0015 * distance
    fidelity = np.clip(fidelity, 0.6, 0.99)

    fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
    ax1.semilogy(distance, success, color="tab:blue")
    ax1.set_xlabel("distance (km)")
    ax1.set_ylabel("success probability")

    ax2 = ax1.twinx()
    ax2.plot(distance, fidelity, color="tab:orange")
    ax2.set_ylabel("fidelity")
    ax2.set_ylim(0.6, 1.0)

    fig.tight_layout()
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
