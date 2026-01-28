import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    eta = np.linspace(0.6, 0.95, 200)
    success = eta ** 2
    fidelity = 0.85 + 0.1 * (eta - 0.6) / 0.35
    fidelity = np.clip(fidelity, 0.8, 0.98)

    fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
    ax1.plot(eta, success, color="tab:purple")
    ax1.set_xlabel("detector efficiency")
    ax1.set_ylabel("success probability (arb.)")

    ax2 = ax1.twinx()
    ax2.plot(eta, fidelity, color="tab:green")
    ax2.set_ylabel("fidelity")
    ax2.set_ylim(0.75, 1.0)

    fig.tight_layout()
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
