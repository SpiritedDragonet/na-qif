import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    fidelity = np.linspace(0.6, 0.95, 60)
    rate = 2.0 * np.exp(-6.0 * (fidelity - 0.6))
    rate = np.maximum(rate, 0.02)

    plt.figure(figsize=(6.0, 3.6))
    plt.plot(rate, fidelity, color="tab:blue")
    plt.scatter(rate[::8], fidelity[::8], color="tab:blue", s=18)
    plt.xlabel("event rate (arb.)")
    plt.ylabel("fidelity")
    plt.xscale("log")
    plt.ylim(0.55, 1.0)
    plt.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
