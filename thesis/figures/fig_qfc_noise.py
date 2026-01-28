import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    mu = np.linspace(0.0, 0.005, 200)
    f0 = 0.98
    alpha = 50.0
    fidelity = f0 - alpha * mu
    fidelity = np.maximum(fidelity, 0.5)

    plt.figure(figsize=(6.0, 3.6))
    plt.plot(mu, fidelity, color="tab:green")
    plt.xlabel("noise rate (arb.)")
    plt.ylabel("fidelity")
    plt.ylim(0.45, 1.0)
    plt.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
