import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    tau = np.linspace(-10.0, 10.0, 400)
    visibility = 0.9
    tau_c = 3.0
    coinc = 0.5 * (1.0 - visibility * np.exp(-(tau / tau_c) ** 2))

    plt.figure(figsize=(6.0, 3.6))
    plt.plot(tau, coinc, color="tab:red")
    plt.xlabel("delay (ns)")
    plt.ylabel("coincidence probability")
    plt.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
