import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    t = np.linspace(0.0, 50.0, 400)
    sigma = 10.0
    t0_a = 25.0
    t0_b = 27.0
    wa = np.exp(-0.5 * ((t - t0_a) / sigma) ** 2)
    wb = np.exp(-0.5 * ((t - t0_b) / sigma) ** 2)

    plt.figure(figsize=(6.0, 3.6))
    plt.plot(t, wa, label="A")
    plt.plot(t, wb, label="B")
    plt.xlabel("t (ns)")
    plt.ylabel("normalized intensity")
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
