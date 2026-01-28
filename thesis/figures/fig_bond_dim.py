import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    t = np.linspace(0.0, 1.0, 200)
    chi = 2.0 + 6.0 * (1.0 - np.exp(-t * 8.0))
    chi = np.minimum(chi + 1.0 * np.exp(-((t - 0.7) / 0.1) ** 2), 10.0)

    plt.figure(figsize=(6.0, 3.6))
    plt.plot(t, chi, color="tab:blue")
    plt.xlabel("normalized time")
    plt.ylabel("bond dimension")
    plt.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
