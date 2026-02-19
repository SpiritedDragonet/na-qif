import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    length_km = np.linspace(0.0, 40.0, 9)
    s_value = 2.0 + 0.6 * np.exp(-length_km / 22.0)

    plt.figure(figsize=(6.0, 3.6))
    plt.plot(length_km, s_value, color="tab:green", marker="o")
    plt.axhline(2.0, color="tab:red", linestyle="--", linewidth=1.0)
    plt.xlabel("fiber length (km)")
    plt.ylabel("CHSH S")
    plt.ylim(1.8, 2.7)
    plt.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
