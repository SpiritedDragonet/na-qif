import pathlib

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    labels = ["H1V2", "V1H2", "H1V1", "H2V2", "H1H2", "V1V2"]
    true_counts = np.array([0.24, 0.23, 0.22, 0.21, 0.05, 0.05])
    dark_counts = np.array([0.04, 0.03, 0.05, 0.05, 0.04, 0.04])

    x = np.arange(len(labels))
    width = 0.7

    plt.figure(figsize=(6.0, 3.6))
    plt.bar(x, true_counts, width, label="true", color="tab:blue")
    plt.bar(x, dark_counts, width, bottom=true_counts, label="dark", color="tab:orange")
    plt.xticks(x, labels)
    plt.ylabel("probability (arb.)")
    plt.xlabel("click pattern")
    plt.ylim(0.0, 0.4)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
