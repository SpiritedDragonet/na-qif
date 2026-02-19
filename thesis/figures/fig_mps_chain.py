import pathlib

import matplotlib.pyplot as plt


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    x_positions = [0, 1.5, 3.0, 4.5]
    y = 0.0

    # Draw bonds
    for i in range(len(x_positions) - 1):
        ax.plot([x_positions[i], x_positions[i + 1]], [y, y], color="black", lw=1.5)

    # Draw sites and physical legs
    for i, x in enumerate(x_positions, start=1):
        ax.scatter([x], [y], s=220, facecolors="white", edgecolors="black", zorder=3)
        ax.plot([x, x], [y, y + 0.8], color="black", lw=1.2)
        ax.text(x, y - 0.35, f"A{i}", ha="center", va="top", fontsize=9)

    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-0.8, 1.2)
    ax.axis("off")
    fig.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
