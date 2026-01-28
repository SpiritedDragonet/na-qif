import pathlib

import matplotlib.pyplot as plt


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    x_positions = [0, 1.2, 2.4, 3.6, 4.8]
    y = 0.0

    # Draw sites
    for x in x_positions:
        ax.scatter([x], [y], s=180, facecolors="white", edgecolors="black", zorder=3)

    # Odd bonds in red, even bonds in blue
    for i in range(len(x_positions) - 1):
        color = "tab:red" if i % 2 == 0 else "tab:blue"
        ax.plot([x_positions[i], x_positions[i + 1]], [y, y], color=color, lw=2.5)

    ax.text(1.2, 0.5, "odd", color="tab:red", fontsize=9, ha="center")
    ax.text(3.0, 0.5, "even", color="tab:blue", fontsize=9, ha="center")

    ax.set_xlim(-0.3, 5.1)
    ax.set_ylim(-0.6, 0.9)
    ax.axis("off")
    fig.tight_layout()

    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
