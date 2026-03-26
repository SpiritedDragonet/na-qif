from __future__ import annotations

import matplotlib.axes
import matplotlib.figure


def frame_all_axes(fig: matplotlib.figure.Figure, *, color: str = "black", linewidth: float = 0.95) -> None:
    for ax in fig.axes:
        spines = getattr(ax, "spines", None)
        if not spines:
            continue
        for side in ("top", "right", "bottom", "left"):
            spine = spines.get(side)
            if spine is None:
                continue
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(linewidth)
