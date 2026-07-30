#!/usr/bin/env python
"""``figures/social/e2_social.png`` — the E2 machine-made pair, restyled for social.

    python scripts/make_e2_social.py

SAME DATA, SAME DOT POSITIONS as ``figures/e2_variance.png``. The jitter draw is reproduced
exactly (``default_rng(0)`` re-seeded per panel, as in ``figures.fig_e2``), so this is a restyle
and not a re-run. ``e2_variance.png`` is never written by this script.

WHAT THE STYLING IS DOING. The research figure is a 2x2 with four colours. Here only the bottom
row survives: both panels are machine-made content, and the only thing that differs between them
is the word the reader was told. That is the claim, so it is the whole picture.

The palette is the Ghost Scale's own opacities, and it is applied to the LABEL the reader was
given, not to the truth:

  * "Labelled correctly"       -> dots at 60% black (Curator tier). Honest machine content.
  * "Falsely labelled human"   -> dots at 100% black (Creator tier). What the reader was told.

So the deceived panel is the loudest thing on the page. That is the lie, drawn at the volume it
was told at.

The four goals are carried by MARKER SHAPE, not colour. Which purpose each reader settled on is
the finding; it has to survive greyscale, feed compression and colour-blind viewing, and four
greys inside a two-grey palette would not. Shape does.

Text is 100% black (headline) or 60% grey (everything else). 60% is exactly the 4.5:1 floor on
white, so the lightest text on the figure is simultaneously the accessible minimum and the
Curator tier.

The two-tone black/white border is the framework's own image marking for machine-mediated visual
content, applied to a figure that is itself machine-made.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "social" / "e2_social.png"

# 1080 x 1350 at 100 dpi, i.e. a 10.8 x 13.5 inch canvas. Every font size below is quoted in
# points on that canvas and none is under 24.
DPI = 100
FIGSIZE = (1080 / DPI, 1350 / DPI)

BLACK = "#000000"          # Creator tier, 100%
GREY60 = (0.4, 0.4, 0.4)   # Curator tier, 60% black == 4.5:1 on white
BORDER_OUTER_PX = 8
BORDER_INNER_PX = 4

# One shape per goal. Deliberately four shapes that stay distinct when a feed resamples them.
GOAL_MARKERS = ["o", "s", "^", "X"]

HEADLINE = "Same images.\nOne word changed."

# (true_provenance, declared_signal, panel label, dot tier, how the within-observer number reads)
#
# Both panels are machine-made. Only the declared signal differs, which is the point.
# The plain line under each label carries the two numbers verbatim from e2_cell_stats.csv. It is
# set as two short lines rather than one: at the 24pt floor, a half-width panel holds about 25
# characters, and "each reader certain: 0.09   readers disagree: 1.38" is 51.
PANELS = [
    ("GHOST", "SIG_GHOST", "Labelled\ncorrectly", GREY60, "each reader unsure"),
    ("GHOST", "SIG_CREATOR", "Falsely labelled\nhuman", BLACK, "each reader certain"),
]

# A thin white stroke under the grey text was tried and dropped: at 24-30pt in a compressed feed
# it reads as halation around the glyphs rather than as weight. Flat 60% is cleaner.


def _load():
    points = pd.read_csv(RESULTS / "e2_points.csv")
    stats = pd.read_csv(RESULTS / "e2_cell_stats.csv")
    cell_stats = {(r.true_provenance, r.declared_signal):
                  {"within": float(r.within), "between": float(r.between)}
                  for r in stats.itertuples()}
    return points[points.seed_rep == 0], cell_stats


def build(points_df, cell_stats, path: Path) -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 24,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "text.color": GREY60,
        "axes.labelcolor": GREY60,
        "xtick.color": GREY60,
        "ytick.color": GREY60,
        "axes.edgecolor": GREY60,
    })
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    axes = [fig.add_axes(rect) for rect in ((0.075, 0.285, 0.405, 0.295),
                                            (0.545, 0.285, 0.405, 0.295))]

    fig.text(0.065, 0.965, HEADLINE, ha="left", va="top", color=BLACK,
             fontsize=60, fontweight="bold", linespacing=1.05)

    for ax, (tier, sig, label, dot_color, sure_word) in zip(axes, PANELS):
        d = points_df[(points_df.true_provenance == tier) & (points_df.declared_signal == sig)]
        # Identical jitter draw to figures.fig_e2: fresh default_rng(0), one uniform per point.
        rng = np.random.default_rng(0)
        yj = rng.uniform(0, 1, size=len(d))
        for g, marker in enumerate(GOAL_MARKERS):
            m = (d.modal_goal.to_numpy() == g)
            if not m.any():
                continue
            ax.scatter(d.within_entropy.to_numpy()[m], yj[m], marker=marker,
                       c=[dot_color], s=90, linewidths=0, alpha=1.0)

        st = cell_stats[(tier, sig)]
        ax.set(xlim=(-0.05, np.log(4) + 0.05), ylim=(-0.06, 1.06), yticks=[],
               xticks=[0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1.0"], fontsize=26)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.6)

        # Label and numbers sit above the axes, left-aligned to the cloud they describe.
        x0 = ax.get_position().x0
        fig.text(x0, 0.700, label, ha="left", va="bottom", color=GREY60,
                 fontsize=34, fontweight="bold", linespacing=1.05)
        fig.text(x0, 0.606, f"{sure_word}: {st['within']:.2f}\n"
                            f"readers disagree: {st['between']:.2f}",
                 ha="left", va="bottom", color=GREY60, fontsize=26, linespacing=1.25)

    # One shared x-axis label under both panels: the scale is the same in each, and repeating it
    # would force it below the 24pt floor to fit a half-width panel.
    fig.text(0.5, 0.215, "how unsure each reader is (nats)", ha="center", va="bottom",
             color=GREY60, fontsize=28)
    fig.text(0.5, 0.178, "certain ←                                        → no idea",
             ha="center", va="bottom", color=GREY60, fontsize=26)

    handles = [Line2D([0], [0], marker=mk, ls="", color=GREY60, markersize=15,
                      markeredgewidth=0, label=f"purpose {g + 1}")
               for g, mk in enumerate(GOAL_MARKERS)]
    # Two rows, not four columns: four "purpose N" entries at the 24pt floor overrun 1080px.
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.065, 0.022),
               ncol=2, fontsize=24, handletextpad=0.4, columnspacing=2.0,
               labelcolor=GREY60, frameon=False,
               title="the purpose each reader settled on", title_fontsize=24,
               alignment="left")

    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    _stamp_border(path)


def _stamp_border(path: Path) -> None:
    """Two-tone image marking: outer black band, inner white band, drawn in exact pixels.

    Done in PIL rather than in matplotlib because the spec is in pixels and a line width in
    points would land on fractional pixels and antialias into grey."""
    img = Image.open(path).convert("RGB")
    a = np.asarray(img).copy()
    b, w = BORDER_OUTER_PX, BORDER_INNER_PX
    a[:b, :] = a[-b:, :] = a[:, :b] = a[:, -b:] = 0
    a[b:b + w, b:-b] = 255
    a[-(b + w):-b, b:-b] = 255
    a[b:-b, b:b + w] = 255
    a[b:-b, -(b + w):-b] = 255
    Image.fromarray(a).save(path)


def main() -> None:
    points, cell_stats = _load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build(points, cell_stats, OUT)
    print(f"wrote {OUT}  ({Image.open(OUT).size[0]}x{Image.open(OUT).size[1]})")


if __name__ == "__main__":
    main()
