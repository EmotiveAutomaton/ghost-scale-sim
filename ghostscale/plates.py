"""The house style for the walkthrough plates.

WHAT A PLATE IS FOR, WHICH IS NOT WHAT A RESEARCH FIGURE IS FOR. The charts in ``figures/`` are
research figures: they show a measurement so it can be checked. A plate shows a FINDING so it can
be understood in about two seconds by someone who has never heard of this project.

Three rules follow from that and they are enforced here rather than left to each script.

1.  THE TITLE STATES THE ANSWER, NOT THE VARIABLES. "A false label moves readers away from the
    truth" rather than "error reduction by provenance and label". If a reader has to consult the
    axes to learn what the point is, the plate has failed.

2.  ONE CONTRAST PER PLATE. Every plate is built around a single comparison, and anything that
    does not serve that comparison is removed. No legends where a direct label will do, no grid
    lines competing with the data, no second y-axis.

3.  THE NUMBER IS ON THE PLATE. Somebody who screenshots this and posts it should be carrying the
    evidence with them, not a picture of a trend.

Everything is drawn from committed verdict files. No plate contains a number typed in by hand.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[1]
PLATE_DIR = REPO / "figures" / "walkthrough"

# --------------------------------------------------------------------------- #
# Palette. Muted, warm-neutral ground with two strong accents, so the eye lands
# on the contrast rather than wandering the chart.
# --------------------------------------------------------------------------- #
INK = "#1b1b1f"          # titles, axis text
MUTED = "#6f6f78"        # subtitles, footers
PAPER = "#faf9f7"        # figure background
GRID = "#e4e2de"

HUMAN = "#2f6f6a"        # the human / intact / correct side of every contrast
MACHINE = "#b4553f"      # the machine / broken / misleading side
NEUTRAL = "#9a9aa2"      # context series that are not the point
HIGHLIGHT = "#c9a227"    # the one thing to look at
COOL = "#3c5a8a"         # a third series where one is genuinely needed

_FONT = "Segoe UI"
_available = {f.name for f in font_manager.fontManager.ttflist}
if _FONT not in _available:
    _FONT = "DejaVu Sans"


def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": _FONT,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 11,
        # Mathtext is only ever used for a single italicised word inside a title. Point it at the
        # house font so the italic does not arrive in a different typeface than its own sentence.
        "mathtext.fontset": "custom",
        "mathtext.rm": _FONT,
        "mathtext.it": f"{_FONT}:italic:bold",
        "mathtext.bf": f"{_FONT}:bold",
    })


def plate(title: str, subtitle: str, footer: str, size=(9.6, 5.6), authored: bool = False,
          second: str = ""):
    """A figure with the finding in the title and the provenance in the footer.

    The proportions are deliberate: 16:9-ish, because these are meant to survive being posted
    somewhere with a feed, and a tall figure gets cropped to nothing.

    THE LAYOUT IS COMPUTED, NOT FIXED, and the first version of this was not. With hard-coded
    y-positions a two-line title overprinted its own subtitle and a long subtitle ran off the
    right edge, which on a plate whose entire job is being read in two seconds is not a cosmetic
    problem. Title and subtitle are wrapped to the figure width and the axes start below whatever
    height they actually needed.
    """
    _apply_rc()

    # THE TITLE SETS ITS OWN SIZE. A three-sentence authored title wrapped to six lines at 19pt
    # and pushed the axes up underneath the subtitle, where the layout clamp let the plot overrun
    # its own text. Cutting the author's words to fit the typography is the wrong way round, so
    # the typography gives way instead: step down through wider, smaller settings and take the
    # first that fits in four lines. Short titles are untouched and still render at 19.
    for _w, _size in ((62, 19), (74, 16.5), (88, 14.5), (104, 13)):
        title_lines = _wrap(title, _w)
        if len(title_lines) <= 4:
            break
    title_size = _size
    title_step = 0.072 * (title_size / 19.0)
    # An empty subtitle means the plate has none, not that it has a blank one. Reserving a line
    # for it leaves a hole between the title and the plot that reads as a mistake.
    sub_lines = _wrap(subtitle, 104) if subtitle.strip() else []

    fig = plt.figure(figsize=size)
    top = 0.955

    # THE SCALE, APPLIED TO THE HEADLINE ITSELF.
    #
    # ``authored`` says a person wrote this title. Those are Polished tier: the words are the
    # author's, the grammar is not, and they run in black. Every other title on the set was
    # written by a machine from the author's results, which is Ghost tier, and those run in grey.
    #
    # A reader who never learns what the colours mean loses nothing. A reader who does gets the
    # project's own instrument applied to the project's own slides, which is the only honest place
    # to start using it.
    fig.text(0.055, top, "\n".join(title_lines), fontsize=title_size, fontweight="bold",
             color=INK if authored else MUTED, va="top", linespacing=1.22)

    sub_y = top - title_step * len(title_lines) - 0.018
    fig.text(0.055, sub_y, "\n".join(sub_lines), fontsize=12, color=MUTED,
             va="top", linespacing=1.32)

    fig.text(0.055, 0.028, footer, fontsize=8.5, color=MUTED, va="bottom")

    axes_top = sub_y - 0.050 * len(sub_lines) - 0.030

    # A SECOND BLOCK OF THE AUTHOR'S OWN WORDS, which is a different thing from a subtitle. It
    # runs black at the same left margin as the title with a blank row above it, and the axes
    # start below it, so the y-axis line cannot cut through it the way it did when this text was
    # drawn inside the plot.
    if second.strip():
        second_lines = _wrap(second, 74)
        fig.text(0.055, axes_top - 0.030, "\n".join(second_lines), fontsize=15,
                 fontweight="bold", color=INK, va="top", linespacing=1.28)
        axes_top -= 0.030 + 0.058 * len(second_lines) + 0.030

    bottom = 0.155
    ax = fig.add_axes([0.075, bottom, 0.86, max(axes_top - bottom, 0.30)])
    return fig, ax


def _wrap(text: str, width: int) -> list:
    """Wrap to a character width, honouring any newlines the caller put in deliberately."""
    import textwrap
    out = []
    for para in str(text).split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out


def on_bar(ax, x, y, text, fontsize=11.5, color=PAPER, ha="center", va="center",
           bar=None, pad=0.90, min_fontsize=7.5):
    """A note printed ON a bar, in the paper colour so it reads against the fill.

    The first version printed these in the bar's own colour, which is invisible.

    THE SECOND BUG WAS WORSE BECAUSE IT WAS SILENT. A note wider than its bar does not wrap and
    does not warn -- it renders straight through both edges and gets clipped, so the flagship
    plate read "ur times further wro / the truth takes you" for as long as nobody looked closely.
    Nothing in the pipeline can catch that: the figure is valid, the file is written, the caption
    is right, and only an eye on the image knows.

    So the text now MEASURES ITSELF against the bar it sits on. Pass ``bar`` (a matplotlib patch)
    and the fontsize shrinks until the rendered width fits inside it. If it will not fit even at
    ``min_fontsize`` the note is moved off the bar and placed beside it in the bar's own colour,
    which is the honest fallback: a legible note next to the bar beats an illegible one on it.
    """
    t = ax.text(x, y, text, color=color, ha=ha, va=va, fontsize=fontsize,
                fontweight="bold", linespacing=1.35, zorder=6)
    if bar is None:
        return t

    import textwrap

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    allowed = bar.get_window_extent(renderer=renderer).width * float(pad)

    def too_wide():
        return t.get_window_extent(renderer=renderer).width > allowed

    # WRAP FIRST, SHRINK ONLY IF WRAPPING RUNS OUT. Relocating the note was tried and was worse:
    # a narrow plot has nowhere to put it, so it ends up clipped in a new position and printed in
    # the bar's own colour over the bar, which is the original invisible-text bug returning by a
    # different door. Wrapping keeps it where it belongs and where the eye already is.
    flat = " ".join(str(text).split())
    width = max(len(line) for line in str(text).split("\n"))
    while too_wide() and width > 8:
        width = max(8, int(width * 0.85))
        t.set_text(textwrap.fill(flat, width))
        fig.canvas.draw()

    size = float(fontsize)
    while too_wide() and size > min_fontsize:
        size = max(min_fontsize, size * 0.92)
        t.set_fontsize(size)
        fig.canvas.draw()
    return t


def audit(fig, name: str = "") -> list:
    """Every problem this file has ever had, checked automatically instead of by eye.

    THE REASON THIS EXISTS. A plate whose text runs off the edge is not an error -- matplotlib
    renders it, the file writes, the caption is right, and the only thing that knows is an eye on
    the image. The flagship plate read "ur times further wro / the truth takes you" through an
    entire outreach plan because nothing in the pipeline could see it.

    Two checks, both for failures that actually happened here:

      CLIPPED   any text whose rendered box leaves the canvas.
      COLLIDING any two texts whose boxes overlap by enough to be unreadable. This is the bug that
                struck "the bar it had to clear" through the y-axis label, and the one that
                overprinted a value on its own history note.

    Returns a list of complaints. Empty means the plate meets the standard: readable at a glance,
    with nothing hidden behind anything else.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    problems = []

    # DEDUPE BY IDENTITY. ``findobj`` walks the tree and a legend's label Text is reachable both
    # through the legend and through the axes, so without this every legend entry collides with
    # itself and the audit cries wolf on plates that are fine.
    seen, texts = set(), []
    for t in fig.findobj(match=lambda o: hasattr(o, "get_text")):
        if id(t) in seen or not getattr(t, "get_text", None):
            continue
        if not str(t.get_text()).strip() or not t.get_visible():
            continue
        seen.add(id(t))
        texts.append(t)

    # Legend entries are excluded from the COLLISION check, though not from the clipping one.
    # Matplotlib lays a legend out itself and its entries never visually overlap; what they do is
    # appear more than once in the object tree, which made the audit report every legend as
    # colliding with itself. An audit that cries wolf gets switched off, so it must not.
    legend_texts = set()
    for ax_ in fig.axes:
        lg = ax_.get_legend()
        if lg is not None:
            legend_texts.update(id(x) for x in lg.findobj(match=lambda o: hasattr(o, "get_text")))

    boxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:                                # noqa: BLE001
            continue
        label = " ".join(str(t.get_text()).split())[:44]
        if bb.x0 < -1.0 or bb.y0 < -1.0 or bb.x1 > W + 1.0 or bb.y1 > H + 1.0:
            problems.append(f"CLIPPED    {name}: {label!r}")
        boxes.append((bb, label, id(t) in legend_texts))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, la, a_leg = boxes[i]
            b, lb, b_leg = boxes[j]
            # Skip only legend-against-legend. The first version skipped anything involving a
            # legend at all, which let a legend sit straight on top of an annotation without a
            # word of complaint -- the exact failure the audit exists to catch, hidden inside the
            # fix for a different one.
            if a_leg and b_leg:
                continue
            ov_w = min(a.x1, b.x1) - max(a.x0, b.x0)
            ov_h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ov_w <= 2 or ov_h <= 2:
                continue
            smaller = min(a.width * a.height, b.width * b.height) or 1.0
            if (ov_w * ov_h) / smaller > 0.25:
                problems.append(f"COLLIDING  {name}: {la!r} x {lb!r}")
    return problems


FRAME = (101, 104, 97)   # the same grey as the rendered panel of the Ghost Scale pair
FRAME_PX = 16            # thicker than strictly needed, so the plate has an edge on any background


def _frame(path: Path) -> None:
    """Put a grey border on a finished plate.

    Applied AFTER savefig rather than as a figure edge, for two reasons. The audit measures text
    against the canvas, so growing the canvas first would move every coordinate it checks. And a
    matplotlib figure edge is drawn INSIDE the canvas, which eats layout the plate has already
    budgeted. Compositing afterwards leaves the plate untouched and adds an edge around it.

    The tone is the grey from the Ghost Scale pair's rendered panel, deliberately: the whole figure
    set is machine-drawn, and the border says so in the project's own vocabulary.
    """
    from PIL import Image

    im = Image.open(path).convert("RGB")
    out = Image.new("RGB", (im.width + 2 * FRAME_PX, im.height + 2 * FRAME_PX), FRAME)
    out.paste(im, (FRAME_PX, FRAME_PX))
    out.save(path)


def save(fig, name: str) -> Path:
    PLATE_DIR.mkdir(parents=True, exist_ok=True)
    path = PLATE_DIR / f"{name}.png"
    for p in audit(fig, name):
        print(f"  !! {p}")
    fig.savefig(path, bbox_inches=None)
    _frame(path)
    plt.close(fig)
    return path


def bar_labels(ax, bars, values, fmt="{:+.2f}", offset=0.02, fontsize=13, color=None):
    """Put the number on the bar. Rule 3."""
    span = max(abs(v) for v in values) or 1.0
    for b, v in zip(bars, values):
        top = b.get_height()
        va = "bottom" if v >= 0 else "top"
        pad = offset * span * (1 if v >= 0 else -1)
        ax.text(b.get_x() + b.get_width() / 2, top + pad, fmt.format(v),
                ha="center", va=va, fontsize=fontsize, fontweight="bold",
                color=color or b.get_facecolor())


def zero_line(ax):
    ax.axhline(0, color=INK, lw=1.1, zorder=1)


def annotate(ax, x, y, text, color=INK, ha="left", va="bottom", fontsize=11, weight="normal",
             **kw):
    ax.text(x, y, text, color=color, ha=ha, va=va, fontsize=fontsize, fontweight=weight, **kw)


def clean_axis(ax, ylabel: str = "", xlabel: str = ""):
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.tick_params(length=0)
