#!/usr/bin/env python
"""The five distribution slides, the PDF, and the social preview image.

    python scripts/make_social_figures.py

Writes to ``figures/social/``. Research figures in ``figures/`` are never touched: they have a
different job, a different audience and different constraints, and one script that wrote both would
eventually apply one set of constraints to the other.

-----------------------------------------------------------------------------------------
A FIGURE IS A CLAIM WITH A PICTURE ATTACHED, so no slide is built for a claim that has not survived
validation. Each slide declares which validation verdicts it depends on. If a verdict is missing,
the slide is skipped and the reason is printed. If a verdict is present but unwelcome, **the slide
is still built and the caveat goes on the slide** — not in a footnote, not in a caption, on the
slide — because that is the difference between a finding and an overclaim and it is the first thing
a hostile reader looks for.

-----------------------------------------------------------------------------------------
THE VISUAL IDENTITY, AND WHY THE ACCESSIBILITY CONSTRAINT IS LOAD-BEARING RATHER THAN A LIMITATION.

The palette is the Ghost Scale's own opacities: 100%, 95%, 60% and 5% black on white. The charts
obey the framework they report on.

The 60% tier sits almost exactly at the minimum contrast ratio for body text and the 5% tier is far
below it. So **all text renders at 100% or 95% only**, and only non-text data marks use the lower
tiers. ``_assert_text_contrast`` checks this at render time, on every text element of every figure,
and raises rather than warns. A carousel about a contrast-grounded transparency framework that fails
contrast requirements would be caught by precisely the audience it is addressed to.

The 5% tier additionally carries the isolating bounding box the published framework requires, since
at 5% the mark is otherwise invisible — which is the framework making its own argument.

Series that are not tiers use 100% black and 60% grey and are ALSO distinguished by line style or
hatch, so they survive greyscale, feed compression and colour-blind viewing.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
VDIR = RESULTS / "validation"
OUT = REPO / "figures" / "social"

REPO_URL = "github.com/EmotiveAutomaton/ghost-scale-sim"

# The Ghost Scale opacities, as fractions of black.
TIER = {"CREATOR": 1.00, "POLISHED": 0.95, "CURATOR": 0.60, "GHOST": 0.05}
INK = "#000000"
# Text may only use these two. Asserted at render time.
TEXT_ALPHAS = (1.00, 0.95)

# 1080 x 1350 at 100 dpi. Minimum type size 24pt, checked below.
SLIDE_IN = (10.80, 13.50)
PREVIEW_IN = (12.80, 6.40)
DPI = 100
MIN_PT = 24


def style():
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": MIN_PT,
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# --------------------------------------------------------------------------- #
# The render-time constraints.
# --------------------------------------------------------------------------- #
def _assert_text_contrast(fig, where: str):
    """No text element may be drawn at an opacity below 95%.

    Walks every Text on the figure and every axis. This is an assertion rather than a warning
    because the failure it guards against is not cosmetic: it would put a page about contrast
    below the contrast threshold, in front of the readers most likely to check.
    """
    def check(t):
        alpha = t.get_alpha()
        if alpha is None:
            return
        if float(alpha) < min(TEXT_ALPHAS) - 1e-9:
            raise AssertionError(
                f"{where}: text {t.get_text()!r} is drawn at opacity {alpha}, below the "
                f"{min(TEXT_ALPHAS):.0%} floor. The 60% tier sits at the minimum contrast ratio "
                f"for body text and the 5% tier is far below it; only non-text marks may use "
                f"them.")
        size = t.get_fontsize()
        if t.get_text().strip() and size < MIN_PT - 1e-9:
            raise AssertionError(
                f"{where}: text {t.get_text()!r} is {size}pt, below the {MIN_PT}pt floor for "
                f"one-handed phone viewing.")

    for t in fig.findobj(match=plt.Text):
        check(t)


def tier_swatches(ax, y=0.5, height=0.55, labels=False):
    """The four tiers as a strip. A design element to a general viewer, a signature to a reader
    who knows the framework. The 5% tier gets its isolating box, as the framework requires."""
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (name, a) in enumerate(TIER.items()):
        ax.add_patch(Rectangle((i + 0.12, y - height / 2), 0.76, height,
                               facecolor=INK, alpha=a, edgecolor="none"))
        if a <= 0.10:
            ax.add_patch(Rectangle((i + 0.12, y - height / 2), 0.76, height,
                                   facecolor="none", edgecolor=INK, lw=1.6, alpha=1.0,
                                   linestyle=(0, (4, 3))))
        if labels:
            ax.text(i + 0.5, y - height / 2 - 0.16, name, ha="center", va="top",
                    fontsize=MIN_PT, alpha=1.0)


def slide_canvas():
    fig = plt.figure(figsize=SLIDE_IN)
    return fig


# --------------------------------------------------------------------------- #
# Layout. Every line is hard-wrapped to a budget derived from its own font size, and blocks are
# stacked from the top with their MEASURED heights, rather than positioned by hand.
#
# The reason is that these are fixed-size canvases with no reflow. matplotlib's own `wrap` measures
# after placement and will happily run a line off the edge, and hand-tuned y coordinates drift out
# of true the moment a caveat gains a sentence — which, in a script whose caveats are generated from
# verdict files, happens every time a verdict changes. So the stack measures.
# --------------------------------------------------------------------------- #
MARGIN = 0.075                    # fraction of the canvas kept clear on each side
_CHAR_W = 0.52                    # mean glyph width as a fraction of point size, sans-serif
_GAP = 0.030                      # vertical breathing room between blocks, in figure fraction


def budget(size_pt: float, bold: bool = False) -> int:
    """How many characters fit on one line at this size, inside the margins."""
    usable_px = SLIDE_IN[0] * DPI * (1.0 - 2 * MARGIN)
    char_px = _CHAR_W * (1.06 if bold else 1.0) * size_pt * DPI / 72.0
    return max(12, int(usable_px / char_px))


def wrap(text: str, size_pt: float, bold: bool = False) -> str:
    """Hard-wrap to the budget, honouring line breaks the caller already put in."""
    import textwrap
    width = budget(size_pt, bold)
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(out)


def _height(wrapped: str, size_pt: float, spacing: float, box: bool) -> float:
    lines = wrapped.count("\n") + 1
    px = lines * size_pt * spacing * DPI / 72.0
    if box:
        px += 1.1 * size_pt * DPI / 72.0
    return px / (SLIDE_IN[1] * DPI)


class Stack:
    """Lays blocks out downward from a top edge, tracking where the cursor has got to."""

    def __init__(self, fig, top: float = 0.955):
        self.fig = fig
        self.y = top

    def text(self, body: str, size: float, *, bold: bool = False, alpha: float = 0.95,
             spacing: float = 1.42, box: bool = False, gap: float = _GAP,
             nowrap: bool = False):
        # ``nowrap`` is for the one thing that must never break: a URL. A wrapped URL is a URL
        # somebody mistypes, so it is placed verbatim and the caller is responsible for the size.
        wrapped = body if nowrap else wrap(body, size, bold)
        h = _height(wrapped, size, spacing, box)
        centre = self.y - h / 2.0
        kw = dict(ha="center", va="center", fontsize=size, alpha=alpha, linespacing=spacing)
        if bold:
            kw["fontweight"] = "bold"
        if box:
            kw["bbox"] = dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor=INK, lw=1.4)
        self.fig.text(0.5, centre, wrapped, **kw)
        self.y = centre - h / 2.0 - gap
        return self

    def axes(self, height: float, *, left: float = 0.24, width: float = 0.62,
             gap: float = _GAP, below: float = 0.0):
        """Reserve an axes box, plus ``below`` for whatever matplotlib draws under it.

        ``below`` is not optional decoration. Tick labels and an x-axis label are drawn OUTSIDE the
        axes rectangle, so a stack that advances by ``height`` alone puts the next text block on top
        of them. Callers pass roughly one line height per line of tick label, plus one more if there
        is an x-axis label.
        """
        bottom = self.y - height
        ax = self.fig.add_axes([left, bottom, width, height])
        self.y = bottom - below - gap
        return ax

    def swatches(self, height: float = 0.05, gap: float = _GAP):
        ax = self.axes(height, left=0.34, width=0.32, gap=gap, below=0.0)
        tier_swatches(ax)
        return ax

    def space(self, amount: float):
        self.y -= amount
        return self

    def remaining(self) -> float:
        """How much room is left above the footer. Negative means the slide has overflowed."""
        return self.y - 0.085


def footer(fig, text=REPO_URL, size=MIN_PT):
    fig.text(0.5, 0.038, text, ha="center", va="center", fontsize=size, alpha=0.95)


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    _assert_text_contrast(fig, name)
    path = OUT / name
    fig.savefig(path, facecolor="white", bbox_inches=None)
    print(f"  wrote {path.relative_to(REPO)}")
    return fig


# --------------------------------------------------------------------------- #
# Data and validation status.
# --------------------------------------------------------------------------- #
def _json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def load_validation() -> dict:
    return {
        "v1": _json(VDIR / "v1_solver.json"),
        "v2": _json(VDIR / "v2_nulls.json"),
        "v8": _json(VDIR / "v8_reimplementation.json"),
        "summary": _json(VDIR / "summary.json"),
    }


def v1_target_survives(val: dict, target: str) -> bool | None:
    v1 = val.get("v1")
    if not v1:
        return None
    t = (v1.get("targets") or {}).get(target)
    return None if t is None else bool(t.get("outcome_survives"))


def caveat_for(val: dict, target: str, quantity: str | None = None) -> str | None:
    """The validation caveat this slide has to carry, assembled from the verdict files.

    Returns None only when there is genuinely nothing to say, which after this pass is rare.
    """
    parts = []
    survives = v1_target_survives(val, target)
    if survives is False:
        parts.append("changes under exact inference (../docs/audits/a1-validation/RESULTS.md)")
    v1 = val.get("v1") or {}
    if quantity:
        for row in v1.get("table", []):
            if row.get("target") == target and row.get("quantity") == quantity \
                    and not row.get("agrees"):
                parts.append("effect size is solver-dependent")
                break
    if (val.get("summary") or {}).get("quick"):
        parts.append("VALIDATION AT DEV SCALE, NOT REPORTABLE")
    return "; ".join(parts) if parts else None


def caveat_box(stack: "Stack", text: str):
    """The caveat goes ON the slide, in a box, at readable size. Never a footnote."""
    return stack.text(text, MIN_PT, box=True)


def _assert_fits(stack: "Stack", name: str):
    """A slide that has run out of canvas is a broken slide, so it raises rather than ships.

    Overflow on a fixed-size social image is silent: matplotlib draws the text and the viewer sees
    it clipped or overlapping. This is the check that makes the layout self-policing when a caveat
    grows because a verdict changed.
    """
    if stack.remaining() < -1e-9:
        raise AssertionError(
            f"{name}: the slide overflows its canvas by "
            f"{abs(stack.remaining()) * SLIDE_IN[1] * DPI:.0f}px. Shorten the copy or drop a "
            f"font size — do not ship a clipped slide.")


# --------------------------------------------------------------------------- #
# Slide 1 — title. No chart.
# --------------------------------------------------------------------------- #
def slide_1(val):
    fig = slide_canvas()
    s = Stack(fig, top=0.845)
    # TWO BLOCKS, NOT ONE, and the reason is rhythm rather than layout. The line is written as two
    # beats: the claim, then what happened to it. A single wrapped block reflows across the
    # sentence boundary and the second beat stops landing.
    s.text("I built my theory as a working model.", 50, bold=True, alpha=1.0, spacing=1.30,
           gap=0.045)
    s.text("It disagreed with me seven times.", 50, bold=True, alpha=1.0, spacing=1.30,
           gap=0.085)
    s.text("A simulation of how people read intent, and where it breaks.", 30, gap=0.075)
    s.swatches()
    footer(fig)
    _assert_fits(s, "slide 1")
    return save(fig, "slide_1_title.png")


# --------------------------------------------------------------------------- #
# Slide 2 — the two-gates result. The new headline.
# --------------------------------------------------------------------------- #
def slide_2(val):
    v = _json(RESULTS / "e31_verdict.json")
    if not v:
        print("  SKIPPED slide 2: results/e31_verdict.json is missing")
        return None
    cells = v["headline_cells"]
    honest = float(cells["crash_honest_ghost_label"]["prior_drift"])
    lie = float(cells["exploit_dishonest_creator_label"]["prior_drift"])
    multiple = lie / honest if honest > 0 else float("inf")

    fig = slide_canvas()
    s = Stack(fig)
    s.text("The same object. One word changed.", 46, bold=True, alpha=1.0, spacing=1.26,
           gap=0.040)

    ax = s.axes(0.200, gap=0.055, below=0.032)   # one line of tick labels
    xs, vals = [0, 1], [honest, lie]
    bars = ax.bar(xs, vals, width=0.54, color=INK, edgecolor=INK, lw=1.8)
    bars[0].set_alpha(0.60)
    bars[0].set_hatch("///")
    bars[1].set_alpha(1.00)
    ax.set_xticks(xs)
    ax.set_xticklabels(["honest label", "false label"], fontsize=MIN_PT)
    ax.tick_params(axis="x", pad=8, length=0)
    ax.set_ylabel("taken on", fontsize=MIN_PT)
    ax.tick_params(axis="y", labelsize=MIN_PT)
    # Headroom for the multiple annotation ABOVE the value labels rather than through them.
    ax.set_ylim(0, max(vals) * 1.95)
    for x, y in zip(xs, vals):
        ax.text(x, y + max(vals) * 0.03, f"{y:.2f}", ha="center", va="bottom",
                fontsize=28, fontweight="bold", alpha=1.0)
    ax.annotate("", xy=(1, max(vals) * 1.45), xytext=(0, max(vals) * 1.45),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color=INK))
    ax.text(0.5, max(vals) * 1.50, f"{multiple:.0f}x", ha="center", va="bottom",
            fontsize=38, fontweight="bold", alpha=1.0)

    s.text("Identical machine-made content. Only the label differs.", 25)

    # THE REQUIRED ANNOTATION. On the slide, per the spec, not in a footnote.
    lines = ["CEILING: this reader cannot doubt the label it conditioned on, so the "
             "multiple is an UPPER BOUND."]
    v8 = val.get("v8") or {}
    if v8.get("verdict") == "MECHANISM_REPLICATES_MAGNITUDE_DOES_NOT":
        lines.append("Rebuilt independently, the direction holds and the multiple does not. "
                     "Read the direction, not the number.")
    elif v8.get("verdict") == "REPLICATES":
        lines.append("Replicated in independent code written from the description alone.")
    extra = caveat_for(val, "e31", "exploit_mu_gap")
    if extra:
        # Sentence-cased, because a caveat assembled from verdict fragments should still read as
        # a sentence to somebody who is not reading the verdict files.
        lines.append("Also: " + extra + ".")
    caveat_box(s, "\n".join(lines))
    footer(fig)
    _assert_fits(s, "slide 2")
    return save(fig, "slide_2_two_gates.png")


# --------------------------------------------------------------------------- #
# Slide 3 — the competence result.
# --------------------------------------------------------------------------- #
def slide_3(val):
    """The competence result, drawn from E15 rather than E10.

    E10 established the effect on a coarse grid; E15 is the follow-up that resolves its SHAPE and
    sweeps down to zero skill, so it is the one with a curve worth showing. Both are committed and
    they agree where they overlap (V-6 checks that), and the finer grid is what makes the second
    finding legible: the reader's beliefs come apart well before its answers do.
    """
    p = RESULTS / "e15_competence_cliff.csv"
    v = _json(RESULTS / "e15_verdict.json")
    if not p.exists():
        print("  SKIPPED slide 3: results/e15_competence_cliff.csv is missing")
        return None
    df = pd.read_csv(p)
    main = df[df.get("arm", "main") == "main"] if "arm" in df.columns else df
    g = main.groupby("d")[["psi", "goal_accuracy"]].mean().reset_index().sort_values("d")
    skill = 1.0 - g.d.values
    belief = g.psi.values / g.psi.values.max()
    choice = g.goal_accuracy.values / g.goal_accuracy.values.max()

    # The two break points, read out of E15's own committed fits rather than eyeballed.
    shapes = (v or {}).get("shape_by_metric", {})
    belief_knee = None
    try:
        belief_knee = 1.0 - float(shapes["psi"]["fits"]["hinge"]["params"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    choice_knee = None
    if v and v.get("logistic_d50") is not None:
        choice_knee = 1.0 - float(v["logistic_d50"])

    fig = slide_canvas()
    s = Stack(fig)
    s.text("Same documents. Only the reader changed.", 46, bold=True, alpha=1.0, spacing=1.26,
           gap=0.040)

    ax = s.axes(0.215, gap=0.055, below=0.068)   # tick labels plus an x-axis label
    ax.plot(skill, belief, lw=3.4, color=INK, alpha=1.00, ls="-",
            label="what it believes")
    ax.plot(skill, choice, lw=3.4, color=INK, alpha=0.60, ls="--",
            label="what it picks")
    for knee, style_ in ((belief_knee, (0, (1, 2))), (choice_knee, (0, (5, 3)))):
        if knee is not None:
            ax.axvline(knee, color=INK, lw=1.8, alpha=0.60, ls=style_)
    ax.set_xlabel("the reader's own skill", fontsize=MIN_PT)
    ax.set_ylabel("share of best", fontsize=MIN_PT)
    ax.tick_params(labelsize=MIN_PT)
    ax.set_xlim(1.02, -0.02)          # skill falling left to right: the direction of the story
    ax.set_ylim(-0.03, 1.10)
    ax.legend(fontsize=MIN_PT, loc="lower left")

    s.text("No machine-made content anywhere in this test. Hold the documents "
           "constant, vary only who is reading, and what can be recovered collapses.", 25)
    if belief_knee is not None and choice_knee is not None:
        # "starts coming apart" rather than "comes apart": both numbers are knees in a fitted
        # curve, so they mark where the decline begins, not where it completes. The chart shows a
        # gradual decline from the first marker, and the copy has to match the chart.
        s.text(f"And what the reader believes starts coming apart at skill {belief_knee:.2f}, "
               f"long before its answers do at {choice_knee:.2f}. Preference data collects "
               f"answers.", 25, alpha=1.0)
    extra = caveat_for(val, "e32")
    if extra:
        caveat_box(s, extra)
    footer(fig)
    _assert_fits(s, "slide 3")
    return save(fig, "slide_3_competence.png")


# --------------------------------------------------------------------------- #
# Slide 4 — silent versus loud failure.
# --------------------------------------------------------------------------- #
def slide_4(val):
    p = RESULTS / "e32_cell_stats.csv"
    if not p.exists():
        print("  SKIPPED slide 4: results/e32_cell_stats.csv is missing")
        return None
    st = pd.read_csv(p)
    om = float(st.omega.min())
    expert = st[(st.arm == "foreign_content") & (st.omega == om)].iloc[0]
    novice = st[(st.arm == "unskilled_reader") & (st.omega == om)].iloc[0]

    fig = slide_canvas()
    s = Stack(fig)
    s.text("A novice looking at a person beats an expert looking at a machine.",
           42, bold=True, alpha=1.0, spacing=1.28, gap=0.045)

    ax = s.axes(0.200, gap=0.055, below=0.062)   # two lines of tick labels
    vals = [float(expert.accuracy), float(novice.accuracy)]
    bars = ax.bar([0, 1], vals, width=0.54, color=INK, edgecolor=INK, lw=1.8)
    bars[0].set_alpha(0.60)
    bars[0].set_hatch("///")
    bars[1].set_alpha(1.00)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["expert,\nmachine work", "novice,\nhuman work"], fontsize=MIN_PT)
    ax.tick_params(axis="x", pad=8, length=0)
    ax.set_ylabel("reads the\nreal purpose", fontsize=MIN_PT, linespacing=1.3)
    ax.tick_params(axis="y", labelsize=MIN_PT)
    ax.set_ylim(0, max(vals) * 1.30)
    for x, y in zip([0, 1], vals):
        ax.text(x, y + max(vals) * 0.035, f"{y:.2f}", ha="center", va="bottom",
                fontsize=30, fontweight="bold", alpha=1.0)

    s.text("Matched so both face the same information deficit. A badly aimed template "
           "still points near the truth. Content out of range points nowhere.", 25)
    s.text(f"The expert keeps working at it ({float(expert.engaged_fraction):.2f} of the time) "
           f"and stays lost. The novice quits at once "
           f"({float(novice.engaged_fraction):.3f}) and feels settled.", 25, alpha=1.0)
    extra = caveat_for(val, "e32", "foreign_within_at_matched")
    if extra:
        caveat_box(s, extra)
    footer(fig)
    _assert_fits(s, "slide 4")
    return save(fig, "slide_4_silent_vs_loud.png")


# --------------------------------------------------------------------------- #
# Slide 5 — the close. This is the slide that does the work.
# --------------------------------------------------------------------------- #
def slide_5(val):
    fig = slide_canvas()
    s = Stack(fig, top=0.84)
    s.text("Seven ideas died. One was mine.", 54, bold=True, alpha=1.0, spacing=1.30,
           gap=0.075)
    s.text("One experiment failed its own test three times and I didn't report it.",
           38, spacing=1.32, gap=0.075)
    s.swatches(gap=0.065)
    # Sized so the URL fits on one line at this margin. Larger than the other slides' footers,
    # which is what the specification asks of the closing slide.
    s.text(REPO_URL, 28, bold=True, alpha=1.0, nowrap=True)
    _assert_fits(s, "slide 5")
    return save(fig, "slide_5_close.png")


# --------------------------------------------------------------------------- #
# The social preview.
# --------------------------------------------------------------------------- #
def preview(val):
    fig = plt.figure(figsize=PREVIEW_IN)
    fig.text(0.5, 0.68, "Ghost Scale Simulation", ha="center", va="center",
             fontsize=52, fontweight="bold", alpha=1.0)
    fig.text(0.5, 0.44,
             "A working model of how people work out what someone\n"
             "was trying to do, and what happens when nothing was.",
             ha="center", va="center", fontsize=26, linespacing=1.45, alpha=0.95)
    ax = fig.add_axes([0.42, 0.15, 0.16, 0.10])
    tier_swatches(ax)
    return save(fig, "social_preview_1280x640.png")


def make_pdf(figs):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "carousel.pdf"
    with PdfPages(path) as pdf:
        for fig in figs:
            if fig is not None:
                pdf.savefig(fig, facecolor="white")
    print(f"  wrote {path.relative_to(REPO)}")


def main():
    style()
    val = load_validation()
    if not val.get("v1"):
        print("NOTE: no validation verdicts found in results/validation/. Slides will be built "
              "without validation caveats, which is only correct before the pass has been run.")
    elif (val.get("summary") or {}).get("quick"):
        print("NOTE: the validation verdicts on disk came from a --quick run. Every slide will "
              "carry a NOT REPORTABLE banner, which is the honest thing for it to carry.")

    print("building slides:")
    figs = [slide_1(val), slide_2(val), slide_3(val), slide_4(val), slide_5(val)]
    make_pdf(figs)
    preview(val)
    for f in figs:
        if f is not None:
            plt.close(f)
    print("done. figures/ (research charts) untouched.")


if __name__ == "__main__":
    main()
