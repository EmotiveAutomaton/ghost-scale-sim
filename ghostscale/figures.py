"""Figure builders. Ghost Scale branding: tiers keep their published names and a fixed
opacity-derived colour ramp (CREATOR opaque -> GHOST faint), echoing the four-tier scale.

LABELLING RULE for every figure in this project. Axis labels and panel titles are written for
someone who has not read the paper. Plain words carry the meaning; the technical name follows in
parentheses only where a reader who HAS read the paper would otherwise lose the thread. So
"how far the recovered preferences drift from the truth (nats)", not "KL(C_recovered || C_true)".
The units stay, because a number without units is not checkable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import constants as K

# Tier colours: a single hue at decreasing opacity, matching the Ghost Scale (100/95/60/5%).
_TIER_BASE = (0.13, 0.20, 0.42)  # deep indigo
TIER_ALPHA = {"CREATOR": 1.00, "POLISHED": 0.75, "CURATOR": 0.45, "GHOST": 0.20}
TIER_COLOR = {name: (*_TIER_BASE, TIER_ALPHA[name]) for name in K.PROVENANCE_NAMES}
TIER_LINE = {"CREATOR": "-", "POLISHED": "-", "CURATOR": "--", "GHOST": ":"}


def goal_label(g: int) -> str:
    """Display name for a goal index. The code calls them G0..G3 because they are abstract and
    interchangeable by construction; a reader seeing the figure for the first time needs a word,
    not an index."""
    return f"purpose {int(g) + 1}"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def _tier_color(name: str):
    # For lines we want visible colour; use solid indigo with tier-scaled darkness.
    a = TIER_ALPHA[name]
    return (_TIER_BASE[0] + (1 - a) * 0.55,
            _TIER_BASE[1] + (1 - a) * 0.55,
            _TIER_BASE[2] + (1 - a) * 0.45)


# --------------------------------------------------------------------------- #
# E1 — the generative crash.
# --------------------------------------------------------------------------- #
def fig_e1(series_df, path: Path) -> None:
    """Two panels: (left) DEEP-engagement survival curves by tier;
    (right) EFE decomposition over time — epistemic (goal) value of DEEP collapsing by tier,
    with the pragmatic effort gap as a reference line."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for name in K.PROVENANCE_NAMES:
        d = series_df[series_df.tier == name].sort_values("t")
        ax.plot(d.t, d.frac_deep, TIER_LINE[name], color=_tier_color(name),
                lw=2, label=name)
    ax.set(xlabel="time spent with the artifact (steps)",
           ylabel="share of readers still looking closely",
           title="Readers give up on work that has no intent behind it",
           ylim=(-0.03, 1.03))
    ax.legend(title="who made it")

    ax = axes[1]
    for name in K.PROVENANCE_NAMES:
        d = series_df[series_df.tier == name].sort_values("t")
        ax.plot(d.t, d.deep_epi_goal, TIER_LINE[name], color=_tier_color(name),
                lw=2, label=name)
    # Reference: the effort gap the epistemic value must clear to keep DEEP.
    if "effort_gap" in series_df.columns:
        gap = float(series_df.effort_gap.iloc[0])
        ax.axhline(gap, color="firebrick", lw=1, ls="-.", alpha=0.8,
                   label=f"what looking closely costs ({gap:.2f})")
    ax.set(xlabel="time spent with the artifact (steps)",
           ylabel="expected payoff from looking closely (nats)",
           title="Below the cost line, looking closely stops being worth it")
    ax.legend(title="who made it")

    fig.suptitle("E1 — The generative crash: attention drains away from work "
                 "that was made without a purpose", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# E2 — IRL convergence failure (the headline figure).
# --------------------------------------------------------------------------- #
def fig_e2(points_df, cell_stats, path: Path) -> None:
    """2x2 panel of confidence-vs-disagreement, one point per observer.

    x = within-observer entropy (confidence: left = confident).
    y = |modal goal - population modal goal| jittered spread shown via colour by modal goal.
    The headline cell (true=GHOST, signal=SIG_CREATOR) shows a tight low-entropy cloud whose
    modal goals nonetheless span all four goals — confident disagreement.
    """
    set_style()
    cells = [("CREATOR", "SIG_CREATOR"), ("CREATOR", "SIG_GHOST"),
             ("GHOST", "SIG_CREATOR"), ("GHOST", "SIG_GHOST")]
    goal_cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4), sharex=True, sharey=True)

    for ax, (tier, sig) in zip(axes.ravel(), cells):
        d = points_df[(points_df.true_provenance == tier) & (points_df.declared_signal == sig)]
        # y-jitter by observer index for visual spread; colour by modal goal.
        rng = np.random.default_rng(0)
        yj = rng.uniform(0, 1, size=len(d))
        ax.scatter(d.within_entropy, yj, c=[goal_cmap(g % 10) for g in d.modal_goal],
                   s=14, alpha=0.75, edgecolors="none")
        st = cell_stats[(tier, sig)]
        highlight = (tier == "GHOST" and sig == "SIG_CREATOR")
        told = "told it was human-made" if sig == "SIG_CREATOR" else "told it was machine-made"
        really = "really human-made" if tier == "CREATOR" else "really machine-made"
        title = (f"{really}, {told}\n"
                 f"each reader is sure: {st['within']:.2f}   "
                 f"readers disagree: {st['between']:.2f}")
        ax.set_title(title, fontweight="bold" if highlight else "normal",
                     color="firebrick" if highlight else "black")
        if highlight:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("firebrick")
                spine.set_linewidth(1.6)
        ax.set(xlim=(-0.05, np.log(4) + 0.05), ylim=(-0.05, 1.05), yticks=[])

    for ax in axes[1, :]:
        ax.set_xlabel("how unsure each reader is (nats)\n"
                      "<- certain of the answer        no idea ->")
    # Legend for goals.
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=goal_cmap(g % 10),
                          label=goal_label(g)) for g in range(len(K.GOAL_NAMES))]
    fig.legend(handles=handles, title="the goal this reader\nsettled on", loc="center right",
               bbox_to_anchor=(1.12, 0.5))
    fig.suptitle("E2 — What hallucination looks like: everyone is certain, "
                 "and nobody agrees\n"
                 "one dot per reader; colour is the purpose that reader decided on",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.99, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(matrix, xlabels, ylabels, path: Path, xlabel="", ylabel="", title="",
                 cbar_label="", cmap="magma", contour_level=None):
    """Heatmap with optional highlighted contour (used to locate a boundary, e.g. E4)."""
    set_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    M = np.asarray(matrix, dtype=float)
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(len(xlabels)), [f"{x:g}" for x in xlabels], rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), [f"{y:g}" for y in ylabels])
    if contour_level is not None and np.nanmin(M) < contour_level < np.nanmax(M):
        ax.contour(M, levels=[contour_level], colors="cyan", linewidths=2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_simple_lines(df, x, y, hue, path: Path, xlabel="", ylabel="", title="",
                      legend_title=None, level_labels=None):
    """Generic line plot used by E3-E6 (one line per hue level).

    ``legend_title`` and ``level_labels`` exist so the legend can say what the levels MEAN
    ("reads the label", "ignores the label") instead of echoing the column name and the raw
    factor level, which are written for the CSV rather than for a reader.
    """
    set_style()
    level_labels = level_labels or {}
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for level, d in df.groupby(hue):
        d = d.sort_values(x)
        ax.plot(d[x], d[y], marker="o", ms=4, lw=1.8, label=level_labels.get(level, str(level)))
    ax.set(xlabel=xlabel or x, ylabel=ylabel or y, title=title)
    ax.legend(title=legend_title if legend_title is not None else hue)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
