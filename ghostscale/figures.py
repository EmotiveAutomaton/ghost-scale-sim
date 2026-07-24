"""Figure builders. Ghost Scale branding: tiers keep their published names and a fixed
opacity-derived colour ramp (CREATOR opaque -> GHOST faint), echoing the four-tier scale.
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
    ax.set(xlabel="timestep", ylabel="P(attention = DEEP)",
           title="Disengagement survival by tier", ylim=(-0.03, 1.03))
    ax.legend(title="provenance")

    ax = axes[1]
    for name in K.PROVENANCE_NAMES:
        d = series_df[series_df.tier == name].sort_values("t")
        ax.plot(d.t, d.deep_epi_goal, TIER_LINE[name], color=_tier_color(name),
                lw=2, label=name)
    # Reference: the effort gap the epistemic value must clear to keep DEEP.
    if "effort_gap" in series_df.columns:
        gap = float(series_df.effort_gap.iloc[0])
        ax.axhline(gap, color="firebrick", lw=1, ls="-.", alpha=0.8,
                   label=f"effort gap ({gap:.2f})")
    ax.set(xlabel="timestep", ylabel="epistemic value of DEEP about goal (nats)",
           title="EFE epistemic term collapses for GHOST")
    ax.legend(title="provenance")

    fig.suptitle("E1 — The generative crash (H1)", fontweight="bold")
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
        title = (f"true={tier}, signal={sig}\n"
                 f"within H={st['within']:.2f}  between H={st['between']:.2f}")
        ax.set_title(title, fontweight="bold" if highlight else "normal",
                     color="firebrick" if highlight else "black")
        if highlight:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("firebrick")
                spine.set_linewidth(1.6)
        ax.set(xlim=(-0.05, np.log(4) + 0.05), ylim=(-0.05, 1.05), yticks=[])

    for ax in axes[1, :]:
        ax.set_xlabel("within-observer entropy (nats)\n<- confident        uncertain ->")
    # Legend for goals.
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=goal_cmap(g % 10),
                          label=K.GOAL_NAMES[g]) for g in range(len(K.GOAL_NAMES))]
    fig.legend(handles=handles, title="modal inferred goal", loc="center right",
               bbox_to_anchor=(1.10, 0.5))
    fig.suptitle("E2 — Confident disagreement is the signature of hallucination (H2)\n"
                 "colour = each observer's confidently inferred goal",
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


def save_simple_lines(df, x, y, hue, path: Path, xlabel="", ylabel="", title=""):
    """Generic line plot used by E3-E6 (one line per hue level)."""
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for level, d in df.groupby(hue):
        d = d.sort_values(x)
        ax.plot(d[x], d[y], marker="o", ms=4, lw=1.8, label=str(level))
    ax.set(xlabel=xlabel or x, ylabel=ylabel or y, title=title)
    ax.legend(title=hue)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
