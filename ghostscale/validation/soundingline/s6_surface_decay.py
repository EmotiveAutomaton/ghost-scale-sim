"""S-6 — does surface thickness decay across an artifact while depth does not?

**LIKE S-3, THIS IS A PURPOSE-BUILT CONSTRUCTION AND NOT THE V5 WORLD.** The shared model has one
decision stream and no per-artifact metabolic budget on the CREATOR's side -- the reserve V6 added
is the reader's. Splitting the creator's output into a cached stream and a paid stream is new
machinery, so it is built here in miniature rather than added to a closed model.

THE CLAIM, from `docs/theory/SURFACE_AND_DEPTH.md`, derived from automaticity:

    content decisions are practised and cached, so they cost little and do not decay across one
    artifact; surface decisions are a performance held consciously, so under a budget they degrade.

THREE CREATORS, and the third carries the sharper prediction:

    practised   a cache covering most content decisions, and a depleting budget
    novice      no cache, so BOTH streams draw on the same budget
    synthetic   no budget at all

    S-1 there  surface density declines across the artifact; content density does not
    S-2 there  the machine signature is not thin depth. It is A SURFACE THAT DOES NOT MOVE.

The second is the one worth having, because it is a positive signature rather than an absence, and
absences are what every AI detector already looks for.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ...config import Config
from ...prereg_v6 import BOOTSTRAP_DRAWS, percentile_interval
from ...v6 import SEED_OFFSET
from . import sl_dir

POSITIONS = 40
BUDGET = 12.0
SURFACE_COST = 0.55
CONTENT_COST = 0.55        # identical, so any difference is the cache and not the price
CACHE_HIT = 0.85           # a practised creator's content decision is cached this often


def _emit(kind: str, rng) -> pd.DataFrame:
    budget = float("inf") if kind == "synthetic" else BUDGET
    cache = CACHE_HIT if kind == "practised" else 0.0
    rows = []
    for t in range(POSITIONS):
        # A content decision is cached, and a cached decision costs nothing to produce.
        cached = rng.random() < cache
        c_cost = 0.0 if cached else CONTENT_COST
        content = 1 if (cached or budget >= c_cost) else 0
        if content and not cached:
            budget -= c_cost
        # A surface decision is never cached: it is held consciously, so it always costs.
        surface = 1 if budget >= SURFACE_COST else 0
        if surface:
            budget -= SURFACE_COST
        rows.append({"t": t, "content": content, "surface": surface,
                     "budget_left": budget if np.isfinite(budget) else -1.0})
    return pd.DataFrame(rows)


def _slope(y: np.ndarray) -> float:
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, 1)[0]) if y.size >= 2 else float("nan")


def run(cfg: Config, n_obs: int = 400) -> dict:
    rng = np.random.default_rng(SEED_OFFSET + 90_600)
    frames = []
    for kind in ("practised", "novice", "synthetic"):
        for i in range(int(n_obs)):
            d = _emit(kind, rng)
            d["kind"] = kind
            d["i"] = i
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(sl_dir() / "s6_surface_decay.csv", index=False)

    out = {}
    for kind in ("practised", "novice", "synthetic"):
        sub = df[df.kind == kind]
        per = sub.groupby("i").apply(
            lambda s: pd.Series({"surface": _slope(s.surface.to_numpy(dtype=float)),
                                 "content": _slope(s.content.to_numpy(dtype=float))}),
            include_groups=False)
        stats = {}
        for col in ("surface", "content"):
            v = per[col].to_numpy(dtype=float)
            draws = [float(np.mean(rng.choice(v, v.size, replace=True)))
                     for _ in range(BOOTSTRAP_DRAWS)]
            lo, hi = percentile_interval(draws)
            # TOL, because a creator with an infinite budget emits a constant stream and
            # polyfit returns a slope of about -1e-17 on it. Without a tolerance that reads as
            # "declines" and the sharpest prediction in the set fails on floating-point dust.
            TOL = 1e-6
            stats[col] = {"slope_per_position": float(v.mean()), "interval": [lo, hi],
                          "declines": bool(np.isfinite(hi) and hi < -TOL),
                          "flat": bool(np.isfinite(lo) and abs(float(v.mean())) < TOL)}
        stats["surface_density"] = float(sub.surface.mean())
        stats["content_density"] = float(sub.content.mean())
        out[kind] = stats

    # THE FIRST PREDICTION IS A COMPARISON, NOT AN ABSOLUTE. A practised creator still pays for
    # the content decisions its cache misses, so its content stream declines a little. The claim
    # is that surface decays FASTER, and that is what gets tested.
    prac = df[df.kind == "practised"]
    per_p = prac.groupby("i").apply(
        lambda z: pd.Series({"surface": _slope(z.surface.to_numpy(dtype=float)),
                             "content": _slope(z.content.to_numpy(dtype=float))}),
        include_groups=False)
    gap = (per_p.content - per_p.surface).to_numpy(dtype=float)
    gdraws = [float(np.mean(rng.choice(gap, gap.size, replace=True)))
              for _ in range(BOOTSTRAP_DRAWS)]
    glo, ghi = percentile_interval(gdraws)
    surface_faster = {"content_slope_minus_surface_slope": float(gap.mean()),
                      "interval": [glo, ghi],
                      "surface_decays_faster": bool(np.isfinite(glo) and glo > 0)}

    p, n, s = out["practised"], out["novice"], out["synthetic"]
    sdoc_s1 = bool(p["surface"]["declines"] and surface_faster["surface_decays_faster"])
    sdoc_s2 = bool(s["surface"]["flat"] and not s["surface"]["declines"])
    novice_both = bool(n["surface"]["declines"] and n["content"]["declines"])

    verdict = {
        "test": "S-6 — does surface decay across an artifact while content does not?",
        "for": "Sounding Line, SURFACE_AND_DEPTH.md",
        "IMPORTANT": (
            "a purpose-built construction, not the V5 world. The shared model has one decision "
            "stream and no creator-side budget; splitting it is new machinery and this repository "
            "is otherwise closed. Read it as what a two-stream creator does."),
        "construction": {"positions": POSITIONS, "budget": BUDGET,
                         "surface_cost": SURFACE_COST, "content_cost": CONTENT_COST,
                         "cache_hit_rate_when_practised": CACHE_HIT,
                         "note": "both streams cost the same. The only difference is the cache."},
        "by_creator": out,
        "practised_surface_decays_faster_than_content": surface_faster,
        "predictions": {
            "S1_surface_decays_FASTER_than_content_for_a_practised_creator": sdoc_s1,
            "S2_a_creator_with_no_budget_shows_a_FLAT_surface": sdoc_s2,
            "a_novice_shows_decay_in_both": novice_both,
        },
        "outcome": ("ALL_THREE_PREDICTIONS_HOLD" if (sdoc_s1 and sdoc_s2 and novice_both)
                    else "SOME_PREDICTIONS_FAIL"),
        "what_would_have_falsified_it": (
            "content decaying as fast as surface for a practised creator, which would make the "
            "cache irrelevant; or a synthetic creator whose surface also declined, which would "
            "remove the positive machine signature and leave only an absence."),
        "the_honest_caveat": (
            "this construction BUILDS IN the asymmetry it then measures: content is cached and "
            "surface is not. It is a consistency check on the theory's arithmetic, not evidence "
            "that real practice works this way. What it earns is the direction of the third "
            "prediction, which does not follow trivially: a budgetless creator is FLAT rather "
            "than merely high."),
    }
    (sl_dir() / "s6_surface_decay.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
