"""Attention as allocation, never as invention (spec §3.3).

Two analogues over the same disjoint evidence channels:

* selection attention: a finite budget of channel inspections; unselected channels are not
  observed (weight 0);
* precision attention: every channel is available, and each channel's log-likelihood is tempered
  by a predeclared nonnegative weight.

Neither can alter ground truth, add a label, or sharpen the correct cue by construction: the
oracle policy is the only one allowed to see which channel is diagnostic, and it exists to be the
ceiling other policies are read against. Policies:

    uniform           every channel, weight 1 (the identity)
    random            a random subset within budget
    salience          the channels the world marks as conspicuous (a nuisance attribute; a
                      concealer can make weak cues conspicuous)
    oracle            the channels ranked by their true diagnosticity for the target
    wrong             the oracle ranking inverted
    learned           weights fitted on labelled TRAINING worlds by held-out log score
    adaptive          local prior first; on surprise, weight moves to target-specific channels
    narrow            precision concentrated on the initially best channel, never re-evaluated
    broad             uniform monitoring with periodic exploration of the remaining channels
"""
from __future__ import annotations

import numpy as np

from . import common as C
from .exact import Model
from .world import CHANNELS

POLICIES = ("uniform", "random", "salience", "oracle", "wrong", "learned", "adaptive", "narrow", "broad")
COST = {"surface": 1.0, "common_structure": 0.5, "group_convention": 0.5, "mechanics": 0.5,
        "goal_consequences": 0.5, "opportunity_set": 1.0, "paid_cost": 0.5, "anomaly": 0.25,
        "communicative_shaping": 0.25, "source_history": 0.5, "process_records": 2.0}
TARGET_SPECIFIC = ("goal_consequences", "mechanics", "process_records", "surface")
WEIGHT_GRID = (0.0, 0.5, 1.0, 2.0)


def salience_of(channels: list, rng, adversarial_weak: str | None = None, quiet: str | None = None) -> dict:
    """A conspicuousness score per channel: a nuisance attribute of the world's surface. Under
    adversarial salience the concealer makes one weak channel conspicuous and the diagnostic one
    quiet."""
    s = {c: float(rng.uniform(0.2, 1.0)) for c in channels}
    if adversarial_weak is not None and adversarial_weak in s:
        s[adversarial_weak] = 1.5
    if quiet is not None and quiet in s:
        s[quiet] = 0.05
    return s


def channel_diagnosticity(model: Model, prior: np.ndarray, items: list, channel: str, truth_key: str = "index") -> float:
    """Mean log-score gain about the target from the channel alone, over labelled items
    [(arts, truth_index)]. This is the oracle ruler; it sees labels and exists only as a ceiling."""
    gains = []
    for arts, ti in items:
        q = model.posterior(prior, arts, (channel,))
        gains.append(C.log_score(q, ti) - C.log_score(prior, ti))
    return float(np.mean(gains)) if gains else 0.0


def select(policy: str, channels: list, budget: float, rng, salience: dict | None = None,
           ranking: list | None = None, learned: dict | None = None) -> list:
    """Channels chosen under a budget for a selection policy. ``ranking`` is the oracle order."""
    cost = lambda c: COST.get(c, 1.0)
    if policy in ("uniform", "adaptive"):
        return list(channels)                      # adaptive allocates precision sequentially (adaptive_read)

    def take(order):
        out, spent = [], 0.0
        for c in order:
            if spent + cost(c) <= budget + 1e-9:
                out.append(c)
                spent += cost(c)
        return out

    if policy == "random":
        return take(list(rng.permutation(channels)))
    if policy == "salience":
        return take(sorted(channels, key=lambda c: -salience.get(c, 0.0)))
    if policy == "oracle":
        return take(list(ranking))
    if policy == "wrong":
        return take(list(ranking)[::-1])
    if policy == "learned":
        return take(sorted(channels, key=lambda c: -(learned or {}).get(c, 0.0)))
    if policy == "narrow":
        return take(list(ranking)[:1]) if ranking else take(list(channels)[:1])
    if policy == "broad":
        return take(list(rng.permutation(channels)))
    raise ValueError(policy)


def precision(policy: str, channels: list, rng, ranking: list | None = None, learned: dict | None = None,
              salience: dict | None = None, budget: float | None = None) -> dict:
    """Nonnegative weights per channel for a precision policy; ``budget`` caps their sum."""
    if policy == "uniform":
        w = {c: 1.0 for c in channels}
    elif policy == "random":
        w = {c: float(rng.choice(WEIGHT_GRID)) for c in channels}
    elif policy == "salience":
        w = {c: float(salience.get(c, 0.5)) for c in channels}
    elif policy == "oracle":
        w = {c: 2.0 if c in ranking[:2] else (1.0 if c in ranking[:4] else 0.25) for c in channels}
    elif policy == "wrong":
        r = list(ranking)[::-1]
        w = {c: 2.0 if c in r[:2] else (1.0 if c in r[:4] else 0.25) for c in channels}
    elif policy == "learned":
        w = {c: float(learned.get(c, 1.0)) for c in channels}
    elif policy == "narrow":
        w = {c: (3.0 if c == ranking[0] else 0.0) for c in channels}
    elif policy == "broad":
        w = {c: 1.0 for c in channels}
    else:
        raise ValueError(policy)
    if budget is not None:
        s = sum(w.values())
        if s > budget and s > 0:
            w = {c: v * budget / s for c, v in w.items()}
    return w


def fit_precision(model: Model, prior: np.ndarray, train_items: list, channels: list, grid=WEIGHT_GRID,
                  passes: int = 2) -> dict:
    """Coordinate ascent on held-out log score over a weight grid, on LABELLED TRAINING items
    [(arts, truth_index)] from training worlds. The learned reader never sees a test label."""
    w = {c: 1.0 for c in channels}

    def score(weights):
        return float(np.mean([C.log_score(model.posterior(prior, arts, channels, weights), ti) for arts, ti in train_items]))

    best = score(w)
    for _ in range(passes):
        for c in channels:
            for g in grid:
                trial = dict(w)
                trial[c] = float(g)
                s = score(trial)
                if s > best + 1e-9:
                    best, w = s, trial
    return w


def information_per_cost(gain: float, channels: list) -> float:
    spent = sum(COST.get(c, 1.0) for c in channels)
    return float(gain / spent) if spent > 0 else 0.0


def adaptive_read(model: Model, prior: np.ndarray, arts: list, channels: list, local_weights: dict,
                  surprise_threshold: float = 2.0, target_weights: dict | None = None) -> dict:
    """Sequential reading: local-prior weights until an artifact's surprise (negative log
    predictive of its surface under the running posterior, relative to the population
    prediction) exceeds a threshold; then weights move to target-specific channels. Returns the
    posterior trajectory and the step at which reallocation happened."""
    tw = target_weights or {c: (2.0 if c in TARGET_SPECIFIC else 0.25) for c in channels}
    q = np.asarray(prior, float).copy()
    weights = dict(local_weights)
    realloc_at = None
    traj = []
    for t, a in enumerate(arts):
        fam = a["family"]
        pred = model.predictive_surface(q, fam, a["domain"])
        base = model.predictive_surface(prior, fam, a["domain"])
        feats = np.asarray(a["features"])
        surprise = float(np.log(np.maximum(base[feats], 1e-12)).sum() - np.log(np.maximum(pred[feats], 1e-12)).sum())
        if realloc_at is None and surprise > surprise_threshold:
            weights = dict(tw)
            realloc_at = t
        q = model.posterior(q, [a], channels, weights)
        traj.append(q.copy())
    return {"posterior": q, "trajectory": traj, "reallocated_at": realloc_at}


def static_read(model: Model, prior: np.ndarray, arts: list, channels: list, weights: dict) -> np.ndarray:
    return model.posterior(prior, arts, channels, weights)


def tunnel_read(model: Model, prior: np.ndarray, arts: list, channels: list, ranking: list,
                mode: str = "narrow", explore_every: int = 4) -> dict:
    """Narrow: precision on the initially best channel for the whole stream. Broad: uniform with
    periodic exploration (every ``explore_every`` artifacts, all channels at weight 1)."""
    q = np.asarray(prior, float).copy()
    traj = []
    for t, a in enumerate(arts):
        if mode == "narrow":
            w = {c: (3.0 if c == ranking[0] else 0.0) for c in channels}
        else:
            w = {c: 1.0 for c in channels} if (t % explore_every == 0) else {c: (2.0 if c == ranking[0] else 0.5) for c in channels}
        q = model.posterior(q, [a], channels, w)
        traj.append(q.copy())
    return {"posterior": q, "trajectory": traj}


def no_information_world(arts: list, rng, channels: list) -> list:
    """Replace every channel observation by a draw that carries nothing about the maker: features
    from a uniform distribution, structure and convention markers uniform, payoff uniform."""
    out = []
    for a in arts:
        b = dict(a)
        nf = int(np.max(a["features"])) + 1 if len(a["features"]) else 1
        b["features"] = rng.integers(0, max(nf, 2), size=len(a["features"]))
        if "structure_obs" in a:
            b["structure_obs"] = rng.integers(0, max(1, max(a["structure_obs"]) + 1), size=len(a["structure_obs"])).tolist()
        if "convention_obs" in a:
            b["convention_obs"] = rng.integers(0, max(nf, 2), size=len(a["convention_obs"])).tolist()
        if "payoff_obs" in a:
            b["payoff_obs"] = int(rng.integers(0, 4))
        b["method"] = int(rng.integers(0, 2)) if a.get("method") is not None else None
        b["log"] = dict(a["log"])
        b["log"]["goal"] = -1
        out.append(b)
    return out


def duplicate_channel(arts: list, src: str = "surface") -> list:
    """Cue duplication (A13): the surface presented twice under two names carries no new
    information; a reader that counts it twice inflates confidence."""
    out = []
    for a in arts:
        b = dict(a)
        b["duplicate_of_surface"] = True
        out.append(b)
    return out
