"""Demonstrated competence against attention history, as independent generators (spec §3.1, trunk E).

Competence K is the accuracy with which a maker realizes intended actions and emits faithful
semantic tokens. Attention history H is a tilt on which surface features the maker emits early
and which transitions it practices, with a reward-linked strength that decays after the reward
reverses. Neither is defined in terms of the other; card E01 asserts it. A reader is an agent
too: its own history is a tilt on which ROUTES it weighs first, corrected by feedback.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from . import joint as J
from .world import N_ACT, N_FEAT, ROUTES, Maker, World, make_maker

HISTORY_LEVELS = ("none", "weak", "strong")
H_STRENGTH = {"none": 0.0, "weak": 0.6, "strong": 1.5}


def agent(world: World, name: str, rng: np.random.Generator, family: int, competence: str, history: str,
          pref: int | None = None, plan: int | None = None, h_feat=None, h_trans=None) -> Maker:
    """A maker (or reader-as-agent) with competence and history set independently."""
    hs = H_STRENGTH[history]
    m = make_maker(world, name, rng, family=family, pref=pref, plan=plan, competence=competence, h_strength=hs,
                   h_feat=h_feat, h_trans=h_trans)
    m.comm["history"] = history
    m.comm["competence"] = competence
    return m


def reverse_reward(m: Maker, at_episode: int) -> Maker:
    m.h_reversed_at = int(at_episode)
    return m


def reader_route_history(rng: np.random.Generator, strength: float, routes=ROUTES, favoured: str | None = None) -> dict:
    """A reader's own attention history over routes: initial tempering weights that favour one
    route by construction (the stale-history bias E02/E05 correct)."""
    h = rng.normal(0.0, 1.0, len(routes))
    if favoured is not None:
        h[list(routes).index(favoured)] += 2.0
    w = C.softmax(strength * h) * len(routes)
    return {r: float(x) for r, x in zip(routes, w)}


def signature(eps: list, steps: int = 3) -> np.ndarray:
    """Early action-bigram frequencies: the residual of an acquisition path at equal skill (E08)."""
    M = np.zeros((N_ACT, N_ACT)) + 0.1
    for ep in eps:
        a = ep["action"]
        for t in range(1, min(steps, len(a))):
            M[a[t - 1], a[t]] += 1
    return M / M.sum()


def early_relevance(ep: dict, m: Maker, fam=None, inv_vocab=None, steps: int = 2) -> float:
    """How strongly the maker's early surface tokens follow its attention tilt: the tilt at the
    emitted token in EXCESS of what the realized action's base emission would give on average.
    Conditioning on the realized action keeps competence (which changes realized actions) out of
    the measure by construction."""
    vals = []
    for t, s in enumerate(ep["surface"][:steps]):
        base = 0.0
        if fam is not None:
            a = int(ep["action"][t]) if inv_vocab is None else int(inv_vocab[ep["action"][t]])
            base = float(fam.feat[a] @ m.h_feat)
        vals.append(m.h_feat[s] - base)
    return float(np.mean(vals))


H_GRID = np.round(np.arange(-1.0, 4.01, 0.05), 2)


def history_signal(eps: list, m: Maker, fam, inv_vocab=None, steps: int = 2) -> float:
    """Early-relevance signal: the maximum-likelihood tilt strength of the maker's early surface
    tokens along its attention direction, given the realized actions. The emission model is the
    world's own (base row of the realized action, tilted by exp(h * h_feat)), so the estimate is
    consistent for the planted strength under any competence: competence changes which base rows
    are in play, not the strength that maximizes their likelihood. Zero planted tilt estimates
    to zero with error shrinking in the number of tokens - a correlation over ten bins does not."""
    pairs = []
    for ep in eps:
        for t, s in enumerate(ep["surface"][:steps]):
            a = int(ep["action"][t]) if inv_vocab is None else int(inv_vocab[ep["action"][t]])
            pairs.append((a, int(s)))
    if not pairs:
        return 0.0
    base = np.log(fam.feat + 1e-300)                            # (N_ACT, N_FEAT)
    best_h, best_ll = 0.0, -np.inf
    for h in H_GRID:
        tilt = base + h * m.h_feat[None, :]
        tilt = tilt - np.log(np.exp(tilt).sum(axis=1, keepdims=True))
        ll = float(sum(tilt[a, s] for a, s in pairs))
        if ll > best_ll + 1e-12:
            best_h, best_ll = float(h), ll
    return best_h


def process_score(reader: J.Reader, eps: list, ep_next: dict, prior: np.ndarray, routes=("action", "semantic", "context")) -> float:
    tabs = reader.route_tables(eps, routes)
    post = J.joint(prior, tabs)
    pred = J.next_episode_action_dist(reader, post)
    return float(np.log(max(pred[int(ep_next["action"][0])], 1e-12)))


# --------------------------------------------------------------------------- #
# Diverse readers (E09): intersection, never naive averaging.
# --------------------------------------------------------------------------- #
def combine_readers(posts: list, method: str, prior: np.ndarray | None = None) -> np.ndarray:
    P = np.stack(posts)
    if method == "average":
        out = P.mean(axis=0)
    elif method == "likelihood_product":                        # calibrated-likelihood intersection: product of likelihood ratios
        pr = prior if prior is not None else np.full(P.shape[1], 1.0 / P.shape[1])
        lr = np.log(np.maximum(P, 1e-300)) - np.log(pr)
        out = C.softmax(lr.sum(axis=0) + np.log(pr))
    elif method == "feasible_set":                              # states every reader keeps above a floor, uniform within
        keep = np.all(P >= 1.0 / P.shape[1] * 0.5, axis=0)
        out = keep.astype(float)
        if out.sum() == 0:
            out = P.mean(axis=0)
    else:
        raise KeyError(method)
    return out / out.sum()
