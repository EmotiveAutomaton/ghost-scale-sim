"""The uptake bridge: q(w_maker | artifacts) -> q_trusted(w) -> C_AIF -> reader policy
(spec section 10).

The downstream task is a small choice task over the goal channels: each action yields an outcome
distribution over channels, and the reader's preferences over channels (its C_AIF) decide the
policy. The bridge mixes the reader's own standing preference with a trusted posterior over the
maker's profile at an explicit uptake weight. Identities (U01): zero weight leaves the policy
bit-identical; a uniform posterior produces no directional movement; C_AIF never contains a
provenance or source preference because the task has none to hold.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def representation(post: dict, family: dict, kind: str) -> np.ndarray:
    """A single profile vector summarising a posterior, by representation kind (U03)."""
    names = list(post)
    ng = len(next(iter(family.values())))
    if kind == "map":
        return np.asarray(family[max(post, key=post.get)], float)
    if kind == "mean":
        w = np.zeros(ng)
        for n in names:
            w += post[n] * family[n]
        return w / w.sum()
    if kind == "lower_confidence":
        # shrink the posterior mean toward uniform in proportion to posterior entropy
        w = representation(post, family, "mean")
        v = np.array([post[n] for n in names])
        h = float(-(v[v > 0] * np.log(v[v > 0])).sum()) / np.log(len(names))
        u = np.full(ng, 1.0 / ng)
        return (1 - h) * w + h * u
    if kind == "confidence_gated":
        best = max(post.values())
        w = representation(post, family, "mean")
        return w if best >= 0.6 else np.full(ng, 1.0 / ng)
    raise ValueError(kind)


def bridge(c_self: np.ndarray, w_trusted: np.ndarray, uptake: float, trust: float = 1.0) -> np.ndarray:
    """C_AIF = (1 - u*trust) * C_self + u*trust * w_trusted, as a preference vector over channels."""
    u = float(uptake) * float(trust)
    if u == 0.0:
        return np.asarray(c_self, float).copy()      # bit-identical: the U01 identity
    out = (1.0 - u) * np.asarray(c_self, float) + u * np.asarray(w_trusted, float)
    return out / out.sum()


def task(rng, ng: int, n_actions: int = 6) -> np.ndarray:
    """Each action's outcome distribution over channels."""
    return rng.dirichlet(np.ones(ng), size=int(n_actions))


def policy(c_aif: np.ndarray, outcomes: np.ndarray, beta: float = 8.0) -> np.ndarray:
    """Softmax policy over actions by expected preference (log-preference utility)."""
    util = outcomes @ np.log(np.maximum(c_aif, _EPS))
    z = np.exp(beta * (util - util.max()))
    return z / z.sum()


def regret(pol: np.ndarray, outcomes: np.ndarray, c_true: np.ndarray) -> float:
    """Expected shortfall of the policy against the best action under the evaluation preference."""
    util = outcomes @ np.log(np.maximum(c_true, _EPS))
    return float(util.max() - pol @ util)


def movement(pol_a: np.ndarray, pol_b: np.ndarray) -> float:
    return float(np.abs(pol_a - pol_b).sum() / 2.0)


def wrong_direction(pol_before: np.ndarray, pol_after: np.ndarray, outcomes: np.ndarray,
                    c_maker: np.ndarray) -> bool:
    """Did the update move the policy AWAY from the maker's preference direction?"""
    util = outcomes @ np.log(np.maximum(c_maker, _EPS))
    return bool(pol_after @ util < pol_before @ util - 1e-12)
