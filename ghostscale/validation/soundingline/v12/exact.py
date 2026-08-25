"""Exact inference for V12: posteriors over maker profiles and regimes, and exact expected
information gain of probes (spec §4.1). This is the reference path; the PyMDP reader is compared
against it, never averaged with it.
"""
from __future__ import annotations

import numpy as np

from ....v11.maker import poe
from .world import REGIMES, World, realization

_EPS = 1e-300


def reader_emission(world: World, template: np.ndarray, habit: np.ndarray | None, w: np.ndarray,
                    domain: int, tier: str, regime_assumption: str, profile: str,
                    construction: str = "A") -> np.ndarray:
    """The emission distribution a reader with ``template`` predicts for a maker of profile
    ``profile`` (weighting ``w``) under a regime assumption. Under construction A this is a
    goal mixture; the regime assumption sets which tail cue is expected."""
    a = world.alpha[tier]
    dom = world.domains[domain]
    if construction == "B":
        base = poe(template, w)
        if habit is not None:
            base = base * habit
        base = base / base.sum()
        d = a * base + (1 - a) * world.synth
        return dom.to_surface(d / d.sum())
    n_slots = len(world.family_names)
    out = np.zeros(world.nf)
    for g in range(world.ng):
        base = template[g] * (habit if habit is not None else 1.0)
        base = base / base.sum()
        if regime_assumption == "bard":
            e = realization(world, base, g, world.cue_of[profile])
        elif regime_assumption == "concealer":
            e = realization(world, base, g, world.cue_of[world.decoy_of[profile]])
        elif regime_assumption == "neutral":
            e = np.mean([realization(world, base, g, s) for s in range(n_slots)], axis=0)
        elif regime_assumption == "plain":
            e = base
        else:
            raise ValueError(regime_assumption)
        out += w[g] * e
    d = a * out + (1 - a) * world.synth
    return dom.to_surface(d / d.sum())


def goal_loglik(world: World, template: np.ndarray, habit, feats: np.ndarray, domain: int,
                tier: str, regime_assumption: str, profile: str) -> np.ndarray:
    """log P(features | goal g) for each g under the reader's own template (construction A),
    for use where the profile enters only through the goal draw."""
    a = world.alpha[tier]
    dom = world.domains[domain]
    n_slots = len(world.family_names)
    out = np.empty(world.ng)
    for g in range(world.ng):
        base = template[g] * (habit if habit is not None else 1.0)
        base = base / base.sum()
        if regime_assumption == "bard":
            e = realization(world, base, g, world.cue_of[profile])
        elif regime_assumption == "concealer":
            e = realization(world, base, g, world.cue_of[world.decoy_of[profile]])
        elif regime_assumption == "neutral":
            e = np.mean([realization(world, base, g, s) for s in range(n_slots)], axis=0)
        else:
            e = base
        d = a * e + (1 - a) * world.synth
        d = dom.to_surface(d / d.sum())
        out[g] = float(np.log(np.maximum(d[feats], _EPS)).sum())
    return out


def profile_loglik_cumulative(world: World, template: np.ndarray, habit, artifacts: list,
                              tier: str, regime_assumption: str = "plain",
                              family: dict | None = None, construction: str = "A") -> dict:
    """name -> cumulative log-likelihood after each artifact. Construction A marginalises the
    per-artifact goal draw: log sum_g w[g] P(a | g)."""
    fam = family if family is not None else world.family
    out = {}
    for name, w in fam.items():
        ll = []
        for art in artifacts:
            feats = np.asarray(art["features"])
            if construction == "B":
                d = reader_emission(world, template, habit, w, art["domain"], tier,
                                    regime_assumption, name, construction="B")
                ll.append(float(np.log(np.maximum(d[feats], _EPS)).sum()))
            else:
                gl = goal_loglik(world, template, habit, feats, art["domain"], tier,
                                 regime_assumption, name)
                m = gl.max()
                ll.append(float(np.log(np.maximum((np.exp(gl - m) * w).sum(), _EPS)) + m))
        out[name] = np.cumsum(ll) if ll else np.zeros(0)
    return out


def posterior(cumlik: dict, at: int, prior: dict | None = None) -> dict:
    names = list(cumlik)
    v = np.array([cumlik[n][at - 1] if at > 0 else 0.0 for n in names])
    if prior is not None:
        v = v + np.log(np.maximum(np.array([prior.get(n, 0.0) for n in names]), _EPS))
    v = np.exp(v - v.max())
    v = v / v.sum()
    return dict(zip(names, v))


def joint_profile_regime_posterior(world: World, template, habit, artifacts, tier,
                                   prior_profile: dict, prior_regime: dict) -> dict:
    """Posterior over (profile, regime) pairs when the regime is uncertain (B02/B03)."""
    out = {}
    for r, pr in prior_regime.items():
        cum = profile_loglik_cumulative(world, template, habit, artifacts, tier, r)
        for name, arr in cum.items():
            ll = float(arr[-1]) if arr.size else 0.0
            out[(name, r)] = ll + np.log(max(prior_profile.get(name, 0.0), 1e-300)) \
                + np.log(max(pr, 1e-300))
    m = max(out.values())
    z = {k: np.exp(v - m) for k, v in out.items()}
    s = sum(z.values())
    return {k: v / s for k, v in z.items()}


def marginal(post_joint: dict, axis: int) -> dict:
    out = {}
    for key, p in post_joint.items():
        out[key[axis]] = out.get(key[axis], 0.0) + p
    return out


def entropy(p: dict) -> float:
    v = np.array([x for x in p.values() if x > 0])
    return float(-(v * np.log(v)).sum())


def expected_information_gain(prior: dict, emissions: dict, n_steps: int, rng,
                              draws: int = 200) -> float:
    """Monte-Carlo EIG (nats) about the hypothesis index from an artifact of ``n_steps`` draws,
    where ``emissions[name]`` is the predicted surface distribution under each hypothesis."""
    names = list(prior)
    p = np.array([prior[n] for n in names])
    E = np.stack([emissions[n] for n in names])
    H0 = -(p[p > 0] * np.log(p[p > 0])).sum()
    posts = []
    for _ in range(int(draws)):
        h = rng.choice(len(names), p=p)
        feats = rng.choice(E.shape[1], size=int(n_steps), p=E[h])
        ll = np.log(np.maximum(E[:, feats], _EPS)).sum(axis=1) + np.log(np.maximum(p, _EPS))
        q = np.exp(ll - ll.max())
        q = q / q.sum()
        posts.append(-(q[q > 0] * np.log(q[q > 0])).sum())
    return float(H0 - np.mean(posts))


def predictive_next_goal(post: dict, family: dict) -> np.ndarray:
    """P(next goal) under the profile posterior: the hidden-continuation predictor."""
    w = np.zeros(len(next(iter(family.values()))))
    for name, p in post.items():
        w += p * family[name]
    return w / w.sum()
