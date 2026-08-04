"""The E36 rollout, reimplemented once so five modules cannot drift apart on it.

E36'S SEEDS, VERBATIM. ``base = 30_000 + mu * 100 + int(beta * 100)`` is lifted from
``v6/e36_process.py``. That is what makes a result here comparable to E36's rather than merely
similar to it, and ``fidelity_check`` asserts the comparison holds before anything else is scored.
"""
from __future__ import annotations

import numpy as np

from ... import metrics
from ... import v6_model as V6
from ...config import Config
from ...v5_model import make_v5_observer
from ...v6 import harness as H
from ...v6.e36_process import BETA_GRID, MU_GRID, RESOLVED_ENTROPY

_EPS = 1e-12


def build(cfg: Config):
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    return H.build_world_and_config(cfg)


def e36_seed(mu: int, beta: float) -> int:
    return 30_000 + int(mu) * 100 + int(float(beta) * 100)


def rollouts(cfg, mus=None, betas=None, n_obs: int = 120, n_timesteps: int = 24,
             forced_k: int = 24):
    """Yield one E36 encounter at a time, with the settling step already located."""
    world, _cfg_b, cfg_r, n_mu, n_sub, ng = build(cfg)
    for mu in (MU_GRID if mus is None else mus):
        for beta in (BETA_GRID if betas is None else betas):
            base = e36_seed(mu, beta)
            for i in range(int(n_obs)):
                art_rng = np.random.default_rng(base * 31 + i)
                g_true = int(art_rng.integers(ng))
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, g_true, int(mu), float(beta), n_timesteps, art_rng)
                agent = make_v5_observer(world, np.random.default_rng(base * 7907 + i))
                enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                      np.random.default_rng(base * 7907 + i),
                                      n_timesteps, forced_k, n_sub, n_mu, ng,
                                      float(world.cfg.signal_model.kappa))
                ents = [float(metrics.within_observer_entropy(p))
                        for p in enc.goal_posteriors_by_step]
                settled = next((t for t, h in enumerate(ents) if h <= RESOLVED_ENTROPY), None)
                yield {"mu": int(mu), "beta": float(beta), "i": i, "g_true": g_true,
                       "enc": enc, "ents": ents, "settled": settled, "n_sub": n_sub,
                       "world": world, "cfg_r": cfg_r, "n_mu": n_mu, "ng": ng}


def usable(rec) -> bool:
    """E36's own admission rule: the split must leave a window on both sides."""
    s, n = rec["settled"], len(rec["ents"])
    return s is not None and 2 <= s <= n - 3


def process_gain(enc, split: int, n_sub: int) -> float:
    before = V6.process_recovery(enc.subgoal_posteriors[:split],
                                 enc.true_modes[:split], n_sub)["process_error_reduction"]
    after = V6.process_recovery(enc.subgoal_posteriors[split:],
                                enc.true_modes[split:], n_sub)["process_error_reduction"]
    return float(after) - float(before)


def resolved_steps(enc, split: int, n_sub: int, threshold: float) -> tuple:
    """SOUNDING LINE'S STATISTIC, MAPPED ONTO THIS READER.

    Their ``unlock`` counts decisions recovered after the purpose settles over decisions
    recovered before. Nothing in that touches the truth: a decision is 'recovered' when their
    extractor emits one. The nearest honest analogue here is a sub-goal posterior that has
    RESOLVED -- entropy below a threshold -- whether or not it resolved onto the right mode.

    Returned as densities rather than raw counts, because the two windows are different lengths
    and a raw count ratio would be a ratio of window sizes.
    """
    hmax = float(np.log(max(n_sub, 2)))
    ents = [float(metrics.within_observer_entropy(np.asarray(q, dtype=float)))
            for q in enc.subgoal_posteriors]
    lim = threshold * hmax
    before = ents[:split]
    after = ents[split:]
    d_before = float(np.mean([e <= lim for e in before])) if before else float("nan")
    d_after = float(np.mean([e <= lim for e in after])) if after else float("nan")
    return d_before, d_after


def concentration(enc, split: int, n_sub: int) -> tuple:
    """A THRESHOLD-FREE VERSION OF THE SAME IDEA, because the thresholded one nearly always
    divides by zero here. Mean (1 - H/Hmax) over each window: how concentrated the reader's
    process belief is, on a scale where 0 is 'no idea' and 1 is 'certain'. Still consults no
    truth, so it is still Sounding Line's kind of statistic rather than E36's."""
    hmax = float(np.log(max(n_sub, 2)))
    conc = [1.0 - float(metrics.within_observer_entropy(np.asarray(q, dtype=float))) / hmax
            for q in enc.subgoal_posteriors]
    before, after = conc[:split], conc[split:]
    return (float(np.mean(before)) if before else float("nan"),
            float(np.mean(after)) if after else float("nan"))


def ratio(before: float, after: float) -> float:
    """Their ratio, with their divide-by-zero problem made explicit rather than hidden."""
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    if before <= _EPS:
        return float("nan") if after <= _EPS else float("inf")
    return float(after / before)
