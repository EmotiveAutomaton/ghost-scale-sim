"""Projection and target-specific correction (spec §1.3 operations 1 and 4; trunk P).

A cheap local prior (self, equal-local, generic local) is corrected by target evidence. This
module supplies the correction rulers (half-life, residual bias, order effect), the evidence
types that may or may not correct (behaviour, process record, biography, group label, stated
preference, source history, each with a truth flag and a reliability the reader assigns), the
bounded readers (fast, deliberative, compute-matched, accuracy- and confidence-rewarded), the
robust reader that separates an outlier from a regime change, feedback learning across targets,
and reader ensembles with independent or correlated errors.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from .exact import Model
from .goals_trust import note_loglik

EVIDENCE_TYPES = ("behaviour", "process_record", "biography", "group_label", "stated_preference", "source_history")
_EPS = 1e-12


def correction_curve(model: Model, prior: np.ndarray, arts: list, truth_index: int, self_index: int | None,
                     channels=("surface",)) -> dict:
    """Posterior on the truth after each artifact, the half-life of the correction (artifacts
    until the truth posterior reaches half its final value), residual mass on the reader's own
    profile, and the order effect (final posterior under reversed order)."""
    cum = model.cumulative(prior, arts, channels)
    truth = cum[:, truth_index]
    final = truth[-1]
    hl = next((i for i in range(1, len(truth)) if truth[i] >= 0.5 * final), len(arts)) if final > truth[0] else len(arts)
    rev = model.cumulative(prior, arts[::-1], channels)
    self_mass = float(cum[-1, self_index]) if self_index is not None and self_index != truth_index else float("nan")
    return {"truth_trajectory": truth.tolist(), "half_life": int(hl), "final_truth": float(final),
            "residual_self_mass": self_mass, "order_effect": float(abs(rev[-1, truth_index] - final)),
            "confidence_final": float(cum[-1].max())}


def evidence_loglik(model: Model, kind: str, item, reliability: float, truth_maker=None) -> np.ndarray:
    """A non-behavioural piece of evidence as a log-likelihood over hypotheses. ``item`` is the
    asserted value (group id, profile name, or profile vector); the reader assigns the note a
    reliability and never treats it as truth (spec P02, G13)."""
    if kind == "group_label":
        vals = np.array([h.group for h in model.hyps])
        return note_loglik(int(item), vals, reliability)
    if kind == "biography":
        vals = np.array([h.profile for h in model.hyps])
        return note_loglik(str(item), vals, reliability)
    if kind == "stated_preference":
        w = np.asarray(item, float)
        d = np.array([C.js(w, h.w) for h in model.hyps])
        return reliability * (-6.0 * d) + (1 - reliability) * 0.0
    if kind == "source_history":
        # a history of past artifacts already scored: item is a posterior over hypotheses
        q = np.asarray(item, float)
        return reliability * np.log(np.maximum(q, _EPS)) + (1 - reliability) * np.log(1.0 / q.size)
    raise ValueError(kind)


def bounded_reader(model: Model, prior: np.ndarray, arts: list, channels, mode: str, rng=None,
                   compute_budget: int | None = None) -> np.ndarray:
    """Readers under incentive and compute regimes (P06). fast: reads a compute-limited prefix and
    stops; deliberative: everything; compute_matched: the same number of artifact reads as
    fast but chosen for surprise; accuracy_rewarded: deliberative; confidence_rewarded:
    deliberative then sharpened (tempered toward its mode)."""
    n = len(arts)
    if mode == "fast":
        k = compute_budget or max(1, n // 4)
        return model.posterior(prior, arts[:k], channels)
    if mode == "compute_matched":
        k = compute_budget or max(1, n // 4)
        q = np.asarray(prior, float)
        picks = []
        for t, a in enumerate(arts):
            if len(picks) >= k:
                break
            pred = model.predictive_surface(q, a["family"], a["domain"])
            s = -float(np.log(np.maximum(pred[np.asarray(a["features"])], _EPS)).mean())
            if s > 1.9 or len(picks) < 1:
                picks.append(a)
                q = model.posterior(q, [a], channels)
        return q
    q = model.posterior(prior, arts, channels)
    if mode == "confidence_rewarded":
        return C.softmax(3.0 * np.log(np.maximum(q, _EPS)))
    return q


def robust_read(model: Model, prior: np.ndarray, arts: list, channels, hazard: float = 0.05,
                mode: str = "robust") -> dict:
    """Outlier versus regime change (P08). robust: a change-point mixture where each artifact
    may open a new maker regime with probability ``hazard``; reset: restart from the prior on
    any surprising artifact; anchor: never let the posterior move more than a cap from the
    prior. Returns the final posterior and the estimated change point."""
    K = model.K
    if mode == "robust":
        # a single optional change point: artifacts before t0 from one maker hypothesis, after t0
        # from another; the marginal likelihood covers the WHOLE stream under both segments
        L = model.loglik(arts, channels)
        n = len(arts)
        lp = np.log(np.maximum(prior, _EPS))
        best, best_lp, cp = None, -np.inf, 0
        for t0 in range(n):
            before = C.logsumexp(lp + L[:t0].sum(axis=0)) if t0 > 0 else 0.0
            v = lp + L[t0:].sum(axis=0)
            change_prior = np.log(1 - hazard) * (n - 1) if t0 == 0 else np.log(hazard) + np.log(1 - hazard) * (n - 2)
            marg = before + C.logsumexp(v) + change_prior
            if marg > best_lp:
                best_lp, best, cp = marg, C.softmax(v), t0
        return {"posterior": best, "change_point": int(cp)}
    if mode == "reset":
        q = np.asarray(prior, float)
        base_pred = None
        resets = 0
        for a in arts:
            pred = model.predictive_surface(q, a["family"], a["domain"])
            base_pred = model.predictive_surface(np.asarray(prior, float), a["family"], a["domain"])
            f = np.asarray(a["features"])
            s = float(np.log(np.maximum(base_pred[f], _EPS)).mean() - np.log(np.maximum(pred[f], _EPS)).mean())
            if s > 0.7:                                     # surprising relative to knowing nothing: reset
                q = np.asarray(prior, float)
                resets += 1
            q = model.posterior(q, [a], channels)
        return {"posterior": q, "change_point": resets}
    if mode == "anchor":
        q = model.posterior(prior, arts, channels)
        cap = 0.6
        return {"posterior": C.normalize((1 - cap) * np.asarray(prior, float) + cap * q), "change_point": 0}
    raise ValueError(mode)


def feedback_weight_update(alpha: float, outcome_correct: bool | None, lr: float = 0.15) -> float:
    """Meta-learning of the self-prior weight from outcome feedback (P04): the mixture weight
    on the local prior rises after a correct prediction and falls after an incorrect one."""
    if outcome_correct is None:
        return alpha
    return float(np.clip(alpha + lr * (1.0 if outcome_correct else -1.0), 0.05, 0.95))


def mixed_prior(local: np.ndarray, population: np.ndarray, alpha: float) -> np.ndarray:
    return C.normalize(alpha * np.asarray(local, float) + (1 - alpha) * np.asarray(population, float))


def ensemble(posteriors: list, method: str, weights=None) -> np.ndarray:
    """Vote: average of one-hot argmaxes (smoothed); bayes: product of posteriors renormalised
    (the independence assumption); mean: linear pool."""
    P = np.stack(posteriors)
    if method == "vote":
        v = np.zeros(P.shape[1]) + 0.1
        for p in P:
            v[int(np.argmax(p))] += 1.0
        return C.normalize(v)
    if method == "bayes":
        return C.softmax(np.log(np.maximum(P, _EPS)).sum(axis=0))
    if method == "mean":
        w = np.ones(P.shape[0]) if weights is None else np.asarray(weights, float)
        return C.normalize((w[:, None] * P).sum(axis=0))
    raise ValueError(method)


def equifinal_pair(model: Model, prior: np.ndarray, arts_a: list, arts_b: list, channels) -> dict:
    """Two histories with identical artifacts must give identical posteriors (abstention is the
    spread of that posterior); a later separating artifact must move them apart (P14)."""
    qa = model.posterior(prior, arts_a, channels)
    qb = model.posterior(prior, arts_b, channels)
    return {"identical_posteriors": float(np.abs(qa - qb).max()), "abstention_entropy": C.entropy(qa),
            "top_mass": float(qa.max())}
