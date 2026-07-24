"""Metrics (Spec §6), each with its formal definition in the docstring.

All information-theoretic quantities are in **nats** (natural log) unless stated.
The low-level helpers (entropy, KL, JS, normalize) are reused by
``generative_model.py`` for its construction-time assertions.

Expected-free-energy (EFE) decomposition uses the pymdp legacy control primitives so
that the pragmatic/epistemic terms we report are exactly the ones the agent scores
policies with:

    pragmatic term  = calc_expected_utility(qo_pi, C)      (higher = more preferred)
    epistemic term  = calc_states_info_gain(A, qs_pi)      (expected info gain about states)

In addition we provide a *factor-restricted* epistemic value about ``creator_goal``
(factor 1) — this is the quantity Spec §6 names as ``epistemic_value`` ("expected
information gain about factor 1"), and the one whose collapse to zero for GHOST content
is the E1 claim.
"""
from __future__ import annotations

import numpy as np

from pymdp.legacy.control import (
    get_expected_states,
    get_expected_obs,
    calc_expected_utility,
    calc_states_info_gain,
)

from .constants import F_GOAL, M_FEATURES

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Low-level information-theoretic helpers (nats).
# --------------------------------------------------------------------------- #
def normalize(v: np.ndarray) -> np.ndarray:
    """Return v / sum(v) along the last axis; safe for zero vectors."""
    v = np.asarray(v, dtype=float)
    s = v.sum(axis=-1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return v / s


def shannon_entropy(p: np.ndarray) -> float:
    """H(p) = -Σ p log p, in nats."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = Σ p log(p/q), in nats."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log((p[mask]) / (q[mask] + _EPS))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, symmetric, in nats. Range [0, ln 2]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# --------------------------------------------------------------------------- #
# EFE decomposition per policy (Spec §6, E1).
# --------------------------------------------------------------------------- #
def policy_efe_terms(agent, policy) -> tuple[float, float]:
    """Return (pragmatic_value, epistemic_value_total) for a policy from the agent's
    current beliefs ``agent.qs``.

    pragmatic = expected utility of the policy's expected observations under C.
    epistemic = expected information gain about *all* hidden states (the salience term
                the agent actually uses in its EFE).
    """
    qs_pi = get_expected_states(agent.qs, agent.B, policy)
    qo_pi = get_expected_obs(qs_pi, agent.A)
    pragmatic = float(calc_expected_utility(qo_pi, agent.C))
    epistemic_total = float(calc_states_info_gain(agent.A, qs_pi))
    return pragmatic, epistemic_total


def pragmatic_value(agent, policy) -> float:
    """Expected utility (pragmatic term of EFE) of ``policy`` under the agent's C."""
    return policy_efe_terms(agent, policy)[0]


def epistemic_value(agent, policy, factor: int = F_GOAL) -> float:
    """Expected information gain about ``factor`` (default: creator_goal, factor 1)
    accumulated over the policy horizon — Spec §6's ``epistemic_value``.

    Computed exactly as the mutual information I(s_factor ; o) under the policy's
    predictive model, summed over the policy's timesteps:

        Σ_t  [ H(q(s_factor)_t) - E_{o~q(o)_t}[ H(q(s_factor | o)_t) ] ]

    For GHOST content under DEEP this collapses toward zero: the features carry (almost)
    no dependence on the goal, so observing them does not reduce goal uncertainty.
    """
    qs_pi = get_expected_states(agent.qs, agent.B, policy)
    total = 0.0
    for qs_t in qs_pi:
        total += _expected_info_gain_about_factor(qs_t, agent.A, factor)
    return float(total)


def _expected_info_gain_about_factor(qs, A, factor: int) -> float:
    """I(s_factor ; o) for one predicted state marginal set ``qs`` and likelihood ``A``.

    Enumerates the (small) joint observation space; exact for this model's cardinalities.
    """
    num_states = [np.asarray(q).size for q in qs]
    # Joint prior over hidden states P(s) as an outer product of the factor marginals.
    prior = np.asarray(qs[0], dtype=float)
    for q in qs[1:]:
        prior = np.multiply.outer(prior, np.asarray(q, dtype=float))
    prior_flat = prior.ravel()

    # Prior marginal entropy over the target factor.
    prior_marg = _marginal(prior, factor, num_states)
    h_prior = shannon_entropy(prior_marg)

    # Flatten each modality's likelihood to (num_obs_m, S).
    S = prior_flat.size
    A_flat = [np.asarray(A[m], dtype=float).reshape(np.asarray(A[m]).shape[0], S)
              for m in range(len(A))]

    exp_post_entropy = 0.0
    # Enumerate joint observations; modalities are conditionally independent given s.
    from itertools import product as _product
    obs_ranges = [range(af.shape[0]) for af in A_flat]
    for obs_tuple in _product(*obs_ranges):
        like = np.ones(S)
        for m, o in enumerate(obs_tuple):
            like = like * A_flat[m][o]
        joint = like * prior_flat
        p_o = joint.sum()
        if p_o <= _EPS:
            continue
        post_flat = joint / p_o
        post = post_flat.reshape(num_states)
        post_marg = _marginal(post, factor, num_states)
        exp_post_entropy += p_o * shannon_entropy(post_marg)

    return max(h_prior - exp_post_entropy, 0.0)


def _marginal(joint: np.ndarray, factor: int, num_states: list[int]) -> np.ndarray:
    joint = joint.reshape(num_states)
    axes = tuple(i for i in range(len(num_states)) if i != factor)
    return joint.sum(axis=axes)


# --------------------------------------------------------------------------- #
# Between/within observer (Spec §6, E2 — the hallucination signature).
# --------------------------------------------------------------------------- #
def between_observer_entropy(posteriors: list[np.ndarray]) -> float:
    """Entropy (nats) of the distribution of *modal* inferred goals across observers.

    This is inter-rater reliability: each observer contributes its argmax goal; we form
    the empirical distribution over goals across the population and take its Shannon
    entropy. High value = observers disagree about the goal.
    """
    if len(posteriors) == 0:
        return 0.0
    num_goals = np.asarray(posteriors[0]).size
    modal = np.array([int(np.argmax(p)) for p in posteriors])
    counts = np.bincount(modal, minlength=num_goals).astype(float)
    return shannon_entropy(counts / counts.sum())


def within_observer_entropy(posterior: np.ndarray) -> float:
    """Shannon entropy (nats) of a single observer's goal posterior.

    Low value = that observer is individually confident about the goal (whether or not
    it is correct).
    """
    return shannon_entropy(np.asarray(posterior, dtype=float))


def mean_within_observer_entropy(posteriors: list[np.ndarray]) -> float:
    """Mean of ``within_observer_entropy`` across observers."""
    if len(posteriors) == 0:
        return 0.0
    return float(np.mean([within_observer_entropy(p) for p in posteriors]))


# --------------------------------------------------------------------------- #
# The unidentifiability-vs-noise diagnostic (Spec §6, §9 N6).
# --------------------------------------------------------------------------- #
def mutual_information_features_goal(A0: np.ndarray, provenance: int, attention: int,
                                     goal_prior: np.ndarray | None = None) -> float:
    """MI (nats) between artifact_features (modality 0) and creator_goal (factor 1),
    conditioned on ``provenance`` and ``attention``.

    Uses P(feature | goal) = A0[:, provenance, :, attention] (shape [num_features,
    num_goals]) and a goal prior (uniform by default):

        I(F; G) = Σ_g P(g) Σ_f P(f|g) log( P(f|g) / P(f) ),   P(f) = Σ_g P(g) P(f|g)

    This is the diagnostic that separates *unidentifiability* from *noise*: the synthetic
    condition must show low MI. Report ``feature_entropy_given_provenance`` alongside it
    — synthetic (structured-unidentifiable) shows low MI **and** low feature entropy;
    the uniform-noise strawman shows low MI **and** high feature entropy.
    """
    P_f_given_g = np.asarray(A0)[:, provenance, :, attention]  # (num_features, num_goals)
    num_goals = P_f_given_g.shape[1]
    if goal_prior is None:
        goal_prior = np.full(num_goals, 1.0 / num_goals)
    goal_prior = np.asarray(goal_prior, dtype=float)
    P_f = P_f_given_g @ goal_prior  # marginal over features
    mi = 0.0
    for g in range(num_goals):
        col = P_f_given_g[:, g]
        mask = col > 0
        mi += goal_prior[g] * float(np.sum(col[mask] * np.log(col[mask] / (P_f[mask] + _EPS))))
    return float(mi)


def feature_entropy_given_provenance(A0: np.ndarray, provenance: int, attention: int,
                                     goal_prior: np.ndarray | None = None) -> float:
    """H(features | provenance) (nats): entropy of the goal-marginalized feature
    distribution P(f) = Σ_g P(g) P(f | g) at this provenance/attention.

    Paired with ``mutual_information_features_goal`` to distinguish structured-
    unidentifiable content (low H) from noise (high H)."""
    P_f_given_g = np.asarray(A0)[:, provenance, :, attention]
    num_goals = P_f_given_g.shape[1]
    if goal_prior is None:
        goal_prior = np.full(num_goals, 1.0 / num_goals)
    P_f = P_f_given_g @ np.asarray(goal_prior, dtype=float)
    return shannon_entropy(P_f)


# --------------------------------------------------------------------------- #
# Psi analogue (Spec §6). Explicitly a reimplementation of the intuition, NOT a
# port of the closed-form Ψ equation. Documented as such in the README.
# --------------------------------------------------------------------------- #
def psi_analogue(posterior: np.ndarray, prior: np.ndarray, kappa: float,
                 engaged: bool) -> float:
    """Discrete stand-in for Ψ:

        psi = [engaged] * (-ln(1 - kappa)) * KL( Q(goal|τ) || P0(goal) )

    The sigmoid gate of the closed-form Ψ is replaced by the binary engagement decision.
    ``kappa`` is clamped below 1 to keep -ln(1-kappa) finite.
    """
    if not engaged:
        return 0.0
    k = min(float(kappa), 1.0 - 1e-6)
    weight = -np.log(1.0 - k)
    return float(weight * kl_divergence(np.asarray(posterior, float), np.asarray(prior, float)))
