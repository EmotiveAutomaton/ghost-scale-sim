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


def between_observer_entropy_corrected(posteriors: list[np.ndarray]) -> float:
    """``between_observer_entropy`` with the Miller-Madow bias correction added.

        H_MM = H_plugin + (m - 1) / 2N        m = goals actually voted for, N = readers

    ADDED BESIDE THE ORIGINAL, NEVER SUBSTITUTED (repair R-4). The plug-in entropy of a count
    vector is biased low by roughly (K-1)/2N nats: negligible at the 4000 readers version 1 ran
    at, 0.025 at 60, 0.099 at 16. Every committed number was produced by the plug-in form, so
    swapping the estimator would change what those numbers mean without changing any file. Both
    are reported everywhere.

    The observed support ``m`` is used rather than the available ``K``, which is what makes this
    a correction rather than a constant offset: a genuinely unanimous population has m = 1 and
    is left alone.
    """
    if len(posteriors) == 0:
        return 0.0
    num_goals = np.asarray(posteriors[0]).size
    modal = np.array([int(np.argmax(p)) for p in posteriors])
    counts = np.bincount(modal, minlength=num_goals).astype(float)
    n = float(counts.sum())
    m = float(np.count_nonzero(counts))
    return float(shannon_entropy(counts / n) + (m - 1.0) / (2.0 * n))


def mean_pairwise_js(posteriors: list[np.ndarray], max_pairs: int = 20000,
                     rng: np.random.Generator | None = None) -> float:
    """Mean Jensen-Shannon divergence between readers' FULL posteriors, in nats (repair R-2).

    THE STATISTIC ``between_observer_entropy`` SHOULD HAVE BEEN PAIRED WITH, and the reason is
    that the modal-goal entropy cannot tell two populations apart that are not remotely alike:

      * readers each CERTAIN OF A DIFFERENT answer, which is the disagreement the framework claims;
      * readers who are ALL EQUALLY LOST and whose single best guess is close to a coin toss.

    Both produce the same count vector and therefore the same entropy. Measured on the committed
    cells, three pairs score within 0.05 of each other on modal-goal entropy while differing by up
    to elevenfold on this. It is near zero when everyone is equally unsure, because their beliefs
    are then nearly identical, and large only when readers genuinely differ.

    Sub-sampled above ``max_pairs`` because the pair count is quadratic: 200 readers is 19,900
    pairs and 4,000 is eight million.
    """
    P = np.asarray([np.asarray(p, dtype=float) for p in posteriors], dtype=float)
    if P.ndim != 2 or P.shape[0] < 2:
        return 0.0
    P = P / P.sum(axis=1, keepdims=True)
    n = P.shape[0]
    if n * (n - 1) // 2 <= max_pairs:
        idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        a, b = rng.integers(0, n, max_pairs), rng.integers(0, n, max_pairs)
        idx = [(int(i), int(j)) for i, j in zip(a, b) if i != j]
    return float(np.mean([js_divergence(P[i], P[j]) for i, j in idx]))


def error_reduction(posterior: np.ndarray, prior: np.ndarray, true_goal: int) -> float:
    """How much closer to the truth the reader got, in nats. Signed (repair R-5).

        error_reduction = KL(truth || prior) - KL(truth || posterior)
                        = log posterior[g*] - log prior[g*]

    THE MEASURE ``psi_analogue`` SHOULD HAVE BEEN, and the difference is not cosmetic.
    ``psi_analogue`` is a DISTANCE, KL(posterior || prior), so it counts any movement as uptake.
    A reader who becomes confidently WRONG has moved just as far from where it started as one who
    became confidently right, and measured on this model it scores 87% as much. That makes the
    reported measure U-shaped in how well the reader actually understood, with a minimum in the
    middle, so any experiment regressing it on a difficulty manipulation whose arms straddle that
    minimum returns a flat result for reasons unconnected to the manipulation.

    This one is monotone in accuracy and goes NEGATIVE when the reader moves away from the truth,
    which is the behaviour the distance cannot express.

    ONE DIRECTION MATTERS AND THE OTHER IS UNDEFINED. Written the other way round, as
    KL(prior || truth) - KL(posterior || truth), every term is infinite: the truth is a point mass
    and KL(p || point mass) diverges for any p with mass elsewhere. Floored at some epsilon it
    returns a finite number that is a function of the epsilon rather than of the data. The form
    above has no such problem and equals the reduction in the surprisal of the truth.
    """
    p = np.asarray(posterior, dtype=float)
    q = np.asarray(prior, dtype=float)
    g = int(true_goal)
    return float(np.log(max(p[g], _EPS)) - np.log(max(q[g], _EPS)))


def trust_factor(kappa: float) -> float:
    """The multiplicative trust weight inside ``psi_analogue``, reported on its own (repair R-5).

    ``psi_analogue`` is this factor times a belief distance. Over the range one experiment sweeps
    the factor alone changes by more than fortyfold, which is larger than most effects in this
    project, so any sweep that varies trust and reports uptake is reporting a product of two
    things. Separated so the two can be read apart.
    """
    return float(-np.log(1.0 - min(float(kappa), 1.0 - 1e-6)))


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
# Calibration (V4.5 A2 / V4 E25).
#
# Regret, accuracy, and the epistemic/pragmatic decomposition already exist. These two are
# the gap, and they are TRANSLATION rather than new information: they restate the
# fabrication result in the vocabulary an ML audience already uses. The observer is not
# merely wrong about the goal, it is MISCALIBRATED about the goal, and V1's E2 shows the
# miscalibration is induced by a LABEL rather than by the content.
#
# Both take the full posterior over goals. Neither is computable from a modal goal and an
# entropy, which is why the three source experiments had to start persisting the posterior;
# see analyses_v4_5 for that deviation.
# --------------------------------------------------------------------------- #
def brier_score(posteriors: np.ndarray, truths: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between the posterior and the one-hot
    outcome, averaged over observers.

        BS = (1/N) Σ_n Σ_c ( p_n[c] - 1[y_n = c] )^2

    Range [0, 2]; lower is better. A uniform posterior over K classes scores 1 - 1/K
    (0.75 at K = 4) regardless of the outcome, which is the reference an honestly-uncertain
    observer earns. Confident-and-wrong approaches 2, and that is the E2 cell.

    This is the STRICTLY PROPER scoring rule of the pair: it rewards an observer for
    reporting its true belief and cannot be gamed by hedging, so a bad Brier score is a
    statement about the belief and not about the reporting convention.
    """
    p = np.asarray(posteriors, dtype=float)
    y = np.asarray(truths, dtype=int)
    if p.ndim == 1:
        p = p[None, :]
    onehot = np.zeros_like(p)
    onehot[np.arange(p.shape[0]), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def expected_calibration_error(posteriors: np.ndarray, truths: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected calibration error on the top-label prediction (the standard formulation).

        ECE = Σ_b (|B_b| / N) · | acc(B_b) - conf(B_b) |

    where observers are binned by their confidence (max posterior mass), ``acc`` is the
    fraction of that bin whose modal goal was correct, and ``conf`` is the bin's mean
    confidence. Range [0, 1]; zero means stated confidence tracks the chance of being right.

    ECE is NOT proper — an observer that always says "25%" and is right 25% of the time
    scores a perfect 0 while knowing nothing — so it is reported ALONGSIDE Brier and never
    instead of it. It is included because it is the number an ML audience reads first, and
    because the specific failure it names is the one the Ghost Scale induces: high
    confidence in a bin whose accuracy is at chance.

    Bins are the equal-width [0,1] bins of the standard definition, so the value is
    comparable to published figures. Empty bins contribute nothing.
    """
    p = np.asarray(posteriors, dtype=float)
    y = np.asarray(truths, dtype=int)
    if p.ndim == 1:
        p = p[None, :]
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    n = p.shape[0]
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for b in range(int(n_bins)):
        lo, hi = edges[b], edges[b + 1]
        # Left-open bins, with the first bin closed on the left so nothing is dropped.
        sel = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        m = int(np.sum(sel))
        if m == 0:
            continue
        ece += (m / n) * abs(float(np.mean(correct[sel])) - float(np.mean(conf[sel])))
    return float(ece)


def calibration_bins(posteriors: np.ndarray, truths: np.ndarray,
                     n_bins: int = 10) -> list[dict]:
    """The reliability-diagram rows behind ``expected_calibration_error``.

    Reported because a single ECE number hides the direction of the error, and the direction
    is the finding: the E2 hallucination cell is overconfident (confidence far above
    accuracy) rather than merely inaccurate.
    """
    p = np.asarray(posteriors, dtype=float)
    y = np.asarray(truths, dtype=int)
    if p.ndim == 1:
        p = p[None, :]
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows = []
    for b in range(int(n_bins)):
        lo, hi = edges[b], edges[b + 1]
        sel = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        m = int(np.sum(sel))
        rows.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "n": m,
            "mean_confidence": float(np.mean(conf[sel])) if m else float("nan"),
            "accuracy": float(np.mean(correct[sel])) if m else float("nan"),
            "gap": float(np.mean(conf[sel]) - np.mean(correct[sel])) if m else float("nan"),
        })
    return rows


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
