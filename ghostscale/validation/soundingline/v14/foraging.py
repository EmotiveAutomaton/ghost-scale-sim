"""Epistemic foraging (spec §3, trunk F): items with planted novelty, complexity, compressibility,
learnable error, learning progress, relevance and cost; policies that pick what to observe next;
and realized held-out prediction gain as the only score.

The trap the spec requires: an UNLEARNABLE item (uniform random tokens) stays surprising forever
and must lose to a STRUCTURED item whose error falls with exposure. A curiosity ruler that
follows raw surprise is caught here.
"""
from __future__ import annotations

import numpy as np

from . import common as C

N_TOK = 8
KINDS = ("novel_explained", "familiar_unresolved", "complex_compressible", "simple_unresolved", "unlearnable_noise", "structured_learnable")


def make_item(rng: np.random.Generator, kind: str, cost: float = 1.0, relevance: float = 1.0) -> dict:
    """An item is a token-generating process. Structured kinds have a fixed categorical parameter
    (learnable); noise is uniform (unlearnable); ``complexity`` is the entropy of the generator."""
    if kind == "unlearnable_noise":
        p = np.full(N_TOK, 1.0 / N_TOK)
        prior_exposure = 8                                                # familiar noise: the learner already knows it is flat, and it stays maximally surprising
    elif kind == "novel_explained":
        p = np.zeros(N_TOK); p[int(rng.integers(N_TOK))] = 1.0        # deterministic: explained on first look
        prior_exposure = 0
    elif kind == "familiar_unresolved":
        p = rng.dirichlet(np.full(N_TOK, 0.5))
        prior_exposure = 6                                                # seen before, still uncertain
    elif kind == "complex_compressible":
        p = rng.dirichlet(np.full(N_TOK, 0.3))                             # high-entropy-looking but a fixed law
        prior_exposure = 0
    elif kind == "simple_unresolved":
        p = np.zeros(N_TOK); p[[0, 1]] = [0.5, 0.5]
        prior_exposure = 1
    else:                                                                 # structured_learnable
        p = np.zeros(N_TOK); p[int(rng.integers(N_TOK))] = 0.85; p += 0.15 / N_TOK; p /= p.sum()
        prior_exposure = 0
    counts0 = np.ones(N_TOK) * 0.5 + (p * prior_exposure * (0.5 if kind == "familiar_unresolved" else 4) if prior_exposure else 0.0)
    return {"kind": kind, "p": p, "cost": float(cost), "relevance": float(relevance), "complexity": C.entropy(p),
            "uncertainty": [float(C.entropy(counts0 / counts0.sum()))],       # the learner's own expected surprise, before any look
            # evidence per exposure: an unresolved familiar item was seen often but its law did not settle
            "counts": np.ones(N_TOK) * 0.5 + (p * prior_exposure * (0.5 if kind == "familiar_unresolved" else 4) if prior_exposure else 0.0), "exposures": prior_exposure,
            "novelty": 1.0 / (1.0 + prior_exposure),
            "resolve_weight": 200.0 if kind == "novel_explained" else 1.0}     # an explained item: one look settles its law


def predictive(item: dict) -> np.ndarray:
    return item["counts"] / item["counts"].sum()


def observe(item: dict, rng: np.random.Generator) -> int:
    q = predictive(item)
    tok = int(rng.choice(N_TOK, p=item["p"]))
    item.setdefault("errors", []).append(float(-np.log(q[tok] + 1e-12)))     # the learner's own surprise, recorded before the update
    item["counts"][tok] += float(item.get("resolve_weight", 1.0))
    item.setdefault("uncertainty", []).append(float(C.entropy(predictive(item))))   # and its expected surprise after it
    item["exposures"] += 1
    item["novelty"] = 1.0 / (1.0 + item["exposures"])
    return tok


def current_error(item: dict) -> float:
    """The learner's OWN expected surprise of the next token: the entropy of its predictive. The
    true generator is never consulted by a policy (it is used only to score realized gain)."""
    return float(C.entropy(predictive(item)))


def true_error(item: dict) -> float:
    """Evaluation only: cross-entropy of the true generator under the learner's predictive."""
    q = predictive(item)
    return float(-(item["p"] * np.log(q + 1e-12)).sum())


def reducible_error(item: dict) -> float:
    """Evaluation only: current true error minus the generator's own entropy."""
    return float(max(0.0, true_error(item) - item["complexity"]))


def expected_learning_progress(item: dict, rng: np.random.Generator, k: int = 3, optimism: float = 0.0) -> float:
    """Learning progress (Oudeyer-style) on the learner's OWN expected surprise: the recent drop of
    its predictive entropy over this item's last looks (a window of up to k looks each side,
    shorter while the record is short). Token-by-token surprise is loud for a sharp law and
    noisy for a rich one; the learner's expected surprise is smooth, and it stays flat for noise
    the learner already knows to be flat. Nothing here consults the true generator."""
    u = item.get("uncertainty", [])
    if len(u) < 2:
        return optimism
    kk = max(1, min(k, len(u) // 2))
    return float(max(0.0, np.mean(u[-2 * kk:-kk]) - np.mean(u[-kk:])))


def expected_information_gain(item: dict, rng: np.random.Generator, draws: int = 8) -> float:
    """Expected information gain of one observation about the item's law: the expected KL
    between the updated and the current predictive under the current predictive (Bayesian
    surprise in expectation). It falls with accumulated evidence and is not larger for a
    peaked belief, which predictive-entropy reduction wrongly is."""
    a = item["counts"]
    q = a / a.sum()
    gain = 0.0
    for tok in range(N_TOK):                                     # one more token, at unit weight: what the next look could teach
        c2 = a.copy(); c2[tok] += 1.0
        gain += q[tok] * C.kl(c2 / c2.sum(), q)
    return float(max(0.0, gain))


ABSTAIN_FLOOR = {"learning_progress": 0.005, "eig_per_cost": 0.005}


def choose(items: list, policy: str, rng: np.random.Generator) -> int:
    """The item to observe next, or -1 when the policy declines to act (nothing clears its floor)."""
    if policy == "random":
        return int(rng.integers(len(items)))
    if policy == "novelty":
        score = [it["novelty"] for it in items]
    elif policy == "complexity":
        score = [it["complexity"] for it in items]
    elif policy == "surprise":
        score = [current_error(it) for it in items]
    elif policy == "learning_progress":
        score = [expected_learning_progress(it, rng) / it["cost"] for it in items]
    elif policy == "eig_per_cost":                              # what the look teaches, weighted by declared relevance, per cost
        score = [expected_information_gain(it, rng) * it["relevance"] * it["relevance" ] / it["cost" ] for it in items ]
    elif policy == "always_forensic":
        score = [it["cost"] for it in items]                              # buys the dearest item
    else:
        raise KeyError(policy)
    score = np.asarray(score, dtype=float)
    best = int(np.argmax(score))
    if policy in ABSTAIN_FLOOR and float(score[best]) < ABSTAIN_FLOOR[policy]:
        return -1
    if policy in ("learning_progress", "surprise", "eig_per_cost") and score.sum() > 0:
        # proportional allocation (IAC-style): a greedy argmax hammers one item past its returns
        return int(rng.choice(len(items), p=np.clip(score, 0, None) / np.clip(score, 0, None).sum()))
    return best


def realized_gain(items: list, before: list, holdout_draws: int = 64, rng=None) -> float:
    """Held-out prediction gain summed over items: log score under the current predictive minus the
    score under the predictive before foraging, on tokens from the true generators."""
    total = 0.0
    for it, q0 in zip(items, before):
        q1 = predictive(it)
        total += float((it["p"] * (np.log(q1 + 1e-12) - np.log(q0 + 1e-12))).sum()) * it["relevance"]
    return total


def forage(items: list, policy: str, budget: float, rng: np.random.Generator) -> dict:
    before = [predictive(it).copy() for it in items]
    spent, picks = 0.0, []
    while True:
        i = choose(items, policy, rng)
        if i < 0 or spent + items[i]["cost"] > budget:
            break
        observe(items[i], rng)
        spent += items[i]["cost"]
        picks.append(i)
    return {"gain": realized_gain(items, before), "spent": spent, "picks": picks,
            "gain_per_cost": realized_gain(items, before) / max(spent, 1e-9)}
