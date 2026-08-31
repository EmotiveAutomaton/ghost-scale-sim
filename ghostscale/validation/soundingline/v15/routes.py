"""Route reliability, shared causes and robust transfer (spec §6, trunk R).

V14 found that learned route weighting beat equal weighting by +0.009 nats and went *negative* out
of family. The audit reading was that reliability weighting is weak precisely where the routes are
already strong, so this trunk moves the question to where it can have an answer: reliability
*dispersion* crossed with evidence dose (R01), sparse feedback with no target labels at test (R02),
correlated evidence the reader is not told about (R03), and domain shift (R04, R05).

The failure this module must be able to exhibit
-----------------------------------------------
R06 is the ease trap: a route made cheap and pleasant to read, carrying nothing. A weighting scheme
that chases ease rather than accuracy will load onto it, and -- this is the part that matters -- its
*held-out accuracy record can stay clean* while it does so, because the captured route agrees with
the others on the easy cases. The attack is only meaningful if the trap can actually spring, so the
ease-driven weighter is implemented honestly rather than as a straw rival.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C

WEIGHTERS = ("equal", "learned", "robust", "ease_driven", "oracle_reliability")


@dataclass
class RouteBank:
    """A set of noisy routes reporting on one hidden value."""

    reliability: np.ndarray                 # [n_routes] probability of reporting the truth
    ease: np.ndarray                        # [n_routes] how cheap the route is to read
    cluster: np.ndarray                     # [n_routes] shared-cause group id
    n_values: int = 4
    meta: dict = field(default_factory=dict)

    @property
    def n_routes(self) -> int:
        return int(self.reliability.size)


def sample_bank(rng, n_routes: int = 4, dispersion: float = 0.3, n_values: int = 4,
                duplicated: bool = False, easy_useless: bool = False) -> RouteBank:
    """``dispersion`` is R01's first axis: how unequal the routes' reliabilities are."""
    base = 0.55
    rel = np.clip(base + dispersion * rng.normal(size=n_routes), 0.28, 0.97)
    ease = np.clip(0.5 + 0.2 * rng.normal(size=n_routes), 0.05, 0.95)
    cluster = np.arange(n_routes)
    if duplicated:
        cluster[1] = cluster[0]                 # routes 0 and 1 share a cause
        rel[1] = rel[0]
    if easy_useless:
        rel[-1] = 1.0 / n_values                # says nothing
        ease[-1] = 0.98                         # and is delightful to read
    return RouteBank(reliability=rel, ease=ease, cluster=cluster, n_values=n_values,
                     meta={"dispersion": float(dispersion), "duplicated": bool(duplicated),
                           "easy_useless": bool(easy_useless)})


def emit(bank: RouteBank, truth: int, rng) -> np.ndarray:
    """One report per route. Routes in a shared-cause cluster err *together*."""
    reports = np.zeros(bank.n_routes, int)
    cluster_noise = {}
    for i in range(bank.n_routes):
        c = int(bank.cluster[i])
        if c not in cluster_noise:
            cluster_noise[c] = rng.random() >= bank.reliability[i]
        wrong = cluster_noise[c]
        if wrong:
            alt = [v for v in range(bank.n_values) if v != truth]
            reports[i] = int(rng.choice(alt))
        else:
            reports[i] = int(truth)
    return reports


def fuse(bank: RouteBank, reports: np.ndarray, weights: np.ndarray,
         shared_cause: bool = False, budget=None) -> np.ndarray:
    """Weighted vote over route reports.

    ``shared_cause=True`` divides each cluster's total weight among its members, which is the
    correction R03 asks whether a reader can find *without being told* the correlation graph.
    """
    w = np.asarray(weights, float).copy()
    if shared_cause:
        for c in set(bank.cluster.tolist()):
            idx = np.flatnonzero(bank.cluster == c)
            if idx.size > 1:
                w[idx] = w[idx] / idx.size
    lg = np.zeros(bank.n_values)
    for i, r in enumerate(reports):
        lg[int(r)] += w[i]
    if budget is not None:
        budget.lik(bank.n_routes)
    return C.softmax(lg)


def learn_weights(bank: RouteBank, rng, n_feedback: int = 40, sparsity: float = 0.0,
                  kind: str = "learned", budget=None) -> np.ndarray:
    """Estimate route weights from *predictive feedback*, never from target labels at test.

    ``sparsity`` is the fraction of episodes with no feedback at all (R02). ``robust`` shrinks
    toward equal weights by the spread of the estimate, which is what should pay off under shift
    (R04). ``ease_driven`` is the planted failure: it weights by how cheap a route is to read.
    """
    if kind == "equal":
        return np.ones(bank.n_routes)
    if kind == "oracle_reliability":
        return np.log(np.maximum(bank.reliability, 1e-6) * bank.n_values)
    if kind == "ease_driven":
        return 3.0 * np.asarray(bank.ease, float)

    hits = np.zeros(bank.n_routes)
    seen = np.zeros(bank.n_routes)
    for _ in range(int(n_feedback)):
        truth = int(rng.integers(bank.n_values))
        rep = emit(bank, truth, rng)
        if rng.random() < sparsity:
            continue                                    # no feedback this episode
        for i, r in enumerate(rep):
            seen[i] += 1.0
            hits[i] += float(int(r) == truth)
        if budget is not None:
            budget.obs(1)
    acc = (hits + 1.0) / (seen + float(bank.n_values))
    w = np.log(np.maximum(acc, 1e-6) * bank.n_values)
    if kind == "robust":
        # shrink toward equal weighting by how uncertain each estimate is: minimax-flavoured,
        # and the point of R04 is that this should cost in-domain and pay after a shift
        se = np.sqrt(np.maximum(acc * (1 - acc), 1e-6) / np.maximum(seen, 1.0))
        shrink = 1.0 / (1.0 + 6.0 * se)
        w = w * shrink
    return w


def score_weighter(bank: RouteBank, weights: np.ndarray, rng, n: int = 200,
                   shared_cause: bool = False, budget=None) -> dict:
    """Held-out predictive score of a weighting scheme on the hidden value."""
    ls, acc, conf = [], [], []
    for _ in range(int(n)):
        truth = int(rng.integers(bank.n_values))
        post = fuse(bank, emit(bank, truth, rng), weights, shared_cause, budget)
        ls.append(C.log_score(post, truth))
        acc.append(float(C.top1(post) == truth))
        conf.append(C.confidence(post))
    return {"log_score": float(np.mean(ls)), "accuracy": float(np.mean(acc)),
            "mean_confidence": float(np.mean(conf)),
            "overconfidence": float(np.mean(conf) - np.mean(acc)),
            "calibration": C.calibration_block(conf, acc)}


def detect_shared_cause(bank: RouteBank, rng, n: int = 240) -> dict:
    """R03: recover the correlation structure from co-agreement alone, with no graph supplied."""
    obs = np.array([emit(bank, int(rng.integers(bank.n_values)), rng) for _ in range(int(n))])
    n_r = bank.n_routes
    agree = np.zeros((n_r, n_r))
    for i in range(n_r):
        for j in range(n_r):
            agree[i, j] = float(np.mean(obs[:, i] == obs[:, j]))
    base = 1.0 / bank.n_values
    excess = agree - base
    np.fill_diagonal(excess, 0.0)
    thresh = float(np.quantile(excess[np.triu_indices(n_r, 1)], 0.75)) + 0.12
    found = {(i, j) for i in range(n_r) for j in range(i + 1, n_r) if excess[i, j] > thresh}
    truth = {(i, j) for i in range(n_r) for j in range(i + 1, n_r)
             if bank.cluster[i] == bank.cluster[j]}
    return {"agreement": agree.tolist(), "detected_pairs": sorted(found),
            "true_pairs": sorted(truth),
            "recall": float(len(found & truth) / max(len(truth), 1)) if truth else float("nan"),
            "false_pairs": len(found - truth)}


def domain_shift(bank: RouteBank, rng, size: float = 0.5) -> RouteBank:
    """Move the reliabilities. ``size`` 0 is no shift, 1 reverses the ordering."""
    rel = bank.reliability.copy()
    order = np.argsort(rel)
    shifted = rel.copy()
    shifted[order] = rel[order[::-1]]
    out = (1.0 - size) * rel + size * shifted
    return RouteBank(reliability=np.clip(out, 0.28, 0.97), ease=bank.ease.copy(),
                     cluster=bank.cluster.copy(), n_values=bank.n_values,
                     meta={**bank.meta, "shift": float(size)})


def transfer_policy(bank: RouteBank, rng, shift: float, policy: str, n_feedback: int = 40) -> dict:
    """R05: after a shift, should the reader reset, partially transfer, or retain old weights?"""
    old = learn_weights(bank, rng, n_feedback, kind="learned")
    new_bank = domain_shift(bank, rng, shift)
    if policy == "retain":
        w = old
    elif policy == "reset":
        w = learn_weights(new_bank, rng, max(n_feedback // 4, 4), kind="learned")
    else:                                                    # partial
        fresh = learn_weights(new_bank, rng, max(n_feedback // 4, 4), kind="learned")
        w = 0.5 * old + 0.5 * fresh
    s = score_weighter(new_bank, w, rng)
    return {"policy": policy, "shift": float(shift), **s}


# --------------------------------------------------------------------------- #
# Costly access under prior ambiguity (R07).
# --------------------------------------------------------------------------- #
def expected_information_gain(bank: RouteBank, prior: np.ndarray, route: int,
                              budget=None) -> float:
    """Expected KL from buying one route's report, under a stated prior."""
    rel = float(bank.reliability[route])
    n = bank.n_values
    total = 0.0
    for r in range(n):
        pr = sum(prior[v] * (rel if v == r else (1 - rel) / (n - 1)) for v in range(n))
        if pr <= 0:
            continue
        post = np.array([prior[v] * (rel if v == r else (1 - rel) / (n - 1)) for v in range(n)])
        post = post / post.sum()
        total += pr * C.kl(post, prior)
    if budget is not None:
        budget.lik(n)
    return float(total)


def robust_expected_information_gain(bank: RouteBank, priors: list, route: int,
                                     budget=None) -> float:
    """Worst case over an *ambiguity set* of priors rather than one nominal prior.

    Go & Isaac's setting: when the prior itself is not known, a design that is excellent under the
    nominal prior can be poor under a neighbour. R07 and F04 both turn on whether paying attention
    to that worst case is worth its cost.
    """
    return float(min(expected_information_gain(bank, p, route, budget) for p in priors))


def purchase_policy(bank: RouteBank, rng, policy: str, cost: float, priors: list,
                    n: int = 120, budget=None) -> dict:
    """Five purchase policies compared on realized gain per cost (R07).

    ``never``, ``always``, ``fixed`` (a preselected route), ``eig`` (best under the nominal prior)
    and ``robust_eig`` (best worst-case). The forensic-style route is the expensive one.
    """
    nominal = priors[0]
    gains, costs, ls = [], [], []
    for _ in range(int(n)):
        truth = int(rng.integers(bank.n_values))
        free = [i for i in range(bank.n_routes - 1)]
        rep = emit(bank, truth, rng)
        base = fuse(bank, rep[free], np.ones(len(free)), budget=budget)
        pricey = bank.n_routes - 1
        if policy == "never":
            buy = False
        elif policy == "always":
            buy = True
        elif policy == "fixed":
            buy = True
        elif policy == "eig":
            buy = expected_information_gain(bank, base, pricey, budget) > cost
        else:
            buy = robust_expected_information_gain(bank, [base] + priors[1:], pricey, budget) > cost
        if buy:
            lg = np.log(np.maximum(base, 1e-300))
            rel = float(bank.reliability[pricey])
            for v in range(bank.n_values):
                lg[v] += np.log(rel if v == int(rep[pricey]) else (1 - rel) / (bank.n_values - 1))
            post = C.softmax(lg)
            costs.append(cost)
        else:
            post = base
            costs.append(0.0)
        gains.append(C.log_score(post, truth) - C.log_score(base, truth))
        ls.append(C.log_score(post, truth))
    spent = float(np.sum(costs))
    return {"policy": policy, "mean_log_score": float(np.mean(ls)),
            "mean_gain": float(np.mean(gains)), "total_cost": spent,
            "gain_per_cost": float(np.sum(gains) / spent) if spent > 0 else float("inf"),
            "purchase_rate": float(np.mean(np.asarray(costs) > 0))}


def weights_change_exact_posterior(bank: RouteBank, rng, n: int = 200) -> dict:
    """R08: are route weights a bounded processing shortcut, or do they change the answer?

    With unlimited budget the correctly-weighted fusion and the exact posterior converge, so any
    advantage must be a finite-budget one. Reported as the gap at full budget and the gap at a
    truncated one.
    """
    exact_w = np.log(np.maximum(bank.reliability, 1e-6) * bank.n_values)
    full = score_weighter(bank, exact_w, np.random.default_rng(rng.integers(0, 2 ** 62)), n)
    equal = score_weighter(bank, np.ones(bank.n_routes),
                           np.random.default_rng(rng.integers(0, 2 ** 62)), n)
    trunc = RouteBank(reliability=bank.reliability[:2], ease=bank.ease[:2],
                      cluster=bank.cluster[:2], n_values=bank.n_values)
    tw = np.log(np.maximum(trunc.reliability, 1e-6) * trunc.n_values)
    budgeted = score_weighter(trunc, tw, np.random.default_rng(rng.integers(0, 2 ** 62)), n)
    budget_equal = score_weighter(trunc, np.ones(2),
                                  np.random.default_rng(rng.integers(0, 2 ** 62)), n)
    return {"full_budget_advantage": full["log_score"] - equal["log_score"],
            "finite_budget_advantage": budgeted["log_score"] - budget_equal["log_score"],
            "converges": bool(abs(full["log_score"] - equal["log_score"]) < 0.35)}
