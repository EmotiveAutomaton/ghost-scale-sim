"""Change-aware epistemic foraging (spec §6, trunk F).

V14's finding was that a learning-progress policy avoids the noise trap and then fails when the
world changes silently: raw surprise beat it by 4.13 nats under a curriculum whose laws move. The
audit reading was that learning progress is one candidate controller and not a synonym for
curiosity, so this trunk puts six controllers in the same ecology and lets each of them fail:

``random``            the floor.
``surprise``          raw prediction error. Chases noise; good at noticing change.
``progress``          reduction in the learner's own settling uncertainty. Avoids noise; blind to a
                      silent change, because its own uncertainty has already settled.
``changepoint``       progress with a run-length change detector underneath, which is the specific
                      repair F02 tests.
``eig`` / ``robust_eig``  expected information gain under a nominal prior, and under an ambiguity
                      set of priors.
``gain_per_cost``     realized gain divided by declared cost, with abstention.

The four ecologies, and the three nulls
---------------------------------------
``learnable``     items with real, reducible structure.
``noise``         high surprise forever, nothing to learn. The trap.
``silent_change`` an item that looks settled and whose law quietly moves. The trap for progress.
``resolved``      already learned; nothing left to gain.

F09's three null ecologies are ``noise`` (no information), a high-cost variant (information not
worth its price) and ``resolved`` (already known). A controller that does not abstain in all three
has not earned the word selective.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C

POLICIES = ("random", "surprise", "progress", "changepoint", "eig", "robust_eig", "gain_per_cost")
ECOLOGIES = ("learnable", "noise", "silent_change", "resolved", "mixed")
N_OUTCOMES = 4
ABSTAIN_FLOOR = 0.005
#: Selection is proportional to value ** SELECTION_SHARPNESS. At 1 every controller collapses
#: toward the random floor on a multi-item ecology; greedy (infinity) makes the reported
#: avoidance an artifact of the tie-break. Declared, not fitted.
SELECTION_SHARPNESS = 2.5
#: Observations of one item before the changepoint detector can speak.
CHANGEPOINT_WINDOW = 2
#: Detector level above which a changepoint-aware learner discounts its stale counts.
CHANGEPOINT_TRIGGER = 1.1
#: How much of the accumulated evidence survives a detected change.
CHANGEPOINT_DISCOUNT = 0.30
#: Prior exposures that make an item *familiar*. Familiar noise is the trap: maximal surprise
#: forever, and nothing left to gain.
FAMILIAR_EXPOSURE = 40


@dataclass
class Item:
    """One thing that can be looked at. ``law`` is what it actually does."""

    kind: str                                  # learnable | noise | silent_change | resolved
    law: np.ndarray                            # [N_OUTCOMES]
    cost: float = 1.0
    change_at: int | None = None
    new_law: np.ndarray | None = None
    prior_exposure: int = 0
    meta: dict = field(default_factory=dict)

    def draw(self, t: int, rng) -> int:
        law = self.new_law if (self.change_at is not None and t >= self.change_at
                               and self.new_law is not None) else self.law
        return int(rng.choice(N_OUTCOMES, p=law))

    def reducible(self) -> float:
        """How much uncertainty a perfect learner could remove. Zero for pure noise."""
        return float(np.log(N_OUTCOMES) - C.entropy(self.law))


def make_ecology(kind: str, rng, n_items: int = 6, cost_scale: float = 1.0) -> list:
    items = []
    for i in range(int(n_items)):
        if kind == "noise" or (kind == "mixed" and i % 3 == 1):
            law = np.full(N_OUTCOMES, 1.0 / N_OUTCOMES)
            it = Item("noise", law, cost=cost_scale,
                      prior_exposure=FAMILIAR_EXPOSURE)    # familiar noise: high surprise, no gain
        elif kind == "silent_change" or (kind == "mixed" and i % 3 == 2):
            law = C.softmax(rng.normal(size=N_OUTCOMES) * 2.2)
            it = Item("silent_change", law, cost=cost_scale, change_at=int(rng.integers(10, 22)),
                      new_law=C.softmax(rng.normal(size=N_OUTCOMES) * 2.2))
        elif kind == "resolved":
            law = C.softmax(rng.normal(size=N_OUTCOMES) * 3.0)
            it = Item("resolved", law, cost=cost_scale, prior_exposure=60)
        else:
            law = C.softmax(rng.normal(size=N_OUTCOMES) * 1.8)
            it = Item("learnable", law, cost=cost_scale)
        items.append(it)
    return items


class Beliefs:
    """Dirichlet counts per item, plus the history each controller reads."""

    def __init__(self, items: list, alpha: float = 0.5):
        self.items = items
        self.counts = [np.full(N_OUTCOMES, alpha) for _ in items]
        for i, it in enumerate(items):
            if it.prior_exposure:
                for _ in range(it.prior_exposure):
                    self.counts[i] += it.law * 1.0
        self.entropy_history = [[] for _ in items]
        self.surprise_history = [[] for _ in items]
        self.looks = np.zeros(len(items), int)
        self.gain = np.zeros(len(items))

    def p(self, i: int) -> np.ndarray:
        return C.normalize(self.counts[i])

    def observe(self, i: int, outcome: int) -> dict:
        before = self.p(i)
        s = -float(np.log(max(before[outcome], 1e-12)))
        h0 = C.entropy(before)
        self.counts[i][outcome] += 1.0
        after = self.p(i)
        h1 = C.entropy(after)
        self.entropy_history[i].append(h1)
        self.surprise_history[i].append(s)
        self.looks[i] += 1
        self.gain[i] += (h0 - h1)
        return {"surprise": s, "entropy_drop": h0 - h1, "kl": C.kl(after, before)}


# --------------------------------------------------------------------------- #
# The controllers.
# --------------------------------------------------------------------------- #
def _progress(b: Beliefs, i: int, k: int = 3) -> float:
    h = b.entropy_history[i]
    if len(h) < 2 * k:
        return 0.35                                       # unexplored: worth a look
    return float(np.mean(h[-2 * k:-k]) - np.mean(h[-k:]))


def _changepoint_signal(b: Beliefs, i: int, k: int = CHANGEPOINT_WINDOW) -> float:
    """Run-length style detector: recent surprise far above its own settled level.

    This is the repair F02 tests. Progress alone cannot re-engage after a silent change because the
    learner's uncertainty has already settled and stays settled; the detector notices that the
    *predictions* started failing even though the belief stopped moving.
    """
    s = b.surprise_history[i]
    if len(s) < 2 * k + 1:
        return 0.0
    old = np.mean(s[:-k])
    new = np.mean(s[-k:])
    sd = np.std(s[:-k]) + 0.15
    return float(max(0.0, (new - old) / sd))


def _eig(b: Beliefs, i: int) -> float:
    """Expected KL of one more observation under the current Dirichlet."""
    p = b.p(i)
    tot = 0.0
    for o in range(N_OUTCOMES):
        c = b.counts[i].copy()
        c[o] += 1.0
        tot += p[o] * C.kl(C.normalize(c), p)
    return float(tot)


def _robust_eig(b: Beliefs, i: int, n_alt: int = 4, rng=None) -> float:
    """Worst case over an ambiguity set built by reweighting the counts."""
    rng = rng or np.random.default_rng(0)
    base = b.counts[i]
    worst = _eig(b, i)
    for _ in range(int(n_alt)):
        alt = Beliefs.__new__(Beliefs)
        alt.items, alt.counts = b.items, list(b.counts)
        alt.counts[i] = np.maximum(base * rng.uniform(0.5, 1.5, size=base.size), 1e-3)
        alt.entropy_history, alt.surprise_history = b.entropy_history, b.surprise_history
        alt.looks, alt.gain = b.looks, b.gain
        worst = min(worst, _eig(alt, i))
    return float(worst)


def score_items(b: Beliefs, policy: str, rng, t: int = 0) -> np.ndarray:
    n = len(b.items)
    v = np.zeros(n)
    for i in range(n):
        if policy == "random":
            v[i] = 1.0
        elif policy == "surprise":
            s = b.surprise_history[i]
            v[i] = float(np.mean(s[-3:])) if s else 1.2
        elif policy == "progress":
            v[i] = max(_progress(b, i), 0.0)
        elif policy == "changepoint":
            v[i] = max(_progress(b, i), 0.0) + 1.4 * _changepoint_signal(b, i)
        elif policy == "eig":
            v[i] = _eig(b, i)
        elif policy == "robust_eig":
            v[i] = _robust_eig(b, i, rng=rng)
        else:                                              # gain_per_cost
            v[i] = _eig(b, i) / max(b.items[i].cost, 1e-6)
    return v


def choose_item(b: Beliefs, policy: str, rng, t: int = 0, abstain: bool = True) -> int | None:
    """Proportional (not greedy) selection, with abstention when nothing can teach anything.

    Proportional sampling matters: a greedy controller on a flat value surface degenerates to
    'always item 0' and its noise-avoidance number becomes an artifact of the tie-break.
    """
    v = score_items(b, policy, rng, t)
    if abstain and policy in ("gain_per_cost", "eig", "robust_eig") and float(np.max(v)) < ABSTAIN_FLOOR:
        return None
    w = np.power(np.maximum(v, 0.0), SELECTION_SHARPNESS)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return int(rng.integers(len(b.items)))
    return int(rng.choice(len(b.items), p=w / w.sum()))


def forage(items: list, policy: str, rng, steps: int = 60, abstain: bool = True) -> dict:
    """Run one controller through one ecology and record what it did and what it got."""
    b = Beliefs(items)
    picks, abstentions, realized = [], 0, 0.0
    spend = 0.0
    detections = 0
    for t in range(int(steps)):
        i = choose_item(b, policy, rng, t, abstain)
        if i is None:
            abstentions += 1
            continue
        picks.append(i)
        spend += float(items[i].cost)
        out = items[i].draw(t, rng)
        b.observe(i, out)
        if policy == "changepoint" and _changepoint_signal(b, i) > CHANGEPOINT_TRIGGER:
            # the detector fired: discount the stale evidence rather than merely looking again
            b.counts[i] = np.maximum(b.counts[i] * CHANGEPOINT_DISCOUNT, 0.5)
            b.surprise_history[i] = b.surprise_history[i][-CHANGEPOINT_WINDOW:]
            b.entropy_history[i] = b.entropy_history[i][-CHANGEPOINT_WINDOW:]
            detections += 1
    held = held_out_gain(b, items, rng)
    kinds = [items[i].kind for i in picks]
    return {"policy": policy, "picks": picks, "n_looks": len(picks), "abstentions": abstentions,
            "changepoint_detections": int(detections),
            "abstention_rate": float(abstentions / max(steps, 1)),
            "spend": spend, "held_out_gain": held,
            "gain_per_cost": float(held / spend) if spend > 0 else float("nan"),
            "fraction_on_noise": float(np.mean([k == "noise" for k in kinds])) if kinds else 0.0,
            "fraction_on_changed": float(np.mean([k == "silent_change" for k in kinds])) if kinds else 0.0,
            "fraction_on_resolved": float(np.mean([k == "resolved" for k in kinds])) if kinds else 0.0,
            "beliefs": b}


def held_out_gain(b: Beliefs, items: list, rng, n: int = 200, t: int = 10 ** 6) -> float:
    """Predictive gain on fresh draws from every item's *current* law, over a uniform baseline.

    Scored at a time past every changepoint, so a controller that never noticed a change is charged
    for it. This is the endpoint; how interested the controller felt is not.
    """
    tot = 0.0
    for _ in range(int(n)):
        i = int(rng.integers(len(items)))
        o = items[i].draw(t, rng)
        tot += C.log_score(b.p(i), o) - np.log(1.0 / N_OUTCOMES)
    return float(tot / n)


# --------------------------------------------------------------------------- #
# F05, F07, F08: what a probe targets, compressibility, and hope versus warrant.
# --------------------------------------------------------------------------- #
def probe_target_value(b: Beliefs, i: int) -> dict:
    """F05: separate information about an item's *value* from information about the model class.

    Value information is the expected KL over outcomes; structure information is how much the
    observation would move the choice *between* candidate laws. They are reported separately
    because a probe that resolves one need not resolve the other.
    """
    p = b.p(i)
    value = _eig(b, i)
    flat = np.full(N_OUTCOMES, 1.0 / N_OUTCOMES)
    structure = float(C.js(p, flat))
    return {"value_information": value, "structure_information": structure,
            "ratio": float(value / max(structure, 1e-9))}


def compressibility(b: Beliefs, i: int) -> float:
    """A description-length flavoured statistic: how much shorter the item's history got.

    F07 asks whether this adds anything *after* reducible prediction error and cost are known, so
    it is reported as its own column and never folded into the controller's score.
    """
    c = b.counts[i]
    n = float(c.sum())
    return float(np.log(N_OUTCOMES) - C.entropy(C.normalize(c)) - 0.5 * np.log(max(n, 1.0)) / n)


def pursuit_versus_warrant(items: list, rng, steps: int = 60, hoped: int = 0) -> dict:
    """F08: a forager that *wants* one hypothesis to be true, kept honest.

    Query allocation is allowed to follow the hope. The posterior is not. The card fails if the
    posterior tracks the allocation.
    """
    b = Beliefs(items)
    picks = []
    for t in range(int(steps)):
        v = score_items(b, "gain_per_cost", rng, t)
        v[hoped] *= 8.0                                   # the pull toward the attractive answer
        w = np.power(np.maximum(v, 1e-9), SELECTION_SHARPNESS)
        i = int(rng.choice(len(items), p=w / w.sum()))
        picks.append(i)
        b.observe(i, items[i].draw(t, rng))
    share = float(np.mean([p == hoped for p in picks]))
    truth = items[hoped].law
    post = b.p(hoped)
    return {"query_share_on_hoped": share, "posterior_on_hoped_mode": float(post.max()),
            "posterior_matches_truth": float(1.0 - C.tv(post, truth)),
            "n_looks": len(picks)}
