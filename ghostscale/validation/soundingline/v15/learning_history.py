"""Endogenous expertise, learning history and residue (spec §3.2, trunk E).

What V14 did and why it does not count
--------------------------------------
V14 moved a "competence" knob and a "history" knob and showed that each moved its own supplied
readout channel. The factors were orthogonal by construction and the reader was handed the matching
features, so the result was a `CONSTRUCTION_IDENTITY`: it demonstrated that the wiring was
connected. V15 is not allowed to repeat it (spec §2), so here:

* competence is **not a parameter**. It is the measured accuracy of a transition model that was
  *learned* from a training history, and two makers reach the same accuracy by different routes.
* the reader is **never given the history**, nor a fixed per-history signature. Curricula are
  randomized per maker (E12), so there is no planted class to recover.
* what the reader gets is behaviour: choices on held-out items. What it is asked for is a *future*
  event -- where the next novel error falls, how fast a reversed skill is reacquired, how far the
  skill transfers -- never the history label on its own.

The four training sources, and the residue each one leaves
----------------------------------------------------------
``practice``      the learner attempts an item and updates toward its own answer, except that
                  the world sometimes shows whether the attempt worked. Self-confirming: it
                  deepens whatever it already does, and only the partial outcome signal keeps
                  it from locking in its first guesses forever.
``instruction``   the correct answer is supplied. Fast and exact on the instructed item, and
                  verbatim -- it generalizes to neighbours least.
``feedback``      the learner attempts, and is told only whether it was right. Corrective where it
                  was wrong; leaves the rest untouched.
``constraint``    an item is *blocked*. The learner never practises it, so the residue is a hole
                  rather than a bias, and the hole is invisible until the constraint lifts.

Matched final skill is enforced by construction (E01): the mixture determines *how* a learner got
there, and the amount of training is solved per maker so that overall accuracy lands in a narrow
band. Two makers of equal skill therefore differ only in error topology, transfer breadth and
relearning rate -- which is what every card in the trunk actually scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C

SOURCES = ("practice", "instruction", "feedback", "constraint")
N_ITEMS = 12                      # distinct transitions a maker can learn
N_RESPONSES = 4                   # answers available for each
#: How much one exposure from each source moves the learner, and how far it spreads to neighbours.
RATE = {"practice": 0.55, "instruction": 1.35, "feedback": 0.95, "constraint": 0.0}
#: How often self-directed practice sees whether the attempt actually worked. At 0 a practice
#: history cannot improve past chance and cannot be skill-matched to an instructed one.
PRACTICE_OUTCOME_VISIBILITY = 0.40
SPREAD = {"practice": 0.35, "instruction": 0.05, "feedback": 0.20, "constraint": 0.0}
TARGET_SKILL = 0.72
SKILL_BAND = 0.05


@dataclass
class Curriculum:
    """A randomized training history. ``items`` and ``sources`` are the same length."""

    items: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    blocked: tuple = ()
    mixture: dict = field(default_factory=dict)
    n_exposures: int = 0

    def counts_by_source(self) -> dict:
        return {s: int(sum(1 for x in self.sources if x == s)) for s in SOURCES}

    def dated_records(self) -> list:
        """(episode index, item, source) triples -- the *dated* evidence E11 compares with a bag."""
        return [{"t": i, "item": it, "source": s}
                for i, (it, s) in enumerate(zip(self.items, self.sources))]


@dataclass
class Learner:
    """A learned response model over items, plus what it never got to practise."""

    weights: np.ndarray                     # [N_ITEMS, N_RESPONSES] log-evidence
    truth: np.ndarray                       # [N_ITEMS] the correct response
    blocked: tuple = ()
    curriculum: Curriculum | None = None
    trace: list = field(default_factory=list)

    def policy(self, item: int, temperature: float = 0.5) -> np.ndarray:
        return C.softmax(self.weights[item] / max(temperature, 1e-3))

    def answer(self, item: int, rng, temperature: float = 0.5) -> int:
        return int(rng.choice(N_RESPONSES, p=self.policy(item, temperature)))

    def skill(self, temperature: float = 0.5, items=None) -> float:
        items = range(N_ITEMS) if items is None else items
        return float(np.mean([self.policy(i, temperature)[int(self.truth[i])] for i in items]))

    def error_profile(self, temperature: float = 0.5) -> np.ndarray:
        """Per-item probability of being wrong. The *shape* of this, not its mean, is the residue."""
        return np.array([1.0 - self.policy(i, temperature)[int(self.truth[i])]
                         for i in range(N_ITEMS)])


def sample_truth(rng) -> np.ndarray:
    return rng.integers(0, N_RESPONSES, size=N_ITEMS)


def sample_curriculum(mixture: dict, n_exposures: int, rng, blocked_k: int = 0) -> Curriculum:
    """Draw a *randomized* curriculum with the given source mixture.

    Randomization is the point (E12): with two fixed transition matrices a reader recovers a
    planted signature and the trunk proves nothing. Here the item order, the item frequencies and
    which items are blocked all vary per maker.
    """
    ms = np.array([mixture.get(s, 0.0) for s in SOURCES], float)
    ms = ms / ms.sum() if ms.sum() > 0 else np.full(len(SOURCES), 1.0 / len(SOURCES))
    blocked = tuple(sorted(rng.choice(N_ITEMS, size=blocked_k, replace=False))) if blocked_k else ()
    weight = rng.dirichlet(np.full(N_ITEMS, 1.6))              # uneven exposure, per maker
    items, sources = [], []
    for _ in range(int(n_exposures)):
        it = int(rng.choice(N_ITEMS, p=weight))
        src = SOURCES[int(rng.choice(len(SOURCES), p=ms))]
        if it in blocked:
            src = "constraint"
        items.append(it)
        sources.append(src)
    return Curriculum(items=items, sources=sources, blocked=blocked,
                      mixture={s: float(m) for s, m in zip(SOURCES, ms)},
                      n_exposures=int(n_exposures))


def train(truth: np.ndarray, cur: Curriculum, rng, temperature: float = 0.5) -> Learner:
    """Run the curriculum. Every source updates the learner differently."""
    w = np.zeros((N_ITEMS, N_RESPONSES))
    lr = Learner(weights=w, truth=truth, blocked=cur.blocked, curriculum=cur)
    for t, (item, src) in enumerate(zip(cur.items, cur.sources)):
        if src == "constraint":
            lr.trace.append({"t": t, "item": item, "source": src, "moved": 0.0})
            continue
        rate, spread = RATE[src], SPREAD[src]
        if src == "instruction":
            target = int(truth[item])
        elif src == "practice":
            target = (int(truth[item]) if rng.random() < PRACTICE_OUTCOME_VISIBILITY
                      else lr.answer(item, rng, temperature))   # own answer, sometimes checked
        else:                                                   # feedback: corrects only if wrong
            got = lr.answer(item, rng, temperature)
            target = int(truth[item]) if got != int(truth[item]) else got
        w[item, target] += rate
        if spread > 0:                                          # generalization to neighbours
            for nb in ((item - 1) % N_ITEMS, (item + 1) % N_ITEMS):
                if nb in cur.blocked:
                    continue
                w[nb, target] += rate * spread
        lr.trace.append({"t": t, "item": item, "source": src, "target": int(target),
                         "moved": float(rate)})
    lr.weights = w
    return lr


def train_to_skill(truth: np.ndarray, mixture: dict, rng, target: float = TARGET_SKILL,
                   blocked_k: int = 0, temperature: float = 0.5, lo: int = 6,
                   hi: int = 420) -> tuple:
    """Solve for the exposure count that lands this mixture at ``target`` final skill.

    This is what makes E01 a real question rather than a definition: the mixture decides *how* the
    learner got there and the exposure count is spent to make sure everyone got to the same place.
    A card comparing two mixtures at matched skill is comparing histories, not amounts of practice.
    """
    best = None
    for _ in range(14):
        mid = (lo + hi) // 2
        cur = sample_curriculum(mixture, mid, np.random.default_rng(rng.integers(0, 2 ** 62)),
                                blocked_k)
        lr = train(truth, cur, np.random.default_rng(rng.integers(0, 2 ** 62)), temperature)
        sk = lr.skill(temperature)
        # keep the CLOSEST candidate, not the last one the bisection happened to try: the
        # objective is noisy because every evaluation redraws a randomized curriculum
        if best is None or abs(sk - target) < abs(best[2] - target):
            best = (lr, cur, sk, mid)
        if sk < target:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    lr, cur, sk, n = best
    return lr, cur, {"final_skill": sk, "exposures": n, "matched": bool(abs(sk - target) <= SKILL_BAND)}


# --------------------------------------------------------------------------- #
# What a reader observes, and the two rival reader models.
# --------------------------------------------------------------------------- #
def observe(lr: Learner, rng, n: int = 24, temperature: float = 0.5, items=None) -> list:
    """Held-out behaviour: the learner answering items. No history, no labels."""
    pool = list(range(N_ITEMS)) if items is None else list(items)
    out = []
    for _ in range(int(n)):
        it = int(rng.choice(pool))
        out.append({"item": it, "response": lr.answer(it, rng, temperature),
                    "correct": int(lr.answer(it, rng, temperature) == int(lr.truth[it]))})
    return out


def attention_only_model(obs: list) -> np.ndarray:
    """The rival V15 has to beat: *expertise is previous attention*.

    It estimates how much each item was attended to from how often the learner answers it
    confidently, and predicts that error falls where attention was low. It has no notion of a
    source, so instruction and feedback are the same event to it.
    """
    seen = np.full(N_ITEMS, 0.5)
    for o in obs:
        seen[o["item"]] += 1.0
    return C.normalize(1.0 / seen)


def learning_record_model(obs: list, truth: np.ndarray) -> np.ndarray:
    """The richer rival: models *what kind* of exposure the behaviour implies.

    A self-confirming history leaves confident wrong answers; an instructed one leaves correct but
    isolated answers; a feedback history leaves correct answers with correct neighbours. The model
    reads the local consistency of the behaviour rather than its volume.
    """
    conf = np.full(N_ITEMS, 0.0)
    corr = np.full(N_ITEMS, 0.0)
    cnt = np.full(N_ITEMS, 1e-6)
    for o in obs:
        cnt[o["item"]] += 1.0
        corr[o["item"]] += float(o["response"] == int(truth[o["item"]]))
    acc = corr / np.maximum(cnt, 1e-6)
    for i in range(N_ITEMS):
        nb = [(i - 1) % N_ITEMS, (i + 1) % N_ITEMS]
        conf[i] = acc[i] - 0.5 * float(np.mean([acc[j] for j in nb]))
    risk = (1.0 - acc) + 0.6 * np.maximum(-conf, 0.0)          # wrong, and isolated from support
    return C.normalize(risk)


def novel_error_score(pred: np.ndarray, lr: Learner, temperature: float = 0.5) -> float:
    """Log score of a predicted error *location* against where the error actually falls."""
    prof = lr.error_profile(temperature)
    where = int(np.argmax(prof))
    return C.log_score(C.normalize(pred), where)


# --------------------------------------------------------------------------- #
# Reversal, transfer and correction (E07, E08, E09).
# --------------------------------------------------------------------------- #
def reverse(truth: np.ndarray, rng, k: int = 4) -> tuple:
    """Reverse the correct answer on ``k`` items. Relearning after this is E08's endpoint."""
    t2 = truth.copy()
    idx = rng.choice(N_ITEMS, size=k, replace=False)
    for i in idx:
        t2[i] = int((truth[i] + 1 + rng.integers(0, N_RESPONSES - 1)) % N_RESPONSES)
    return t2, tuple(sorted(int(i) for i in idx))


def relearning_curve(lr: Learner, new_truth: np.ndarray, rng, steps: int = 24,
                     temperature: float = 0.5) -> dict:
    """How fast the learner reacquires after a reversal. The path predicts this; skill does not."""
    w = lr.weights.copy()
    tmp = Learner(weights=w, truth=new_truth, blocked=lr.blocked)
    trace = []
    for t in range(int(steps)):
        item = int(rng.integers(N_ITEMS))
        got = tmp.answer(item, rng, temperature)
        target = int(new_truth[item]) if got != int(new_truth[item]) else got
        w[item, target] += RATE["feedback"]
        trace.append(tmp.skill(temperature))
    half = next((i for i, s in enumerate(trace) if s > 0.5 * (trace[0] + max(trace))), None)
    return {"trace": trace, "start": trace[0] if trace else float("nan"),
            "end": trace[-1] if trace else float("nan"), "half_life": half,
            "gain": (trace[-1] - trace[0]) if trace else float("nan")}


def transfer_breadth(lr: Learner, rng, temperature: float = 0.5) -> dict:
    """Skill on *neighbouring* items the learner was never directly trained on.

    Two makers matched on overall skill can differ here by a lot: verbatim instruction transfers
    least, self-directed practice most. E07's endpoint.
    """
    trained = {o for o in (lr.curriculum.items if lr.curriculum else [])}
    untrained = [i for i in range(N_ITEMS) if i not in trained]
    return {"trained_skill": lr.skill(temperature, sorted(trained)) if trained else float("nan"),
            "untrained_skill": lr.skill(temperature, untrained) if untrained else float("nan"),
            "n_untrained": len(untrained),
            "breadth": (lr.skill(temperature, untrained) - lr.skill(temperature, sorted(trained)))
            if (trained and untrained) else float("nan")}


def correct_residue(lr: Learner, rng, targeted: bool, n: int = 18,
                    temperature: float = 0.5) -> dict:
    """Targeted evidence on the worst items versus scattered evidence (E09).

    The question is whether a stale bias can be removed *without* costing the valid skill beside
    it, so both quantities are reported and neither is allowed to stand alone.
    """
    before_profile = lr.error_profile(temperature)
    before_skill = lr.skill(temperature)
    worst = list(np.argsort(-before_profile)[:4])
    w = lr.weights.copy()
    tmp = Learner(weights=w, truth=lr.truth, blocked=lr.blocked, curriculum=lr.curriculum)
    for _ in range(int(n)):
        item = int(rng.choice(worst)) if targeted else int(rng.integers(N_ITEMS))
        w[item, int(lr.truth[item])] += RATE["feedback"]
    after_profile = tmp.error_profile(temperature)
    return {"targeted": bool(targeted),
            "bias_removed": float(before_profile[worst].mean() - after_profile[worst].mean()),
            "skill_before": before_skill, "skill_after": tmp.skill(temperature),
            "skill_cost": float(before_skill - tmp.skill(temperature))}


def behaviour_signature(obs: list, truth: np.ndarray) -> np.ndarray:
    """Permutation-invariant summary of a learner's behaviour.

    Item identity is randomized per maker, so any feature indexed *by item* is meaningless across
    makers. What survives relabelling is the shape of the competence distribution and how locally
    consistent it is:

    * the sorted per-item accuracy curve -- instruction leaves a spiky profile, practice a smooth
      one, a constraint leaves a hole;
    * its spread and its extremes;
    * neighbour consistency -- feedback repairs an item and leaves its neighbours alone, practice
      spreads, so the correlation between an item's accuracy and its neighbours' differs.
    """
    n = len(truth)
    corr = np.zeros(n)
    cnt = np.full(n, 1e-6)
    for o in obs:
        cnt[o["item"]] += 1.0
        corr[o["item"]] += float(o["response"] == int(truth[o["item"]]))
    acc = corr / np.maximum(cnt, 1e-6)
    seen = cnt > 1e-3
    srt = np.sort(acc)
    q = np.quantile(srt, [0.0, 0.25, 0.5, 0.75, 1.0])
    nb = np.array([0.5 * (acc[(i - 1) % n] + acc[(i + 1) % n]) for i in range(n)])
    consistency = float(np.corrcoef(acc, nb)[0, 1]) if acc.std() > 1e-9 and nb.std() > 1e-9 else 0.0
    return np.array([*q, float(acc.mean()), float(acc.std()), consistency,
                     float(np.mean(~seen)), float(np.mean(acc < 0.25))])


def history_posterior(obs: list, truth: np.ndarray, candidates: dict, rng,
                      temperature: float = 0.5, n_sim: int = 6) -> dict:
    """Posterior over which *mixture* produced this behaviour, by forward simulation.

    No supplied history feature and no fixed signature: each candidate mixture is trained to the
    same skill on fresh randomized curricula, and scored on permutation-invariant behaviour
    features. E02's estimator, and it is allowed to sit at chance -- but not below it, which is
    what the earlier item-indexed version did.
    """
    target = behaviour_signature(obs, truth)
    lls = {}
    for name, mix in candidates.items():
        sims = []
        for _ in range(int(n_sim)):
            sub = np.random.default_rng(rng.integers(0, 2 ** 62))
            lr, _, _ = train_to_skill(truth, mix, sub, temperature=temperature)
            sims.append(behaviour_signature(observe(lr, sub, n=len(obs), temperature=temperature),
                                            truth))
        m = np.mean(sims, axis=0)
        sd = np.std(sims, axis=0) + 0.05
        lls[name] = float(-0.5 * np.sum(((target - m) / sd) ** 2))
    keys = list(lls)
    post = C.softmax(np.array([lls[k] for k in keys]))
    return {k: float(v) for k, v in zip(keys, post)}
