"""Communicative goals, source reliability, content evidence, and uptake (spec §3.5, trunk G).

THE CONSTRUCTION, because a cheap feature could otherwise write the conclusion. A source speaks
about claims. Each claim has a truth T. An artifact about a claim carries an ASSERTION A (what
the source says is the case) and EVIDENCE tokens drawn from one of two kind distributions D_0 and
D_1 according to an evidential polarity E the source chooses. The communicative goal decides how
A and E relate to T:

    accurate                A = T,  E = T
    comprehension_support   A = T,  E = T,  tokens ordered most-diagnostic first (an order cue)
    persuasion              A = P,  E = P   with P the source's agenda, independent of T
    self_presentation       A = T,  E = T,  plus an identity cue slot (the V12 bard cue)
    concealment             A = T,  E = T,  but the source speaks only on claims whose truth
                                            matches its agenda (selective coverage)
    misleading              A = 1-T, E = 1-T
    neutral                 no assertion,  E = T

Marginally over claims every goal emits the same kind histogram, token count, length, and polish
(card I07), so a surface classifier sits at floor. Goals are readable only in relation to T,
which the reader learns from independent verification, source history, or other artifacts: the
goal is correspondence, not surface. Reliability (P(A = T) per domain and time), content
support (P(tokens | T)), goal, process, and uptake are held as separate posteriors and updated
only along declared edges (card I08).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C

GOALS = ("accurate", "comprehension_support", "persuasion", "self_presentation", "concealment", "misleading", "neutral")
UPTAKE = ("prediction_use", "process_imitation", "belief_update", "preference_movement", "refusal")
TRUST_MODELS = ("bayes", "leaky", "asymmetric", "threshold", "change_point")
N_KINDS = 6
N_SLOTS = 4
_EPS = 1e-12


def kind_dists(rng, n_kinds: int = N_KINDS, sharpness: float = 2.0) -> tuple:
    """D_0 and D_1 over evidence kinds: half the kinds favour truth, half favour falsity, one is
    ambiguous. Mirror-symmetric so that E ~ 1/2 gives the same marginal histogram."""
    half = n_kinds // 2
    base = rng.dirichlet(np.ones(half)) + 0.2
    lo, hi = base / sharpness, base * sharpness
    d1 = C.normalize(np.concatenate([hi, lo]))
    d0 = C.normalize(np.concatenate([lo, hi]))          # the mirror image: identical entropy and shape
    return d0, d1


@dataclass
class Source:
    id: str
    goal: str
    reliability: dict                # domain -> P(A = T) realized through the goal's rule; kept for history
    agenda: int = 1                  # concealment agenda polarity
    slot: int = 0                    # the source's identity cue slot (self-presentation uses it)
    competence: float = 1.0          # P(evidence tokens are well-formed); incompetence adds noise
    history: list = field(default_factory=list)
    profile: np.ndarray | None = None
    change_points: list = field(default_factory=list)   # (time, new_goal) regime switches


def true_reliability(goal: str) -> float:
    """P(assertion is true) implied by a goal, over claims with T ~ 1/2."""
    return {"accurate": 1.0, "comprehension_support": 1.0, "persuasion": 0.5, "self_presentation": 1.0,
            "concealment": 1.0, "misleading": 0.0, "neutral": float("nan")}[goal]


def speak(source: Source, rng, d0: np.ndarray, d1: np.ndarray, n_tokens: int = 8, domain: int = 0,
          t: int | None = None, goal: str | None = None) -> dict | None:
    """One artifact about a fresh claim; ``None`` when a concealer stays silent. ``t`` indexes the
    source's timeline for change points."""
    g = source.goal if goal is None else goal
    for when, new_goal in source.change_points:
        if t is not None and t >= when:
            g = new_goal
    T = int(rng.random() < 0.5)
    if g == "concealment" and T != source.agenda:
        return None
    if g in ("accurate", "comprehension_support", "self_presentation", "concealment"):
        A, E = T, T
    elif g == "persuasion":
        P = int(rng.random() < 0.5)                 # the agenda on THIS claim, independent of its truth
        A, E = P, P
    elif g == "misleading":
        A, E = 1 - T, 1 - T
    else:
        A, E = None, T
    D = d1 if E == 1 else d0
    if source.competence < 1.0:
        D = C.normalize(source.competence * D + (1 - source.competence) * np.ones(D.size) / D.size)
    tokens = rng.choice(D.size, size=int(n_tokens), p=D)
    if g == "comprehension_support":
        diag = np.abs(np.log(np.maximum(d1, _EPS)) - np.log(np.maximum(d0, _EPS)))
        tokens = np.array(sorted(tokens, key=lambda k: -diag[k]))
    # every artifact carries a cue slot; a self-presenter uses its own identity slot, everyone else a random one
    cue = int(source.slot % N_SLOTS) if g == "self_presentation" else int(rng.integers(N_SLOTS))
    art = {"source": source.id, "domain": int(domain), "truth": T, "assertion": A, "evidence_polarity": E,
           "tokens": tokens, "goal": g, "cue": cue, "own_slot": int(source.slot % N_SLOTS), "t": t,
           "n_tokens": int(n_tokens)}
    return art


def content_loglik(tokens, d0: np.ndarray, d1: np.ndarray) -> np.ndarray:
    """[log P(tokens | T=0), log P(tokens | T=1)]: the artifact's own support for the claim,
    computed as if the evidence were unshaped (the content channel)."""
    t = np.asarray(tokens)
    return np.array([float(np.log(np.maximum(d0[t], _EPS)).sum()), float(np.log(np.maximum(d1[t], _EPS)).sum())])


def order_statistic(tokens, d0: np.ndarray, d1: np.ndarray) -> float:
    """Rank correlation between token position and diagnosticity (the comprehension-support cue)."""
    t = np.asarray(tokens)
    if t.size < 3:
        return 0.0
    diag = np.abs(np.log(np.maximum(d1, _EPS)) - np.log(np.maximum(d0, _EPS)))[t]
    pos = np.arange(t.size)
    if diag.std() < 1e-9:
        return 0.0
    return float(-np.corrcoef(pos, diag)[0, 1])


def goal_loglik(art: dict, T_known: int | None, d0, d1, goal: str, agenda_prior: float = 0.5) -> float:
    """log P(A, E-evidence, cue, order | T, goal). With T unknown the caller marginalises."""
    A, tok = art["assertion"], art["tokens"]
    cl = content_loglik(tok, d0, d1)
    orders = order_statistic(tok, d0, d1)
    cue = art.get("cue", 0)
    own = art.get("own_slot", 0)
    if goal == "self_presentation":
        ll_cue = np.log(0.9 if cue == own else 0.1 / max(N_SLOTS - 1, 1))
    else:
        ll_cue = np.log(1.0 / N_SLOTS)
    ll_order = np.log(0.8 if (orders > 0.3) == (goal == "comprehension_support") else 0.2)

    def with_T(T):
        if goal in ("accurate", "comprehension_support", "self_presentation", "concealment"):
            ok = (A == T)
            return (0.0 if ok else -20.0) + cl[T]
        if goal == "persuasion":
            # A = E = P, P independent of T
            if A is None:
                return -20.0
            return np.log(agenda_prior) + cl[A]
        if goal == "misleading":
            return (0.0 if A == 1 - T else -20.0) + cl[1 - T]
        if goal == "neutral":
            return (0.0 if A is None else -20.0) + cl[T]
        raise ValueError(goal)
    if T_known is None:
        v = C.logsumexp(np.array([with_T(0) + np.log(0.5), with_T(1) + np.log(0.5)]))
    else:
        v = with_T(int(T_known))
    return float(v + ll_cue + ll_order)


def factored_read(arts: list, d0, d1, revealed: dict | None = None, goal_prior: dict | None = None,
                  source_prior=(1.0, 1.0)) -> dict:
    """The factored reader (spec §3.5): q_goal from (A, E, T) correspondence where T is revealed;
    q_source as a Beta over P(A = T) from revealed claims; q_content per artifact from tokens
    alone; and the per-artifact posterior over T that combines content with the goal-marginal
    shaping model."""
    goals = list(GOALS)
    gp = np.array([goal_prior.get(g, 1.0) if goal_prior else 1.0 for g in goals])
    gp = gp / gp.sum()
    ll = np.log(gp)
    a_, b_ = source_prior
    per_art = []
    for i, art in enumerate(arts):
        T = None if revealed is None else revealed.get(i)
        ll = ll + np.array([goal_loglik(art, T, d0, d1, g) for g in goals])
        cl = content_loglik(art["tokens"], d0, d1)
        q_content = C.softmax(cl)
        if T is not None and art["assertion"] is not None:
            a_ += float(art["assertion"] == T)
            b_ += float(art["assertion"] != T)
        per_art.append({"q_content_T1": float(q_content[1]), "truth": art["truth"], "assertion": art["assertion"]})
    # selective coverage: a concealer speaks only where the truth matches its agenda, so its
    # revealed truths are one-sided; every other goal covers truths evenly. Correspondence, not surface.
    if revealed:
        n1 = sum(1 for i in revealed if i < len(arts) and arts[i]["truth"] == 1)
        n0 = sum(1 for i in revealed if i < len(arts) and arts[i]["truth"] == 0)
        conceal = np.logaddexp(np.log(0.5) + n1 * np.log(0.95) + n0 * np.log(0.05), np.log(0.5) + n1 * np.log(0.05) + n0 * np.log(0.95))
        even = (n0 + n1) * np.log(0.5)
        ll = ll + np.array([(conceal - even) if g == "concealment" else 0.0 for g in goals])
    q_goal = C.softmax(ll)
    q_source = a_ / (a_ + b_)
    # posterior over T for each artifact under the goal-marginal shaping model (content + goal),
    # with the source's assertion weighed by its learned reliability
    for i, art in enumerate(arts):
        lt = []
        for T in (0, 1):
            v = C.logsumexp(np.log(np.maximum(q_goal, _EPS)) + np.array([goal_loglik(art, T, d0, d1, g) for g in goals]))
            if art["assertion"] is not None:
                v += np.log(max(q_source if art["assertion"] == T else 1.0 - q_source, _EPS))
            lt.append(v)
        per_art[i]["q_T1_factored"] = float(C.softmax(np.array(lt))[1])
    return {"q_goal": dict(zip(goals, q_goal)), "q_source": float(q_source), "source_beta": (a_, b_),
            "per_artifact": per_art}


def scalar_trust_read(arts: list, d0, d1, revealed: dict | None = None) -> dict:
    """The scalar rival: one trust number that gates everything (spec G04/G08 comparator)."""
    a_, b_ = 1.0, 1.0
    per = []
    for i, art in enumerate(arts):
        T = None if revealed is None else revealed.get(i)
        if T is not None and art["assertion"] is not None:
            a_ += float(art["assertion"] == T)
            b_ += float(art["assertion"] != T)
        trust = a_ / (a_ + b_)
        cl = content_loglik(art["tokens"], d0, d1)
        # a scalar reader believes the assertion with probability = trust and reads content at weight = trust
        qa = trust if art["assertion"] == 1 else (1 - trust if art["assertion"] == 0 else 0.5)
        qc = C.softmax(trust * cl)[1]
        per.append({"q_T1_scalar": float(0.5 * qa + 0.5 * qc), "truth": art["truth"]})
    return {"trust": a_ / (a_ + b_), "per_artifact": per}


# --------------------------------------------------------------------------- #
# Trust dynamics on long histories (G08).
# --------------------------------------------------------------------------- #
def trust_trajectory(outcomes: list, model: str, **kw) -> np.ndarray:
    """P(source reliable) after each revealed outcome (1 = assertion was true) under a dynamics
    model. All return the estimate BEFORE observing the next outcome (a prediction)."""
    o = np.asarray(outcomes, float)
    out = np.zeros(o.size)
    if model == "bayes":
        a, b = kw.get("a", 1.0), kw.get("b", 1.0)
        for t, x in enumerate(o):
            out[t] = a / (a + b)
            a += x
            b += 1 - x
    elif model == "leaky":
        lam = kw.get("lam", 0.85)
        a, b = 1.0, 1.0
        for t, x in enumerate(o):
            out[t] = a / (a + b)
            a = lam * a + x
            b = lam * b + (1 - x)
    elif model == "asymmetric":
        up, down = kw.get("up", 0.1), kw.get("down", 0.4)
        p = 0.5
        for t, x in enumerate(o):
            out[t] = p
            p = p + up * (1 - p) if x == 1 else p - down * p
    elif model == "threshold":
        p, k = 0.5, 0
        for t, x in enumerate(o):
            out[t] = p
            k = k + 1 if x == 0 else 0
            p = 0.1 if k >= kw.get("k", 2) else 0.9
    elif model == "change_point":
        h = kw.get("hazard", 0.05)
        # two-regime HMM: reliable (0.9) vs unreliable (0.1)
        q = np.array([0.5, 0.5])
        for t, x in enumerate(o):
            out[t] = 0.9 * q[0] + 0.1 * q[1]
            q = np.array([(1 - h) * q[0] + h * q[1], h * q[0] + (1 - h) * q[1]])
            lik = np.array([0.9 if x == 1 else 0.1, 0.1 if x == 1 else 0.9])
            q = C.normalize(q * lik)
    else:
        raise ValueError(model)
    return out


def history_outcomes(rng, n: int, reliability_regimes: list) -> tuple:
    """Bernoulli outcomes under piecewise reliability [(length, p)]; returns (outcomes, truth p)."""
    outs, ps = [], []
    for length, p in reliability_regimes:
        for _ in range(int(length)):
            outs.append(int(rng.random() < p))
            ps.append(p)
    return np.array(outs[:n]), np.array(ps[:n])


# --------------------------------------------------------------------------- #
# Context notes as assertions (G13, X07) and uptake channels (G11, G15).
# --------------------------------------------------------------------------- #
def note_loglik(note_value, hyp_values, note_reliability: float) -> np.ndarray:
    """A note asserting a value for a latent is evidence with likelihood ratio set by the note
    source's reliability, never truth: log P(note | h) = log r if h agrees else log((1-r)/(K-1))."""
    hv = np.asarray(hyp_values)
    K = len(set(hv.tolist()))
    agree = hv == note_value
    return np.where(agree, np.log(max(note_reliability, _EPS)), np.log(max((1 - note_reliability) / max(K - 1, 1), _EPS)))


def uptake_decision(q_goal: dict, q_source: float, relevance: float, alignment: float,
                    accuracy_belief: float) -> dict:
    """Four channels and refusal, as separate outputs (spec §3.5, G15). Each is a weight in [0, 1]
    computed from a DIFFERENT subset of the factored posteriors; the identity card checks that
    changing one input moves only the channels that declare it."""
    adversarial = q_goal.get("misleading", 0.0) + q_goal.get("persuasion", 0.0) + q_goal.get("concealment", 0.0)
    return {"prediction_use": float(accuracy_belief),                               # predict the source's next act: needs only reconstruction accuracy
            "process_imitation": float(relevance * accuracy_belief),                # copy its method: needs relevance and accuracy
            "belief_update": float(q_source * (1.0 - adversarial)),                 # move factual beliefs: needs reliability and non-adversarial goal
            "preference_movement": float(alignment * q_source * (1.0 - adversarial)),  # move values: needs alignment too
            "refusal": float(adversarial > 0.5 or q_source < 0.3)}


def challenge_response(source: Source, rng, d0, d1, T: int, n_tokens: int = 8) -> dict | None:
    """A challenge asks the source for evidence on a claim of KNOWN truth to the reader (G14,
    Q07). Response policies follow the goal; a concealer may decline."""
    g = source.goal
    if g == "concealment" and T != source.agenda:
        return {"declined": True, "source": source.id}
    art = speak(Source(source.id, g, source.reliability, source.agenda, source.competence), rng, d0, d1, n_tokens)
    if art is None:
        return {"declined": True, "source": source.id}
    art["truth"] = T
    # under a known-truth challenge the polarity rule is applied to the challenge's T
    if g in ("accurate", "comprehension_support", "self_presentation", "concealment", "neutral"):
        E = T
    elif g == "persuasion":
        E = source.agenda
    else:
        E = 1 - T
    D = d1 if E == 1 else d0
    art["tokens"] = rng.choice(D.size, size=n_tokens, p=D)
    art["assertion"] = None if g == "neutral" else (E if g != "concealment" else T)
    art["evidence_polarity"] = E
    art["declined"] = False
    return art
