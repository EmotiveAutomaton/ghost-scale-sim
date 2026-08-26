"""Opportunity and multidimensional cost (spec §3.4).

Every choice record carries the alternatives actually available, the alternatives the maker
believed available, the alternatives visible to the reader, the chosen and forgone actions, the
counterfactual consequence of each action, a cost VECTOR per action, the actor's competence and
cost uncertainty, and whether each cost was voluntary, imposed, sunk, anticipated, or discovered
late. Paid cost can be caused by motivation, competence, knowledge, constraint, exploration, risk
tolerance, social obligation, or error; the world plants factorial cells in which each rival is
true, and a reader gets no credit for "the maker cared more".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C

COST_DIMS = ("time", "execution", "cognitive", "epistemic", "opportunity", "social", "risk", "imposed")
CAUSES = ("motivation", "competence", "knowledge", "constraint", "risk_tolerance", "social_obligation", "error")
ECOLOGIES = ("craft", "bureaucratic", "hazardous", "collegial", "frontier", "scarce")
DISCOVERY_ECOLOGIES = ECOLOGIES[:4]
_EPS = 1e-12

#: Levels used by factorial cards. Each cause has a "low" and a "high" that produce the same
#: observed effort in the matched cell by construction of the menus.
LEVELS = {"motivation": (0.6, 1.6), "competence": (0.45, 1.0), "knowledge": (0.5, 1.0),
          "constraint": (0.0, 1.0), "risk_tolerance": (0.1, 0.9), "social_obligation": (0.1, 0.9),
          "error": (0.05, 0.4)}


@dataclass
class Actor:
    w: np.ndarray                       # goal profile, values the reward vector
    motivation: float = 1.0             # goal strength s
    competence: float = 1.0             # divides execution and cognitive cost
    knowledge: float = 1.0              # P(sees a non-mandatory option); believed-cost precision
    constraint: float = 0.0             # P(an imposed mandatory option is present)
    risk_tolerance: float = 0.5         # 1 - weight on outcome variance
    social_obligation: float = 0.5      # weight on social cost
    error: float = 0.1                  # choice noise: beta = BETA * (1 - error)
    curiosity: float = 0.3              # reward for information
    weights: dict = field(default_factory=dict)   # dimension weights (portable tradeoffs)

    def dim_weights(self) -> np.ndarray:
        base = {"time": 1.0, "execution": 1.0, "cognitive": 1.0, "epistemic": 1.0, "opportunity": 0.0,
                "social": self.social_obligation, "risk": 1.0 - self.risk_tolerance, "imposed": 0.0}
        base.update(self.weights)
        return np.array([base[d] for d in COST_DIMS])


BETA = 8.0


def ecology_costs(ecology: str, payoff_value: np.ndarray, n: int, rng) -> np.ndarray:
    """(n, 8) cost vectors whose relation to reward and competence depends on the ecology."""
    c = np.zeros((n, len(COST_DIMS)))
    u = rng.uniform(0.0, 0.5, size=(n, len(COST_DIMS)))
    c[:] = u
    if ecology == "craft":            # execution cost rises with the payoff on offer
        c[:, 1] = 0.2 + 0.6 * payoff_value
        c[:, 2] = 0.1 + 0.3 * payoff_value
    elif ecology == "bureaucratic":   # imposed and time costs dominate, unrelated to payoff
        c[:, 7] = rng.uniform(0.0, 0.6, n)
        c[:, 0] = rng.uniform(0.2, 0.6, n)
    elif ecology == "hazardous":      # risk (variance) is the main price of high payoff
        c[:, 6] = 0.1 + 0.7 * payoff_value
        c[:, 1] = rng.uniform(0.0, 0.2, n)
    elif ecology == "collegial":      # social cost correlates with payoff; execution is cheap
        c[:, 5] = 0.1 + 0.6 * payoff_value
        c[:, 1] = rng.uniform(0.0, 0.15, n)
    elif ecology == "frontier":       # epistemic cost high; information valuable (E2 fresh)
        c[:, 3] = 0.2 + 0.6 * rng.uniform(0, 1, n)
        c[:, 1] = 0.2 * payoff_value
    elif ecology == "scarce":         # time and opportunity dominate (E2 fresh)
        c[:, 0] = 0.3 + 0.5 * payoff_value
        c[:, 2] = rng.uniform(0.2, 0.5, n)
    return c


def menu(rng, ng: int, n_options: int = 4, ecology: str = "craft", w_for_value=None, conc: float = 1.0,
         mandatory: bool = False) -> dict:
    payoff = rng.dirichlet(np.full(ng, conc), size=n_options)
    scale = rng.uniform(0.5, 1.5, size=n_options)
    payoff = payoff * scale[:, None]
    value = payoff.sum(axis=1) / payoff.sum(axis=1).max()
    cost = ecology_costs(ecology, value, n_options, rng)
    variance = rng.uniform(0.0, 1.0, n_options) * (1.0 + cost[:, 6])
    info = rng.uniform(0.0, 1.0, n_options)
    mand = np.zeros(n_options, dtype=bool)
    if mandatory:
        mand[int(rng.integers(n_options))] = True
    return {"payoff": payoff, "cost": cost, "variance": variance, "info": info, "mandatory": mand,
            "ecology": ecology, "n": int(n_options)}


def believed_menu(actor: Actor, m: dict, rng) -> dict:
    """What the maker believes is on offer: options unseen with probability 1-knowledge, and
    believed costs perturbed in proportion to ignorance."""
    n = m["n"]
    seen = np.array([True if m["mandatory"][i] else (rng.random() < actor.knowledge) for i in range(n)])
    if not seen.any():
        seen[int(rng.integers(n))] = True
    noise = (1.0 - actor.knowledge) * rng.normal(0.0, 0.3, size=m["cost"].shape)
    cost_b = np.maximum(m["cost"] + noise, 0.0)
    return {**m, "seen": seen, "cost_believed": cost_b}


def utility(actor: Actor, m: dict, believed: bool = True) -> np.ndarray:
    pay = m["payoff"] @ actor.w
    cost = m.get("cost_believed", m["cost"]) if believed else m["cost"]
    cost = cost.copy()
    cost[:, 1] = cost[:, 1] / max(actor.competence, 1e-6)
    cost[:, 2] = cost[:, 2] / max(actor.competence, 1e-6)
    dw = actor.dim_weights()
    u = actor.motivation * pay - cost @ dw - (1.0 - actor.risk_tolerance) * m["variance"] \
        + actor.curiosity * m["info"] - m["cost"][:, 3] * 0.0
    return u


def choose(actor: Actor, m: dict, rng, visible_to_reader: str = "full") -> dict:
    """The maker's choice with the full record. Mandatory options force the choice (imposed)."""
    b = believed_menu(actor, m, rng)
    u = utility(actor, b)
    beta = BETA * (1.0 - actor.error)
    logits = beta * u
    logits[~b["seen"]] = -np.inf
    if b["mandatory"].any():
        a = int(np.argmax(b["mandatory"]))
        mode = "imposed"
    else:
        p = C.softmax(logits)
        a = int(rng.choice(m["n"], p=p))
        mode = "voluntary"
    true_u = utility(actor, m, believed=False)
    srt = np.sort(true_u)
    counterfactual = m["payoff"] @ actor.w
    late = bool(rng.random() < 0.2 * (1.0 - actor.knowledge))
    rec = {"choice": a, "mode": mode, "payoff": m["payoff"], "cost": m["cost"], "variance": m["variance"],
           "info": m["info"], "mandatory": m["mandatory"], "n": m["n"], "ecology": m["ecology"],
           "available": list(range(m["n"])), "believed_available": [int(i) for i in np.flatnonzero(b["seen"])],
           "forgone": [i for i in range(m["n"]) if i != a],
           "counterfactual_value": counterfactual, "opportunity_cost": float(srt[-1] - true_u[a]) if m["n"] > 1 else 0.0,
           "opportunity_strength": float(srt[-1] - srt[-2]) if m["n"] > 1 else 0.0,
           "competence": actor.competence, "cost_uncertainty": float(1.0 - actor.knowledge),
           "cost_flags": {"voluntary": mode == "voluntary", "imposed": mode == "imposed",
                          "anticipated": not late, "discovered_late": late, "sunk": False},
           "paid_cost": m["cost"][a]}
    if visible_to_reader == "full":
        rec["visible"] = list(range(m["n"]))
    elif visible_to_reader == "chosen_only":
        rec["visible"] = [a]
    return rec


def stream(actor: Actor, rng, n: int, ng: int, ecology: str = "craft", n_options: int = 4,
           mandatory_rate: float | None = None, conc: float = 1.0) -> list:
    out = []
    for _ in range(int(n)):
        mand = (rng.random() < (actor.constraint if mandatory_rate is None else mandatory_rate))
        m = menu(rng, ng, n_options, ecology, conc=conc, mandatory=mand)
        out.append(choose(actor, m, rng))
    return out


# --------------------------------------------------------------------------- #
# Readers.
# --------------------------------------------------------------------------- #
def loglik(actor: Actor, rec: dict, cost_fn=None, menu_view: str = "full", size_view: int | None = None) -> float:
    """log P(choice | actor params) under the exact softmax, with optional misspecifications:
    ``cost_fn`` transforms the cost vector (weighting families, misspecified cost models);
    ``menu_view`` restricts the menu the reader models; ``size_view`` replaces the menu size the
    reader believes (choice-set-size neglect)."""
    if rec["mode"] == "imposed" and menu_view != "ignore_flags":
        return 0.0                                     # an imposed choice carries no preference evidence
    m = {"payoff": np.asarray(rec["payoff"]), "cost": np.asarray(rec["cost"]), "variance": np.asarray(rec["variance"]),
         "info": np.asarray(rec["info"]), "n": rec["n"]}
    if cost_fn is not None:
        m["cost"] = cost_fn(m["cost"])
    u = utility(actor, m, believed=False)
    a = int(rec["choice"])
    beta = BETA * (1.0 - actor.error)
    if menu_view == "outcome_only":
        # the reader sees only what was chosen and its payoff, not the menu: likelihood is the
        # profile's affinity for the chosen payoff against an average alternative
        z = beta * np.array([u[a], float(np.mean(u))])
        return float(z[0] - C.logsumexp(z))
    if size_view is not None and size_view < rec["n"]:
        # neglect: only ``size_view`` alternatives are modelled, the chosen plus a fixed random subset
        # of the others (fixed by the record, so the neglect is a property of the reader, not noise)
        rr = np.random.default_rng(int(abs(float(np.asarray(rec["payoff"]).sum() * 1e6))) % (2 ** 31))
        others = [i for i in range(rec["n"]) if i != a]
        keep = [a] + list(rr.choice(others, size=max(size_view - 1, 0), replace=False))
        z = beta * u[keep]
        return float(z[0] - C.logsumexp(z))
    z = beta * u
    return float(z[a] - C.logsumexp(z))


def cause_grid(levels: dict) -> list:
    """Every combination of the declared cause levels, as kwargs for Actor."""
    keys = list(levels)
    out = [{}]
    for k in keys:
        out = [{**o, k: float(v)} for o in out for v in levels[k]]
    return out


def posterior(profiles: dict, recs: list, causes: dict | None = None, prior=None, **kw) -> dict:
    """Joint posterior over profile names x cause cells from records. ``causes`` maps each cause
    to its candidate levels (the reader's factored grid); absent causes take Actor defaults."""
    cells = cause_grid(causes or {})
    names = list(profiles)
    ll = np.zeros((len(names), len(cells)))
    for i, n in enumerate(names):
        for j, cell in enumerate(cells):
            actor = Actor(np.asarray(profiles[n], float), **cell)
            ll[i, j] = sum(loglik(actor, r, **kw) for r in recs)
    if prior is not None:
        ll += np.log(np.maximum(np.asarray(prior, float), _EPS))
    P = C.softmax(ll.ravel()).reshape(ll.shape)
    return {"P": P, "names": names, "cells": cells, "profile": dict(zip(names, P.sum(axis=1))),
            "cause": [float(x) for x in P.sum(axis=0)]}


def predict_choice(post: dict, profiles: dict, m: dict, **kw) -> np.ndarray:
    """Posterior-predictive over a new menu's options."""
    out = np.zeros(m["n"])
    for i, n in enumerate(post["names"]):
        for j, cell in enumerate(post["cells"]):
            p = post["P"][i, j]
            if p <= 0:
                continue
            actor = Actor(np.asarray(profiles[n], float), **cell)
            mm = {"payoff": np.asarray(m["payoff"]), "cost": np.asarray(m["cost"]), "variance": np.asarray(m["variance"]),
                  "info": np.asarray(m["info"]), "n": m["n"]}
            if kw.get("cost_fn") is not None:
                mm["cost"] = kw["cost_fn"](mm["cost"])
            u = utility(actor, mm, believed=False)
            out += p * C.softmax(BETA * (1.0 - actor.error) * u)
    return C.normalize(out)


def total_cost_fn(cost: np.ndarray) -> np.ndarray:
    """The cost vector collapsed to its sum on the execution dimension (a total-cost-only reader)."""
    out = np.zeros_like(cost)
    out[:, 1] = cost.sum(axis=1)
    return out


WEIGHTING = {
    "linear": lambda c: c,
    "logarithmic": lambda c: np.log1p(3.0 * c) / np.log1p(3.0),
    "saturating": lambda c: 2.0 * c / (1.0 + 2.0 * c),
    "threshold": lambda c: (c > 0.3).astype(float) * 0.6,
    "rank_based": lambda c: np.argsort(np.argsort(c, axis=0), axis=0) / max(c.shape[0] - 1, 1) * 0.6,
    "resource_rational": lambda c: np.where(c > np.median(c, axis=0, keepdims=True), c, 0.0),
}


def learned_monotone(train_costs: np.ndarray, train_gain: np.ndarray, bins: int = 6):
    """A monotone (isotonic, pooled-adjacent-violators) map fitted on training records."""
    x = np.asarray(train_costs, float).ravel()
    y = np.asarray(train_gain, float).ravel()
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    centres, vals = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi)
        if sel.any():
            centres.append(float(x[sel].mean()))
            vals.append(float(y[sel].mean()))
    vals = np.array(vals)
    for i in range(1, len(vals)):                     # pool adjacent violators
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]
    centres = np.array(centres)

    def fn(c):
        return np.interp(c, centres, vals) if centres.size else c
    return fn


def frequency_predict(recs: list, n: int) -> np.ndarray:
    freq = np.bincount([int(r["choice"]) for r in recs], minlength=n).astype(float) + 0.5
    return freq / freq.sum()


def neglect_reader_size(n: int, gamma: float = 0.35) -> int:
    """The planted human-like heuristic: a menu of n reads as 2 + gamma * (n - 2) alternatives."""
    return int(max(2, round(2 + gamma * (n - 2))))


def hidden_menu(rec: dict, rng, hide: int = 1, add_false: int = 0) -> dict:
    """The reader's view when alternatives are missing or false (O16, X09)."""
    r = dict(rec)
    n = rec["n"]
    keep = [i for i in range(n) if i != rec["choice"]]
    rng.shuffle(keep)
    keep = keep[max(0, len(keep) - hide):] if hide < len(keep) else []
    visible = sorted([rec["choice"]] + keep)
    pay = np.asarray(rec["payoff"])[visible]
    cost = np.asarray(rec["cost"])[visible]
    var = np.asarray(rec["variance"])[visible]
    info = np.asarray(rec["info"])[visible]
    if add_false:
        pay = np.vstack([pay, rng.dirichlet(np.ones(pay.shape[1]), size=add_false)])
        cost = np.vstack([cost, rng.uniform(0, 0.5, size=(add_false, cost.shape[1]))])
        var = np.concatenate([var, rng.uniform(0, 1, add_false)])
        info = np.concatenate([info, rng.uniform(0, 1, add_false)])
    r.update({"payoff": pay, "cost": cost, "variance": var, "info": info, "n": int(pay.shape[0]),
              "choice": visible.index(rec["choice"]), "menu_incomplete": hide > 0, "menu_false": add_false > 0})
    return r
