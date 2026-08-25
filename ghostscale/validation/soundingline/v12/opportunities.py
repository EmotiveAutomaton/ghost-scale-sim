"""Opportunity records and the choice world used by trunk R (spec §3.6, §11).

Values are inferred from OPPORTUNITIES, not action counts: the same choice under a near tie is
weak evidence and under a large opposing cost is strong evidence. The choice world makes that
explicit. A maker with standing profile w faces menus of options; each option pays a vector over
the goal channels and carries a cost. The maker chooses by softmax over w . payoff - cost, at a
rationality beta; habits add option-specific tilts that repeat; expertise scales how well the
maker sees the payoffs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-300


@dataclass
class ChoiceWorld:
    ng: int
    family: dict
    family_names: list
    n_options: int = 4
    beta: float = 8.0


def menu(cw: ChoiceWorld, rng, cost_scale: float = 0.4, n_options: int | None = None) -> dict:
    n = cw.n_options if n_options is None else int(n_options)
    payoff = rng.dirichlet(np.ones(cw.ng), size=n)           # each option's channel payoffs
    cost = rng.uniform(0.0, cost_scale, size=n)
    return {"payoff": payoff, "cost": cost}


def utilities(cw: ChoiceWorld, w: np.ndarray, m: dict, habit: np.ndarray | None = None,
              k: float = 0.0, rng=None) -> np.ndarray:
    pay = m["payoff"]
    if k > 0 and rng is not None:
        pay = (1 - k) * pay + k * rng.dirichlet(np.ones(cw.ng), size=pay.shape[0])
    u = pay @ np.asarray(w, float) - m["cost"]
    if habit is not None:
        u = u + habit[: u.size]
    return u


def choose(cw: ChoiceWorld, w: np.ndarray, m: dict, rng, habit=None, k: float = 0.0,
           beta: float | None = None) -> dict:
    """The maker's choice with its full opportunity record."""
    b = cw.beta if beta is None else float(beta)
    u = utilities(cw, w, m, habit, k, rng)
    p = np.exp(b * (u - u.max()))
    p = p / p.sum()
    a = int(rng.choice(u.size, p=p))
    srt = np.sort(u)
    return {"choice": a, "utilities": u.tolist(), "cost": m["cost"].tolist(),
            "payoff": m["payoff"].tolist(),
            "opportunity_strength": float(srt[-1] - srt[-2]) if u.size > 1 else 0.0,
            "mode": "habitual" if (habit is not None and habit[a] > 0.5 * abs(u).max()) else "deliberated"}


def choice_loglik(cw: ChoiceWorld, w: np.ndarray, m: dict, a: int,
                  beta: float | None = None) -> float:
    b = cw.beta if beta is None else float(beta)
    u = m["payoff"] @ np.asarray(w, float) - m["cost"]
    z = b * (u - u.max())
    return float(z[a] - np.log(np.exp(z).sum()))


def profile_posterior_from_choices(cw: ChoiceWorld, records: list, prior: dict | None = None,
                                   beta: float | None = None, use_costs: bool = True) -> dict:
    """Exact posterior over the family from a sequence of (menu, choice) records.
    ``use_costs=False`` scores only which option was chosen, ignoring what it cost: the
    frequency reader that the opportunity-strength ruler (R05) must beat."""
    names = cw.family_names
    ll = np.zeros(len(names))
    for rec in records:
        m = {"payoff": np.asarray(rec["payoff"]), "cost": (np.asarray(rec["cost"]) if use_costs
                                                             else np.zeros(len(rec["cost"])))}
        for i, n in enumerate(names):
            ll[i] += choice_loglik(cw, cw.family[n], m, int(rec["choice"]), beta)
    if prior is not None:
        ll += np.log(np.maximum(np.array([prior.get(n, 0.0) for n in names]), _EPS))
    v = np.exp(ll - ll.max())
    v = v / v.sum()
    return dict(zip(names, v))


def stream_choices(cw: ChoiceWorld, w: np.ndarray, rng, n: int, cost_scale: float = 0.4,
                   habit=None, k: float = 0.0, beta=None) -> list:
    out = []
    for _ in range(int(n)):
        m = menu(cw, rng, cost_scale)
        rec = choose(cw, w, m, rng, habit, k, beta)
        out.append(rec)
    return out


def predict_choice(cw: ChoiceWorld, post: dict, m: dict, beta=None) -> np.ndarray:
    """Posterior-predictive distribution over the options of a new menu."""
    b = cw.beta if beta is None else float(beta)
    p = np.zeros(len(m["cost"]))
    for n, q in post.items():
        u = np.asarray(m["payoff"]) @ cw.family[n] - np.asarray(m["cost"])
        z = np.exp(b * (u - u.max()))
        p += q * z / z.sum()
    return p / p.sum()
