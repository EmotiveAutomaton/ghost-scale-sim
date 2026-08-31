"""Persistent tendency, current value, change and concealment (spec §3.2, trunk V).

The trunk's hard requirement (spec §6 V04, pre-mortem item 11) is that *changed value* and *better
concealment* must not differ by a telltale template. So both are produced by one planner::

    utility(option) = preference . features(option)
                    - audience_cost(option) * concealment
                    - effort(option) / competence
                    + signal_value(option) * signalling

A maker whose preference reversed and a maker who merely got better at hiding the same preference
are two settings of the same equation, and every public choice they make comes out of the same
code path. What separates them is *where* the choices differ: concealment moves the high-audience
options only, a real change moves the private ones too. That is V04's endpoint and it is an
intervention, not a signature.

The cost trunk (V05, V06, V09) has the same shape. A paid cost is not evidence of preference until
low competence, an imposed constraint, a signalling motive and a different cost function have all
been given the same chance to explain it, so the posterior is over a *cost vector* rather than over
preference alone. Forgone alternatives are separate evidence again: the option set the maker did
not take is what distinguishes an unavailable option from a rejected one (V06).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C

N_OPTIONS = 6
N_FEATURES = 4
#: The rival explanations a paid cost has to survive before it counts as preference (V05).
COST_OWNERS = ("preference", "competence", "constraint", "signalling", "cost_function")


@dataclass
class ValueWorld:
    features: np.ndarray                    # [N_OPTIONS, N_FEATURES]
    preference: np.ndarray                  # [N_FEATURES]
    audience_cost: np.ndarray               # [N_OPTIONS] how visible each option is
    effort: np.ndarray                      # [N_OPTIONS]
    signal_value: np.ndarray                # [N_OPTIONS]
    concealment: float = 0.0
    competence: float = 0.85
    signalling: float = 0.0
    cost_function: str = "linear"
    temperature: float = 0.6
    meta: dict = field(default_factory=dict)

    def copy_with(self, **kw) -> "ValueWorld":
        d = {"features": self.features, "preference": self.preference,
             "audience_cost": self.audience_cost, "effort": self.effort,
             "signal_value": self.signal_value, "concealment": self.concealment,
             "competence": self.competence, "signalling": self.signalling,
             "cost_function": self.cost_function, "temperature": self.temperature}
        d.update(kw)
        return ValueWorld(**d)


def sample_value_world(rng, concealment: float = 0.0, competence: float = 0.85,
                       signalling: float = 0.0, cost_function: str = "linear",
                       temperature: float = 0.6) -> ValueWorld:
    return ValueWorld(features=rng.normal(size=(N_OPTIONS, N_FEATURES)),
                      preference=rng.normal(size=N_FEATURES),
                      audience_cost=np.abs(rng.normal(size=N_OPTIONS)),
                      effort=np.abs(rng.normal(size=N_OPTIONS)) * 0.8,
                      signal_value=rng.normal(size=N_OPTIONS) * 0.7,
                      concealment=concealment, competence=competence, signalling=signalling,
                      cost_function=cost_function, temperature=temperature)


def _effort_term(w: ValueWorld) -> np.ndarray:
    if w.cost_function == "quadratic":
        return (w.effort ** 2) / max(w.competence, 1e-3)
    if w.cost_function == "threshold":
        return np.where(w.effort > np.median(w.effort), 2.0, 0.2) / max(w.competence, 1e-3)
    return w.effort / max(w.competence, 1e-3)


def utility(w: ValueWorld, public: bool = True, available=None) -> np.ndarray:
    """One equation for every maker. Concealment only bites where the audience is watching."""
    u = w.features @ w.preference - _effort_term(w) + w.signalling * w.signal_value
    if public:
        # Concealment is a cost on being SEEN PREFERRING: a visible option aligned with the
        # preference is pushed down, a visible option against it is pushed up. Penalising only
        # the positive part shrinks the leaders proportionally and barely reorders them, which
        # left a concealing maker's public and private policies 0.011 apart.
        u = u - 1.8 * w.concealment * w.audience_cost * (w.features @ w.preference)
    if available is not None:
        u = np.where(np.asarray(available, bool), u, -1e9)
    return u


def choose(w: ValueWorld, rng, public: bool = True, available=None) -> dict:
    u = utility(w, public, available)
    p = C.softmax(u / max(w.temperature, 1e-3))
    pick = int(rng.choice(N_OPTIONS, p=p))
    avail = np.ones(N_OPTIONS, bool) if available is None else np.asarray(available, bool)
    return {"choice": pick, "policy": p, "public": bool(public),
            "opportunity": [int(i) for i in np.flatnonzero(avail)],
            "forgone": [int(i) for i in np.flatnonzero(avail) if i != pick],
            "paid_cost": float(w.effort[pick]), "audience_cost": float(w.audience_cost[pick])}


# --------------------------------------------------------------------------- #
# The four rivals V02-V04 have to separate, all built from the same planner.
# --------------------------------------------------------------------------- #
def make_rivals(base: ValueWorld, rng) -> dict:
    """``unchanged``, ``changed_preference``, ``changed_goal``, ``concealment``, ``stale_residue``.

    ``changed_goal`` moves only this episode's weighting and reverts; ``changed_preference`` moves
    the standing vector; ``concealment`` leaves the preference alone and raises the hiding term;
    ``stale_residue`` leaves the preference alone and adds a lagging effort bias from an old skill.
    """
    flip = -base.preference + rng.normal(size=N_FEATURES) * 0.15
    episode_only = base.preference + rng.normal(size=N_FEATURES) * 1.4
    stale = base.effort + np.abs(rng.normal(size=N_OPTIONS)) * 0.9
    return {"unchanged": base,
            "changed_preference": base.copy_with(preference=flip),
            "changed_goal": base.copy_with(preference=episode_only),
            "concealment": base.copy_with(concealment=1.6),
            "stale_residue": base.copy_with(effort=stale)}


def public_private_signature(w: ValueWorld, rng, n: int = 30) -> dict:
    """The measurement V04 turns on: how far public and private choices diverge.

    Concealment separates them; a real preference change moves both together. Neither is visible
    from public choices alone, which is the point.
    """
    # Computed on the POLICIES. Sampling forty choices at a peaked temperature returns two
    # identical point masses whatever the concealment, which made this read 0.000 for the
    # concealment rival -- the same number as for the unchanged one.
    pol_pub = C.softmax(utility(w, public=True) / max(w.temperature, 1e-3))
    pol_priv = C.softmax(utility(w, public=False) / max(w.temperature, 1e-3))
    pub, priv = [], []
    for _ in range(int(n)):
        pub.append(choose(w, rng, public=True)["choice"])
        priv.append(choose(w, rng, public=False)["choice"])
    hp = np.bincount(pub, minlength=N_OPTIONS) / max(len(pub), 1)
    hv = np.bincount(priv, minlength=N_OPTIONS) / max(len(priv), 1)
    return {"public": pol_pub.tolist(), "private": pol_priv.tolist(),
            "divergence": float(C.tv(pol_pub, pol_priv)),
            "sampled_public": hp.tolist(), "sampled_private": hv.tolist(),
            "sampled_divergence": float(C.tv(hp, hv))}


def rival_posterior(observed: list, rivals: dict, rng, n_sim: int = 24, public: bool = True,
                    budget=None) -> dict:
    """Posterior over which rival produced a set of observed choices, by forward simulation."""
    def hist(choices):
        return np.bincount(np.asarray(choices, int), minlength=N_OPTIONS) / max(len(choices), 1)

    target = hist(observed)
    lls = {}
    for name, w in rivals.items():
        sims = [hist([choose(w, np.random.default_rng(rng.integers(0, 2 ** 62)), public)["choice"]
                      for _ in range(len(observed))]) for _ in range(int(n_sim))]
        m, sd = np.mean(sims, axis=0), np.std(sims, axis=0) + 0.05
        lls[name] = float(-0.5 * np.sum(((target - m) / sd) ** 2))
        if budget is not None:
            budget.lik(int(n_sim))
    keys = list(lls)
    return {k: float(v) for k, v in zip(keys, C.softmax(np.array([lls[k] for k in keys])))}


# --------------------------------------------------------------------------- #
# Cost attribution (V05, V06, V09) and reward equivalence (V08).
# --------------------------------------------------------------------------- #
def cost_vector_posterior(observed: list, base: ValueWorld, rng, n_sim: int = 20) -> dict:
    """Which owner explains a paid cost: preference, competence, constraint, signalling, or a
    different cost function. Never preference alone."""
    variants = {
        "preference": base.copy_with(preference=base.preference * 1.8),
        "competence": base.copy_with(competence=0.35),
        "constraint": base.copy_with(effort=base.effort * 2.2),
        "signalling": base.copy_with(signalling=1.8),
        "cost_function": base.copy_with(cost_function="quadratic"),
    }
    return rival_posterior(observed, variants, rng, n_sim)


def opportunity_information(w: ValueWorld, rng, n: int = 40) -> dict:
    """V06: do the forgone alternatives add anything beyond the chosen option and its scalar cost?

    Two readers are compared on recovering the preference direction: one sees choice and cost, the
    other also sees what was on the table and declined.
    """
    rows = []
    for _ in range(int(n)):
        avail = rng.random(N_OPTIONS) < 0.7
        if avail.sum() < 2:
            avail[:2] = True
        rows.append(choose(w, rng, public=True, available=avail))
    # cost-only reader: regress features of the chosen option on nothing but its own cost
    X_cost = np.array([w.features[r["choice"]] for r in rows])
    # opportunity-aware reader: chosen minus the mean of what was declined
    X_opp = np.array([w.features[r["choice"]]
                      - (w.features[r["forgone"]].mean(axis=0) if r["forgone"]
                         else np.zeros(N_FEATURES)) for r in rows])
    truth = C.normalize(np.abs(w.preference))

    def align(X):
        v = X.mean(axis=0)
        if np.linalg.norm(v) < 1e-9:
            return 0.0
        return float(np.dot(v / np.linalg.norm(v),
                            w.preference / np.linalg.norm(w.preference)))
    return {"cost_only_alignment": align(X_cost), "opportunity_alignment": align(X_opp),
            "advantage": align(X_opp) - align(X_cost), "n": len(rows)}


def feasible_reward_set(observed: list, w: ValueWorld, rng, n_draw: int = 400,
                        public: bool = True) -> dict:
    """V08: every reward vector that predicts *every* observed choice, kept as a class.

    Returned as coverage and diameter rather than a point estimate, because forcing a unique value
    where the record cannot identify one is the failure this trunk exists to avoid.
    """
    cands = rng.normal(size=(int(n_draw), N_FEATURES))
    keep = []
    for c in cands:
        w2 = w.copy_with(preference=c)
        u = utility(w2, public)
        ok = all(int(np.argmax(u)) == ch or u[ch] >= np.quantile(u, 0.7) for ch in observed)
        if ok:
            keep.append(c)
    if not keep:
        return {"n_feasible": 0, "coverage": 0.0, "diameter": float("nan"),
                "contains_truth": False}
    K = np.array(keep)
    unit = K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-9)
    tru = w.preference / max(np.linalg.norm(w.preference), 1e-9)
    return {"n_feasible": len(keep), "coverage": float(len(keep) / n_draw),
            "diameter": float(np.max(unit @ unit.T * -1 + 1)),
            "mean_alignment_to_truth": float(np.mean(unit @ tru)),
            "contains_truth": bool(np.max(unit @ tru) > 0.9)}


def suboptimality_shrinks_set(w: ValueWorld, rng, n: int = 24) -> dict:
    """V09: do mistakes shrink the compatible preference set, or only optimal choices?

    Three records of the same length: optimal-only, varied-competence, and one containing outright
    errors. Fewer feasible reward vectors means a more informative record.
    """
    out = {}
    for name, comp in (("optimal_only", 0.99), ("varied_competence", 0.7), ("with_errors", 0.45)):
        w2 = w.copy_with(competence=comp)
        obs = [choose(w2, rng, public=True)["choice"] for _ in range(int(n))]
        out[name] = feasible_reward_set(obs, w2, rng)
    return out


def dated_trajectory(base: ValueWorld, rng, n_episodes: int = 10,
                     change_at: int | None = 5) -> dict:
    """V07: a directional change, observed as dated works and as an undated bag."""
    w = base
    eps = []
    for t in range(int(n_episodes)):
        if change_at is not None and t == change_at:
            w = w.copy_with(preference=-base.preference)
        eps.append({"t": t, "choice": choose(w, rng, public=True)["choice"]})
    return {"episodes": eps, "change_at": change_at,
            "dated": [(e["t"], e["choice"]) for e in eps],
            "bag": sorted(e["choice"] for e in eps)}


def change_point_score(traj: dict) -> dict:
    """Where a dated record says the change happened, and what a bag can say (nothing)."""
    ch = [c for _, c in traj["dated"]]
    n = len(ch)
    best, best_t = -1e18, None
    for t in range(2, n - 1):
        a = np.bincount(ch[:t], minlength=N_OPTIONS) + 0.5
        b = np.bincount(ch[t:], minlength=N_OPTIONS) + 0.5
        sep = C.tv(C.normalize(a), C.normalize(b))
        if sep > best:
            best, best_t = sep, t
    return {"detected_at": best_t, "separation": float(best),
            "truth": traj["change_at"],
            "error": abs((best_t or 0) - (traj["change_at"] or 0)),
            "bag_can_detect": False}
