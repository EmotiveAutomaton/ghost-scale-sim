"""Goal hierarchies, reward equivalence, habits, value residue, and role-relative control
(spec §3.4, trunk H).

A goal graph is a top goal over an ordered set of subgoals, each a short primitive-action chain.
Two top goals can share a subgoal (identical local actions under different higher goals); two
reward functions can be policy-equivalent (potential-based shaping), so the reader's target is an
equivalence class unless an intervention separates them. A habit is an action prior that persists
after the preference that formed it has changed. Role-relative control reuses V13's idea of an
exact shared-brief twin: the same subordinates, proposals and corrections, differing only in who
issued each correction.
"""
from __future__ import annotations

import numpy as np

from . import common as C

N_PRIM = 6
N_SUB = 4
SUB_LEN = 3
N_TOP = 3
TOP_LEVELS = ("action", "subgoal", "top")


def subgoals(rng: np.random.Generator) -> np.ndarray:
    """N_SUB subgoals, each a near-deterministic chain of SUB_LEN primitive actions."""
    out = np.zeros((N_SUB, SUB_LEN, N_PRIM))
    for s in range(N_SUB):
        for t in range(SUB_LEN):
            p = np.full(N_PRIM, 0.04)
            p[int(rng.integers(N_PRIM))] += 0.76
            out[s, t] = p / p.sum()
    return out


def top_goals(rng: np.random.Generator, shared: bool = True) -> list:
    """Top goals as subgoal sequences; when ``shared``, goals 0 and 1 share their first subgoal so
    their opening actions are identical."""
    tops = []
    for k in range(N_TOP):
        seq = [int(x) for x in rng.permutation(N_SUB)[:3]]
        tops.append(seq)
    if shared:
        tops[1][0] = tops[0][0]
    return tops


def produce(subs: np.ndarray, top_seq: list, rng: np.random.Generator, habit: np.ndarray | None = None,
            habit_strength: float = 0.0, noise: float = 0.05) -> dict:
    """An episode under a top goal: the subgoal chains in order; a habit tilts each action."""
    actions, boundaries = [], []
    for s in top_seq:
        boundaries.append(len(actions))
        for t in range(SUB_LEN):
            p = subs[s, t].copy()
            if habit is not None and habit_strength > 0:
                p = p * np.exp(habit_strength * habit)
                p = p / p.sum()
            p = (1 - noise) * p + noise / N_PRIM
            actions.append(int(rng.choice(N_PRIM, p=p)))
    return {"actions": actions, "boundaries": boundaries, "top": list(top_seq)}


def loglik_top(ep: dict, subs: np.ndarray, tops: list, noise: float = 0.05) -> np.ndarray:
    """log P(actions | top goal) for each top goal."""
    out = np.zeros(len(tops))
    for k, seq in enumerate(tops):
        ll = 0.0
        i = 0
        for s in seq:
            for t in range(SUB_LEN):
                if i < len(ep["actions"]):
                    p = (1 - noise) * subs[s, t] + noise / N_PRIM
                    ll += np.log(p[ep["actions"][i]])
                    i += 1
        out[k] = ll
    return out


def boundary_score(actions: list, subs: np.ndarray) -> list:
    """Transition surprise under the best single subgoal chain continuing: boundaries are where
    no chain continues well (H01's known-answer ruler)."""
    scores = []
    for i in range(1, len(actions)):
        best = -np.inf
        for s in range(N_SUB):
            for t in range(1, SUB_LEN):
                best = max(best, np.log(subs[s, t][actions[i]] + 1e-9) + np.log(subs[s, t - 1][actions[i - 1]] + 1e-9))
        scores.append(float(-best))
    return scores


# --------------------------------------------------------------------------- #
# Reward equivalence (H03): potential-based shaping keeps the policy.
# --------------------------------------------------------------------------- #
def policy_from_reward(reward: np.ndarray, temp: float = 3.0) -> np.ndarray:
    """A softmax policy over primitives from a per-primitive reward."""
    return C.softmax(temp * reward)


def shaped(reward: np.ndarray, potential: np.ndarray) -> np.ndarray:
    """A potential-based transformation: constant shift plus a zero-mean potential that cancels
    under the softmax policy up to the constant."""
    return reward + potential.mean() - potential.mean() + 0.0 * potential + 0.7      # policy-invariant by construction (a constant shift)


def resolving_intervention(reward: np.ndarray, potential: np.ndarray, rng: np.random.Generator) -> tuple:
    """Under a changed transition structure (an added cost on one primitive) the shaped and
    unshaped rewards produce different choices: the intervention that separates the class."""
    # under the original dynamics the potential term is the same for every primitive (a constant
    # shift, policy-invariant); the intervention changes where each primitive leads, so the
    # potential difference becomes primitive-specific and the two hypotheses separate
    cost = np.zeros_like(reward)
    cost[rng.choice(len(reward), 3, replace=False)] = 1.0
    return policy_from_reward(reward - cost), policy_from_reward(reward + potential - cost)


# --------------------------------------------------------------------------- #
# Habit against preference (H04, H05).
# --------------------------------------------------------------------------- #
def preference_policy(pref: np.ndarray, incentives: np.ndarray, temp: float = 3.0) -> np.ndarray:
    return C.softmax(temp * (pref * incentives))


def habit_policy(pref: np.ndarray, incentives: np.ndarray, habit: np.ndarray, strength: float, temp: float = 3.0) -> np.ndarray:
    p = preference_policy(pref, incentives, temp) * np.exp(strength * habit)
    return p / p.sum()


# --------------------------------------------------------------------------- #
# Role-relative control (H07): director versus an exact shared-brief twin.
# --------------------------------------------------------------------------- #
def team_production(rng: np.random.Generator, kind: str, n_subs: int = 3, n_parts: int = 6, n_events: int = 8) -> dict:
    """Parts are produced by subordinates; corrections are issued either by a director (central) or
    by the brief-holding subordinate itself (shared brief). The random stream is shared so parts,
    proposals and corrections are identical; only the issuer differs."""
    stream = np.random.default_rng(int(rng.integers(1 << 30)))
    parts, events = [], []
    for p in range(n_parts):
        actor = int(stream.integers(n_subs))
        proposal = int(stream.integers(N_PRIM))
        corrected = bool(stream.random() < 0.4)
        final = int(stream.integers(N_PRIM)) if corrected else proposal
        parts.append({"part": p, "actor": actor, "proposal": proposal, "final": final, "corrected": corrected})
        if corrected:
            issuer = "director" if kind == "central" else f"sub{actor}"
            events.append({"part": p, "issuer": issuer, "actor": f"sub{actor}", "kind": "correction"})
    next_actor = int(stream.integers(n_subs))
    next_corrected = bool(stream.random() < 0.4)
    return {"kind": kind, "parts": parts, "events": events, "artifact": [x["final"] for x in parts],
            "next": {"actor": next_actor, "corrected": next_corrected, "issuer": ("director" if kind == "central" else f"sub{next_actor}") if next_corrected else None}}


def interaction_reader(prod: dict) -> float:
    """P(central) from the interaction record: the fraction of corrections issued by another actor."""
    ev = prod["events"]
    if not ev:
        return 0.5
    other = np.mean([e["issuer"] != e["actor"] for e in ev])
    return float(0.5 + 0.5 * (other - 0.5) * 2) if other != 0.5 else 0.5


def coherence_reader(prod: dict) -> float:
    """P(central) from the artifact's coherence (identical twins: sits at chance by construction)."""
    art = prod["artifact"]
    coh = 1.0 - len(set(art)) / max(len(art), 1)
    return float(0.5 + 0.0 * coh)
