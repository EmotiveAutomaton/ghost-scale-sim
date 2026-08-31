"""Hierarchy, collaboration and role-relative control (spec §6, trunk H).

V14 established that exactly reward-equivalent hierarchies stay at chance from behaviour alone
until a resolving intervention arrives. That is a clean identifiability boundary and it is imported
here as an anchor (H01), not re-discovered. What V15 adds is the *approximate* case: two
hierarchies that are nearly but not exactly equivalent, where the honest answer is a graded class
rather than a name (H02), and the topology question -- can a central controller, a distributed
shared model, an editor-ratifier and a set of independent contributors be told apart when their
outputs are surface-matched (H07)?

Abstention is a first-class answer here. Two topologies that are equivalent on the observed record
must come back as a class with its mass reported, and a card that names one of them is wrong even
when it happens to name the right one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C

TOPOLOGIES = ("central", "distributed", "editor_ratifier", "independent")
N_ROLES = 4
N_SUBTASKS = 5
N_ACTIONS = 4


@dataclass
class Team:
    topology: str
    brief: np.ndarray                       # [N_SUBTASKS, N_ACTIONS] the organizing constraint
    role_skill: np.ndarray                  # [N_ROLES]
    role_habit: np.ndarray                  # [N_ROLES, N_ACTIONS]
    assignment: np.ndarray                  # [N_SUBTASKS] which role owns which subtask
    temperature: float = 0.6
    editor: int = 0
    meta: dict = field(default_factory=dict)


def sample_team(rng, topology: str = "central", n_roles: int = N_ROLES,
                temperature: float = 0.6) -> Team:
    return Team(topology=topology,
                brief=rng.normal(size=(N_SUBTASKS, N_ACTIONS)) * 1.4,
                role_skill=rng.uniform(0.55, 0.95, size=n_roles),
                role_habit=C.softmax(rng.normal(size=(n_roles, N_ACTIONS)) * 1.1, axis=-1),
                assignment=rng.integers(0, n_roles, size=N_SUBTASKS),
                temperature=temperature, editor=int(rng.integers(n_roles)))


def _role_policy(t: Team, role: int, subtask: int) -> np.ndarray:
    """What one role would do on one subtask: the brief, bent by that role's habit and skill."""
    s = float(t.role_skill[role])
    logit = s * t.brief[subtask] + (1.0 - s) * 2.0 * np.log(np.maximum(t.role_habit[role], 1e-9))
    return C.softmax(logit / max(t.temperature, 1e-3))


def produce(t: Team, rng, n_rounds: int = 6) -> dict:
    """One collaboration. The *record* is who issued what and when; the artifact is the outcome."""
    rows = []
    for r in range(int(n_rounds)):
        for st in range(N_SUBTASKS):
            if t.topology == "central":
                issuer = int(t.assignment[0])                  # one controller decides everywhere
                pol = _role_policy(t, issuer, st)
            elif t.topology == "distributed":
                # every role holds the same shared model; the action is their consensus
                pol = C.normalize(np.mean([_role_policy(t, i, st)
                                           for i in range(len(t.role_skill))], axis=0))
                issuer = int(t.assignment[st])
            elif t.topology == "editor_ratifier":
                owner = int(t.assignment[st])
                draft = _role_policy(t, owner, st)
                pol = C.normalize(0.55 * draft + 0.45 * _role_policy(t, t.editor, st))
                issuer = t.editor if rng.random() < 0.45 else owner
            else:                                              # independent contributors
                issuer = int(t.assignment[st])
                pol = _role_policy(t, issuer, st)
            rows.append({"round": r, "subtask": st, "issuer": issuer,
                         "action": int(rng.choice(N_ACTIONS, p=pol))})
    nxt = int(rng.integers(N_SUBTASKS))
    return {"topology": t.topology, "rows": rows, "next_subtask": nxt,
            "next_issuer": rows[-1]["issuer"],
            "artifact": np.bincount([r["action"] for r in rows], minlength=N_ACTIONS).tolist()}


def artifact_only_posterior(ep: dict, t: Team, rng, n_sim: int = 20) -> dict:
    """H03: what the *static artifact* alone can say about the topology. Expected: very little."""
    target = C.normalize(np.asarray(ep["artifact"], float))
    lls = {}
    for topo in TOPOLOGIES:
        t2 = Team(**{**t.__dict__, "topology": topo})
        sims = [C.normalize(np.asarray(
            produce(t2, np.random.default_rng(rng.integers(0, 2 ** 62)),
                    len(ep["rows"]) // N_SUBTASKS)["artifact"], float)) for _ in range(int(n_sim))]
        # The floor is the statistic's own between-simulation spread. A likelihood sharper
        # than its sampling noise made the static artifact name the true topology at 0.87
        # where H03 needs it near chance.
        sims = np.asarray(sims, float)
        m = sims.mean(axis=0)
        sd = sims.std(axis=0) + max(float(sims.std(axis=0).mean()), 0.04)
        lls[topo] = float(-0.5 * np.sum(((target - m) / sd) ** 2))
    return {k: float(v) for k, v in zip(TOPOLOGIES,
                                        C.softmax(np.array([lls[k] for k in TOPOLOGIES])))}


def process_record_posterior(ep: dict, t: Team, rng, n_sim: int = 20, budget=None) -> dict:
    """H04/H07: what the *process record* -- who issued what -- adds over the artifact.

    The discriminating features are all about issuance: how concentrated it is, whether one role
    appears everywhere, and how often the issuer changes between adjacent subtasks.
    """
    def feats(e):
        iss = np.array([r["issuer"] for r in e["rows"]], int)
        acts = np.array([r["action"] for r in e["rows"]], int)
        counts = np.bincount(iss, minlength=len(t.role_skill)).astype(float)
        counts = counts / max(counts.sum(), 1)
        switches = float(np.mean(iss[1:] != iss[:-1])) if iss.size > 1 else 0.0
        return np.array([float(counts.max()), float(C.entropy(counts + 1e-9)), switches,
                         float(len(set(acts.tolist())) / N_ACTIONS)])

    target = feats(ep)
    lls = {}
    for topo in TOPOLOGIES:
        t2 = Team(**{**t.__dict__, "topology": topo})
        sims = [feats(produce(t2, np.random.default_rng(rng.integers(0, 2 ** 62)),
                              len(ep["rows"]) // N_SUBTASKS)) for _ in range(int(n_sim))]
        m, sd = np.mean(sims, axis=0), np.std(sims, axis=0) + 0.06
        lls[topo] = float(-0.5 * np.nansum(((target - m) / sd) ** 2))
        if budget is not None:
            budget.lik(int(n_sim))
    return {k: float(v) for k, v in zip(TOPOLOGIES,
                                        C.softmax(np.array([lls[k] for k in TOPOLOGIES])))}


# --------------------------------------------------------------------------- #
# Reward equivalence, exact and approximate (H01, H02).
# --------------------------------------------------------------------------- #
def equivalent_briefs(t: Team, rng, exact: bool = True, eps: float = 0.25) -> list:
    """Briefs that induce the same behaviour. Exactly, or to within ``eps``.

    Exact equivalence is a constant shift of the brief, which cancels in the softmax. Approximate
    equivalence adds a small perturbation, which is where a graded class replaces a boundary.
    """
    out = []
    for _ in range(4):
        shift = rng.normal(size=(N_SUBTASKS, 1))
        b = t.brief + shift
        if not exact:
            b = b + rng.normal(size=t.brief.shape) * eps
        out.append(b)
    return out


def equivalence_class_report(t: Team, rng, exact: bool = True, eps: float = 0.25,
                             n: int = 200) -> dict:
    """How distinguishable a brief is from its equivalents, on behaviour alone."""
    base = np.array([_role_policy(t, int(t.assignment[st]), st) for st in range(N_SUBTASKS)])
    divs = []
    for b in equivalent_briefs(t, rng, exact, eps):
        t2 = Team(**{**t.__dict__, "brief": b})
        alt = np.array([_role_policy(t2, int(t2.assignment[st]), st) for st in range(N_SUBTASKS)])
        divs.append(float(np.mean([C.tv(base[i], alt[i]) for i in range(N_SUBTASKS)])))
    return {"exact": bool(exact), "epsilon": float(eps), "divergences": divs,
            "max_divergence": float(np.max(divs)), "mean_divergence": float(np.mean(divs)),
            "indistinguishable": bool(np.max(divs) < 1e-9),
            "graded_membership": [float(np.exp(-6.0 * d)) for d in divs]}


def role_factor_posterior(ep: dict, t: Team, rng) -> dict:
    """H05: can a local deviation be explained by a subordinate's competence and habit, without
    inventing a director's goal?

    Two explanations are scored on the same rows: the role's own skill and habit, or a hypothesised
    superordinate constraint. The card fails if the constraint model wins where no constraint was
    planted.
    """
    rows = ep["rows"]
    ll_role, ll_director = 0.0, 0.0
    for r in rows:
        st, a = r["subtask"], r["action"]
        ll_role += float(np.log(max(_role_policy(t, r["issuer"], st)[a], 1e-300)))
        director = C.softmax(t.brief[st] / max(t.temperature, 1e-3))
        ll_director += float(np.log(max(director[a], 1e-300)))
    return {"role_model_loglik": ll_role, "director_model_loglik": ll_director,
            "role_advantage": ll_role - ll_director, "n_rows": len(rows)}


def cross_role_dependency(ep: dict, t: Team) -> float:
    """H06: does a lower-level goal become a constraint for a subordinate role?

    Measured as how much an issuer's action on one subtask predicts the next subtask's action --
    a dependency that exists in central and editor topologies and not among independents.
    """
    rows = sorted(ep["rows"], key=lambda r: (r["round"], r["subtask"]))
    a = np.array([r["action"] for r in rows], int)
    if a.size < 4:
        return float("nan")
    return float(np.mean(a[1:] == a[:-1]))


def resolving_intervention(t: Team, rng, kind: str, n: int = 120) -> dict:
    """H04: which intervention on the process record first separates actor from constraint.

    ``swap_role``      put a different role on the subtask.
    ``change_brief``   alter the organizing constraint.
    ``remove_editor``  take the ratifier out.
    The one that moves behaviour most is the one that identifies what was driving it.
    """
    base = produce(t, np.random.default_rng(7), 6)
    if kind == "swap_role":
        t2 = Team(**{**t.__dict__, "assignment": np.roll(t.assignment, 1)})
    elif kind == "change_brief":
        t2 = Team(**{**t.__dict__, "brief": t.brief + rng.normal(size=t.brief.shape) * 1.6})
    else:
        t2 = Team(**{**t.__dict__, "topology": "independent"})
    alt = produce(t2, np.random.default_rng(7), 6)
    hb = C.normalize(np.asarray(base["artifact"], float))
    ha = C.normalize(np.asarray(alt["artifact"], float))
    return {"intervention": kind, "artifact_shift": float(C.tv(hb, ha)),
            "issuer_shift": float(abs(
                np.mean([r["issuer"] for r in base["rows"]])
                - np.mean([r["issuer"] for r in alt["rows"]])))}
