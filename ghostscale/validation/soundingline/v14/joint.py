"""Exact joint, staged, independent and oracle estimators over the (plan, goal, preference) grid
(spec §4.1), with matched information and compute.

THE GRID. A state is (process plan, CURRENT episode goal, standing preference). Past episodes had
their own goals, drawn from the preference; their evidence informs the plan and the preference
with those goals integrated out, and the current episode's evidence conditions the grid's goal.

THE COMPUTE MATCH. Every estimator consumes the same per-route log-likelihood table, computed
once per unit and shared; ``evaluations`` reports the count so the equality is asserted, not
assumed. Estimators differ only in how they combine the table:

    joint         the exact posterior over the 96 states from every route;
    independent   each latent from its own route only (plan: action + forensic; goal: semantic;
                  preference: context), the other latents integrated under the prior - the
                  ablation of cross-latent messages;
    staged        a plug-in pipeline in a declared order: the first latent's marginal (from every
                  route) is committed to its mode, the next latent is inferred with that
                  commitment fixed, and so on. Under exact inference a soft handoff collapses
                  into the joint, so the commitment IS the staging (docs/versions/v14-routed-reader/SPEC.md,
                  judgment call 1);
    oracle        the true values of the other two latents are supplied.
Baselines (frequency, last choice, surface centroid) live in the cards that use them.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from .world import (N_ACT, N_FEAT, N_GOAL, N_PLAN, N_PREF, N_DECISIONS, Family, Maker, World, equivalent)

_TINY = 1e-300
LATENTS = ("process", "goal", "preference")
STATES = [(pl, g, pr) for pl in range(N_PLAN) for g in range(N_GOAL) for pr in range(N_PREF)]
N_STATES = len(STATES)
_PL = np.array([s[0] for s in STATES])
_G = np.array([s[1] for s in STATES])
_PR = np.array([s[2] for s in STATES])
_IDX = {s: i for i, s in enumerate(STATES)}
_AXIS = {"process": _PL, "goal": _G, "preference": _PR}
_SIZE = {"process": N_PLAN, "goal": N_GOAL, "preference": N_PREF}


def state_index(plan: int, goal: int, pref: int) -> int:
    return _IDX[(int(plan), int(goal), int(pref))]


def truth_of(m: Maker, ep: dict | None = None) -> tuple:
    """(plan, goal of the given or current episode, preference)."""
    return (int(m.plan), int(ep["goal"]) if ep is not None else int(m.goal), int(m.pref))


class Reader:
    """A reader's likelihood is its own model of execution: its competence parameters are the
    noise it assumes. ``family`` selects the world structure it reads with."""

    def __init__(self, world: World, family: int, k_exec: float = 0.75, k_obs: float = 0.8, name: str = "reader", template_blur: float = 0.0):
        self.world, self.fam_index, self.k_exec, self.k_obs, self.name = world, int(family), float(k_exec), float(k_obs), name
        self.fam: Family = world.family(family)
        self.inv_vocab = np.argsort(self.fam.vocab)
        # the reader's competence is the quality of its plan templates: a blurred template is a
        # less competent reader (its expertise is its likelihood); execution noise stays matched
        self.template_blur = float(template_blur)
        b = self.template_blur
        self.plan = (1 - b) * self.fam.plan + b / N_ACT
        self.plan0 = (1 - b) * self.fam.plan0 + b / N_ACT
        # log P(goal | preference): (N_GOAL, N_PREF)
        gd = np.stack([C.softmax(world.params.goal_temp * np.log(self.fam.prefs[pr] + 1e-9)) for pr in range(N_PREF)], axis=1)
        self.log_gd = np.log(gd + _TINY)

    # ---- per-route log-likelihood tables for ONE episode over the full grid ------------------ #
    def ll_action(self, ep: dict, k_exec: float | None = None) -> np.ndarray:
        ke = self.k_exec if k_exec is None else float(k_exec)
        acts = [int(self.inv_vocab[a]) for a in ep["action"]]
        plan, plan0 = self.plan, self.plan0
        out = np.zeros((N_PLAN, N_GOAL))
        for pl in range(N_PLAN):
            for g in range(N_GOAL):
                ll = np.log(ke * plan0[pl, g, acts[0]] + (1 - ke) / N_ACT)
                for t in range(1, len(acts)):
                    ll += np.log(ke * plan[pl, g, acts[t - 1], acts[t]] + (1 - ke) / N_ACT)
                out[pl, g] = ll
        return np.repeat(out[:, :, None], N_PREF, axis=2).reshape(-1)

    def ll_semantic(self, ep: dict, k_obs: float | None = None) -> np.ndarray:
        ko = self.k_obs if k_obs is None else float(k_obs)
        per_goal = np.zeros(N_GOAL)
        for g in range(N_GOAL):
            per_goal[g] = sum(np.log(ko * self.fam.sem[g][tok] + (1 - ko) / N_FEAT) for tok in ep["semantic"])
        return per_goal[_G]

    def ll_context(self, ep: dict) -> np.ndarray:
        per_pref = np.zeros(N_PREF)
        temp = self.world.params.context_temp
        for pr in range(N_PREF):
            ll = 0.0
            for d, ch in enumerate(ep["context"]["choices"][:N_DECISIONS]):
                ll += np.log(C.softmax(temp * (self.fam.align[d] @ self.fam.prefs[pr]))[ch] + _TINY)
            per_pref[pr] = ll
        return per_pref[_PR]

    def ll_forensic(self, ep: dict) -> np.ndarray:
        acc = self.world.params.forensic_acc
        head = ep["forensic"]["intended_head"]
        sig = int(ep["forensic"]["signature"])
        out = np.zeros((N_PLAN, N_GOAL))
        for pl in range(N_PLAN):
            for g in range(N_GOAL):
                ll = np.log(self.plan0[pl, g, head[0]] + _TINY)
                if len(head) > 1:
                    ll += np.log(self.plan[pl, g, head[0], head[1]] + _TINY)
                ll += np.log(acc if sig == pl else (1 - acc) / (N_PLAN - 1))
                out[pl, g] = ll
        return np.repeat(out[:, :, None], N_PREF, axis=2).reshape(-1)

    def collapse_past(self, ll: np.ndarray) -> np.ndarray:
        """A past episode's table: integrate its goal out under P(goal | preference) and broadcast
        the result over the current goal axis."""
        cube = ll.reshape(N_PLAN, N_GOAL, N_PREF) + self.log_gd[None, :, :]
        m = cube.max(axis=1, keepdims=True)
        collapsed = m[:, 0, :] + np.log(np.exp(cube - m).sum(axis=1))          # (N_PLAN, N_PREF)
        return np.repeat(collapsed[:, None, :], N_GOAL, axis=1).reshape(-1)

    def episode_tables(self, ep: dict, routes, k_exec: float | None = None) -> dict:
        out = {}
        if "action" in routes:
            out["action"] = self.ll_action(ep, k_exec)
        if "semantic" in routes:
            out["semantic"] = self.ll_semantic(ep)
        if "context" in routes:
            out["context"] = self.ll_context(ep)
        if "forensic" in routes:
            out["forensic"] = self.ll_forensic(ep)
        return out

    def route_tables(self, eps: list, routes=("action", "semantic", "context"), k_exec: float | None = None) -> dict:
        """Summed per-route log-likelihoods; the last episode is the current one (its goal is the
        grid's goal), earlier episodes are collapsed. Computed once and shared by every estimator."""
        tabs = {r: np.zeros(N_STATES) for r in routes}
        for i, ep in enumerate(eps):
            step = self.episode_tables(ep, routes, k_exec)
            current = i == len(eps) - 1
            for r in routes:
                tabs[r] += step[r] if current else self.collapse_past(step[r])
        return tabs


def evaluations(tabs: dict) -> int:
    """The compute every estimator consumed: one table evaluation per state per route."""
    return int(N_STATES * len(tabs))


# --------------------------------------------------------------------------- #
# Priors and marginals.
# --------------------------------------------------------------------------- #
def uniform_prior() -> np.ndarray:
    return np.full(N_STATES, 1.0 / N_STATES)


def prior_from(plan_p=None, goal_p=None, pref_p=None) -> np.ndarray:
    pp = np.full(N_PLAN, 1.0 / N_PLAN) if plan_p is None else np.asarray(plan_p, float)
    gp = np.full(N_GOAL, 1.0 / N_GOAL) if goal_p is None else np.asarray(goal_p, float)
    rp = np.full(N_PREF, 1.0 / N_PREF) if pref_p is None else np.asarray(pref_p, float)
    p = pp[_PL] * gp[_G] * rp[_PR]
    return p / p.sum()


def marginal(post: np.ndarray, which: str) -> np.ndarray:
    out = np.zeros(_SIZE[which])
    np.add.at(out, _AXIS[which], post)
    return out


def posterior(prior: np.ndarray, ll: np.ndarray) -> np.ndarray:
    return C.softmax(np.log(np.maximum(prior, _TINY)) + ll)


def combined(tabs: dict, weights: dict | None = None) -> np.ndarray:
    ll = np.zeros(N_STATES)
    for r, t in tabs.items():
        ll += (1.0 if weights is None else float(weights.get(r, 1.0))) * t
    return ll


# --------------------------------------------------------------------------- #
# Estimators.
# --------------------------------------------------------------------------- #
def joint(prior: np.ndarray, tabs: dict, weights: dict | None = None) -> np.ndarray:
    return posterior(prior, combined(tabs, weights))


ROUTE_OF = {"process": ("action", "forensic"), "goal": ("semantic",), "preference": ("context",)}


def independent(prior: np.ndarray, tabs: dict) -> np.ndarray:
    """Each latent from its own route(s) only; the product of the three marginals as a joint."""
    margs = {}
    for lat, routes in ROUTE_OF.items():
        ll = np.zeros(N_STATES)
        for r in routes:
            if r in tabs:
                ll += tabs[r]
        margs[lat] = marginal(posterior(prior, ll), lat)
    return prior_from(margs["process"], margs["goal"], margs["preference"])


def staged(prior: np.ndarray, tabs: dict, order=("goal", "process", "preference"), weights: dict | None = None) -> np.ndarray:
    """Plug-in pipeline: commit each latent to its mode before inferring the next."""
    full = posterior(prior, combined(tabs, weights))
    mask = np.ones(N_STATES, bool)
    for lat in order[:-1]:
        sub = full * mask
        sub = sub / max(sub.sum(), _TINY)
        mode = int(np.argmax(marginal(sub, lat)))
        mask &= _AXIS[lat] == mode
    out = full * mask
    return out / max(out.sum(), _TINY)


def oracle(prior: np.ndarray, tabs: dict, truth: tuple, which: str) -> np.ndarray:
    """Posterior over ``which`` with the other two latents fixed at their true values."""
    mask = np.ones(N_STATES, bool)
    for lat, val in zip(LATENTS, truth):
        if lat != which:
            mask &= _AXIS[lat] == int(val)
    pr = prior * mask
    post = posterior(pr / max(pr.sum(), _TINY), combined(tabs))
    return marginal(post, which)


ESTIMATORS = ("independent", "goal_process_preference", "process_goal_preference", "preference_goal_process", "joint")


def estimate(name: str, prior: np.ndarray, tabs: dict, weights: dict | None = None) -> np.ndarray:
    if name == "joint":
        return joint(prior, tabs, weights)
    if name == "independent":
        return independent(prior, tabs)
    return staged(prior, tabs, tuple(name.split("_")), weights)


# --------------------------------------------------------------------------- #
# Predictions from a posterior (the prospective endpoints).
# --------------------------------------------------------------------------- #
def next_action_dist(reader: Reader, post: np.ndarray, last_action: int, k_exec: float | None = None) -> np.ndarray:
    """The next action WITHIN the current episode (the grid's goal applies), in vocabulary labels."""
    ke = reader.k_exec if k_exec is None else float(k_exec)
    last = int(reader.inv_vocab[last_action])
    out = np.zeros(N_ACT)
    plan = reader.plan
    for i, (pl, g, _) in enumerate(STATES):
        if post[i] > 1e-12:
            out += post[i] * (ke * plan[pl, g, last] + (1 - ke) / N_ACT)
    out = out / max(out.sum(), _TINY)
    return out[reader.inv_vocab]


def next_episode_second_action_dist(reader: Reader, post: np.ndarray, first_action: int, k_exec: float | None = None) -> np.ndarray:
    """The second action of the NEXT episode given its first, the goal redrawn from the preference."""
    ke = reader.k_exec if k_exec is None else float(k_exec)
    first = int(reader.inv_vocab[first_action])
    gd = np.exp(reader.log_gd)
    out = np.zeros(N_ACT)
    for i, (pl, _, pr) in enumerate(STATES):
        if post[i] > 1e-12:
            nxt = gd[:, pr] @ reader.plan[pl][:, first, :]
            out += post[i] * (ke * nxt + (1 - ke) / N_ACT)
    out = out / max(out.sum(), _TINY)
    return out[reader.inv_vocab]


def next_episode_action_dist(reader: Reader, post: np.ndarray, k_exec: float | None = None) -> np.ndarray:
    """The first action of the NEXT episode: its goal is redrawn from the preference."""
    ke = reader.k_exec if k_exec is None else float(k_exec)
    gd = np.exp(reader.log_gd)                                   # (N_GOAL, N_PREF)
    out = np.zeros(N_ACT)
    for i, (pl, _, pr) in enumerate(STATES):
        if post[i] > 1e-12:
            first = gd[:, pr] @ reader.plan0[pl]                 # (N_ACT,)
            out += post[i] * (ke * first + (1 - ke) / N_ACT)
    out = out / max(out.sum(), _TINY)
    return out[reader.inv_vocab]


def next_goal_dist(reader: Reader, post: np.ndarray) -> np.ndarray:
    pm = marginal(post, "preference")
    out = np.exp(reader.log_gd) @ pm
    return out / out.sum()


def next_choice_dist(reader: Reader, post: np.ndarray, decision: int) -> np.ndarray:
    pm = marginal(post, "preference")
    out = np.zeros(reader.fam.align.shape[1])
    temp = reader.world.params.context_temp
    for pr in range(N_PREF):
        out += pm[pr] * C.softmax(temp * (reader.fam.align[decision] @ reader.fam.prefs[pr]))
    return out / out.sum()


def class_mass(post: np.ndarray, truth: tuple) -> float:
    """Posterior mass on the process-equivalence class of the truth (same preference)."""
    pl, g, pr = truth
    return float(sum(post[state_index(a, b, pr)] for a, b in equivalent(pl, g)))


def top_state_correct(post: np.ndarray, truth: tuple, up_to_class: bool = True) -> bool:
    s = STATES[int(np.argmax(post))]
    if s == tuple(int(x) for x in truth):
        return True
    return up_to_class and s[2] == truth[2] and (s[0], s[1]) in equivalent(truth[0], truth[1])


# --------------------------------------------------------------------------- #
# Trajectories (spec §4.2): the posterior after every episode.
# --------------------------------------------------------------------------- #
def trajectory(reader: Reader, prior: np.ndarray, eps: list, m: Maker, routes=("action", "semantic", "context"),
               estimator: str = "joint") -> list:
    out = []
    for dose in range(1, len(eps) + 1):
        tabs = reader.route_tables(eps[:dose], routes)
        post = estimate(estimator, prior, tabs)
        truth = truth_of(m, eps[dose - 1])
        out.append({"dose": dose, "entropy": C.entropy(post), "p_truth": float(post[state_index(*truth)]),
                    "class_mass": class_mass(post, truth), "top1": bool(top_state_correct(post, truth)),
                    "goal_p": float(marginal(post, "goal")[truth[1]]), "process_p": float(marginal(post, "process")[truth[0]]),
                    "pref_p": float(marginal(post, "preference")[truth[2]])})
    return out


def first_improving_dose(traj: list, key: str, baseline: float) -> int | None:
    for row in traj:
        if row[key] > baseline:
            return int(row["dose"])
    return None
