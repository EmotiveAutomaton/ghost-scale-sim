"""The V14 constructed world (spec §3): maker state, episodes, and four disjoint evidence routes.

A maker at episode t is M_t = (K, H, V, G, Pi, B, A^m, A^a, C, R, O). Here:

    K   competence: probability an intended action is realized (p_exec) and that a semantic
        token is emitted faithfully (p_obs);
    H   attention history: a feature tilt applied to the maker's EARLY surface emissions, a
        practiced-transition tilt over action bigrams, and a reward-linked strength that decays
        after its reward reverses;
    V   standing preference: one of N_PREF profiles over the N_GOAL goal dimensions; it selects
        each episode's goal and every context-route choice;
    G   the episode goal;
    Pi  the process plan: a first-order policy over action chains conditioned on the goal. Plan
        PLAN_HABIT reproduces PLAN_DIRECT exactly under goal 0, so (Pi, G) = (0, 0) and (3, 0) are
        a process-equivalence class on every route except forensic. That is the equifinality the
        J, H and X cards require, built in rather than found;
    B, A^m, A^a, C, R  communication variables (communication.py);
    O   the opportunity set: context-route menus with per-option goal alignment.

Routes are DISJOINT observations of one episode:
    action    the realized action chain (plan, goal and competence);
    semantic  claim tokens drawn from the goal's distribution with competence noise;
    context   menu choices that follow the standing preference, plus role and source tags;
    forensic  a costly optional observation: the INTENDED first actions before competence noise
              and a plan signature token, which is what resolves plan equivalence.
Nothing here names a person; every quantity is a planted variable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C

N_ACT = 6
N_FEAT = 10
N_GOAL = 4
N_PLAN = 4
N_PREF = 6
N_SEM = 5          # semantic tokens per episode
T_STEPS = 6
N_MENU = 3         # options per context decision
N_DECISIONS = 4
PLAN_DIRECT, PLAN_EXPLORE, PLAN_STICKY, PLAN_HABIT = 0, 1, 2, 3
ROUTES = ("action", "semantic", "context", "forensic")
ROUTE_COST = {"action": 1.0, "semantic": 0.5, "context": 0.7, "forensic": 3.0}     # declared processing penalties (ease is planted, never felt)
COMPETENCE_LEVELS = {"low": (0.55, 0.6), "mid": (0.75, 0.8), "high": (0.95, 0.95)}
PLAN_EQUIVALENCE = {(PLAN_HABIT, 0): (PLAN_DIRECT, 0), (PLAN_DIRECT, 0): (PLAN_HABIT, 0)}


@dataclass
class WorldParams:
    n_families: int = 2
    vocabulary: int = 0                 # index of the action-vocabulary permutation (transfer lane uses fresh ones)
    goal_temp: float = 2.5              # how sharply the standing preference selects episode goals
    context_temp: float = 3.0           # how sharply preference selects a menu option
    forensic_acc: float = 0.9
    equifinal_x: int = 1
    h_early_steps: int = 2
    h_decay: float = 0.7


@dataclass
class Family:
    index: int
    plan: np.ndarray                    # (N_PLAN, N_GOAL, N_ACT, N_ACT) transition rows
    plan0: np.ndarray                   # (N_PLAN, N_GOAL, N_ACT) first-action distributions
    sem: np.ndarray                     # (N_GOAL, N_FEAT)
    feat: np.ndarray                    # (N_ACT, N_FEAT) surface emission per action
    prefs: np.ndarray                   # (N_PREF, N_GOAL) standing preference profiles
    align: np.ndarray                   # (N_DECISIONS, N_MENU, N_GOAL) menu option alignments
    vocab: np.ndarray                   # action relabelling for this family's vocabulary


@dataclass
class Maker:
    id: str
    family: int
    pref: int
    plan: int
    k_exec: float
    k_obs: float
    h_feat: np.ndarray                  # (N_FEAT,) attention-history tilt on early surface tokens
    h_trans: np.ndarray                 # (N_ACT, N_ACT) practiced-transition tilt (log-space)
    h_strength: float = 0.0
    h_reversed_at: int | None = None
    goal: int = 0
    comm: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    def truth(self) -> tuple:
        return (self.plan, self.goal, self.pref)


@dataclass
class World:
    wid: int
    lane: str
    params: WorldParams
    families: list

    @property
    def n_families(self) -> int:
        return len(self.families)

    def family(self, i: int) -> Family:
        return self.families[int(i)]


# --------------------------------------------------------------------------- #
# Construction.
# --------------------------------------------------------------------------- #
def _row_dirichlet(rng, n, conc):
    return rng.dirichlet(np.full(n, conc))


def make_family(index: int, rng: np.random.Generator, p: WorldParams, vocab_index: int = 0) -> Family:
    plan = np.zeros((N_PLAN, N_GOAL, N_ACT, N_ACT))
    plan0 = np.zeros((N_PLAN, N_GOAL, N_ACT))
    for g in range(N_GOAL):
        # DIRECT: peaked transitions toward a goal-specific chain
        for a in range(N_ACT):
            plan[PLAN_DIRECT, g, a] = _row_dirichlet(rng, N_ACT, 0.4)
        plan0[PLAN_DIRECT, g] = _row_dirichlet(rng, N_ACT, 0.5)
        # EXPLORE: near-uniform early, then the same goal-specific chain (mixture 50/50)
        plan[PLAN_EXPLORE, g] = 0.5 * plan[PLAN_DIRECT, g] + 0.5 / N_ACT
        plan0[PLAN_EXPLORE, g] = np.full(N_ACT, 1.0 / N_ACT)
        # STICKY: strong self-transitions
        stick = np.eye(N_ACT) * 0.6 + 0.4 * plan[PLAN_DIRECT, g]
        plan[PLAN_STICKY, g] = stick / stick.sum(axis=1, keepdims=True)
        plan0[PLAN_STICKY, g] = plan0[PLAN_DIRECT, g]
        # HABIT: its own chain under every goal except 0, where it equals DIRECT exactly
        if g == 0:
            plan[PLAN_HABIT, g] = plan[PLAN_DIRECT, g]
            plan0[PLAN_HABIT, g] = plan0[PLAN_DIRECT, g]
        else:
            for a in range(N_ACT):
                plan[PLAN_HABIT, g, a] = _row_dirichlet(rng, N_ACT, 0.4)
            plan0[PLAN_HABIT, g] = _row_dirichlet(rng, N_ACT, 0.5)
    sem = np.stack([_row_dirichlet(rng, N_FEAT, 0.5) for _ in range(N_GOAL)])
    feat = np.stack([_row_dirichlet(rng, N_FEAT, 0.7) for _ in range(N_ACT)])
    prefs = np.stack([_row_dirichlet(rng, N_GOAL, 0.8) for _ in range(N_PREF)])
    align = rng.normal(0.0, 1.0, size=(N_DECISIONS, N_MENU, N_GOAL))
    vocab = np.arange(N_ACT) if vocab_index == 0 else np.random.default_rng(C.seed(f"vocab|{vocab_index}")).permutation(N_ACT)
    return Family(index, plan, plan0, sem, feat, prefs, align, vocab)


def make_world(wid: int, lane: str, params: WorldParams | None = None, rng: np.random.Generator | None = None) -> World:
    p = params or WorldParams()
    rng = rng if rng is not None else np.random.default_rng(C.world_seed(lane, wid) + 1)
    fams = [make_family(i, rng, p, vocab_index=p.vocabulary + (i if p.vocabulary else 0)) for i in range(p.n_families)]
    return World(wid, lane, p, fams)


def competence_of(level: str) -> tuple:
    return COMPETENCE_LEVELS[level]


def make_maker(world: World, name: str, rng: np.random.Generator, family: int = 0, pref: int | None = None,
               plan: int | None = None, competence: str | None = None, k_exec: float | None = None,
               k_obs: float | None = None, h_strength: float | None = None, h_feat: np.ndarray | None = None,
               h_trans: np.ndarray | None = None) -> Maker:
    pref = int(rng.integers(N_PREF)) if pref is None else int(pref)
    plan = int(rng.integers(N_PLAN)) if plan is None else int(plan)
    if competence is not None:
        ke, ko = competence_of(competence)
    else:
        ke, ko = COMPETENCE_LEVELS[["low", "mid", "high"][int(rng.integers(3))]]
    ke = float(k_exec if k_exec is not None else ke)
    ko = float(k_obs if k_obs is not None else ko)
    hs = float(rng.uniform(0.0, 1.0) if h_strength is None else h_strength)
    hf = h_feat if h_feat is not None else rng.normal(0.0, 1.0, N_FEAT)
    ht = h_trans if h_trans is not None else rng.normal(0.0, 0.8, (N_ACT, N_ACT))
    m = Maker(name, int(family), pref, plan, ke, ko, np.asarray(hf, float), np.asarray(ht, float), hs)
    m.goal = draw_goal(world, m, rng)
    return m


def population(world: World, n: int, rng: np.random.Generator, family: int | None = None) -> list:
    out = []
    for i in range(int(n)):
        fam = int(rng.integers(world.n_families)) if family is None else int(family)
        out.append(make_maker(world, f"m{i}", rng, family=fam))
    return out


# --------------------------------------------------------------------------- #
# Goals, episodes, routes.
# --------------------------------------------------------------------------- #
def goal_dist(world: World, m: Maker) -> np.ndarray:
    fam = world.family(m.family)
    return C.softmax(world.params.goal_temp * np.log(fam.prefs[m.pref] + 1e-9))


def draw_goal(world: World, m: Maker, rng: np.random.Generator) -> int:
    return int(rng.choice(N_GOAL, p=goal_dist(world, m)))


def effective_h(m: Maker, episode: int = 0, decay: float = 0.7) -> float:
    """Reward-linked attention strength; decays geometrically per episode after its reward reversed."""
    if m.h_reversed_at is None or episode < m.h_reversed_at:
        return float(m.h_strength)
    return float(m.h_strength * decay ** (episode - m.h_reversed_at + 1))


def action_policy(fam: Family, m: Maker, g: int, prev: int | None, h_eff: float) -> np.ndarray:
    base = fam.plan0[m.plan, g] if prev is None else fam.plan[m.plan, g, prev]
    if prev is None or h_eff <= 0:
        return base
    tilt = np.exp(h_eff * m.h_trans[prev])
    p = base * tilt
    return p / p.sum()


def episode(world: World, m: Maker, rng: np.random.Generator, index: int = 0, goal: int | None = None,
            steps: int = T_STEPS) -> dict:
    """One episode: intended and realized actions, surface tokens, semantic tokens, context choices,
    forensic fields. ``index`` is the episode number in the maker's life (history decay)."""
    fam = world.family(m.family)
    g = int(m.goal if goal is None else goal)
    h_eff = effective_h(m, index, world.params.h_decay)
    intended, realized, surface = [], [], []
    prev = None
    for t in range(int(steps)):
        p = action_policy(fam, m, g, prev, h_eff)
        a = int(rng.choice(N_ACT, p=p))
        intended.append(a)
        real = a if rng.random() < m.k_exec else int(rng.integers(N_ACT))
        realized.append(real)
        fp = fam.feat[real].copy()
        if t < world.params.h_early_steps and h_eff > 0:
            fp = fp * np.exp(h_eff * m.h_feat)
            fp = fp / fp.sum()
        surface.append(int(rng.choice(N_FEAT, p=fp)))
        prev = real
    sem = []
    for _ in range(N_SEM):
        tok = int(rng.choice(N_FEAT, p=fam.sem[g])) if rng.random() < m.k_obs else int(rng.integers(N_FEAT))
        sem.append(tok)
    choices = []
    pref = fam.prefs[m.pref]
    for d in range(N_DECISIONS):
        util = fam.align[d] @ pref
        choices.append(int(rng.choice(N_MENU, p=C.softmax(world.params.context_temp * util))))
    sig = m.plan if rng.random() < world.params.forensic_acc else int(rng.integers(N_PLAN))
    return {"maker": m.id, "family": m.family, "index": int(index), "goal": g, "plan": m.plan, "pref": m.pref,
            "intended": intended, "action": [int(fam.vocab[a]) for a in realized], "surface": surface, "semantic": sem,
            "context": {"choices": choices, "role": int(m.pref % 3), "source": dict(m.source)},
            "forensic": {"intended_head": intended[:2], "signature": int(sig), "k_exec": m.k_exec}}


def stream(world: World, m: Maker, rng: np.random.Generator, n: int, start: int = 0, regoal: bool = True) -> list:
    """``n`` episodes; the goal is redrawn from the standing preference each episode when ``regoal``."""
    out = []
    for i in range(int(n)):
        if regoal and i > 0:
            m.goal = draw_goal(world, m, rng)
        out.append(episode(world, m, rng, index=start + i))
    return out


def equivalent(plan: int, goal: int) -> tuple:
    """The process-equivalence class of (plan, goal) on the non-forensic routes."""
    other = PLAN_EQUIVALENCE.get((int(plan), int(goal)))
    return ((int(plan), int(goal)),) if other is None else ((int(plan), int(goal)), other)


def surface_histogram(ep: dict) -> np.ndarray:
    h = np.bincount(ep["surface"], minlength=N_FEAT).astype(float)
    return h / max(h.sum(), 1.0)


def relabel(ep: dict, perm: np.ndarray) -> dict:
    """A nuisance relabelling of surface tokens (attack X01, test 12): the latent is untouched."""
    out = dict(ep)
    out["surface"] = [int(perm[s]) for s in ep["surface"]]
    return out
