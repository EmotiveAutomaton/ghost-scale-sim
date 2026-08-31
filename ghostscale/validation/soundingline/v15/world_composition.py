"""Generator family 2 of 3: the **composition world** (spec §5.2).

Parts, edits, stopping and role-relative constraints. A maker holds a working composition -- a
vector of part states -- and repeatedly chooses an edit. What it edits and when it stops are the
observable trace; why is not.

Independence from the chain family
----------------------------------
Spec §5 forbids a family that relabels another family's transition table, and card I06 audits it.
Nothing here is imported from ``world_chain``. The differences are structural, not cosmetic:

* the chain family emits tokens from a *static* table indexed by the latent triple; here the
  emission is a function of the **evolving composition state**, so the same latent produces
  different tokens as the work proceeds;
* the chain family's policy is a random logit table; here the policy is **value-driven** -- an edit
  is chosen by how much it moves the composition toward the goal's target property, so competence
  and temperature act on a computed advantage rather than on fixed weights;
* the chain family has no terminal condition; here **stopping is endogenous**, triggered by a
  satisfaction threshold that belongs to the process rather than to the goal, which is what lets
  G09 ask whether a stopping rule is recoverable independently of the content goal.

Latent components, in the shared vocabulary:

``process``   the edit strategy: sweep, deepen, repair, explore.
``goal``      which property of the composition is being maximized this episode.
``tendency``  the standing preference over part *kinds*, expressed across episodes and contexts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C
from .ontology import (fit_uniform_marginals,
                       COMPONENTS, Episode, Latent, overlap_index, pairwise_coupling,
                       realized_route_information, target_coupling_nats)

FAMILY = "composition"
ROUTES = ("edits", "parts", "review", "trace")
HOME = {"edits": "process", "parts": "goal", "review": "tendency", "trace": "process"}
N_TOKENS = 6
N_PARTS = 5
N_ACTIONS = 5                    # edit operations: extend, refine, cut, reorder, join
STRATEGIES = ("sweep", "deepen", "repair", "explore")


@dataclass
class CompositionWorld:
    knobs: object
    n_p: int
    n_g: int
    n_v: int
    prior: np.ndarray
    emission: dict                       # route -> [n_p, n_g, n_v, N_TOKENS] (state-averaged view)
    policy: np.ndarray                   # [n_p, n_g, n_context, N_ACTIONS] (state-averaged view)
    contexts: int = 3
    goal_weights: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    kind_of_part: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=int))
    tendency_context: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    satisfaction: np.ndarray = field(default_factory=lambda: np.zeros(1))
    context_bias: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    context_threshold: np.ndarray = field(default_factory=lambda: np.zeros(1))
    beta: float = 0.0
    meta: dict = field(default_factory=dict)

    def latent_space(self) -> list:
        return [(p, g, v) for p in range(self.n_p) for g in range(self.n_g)
                for v in range(self.n_v)]

    def n_latent(self) -> int:
        return self.n_p * self.n_g * self.n_v


# --------------------------------------------------------------------------- #
# Construction. The coupling is built by conditional tilting, not by mixing.
# --------------------------------------------------------------------------- #
def _prior(n_p: int, n_g: int, n_v: int, kappa: float, rng) -> tuple:
    """Coupling by *conditional concentration*: the tendency narrows which goals are live, and the
    goal narrows which strategies are live.

    This reaches the same target coupling as the chain family by a different route -- a chain of
    conditionals rather than a mixture with a permutation table -- which is what makes agreement
    between the two families evidence rather than a shared implementation.
    """
    uniform = np.full((n_p, n_g, n_v), 1.0 / (n_p * n_g * n_v))
    if kappa <= 0.0:
        return uniform, 0.0, pairwise_coupling(uniform)
    rank_g = np.argsort(rng.normal(size=(n_v, n_g)), axis=1)
    rank_p = np.argsort(rng.normal(size=(n_g, n_p)), axis=1)
    target = target_coupling_nats(kappa, n_p, n_g)

    def build(s: float) -> np.ndarray:
        pv = np.full(n_v, 1.0 / n_v)
        pg_v = np.zeros((n_v, n_g))
        for v in range(n_v):
            pg_v[v] = C.softmax(-s * np.argsort(rank_g[v]).astype(float))
        pp_g = np.zeros((n_g, n_p))
        for g in range(n_g):
            pp_g[g] = C.softmax(-s * np.argsort(rank_p[g]).astype(float))
        out = np.zeros((n_p, n_g, n_v))
        for v in range(n_v):
            for g in range(n_g):
                out[:, g, v] = pv[v] * pg_v[v, g] * pp_g[g]
        # Fit back onto uniform one-dimensional marginals. Without this the conditional chain
        # concentrates the goal and process marginals as the tilt rises, so "coupling" would move
        # the marginals here and not in the chain family -- and a cross-family agreement (C14)
        # would be comparing two different knobs.
        return fit_uniform_marginals(out)

    def mean_pw(w) -> float:
        pc = pairwise_coupling(w)
        return float(np.mean([pc["process_goal"], pc["process_tendency"], pc["goal_tendency"]]))

    lo, hi = 0.0, 6.0
    if mean_pw(build(hi)) <= target:
        prior = build(hi)
        s = hi
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if mean_pw(build(mid)) < target:
                lo = mid
            else:
                hi = mid
        s = 0.5 * (lo + hi)
        prior = build(s)
    return prior, float(s), pairwise_coupling(prior)


def sample_world(knobs, rng) -> CompositionWorld:
    n_p, n_g, n_v = knobs.n_process, knobs.n_goal, knobs.n_tendency
    s = rng.integers(0, 2 ** 62, size=5)
    r_prior, r_goal, r_part, r_ctx, r_emit = (np.random.default_rng(int(x)) for x in s)

    prior, beta, coupling = _prior(n_p, n_g, n_v, knobs.kappa, r_prior)
    # a goal is a weighting over part *properties*; an edit's value is how much it improves it
    goal_weights = r_goal.normal(size=(n_g, N_PARTS))
    kind_of_part = r_part.integers(0, max(n_v, 2), size=N_PARTS)
    tc = C.softmax(r_ctx.normal(size=(n_v, 3)) * 1.1, axis=-1)
    context_bias = r_ctx.normal(size=(3, N_PARTS)) * 2.0
    # the role moves the bar for 'good enough', so a maker's stopping point is not a function
    # of its latent triple alone (spec §5.2's role-relative constraints)
    context_threshold = r_ctx.uniform(-0.16, 0.16, size=3)
    satisfaction = 0.34 + 0.26 * np.arange(n_p) / max(n_p - 1, 1)      # process sets the threshold

    w = CompositionWorld(knobs=knobs, n_p=n_p, n_g=n_g, n_v=n_v, prior=prior, emission={},
                         policy=np.zeros((n_p, n_g, 3, N_ACTIONS)), contexts=3,
                         goal_weights=goal_weights, kind_of_part=kind_of_part,
                         tendency_context=tc, satisfaction=satisfaction, beta=beta,
                         context_bias=context_bias, context_threshold=context_threshold)
    w.emission = _emission_view(w, knobs, r_emit)
    w.policy = _policy_view(w)
    ri = realized_route_information(w.emission, prior)
    ref = np.full_like(prior, 1.0 / prior.size)
    ri_ref = realized_route_information(w.emission, ref)
    w.meta = {"family": FAMILY, "pairwise_coupling": coupling,
              "realized_coupling": float(np.mean([coupling["process_goal"],
                                                  coupling["process_tendency"],
                                                  coupling["goal_tendency"]])),
              "target_coupling": target_coupling_nats(knobs.kappa, n_p, n_g),
              "route_information": ri, "route_information_reference": ri_ref,
              "overlap_index": overlap_index(ri_ref), "tilt": beta,
              "marginal_uniformity": float(max(
                  abs(prior.sum(axis=(1, 2)) - 1.0 / n_p).max(),
                  abs(prior.sum(axis=(0, 2)) - 1.0 / n_g).max(),
                  abs(prior.sum(axis=(0, 1)) - 1.0 / n_v).max()))}
    return w


def _initial_state(rng) -> np.ndarray:
    return rng.random(N_PARTS) * 0.3


def _edit_effect(op: int, part: int, state: np.ndarray) -> np.ndarray:
    """What an edit does to the composition. Deterministic given (op, part, state)."""
    s = state.copy()
    if op == 0:                                   # extend: raise this part
        s[part] = min(1.0, s[part] + 0.30)
    elif op == 1:                                 # refine: small gain, no risk
        s[part] = min(1.0, s[part] + 0.12)
    elif op == 2:                                 # cut: lower this part, raise its neighbour
        s[part] = max(0.0, s[part] - 0.20)
        s[(part + 1) % N_PARTS] = min(1.0, s[(part + 1) % N_PARTS] + 0.10)
    elif op == 3:                                 # reorder: swap two parts
        j = (part + 2) % N_PARTS
        s[part], s[j] = s[j], s[part]
    else:                                         # join: average a pair upward
        j = (part + 1) % N_PARTS
        m = 0.5 * (s[part] + s[j]) + 0.10
        s[part] = s[j] = min(1.0, m)
    return s


def _quality(world: CompositionWorld, g: int, state: np.ndarray) -> float:
    return float(np.dot(C.softmax(world.goal_weights[g]), state))


def _bar(world: CompositionWorld, p: int, context: int) -> float:
    """The satisfaction bar: the process sets it, the role shifts it."""
    n = world.context_threshold.shape[0]
    return float(world.satisfaction[p] + world.context_threshold[context % n])


def _strategy_bias(p: int, part: int, step: int, state: np.ndarray) -> float:
    """The process is *where* the maker looks next, independent of what the goal wants."""
    name = STRATEGIES[p % len(STRATEGIES)]
    if name == "sweep":
        return 2.0 if part == step % N_PARTS else 0.0
    if name == "deepen":
        return 2.0 if part == int(np.argmax(state)) else 0.0
    if name == "repair":
        return 2.0 if part == int(np.argmin(state)) else 0.0
    return 0.6                                    # explore: flat over parts


def _choice_logits(world: CompositionWorld, p: int, g: int, state: np.ndarray, step: int,
                   context: int = 0):
    """Value of every (op, part) pair: goal-driven improvement, process-driven attention, and
    the role-relative constraint the context puts on which parts are this maker's to touch.

    The context term is what stops a *label* from being sufficient: two makers with the same
    process, goal and tendency act differently in different roles, so a decontextualized
    pointer to the triple cannot reproduce the choice. Without it ``label_only`` and the state
    oracle are literally the same reader and the pointer-versus-state comparison is vacuous."""
    n = N_ACTIONS * N_PARTS
    val = np.zeros(n)
    base = _quality(world, g, state)
    cb = world.context_bias[context % world.context_bias.shape[0]]
    for op in range(N_ACTIONS):
        for part in range(N_PARTS):
            nxt = _edit_effect(op, part, state)
            val[op * N_PARTS + part] = 4.0 * (_quality(world, g, nxt) - base) \
                + _strategy_bias(p, part, step, state) + float(cb[part])
    return val


def sample_latent(world: CompositionWorld, rng) -> Latent:
    flat = world.prior.ravel()
    i = int(rng.choice(flat.size, p=flat / flat.sum()))
    p, g, v = np.unravel_index(i, world.prior.shape)
    return Latent(process=int(p), goal=int(g), tendency=int(v))


def rollout(world: CompositionWorld, latent: Latent, rng, n_steps: int = 12) -> Episode:
    k = world.knobs
    p, g, v = latent.process, latent.goal, latent.tendency
    context = int(rng.choice(world.contexts, p=world.tendency_context[v]))
    state = _initial_state(rng)
    routes = {r: [] for r in ROUTES}
    ops, parts, states, stopped_at = [], [], [], None

    for t in range(n_steps):
        logits = _choice_logits(world, p, g, state, t, context)
        pol = C.softmax(logits / max(k.temperature, 1e-3))
        idx = int(rng.choice(pol.size, p=pol))
        if rng.random() >= k.competence:
            idx = int(rng.integers(pol.size))                 # a slip: intention missed
        op, part = divmod(idx, N_PARTS)
        ops.append(op)
        parts.append(part)
        states.append(state.copy())
        # tokens are functions of the evolving state, not of a static table
        routes["edits"].append(int(op % N_TOKENS))
        routes["parts"].append(int((part + int(world.kind_of_part[part])) % N_TOKENS))
        routes["review"].append(int((int(world.kind_of_part[part]) * 2 + (v % 3)) % N_TOKENS))
        routes["trace"].append(int((op * 2 + p) % N_TOKENS))
        state = _edit_effect(op, part, state)
        if stopped_at is None and _quality(world, g, state) >= _bar(world, p, context):
            stopped_at = t + 1

    nxt_ctx = int(rng.choice(world.contexts, p=world.tendency_context[v]))
    final_logits = _choice_logits(world, p, g, state, n_steps, nxt_ctx)
    final_pol = C.softmax(final_logits / max(k.temperature, 1e-3))
    next_idx = int(rng.choice(final_pol.size, p=final_pol))
    hidden = {"next_edit": int(next_idx // N_PARTS),
              "next_action": int(next_idx // N_PARTS),
              "stop_or_continue": int(_quality(world, g, state) >= _bar(world, p, nxt_ctx)),
              "changed_context_choice": int(np.argmax(world.tendency_context[v]))}
    return Episode(routes=routes, context=context,
                   opportunity=tuple(range(N_ACTIONS)), hidden=hidden, latent=latent,
                   meta={"family": FAMILY, "actions": ops, "parts": parts,
                         "states": [s.tolist() for s in states], "final_state": state.tolist(),
                         "stopped_at": stopped_at, "next_context": nxt_ctx})


# --------------------------------------------------------------------------- #
# Reader-side views. The tournament needs table-shaped summaries of a dynamic process.
# --------------------------------------------------------------------------- #
def _emission_view(world: CompositionWorld, knobs, rng) -> dict:
    """Marginal token tables ``[n_p, n_g, n_v, N_TOKENS]``, obtained by averaging the generative
    emission over reachable composition states.

    These are the reader's likelihood, and they are deliberately a *summary* of a process the
    reader does not simulate: an approximation the chain family does not have, and one reason the
    two families disagree about how hard the same nominal coupling is to exploit.
    """
    n_p, n_g, n_v = world.n_p, world.n_g, world.n_v
    tab = {r: np.full((n_p, n_g, n_v, N_TOKENS), 1e-3) for r in ROUTES}
    for p in range(n_p):
        for g in range(n_g):
            for v in range(n_v):
                st = np.linspace(0.1, 0.8, N_PARTS)
                for t in range(10):
                    logits = _choice_logits(world, p, g, st, t, 0)
                    pol = C.softmax(logits / max(knobs.temperature, 1e-3))
                    for idx in range(pol.size):
                        op, part = divmod(idx, N_PARTS)
                        wgt = float(pol[idx])
                        tab["edits"][p, g, v, op % N_TOKENS] += wgt
                        tab["parts"][p, g, v, (part + int(world.kind_of_part[part])) % N_TOKENS] += wgt
                        tab["review"][p, g, v, (int(world.kind_of_part[part]) * 2 + (v % 3)) % N_TOKENS] += wgt
                        tab["trace"][p, g, v, (op * 2 + p) % N_TOKENS] += wgt
                    st = _edit_effect(int(np.argmax(pol)) // N_PARTS,
                                      int(np.argmax(pol)) % N_PARTS, st)
    out = {}
    for r, t in tab.items():
        t = t / t.sum(axis=-1, keepdims=True)
        if knobs.overlap > 0:                       # blend toward the component-average table
            flat = t.mean(axis=(0, 1, 2), keepdims=True)
            t = (1 - 0.5 * knobs.overlap) * t + 0.5 * knobs.overlap * flat
            t = t / t.sum(axis=-1, keepdims=True)
        out[r] = t
    if knobs.dependence == "redundant":
        out["parts"] = out["edits"].copy()
    return out


def _policy_view(world: CompositionWorld) -> np.ndarray:
    """``p(edit operation | process, goal, context)`` averaged over reachable states."""
    n_p, n_g = world.n_p, world.n_g
    out = np.zeros((n_p, n_g, world.contexts, N_ACTIONS))
    for p in range(n_p):
        for g in range(n_g):
            for c0 in range(world.contexts):
                st = np.linspace(0.1, 0.8, N_PARTS)
                acc = np.zeros(N_ACTIONS)
                for t in range(10):
                    logits = _choice_logits(world, p, g, st, t, c0)
                    pol = C.softmax(logits / max(world.knobs.temperature, 1e-3))
                    for idx in range(pol.size):
                        acc[idx // N_PARTS] += float(pol[idx])
                    st = _edit_effect(int(np.argmax(pol)) // N_PARTS,
                                      int(np.argmax(pol)) % N_PARTS, st)
                out[p, g, c0] = C.normalize(acc)
    return out


# --------------------------------------------------------------------------- #
# The shared reader surface.
# --------------------------------------------------------------------------- #
def route_loglik(world, triple, episode, upto: int, routes=None, budget=None) -> float:
    p, g, v = triple
    tot = 0.0
    for r in (routes or ROUTES):
        toks = episode.routes.get(r, [])[:upto]
        if not toks:
            continue
        tab = world.emission[r][p, g, v]
        tot += float(np.log(np.maximum(tab[np.asarray(toks, int)], 1e-300)).sum())
    if budget is not None:
        budget.lik(len(routes or ROUTES))
    return tot


def _traj_policies(world, episode) -> np.ndarray:
    """``[n_p, n_g, n_steps, N_ACTIONS * N_PARTS]``: the choice distribution each (process, goal)
    pair would have had at each *observed* composition state.

    The state trajectory is a deterministic function of the observed edits, so it does not depend
    on the latent and is computed once per episode. This is what makes the composition family's
    likelihood genuinely state-dependent instead of a static table -- and it is the reason this
    family disagrees with the chain family about how much a given nominal coupling is worth.
    """
    cached = episode.meta.get("_pol_cache")
    if cached is not None:
        return cached
    states = [np.asarray(x, float) for x in episode.meta.get("states", [])]
    n_p, n_g = world.n_p, world.n_g
    T = len(states)
    out = np.zeros((n_p, n_g, T, N_ACTIONS * N_PARTS))
    tau = max(world.knobs.temperature, 1e-3)
    for p in range(n_p):
        for g in range(n_g):
            for t, st in enumerate(states):
                out[p, g, t] = C.softmax(
                    _choice_logits(world, p, g, st, t, episode.context) / tau)
    episode.meta["_pol_cache"] = out
    return out


def action_loglik(world, triple, episode, upto: int, budget=None) -> float:
    p, g, _ = triple
    acts = episode.meta.get("actions", [])[:upto]
    parts = episode.meta.get("parts", [])[:upto]
    if not acts:
        return 0.0
    pol = _traj_policies(world, episode)
    k = world.knobs
    n = pol.shape[-1]
    tot = 0.0
    for t, (op, part) in enumerate(zip(acts, parts)):
        if t >= pol.shape[2]:
            break
        pr = k.competence * pol[p, g, t, op * N_PARTS + part] + (1.0 - k.competence) / n
        tot += float(np.log(max(pr, 1e-300)))
    if budget is not None:
        budget.lik(len(acts))
    return tot


def loglik(world, triple, episode, upto: int, routes=None, budget=None) -> float:
    return (route_loglik(world, triple, episode, upto, routes, budget)
            + action_loglik(world, triple, episode, upto, budget))


def step_loglik(world, triple, episode, t: int, budget=None) -> float:
    p, g, v = triple
    tot = 0.0
    for r in ROUTES:
        toks = episode.routes.get(r, [])
        if t < len(toks):
            tot += float(np.log(max(world.emission[r][p, g, v][toks[t]], 1e-300)))
    acts = episode.meta.get("actions", [])
    parts = episode.meta.get("parts", [])
    pol = _traj_policies(world, episode)
    if t < len(acts) and t < pol.shape[2]:
        k = world.knobs
        n = pol.shape[-1]
        pr = k.competence * pol[p, g, t, acts[t] * N_PARTS + parts[t]] + (1.0 - k.competence) / n
        tot += float(np.log(max(pr, 1e-300)))
    if budget is not None:
        budget.lik(len(ROUTES) + 1)
    return tot


def log_prior(world, triple) -> float:
    return float(np.log(max(world.prior[triple], 1e-300)))


def endpoint_dist(world, triple, episode, endpoint: str = "next_edit", budget=None) -> np.ndarray:
    p, g, v = triple
    if budget is not None:
        budget.lik(1)
    if endpoint in ("next_edit", "next_action"):
        st = np.asarray(episode.meta.get("final_state", np.zeros(N_PARTS)), float)
        tau = max(world.knobs.temperature, 1e-3)
        ctx = episode.meta.get("next_context", episode.context)
        full = C.softmax(
            _choice_logits(world, p, g, st, int(episode.n_steps()), ctx) / tau)
        op = full.reshape(N_ACTIONS, N_PARTS).sum(axis=1)
        k = world.knobs
        return k.competence * C.normalize(op) + (1.0 - k.competence) / N_ACTIONS
    if endpoint == "stop_or_continue":
        st = np.asarray(episode.meta.get("final_state", np.zeros(N_PARTS)), float)
        ctx = episode.meta.get("next_context", episode.context)
        margin = (_quality(world, g, st) - _bar(world, p, ctx)) * 6.0
        return C.softmax(np.array([-margin, margin]))
    if endpoint == "changed_context_choice":
        return np.asarray(world.tendency_context[v], float)
    raise ValueError(endpoint)


def endpoint_size(endpoint: str, world) -> int:
    if endpoint == "stop_or_continue":
        return 2
    if endpoint == "changed_context_choice":
        return world.contexts
    return N_ACTIONS


def marginal_prior(world) -> dict:
    p = world.prior
    return {"process": p.sum(axis=(1, 2)), "goal": p.sum(axis=(0, 2)),
            "tendency": p.sum(axis=(0, 1))}


def surface_predictor(episode, upto: int, endpoint: str, world) -> np.ndarray:
    n = endpoint_size(endpoint, world)
    if endpoint == "stop_or_continue":
        st = np.asarray(episode.meta.get("states", [[0.0] * N_PARTS])[min(upto, len(
            episode.meta.get("states", [[0.0]])) - 1)], float)
        m = float(st.mean())
        return C.normalize(np.array([1.0 - m, m]))
    if endpoint == "changed_context_choice":
        return np.full(n, 1.0 / n)
    acts = episode.meta.get("actions", [])[:upto]
    counts = np.full(n, 0.5)
    for a in acts:
        counts[a] += 1.0
    if acts:
        counts[acts[-1]] += 1.5
    return counts / counts.sum()


def context_realized(world, triple, episode) -> dict:
    p, g, v = triple
    st = np.asarray(episode.meta.get("final_state", np.zeros(N_PARTS)), float)
    return {"context": int(episode.context), "opportunity": list(episode.opportunity),
            "action_distribution": world.policy[p, g, episode.context].tolist(),
            "process_constraints": {"strategy": STRATEGIES[p % len(STRATEGIES)],
                                    "competence": float(world.knobs.competence)},
            "stopping_rule": {"kind": "role_relative_satisfaction_threshold",
                              "threshold": _bar(world, p, episode.context),
                              "process_component": float(world.satisfaction[p]),
                              "quality_now": _quality(world, g, st)},
            "intervention_delta": {"context_swap": float(C.tv(
                world.policy[p, g, episode.context],
                world.policy[p, g, (episode.context + 1) % world.contexts]))}}
