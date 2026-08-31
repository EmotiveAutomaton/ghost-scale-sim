"""Generator family 1 of 3: the **chain world** (spec §5.1).

Discrete action/revision chains small enough to enumerate exactly. A maker has a process (how it
works), a foreground goal (what it is after this episode) and a standing tendency (what it prefers
across episodes). Four routes carry evidence and none of them shares a token with another.

This family's job in the program is to be the one where *exact* answers exist. Every approximate
architecture must reproduce its posteriors before it is allowed to run at scale (M01), and the
coupling/access phase boundaries are measured here first (C02-C09) and then required to transfer
to families 2 and 3 (C14).

How coupling is built, and why the null at zero is exact rather than approximate
-------------------------------------------------------------------------------
The latent prior is a mixture of a uniform table and a *permutation-chain* table -- tendency picks
a goal, goal picks a process -- with the mixture weight solved by bisection so that the realized
pairwise mutual information hits ``ontology.target_coupling_nats(kappa)``, and the result fitted
back onto uniform one-dimensional marginals. Two properties matter and both were paid for:

* at ``kappa = 0`` the mixture weight is exactly zero, the prior is exactly a product, and the
  exact joint posterior equals the product of the independent marginals to floating point. That is
  the anchor: V14's regime is the corner of this atlas, and its +0.011 nats has to reappear there
  as approximately nothing.
* every marginal stays uniform at every coupling level, so the coupling knob adds dependence and
  *only* dependence. A log-linear tilt was tried first and is wrong for this job: mutual
  information is not monotone in the tilt strength, so past some point the prior collapses toward a
  point mass and the coupling falls back to zero -- a bisection walks the wrong way and returns
  degenerate worlds, which is what the first implementation did.

Overlap is the second knob and is separate from coupling. At ``overlap = 0`` each route's emission
depends on exactly one component; as it rises each route mixes in the other two, so a reader that
assigns routes to components one-to-one is using a wrong likelihood even when the prior factorizes.

``dependence`` is the third: ``redundant`` duplicates one component's evidence across two routes
(a naive fuser double-counts it); ``synergistic`` keys tokens on the *sum* of two components mod
the alphabet, so each route alone is uniform and only the pair is informative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C
from .ontology import (COMPONENTS, Episode, Latent, fit_uniform_marginals,
                       overlap_index, pairwise_coupling, realized_route_information,
                       target_coupling_nats)

FAMILY = "chain"
ROUTES = ("action", "semantic", "context", "forensic")
#: Which latent component each route is "about" at overlap 0. Forensic is a costly process route.
HOME = {"action": "process", "semantic": "goal", "context": "tendency", "forensic": "process"}
N_TOKENS = 6
N_ACTIONS = 5


@dataclass
class ChainWorld:
    knobs: object
    n_p: int
    n_g: int
    n_v: int
    prior: np.ndarray                       # [n_p, n_g, n_v]
    emission: dict                          # route -> [n_p, n_g, n_v, N_TOKENS]
    policy: np.ndarray                      # [n_p, n_g, n_context, N_ACTIONS]
    contexts: int = 3
    beta: float = 0.0
    tendency_context: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    meta: dict = field(default_factory=dict)

    def latent_space(self) -> list:
        return [(p, g, v) for p in range(self.n_p) for g in range(self.n_g)
                for v in range(self.n_v)]

    def n_latent(self) -> int:
        return self.n_p * self.n_g * self.n_v


# --------------------------------------------------------------------------- #
# Construction.
# --------------------------------------------------------------------------- #
def _coupled_prior(n_p: int, n_g: int, n_v: int, kappa: float, rng) -> tuple:
    """A prior whose pairwise coupling rises with ``kappa`` while every marginal stays uniform.

    Built as a mixture ``(1 - lam) * uniform + lam * structured``, with ``lam`` found by bisection
    against ``ontology.target_coupling_nats``. A log-linear tilt was tried first and is wrong for
    this job: mutual information is *not* monotone in the tilt strength -- past some point the
    prior collapses toward a point mass and the coupling falls back to zero, so a bisection walks
    the wrong way and returns a degenerate world. The mixture is monotone from 0 to its ceiling,
    and when the target is above that ceiling the realized value is recorded and used as the
    phase-diagram axis instead of the nominal knob.
    """
    uniform = np.full((n_p, n_g, n_v), 1.0 / (n_p * n_g * n_v))
    if kappa <= 0.0:
        return uniform, 0.0, pairwise_coupling(uniform)

    # Permutation chain tendency -> goal -> process. With equal component sizes every one-
    # dimensional marginal of this table is already exactly uniform, so all three *pairwise*
    # couplings can reach ~log(n) together. A random many-to-one map cannot: it concentrates the
    # goal marginal, the uniform fit then fights the structure, and the coupling ceiling collapses
    # to a fifth of what the atlas needs (measured at 0.19 nats against a 1.04-nat target).
    eps = 0.004
    n = min(n_p, n_g, n_v)
    tau = rng.permutation(n_g)[:n]                       # tendency v -> goal
    sig = rng.permutation(n_p)[:n]                       # goal index -> process
    structured = np.full((n_p, n_g, n_v), eps)
    for v in range(n_v):
        g = int(tau[v % n])
        p = int(sig[v % n])
        structured[p, g, v] += 1.0
    structured = fit_uniform_marginals(structured)

    def mix(lam: float) -> np.ndarray:
        return (1.0 - lam) * uniform + lam * structured

    def mean_pairwise(w) -> float:
        pc = pairwise_coupling(w)
        return float(np.mean([pc["process_goal"], pc["process_tendency"], pc["goal_tendency"]]))

    target = target_coupling_nats(kappa, n_p, n_g)
    lo, hi = 0.0, 1.0
    if mean_pairwise(structured) <= target:
        prior = structured
        lam = 1.0
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if mean_pairwise(mix(mid)) < target:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        prior = mix(lam)
    return prior, float(lam), pairwise_coupling(prior)


def _emission_tables(n_p: int, n_g: int, n_v: int, overlap: float, dependence: str, rng) -> dict:
    """One token table per route. ``overlap`` mixes non-home components in; ``dependence`` selects
    an independent, redundant or synergistic construction."""
    sizes = {"process": n_p, "goal": n_g, "tendency": n_v}
    feat = {c: rng.normal(size=(sizes[c], N_TOKENS)) * 1.6 for c in COMPONENTS}
    out = {}
    for r in ROUTES:
        home = HOME[r]
        others = [c for c in COMPONENTS if c != home]
        w_home = 1.0 - 0.55 * float(overlap)
        w_other = 0.55 * float(overlap) / len(others)
        logit = np.zeros((n_p, n_g, n_v, N_TOKENS))
        idx = {"process": 0, "goal": 1, "tendency": 2}
        for c, w in [(home, w_home)] + [(o, w_other) for o in others]:
            if w <= 0:
                continue
            f = feat[c]
            shape = [1, 1, 1, N_TOKENS]
            shape[idx[c]] = sizes[c]
            logit = logit + w * f.reshape(shape)
        if r == "forensic":
            logit = logit * 2.2                                  # costly but sharp (V14's forensic)
        out[r] = C.softmax(logit, axis=-1)

    if dependence == "redundant":
        # the semantic route is re-keyed onto the action route's component and feature: the same
        # cause reaches the reader twice, dressed as two independent paraphrases (attack X04)
        out["semantic"] = out["action"].copy()
    elif dependence == "synergistic":
        # tokens key on (process + goal) mod N_TOKENS: each route alone is uniform in each
        # component, only the pair identifies anything
        sharp = 3.0
        base = np.zeros((n_p, n_g, n_v, N_TOKENS))
        for p in range(n_p):
            for g in range(n_g):
                base[p, g, :, (p + g) % N_TOKENS] = sharp
        out["action"] = C.softmax(base, axis=-1)
        base2 = np.zeros((n_p, n_g, n_v, N_TOKENS))
        for p in range(n_p):
            for g in range(n_g):
                base2[p, g, :, (p + 2 * g) % N_TOKENS] = sharp
        out["semantic"] = C.softmax(base2, axis=-1)
    return out


def _policy(n_p: int, n_g: int, n_ctx: int, temperature: float, rng) -> np.ndarray:
    """p(action | process, goal, context). The process shapes *how*, the goal shapes *toward what*."""
    proc = rng.normal(size=(n_p, N_ACTIONS)) * 1.5
    goal = rng.normal(size=(n_g, N_ACTIONS)) * 1.5
    ctx = rng.normal(size=(n_ctx, N_ACTIONS)) * 0.7
    logit = proc[:, None, None, :] + goal[None, :, None, :] + ctx[None, None, :, :]
    return C.softmax(logit / max(float(temperature), 1e-3), axis=-1)


def sample_world(knobs, rng) -> ChainWorld:
    """Each part of the world draws from its own substream, so that turning one knob does not
    silently redraw another part through a shifted random stream. A factorial design whose cells
    differ in more than the factor is not a factorial design."""
    n_p, n_g, n_v = knobs.n_process, knobs.n_goal, knobs.n_tendency
    s = rng.integers(0, 2 ** 62, size=4)
    r_prior, r_emit, r_pol, r_ctx = (np.random.default_rng(int(x)) for x in s)

    prior, lam, coupling = _coupled_prior(n_p, n_g, n_v, knobs.kappa, r_prior)
    emission = _emission_tables(n_p, n_g, n_v, knobs.overlap, knobs.dependence, r_emit)
    n_ctx = 3
    policy = _policy(n_p, n_g, n_ctx, knobs.temperature, r_pol)
    tc = C.softmax(r_ctx.normal(size=(n_v, n_ctx)) * 1.2, axis=-1)   # tendency -> context choice
    w = ChainWorld(knobs=knobs, n_p=n_p, n_g=n_g, n_v=n_v, prior=prior, emission=emission,
                   policy=policy, contexts=n_ctx, beta=lam, tendency_context=tc)
    ri = realized_route_information(emission, prior)
    # The overlap index must describe the EMISSION, so it is measured against a uniform reference
    # prior. Measured under the world's own prior it reads 0.89 at zero overlap as soon as coupling
    # is on, because a coupled prior lets a token about one component speak about all three -- that
    # is the coupling knob leaking into the overlap receipt, and it would make the two axes of the
    # phase surface impossible to tell apart.
    ref = np.full_like(prior, 1.0 / prior.size)
    ri_ref = realized_route_information(emission, ref)
    w.meta = {"family": FAMILY, "pairwise_coupling": coupling,
              "realized_coupling": float(np.mean([coupling["process_goal"],
                                                  coupling["process_tendency"],
                                                  coupling["goal_tendency"]])),
              "target_coupling": target_coupling_nats(knobs.kappa, n_p, n_g),
              "route_information": ri, "route_information_reference": ri_ref,
              "overlap_index": overlap_index(ri_ref),
              "overlap_index_under_world_prior": overlap_index(ri),
              "mixture_lambda": lam,
              "marginal_uniformity": float(max(
                  abs(prior.sum(axis=(1, 2)) - 1.0 / n_p).max(),
                  abs(prior.sum(axis=(0, 2)) - 1.0 / n_g).max(),
                  abs(prior.sum(axis=(0, 1)) - 1.0 / n_v).max()))}
    return w


def sample_latent(world: ChainWorld, rng) -> Latent:
    flat = world.prior.ravel()
    i = int(rng.choice(flat.size, p=flat / flat.sum()))
    p, g, v = np.unravel_index(i, world.prior.shape)
    return Latent(process=int(p), goal=int(g), tendency=int(v))


# --------------------------------------------------------------------------- #
# Rollout.
# --------------------------------------------------------------------------- #
def rollout(world: ChainWorld, latent: Latent, rng, n_steps: int = 12) -> Episode:
    k = world.knobs
    p, g, v = latent.process, latent.goal, latent.tendency
    context = int(rng.choice(world.contexts, p=world.tendency_context[v]))
    goal_t = [g] * n_steps
    if k.drift == "abrupt":
        cut = n_steps // 2
        g2 = int((g + 1 + rng.integers(0, max(world.n_g - 1, 1))) % world.n_g)
        goal_t = [g] * cut + [g2] * (n_steps - cut)
    elif k.drift == "gradual":
        g2 = int((g + 1) % world.n_g)
        for t in range(n_steps):
            if rng.random() < t / max(n_steps - 1, 1) * 0.6:
                goal_t[t] = g2

    routes = {r: [] for r in ROUTES}
    actions, intended = [], []
    for t in range(n_steps):
        gt = goal_t[t]
        pol = world.policy[p, gt, context]
        a_int = int(rng.choice(N_ACTIONS, p=pol))
        a = a_int if rng.random() < k.competence else int(rng.integers(N_ACTIONS))
        intended.append(a_int)
        actions.append(a)
        for r in ROUTES:
            routes[r].append(int(rng.choice(N_TOKENS, p=world.emission[r][p, gt, v])))

    # the hidden events this family scores (spec §4.1)
    nxt_ctx = int(rng.choice(world.contexts, p=world.tendency_context[v]))
    pol_next = world.policy[p, goal_t[-1], nxt_ctx]
    hidden = {"next_action": int(rng.choice(N_ACTIONS, p=pol_next)),
              "changed_context_choice": int(np.argmax(world.tendency_context[v]
                                                      + 1e-9 * rng.normal(size=world.contexts))),
              "next_episode_first_choice": int(rng.choice(
                  N_ACTIONS, p=world.policy[p, int(np.argmax(world.prior.sum(axis=(0, 2)))), nxt_ctx]))}
    return Episode(routes=routes, context=context,
                   opportunity=tuple(range(N_ACTIONS)), hidden=hidden, latent=latent,
                   meta={"family": FAMILY, "actions": actions, "intended": intended,
                         "goal_t": goal_t, "next_context": nxt_ctx})


# --------------------------------------------------------------------------- #
# Likelihood and prediction. Every call debits a budget (spec §3.5).
# --------------------------------------------------------------------------- #
def route_loglik(world: ChainWorld, triple, episode: Episode, upto: int, routes=None,
                 budget=None) -> float:
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


def action_loglik(world: ChainWorld, triple, episode: Episode, upto: int, budget=None) -> float:
    p, g, _ = triple
    acts = episode.meta.get("actions", [])[:upto]
    if not acts:
        return 0.0
    k = world.knobs
    pol = world.policy[p, g, episode.context]
    mix = k.competence * pol + (1.0 - k.competence) / N_ACTIONS
    if budget is not None:
        budget.lik(1)
    return float(np.log(np.maximum(mix[np.asarray(acts, int)], 1e-300)).sum())


def loglik(world: ChainWorld, triple, episode: Episode, upto: int, routes=None,
           budget=None) -> float:
    return (route_loglik(world, triple, episode, upto, routes, budget)
            + action_loglik(world, triple, episode, upto, budget))


def step_loglik(world: ChainWorld, triple, episode: Episode, t: int, budget=None) -> float:
    """One step's contribution only. The sequential filters need slices, not prefixes; summing
    ``loglik`` prefixes would count every earlier step again at every step."""
    p, g, v = triple
    tot = 0.0
    for r in ROUTES:
        toks = episode.routes.get(r, [])
        if t < len(toks):
            tot += float(np.log(max(world.emission[r][p, g, v][toks[t]], 1e-300)))
    acts = episode.meta.get("actions", [])
    if t < len(acts):
        k = world.knobs
        pol = world.policy[p, g, episode.context]
        mix = k.competence * pol + (1.0 - k.competence) / N_ACTIONS
        tot += float(np.log(max(mix[acts[t]], 1e-300)))
    if budget is not None:
        budget.lik(len(ROUTES) + 1)
    return tot


def log_prior(world: ChainWorld, triple) -> float:
    return float(np.log(max(world.prior[triple], 1e-300)))


def endpoint_dist(world: ChainWorld, triple, episode: Episode, endpoint: str = "next_action",
                  budget=None) -> np.ndarray:
    """p(hidden event | latent, observed context). Never sees the hidden event."""
    p, g, v = triple
    if budget is not None:
        budget.lik(1)
    if endpoint == "next_action":
        ctx = episode.meta.get("next_context", episode.context)
        pol = world.policy[p, g, ctx]
        k = world.knobs
        return k.competence * pol + (1.0 - k.competence) / N_ACTIONS
    if endpoint == "changed_context_choice":
        return np.asarray(world.tendency_context[v], float)
    if endpoint == "next_episode_first_choice":
        gbar = int(np.argmax(world.prior.sum(axis=(0, 2))))
        ctx = episode.meta.get("next_context", episode.context)
        return np.asarray(world.policy[p, gbar, ctx], float)
    raise ValueError(endpoint)


def endpoint_size(endpoint: str, world: ChainWorld) -> int:
    return world.contexts if endpoint == "changed_context_choice" else N_ACTIONS


def marginal_prior(world: ChainWorld) -> dict:
    p = world.prior
    return {"process": p.sum(axis=(1, 2)), "goal": p.sum(axis=(0, 2)),
            "tendency": p.sum(axis=(0, 1))}


def surface_predictor(episode: Episode, upto: int, endpoint: str, world: ChainWorld) -> np.ndarray:
    """Frequency and last-action baselines with no maker model at all (spec §3.4.1)."""
    n = endpoint_size(endpoint, world)
    if endpoint != "next_action":
        return np.full(n, 1.0 / n)
    acts = episode.meta.get("actions", [])[:upto]
    counts = np.full(n, 0.5)
    for a in acts:
        counts[a] += 1.0
    if acts:
        counts[acts[-1]] += 1.5                                  # last action, weighted
    return counts / counts.sum()


def context_realized(world: ChainWorld, triple, episode: Episode) -> dict:
    """The context-realized candidate a bare label is compared against (spec §3.3)."""
    p, g, v = triple
    return {"context": int(episode.context), "opportunity": list(episode.opportunity),
            "action_distribution": world.policy[p, g, episode.context].tolist(),
            "process_constraints": {"process": int(p), "competence": float(world.knobs.competence)},
            "stopping_rule": {"kind": "fixed_length", "n": int(episode.n_steps())},
            "intervention_delta": {
                "context_swap": float(C.tv(world.policy[p, g, episode.context],
                                           world.policy[p, g, (episode.context + 1) % world.contexts]))}}
