"""Generator family 3 of 3: the **communication world** (spec §5.3).

Evidence selection, audience effects, correction and private costly action. A maker holds a private
belief, an audience it is steering, a selection policy and a standing motive, and emits an
assertion, a chosen subset of evidence, a correction record and -- sometimes -- a costly private
action.

The one thing this family exists to get right
---------------------------------------------
V14 separated a sincere fanatic from a strategic propagandist at 90% using an off-audience action
that was *generated directly from the hidden belief and read with the matching likelihood*. That is
a construction identity, not a discovery, and spec §2 requires it be removed. Here the private
action is produced by a planner::

    p(private action) = softmax( (alignment(action, belief) * stake - cost(action)) / temperature )

so the same action can come from a strong belief paying a high cost or a weak belief paying a low
one, and a reader that wants the motive has to model the cost as well. S04 sweeps the noise, S05
crosses the motives (a fanatic may strategically teach, a propagandist may privately believe), and
the discriminator is allowed to fail.

Independence from the other two families
----------------------------------------
* the prior is built by **tempering a Dirichlet draw** onto the coupling target -- not the chain
  family's permutation mixture, not the composition family's conditional chain;
* the emission is a **belief-marginalized planner integral**: the reader's likelihood over the
  latent triple integrates out the private belief, which neither other family has;
* the "actions" channel is an **evidence-selection** distribution over a pool, not a motor policy.

Latent components, in the shared vocabulary:

``process``   the evidence-selection policy: sample_all, cherry_pick, balanced, escalate.
``goal``      the audience state being steered toward this episode.
``tendency``  the standing motive: sincere, strategic, mixed, contrarian.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C
from .ontology import (fit_uniform_marginals,
                       Episode, Latent, overlap_index, pairwise_coupling,
                       realized_route_information, target_coupling_nats)

FAMILY = "communication"
ROUTES = ("assertion", "evidence", "correction", "private")
HOME = {"assertion": "goal", "evidence": "process", "correction": "tendency",
        "private": "tendency"}
N_TOKENS = 6
N_ACTIONS = 5                       # which evidence item is presented next
N_EVIDENCE = 5
N_BELIEF = 4
SELECTION = ("sample_all", "cherry_pick", "balanced", "escalate")
MOTIVE = ("sincere", "strategic", "mixed", "contrarian")
#: Motives that emit the *same* artifact. Spec §2: the static-artifact boundary must be a
#: boundary, not a signature -- a sincere fanatic and a strategic propagandist assert the same
#: thing, choose evidence the same way and correct at the same rate. Only a purchased
#: counterfactual separates them, and S04 is allowed to find that even that fails.
SURFACE_PROFILE = {"sincere": 0, "strategic": 0, "mixed": 1, "contrarian": 1}
COLLISION_CLASSES = {0: ("sincere", "strategic"), 1: ("mixed", "contrarian")}


def motive_of(v: int) -> str:
    return MOTIVE[v % len(MOTIVE)]


def profile_of(v: int) -> int:
    return SURFACE_PROFILE[motive_of(v)]


def collision_class(v: int) -> tuple:
    """The motives this one is observationally equivalent to on the artifact alone."""
    return COLLISION_CLASSES[profile_of(v)]


@dataclass
class CommunicationWorld:
    knobs: object
    n_p: int
    n_g: int
    n_v: int
    prior: np.ndarray
    emission: dict
    policy: np.ndarray                   # [n_p, n_g, n_context, N_ACTIONS] evidence selection
    contexts: int = 3
    evidence_support: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    belief_prior: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    action_cost: np.ndarray = field(default_factory=lambda: np.zeros(1))
    stake: float = 1.0
    tendency_context: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    beta: float = 0.0
    meta: dict = field(default_factory=dict)

    def latent_space(self) -> list:
        return [(p, g, v) for p in range(self.n_p) for g in range(self.n_g)
                for v in range(self.n_v)]

    def n_latent(self) -> int:
        return self.n_p * self.n_g * self.n_v


# --------------------------------------------------------------------------- #
# Construction.
# --------------------------------------------------------------------------- #
def _prior(n_p: int, n_g: int, n_v: int, kappa: float, rng) -> tuple:
    """Tempered Dirichlet: draw a rough joint, sharpen it until its coupling hits the target, and
    fit it back onto uniform marginals. A third construction reaching the same declared semantics.
    """
    uniform = np.full((n_p, n_g, n_v), 1.0 / (n_p * n_g * n_v))
    if kappa <= 0.0:
        return uniform, 0.0, pairwise_coupling(uniform)
    raw = rng.dirichlet(np.full(n_p * n_g * n_v, 0.12)).reshape(n_p, n_g, n_v)
    lograw = np.log(np.maximum(raw, 1e-300))
    target = target_coupling_nats(kappa, n_p, n_g)

    def build(t: float) -> np.ndarray:
        # temper in log space and renormalize before fitting: raw ** t underflows
        return fit_uniform_marginals(C.softmax((t * lograw).ravel()).reshape(raw.shape))

    def mean_pw(w) -> float:
        pc = pairwise_coupling(w)
        return float(np.mean([pc["process_goal"], pc["process_tendency"], pc["goal_tendency"]]))

    lo, hi = 0.0, 6.0
    if mean_pw(build(hi)) <= target:
        prior, t = build(hi), hi
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if mean_pw(build(mid)) < target:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        prior = build(t)
    return prior, float(t), pairwise_coupling(prior)


def _selection_distribution(world, p: int, g: int, context: int) -> np.ndarray:
    """Which evidence item this selection policy shows next, given the audience it is steering.

    ``sample_all`` is near-uniform over the pool; ``cherry_pick`` concentrates on items that
    support the target audience state; ``balanced`` mixes support and counter-evidence;
    ``escalate`` walks up the support ordering. S07 asks whether a reader can tell a selected
    sample from a random one, so these must differ in *shape* and not only in mean.
    """
    sup = world.evidence_support[:, g]                      # how much each item supports goal g
    name = SELECTION[p % len(SELECTION)]
    if name == "sample_all":
        logit = np.zeros(N_EVIDENCE)
    elif name == "cherry_pick":
        logit = 3.0 * sup
    elif name == "balanced":
        logit = 1.1 * np.abs(sup - sup.mean())
    else:
        logit = 2.2 * np.argsort(np.argsort(sup)).astype(float) / max(N_EVIDENCE - 1, 1)
    logit = logit + 0.55 * world.evidence_support[:, (g + context) % world.n_g]
    return C.softmax(logit)


def _private_action_distribution(world, v: int, belief: int) -> np.ndarray:
    """The costly private action, produced by a planner rather than read off the belief.

    ``alignment`` is how well an action matches what the belief would recommend, ``stake`` how much
    that matters, ``action_cost`` what the action costs. A high-cost action taken by a weak belief
    and a low-cost action taken by a strong one can land on the same token, which is exactly the
    identification problem S03-S05 are for.
    """
    align = np.cos(np.linspace(0, np.pi, N_ACTIONS) - belief * np.pi / max(N_BELIEF - 1, 1))
    prof = profile_of(v)
    # The *observed* private action depends on the surface profile only, so it cannot be read
    # as a belief readout and cannot separate a collision pair. Spec §2 and pre-mortem item 12.
    stake = world.stake * (1.0 if prof == 0 else 0.8)
    util = stake * (align if prof == 0 else -align) - world.action_cost
    return C.softmax(util / 0.6)


def sample_world(knobs, rng) -> CommunicationWorld:
    n_p, n_g, n_v = knobs.n_process, knobs.n_goal, knobs.n_tendency
    s = rng.integers(0, 2 ** 62, size=5)
    r_prior, r_ev, r_bel, r_cost, r_ctx = (np.random.default_rng(int(x)) for x in s)

    prior, beta, coupling = _prior(n_p, n_g, n_v, knobs.kappa, r_prior)
    evidence_support = r_ev.normal(size=(N_EVIDENCE, n_g))
    # indexed by SURFACE PROFILE, not by motive: two motives that share a profile must emit
    # the same private-action distribution or the artifact separates them for free
    n_prof = len(set(SURFACE_PROFILE.values()))
    belief_prior = C.softmax(r_bel.normal(size=(n_prof, N_BELIEF)) * 1.4, axis=-1)
    action_cost = np.abs(r_cost.normal(size=N_ACTIONS)) * 0.8
    tc = C.softmax(r_ctx.normal(size=(n_v, 3)) * 1.1, axis=-1)

    w = CommunicationWorld(knobs=knobs, n_p=n_p, n_g=n_g, n_v=n_v, prior=prior, emission={},
                           policy=np.zeros((n_p, n_g, 3, N_ACTIONS)), contexts=3,
                           evidence_support=evidence_support, belief_prior=belief_prior,
                           action_cost=action_cost, stake=1.6, tendency_context=tc, beta=beta)
    w.emission = _emission(w, knobs)
    w.policy = np.stack([[[_selection_distribution(w, p, g, c) for c in range(3)]
                          for g in range(n_g)] for p in range(n_p)])
    ri = realized_route_information(w.emission, prior)
    ref = np.full_like(prior, 1.0 / prior.size)
    ri_ref = realized_route_information(w.emission, ref)
    w.meta = {"family": FAMILY, "pairwise_coupling": coupling,
              "realized_coupling": float(np.mean([coupling["process_goal"],
                                                  coupling["process_tendency"],
                                                  coupling["goal_tendency"]])),
              "target_coupling": target_coupling_nats(knobs.kappa, n_p, n_g),
              "route_information": ri, "route_information_reference": ri_ref,
              "overlap_index": overlap_index(ri_ref), "temper": beta,
              "marginal_uniformity": float(max(
                  abs(prior.sum(axis=(1, 2)) - 1.0 / n_p).max(),
                  abs(prior.sum(axis=(0, 2)) - 1.0 / n_g).max(),
                  abs(prior.sum(axis=(0, 1)) - 1.0 / n_v).max()))}
    return w


def _emission(world, knobs) -> dict:
    """Token tables, with the private belief integrated out of the private-action channel."""
    n_p, n_g, n_v = world.n_p, world.n_g, world.n_v
    out = {r: np.zeros((n_p, n_g, n_v, N_TOKENS)) for r in ROUTES}
    for p in range(n_p):
        for g in range(n_g):
            for v in range(n_v):
                sel = _selection_distribution(world, p, g, 0)
                ev = np.zeros(N_TOKENS)
                for i in range(N_EVIDENCE):
                    ev[i % N_TOKENS] += sel[i]
                out["evidence"][p, g, v] = ev + 0.02

                # the assertion is about the audience state being steered toward
                a = np.zeros(N_TOKENS)
                a[g % N_TOKENS] += 1.6
                a[(g + 2) % N_TOKENS] += 0.5
                out["assertion"][p, g, v] = C.softmax(a)

                # correction record: a standing property of the motive
                cidx = 0.0 if profile_of(v) == 0 else 1.2
                cc = np.zeros(N_TOKENS)
                for k in range(N_TOKENS):
                    cc[k] = -abs(k - (cidx * (N_TOKENS - 1) / 1.4))
                out["correction"][p, g, v] = C.softmax(1.3 * cc)

                # private action: integrate the private belief out
                pa = np.zeros(N_ACTIONS)
                for b in range(N_BELIEF):
                    pa += world.belief_prior[profile_of(v), b] \
                        * _private_action_distribution(world, v, b)
                pt = np.zeros(N_TOKENS)
                for i in range(N_ACTIONS):
                    pt[i % N_TOKENS] += pa[i]
                out["private"][p, g, v] = pt + 0.02

    for r in out:
        out[r] = out[r] / out[r].sum(axis=-1, keepdims=True)
        if knobs.overlap > 0:
            flat = out[r].mean(axis=(0, 1, 2), keepdims=True)
            m = (1 - 0.5 * knobs.overlap) * out[r] + 0.5 * knobs.overlap * flat
            out[r] = m / m.sum(axis=-1, keepdims=True)
    if knobs.dependence == "redundant":
        out["correction"] = out["private"].copy()
    return out


def sample_latent(world, rng) -> Latent:
    flat = world.prior.ravel()
    i = int(rng.choice(flat.size, p=flat / flat.sum()))
    p, g, v = np.unravel_index(i, world.prior.shape)
    b = int(rng.choice(N_BELIEF, p=world.belief_prior[profile_of(v)]))
    return Latent(process=int(p), goal=int(g), tendency=int(v), extra={"belief": b})


def rollout(world, latent: Latent, rng, n_steps: int = 12) -> Episode:
    k = world.knobs
    p, g, v = latent.process, latent.goal, latent.tendency
    belief = int(latent.extra.get("belief", 0))
    context = int(rng.choice(world.contexts, p=world.tendency_context[v]))
    routes = {r: [] for r in ROUTES}
    shown, priv_actions = [], []

    for t in range(n_steps):
        sel = _selection_distribution(world, p, g, context)
        item = int(rng.choice(N_EVIDENCE, p=sel))
        if rng.random() >= k.competence:
            item = int(rng.integers(N_EVIDENCE))
        shown.append(item)
        pa = _private_action_distribution(world, v, belief)
        act = int(rng.choice(N_ACTIONS, p=pa))
        priv_actions.append(act)
        routes["evidence"].append(int(item % N_TOKENS))
        routes["assertion"].append(int(rng.choice(N_TOKENS, p=world.emission["assertion"][p, g, v])))
        routes["correction"].append(int(rng.choice(N_TOKENS, p=world.emission["correction"][p, g, v])))
        routes["private"].append(int(act % N_TOKENS))

    nxt_ctx = int(rng.choice(world.contexts, p=world.tendency_context[v]))
    sel_next = _selection_distribution(world, p, g, nxt_ctx)
    hidden = {"next_evidence_selection": int(rng.choice(N_EVIDENCE, p=sel_next)),
              "next_action": int(rng.choice(N_EVIDENCE, p=sel_next)),
              "changed_context_choice": int(np.argmax(world.tendency_context[v]))}
    return Episode(routes=routes, context=context, opportunity=tuple(range(N_EVIDENCE)),
                   hidden=hidden, latent=latent,
                   meta={"family": FAMILY, "actions": shown, "private_actions": priv_actions,
                         "belief": belief, "next_context": nxt_ctx})


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


def action_loglik(world, triple, episode, upto: int, budget=None) -> float:
    p, g, _ = triple
    acts = episode.meta.get("actions", [])[:upto]
    if not acts:
        return 0.0
    k = world.knobs
    sel = _selection_distribution(world, p, g, episode.context)
    mix = k.competence * sel + (1.0 - k.competence) / N_EVIDENCE
    if budget is not None:
        budget.lik(1)
    return float(np.log(np.maximum(mix[np.asarray(acts, int)], 1e-300)).sum())


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
    if t < len(acts):
        k = world.knobs
        sel = _selection_distribution(world, p, g, episode.context)
        mix = k.competence * sel + (1.0 - k.competence) / N_EVIDENCE
        tot += float(np.log(max(mix[acts[t]], 1e-300)))
    if budget is not None:
        budget.lik(len(ROUTES) + 1)
    return tot


def log_prior(world, triple) -> float:
    return float(np.log(max(world.prior[triple], 1e-300)))


def endpoint_dist(world, triple, episode, endpoint: str = "next_evidence_selection",
                  budget=None) -> np.ndarray:
    p, g, v = triple
    if budget is not None:
        budget.lik(1)
    if endpoint in ("next_evidence_selection", "next_action"):
        ctx = episode.meta.get("next_context", episode.context)
        return np.asarray(_selection_distribution(world, p, g, ctx), float)
    if endpoint == "changed_context_choice":
        return np.asarray(world.tendency_context[v], float)
    raise ValueError(endpoint)


def endpoint_size(endpoint: str, world) -> int:
    return world.contexts if endpoint == "changed_context_choice" else N_EVIDENCE


def marginal_prior(world) -> dict:
    p = world.prior
    return {"process": p.sum(axis=(1, 2)), "goal": p.sum(axis=(0, 2)),
            "tendency": p.sum(axis=(0, 1))}


def surface_predictor(episode, upto: int, endpoint: str, world) -> np.ndarray:
    n = endpoint_size(endpoint, world)
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
    return {"context": int(episode.context), "opportunity": list(episode.opportunity),
            "action_distribution": _selection_distribution(world, p, g, episode.context).tolist(),
            "process_constraints": {"selection_policy": SELECTION[p % len(SELECTION)],
                                    "motive": MOTIVE[v % len(MOTIVE)],
                                    "competence": float(world.knobs.competence)},
            "stopping_rule": {"kind": "fixed_length", "n": int(episode.n_steps())},
            "intervention_delta": {"context_swap": float(C.tv(
                _selection_distribution(world, p, g, episode.context),
                _selection_distribution(world, p, g, (episode.context + 1) % world.contexts)))}}


# --------------------------------------------------------------------------- #
# Source-trunk instruments (spec §6 trunk S).
# --------------------------------------------------------------------------- #
def motive_posterior(world, post: np.ndarray) -> dict:
    """Collapse a latent posterior onto the standing motive, which is what trunk S asks about."""
    m = post.sum(axis=(0, 1))
    return {MOTIVE[i % len(MOTIVE)]: float(m[i]) for i in range(post.shape[2])}


def counterfactual_probe(world, latent: Latent, probe: str, rng) -> dict:
    """One counterfactual opportunity (spec §6 S03), generated through the same planner.

    ``audience_persuaded``  the audience already believes the target: steering has no value.
    ``private_cost``        the private action's cost is raised.
    ``correction``          a public correction opportunity is offered.
    ``evidence_choice``     a forced choice between a supporting and an undermining item.
    """
    v, g, p = latent.tendency, latent.goal, latent.process
    belief = int(latent.extra.get("belief", 0))
    motive = MOTIVE[v % len(MOTIVE)]
    if probe == "audience_persuaded":
        # a strategic source has nothing left to gain; a sincere one still acts on its belief
        stake = world.stake * (0.05 if motive in ("strategic", "mixed") else 1.0)
        util = stake * np.cos(np.linspace(0, np.pi, N_ACTIONS)
                              - belief * np.pi / max(N_BELIEF - 1, 1)) - world.action_cost
        d = C.softmax(util / 0.6)
    elif probe == "private_cost":
        util = world.stake * np.cos(np.linspace(0, np.pi, N_ACTIONS)
                                    - belief * np.pi / max(N_BELIEF - 1, 1)) - 2.5 * world.action_cost
        d = C.softmax(util / 0.6)
    elif probe == "correction":
        rate = {"sincere": 0.75, "strategic": 0.15, "mixed": 0.45, "contrarian": 0.3}[motive]
        d = np.array([1.0 - rate, rate])
    elif probe == "evidence_choice":
        sup = world.evidence_support[:, g]
        sharp = {"sincere": 0.8, "strategic": 3.0, "mixed": 1.6, "contrarian": 2.0}[motive]
        d = C.softmax(sharp * sup)
    else:
        raise ValueError(probe)
    return {"probe": probe, "distribution": d, "draw": int(rng.choice(len(d), p=d))}


def probe_likelihood(world, triple, probe: str, outcome: int, budget=None) -> float:
    """The reader's likelihood for a probe outcome, marginalizing the private belief it cannot
    see. This is the channel that replaces V14's direct belief readout."""
    p, g, v = triple
    acc = 0.0
    for b in range(N_BELIEF):
        stub = Latent(process=p, goal=g, tendency=v, extra={"belief": b})
        d = counterfactual_probe(world, stub, probe, np.random.default_rng(0))["distribution"]
        acc += float(world.belief_prior[profile_of(v), b]) * float(d[outcome % len(d)])
    if budget is not None:
        budget.lik(N_BELIEF)
    return float(np.log(max(acc, 1e-300)))
