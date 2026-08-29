"""Routes: information, planted ease, learned reliability, conflict, hypothesis expansion,
fusion under duplicated evidence, forensic purchase, and transfer (spec §3.2, trunk R).

Ease is planted: a declared processing penalty per route (world.ROUTE_COST) and the number of
evidence doses a route needs to bring the target's posterior entropy under a threshold. It is
never called fluency. Reliability is LEARNED from feedback on training makers (a held-out
prediction score per route) and applied to test makers as a tempering weight per route, with
no target label at test time.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from . import joint as J
from .world import N_ACT, N_GOAL, N_PLAN, N_PREF, ROUTE_COST, ROUTES, Maker, World, episode, make_maker

_TINY = 1e-300


# --------------------------------------------------------------------------- #
# Information and ease.
# --------------------------------------------------------------------------- #
def route_information(world: World, reader: J.Reader, rng: np.random.Generator, n: int = 48, doses: int = 1) -> dict:
    """Expected reduction of each latent's marginal entropy from one route alone (nats), estimated
    by sampling states from the uniform prior and episodes from a maker with that state."""
    prior = J.uniform_prior()
    h0 = {lat: C.entropy(J.marginal(prior, lat)) for lat in J.LATENTS}
    acc = {r: {lat: [] for lat in J.LATENTS} for r in ROUTES}
    for _ in range(int(n)):
        pl, g, pr = J.STATES[int(rng.integers(J.N_STATES))]
        m = make_maker(world, "probe", rng, family=reader.fam_index, pref=pr, plan=pl, competence="mid")
        eps = [episode(world, m, rng, index=i, goal=g) for i in range(doses)]
        tabs = reader.route_tables(eps, ROUTES)
        for r in ROUTES:
            post = J.posterior(prior, tabs[r])
            for lat in J.LATENTS:
                acc[r][lat].append(h0[lat] - C.entropy(J.marginal(post, lat)))
    return {r: {lat: float(np.mean(v)) for lat, v in d.items()} for r, d in acc.items()}


def doses_to_entropy(world: World, reader: J.Reader, m: Maker, rng: np.random.Generator, route: str, latent: str,
                     threshold: float, max_doses: int = 12) -> int:
    prior = J.uniform_prior()
    tabs = {route: np.zeros(J.N_STATES)}
    for d in range(1, max_doses + 1):
        ep = episode(world, m, rng, index=d)
        tabs[route] += reader.route_tables([ep], (route,))[route]
        if C.entropy(J.marginal(J.posterior(prior, tabs[route]), latent)) <= threshold:
            return d
    return max_doses + 1


def ease(route: str, penalty_override: dict | None = None) -> float:
    return float((penalty_override or ROUTE_COST)[route])


# --------------------------------------------------------------------------- #
# Learned reliability (R02): feedback on training makers, weights applied at test.
# --------------------------------------------------------------------------- #
T_SEEN = 3


def partial(ep: dict, t: int = T_SEEN) -> dict:
    out = dict(ep)
    out["action"] = list(ep["action"][:t])
    return out


def within_gain(reader: J.Reader, post: np.ndarray, prior: np.ndarray, ep_full: dict, t: int = T_SEEN) -> float:
    """Log score of step t+1 of the current episode from its first t steps, over the prior."""
    a, last = int(ep_full["action"][t]), int(ep_full["action"][t - 1])
    pred, base = J.next_action_dist(reader, post, last), J.next_action_dist(reader, prior, last)
    return float(np.log(max(pred[a], 1e-12)) - np.log(max(base[a], 1e-12)))


def make_training(world: World, r: np.random.Generator, n: int, fam: int = 0, k_eps: int = 2, noise_fn=None) -> list:
    """Training items: (episodes seen, the full current episode). The current episode is seen
    to step T_SEEN; its next step is the feedback target."""
    out = []
    for i in range(int(n)):
        m = make_maker(world, f"t{i}", r, family=fam, competence="mid")
        eps = [episode(world, m, r, index=k) for k in range(k_eps + 1)]
        if noise_fn is not None:
            eps = [noise_fn(e) for e in eps]
        out.append((eps[:k_eps] + [partial(eps[k_eps])], eps[k_eps]))
    return out


def route_gain(reader: J.Reader, eps_seen: list, ep_full: dict, route: str, prior: np.ndarray) -> float:
    """Held-out within-episode next-step log score of a route-only posterior over the prior."""
    tabs = reader.route_tables(eps_seen, (route,))
    return within_gain(reader, J.posterior(prior, tabs[route]), prior, ep_full)


def learn_reliability(reader: J.Reader, training: list, prior: np.ndarray, routes=ROUTES, beta: float = 8.0) -> dict:
    """``training``: list of (episodes seen, full current episode). The reliability learner fits
    tempering weights in {0, 0.5, 1} per route to the training feedback (the held-out next step),
    by exhaustive search over the grid; it never sees a target label at test. Also returns each
    route's gain alone, for the record."""
    import itertools
    tabs = [reader.route_tables(eps_seen, routes) for eps_seen, _ in training]
    gains = {r: [] for r in routes}
    for tb, (eps_seen, ep_full) in zip(tabs, training):
        for r in routes:
            gains[r].append(within_gain(reader, J.posterior(prior, tb[r]), prior, ep_full))
    best, best_score = None, -np.inf
    for combo in itertools.product((0.0, 0.5, 1.0), repeat=len(routes)):
        if max(combo) < 1.0:
            continue                                             # tempering only: some route keeps full weight
        w = dict(zip(routes, combo))
        score = float(np.mean([within_gain(reader, J.joint(prior, tb, w), prior, ep_full) for tb, (_, ep_full) in zip(tabs, training)]))
        if score > best_score + 1e-12:
            best, best_score = w, score
    return {r: float(best[r]) for r in routes}, {r: float(np.mean(gains[r])) for r in routes}


def weights_named(kind: str, routes=ROUTES, learned: dict | None = None, rng=None, ease_map: dict | None = None) -> dict:
    n = len(routes)
    if kind == "learned":
        return dict(learned)
    if kind == "equal":
        return {r: 1.0 for r in routes}
    if kind == "random":
        w = rng.dirichlet(np.ones(n)); w = w / w.max()
        return {r: float(x) for r, x in zip(routes, w)}
    if kind == "ease":                                          # the ease-driven reader: cheaper routes weigh more
        e = np.array([ease(r, ease_map) for r in routes])
        w = (1.0 / e); w = w / w.max()
        return {r: float(x) for r, x in zip(routes, w)}
    if kind == "fixed_action":
        return {r: (1.0 if r == "action" else 0.0) for r in routes}
    raise KeyError(kind)


# --------------------------------------------------------------------------- #
# Conflict and hypothesis expansion (R05).
# --------------------------------------------------------------------------- #
def conflict(prior: np.ndarray, tabs: dict) -> float:
    """Mean pairwise Jensen-Shannon divergence between route-only posteriors."""
    posts = [J.posterior(prior, t) for t in tabs.values()]
    if len(posts) < 2:
        return 0.0
    vals = [C.js(posts[i], posts[j]) for i in range(len(posts)) for j in range(i + 1, len(posts))]
    return float(np.mean(vals))


def strategic_semantic_table(reader: J.Reader, eps: list) -> np.ndarray:
    """Semantic likelihood under a STRATEGIC source: tokens advertise a goal other than the true one
    (uniform mixture over the other goals). Used to expand the latent set with a source flag."""
    per_goal = np.zeros(N_GOAL)
    for ep in eps:
        for g in range(N_GOAL):
            others = [gg for gg in range(N_GOAL) if gg != g]
            ll_others = [sum(np.log(reader.k_obs * reader.fam.sem[gg][tok] + (1 - reader.k_obs) / 10) for tok in ep["semantic"]) for gg in others]
            per_goal[g] += C.logsumexp(np.array(ll_others)) - np.log(len(others))
    return per_goal[J._G]


def expanded_posterior(prior: np.ndarray, tabs: dict, strategic_tab: np.ndarray, p_strategic: float = 0.2) -> tuple:
    """Joint over (state, source flag). Returns the state posterior (flag marginalized) and the
    posterior probability that the source was strategic."""
    honest = np.log(np.maximum(prior, _TINY)) + J.combined(tabs) + np.log(1 - p_strategic)
    strat = np.log(np.maximum(prior, _TINY)) + J.combined({k: v for k, v in tabs.items() if k != "semantic"}) + strategic_tab + np.log(p_strategic)
    both = np.concatenate([honest, strat])
    post = C.softmax(both)
    return post[:J.N_STATES] + post[J.N_STATES:], float(post[J.N_STATES:].sum())


# --------------------------------------------------------------------------- #
# Fusion under duplicated or correlated evidence (R07, attack X04).
# --------------------------------------------------------------------------- #
def duplicate_semantic(ep: dict, rng: np.random.Generator, kind: str = "duplicate") -> dict:
    """A second copy of the semantic evidence: an exact duplicate, or a paraphrase (the same
    tokens in a new order with one token resampled from the same goal distribution)."""
    out = dict(ep)
    sem = list(ep["semantic"])
    if kind == "paraphrase":
        sem = [int(x) for x in rng.permutation(sem)]
    out["semantic_dup"] = sem
    out["dup_source"] = "same_cause"
    return out


def fused_tables(reader: J.Reader, eps: list, routes, fusion: str = "naive") -> dict:
    """``naive`` counts a duplicate as fresh evidence; ``shared_cause`` counts one copy per source."""
    tabs = reader.route_tables(eps, routes)
    if "semantic" in routes:
        for ep in eps:
            if ep.get("semantic_dup") is not None and fusion == "naive":
                tabs["semantic"] += reader.ll_semantic({"semantic": ep["semantic_dup"]})
    return tabs


# --------------------------------------------------------------------------- #
# Forensic purchase (R06): expected information gain per cost.
# --------------------------------------------------------------------------- #
def forensic_eig(world: World, reader: J.Reader, prior: np.ndarray, tabs: dict, latent: str, rng: np.random.Generator,
                 draws: int = 24) -> float:
    """Expected reduction in the latent's marginal entropy from one forensic observation, by
    sampling states from the current posterior and simulating the forensic fields."""
    post = J.joint(prior, tabs)
    h0 = C.entropy(J.marginal(post, latent))
    red = []
    for _ in range(int(draws)):
        pl, g, pr = J.STATES[int(rng.choice(J.N_STATES, p=post))]
        m = make_maker(world, "sim", rng, family=reader.fam_index, pref=pr, plan=pl, competence="mid")
        ep = episode(world, m, rng, goal=g)
        t2 = dict(tabs)
        t2["forensic"] = reader.ll_forensic(ep)
        red.append(h0 - C.entropy(J.marginal(J.joint(prior, t2), latent)))
    return float(np.mean(red))


def purchase_policy(kind: str, eig: float, cost: float, rng: np.random.Generator, threshold: float = 0.0) -> bool:
    if kind == "always":
        return True
    if kind == "never":
        return False
    if kind == "random":
        return bool(rng.random() < 0.5)
    return bool(eig / max(cost, 1e-9) > threshold)          # exact: buy when gain per cost clears the bar


# --------------------------------------------------------------------------- #
# Transfer (R08): a new domain is a new vocabulary and new route reliabilities.
# --------------------------------------------------------------------------- #
def transfer_weights(kind: str, learned_old: dict, learned_new: dict, routes=ROUTES) -> dict:
    if kind == "reset":
        return {r: 1.0 for r in routes}
    if kind == "full":
        return dict(learned_old)
    return {r: 0.5 * learned_old[r] + 0.5 for r in routes}     # partial: shrink toward equal
