"""Exact enumeration over the latent grid, and the cheap exact rivals built from it.

Every family exposes the same small surface -- ``latent_space``, ``log_prior``, ``loglik``,
``endpoint_dist``, ``endpoint_size``, ``marginal_prior`` -- so this module is written once against
the *module object* of a family and never against a particular world. ``F`` below is always that
module.

Exactness is the anchor of the whole program (spec §3.5): approximate architectures must reproduce
these posteriors on small worlds before they are allowed to run anywhere else, and card M01 is the
receipt. Nothing here is approximate, so nothing here has a tuning parameter.

The three exact readers
-----------------------
``joint_posterior``
    the full posterior over the latent grid under the world's own coupled prior.
``independent_posterior``
    each component inferred separately from the routes that are *about* it, under a factorized
    prior, and then multiplied. This is V14's "independent marginals" rival, generalized. Where the
    prior factorizes and each route touches one component it equals the joint exactly, which is the
    identity that makes the atlas's zero corner meaningful rather than merely small.
``staged_posterior``
    commit to one component, then infer the next conditioned on that commitment. Fixed orders and
    an adaptive order chosen from current uncertainty. A staged reader cannot revise a wrong early
    commitment, which is the failure M06 measures.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from .ontology import COMPONENTS

CHANNELS_ALL = ("routes", "actions")


# --------------------------------------------------------------------------- #
# The log-likelihood table over the whole grid: the one primitive everything reuses.
# --------------------------------------------------------------------------- #
def loglik_table(F, world, ep, upto: int, channels=CHANNELS_ALL, routes=None, budget=None,
                 model=None) -> np.ndarray:
    """``[n_p, n_g, n_v]`` of log p(observations up to ``upto`` | latent).

    ``model`` is an optional reader-side view of the world (a misspecified copy). Passing the
    world itself is the correctly-specified case.
    """
    w = model if model is not None else world
    shape = (world.n_p, world.n_g, world.n_v)
    out = np.zeros(shape)
    for p in range(shape[0]):
        for g in range(shape[1]):
            for v in range(shape[2]):
                t = (p, g, v)
                s = 0.0
                if "routes" in channels:
                    s += F.route_loglik(w, t, ep, upto, routes, budget)
                if "actions" in channels:
                    s += F.action_loglik(w, t, ep, upto, budget)
                out[p, g, v] = s
    return out


def log_prior_table(F, world, model=None) -> np.ndarray:
    w = model if model is not None else world
    return np.log(np.maximum(np.asarray(w.prior, float), 1e-300))


def joint_posterior(F, world, ep, upto: int, channels=CHANNELS_ALL, routes=None, budget=None,
                    model=None, prior_override=None) -> np.ndarray:
    lg = loglik_table(F, world, ep, upto, channels, routes, budget, model)
    lp = np.log(np.maximum(prior_override, 1e-300)) if prior_override is not None \
        else log_prior_table(F, world, model)
    return C.softmax((lg + lp).ravel()).reshape(lg.shape)


def factorized_prior(F, world, model=None) -> np.ndarray:
    """The product of the world's own one-dimensional marginals: the prior an independent reader
    uses. Under the chain family's marginal-preserving coupling this is exactly uniform, so the
    independent reader is never handicapped by a wrong marginal -- only by the missing dependence.
    """
    w = model if model is not None else world
    m = F.marginal_prior(w)
    return np.einsum("i,j,k->ijk", m["process"], m["goal"], m["tendency"])


def independent_posterior(F, world, ep, upto: int, home_routes: dict | None = None,
                          channels=CHANNELS_ALL, budget=None, model=None,
                          routed: bool = False) -> np.ndarray:
    """The independent-marginals rival: infer under a factorized prior, then project onto a
    product of marginals.

    The evidence is used **once**. An earlier version of this function gave each component its own
    home routes *and* left the shared action channel in every component's likelihood, which
    multiplied the policy evidence by three and made the rival sharper than the exact posterior on
    a policy-generated endpoint -- the joint came out 0.017 nats *behind* exact inference, which is
    impossible for a correctly specified Bayes posterior and was the tell.

    ``routed=True`` selects the cheaper variant that assigns each component only the routes that
    are about it. That reader is a genuine rival for the routing trunk, and it is a different
    object from this one: it discards cross-route evidence rather than merely discarding the
    prior's dependence.
    """
    fp = factorized_prior(F, world, model)
    if routed:
        marg = {}
        hr = home_routes or {}
        for i, comp in enumerate(COMPONENTS):
            lg = loglik_table(F, world, ep, upto, channels=("routes",),
                              routes=hr.get(comp, ()), budget=budget, model=model)
            w_ = fp * np.exp(lg - lg.max())
            axes = tuple(a for a in range(3) if a != i)
            marg[comp] = C.normalize(w_.sum(axis=axes))
        return np.einsum("i,j,k->ijk", marg["process"], marg["goal"], marg["tendency"])

    lg = loglik_table(F, world, ep, upto, channels=channels, routes=None, budget=budget,
                      model=model)
    post = C.softmax((lg + np.log(np.maximum(fp, 1e-300))).ravel()).reshape(lg.shape)
    m = [post.sum(axis=tuple(a for a in range(3) if a != i)) for i in range(3)]
    return np.einsum("i,j,k->ijk", *[C.normalize(x) for x in m])


def staged_posterior(F, world, ep, upto: int, order=("process", "goal", "tendency"),
                     channels=CHANNELS_ALL, budget=None, model=None, adaptive: bool = False,
                     home_routes: dict | None = None) -> tuple:
    """Commit to components one at a time. Returns (posterior, committed order).

    A commitment is a hard argmax: the reader picks a value and conditions everything later on it.
    That is what makes the order matter and what makes an early error unrecoverable.
    """
    lg = loglik_table(F, world, ep, upto, channels, None, budget, model)
    lp = log_prior_table(F, world, model)
    post = C.softmax((lg + lp).ravel()).reshape(lg.shape)
    axis_of = {c: i for i, c in enumerate(COMPONENTS)}
    fixed, chosen = {}, []
    remaining = list(order)
    while remaining:
        if adaptive:
            ent = {}
            for c in remaining:
                i = axis_of[c]
                m = post.sum(axis=tuple(a for a in range(3) if a != i))
                ent[c] = C.entropy(C.normalize(m))
            comp = min(remaining, key=lambda c: ent[c])       # most certain first
        else:
            comp = remaining[0]
        remaining.remove(comp)
        i = axis_of[comp]
        m = post.sum(axis=tuple(a for a in range(3) if a != i))
        val = int(np.argmax(m))
        fixed[comp] = val
        chosen.append(comp)
        sl = [slice(None)] * 3
        sl[i] = val
        mask = np.zeros_like(post)
        mask[tuple(sl)] = 1.0
        post = C.normalize((post * mask).ravel()).reshape(post.shape)
    return post, chosen


# --------------------------------------------------------------------------- #
# Turning a latent posterior into a prediction about a hidden event.
# --------------------------------------------------------------------------- #
def predictive(F, world, ep, post: np.ndarray, endpoint: str, budget=None, model=None,
               cache: dict | None = None) -> np.ndarray:
    """Mix each latent's endpoint distribution by its posterior mass.

    The endpoint distributions come from the *world*, not the reader's model: every architecture
    is asked to predict the same real event, and only its posterior over latents differs. A reader
    whose model is wrong is punished through its posterior, which is the intended channel.
    """
    n = F.endpoint_size(endpoint, world)
    out = np.zeros(n)
    it = np.nditer(post, flags=["multi_index"])
    for m in it:
        w = float(m)
        if w <= 1e-9:
            continue
        t = it.multi_index
        if cache is not None:
            d = cache.get(t)
            if d is None:
                d = F.endpoint_dist(world, t, ep, endpoint, budget)
                cache[t] = d
        else:
            d = F.endpoint_dist(world, t, ep, endpoint, budget)
        out += w * np.asarray(d, float)
    return C.normalize(out)


def oracle_state_predictive(F, world, ep, endpoint: str, budget=None) -> np.ndarray:
    """Upper bound: the true latent, exactly. Never promotable (spec §3.4.11)."""
    return np.asarray(F.endpoint_dist(world, ep.latent.triple(), ep, endpoint, budget), float)


# --------------------------------------------------------------------------- #
# Brute-force checks for the identity cards (I07).
# --------------------------------------------------------------------------- #
def brute_force_posterior(F, world, ep, upto: int, channels=CHANNELS_ALL) -> np.ndarray:
    """The same posterior computed the slow, obviously-correct way: unnormalized products in
    linear space, no log-sum-exp, no vectorisation. Disagreement means the fast path is wrong."""
    shape = (world.n_p, world.n_g, world.n_v)
    out = np.zeros(shape)
    for p in range(shape[0]):
        for g in range(shape[1]):
            for v in range(shape[2]):
                t = (p, g, v)
                acc = float(world.prior[t])
                if "routes" in channels:
                    for r in F.ROUTES:
                        for tok in ep.routes.get(r, [])[:upto]:
                            acc *= float(world.emission[r][p, g, v][tok])
                if "actions" in channels:
                    k = world.knobs
                    pol = world.policy[p, g, ep.context]
                    mix = k.competence * pol + (1.0 - k.competence) / len(pol)
                    for a in ep.meta.get("actions", [])[:upto]:
                        acc *= float(mix[a])
                out[t] = acc
    s = out.sum()
    return out / s if s > 0 else np.full(shape, 1.0 / out.size)


def relabel_invariance(F, world, ep, upto: int, perm: np.ndarray, axis: int = 0) -> float:
    """Permuting the *names* of a latent's values must permute the posterior and change nothing
    else. Returns the max absolute deviation."""
    base = joint_posterior(F, world, ep, upto)
    import copy
    w2 = copy.deepcopy(world)
    w2.prior = np.take(world.prior, perm, axis=axis)
    for r in w2.emission:
        w2.emission[r] = np.take(world.emission[r], perm, axis=axis)
    if axis == 0:
        w2.policy = np.take(world.policy, perm, axis=0)
    elif axis == 1:
        w2.policy = np.take(world.policy, perm, axis=1)
    else:
        w2.tendency_context = np.take(world.tendency_context, perm, axis=0)
    got = joint_posterior(F, w2, ep, upto)
    return float(np.abs(np.take(base, perm, axis=axis) - got).max())
