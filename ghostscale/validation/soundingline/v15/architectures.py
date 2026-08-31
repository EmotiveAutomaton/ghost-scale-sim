"""The reader tournament (spec §3.4, §3.5): eleven architectures, one set of observations, and a
budget receipt for every one of them.

The rule that shapes this module is spec §3.5 and pre-mortem item 2: *a fashionable model must not
win by receiving more hypotheses, observations or compute*. So every reader is built behind the
same call, every likelihood evaluation debits a counter, and a card that reports an architecture
comparison without a budget receipt is refused at reduce time rather than trusted.

The eleven, and what each one is for
------------------------------------
``surface``            frequency and last-action baselines; no maker model at all.
``label_only``         correct latent labels, decontextualized. Spec §3.3's pointer-versus-state
                       test: a correct label that does not improve a hidden event is not
                       understanding, and this reader is how that is measured rather than asserted.
``independent``        per-component marginals multiplied. V14's rival, generalized.
``staged``             fixed and adaptive commitment orders; cannot revise an early commitment.
``joint_exact``        the full posterior. Only where the grid can be enumerated.
``factor_graph``       loopy belief propagation on pairwise factors; structured, not exhaustive.
``particle``           sequential Monte Carlo; can lose the right hypothesis.
``expand``             starts reduced and adds variables from a predeclared library.
``direct_predictor``   predicts the hidden event from observable features with no latent model.
                       Expected to win in domain and fail under intervention (M07).
``oracle_model_space``  correct variables and likelihood family, never hidden truth. The ceiling a
                       *method* may reach.
``oracle_state``       the true latent. An upper bound, never promotable.

Misspecification is applied to the reader's model, never to the generator. ``oracle_model_space``
is the one non-oracle-truth reader exempt from it, which is what makes "how much did being wrong
about the model cost?" a measurable quantity rather than a figure of speech.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import common as C
from . import exact as EX
from . import expansion as EXP
from .ontology import COMPONENTS, likelihood_family_error, reader_components
from .particles import ParticleFilter
from .schemas import NON_PROMOTABLE

#: Which routes each component is "about" when a reader assigns routes one-to-one.
HOME_ROUTES = {"process": ("action", "forensic"), "goal": ("semantic",),
               "tendency": ("context",)}

ALL = ("surface", "label_only", "independent", "independent_routed", "staged", "joint_exact",
       "factor_graph", "particle", "expand", "direct_predictor", "oracle_model_space",
       "oracle_state")
#: The comparison most architecture cards run: cheap rivals, the joint, and the two ceilings.
CORE = ("surface", "label_only", "independent", "staged", "joint_exact", "particle",
        "oracle_state")


@dataclass
class Reading:
    name: str
    dist: np.ndarray
    posterior: object = None
    budget: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def score(self, truth) -> dict:
        return {"log_score": C.log_score(self.dist, truth), "brier": C.brier(self.dist, truth),
                "correct": float(C.top1(self.dist) == truth),
                "confidence": C.confidence(self.dist)}


# --------------------------------------------------------------------------- #
# The reader's model of the world, which may be wrong (spec §5, X05-X07).
# --------------------------------------------------------------------------- #
def reader_model(F, world, model_space: str, rng):
    """A reader-side view of the world. ``correct`` returns the world itself."""
    if model_space in ("correct", "missing_latent", "extra_latent"):
        return world                       # these change the reader's variable set, not its tables
    err = likelihood_family_error(model_space)
    if err <= 0:
        return world
    w = copy.deepcopy(world)
    # A wrong likelihood family that is superficially well calibrated: the *marginal* token
    # distribution of every route is preserved exactly, and only the conditional structure is
    # scrambled. A reader checking its calibration in training sees nothing wrong.
    for r, tab in world.emission.items():
        perm = rng.permutation(tab.shape[-1])
        scrambled = np.take(tab, perm, axis=-1)
        mixed = (1.0 - err) * tab + err * scrambled
        w.emission[r] = mixed / mixed.sum(axis=-1, keepdims=True)
    return w


def _components_for(model_space: str) -> tuple:
    comps = reader_components(model_space)
    return tuple(c for c in comps if c in COMPONENTS)


# --------------------------------------------------------------------------- #
# Factor-graph reader: loopy belief propagation on pairwise factors.
# --------------------------------------------------------------------------- #
def factor_graph_posterior(F, world, ep, upto: int, budget=None, model=None, iters: int = 30,
                           damping: float = 0.3) -> tuple:
    """Structured message passing without enumerating the joint grid.

    The prior is represented by its three pairwise marginals (a pairwise approximation of the true
    joint), the observations by unary factors. On the resulting 3-cycle, belief propagation is
    *loopy*: it converges quickly here but is not exact, and the gap it leaves is precisely what
    M01 measures rather than assumes away.
    """
    w = model if model is not None else world
    prior = np.asarray(w.prior, float)
    sizes = prior.shape
    pw = {(0, 1): prior.sum(axis=2), (1, 2): prior.sum(axis=0), (0, 2): prior.sum(axis=1)}

    unary = []
    for i, comp in enumerate(COMPONENTS):
        lg = np.zeros(sizes[i])
        for val in range(sizes[i]):
            acc = 0.0
            for r in HOME_ROUTES[comp]:
                toks = ep.routes.get(r, [])[:upto]
                if not toks:
                    continue
                tab = w.emission[r]
                sub = tab[val].mean(axis=(0, 1)) if i == 0 else (
                    tab[:, val].mean(axis=(0, 1)) if i == 1 else tab[:, :, val].mean(axis=(0, 1)))
                sub = sub / sub.sum()
                acc += float(np.log(np.maximum(sub[np.asarray(toks, int)], 1e-300)).sum())
            lg[val] = acc
        unary.append(C.softmax(lg))
        if budget is not None:
            budget.lik(sizes[i])

    msg = {(a, b): np.ones(sizes[b]) / sizes[b] for a, b in
           [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]}
    for _ in range(iters):
        new = {}
        for (a, b) in msg:
            key = (a, b) if (a, b) in pw else (b, a)
            fac = pw[key] if key == (a, b) else pw[key].T
            incoming = np.ones(sizes[a])
            for c in range(3):
                if c in (a, b):
                    continue
                incoming = incoming * msg[(c, a)]
            belief_a = unary[a] * incoming
            m = fac.T @ belief_a if fac.shape[0] == sizes[a] else fac @ belief_a
            new[(a, b)] = C.normalize(m)
            if budget is not None:
                budget.lik(1)
        for k in msg:
            msg[k] = C.normalize((1 - damping) * new[k] + damping * msg[k])

    beliefs = []
    for i in range(3):
        b = unary[i].copy()
        for c in range(3):
            if c != i:
                b = b * msg[(c, i)]
        beliefs.append(C.normalize(b))
    post = np.einsum("i,j,k->ijk", *beliefs)
    return post, {"iterations": iters, "damping": damping}


# --------------------------------------------------------------------------- #
# Direct predictor: no latent labels anywhere (spec §3.4.9).
# --------------------------------------------------------------------------- #
class DirectPredictor:
    """Counts ``p(hidden event | last action, context)`` from training episodes of this world.

    It never sees a latent label, in training or at test. Card M07 asks whether it wins in domain
    and then fails when the context is intervened on -- the interpretable-model trade the whole
    trunk is about.
    """

    def __init__(self, n_out: int, n_ctx: int, n_act: int, alpha: float = 0.5):
        self.table = np.full((n_ctx, n_act + 1, n_out), alpha)
        self.n_out, self.n_ctx, self.n_act = n_out, n_ctx, n_act

    def fit(self, episodes, endpoint: str, budget=None):
        for ep in episodes:
            acts = ep.meta.get("actions", [])
            last = acts[-1] if acts else self.n_act
            y = ep.hidden.get(endpoint)
            if y is None:
                continue
            self.table[ep.context % self.n_ctx, last, int(y)] += 1.0
            if budget is not None:
                budget.obs(1)
        return self

    def predict(self, ep, upto: int, budget=None) -> np.ndarray:
        acts = ep.meta.get("actions", [])[:upto]
        last = acts[-1] if acts else self.n_act
        if budget is not None:
            budget.lik(1)
        return C.normalize(self.table[ep.context % self.n_ctx, last])


# --------------------------------------------------------------------------- #
# Label-only reader (spec §3.3).
# --------------------------------------------------------------------------- #
def label_only_predictive(F, world, ep, endpoint: str, budget=None,
                          labels=None) -> np.ndarray:
    """Correct latent labels, stripped of context.

    The reader is handed the true triple and then asked to predict *without* the context-realized
    policy: the endpoint distribution is averaged over the contexts the label could have occurred
    in. This is what a decontextualized pointer buys, and it is the honest version of "the model
    knew the answer and still could not use it".
    """
    t = tuple(labels) if labels is not None else ep.latent.triple()
    n = F.endpoint_size(endpoint, world)
    out = np.zeros(n)
    n_ctx = getattr(world, "contexts", 1)
    for ctx in range(n_ctx):
        stub = copy.copy(ep)
        stub.meta = dict(ep.meta)
        stub.meta["next_context"] = ctx
        stub.context = ctx
        out += np.asarray(F.endpoint_dist(world, t, stub, endpoint, budget), float)
    return C.normalize(out)


# --------------------------------------------------------------------------- #
# The single entry point.
# --------------------------------------------------------------------------- #
def read(name: str, F, world, ep, upto: int, endpoint: str, *, rng, cfg: dict | None = None,
         training=None, calibration=None) -> Reading:
    cfg = dict(cfg or {})
    b = C.Budget()
    ms = cfg.get("model_space", getattr(world.knobs, "model_space", "correct"))
    model = world if name == "oracle_model_space" else reader_model(F, world, ms, rng)
    comps = COMPONENTS if name == "oracle_model_space" else _components_for(ms)

    with b.timing():
        if name == "surface":
            d, post, extra = F.surface_predictor(ep, upto, endpoint, world), None, {}
            b.lik(1)
        elif name == "label_only":
            d, post, extra = label_only_predictive(F, world, ep, endpoint, b), None, {}
        elif name == "independent":
            post = EX.independent_posterior(F, world, ep, upto, HOME_ROUTES, budget=b, model=model)
            d, extra = EX.predictive(F, world, ep, post, endpoint, b), {}
        elif name == "independent_routed":
            post = EX.independent_posterior(F, world, ep, upto, HOME_ROUTES, budget=b, model=model,
                                            routed=True)
            d, extra = EX.predictive(F, world, ep, post, endpoint, b), {"routed": True}
        elif name == "staged":
            order = tuple(cfg.get("order", ("process", "goal", "tendency")))
            post, chosen = EX.staged_posterior(F, world, ep, upto, order, budget=b, model=model,
                                               adaptive=bool(cfg.get("adaptive", False)))
            d, extra = EX.predictive(F, world, ep, post, endpoint, b), {"order": chosen}
        elif name in ("joint_exact", "oracle_model_space"):
            post = EX.joint_posterior(F, world, ep, upto, budget=b, model=model)
            if set(comps) != set(COMPONENTS):
                post = EXP._restricted_posterior(F, world, ep, upto, comps, budget=b, model=model)
            d, extra = EX.predictive(F, world, ep, post, endpoint, b), {"components": list(comps)}
        elif name == "factor_graph":
            post, extra = factor_graph_posterior(F, world, ep, upto, budget=b, model=model)
            d = EX.predictive(F, world, ep, post, endpoint, b)
        elif name == "particle":
            pf = ParticleFilter(F, world, int(cfg.get("n_particles", 240)), rng,
                                jitter=float(cfg.get("jitter", 0.0)), model=model, budget=b)
            post = pf.run(ep, upto)
            d = EX.predictive(F, world, ep, post, endpoint, b)
            extra = {"unique_particles": pf.unique(), "final_ess": pf.ess()}
        elif name == "expand":
            out = EXP.run_expansion(F, world, ep, upto, endpoint,
                                    selector=cfg.get("selector", "expected_value"),
                                    start=tuple(cfg.get("start", ("process", "goal"))),
                                    truly_missing=tuple(cfg.get("truly_missing", ())),
                                    rng=rng, budget=b, calibration=calibration or [])
            d, post = out.pop("prediction"), out.pop("posterior")
            extra = out
        elif name == "direct_predictor":
            dp = DirectPredictor(F.endpoint_size(endpoint, world), getattr(world, "contexts", 1),
                                 F.N_ACTIONS)
            dp.fit(training or [], endpoint, b)
            d, post, extra = dp.predict(ep, upto, b), None, {"n_train": len(training or [])}
        elif name == "oracle_state":
            d, post, extra = EX.oracle_state_predictive(F, world, ep, endpoint, b), None, {}
        else:
            raise ValueError(name)

    return Reading(name=name, dist=np.asarray(d, float), posterior=post,
                   budget=b.to_dict(), extra=extra)


def tournament(F, world, ep, upto: int, endpoint: str, names=CORE, *, rng, cfg=None,
               training=None, calibration=None) -> dict:
    out = {}
    for n in names:
        sub = np.random.default_rng(rng.integers(0, 2 ** 62))
        out[n] = read(n, F, world, ep, upto, endpoint, rng=sub, cfg=cfg, training=training,
                      calibration=calibration)
    return out


def score_tournament(readings: dict, truth, extra_row: dict | None = None) -> list:
    rows = []
    for name, r in readings.items():
        row = {"architecture": name, "promotable": name not in NON_PROMOTABLE}
        row.update(r.score(truth))
        row.update({f"budget_{k}": v for k, v in r.budget.items()})
        if extra_row:
            row.update(extra_row)
        rows.append(row)
    return rows


def particles_for_budget(target_evals: int, n_steps: int, per_step: int = 5) -> int:
    """How many particles spend roughly ``target_evals`` likelihood evaluations.

    Budget-matching is meaningless unless someone computes it, so this is the function cards call
    to size the particle filter against the exact reader rather than picking a round number.
    """
    denom = max(int(n_steps) * int(per_step), 1)
    return int(max(16, min(4000, target_evals // denom)))
