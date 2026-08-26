"""Prior routes and joint matching (spec §3.2).

Twelve routes over the hypothesis grid of a reader's ``Model``:

 1 measured self prior            softmax(-beta * JS(w_self_hat, w_h)) around the reader's own group
 2 equally local non-self prior   the same construction around ANOTHER reader's measured self-model,
                                  entropy-matched by temperature and distance-matched by choosing the
                                  other reader whose expected divergence to truth is closest
 3 within-common population       empirical grid frequencies among the family's makers
 4 within-group prior             the same within the reader's group
 5 within-expertise prior         the same within the reader's ecology
 6 broad all-family population    uniform over families, population within each
 7 entropy-and-distance-matched   a generic local prior: centre optimised so that entropy AND
   generic local prior            expected divergence to truth over the world family match self
 8 random local prior             entropy-matched, centred on a random grid profile
 9 permuted-self prior            the self vector under a random relabelling of the grid
10 anti-similar self prior        the self construction around the decoy profile
11 target-specific learned prior  the posterior after the target's own earlier history
12 oracle ceiling                 mass 0.98 on the truth

Matching (routes 1, 2, 7, 8, 9): prior entropy, expected divergence to truth over the SAMPLED
WORLD FAMILY (the population, never the scored targets), free parameters (one centre, one
temperature), coordinate access (the grid), compute and evidence (shared likelihood). Where
simultaneous matching is impossible the optimiser reports the residual imbalance and a
sensitivity bound (the gain a reweighting to exact match would add or remove).
"""
from __future__ import annotations

import numpy as np

from . import common as C
from .exact import Model

ROUTES = ("self", "equal_local", "within_common", "within_group", "within_expertise", "all_family",
          "generic_local", "random_local", "permuted_self", "anti_similar", "target_learned", "oracle")
BETA_SELF = 6.0
_EPS = 1e-12


def _dist_vector(model: Model, family: int, centre_w: np.ndarray, centre_group: int | None,
                 group_pull: float = 0.5) -> np.ndarray:
    """Distance from a centre to every hypothesis: JS on the profile plus a group term."""
    d = np.full(model.K, 50.0)
    for h in model.hyps:
        if h.family != family:
            continue
        dd = C.js(centre_w, h.w)
        if centre_group is not None:
            dd += group_pull * float(h.group != centre_group)
        d[h.index] = dd
    return d


def local_prior(model: Model, family: int, centre_w: np.ndarray, centre_group: int | None,
                beta: float = BETA_SELF) -> np.ndarray:
    d = _dist_vector(model, family, centre_w, centre_group)
    return C.softmax(-beta * d)


def entropy_matched(model: Model, family: int, centre_w, centre_group, target_entropy: float) -> tuple:
    """Bisection on the temperature so the local prior has the target entropy."""
    d = _dist_vector(model, family, centre_w, centre_group)
    lo, hi = 0.0, 400.0
    for _ in range(60):
        beta = 0.5 * (lo + hi)
        v = C.softmax(-beta * d)
        if C.entropy(v) > target_entropy:
            lo = beta
        else:
            hi = beta
    beta = 0.5 * (lo + hi)
    return C.softmax(-beta * d), beta


def expected_divergence(model: Model, prior: np.ndarray, makers: list, regime: str = "plain") -> float:
    """E over the sampled population of KL(one-hot truth || prior) = mean -log prior(truth)."""
    vals = []
    for m in makers:
        try:
            ti = model.truth_index(m, regime)
        except KeyError:
            continue
        vals.append(-np.log(max(float(prior[ti]), _EPS)))
    return float(np.mean(vals)) if vals else float("nan")


def population_prior(model: Model, makers: list, family: int | None = None, group: int | None = None,
                     ecology: int | None = None, exclude: str | None = None, pseudo: float = 0.5,
                     regime: str = "plain") -> np.ndarray:
    v = np.full(model.K, 0.0)
    fams = set()
    for m in makers:
        if exclude is not None and m.id == exclude:
            continue
        if family is not None and m.family != family:
            continue
        if group is not None and m.group != group:
            continue
        if ecology is not None and m.ecology != ecology:
            continue
        try:
            v[model.truth_index(m, regime)] += 1.0
        except KeyError:
            continue
        fams.add(m.family)
    for h in model.hyps:
        if (family is None and (fams and h.family in fams)) or (family is not None and h.family == family):
            if group is None or h.group == group:
                v[h.index] += pseudo
    if v.sum() <= 0:
        v[:] = 1.0
    return C.normalize(v)


def all_family_prior(model: Model, makers: list, regime: str = "plain") -> np.ndarray:
    v = np.zeros(model.K)
    for f in model.families:
        pf = population_prior(model, makers, family=f, regime=regime)
        v += pf / len(model.families)
    return C.normalize(v)


def permuted(prior: np.ndarray, model: Model, family: int, rng) -> np.ndarray:
    idx = np.array(model.by_family[family])
    vals = prior[idx].copy()
    perm = rng.permutation(idx.size)
    while np.all(perm == np.arange(idx.size)):
        perm = rng.permutation(idx.size)
    out = prior.copy()
    out[idx] = vals[perm]
    return C.normalize(out)


def oracle(model: Model, m, mass: float = 0.98, regime: str = "plain") -> np.ndarray:
    ti = model.truth_index(m, regime)
    v = np.full(model.K, (1 - mass) / max(model.K - 1, 1))
    v[ti] = mass
    return v


def generic_local_matched(model: Model, family: int, self_prior: np.ndarray, makers: list,
                          rng, n_candidates: int = 24, regime: str = "plain") -> tuple:
    """Route 7: search centres (grid profiles, group members' means, and random simplex points)
    with entropy matched by temperature, and choose the one whose expected divergence to truth
    over the population is closest to the self prior's. Returns (prior, report)."""
    fam = model.world.family(family)
    target_h = C.entropy(self_prior)
    target_d = expected_divergence(model, self_prior, makers, regime)
    cands = [(w, g) for w in fam.grid for g in range(len(fam.groups))]
    for _ in range(n_candidates):
        cands.append((rng.dirichlet(np.ones(fam.ng)), int(rng.integers(len(fam.groups)))))
    best, best_gap, best_rep = None, float("inf"), None
    for w, g in cands:
        pr, beta = entropy_matched(model, family, w, g, target_h)
        d = expected_divergence(model, pr, makers, regime)
        gap = abs(d - target_d)
        if gap < best_gap:
            best, best_gap = pr, gap
            best_rep = {"centre_group": int(g), "beta": float(beta), "expected_divergence": d}
    rep = {"target_entropy": target_h, "matched_entropy": C.entropy(best), "target_expected_divergence": target_d,
           **best_rep, "residual_divergence_gap": float(best_gap), "free_parameters": 2,
           "coordinate_access": "grid", "n_candidates": len(cands)}
    return best, rep


def sensitivity_bound(gain_by_divergence: list) -> dict:
    """Given (divergence, gain) rows for a control route across candidate centres, the slope of
    gain on divergence bounds how much of a self-minus-control gain a residual distance gap could
    manufacture: |slope| * residual."""
    if len(gain_by_divergence) < 3:
        return {"slope": float("nan")}
    x = np.array([d for d, _ in gain_by_divergence])
    y = np.array([g for _, g in gain_by_divergence])
    if x.std() < 1e-9:
        return {"slope": 0.0}
    return {"slope": float(np.cov(x, y, bias=True)[0, 1] / x.var())}


def routes_for(model: Model, reader, self_hat: dict, makers: list, others: list, rng, target,
               regime: str = "plain", target_history: list | None = None,
               channels=("surface",)) -> tuple:
    """Every route for one reader against one target. ``others`` are other readers' measured
    self-models [(reader, self_hat)] for the equally-local non-self prior. Returns (priors, report)."""
    fam_id = reader.family
    fam = model.world.family(fam_id)
    sp = local_prior(model, fam_id, self_hat["w_hat"], self_hat.get("group_hat", reader.group), BETA_SELF)
    H = C.entropy(sp)
    D = expected_divergence(model, sp, makers, regime)
    # route 2: the other reader whose entropy-matched local prior has the closest expected divergence
    eq, eq_rep = None, None
    cands = []
    for r2, sh2 in others:
        if r2.id == reader.id or r2.family != fam_id:
            continue
        pr, beta = entropy_matched(model, fam_id, sh2["w_hat"], sh2.get("group_hat", r2.group), H)
        d = expected_divergence(model, pr, makers, regime)
        cands.append((abs(d - D), pr, {"other_reader": r2.id, "beta": float(beta), "expected_divergence": d}))
    if cands:
        cands.sort(key=lambda t: t[0])
        eq, eq_rep = cands[0][1], {**cands[0][2], "residual_divergence_gap": float(cands[0][0])}
    gl, gl_rep = generic_local_matched(model, fam_id, sp, makers, rng, regime=regime)
    rl, rl_beta = entropy_matched(model, fam_id, fam.grid[int(rng.integers(len(fam.grid)))],
                                  int(rng.integers(len(fam.groups))), H)
    anti_w = fam.grid[fam.grid_names.index(fam.decoy_of[self_hat.get("label_hat", reader.label)])]
    anti, _ = entropy_matched(model, fam_id, anti_w, (reader.group + 1) % len(fam.groups), H)
    priors = {"self": sp, "equal_local": eq if eq is not None else gl,
              "within_common": population_prior(model, makers, family=fam_id, exclude=target.id, regime=regime),
              "within_group": population_prior(model, makers, family=fam_id, group=reader.group, exclude=target.id, regime=regime),
              "within_expertise": population_prior(model, makers, family=fam_id, ecology=reader.ecology, exclude=target.id, regime=regime),
              "all_family": all_family_prior(model, makers, regime=regime),
              "generic_local": gl, "random_local": rl, "permuted_self": permuted(sp, model, fam_id, rng),
              "anti_similar": anti, "oracle": oracle(model, target, regime=regime)}
    if target_history:
        priors["target_learned"] = model.posterior(priors["within_common"], target_history, channels)
    rep = {"self_entropy": H, "self_expected_divergence": D, "equal_local": eq_rep, "generic_local": gl_rep,
           "random_local_beta": float(rl_beta),
           "entropy_by_route": {k: C.entropy(v) for k, v in priors.items()},
           "expected_divergence_by_route": {k: expected_divergence(model, v, makers, regime) for k, v in priors.items()}}
    return priors, rep


def measure_self(world, reader, model_for_reader: Model, rng, n_train: int = 24, n_holdout: int = 8,
                 domain: int = 0, k_seen: int = 6, channels=("surface",)) -> dict:
    """The reader produces artifacts as a maker; its profile and group are estimated from them by
    exact inference under its own templates, and scored on held-out self-artifacts by within-
    artifact continuation against frequency and family-population baselines (spec C01)."""
    from .world import stream
    from .exact import reader_model
    if model_for_reader is None:
        model_for_reader = reader_model(world, reader, families=[reader.family])
    arts = stream(world, reader, domain, rng, n_train + n_holdout)
    train, hold = arts[:n_train], arts[n_train:]
    fam = world.family(reader.family)
    prior = np.zeros(model_for_reader.K)
    for i in model_for_reader.by_family[reader.family]:
        prior[i] = 1.0
    prior = C.normalize(prior)
    post = model_for_reader.posterior(prior, train, channels)
    w_hat = model_for_reader.profile_mean(post, reader.family)
    grp = model_for_reader.marginal(post, "group")
    group_hat = max(grp, key=grp.get)
    prof = model_for_reader.marginal(post, "profile")
    label_hat = max(prof, key=prof.get)
    # continuation: infer the held-out artifact's goal from its prefix, predict the rest
    from .exact import prefix_continuation, frequency_continuation
    pop_prior = prior
    ls_self = [prefix_continuation(model_for_reader, post, a, k_seen) for a in hold]
    ls_pop = [prefix_continuation(model_for_reader, pop_prior, a, k_seen) for a in hold]
    ls_freq = [frequency_continuation(train, a, k_seen, fam.nf) for a in hold]
    # transitions: the reader's own method choice given the goal, against pooled method frequency
    own_mp = model_for_reader.method_prefs[reader.family]
    mfreq = np.full(2, 0.5)
    for a in train:
        if a.get("method") is not None:
            mfreq[int(a["method"])] += 1
    mfreq = mfreq / mfreq.sum()
    ls_trans_self, ls_trans_freq = [], []
    for a in hold:
        if a.get("method") is None or a["goal"] < 0:
            continue
        ls_trans_self.append(float(np.log(max(own_mp[a["goal"], a["method"]], 1e-12))))
        ls_trans_freq.append(float(np.log(max(mfreq[a["method"]], 1e-12))))
    return {"w_hat": w_hat, "group_hat": int(group_hat), "label_hat": label_hat, "posterior": post,
            "heldout_logscore_self_model": float(np.mean(ls_self)),
            "heldout_logscore_frequency": float(np.mean(ls_freq)),
            "heldout_logscore_population": float(np.mean(ls_pop)),
            "heldout_transition_self": float(np.mean(ls_trans_self)) if ls_trans_self else 0.0,
            "heldout_transition_frequency": float(np.mean(ls_trans_freq)) if ls_trans_freq else 0.0,
            "label_correct": float(label_hat == reader.label), "group_correct": float(group_hat == reader.group),
            "n_train": n_train, "n_holdout": n_holdout, "k_seen": k_seen}
