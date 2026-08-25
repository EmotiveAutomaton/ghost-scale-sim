"""Self and other models: measuring a reader's own production model, deriving priors from it,
information-matched generic controls, and the similarity rulers (spec §3.2-3.4, §7).

Self-first means a prior over maker hypotheses derived from a MEASURED self-model, never a
sentence. The central threat is trivial prior advantage, so every self-first prior travels with
an information-matched generic prior (same entropy, same parameter count, no coordinate
correspondence) and a permuted-self prior (every marginal preserved, correspondence destroyed).
"""
from __future__ import annotations

import numpy as np

from .exact import posterior, profile_loglik_cumulative
from .world import Maker, World, stream

_EPS = 1e-12


def js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, float) + _EPS
    q = np.asarray(q, float) + _EPS
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def kl(p: dict, q: dict) -> float:
    out = 0.0
    for k, pv in p.items():
        if pv > 0:
            out += pv * np.log(pv / max(q.get(k, 0.0), _EPS))
    return float(out)


def entropy_of(prior: dict) -> float:
    v = np.array([x for x in prior.values() if x > 0])
    return float(-(v * np.log(v)).sum())


# --------------------------------------------------------------------------- #
# S01: measure the self-model behaviourally.
# --------------------------------------------------------------------------- #
def continuation_logscore(world: World, template, habit, w_hat: np.ndarray, art: dict,
                          k_seen: int, tier: str = "CREATOR") -> float:
    """Mean log score of an artifact's remaining features given its first ``k_seen`` features,
    under a maker model (template, habit, profile): the goal is inferred within the artifact from
    the prefix, then the continuation is predicted from the goal posterior."""
    feats = np.asarray(art["features"])
    seen, rest = feats[:k_seen], feats[k_seen:]
    dom = world.domains[art["domain"]]
    a = world.alpha[tier]
    ems = []
    for g in range(world.ng):
        base = template[g] * (habit if habit is not None else 1.0)
        base = base / base.sum()
        d = a * base + (1 - a) * world.synth
        ems.append(dom.to_surface(d / d.sum()))
    ems = np.stack(ems)
    ll = np.log(np.maximum(ems[:, seen], _EPS)).sum(axis=1) + np.log(np.maximum(w_hat, _EPS))
    q = np.exp(ll - ll.max())
    q = q / q.sum()
    pred = q @ ems
    return float(np.log(np.maximum(pred[rest], _EPS)).mean())


def frequency_continuation_logscore(train: list, art: dict, k_seen: int, nf: int) -> float:
    """The frequency baseline: pooled feature frequencies from the training artifacts."""
    counts = np.full(nf, 0.5)
    for a in train:
        counts += np.bincount(np.asarray(a["features"]), minlength=nf)
    p = counts / counts.sum()
    rest = np.asarray(art["features"])[k_seen:]
    return float(np.log(np.maximum(p[rest], _EPS)).mean())


def measure_self(world: World, reader: Maker, rng, n_artifacts: int = 24, domain: int = 0,
                 n_holdout: int = 8, k_seen: int = 6) -> dict:
    """The reader produces artifacts as a maker; its self-profile is estimated from them and
    scored on held-out self-artifacts by CONTINUATION: predict the rest of a held-out artifact
    from its first k_seen features, against a pooled-frequency baseline and a population model."""
    arts = stream(world, reader, domain, rng, n_artifacts + n_holdout)
    train, hold = arts[:n_artifacts], arts[n_artifacts:]
    cum = profile_loglik_cumulative(world, reader.template, reader.habit[domain], train,
                                    reader.tier, "plain")
    post = posterior(cum, len(train))
    w_hat = np.zeros(world.ng)
    for name, p in post.items():
        w_hat += p * world.family[name]
    self_ls = float(np.mean([continuation_logscore(world, reader.template, reader.habit[domain],
                                                   w_hat, a, k_seen, reader.tier) for a in hold]))
    pop_ls = float(np.mean([continuation_logscore(world, world.sig, None,
                                                  np.full(world.ng, 1.0 / world.ng), a, k_seen,
                                                  reader.tier) for a in hold]))
    freq_ls = float(np.mean([frequency_continuation_logscore(train, a, k_seen, world.nf)
                             for a in hold]))
    return {"w_hat": w_hat, "posterior": post, "heldout_logscore_self_model": self_ls,
            "heldout_logscore_population": pop_ls, "heldout_logscore_frequency": freq_ls,
            "top_profile": max(post, key=post.get), "n_train": len(train), "n_holdout": len(hold),
            "k_seen": k_seen}


# --------------------------------------------------------------------------- #
# Priors over the hypothesis family.
# --------------------------------------------------------------------------- #
def self_first_prior(world: World, w_self_hat: np.ndarray, beta: float = 6.0) -> dict:
    d = np.array([js(w_self_hat, world.family[n]) for n in world.family_names])
    v = np.exp(-beta * d)
    v = v / v.sum()
    return dict(zip(world.family_names, v))


def population_prior(world: World, makers: list, pseudo: float = 0.5) -> dict:
    counts = {n: pseudo for n in world.family_names}
    for m in makers:
        counts[m.profile] = counts.get(m.profile, 0.0) + 1.0
    s = sum(counts.values())
    return {n: c / s for n, c in counts.items()}


def uniform_prior(world: World) -> dict:
    return {n: 1.0 / len(world.family_names) for n in world.family_names}


def oracle_prior(world: World, truth: str, mass: float = 0.98) -> dict:
    rest = (1.0 - mass) / (len(world.family_names) - 1)
    return {n: (mass if n == truth else rest) for n in world.family_names}


def permuted_self_prior(prior: dict, rng) -> dict:
    """Every marginal preserved, correspondence destroyed: the same probability vector assigned to
    a random relabelling of the family."""
    names = list(prior)
    vals = np.array([prior[n] for n in names])
    perm = rng.permutation(len(names))
    while np.all(perm == np.arange(len(names))):
        perm = rng.permutation(len(names))
    return dict(zip(names, vals[perm]))


def _entropy_matched(world: World, centre: np.ndarray, target_entropy: float) -> dict:
    """A prior centred on ``centre`` with the same entropy as a reference prior, found by
    bisection on the temperature."""
    d = np.array([js(centre, world.family[n]) for n in world.family_names])
    lo, hi = 0.0, 200.0
    for _ in range(60):
        beta = 0.5 * (lo + hi)
        v = np.exp(-beta * d)
        v = v / v.sum()
        h = float(-(v[v > 0] * np.log(v[v > 0])).sum())
        if h > target_entropy:
            lo = beta
        else:
            hi = beta
    v = np.exp(-0.5 * (lo + hi) * d)
    v = v / v.sum()
    return dict(zip(world.family_names, v))


def information_matched_generic(world: World, self_prior: dict, makers: list) -> dict:
    """Same entropy as the self prior, centred on the population mean profile: same information,
    no self-to-maker coordinate correspondence."""
    mean_w = np.mean([m.w for m in makers], axis=0)
    return _entropy_matched(world, mean_w, entropy_of(self_prior))


def random_local_prior(world: World, self_prior: dict, rng) -> dict:
    """Same entropy as the self prior, centred on a random family member."""
    centre = world.family[world.family_names[int(rng.integers(len(world.family_names)))]]
    return _entropy_matched(world, centre, entropy_of(self_prior))


def expected_kl_to_truth(prior: dict, makers: list) -> float:
    """Average KL from the truth's one-hot to the prior, over the sampled makers: the distance
    a prior has to travel, which information matching must equalise."""
    return float(np.mean([-np.log(max(prior.get(m.profile, 0.0), _EPS)) for m in makers]))


# --------------------------------------------------------------------------- #
# Similarity rulers (S02).
# --------------------------------------------------------------------------- #
def similarity_axes(world: World, reader: Maker, maker: Maker, domain: int = 0) -> dict:
    """Per-axis distances between a reader and a maker. Lower is more similar."""
    from .exact import reader_emission
    prof = js(reader.w, maker.w)
    obs = float(np.mean([js(reader.template[g], maker.template[g]) for g in range(world.ng)]))
    hab = js(reader.habit[domain] / reader.habit[domain].sum(),
             maker.habit[domain] / maker.habit[domain].sum())
    e_r = reader_emission(world, reader.template, reader.habit[domain], reader.w, domain,
                          reader.tier, reader.regime if reader.regime != "neutral" else "neutral",
                          reader.profile)
    e_m = reader_emission(world, maker.template, maker.habit[domain], maker.w, domain,
                          maker.tier, maker.regime, maker.profile)
    policy = js(e_r, e_m)
    return {"profile": prof, "observation": obs, "habit": hab, "policy": policy,
            "regime_match": float(reader.regime == maker.regime),
            "surface_same_domain": 1.0}
