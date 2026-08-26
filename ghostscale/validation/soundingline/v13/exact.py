"""Exact inference for V13 (spec §4.1): a reader model over a hypothesis grid, channel-factorized
likelihoods, tempering and selection, and exact expected information gain.

A hypothesis is a tuple (family, group, profile, regime). The reader's own execution templates
supply the emission under each hypothesis (its expertise is its likelihood), so a route's gain is
the prior's gain and nothing else, as in V12. Channels are DISJOINT observations (world.py), so

    log P(artifact | h) = sum_c  lambda_c * log P(obs_c | h)

with lambda_c = 1 for every channel reproducing the plain posterior bit for bit (card I05).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import common as C
from .world import Family, Maker, N_METHODS, REGIMES, World, base_dist, poe, realization

_EPS = 1e-300
DEFAULT_CHANNELS = ("surface",)


@dataclass
class Hyp:
    index: int
    family: int
    group: int
    profile: str
    w: np.ndarray
    regime: str
    key: tuple


class Model:
    """Emissions under every hypothesis for one reader (its templates), all families requested."""

    def __init__(self, world: World, families: list | None = None, templates: dict | None = None,
                 method_prefs: dict | None = None, regimes: tuple = ("plain",), tier: str = "CREATOR",
                 groups: str | list = "all", habit: dict | None = None, robust_floor: float = 0.02):
        self.world = world
        self.robust_floor = float(robust_floor)      # the reader allows for execution noise it cannot model
        fids = list(range(world.n_families)) if families is None else list(families)
        self.families = fids
        self.tier = tier
        self.hyps = []
        self.by_family = {}
        for f in fids:
            fam = world.family(f)
            gids = list(range(len(fam.groups))) if groups == "all" else list(groups)
            for g in gids:
                for name, w in zip(fam.grid_names, fam.grid):
                    for r in regimes:
                        h = Hyp(len(self.hyps), f, g, name, w, r, (f, g, name, r))
                        self.hyps.append(h)
                        self.by_family.setdefault(f, []).append(h.index)
        self.K = len(self.hyps)
        self.templates = templates or {f: world.family(f).methods for f in fids}
        self.method_prefs = method_prefs or {f: world.family(f).method_pref for f in fids}
        self.habit = habit or {}
        self._cache = {}

    # -- emissions -------------------------------------------------------------------------- #
    def emission(self, h: Hyp, g: int, j: int | None, domain: int, canonical: bool = False) -> np.ndarray:
        key = (h.index, g, j, domain, canonical)
        if key in self._cache:
            return self._cache[key]
        fam = self.world.family(h.family)
        grp = fam.groups[h.group]
        tmpl, mp = self.templates[h.family], self.method_prefs[h.family]
        hab = self.habit.get((h.family, domain))
        if fam.link == "poe":
            sig = np.einsum("gj,gjf->gf", mp, tmpl)
            base = poe(sig, h.w) * grp.conv_mult
            base = base / base.sum()
        else:
            base = base_dist(fam, tmpl, mp, g, j, grp.conv_mult, grp.conv_add, hab)
        cs = self.world.params.cue_strength
        if h.regime == "bard":
            e = realization(fam, base, fam.cue_of[h.profile], cs)
        elif h.regime == "concealer":
            e = realization(fam, base, fam.cue_of[fam.decoy_of[h.profile]], cs)
        elif h.regime == "neutral":
            e = np.mean([realization(fam, base, s, cs) for s in range(max(1, fam.tail.size))], axis=0)
        else:
            e = base
        a = self.world.params.alpha[self.tier]
        d = a * e + (1 - a) * fam.synth
        d = (1 - self.robust_floor) * d / d.sum() + self.robust_floor / fam.nf
        out = d if canonical else fam.domains[domain].to_surface(d)
        self._cache[key] = out
        return out

    def goal_matrix(self, h: Hyp, domain: int, canonical: bool = False) -> np.ndarray:
        """(ng, nf) log-emission per goal, methods marginalised under the reader's preference."""
        fam = self.world.family(h.family)
        if fam.link == "poe":
            return np.log(np.maximum(self.emission(h, 0, None, domain, canonical), _EPS))[None, :]
        return np.stack([np.log(np.maximum(self.emission(h, g, None, domain, canonical), _EPS)) for g in range(fam.ng)])

    # -- channel log-likelihoods ------------------------------------------------------------- #
    def channel_loglik(self, art: dict, channel: str) -> np.ndarray:
        """log P(obs_channel | h) for every hypothesis. Unknown or absent channels are zero."""
        out = np.zeros(self.K)
        fa = int(art["family"])
        for h in self.hyps:
            if h.family != fa:
                # a hypothesis from another family cannot produce this family's artifact: the
                # penalty scales with the evidence so it can never be outrun by a long artifact
                if channel == "surface":
                    out[h.index] = -30.0 * max(1, len(art["features"]))
                elif channel == "common_structure":
                    out[h.index] = -30.0 * max(1, len(art.get("structure_obs", [1])))
                continue
            fam = self.world.family(fa)
            if channel == "surface":
                feats = np.asarray(art["features"])
                counts = np.bincount(feats, minlength=fam.nf).astype(float)
                LM = self.goal_matrix(h, art["domain"])
                per_goal = LM @ counts
                if fam.link == "poe":
                    out[h.index] = float(per_goal[0])
                else:
                    out[h.index] = float(C.logsumexp(per_goal + np.log(np.maximum(h.w, _EPS))))
            elif channel == "mechanics" and art.get("method") is not None:
                mp = self.method_prefs[fa]
                g = art["goal"] if art.get("goal", -1) >= 0 else None
                if g is None:
                    continue
                # mechanics carries goal-conditional method evidence; weight by the hypothesis's goal mass
                lik = float((h.w * mp[:, int(art["method"])]).sum())
                out[h.index] = float(np.log(max(lik, 1e-12)))
            elif channel == "goal_consequences" and "payoff_obs" in art:
                pn = self.world.params.payoff_noise
                po = int(art["payoff_obs"])
                lik = h.w[po] * (1 - pn) + (1 - h.w[po]) * pn / max(fam.ng - 1, 1)
                out[h.index] = float(np.log(max(lik, 1e-12)))
            elif channel == "common_structure" and "structure_obs" in art:
                LM = self.goal_matrix(h, art["domain"], canonical=True)
                E = np.exp(LM)
                blocks = fam.blocks + [list(fam.tail)]
                B = np.stack([E[:, b].sum(axis=1) for b in blocks], axis=1)      # (ng, n_blocks)
                B = B / B.sum(axis=1, keepdims=True)
                obs = np.asarray(art["structure_obs"])
                cnt = np.bincount(obs, minlength=B.shape[1]).astype(float)
                per_goal = np.log(np.maximum(B, _EPS)) @ cnt
                out[h.index] = float(C.logsumexp(per_goal + np.log(np.maximum(h.w, _EPS)))) if fam.link == "draw" else float(per_goal[0])
            elif channel == "group_convention" and "convention_obs" in art:
                grp = fam.groups[h.group]
                conv = grp.conv_add if fam.structure == "additive" else np.maximum(grp.conv_mult - 1.0 + 1e-6, 1e-6)
                conv = conv / conv.sum()
                obs = np.asarray(art["convention_obs"])
                out[h.index] = float(np.log(np.maximum(conv[obs], _EPS)).sum())
            elif channel == "communicative_shaping":
                # the observed cue slot under the hypothesis's regime
                s = int(art["slot"])
                if h.regime == "bard":
                    lik = 0.9 if s == fam.cue_of[h.profile] else 0.1 / max(fam.tail.size - 1, 1)
                elif h.regime == "concealer":
                    lik = 0.9 if s == fam.cue_of[fam.decoy_of[h.profile]] else 0.1 / max(fam.tail.size - 1, 1)
                else:
                    lik = 1.0 / max(fam.tail.size, 1)
                out[h.index] = float(np.log(lik))
            elif channel == "process_records":
                g = int(art["log"]["goal"])
                if g >= 0:
                    out[h.index] = float(np.log(max(h.w[g], 1e-12)))
            elif channel == "anomaly":
                occ = art.get("anomaly", {}).get("occurred", False)
                # the anomaly channel is regime- and expertise-diagnostic, not profile-diagnostic
                handling = art.get("anomaly", {}).get("handling", "none")
                if occ and h.regime in ("bard", "concealer", "neutral"):
                    lik = {"bard": {"repaired": 0.8, "retained": 0.15, "concealed": 0.05},
                           "concealer": {"repaired": 0.15, "retained": 0.15, "concealed": 0.7},
                           "neutral": {"repaired": 0.45, "retained": 0.45, "concealed": 0.1}}[h.regime].get(handling, 0.1)
                    out[h.index] = float(np.log(lik))
        return out

    def loglik(self, arts: list, channels=DEFAULT_CHANNELS, weights: dict | None = None) -> np.ndarray:
        """(n_art, K) tempered channel log-likelihoods."""
        L = np.zeros((len(arts), self.K))
        for i, a in enumerate(arts):
            for c in channels:
                lam = 1.0 if weights is None else float(weights.get(c, 1.0))
                if lam == 0.0:
                    continue
                L[i] += lam * self.channel_loglik(a, c)
        return L

    def posterior(self, prior: np.ndarray, arts: list, channels=DEFAULT_CHANNELS,
                  weights: dict | None = None, upto: int | None = None) -> np.ndarray:
        L = self.loglik(arts if upto is None else arts[:upto], channels, weights)
        v = np.log(np.maximum(np.asarray(prior, float), _EPS)) + L.sum(axis=0)
        return C.softmax(v)

    def cumulative(self, prior: np.ndarray, arts: list, channels=DEFAULT_CHANNELS,
                   weights: dict | None = None) -> np.ndarray:
        """(n_art + 1, K) posteriors after 0..n artifacts."""
        L = self.loglik(arts, channels, weights)
        lp = np.log(np.maximum(np.asarray(prior, float), _EPS))
        cum = np.concatenate([np.zeros((1, self.K)), np.cumsum(L, axis=0)], axis=0)
        return C.softmax(lp[None, :] + cum, axis=1)

    # -- marginals and predictions ------------------------------------------------------------ #
    def marginal(self, post: np.ndarray, key: str) -> dict:
        out = {}
        for h in self.hyps:
            k = getattr(h, key)
            out[k] = out.get(k, 0.0) + float(post[h.index])
        return out

    def truth_index(self, m: Maker, regime: str = "plain") -> int:
        for h in self.hyps:
            if h.family == m.family and h.group == m.group and h.profile == m.label and h.regime == regime:
                return h.index
        for h in self.hyps:                                   # regime-free grid
            if h.family == m.family and h.group == m.group and h.profile == m.label:
                return h.index
        raise KeyError((m.family, m.group, m.label, regime))

    def profile_mean(self, post: np.ndarray, family: int) -> np.ndarray:
        fam = self.world.family(family)
        w = np.zeros(fam.ng)
        tot = 0.0
        for h in self.hyps:
            if h.family == family:
                w += post[h.index] * h.w
                tot += post[h.index]
        return C.normalize(w) if tot > 0 else np.full(fam.ng, 1.0 / fam.ng)

    def next_goal(self, post: np.ndarray, family: int) -> np.ndarray:
        return self.profile_mean(post, family)

    def predictive_surface(self, post: np.ndarray, family: int, domain: int) -> np.ndarray:
        """Posterior-predictive feature distribution for the next artifact (continuation score)."""
        fam = self.world.family(family)
        out = np.zeros(fam.nf)
        for h in self.hyps:
            if h.family != family or post[h.index] <= 0:
                continue
            E = np.exp(self.goal_matrix(h, domain))
            mix = h.w @ E if fam.link == "draw" else E[0]
            out += post[h.index] * mix
        return C.normalize(out) if out.sum() > 0 else np.full(fam.nf, 1.0 / fam.nf)

    # -- expected information gain ------------------------------------------------------------ #
    def eig(self, prior: np.ndarray, probe_fn, rng, draws: int = 100, groups: np.ndarray | None = None,
            channels=DEFAULT_CHANNELS) -> float:
        """Monte-Carlo EIG (nats) about the hypothesis index (or about ``groups`` of hypotheses)
        of one artifact produced by ``probe_fn(h_true, rng) -> artifact``."""
        p = np.asarray(prior, float)
        H0 = C.entropy(p if groups is None else np.bincount(groups, weights=p, minlength=int(groups.max()) + 1))
        posts = []
        for _ in range(int(draws)):
            hi = int(rng.choice(self.K, p=p))
            art = probe_fn(self.hyps[hi], rng)
            q = self.posterior(p, [art], channels)
            if groups is not None:
                q = np.bincount(groups, weights=q, minlength=int(groups.max()) + 1)
            posts.append(C.entropy(q))
        return float(H0 - np.mean(posts))


def reader_model(world: World, reader: Maker, families: list | None = None, regimes: tuple = ("plain",),
                 tier: str | None = None, own_habit: bool = False) -> Model:
    """A Model whose likelihood is the READER's: its own corrupted templates and method
    preferences in its own family (its expertise), family defaults elsewhere (it has no
    execution mapping there), and its own habit tilt when asked."""
    fids = list(range(world.n_families)) if families is None else list(families)
    templates = {f: (reader.template if f == reader.family else world.family(f).methods) for f in fids}
    prefs = {f: (reader.method_pref if f == reader.family else world.family(f).method_pref) for f in fids}
    habit = {}
    if own_habit:
        for d in reader.habit:
            habit[(reader.family, d)] = reader.habit[d]
    return Model(world, fids, templates, prefs, regimes, tier or reader.tier, habit=habit)


def uniform_prior(model: Model, families: list | None = None) -> np.ndarray:
    v = np.ones(model.K)
    if families is not None:
        v[:] = 0.0
        for f in families:
            for i in model.by_family.get(f, []):
                v[i] = 1.0
    return C.normalize(v)


def score_rows(model: Model, post: np.ndarray, m: Maker, regime: str = "plain") -> dict:
    """Proper scores of a posterior against the maker's planted label, group and profile mean."""
    ti = model.truth_index(m, regime)
    prof = model.marginal(post, "profile")
    grp = model.marginal(post, "group")
    return {"ls": C.log_score(post, ti), "ls_profile": float(np.log(max(prof.get(m.label, 0.0), 1e-12))),
            "ls_group": float(np.log(max(grp.get(m.group, 0.0), 1e-12))),
            "brier": C.brier(post, ti), "top1": float(int(np.argmax(post)) == ti),
            "conf": float(post.max()),
            "l1": float(np.abs(model.profile_mean(post, m.family) - m.w).sum())}

def prefix_continuation(model: Model, post: np.ndarray, art: dict, k_seen: int) -> float:
    """Mean log score of an artifact's remaining features given its first ``k_seen``: the goal is
    inferred within the artifact from the prefix under the posterior's maker model, then the
    continuation is predicted from that goal posterior (the V12 continuation ruler)."""
    fam = model.world.family(art["family"])
    feats = np.asarray(art["features"])
    seen, rest = feats[:k_seen], feats[k_seen:]
    if rest.size == 0:
        return 0.0
    pred = np.zeros(fam.nf)
    for h in model.hyps:
        if h.family != art["family"] or post[h.index] <= 0:
            continue
        LM = model.goal_matrix(h, art["domain"])
        E = np.exp(LM)
        if fam.link == "draw":
            lg = LM[:, seen].sum(axis=1) + np.log(np.maximum(h.w, _EPS))
            q = C.softmax(lg)
            pred += post[h.index] * (q @ E)
        else:
            pred += post[h.index] * E[0]
    pred = C.normalize(pred)
    return float(np.log(np.maximum(pred[rest], 1e-12)).mean())


def frequency_continuation(train: list, art: dict, k_seen: int, nf: int) -> float:
    counts = np.full(nf, 0.5)
    for a in train:
        counts += np.bincount(np.asarray(a["features"]), minlength=nf)
    p = counts / counts.sum()
    rest = np.asarray(art["features"])[k_seen:]
    return float(np.log(np.maximum(p[rest], 1e-12)).mean()) if rest.size else 0.0
