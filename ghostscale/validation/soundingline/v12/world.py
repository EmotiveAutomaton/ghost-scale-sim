"""The V12 world: makers with profile w, expertise k, habits h, communicative regime r, in two
surface domains, producing artifacts with opportunity records (spec section 3).

Default construction reproduces V11's objects exactly (card I01 asserts it); randomized
constructions (card I02, X07) vary the declared parameters within valid ranges.

THE REGIME CONSTRUCTION, because it is the one place a cheap feature could write the conclusion.
Every emission has a goal pair (task-relevant mass) and a tail. Among goal-equivalent realizations
the maker may choose WHICH tail feature carries a small extra cue. Each profile in the family owns
one cue slot. A bard emits its own profile's cue, a neutral maker a random one, a concealer the
cue of a decoy profile. Pair mass, entropy, length, and effort are identical across regimes by
construction: the regimes differ only in whether the cue corresponds to the truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....generative_model import build_observer_signature
from ....v11.maker import poe, profile_family

_EPS = 1e-300
REGIMES = ("bard", "neutral", "concealer")


@dataclass
class WorldParams:
    n_goals: int = 4
    n_features: int = 8
    sig_peak: float = 0.9
    sig_floor: float = 0.01
    synth_conc: float = 0.03
    synth_seed: int = 0
    synth_floor: float = 1e-3
    alpha: dict = field(default_factory=lambda: {"CREATOR": 1.0, "POLISHED": 0.95,
                                                 "CURATOR": 0.6, "GHOST": 0.05})
    peaked_mass: float = 0.70
    bimodal_mass: float = 0.40
    cue_strength: float = 0.35      # share of tail mass moved onto the regime cue feature
    habit_strength: float = 0.25    # multiplicative tilt of the habit vector
    n_steps: int = 24


def _pairs(ng: int, nf: int):
    return [[2 * g, 2 * g + 1] for g in range(ng)]


def build_signatures(p: WorldParams) -> np.ndarray:
    pairs = _pairs(p.n_goals, p.n_features)
    sig = np.zeros((p.n_goals, p.n_features))
    for g, (a, b) in enumerate(pairs):
        v = np.full(p.n_features, p.sig_floor)
        v[a] += p.sig_peak / 2.0
        v[b] += p.sig_peak / 2.0
        sig[g] = v / v.sum()
    return sig


def build_synth(p: WorldParams) -> np.ndarray:
    rng = np.random.default_rng(int(p.synth_seed))
    raw = rng.dirichlet(np.full(p.n_features, p.synth_conc))
    pairs = _pairs(p.n_goals, p.n_features)
    highs = np.mean([max(raw[a], raw[b]) for a, b in pairs])
    lows = np.mean([min(raw[a], raw[b]) for a, b in pairs])
    out = np.zeros_like(raw)
    for a, b in pairs:
        out[a], out[b] = highs, lows
    out = np.maximum(out, p.synth_floor)
    return out / out.sum()


def build_family(p: WorldParams) -> dict:
    ng = p.n_goals
    fam = {"uniform": np.full(ng, 1.0 / ng)}
    off = (1.0 - p.peaked_mass) / (ng - 1)
    for k in range(ng):
        w = np.full(ng, off)
        w[k] = p.peaked_mass
        fam[f"peaked_{k}"] = w
    bm = np.full(ng, (1.0 - 2 * p.bimodal_mass) / (ng - 2))
    bm[0] = bm[1] = p.bimodal_mass
    fam["bimodal"] = bm
    return fam


@dataclass
class Domain:
    name: str
    perm: np.ndarray           # surface relabelling of features

    def to_surface(self, dist: np.ndarray) -> np.ndarray:
        out = np.empty_like(dist)
        out[self.perm] = dist
        return out


@dataclass
class World:
    params: WorldParams
    sig: np.ndarray
    synth: np.ndarray
    family: dict
    family_names: list
    alpha: dict
    domains: list
    cue_of: dict               # profile name -> cue slot within the tail
    decoy_of: dict             # profile name -> decoy profile name (a derangement)

    @property
    def ng(self):
        return self.sig.shape[0]

    @property
    def nf(self):
        return self.sig.shape[1]


def make_world(cfg=None, params: WorldParams | None = None, rng=None) -> World:
    """Default (params None) reproduces V11's world from the repository config; otherwise the
    parametric construction. The identity between the two at default parameters is tested."""
    if params is None:
        from ....v11.maker import build_maker_world
        from .... import constants as K
        mw = build_maker_world(cfg)
        p = WorldParams()
        p.synth_seed = int(cfg.artifact_model.noise_free_synth_seed)
        p.synth_conc = float(cfg.artifact_model.noise_free_synth_concentration)
        sig, synth = mw.sig, mw.synth
        alpha = {name: float(mw.alpha[i]) for i, name in enumerate(K.PROVENANCE_NAMES)}
        fam = profile_family(sig.shape[0])
    else:
        p = params
        sig, synth, alpha, fam = build_signatures(p), build_synth(p), dict(p.alpha), build_family(p)
    names = list(fam)
    rng = rng if rng is not None else np.random.default_rng(0)
    nf = sig.shape[1]
    perm1 = rng.permutation(nf)
    while np.all(perm1 == np.arange(nf)):
        perm1 = rng.permutation(nf)
    domains = [Domain("native", np.arange(nf)), Domain("dialect", perm1)]
    cue_of = {n: i for i, n in enumerate(names)}
    decoy_of = {names[i]: names[(i + 1) % len(names)] for i in range(len(names))}
    return World(params=p, sig=sig, synth=synth, family=fam, family_names=names, alpha=alpha,
                 domains=domains, cue_of=cue_of, decoy_of=decoy_of)


# --------------------------------------------------------------------------- #
# Makers.
# --------------------------------------------------------------------------- #
@dataclass
class Maker:
    id: str
    profile: str
    w: np.ndarray
    k: float                     # expertise corruption of the maker's own templates (0 = expert)
    template: np.ndarray         # (goals, features), the maker's execution mapping
    habit: dict                  # domain index -> multiplicative tilt vector (features,)
    regime: str
    construction: str = "A"
    mask: np.ndarray | None = None
    tier: str = "CREATOR"


def make_maker(world: World, mid: str, profile: str, rng, k: float = 0.0, regime: str = "neutral",
               construction: str = "A", habit_strength: float | None = None,
               habit_share_stable: float = 0.5, mask=None, tier: str = "CREATOR") -> Maker:
    w = world.family[profile].copy()
    if mask is not None:
        w = w * np.asarray(mask, dtype=float)
        w = w / w.sum()
    template = build_observer_signature(world.sig, float(k), rng) if k > 0 else world.sig.copy()
    hs = world.params.habit_strength if habit_strength is None else float(habit_strength)
    stable = rng.normal(0.0, 1.0, world.nf)
    habit = {}
    for d in range(len(world.domains)):
        local = rng.normal(0.0, 1.0, world.nf)
        tilt = habit_share_stable * stable + (1.0 - habit_share_stable) * local
        habit[d] = 1.0 + hs * (tilt / (np.abs(tilt).max() + 1e-12))
    return Maker(id=mid, profile=profile, w=w, k=float(k), template=template, habit=habit,
                 regime=regime, construction=construction, mask=mask, tier=tier)


def tail_features(world: World, g: int) -> np.ndarray:
    pair = np.argsort(world.sig[g])[-2:]
    return np.setdiff1d(np.arange(world.nf), pair)


def realization(world: World, base: np.ndarray, g: int, cue_slot: int | None) -> np.ndarray:
    """A goal-equivalent realization: same pair mass, tail mass partly moved to one cue feature.
    cue_slot None returns the unshaped base."""
    if cue_slot is None:
        return base / base.sum()
    tail = tail_features(world, g)
    cue = tail[int(cue_slot) % tail.size]
    out = base.copy()
    tail_mass = out[tail].sum()
    moved = world.params.cue_strength * tail_mass
    out[tail] *= (1.0 - world.params.cue_strength)
    out[cue] += moved
    return out / out.sum()


def emission(world: World, maker: Maker, g: int, domain: int, rng,
             regime: str | None = None) -> tuple:
    """The emission distribution for goal g in a domain, plus the opportunity record."""
    regime = maker.regime if regime is None else regime
    base = maker.template[g] * maker.habit[domain]
    base = base / base.sum()
    n_slots = len(world.family_names)
    alternatives = [realization(world, base, g, s) for s in range(n_slots)]
    if regime == "bard":
        chosen = world.cue_of[maker.profile]
        mode = "deliberated"
    elif regime == "concealer":
        chosen = world.cue_of[world.decoy_of[maker.profile]]
        mode = "deliberated"
    else:
        chosen = int(rng.integers(n_slots))
        mode = "habitual"
    dist = alternatives[chosen]
    a = world.alpha[maker.tier]
    dist = a * dist + (1.0 - a) * world.synth
    dist = dist / dist.sum()
    pair = np.argsort(world.sig[g])[-2:]
    record = {"goal": int(g), "alternatives": n_slots, "chosen_slot": int(chosen),
              "mode": mode,
              "task_mass_by_alternative": [float(alt[pair].sum()) for alt in alternatives]}
    return world.domains[domain].to_surface(dist), record


def artifact(world: World, maker: Maker, domain: int, rng, n_steps: int | None = None,
             commission: int | None = None, a_amp: float = 4.0, lam: float = 0.5) -> dict:
    """One artifact: features (surface coordinates), the goal(s) that produced it, and its
    opportunity records. Construction A draws one goal; B emits the conjunction; a commission
    amplifies a channel multiplicatively (V11 S-14) with a compliance share."""
    from ....v11.maker import amplify
    n = world.params.n_steps if n_steps is None else int(n_steps)
    a = world.alpha[maker.tier]
    dom = world.domains[domain]
    if commission is not None:
        w_amp = amplify(maker.w, int(commission), a_amp)
        how = poe(maker.template, w_amp)
        base = lam * maker.template[int(commission)] + (1.0 - lam) * how
        base = base * maker.habit[domain]
        base = base / base.sum()
        dist = a * base + (1 - a) * world.synth
        dist = dom.to_surface(dist / dist.sum())
        feats = rng.choice(world.nf, size=n, p=dist)
        return {"features": feats, "goals": [int(commission)] * n, "records": [],
                "domain": domain, "commission": int(commission)}
    if maker.construction == "B":
        base = poe(maker.template, maker.w) * maker.habit[domain]
        base = base / base.sum()
        dist = a * base + (1 - a) * world.synth
        dist = dom.to_surface(dist / dist.sum())
        feats = rng.choice(world.nf, size=n, p=dist)
        return {"features": feats, "goals": [-1] * n, "records": [], "domain": domain,
                "commission": None}
    g = int(rng.choice(world.ng, p=maker.w))
    dist, rec = emission(world, maker, g, domain, rng)
    feats = rng.choice(world.nf, size=n, p=dist)
    return {"features": feats, "goals": [g] * n, "records": [rec], "domain": domain,
            "commission": None}


def stream(world: World, maker: Maker, domain: int, rng, n_artifacts: int,
           n_steps: int | None = None) -> list:
    return [artifact(world, maker, domain, rng, n_steps) for _ in range(int(n_artifacts))]


def population(world: World, n: int, rng, profiles=None, k_choices=(0.0,), regimes=("neutral",),
               construction: str = "A", prefix: str = "m", **kw) -> list:
    profiles = profiles or world.family_names
    makers = []
    for i in range(int(n)):
        prof = profiles[i % len(profiles)]
        k = float(k_choices[i % len(k_choices)])
        reg = regimes[i % len(regimes)]
        makers.append(make_maker(world, f"{prefix}{i}", prof, rng, k=k, regime=reg,
                                 construction=construction, **kw))
    return makers


# --------------------------------------------------------------------------- #
# Randomized constructions for severity (I02, X07).
# --------------------------------------------------------------------------- #
RANGES = {
    "sig_peak": (0.6, 0.95), "sig_floor": (0.002, 0.03), "synth_conc": (0.02, 0.3),
    "peaked_mass": (0.5, 0.9), "bimodal_mass": (0.3, 0.45), "cue_strength": (0.1, 0.6),
    "habit_strength": (0.0, 0.5), "curator_alpha": (0.4, 0.8), "n_steps": (8, 32),
}


def random_params(rng, n_goals: int = 4, n_features: int = 8) -> WorldParams:
    p = WorldParams(n_goals=n_goals, n_features=n_features)
    for key, (lo, hi) in RANGES.items():
        if key == "curator_alpha":
            p.alpha["CURATOR"] = float(rng.uniform(lo, hi))
        elif key == "n_steps":
            p.n_steps = int(rng.integers(lo, hi + 1))
        else:
            setattr(p, key, float(rng.uniform(lo, hi)))
    p.synth_seed = int(rng.integers(1, 1_000_000))
    return p


def params_to_dict(p: WorldParams) -> dict:
    d = dict(p.__dict__)
    d["alpha"] = dict(p.alpha)
    return d
