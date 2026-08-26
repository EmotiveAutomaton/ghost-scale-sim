"""The V13 world: a nested similarity basin (spec §3.1).

A maker is a composition, not a flat profile:

    M = F(z_common, z_group, z_expertise, z_individual, z_state, c_episode)

* ``z_common``     a FAMILY: a factorization of goals into features (pair, chain, additive,
                   gated, sparse, mixture). Families differ in how goals reach features and in
                   which feature slots a maker may act on, so a reader cannot obtain the right
                   likelihood from a label; it needs the family's structure.
* ``z_group``      a convention shared by a subset of the family's makers (a multiplicative or
                   additive tilt on features) plus a group-typical profile. A claimed label
                   travels with it and can be true or false.
* ``z_expertise``  an ECOLOGY: template corruption (observation competence), method-choice noise
                   (transition competence), mistake rate, and menu knowledge.
* ``z_individual`` a continuous profile over goals drawn around the group's typical profile,
                   labelled by its nearest grid hypothesis, plus persistent habits per domain.
* ``z_state``      a current-goal pressure, an attention target the maker itself allocates
                   (sharpening goal, mechanics, or surface decisions), and a temporary length
                   pressure.
* ``c_episode``    domain (surface permutation), commission, regime cue, tools; opportunity and
                   cost records are attached by ``costs.py``.

Every artifact carries DISJOINT evidence channels, each an independent draw given the latents,
so a reader's joint likelihood factorizes over channels and attention can select or temper them
without redefining the evidence (spec §3.3): surface, common_structure, group_convention,
mechanics, goal_consequences, anomaly, communicative_shaping, process_records; opportunity_set,
paid_cost and source_history are attached by the cost and trust modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C

STRUCTURES = ("pair", "chain", "additive", "gated", "sparse", "mixture")
DISCOVERY_STRUCTURES = STRUCTURES[:4]
FRESH_STRUCTURES = STRUCTURES[4:]
REGIMES = ("neutral", "bard", "concealer")
CHANNELS = ("surface", "common_structure", "group_convention", "mechanics", "goal_consequences",
            "opportunity_set", "paid_cost", "anomaly", "communicative_shaping", "source_history",
            "process_records")
ATTENTION_TARGETS = ("goal", "mechanics", "surface")
N_METHODS = 2
_EPS = 1e-300


@dataclass
class WorldParams:
    n_families: int = 4
    fresh_families: int = 0            # E1 / transfer: families from FRESH_STRUCTURES
    n_groups: int = 4
    n_ecologies: int = 4
    n_domains: int = 2
    n_steps: int = 24
    sig_peak: float = 0.9
    sig_floor: float = 0.01
    cue_strength: float = 0.35
    habit_strength: float = 0.25
    convention_strength: float = 0.30
    group_conc: float = 12.0           # concentration of the group's distribution over profile labels
    profile_jitter: float = 0.15       # share of an individual's profile that is continuous jitter off its label
    synth_conc: float = 0.03
    synth_seed: int = 0
    payoff_noise: float = 0.15         # goal-consequence channel: P(observed payoff != goal)
    structure_draws: int = 6           # common-structure channel: coarse draws per artifact
    convention_draws: int = 6          # group-convention channel: marker draws per artifact
    alpha: dict = field(default_factory=lambda: {"CREATOR": 1.0, "POLISHED": 0.95, "CURATOR": 0.6, "GHOST": 0.05})
    rare_makers: bool = False          # E6: anti-similar and falsely-similar makers in the population
    structures: tuple = DISCOVERY_STRUCTURES


RANGES = {"sig_peak": (0.75, 0.95), "sig_floor": (0.002, 0.03), "cue_strength": (0.15, 0.5),
          "habit_strength": (0.0, 0.25), "convention_strength": (0.2, 0.45), "group_conc": (1.0, 6.0),
          "profile_jitter": (0.05, 0.25), "synth_conc": (0.02, 0.3), "payoff_noise": (0.05, 0.3), "n_steps": (12, 32)}


def random_params(rng, base: WorldParams | None = None) -> WorldParams:
    p = WorldParams() if base is None else WorldParams(**{**base.__dict__, "alpha": dict(base.alpha)})
    for key, (lo, hi) in RANGES.items():
        if key == "n_steps":
            p.n_steps = int(rng.integers(lo, hi + 1))
        else:
            setattr(p, key, float(rng.uniform(lo, hi)))
    p.alpha["CURATOR"] = float(rng.uniform(0.4, 0.8))
    p.synth_seed = int(rng.integers(1, 1_000_000))
    return p


def params_to_dict(p: WorldParams) -> dict:
    d = dict(p.__dict__)
    d["alpha"] = dict(p.alpha)
    d["structures"] = list(p.structures)
    return d


# --------------------------------------------------------------------------- #
# Families: six factorizations.
# --------------------------------------------------------------------------- #
@dataclass
class Family:
    id: int
    structure: str
    ng: int
    nf: int
    sig: np.ndarray                 # (ng, nf) canonical goal signatures (methods marginalised)
    methods: np.ndarray             # (ng, nm, nf) method-specific realizations
    method_pref: np.ndarray         # (ng, nm) the family's default method preference
    blocks: list                    # per goal, the feature indices that carry it (common structure)
    tail: np.ndarray                # feature indices no goal owns: the cue slots
    mask: np.ndarray | None         # gated: (ng, nf) boolean gates
    grid_names: list                # profile hypothesis names
    grid: np.ndarray                # (K_p, ng) profile hypotheses
    link: str                       # "draw": one goal per artifact; "poe": conjunction
    groups: list = field(default_factory=list)
    ecologies: list = field(default_factory=list)
    domains: list = field(default_factory=list)
    synth: np.ndarray | None = None
    cue_of: dict = field(default_factory=dict)      # profile name -> tail slot index
    decoy_of: dict = field(default_factory=dict)    # profile name -> decoy profile name


@dataclass
class Group:
    id: int
    family: int
    conv_mult: np.ndarray
    conv_add: np.ndarray
    mean_profile: np.ndarray
    claimed: str
    label_probs: np.ndarray | None = None      # the group's distribution over profile labels (its typical profiles)


@dataclass
class Ecology:
    id: int
    family: int
    k: float
    seq_noise: float
    mistake_rate: float
    menu_knowledge: float


@dataclass
class Domain:
    id: int
    perm: np.ndarray

    def to_surface(self, dist):
        out = np.empty_like(dist)
        out[self.perm] = dist
        return out

    def to_canonical_index(self, feats):
        inv = np.argsort(self.perm)
        return inv[np.asarray(feats)]


def _profile_grid(ng: int, rng) -> tuple:
    names, rows = ["uniform"], [np.full(ng, 1.0 / ng)]
    peak = 0.70
    off = (1 - peak) / (ng - 1)
    for k in range(ng):
        w = np.full(ng, off)
        w[k] = peak
        names.append(f"peaked_{k}")
        rows.append(w)
    pairs = [(0, 1), (2, 3)] if ng >= 4 else [(0, 1)]
    for a, b in pairs:
        w = np.full(ng, (1 - 0.8) / (ng - 2))
        w[a] = w[b] = 0.4
        names.append(f"bimodal_{a}{b}")
        rows.append(w)
    return names, np.stack(rows)


def _peaked(nf, idx, peak, floor):
    v = np.full(nf, floor)
    v[np.asarray(idx)] += peak / len(idx)
    return v / v.sum()


def build_family(fid: int, structure: str, rng, p: WorldParams) -> Family:
    ng = int(rng.integers(4, 6))
    mask = None
    if structure == "pair":
        nf = 2 * ng + int(rng.integers(2, 5))
        blocks = [[2 * g, 2 * g + 1] for g in range(ng)]
        methods = np.stack([np.stack([_peaked(nf, b, p.sig_peak, p.sig_floor) * (np.eye(nf)[b[0]] * 0.5 + 1)
                                      for _ in range(1)] * 0) for b in blocks]) if False else None
        methods = np.zeros((ng, N_METHODS, nf))
        for g, b in enumerate(blocks):
            for j in range(N_METHODS):
                v = np.full(nf, p.sig_floor)
                v[b[0]] += p.sig_peak * (0.65 if j == 0 else 0.35)
                v[b[1]] += p.sig_peak * (0.35 if j == 0 else 0.65)
                methods[g, j] = v / v.sum()
    elif structure == "chain":
        nf = 3 * ng + int(rng.integers(2, 4))
        blocks = [[3 * g, 3 * g + 1, 3 * g + 2] for g in range(ng)]
        methods = np.zeros((ng, N_METHODS, nf))
        for g, b in enumerate(blocks):
            methods[g, 0] = _peaked(nf, [b[0], b[1]], p.sig_peak, p.sig_floor)
            methods[g, 1] = _peaked(nf, [b[1], b[2]], p.sig_peak, p.sig_floor)
    elif structure == "additive":
        nf = 2 * ng + int(rng.integers(2, 5))
        blocks = [[2 * g, 2 * g + 1] for g in range(ng)]
        methods = np.zeros((ng, N_METHODS, nf))
        for g, b in enumerate(blocks):
            dense = rng.dirichlet(np.full(nf, 0.5))
            for j in range(N_METHODS):
                v = 0.5 * _peaked(nf, b, p.sig_peak, p.sig_floor) + 0.5 * dense
                v = v * (1.0 + 0.4 * (rng.random(nf) if j == 1 else 0.0))
                methods[g, j] = v / v.sum()
    elif structure == "gated":
        nf = 2 * ng + int(rng.integers(2, 5))
        blocks = [[2 * g, 2 * g + 1] for g in range(ng)]
        mask = np.zeros((ng, nf), dtype=bool)
        methods = np.zeros((ng, N_METHODS, nf))
        for g, b in enumerate(blocks):
            allowed = set(b) | set(rng.choice(nf, size=nf // 2, replace=False).tolist())
            mask[g, list(allowed)] = True
            dense = rng.dirichlet(np.ones(nf)) * mask[g]
            for j in range(N_METHODS):
                v = _peaked(nf, b, p.sig_peak, p.sig_floor) * mask[g] + 0.3 * dense * (1 + j)
                methods[g, j] = v / v.sum()
    elif structure == "sparse":
        nf = 2 * ng + int(rng.integers(3, 6))
        blocks = [sorted(rng.choice(nf - 2, size=2, replace=False).tolist()) for _ in range(ng)]
        methods = np.zeros((ng, N_METHODS, nf))
        for g, b in enumerate(blocks):
            for j in range(N_METHODS):
                v = np.full(nf, p.sig_floor)
                v[b[0]] += p.sig_peak * (0.7 if j == 0 else 0.3)
                v[b[1]] += p.sig_peak * (0.3 if j == 0 else 0.7)
                methods[g, j] = v / v.sum()
    elif structure == "mixture":
        nf = 2 * ng + int(rng.integers(2, 5))
        blocks = [[2 * g, 2 * g + 1] for g in range(ng)]
        methods = np.zeros((ng, N_METHODS, nf))
        for g, b in enumerate(blocks):
            for j in range(N_METHODS):
                methods[g, j] = _peaked(nf, b, p.sig_peak * (1.0 if j == 0 else 0.8), p.sig_floor)
    else:
        raise ValueError(structure)
    owned = sorted({f for b in blocks for f in b})
    tail = np.array([f for f in range(nf) if f not in owned]) if len(owned) < nf else np.array([nf - 1])
    method_pref = np.full((ng, N_METHODS), 1.0 / N_METHODS)
    method_pref[:, 0] = 0.7
    method_pref[:, 1] = 0.3
    sig = np.einsum("gj,gjf->gf", method_pref, methods)
    names, grid = _profile_grid(ng, rng)
    fam = Family(id=fid, structure=structure, ng=ng, nf=nf, sig=sig, methods=methods, method_pref=method_pref,
                 blocks=blocks, tail=tail, mask=mask, grid_names=names, grid=grid,
                 link="poe" if structure == "mixture" else "draw")
    raw = np.random.default_rng(int(p.synth_seed) + fid).dirichlet(np.full(nf, p.synth_conc))
    fam.synth = np.maximum(raw, 1e-3) / np.maximum(raw, 1e-3).sum()
    fam.cue_of = {n: i % max(1, tail.size) for i, n in enumerate(names)}
    fam.decoy_of = {names[i]: names[(i + 1) % len(names)] for i in range(len(names))}
    return fam


def build_groups(fam: Family, rng, p: WorldParams) -> list:
    out = []
    labels = ["A", "B", "C", "D", "E", "F"]
    for gi in range(p.n_groups):
        tilt = rng.normal(0.0, 1.0, fam.nf)
        tilt = tilt / (np.abs(tilt).max() + 1e-12)
        conv_mult = 1.0 + p.convention_strength * tilt
        conv_add = p.convention_strength * rng.dirichlet(np.full(fam.nf, 0.5))
        lp = rng.dirichlet(np.full(len(fam.grid_names), max(p.group_conc, 0.2)))
        mean = lp @ fam.grid
        out.append(Group(id=gi, family=fam.id, conv_mult=conv_mult, conv_add=conv_add,
                         mean_profile=mean / mean.sum(), claimed=labels[gi], label_probs=lp))
    return out


def build_ecologies(fam: Family, rng, p: WorldParams) -> list:
    ks = [0.0, 0.15, 0.35, 0.6]
    sq = [0.05, 0.15, 0.3, 0.5]
    mr = [0.02, 0.05, 0.10, 0.20]
    mk = [1.0, 0.8, 0.6, 0.4]
    order = rng.permutation(4)
    out = []
    for ei in range(p.n_ecologies):
        i = int(order[ei % 4])
        out.append(Ecology(id=ei, family=fam.id, k=ks[i], seq_noise=sq[i], mistake_rate=mr[i], menu_knowledge=mk[i]))
    return out


def build_domains(fam: Family, rng, p: WorldParams, extra: int = 0) -> list:
    out = [Domain(0, np.arange(fam.nf))]
    for d in range(1, p.n_domains + extra):
        perm = rng.permutation(fam.nf)
        while np.all(perm == np.arange(fam.nf)):
            perm = rng.permutation(fam.nf)
        out.append(Domain(d, perm))
    return out


@dataclass
class World:
    wid: int
    lane: str
    params: WorldParams
    families: list

    def family(self, fid: int) -> Family:
        return self.families[fid]

    @property
    def n_families(self):
        return len(self.families)


def make_world(wid: int, lane: str, params: WorldParams | None = None, rng=None,
               extra_domains: int = 0) -> World:
    rng = rng if rng is not None else np.random.default_rng(C.world_seed(lane, wid))
    p = params if params is not None else random_params(rng)
    fams = []
    structures = list(p.structures)[: p.n_families] + list(FRESH_STRUCTURES)[: p.fresh_families]
    for fid, st in enumerate(structures):
        fam = build_family(fid, st, rng, p)
        fam.groups = build_groups(fam, rng, p)
        fam.ecologies = build_ecologies(fam, rng, p)
        fam.domains = build_domains(fam, rng, p, extra_domains)
        fams.append(fam)
    return World(wid=wid, lane=lane, params=p, families=fams)


# --------------------------------------------------------------------------- #
# Makers.
# --------------------------------------------------------------------------- #
@dataclass
class Maker:
    id: str
    family: int
    group: int
    ecology: int
    w: np.ndarray
    label: str
    k: float
    template: np.ndarray            # (ng, nm, nf): the maker's own method realizations
    method_pref: np.ndarray         # (ng, nm)
    habit: dict                     # domain -> (nf,) multiplicative tilt
    tier: str = "CREATOR"
    regime: str = "neutral"
    claimed_group: int | None = None
    attention: str = "goal"
    pressure: float = 0.0
    habit_strength: float = 0.25
    seq_noise: float = 0.1
    mistake_rate: float = 0.05
    cost: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    @property
    def sig(self):
        return np.einsum("gj,gjf->gf", self.method_pref, self.template)


def corrupt(methods: np.ndarray, k: float, rng) -> np.ndarray:
    """Observation competence: a less competent maker executes less distinctly. Each method
    realization is blurred toward the family's average emission (what every goal looks like on
    average), with a small idiosyncratic component. Low competence is undifferentiated execution,
    not random junk."""
    if k <= 0:
        return methods.copy()
    out = np.empty_like(methods)
    avg = methods.mean(axis=(0, 1))
    avg = avg / avg.sum()
    for g in range(methods.shape[0]):
        for j in range(methods.shape[1]):
            noise = rng.dirichlet(np.ones(methods.shape[2]) * 2.0)
            out[g, j] = (1 - k) * methods[g, j] + k * (0.75 * avg + 0.25 * noise)
            out[g, j] /= out[g, j].sum()
    return out


def nearest_label(fam: Family, w: np.ndarray) -> str:
    d = [C.js(w, fam.grid[i]) for i in range(len(fam.grid_names))]
    return fam.grid_names[int(np.argmin(d))]


def make_maker(world: World, mid: str, rng, family: int = 0, group: int | None = None,
               ecology: int | None = None, w: np.ndarray | None = None, label: str | None = None,
               k: float | None = None, regime: str = "neutral", tier: str = "CREATOR",
               claimed_group: int | None = None, attention: str | None = None,
               pressure: float | None = None, habit_strength: float | None = None,
               seq_noise: float | None = None, typicality: float | None = None) -> Maker:
    fam = world.family(family)
    p = world.params
    gi = int(rng.integers(len(fam.groups))) if group is None else int(group)
    ei = int(rng.integers(len(fam.ecologies))) if ecology is None else int(ecology)
    grp, eco = fam.groups[gi], fam.ecologies[ei]
    if w is None:
        if label is None:
            lp = grp.label_probs if grp.label_probs is not None else np.full(len(fam.grid_names), 1.0 / len(fam.grid_names))
            if typicality is not None:
                lp = C.normalize(lp ** float(typicality))
            label = fam.grid_names[int(rng.choice(len(fam.grid_names), p=lp))]
        base = fam.grid[fam.grid_names.index(label)]
        jit = p.profile_jitter
        w = C.normalize((1.0 - jit) * base + jit * rng.dirichlet(np.ones(fam.ng)))
    w = C.normalize(np.asarray(w, float))
    kk = eco.k if k is None else float(k)
    template = corrupt(fam.methods, kk, rng)
    sn = eco.seq_noise if seq_noise is None else float(seq_noise)
    mp = (1 - sn) * fam.method_pref + sn * rng.dirichlet(np.ones(N_METHODS), size=fam.ng)
    mp = mp / mp.sum(axis=1, keepdims=True)
    hs = p.habit_strength if habit_strength is None else float(habit_strength)
    stable = rng.normal(0.0, 1.0, fam.nf)
    habit = {}
    for d in range(len(fam.domains)):
        local = rng.normal(0.0, 1.0, fam.nf)
        tilt = 0.5 * stable + 0.5 * local
        habit[d] = 1.0 + hs * (tilt / (np.abs(tilt).max() + 1e-12))
    att = ATTENTION_TARGETS[int(rng.integers(3))] if attention is None else attention
    fixed_slot = int(rng.integers(max(1, fam.tail.size)))
    mk = Maker(id=mid, family=family, group=gi, ecology=ei, w=w, label=nearest_label(fam, w), k=kk,
                 template=template, method_pref=mp, habit=habit, tier=tier, regime=regime,
                 claimed_group=gi if claimed_group is None else int(claimed_group), attention=att,
                 pressure=float(rng.uniform(0.0, 0.03)) if pressure is None else float(pressure),
                 habit_strength=hs, seq_noise=sn, mistake_rate=eco.mistake_rate)
    mk.cost["fixed_slot"] = fixed_slot
    return mk


def population(world: World, n: int, rng, family: int | None = None, prefix: str = "m", **kw) -> list:
    """``n`` makers spread over families (or within one), groups and ecologies stratified."""
    out = []
    nfam = world.n_families
    for i in range(int(n)):
        f = (i % nfam) if family is None else int(family)
        fam = world.family(f)
        j = i // (nfam if family is None else 1)
        g = j % len(fam.groups)
        e = (j // len(fam.groups)) % len(fam.ecologies)
        out.append(make_maker(world, f"{prefix}{i}", rng, family=f, group=g, ecology=e, **kw))
    if world.params.rare_makers and n >= 8:
        for i in range(max(1, n // 16)):
            m = out[i]
            fam = world.family(m.family)
            m.claimed_group = (m.group + 1) % len(fam.groups)         # falsely-similar: claims another group
            m2 = out[-1 - i]
            m2.w = C.normalize(1.0 - m2.w + 0.05)                     # anti-similar: inverted profile
            m2.label = nearest_label(world.family(m2.family), m2.w)
    return out


# --------------------------------------------------------------------------- #
# Emission.
# --------------------------------------------------------------------------- #
def base_dist(fam: Family, template: np.ndarray, method_pref: np.ndarray, g: int, j: int | None,
              conv_mult: np.ndarray | None, conv_add: np.ndarray | None, habit) -> np.ndarray:
    """The canonical-coordinate distribution of goal ``g`` (method ``j`` or marginal)."""
    if j is None:
        base = np.einsum("j,jf->f", method_pref[g], template[g])
    else:
        base = template[g, int(j)].copy()
    if fam.structure == "additive" and conv_add is not None:
        base = base + conv_add
    elif conv_mult is not None:
        base = base * conv_mult
    if habit is not None:
        base = base * habit
    if fam.mask is not None:
        base = base * fam.mask[g] + 1e-6
    return base / base.sum()


def realization(fam: Family, base: np.ndarray, slot: int | None, cue_strength: float) -> np.ndarray:
    """Goal-equivalent realization: part of the tail mass is moved onto one cue slot. Pair mass and
    entropy across slots are matched by construction (tested)."""
    if slot is None or fam.tail.size == 0:
        return base / base.sum()
    cue = fam.tail[int(slot) % fam.tail.size]
    out = base.copy()
    tail_mass = out[fam.tail].sum()
    moved = cue_strength * tail_mass
    out[fam.tail] *= (1.0 - cue_strength)
    out[cue] += moved
    return out / out.sum()


def poe(template_sig: np.ndarray, w: np.ndarray) -> np.ndarray:
    logp = (np.asarray(w, float)[:, None] * np.log(np.maximum(template_sig, _EPS))).sum(axis=0)
    v = np.exp(logp - logp.max())
    return v / v.sum()


def maker_emission(world: World, m: Maker, g: int, j: int | None, domain: int, slot: int | None,
                   canonical: bool = False) -> np.ndarray:
    fam = world.family(m.family)
    grp = fam.groups[m.group]
    if fam.link == "poe":
        sig = np.einsum("gj,gjf->gf", m.method_pref, m.template)
        base = poe(sig, m.w)
        base = base * grp.conv_mult * m.habit[domain]
        base = base / base.sum()
    else:
        base = base_dist(fam, m.template, m.method_pref, g, j, grp.conv_mult, grp.conv_add, m.habit[domain])
    dist = realization(fam, base, slot, world.params.cue_strength)
    a = world.params.alpha[m.tier]
    dist = a * dist + (1 - a) * fam.synth
    dist = dist / dist.sum()
    return dist if canonical else fam.domains[domain].to_surface(dist)


def cue_slot_for(fam: Family, m: Maker, rng, regime: str | None = None) -> int:
    regime = m.regime if regime is None else regime
    if regime == "bard":
        return fam.cue_of[m.label]
    if regime == "concealer":
        return fam.cue_of[fam.decoy_of[m.label]]
    if m.attention == "surface":
        return int(m.cost.get("fixed_slot", 0)) % max(1, fam.tail.size)   # a surface-attending maker keeps one cue slot
    return int(rng.integers(max(1, fam.tail.size)))


def draw_goal(world: World, m: Maker, rng) -> int:
    fam = world.family(m.family)
    w = m.w.copy()
    if m.pressure > 0:
        w[int(np.argmax(m.w))] += m.pressure
    w = w / w.sum()
    return int(rng.choice(fam.ng, p=w))


def draw_method(world: World, m: Maker, g: int, rng) -> int:
    mp = m.method_pref[g].copy()
    if m.attention == "mechanics":
        mp = mp ** 2.0
        mp = mp / mp.sum()
    return int(rng.choice(N_METHODS, p=mp))


def artifact(world: World, m: Maker, domain: int, rng, n_steps: int | None = None,
             commission: int | None = None, regime: str | None = None, with_channels: bool = True) -> dict:
    """One artifact with its disjoint evidence channels and its process log."""
    fam = world.family(m.family)
    p = world.params
    n = p.n_steps if n_steps is None else int(n_steps)
    if fam.link == "poe":
        g, j = -1, None
    else:
        g = draw_goal(world, m, rng) if commission is None else int(commission)
        j = draw_method(world, m, g, rng)
    slot = cue_slot_for(fam, m, rng, regime)
    if m.attention == "surface" and regime != "neutral" and m.regime != "neutral":
        pass                                            # a surface-attending maker keeps its cue as is
    dist = maker_emission(world, m, g, j, domain, slot)
    feats = rng.choice(fam.nf, size=n, p=dist)
    anomaly = {"occurred": False, "handling": "none", "origin": "none"}
    if fam.link == "draw" and rng.random() < m.mistake_rate:
        # a mistake: one block of features produced under the wrong method
        wrong = 1 - j
        d2 = maker_emission(world, m, g, wrong, domain, slot)
        k0 = int(rng.integers(0, max(1, n - n // 4 + 1)))
        feats[k0: k0 + n // 4] = rng.choice(fam.nf, size=min(n // 4, n - k0), p=d2)
        u_h = rng.random()
        if m.regime == "bard":                              # the generator samples exactly the regime semantics the reader's model states
            handling = "repaired" if u_h < 0.8 else ("retained" if u_h < 0.95 else "concealed")
        elif m.regime == "concealer":
            handling = "concealed" if u_h < 0.7 else ("repaired" if u_h < 0.85 else "retained")
        else:
            handling = "repaired" if u_h < 0.45 else ("retained" if u_h < 0.9 else "concealed")
        if handling == "repaired":
            feats[k0: k0 + n // 4] = rng.choice(fam.nf, size=min(n // 4, n - k0), p=dist)
        anomaly = {"occurred": True, "handling": handling, "origin": "unintended", "at": k0}
    art = {"features": feats, "goal": int(g), "method": (None if j is None else int(j)), "slot": int(slot),
           "domain": int(domain), "family": int(m.family), "maker": m.id, "commission": commission,
           "anomaly": anomaly, "n": int(n)}
    if with_channels and fam.link == "draw":
        # goal consequences: the payoff channel names the goal with noise; a maker attending to
        # its goal makes the consequences clearer
        pn = p.payoff_noise * (0.3 if m.attention == "goal" else 1.0)
        obs = g if rng.random() >= pn else int(rng.choice([x for x in range(fam.ng) if x != g]))
        art["payoff_obs"] = int(obs)
        # common structure: coarse block-membership draws from the same emission (canonical coordinates)
        can = maker_emission(world, m, g, j, domain, slot, canonical=True)
        block_p = np.array([can[b].sum() for b in fam.blocks] + [can[fam.tail].sum()])
        block_p = block_p / block_p.sum()
        art["structure_obs"] = rng.choice(len(block_p), size=int(p.structure_draws), p=block_p).tolist()
        # group convention: marker draws from the convention itself
        conv = fam.groups[m.group].conv_add if fam.structure == "additive" else np.maximum(fam.groups[m.group].conv_mult - 1.0 + 1e-6, 1e-6)
        conv = conv / conv.sum()
        art["convention_obs"] = rng.choice(fam.nf, size=int(p.convention_draws), p=conv).tolist()
    art["log"] = {"goal": int(g), "method": art["method"], "slot": int(slot), "anomaly": anomaly,
                  "attention": m.attention, "group": int(m.group), "ecology": int(m.ecology)}
    return art


def stream(world: World, m: Maker, domain: int, rng, n_artifacts: int, n_steps: int | None = None,
           **kw) -> list:
    return [artifact(world, m, domain, rng, n_steps, **kw) for _ in range(int(n_artifacts))]


# --------------------------------------------------------------------------- #
# Surface statistics and similarity rulers (spec §3.1: cross actual, claimed and cheap surface).
# --------------------------------------------------------------------------- #
def histogram(feats, nf: int) -> np.ndarray:
    h = np.bincount(np.asarray(feats), minlength=nf).astype(float) + 1e-9
    return h / h.sum()


def surface_stats(arts: list, nf: int) -> dict:
    feats = np.concatenate([a["features"] for a in arts])
    h = histogram(feats, nf)
    return {"entropy": C.entropy(h), "length": float(np.mean([len(a["features"]) for a in arts])),
            "histogram": h}


def similarity(world: World, reader: Maker, maker: Maker, domain: int = 0, arts_r: list | None = None,
               arts_m: list | None = None) -> dict:
    """Per-level distances (lower = more similar). ``claimed`` and ``cheap_surface`` are the two
    imposters: a label and a raw histogram."""
    fr, fm = world.family(reader.family), world.family(maker.family)
    same_family = reader.family == maker.family
    if same_family:
        common = 0.0
        group = C.js(np.maximum(fr.groups[reader.group].conv_mult, 1e-6), np.maximum(fm.groups[maker.group].conv_mult, 1e-6))
        individual = C.js(reader.w, maker.w)
        state = abs(reader.pressure - maker.pressure) + float(reader.attention != maker.attention)
    else:
        common = 1.0 + float(fr.structure != fm.structure)
        group, individual, state = 1.0, 1.0, 1.0
    expertise = abs(reader.k - maker.k) + abs(reader.seq_noise - maker.seq_noise)
    out = {"common": common, "group": group, "expertise": expertise, "individual": individual, "state": state,
           "claimed": float(reader.claimed_group != maker.claimed_group) if same_family else 1.0,
           "surface_domain": 0.0}
    if arts_r is not None and arts_m is not None:
        nf = max(fr.nf, fm.nf)
        out["cheap_surface"] = C.js(histogram(np.concatenate([a["features"] for a in arts_r]), nf),
                                    histogram(np.concatenate([a["features"] for a in arts_m]), nf))
    return out


def relabel_family(fam: Family, perm_goals: np.ndarray, perm_feats: np.ndarray) -> Family:
    """A coordinate relabelling of a family (metamorphic relation I14 / X18): the same world under
    permuted goal and feature indices."""
    import copy
    f2 = copy.deepcopy(fam)
    inv_f = np.argsort(perm_feats)
    f2.methods = fam.methods[perm_goals][:, :, perm_feats]
    f2.method_pref = fam.method_pref[perm_goals]
    f2.sig = np.einsum("gj,gjf->gf", f2.method_pref, f2.methods)
    f2.blocks = [[int(inv_f[f]) for f in fam.blocks[g]] for g in perm_goals]
    f2.tail = np.array(sorted(int(inv_f[f]) for f in fam.tail))
    if fam.mask is not None:
        f2.mask = fam.mask[perm_goals][:, perm_feats]
    f2.grid = fam.grid[:, perm_goals]
    f2.synth = fam.synth[perm_feats]
    for g in f2.groups:
        g.conv_mult = g.conv_mult[perm_feats]
        g.conv_add = g.conv_add[perm_feats]
        g.mean_profile = g.mean_profile[perm_goals]
    return f2
