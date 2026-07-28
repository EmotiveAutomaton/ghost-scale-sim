"""V4 C1/C2 — hypothesis-space overlap, the foreign signature family, and EXPLORE.

THE REFRAME. V1-V3 modelled synthetic content as GOAL-EMPTY: ``noise_free_synth`` is
structured but goal-independent, so MI(features; goal) goes to zero and the observer
disengages because nothing is identifiable. V4 models it as GOAL-FOREIGN: produced by a real
policy over a real goal whose signature lies outside the observer's likelihood family. The
crash now has to come from MISSPECIFICATION rather than unidentifiability, which is a
mechanism that predicts confident wrong answers directly.

-----------------------------------------------------------------------------------------
WHY THE FEATURE SPACE DOUBLES, AND WHY THAT WAS NOT OPTIONAL (decision D1).

V4 spec C1 requires ``foreign_basis[k]`` to be drawn from "a feature partition disjoint from
the dominant support of ``sig_true``". At V1-V3 cardinality there is no such partition:
F = 8 and the four goal pairs [[0,1],[2,3],[4,5],[6,7]] tile all eight features. The only two
options were to overlap the observer's support, which is the spec's own pre-mortem failure #1
("foreign content becomes unidentifiable rather than foreign, and V4 reports V3's results in
new vocabulary"), or to widen the space.

So V4 runs at F = 16:

    features  0-7   HUMAN block, four goal pairs, the V1-V3 structure carried over
    features  8-15  FOREIGN block, four foreign pairs, disjoint by construction

The consequence is that V4 does NOT share a seed stream with V1-V3 and cannot reproduce them
bit-exactly. N16 is therefore a BEHAVIOURAL regression with a pre-registered tolerance rather
than the bit-exactness N8 gets. That is a real weakening and it is logged as V4 deviation 1.

-----------------------------------------------------------------------------------------
WHY sig_EXPLORE IS CONSTRUCTED HERE AND NOT WHERE IT LOOKS LIKE IT SHOULD BE (decision D2).

V4 spec C2 defines ``sig_EXPLORE = normalize(mean_g(sig_i[g]))`` and then says, in bold, that
this is NOT uniform over features, because "a uniform-over-features EXPLORE would absorb
anything, including noise, and would trivially destroy every result."

At V1-V3 cardinality those two objects are the same thing. Measured before any V4 code was
written, on the shipped V3 config:

    sig_EXPLORE = [0.125] * 8          max |EXPLORE - uniform| = 1.4e-17
    KL(noise_free_synth || sig_EXPLORE) = 0.565
    KL(noise_free_synth || sig[0])      = 2.113

The four signatures are one shape permuted across four pairs that tile the whole space, so
averaging them is EXACTLY flat, and the result sits four times closer to synthetic content
than any goal the observer actually holds. E19 run that way could only ever return "EXPLORE
absorbed the foreign content", which the spec instructs be reported as evidence that the
framework has a serious problem. It would have been an artifact of the feature partition.

The F = 16 split fixes this without touching the construction. The mean over the four HUMAN
signatures is flat across the human block and at floor across the foreign block, which is
precisely the "human-shaped but goal-agnostic" object the epistemic-foraging argument
describes. It is uniform *within the observer's own policy space*, which is what the FEP
argument actually says, and it is not uniform over features. E19 becomes a test with two
reachable outcomes instead of one.

This module therefore refuses to build an EXPLORE signature that is flat over the whole
feature space. See ``assert_explore_is_not_globally_uniform``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .metrics import js_divergence, shannon_entropy

_LOG_FLOOR = 1e-12

# The V4 feature partition. Human block first so that features 0-7 keep their V1-V3 meaning.
HUMAN_PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]]
FOREIGN_PAIRS = [[8, 9], [10, 11], [12, 13], [14, 15]]
HUMAN_FEATURES = [f for pair in HUMAN_PAIRS for f in pair]
FOREIGN_FEATURES = [f for pair in FOREIGN_PAIRS for f in pair]
NUM_FEATURES_V4 = len(HUMAN_FEATURES) + len(FOREIGN_FEATURES)
NUM_REAL_GOALS = len(HUMAN_PAIRS)

# Index of EXPLORE in the observer's goal space when it is enabled. It is appended AFTER the
# real goals so that goal indices 0..3 keep meaning what they mean in V1-V3, and any metric
# that slices [:NUM_REAL_GOALS] gets the comparable quantity.
EXPLORE = NUM_REAL_GOALS


@dataclass
class V4Signatures:
    """Everything the world and the observer need, built once and frozen."""
    sig_observer: np.ndarray        # (num_observer_goals, 16) the observer's likelihood family
    sig_true: np.ndarray            # (4, 16) the four real human goal signatures
    sig_explore: np.ndarray         # (16,) the epistemic-foraging signature
    sig_foreign: np.ndarray         # (4, 16) the foreign family at this omega
    foreign_basis: np.ndarray       # (4, 16) omega = 0 limit of sig_foreign
    omega: float
    include_explore: bool
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Construction.
# --------------------------------------------------------------------------- #
def _peaked_over(pair, num_features: int, peak: float, floor: float) -> np.ndarray:
    v = np.full(num_features, floor)
    v[pair[0]] += peak / 2.0
    v[pair[1]] += peak / 2.0
    return v / v.sum()


def build_human_signatures(cfg: Config) -> np.ndarray:
    """The four real goal signatures, V1-V3's construction lifted into the 16-feature space.

    Each is peaked on its own HUMAN pair and at floor everywhere else, including across the
    whole foreign block. That last part is the load-bearing bit: it is what makes the foreign
    block unreachable by any in-family hypothesis.
    """
    peak = float(cfg.artifact_model.sig_peak_mass)
    floor = float(cfg.artifact_model.sig_floor)
    return np.stack([_peaked_over(p, NUM_FEATURES_V4, peak, floor) for p in HUMAN_PAIRS])


def build_explore_signature(sig_true: np.ndarray) -> np.ndarray:
    """sig_EXPLORE = normalize(mean_g(sig_true[g])) — V4 spec C2, unmodified.

    Human-shaped but goal-agnostic. An exploring human still moves like a human, so the
    flattening happens WITHIN the agent's own policy space rather than globally. In this
    construction that means flat across the human block and at floor across the foreign one.
    """
    ex = np.asarray(sig_true, dtype=float).mean(axis=0)
    return ex / ex.sum()


def build_foreign_basis(cfg: Config, seed: int) -> np.ndarray:
    """``foreign_basis[k]`` — the omega = 0 limit. Drawn once, deterministically.

    Each foreign goal OWNS a foreign pair, the way each human goal owns a human pair, and its
    shape on top of that is a Dirichlet draw rather than the fixed peak the human family uses.
    Concretely, the Dirichlet concentration vector is ``base`` everywhere on the foreign block
    with ``anchor`` added on goal k's own pair.

    WHY ANCHORED RATHER THAN A PLAIN DRAW, and this was decided by measurement rather than by
    preference. An unanchored low-concentration Dirichlet lets two foreign goals peak on the
    same feature by chance, which collapses MI(features; foreign_goal) and makes it lurch with
    the seed. Measured on the shipped config before the anchor existed:

        human family                          MI = 1.0065 nats
        plain Dirichlet, concentration 0.25   MI = 0.6427
        plain Dirichlet, concentration 0.10   MI = 0.7269
        plain Dirichlet, concentration 0.05   MI = 0.5977   (non-monotone: collisions)

    All three fail N18, and tightening the concentration does not reliably fix it. The failure
    is structural: the human family gets its goal-directedness from a DESIGNED partition, and
    a foreign family that is meant to be exactly as goal-directed needs the same guarantee.
    Anchoring supplies it while leaving the within-pair shape random, so the foreign family is
    not merely the human family translated by eight indices.

    This is what C1 means by "structured, goal-dependent, and genuinely out-of-family rather
    than merely different". The ONE thing that differs between the two families is whether the
    observer's hypothesis space covers the block they live on, which is the variable V4 exists
    to isolate.
    """
    base = float(cfg.get("v4.foreign.concentration", 0.25))
    anchor = float(cfg.get("v4.foreign.anchor", 6.0))
    floor = float(cfg.get("v4.foreign.floor", 1.0e-3))
    rng = np.random.default_rng(int(seed))
    basis = np.zeros((NUM_REAL_GOALS, NUM_FEATURES_V4))
    for k in range(NUM_REAL_GOALS):
        conc = np.full(len(FOREIGN_FEATURES), base)
        for f in FOREIGN_PAIRS[k]:
            conc[FOREIGN_FEATURES.index(f)] += anchor
        draw = rng.dirichlet(conc)
        v = np.full(NUM_FEATURES_V4, floor)
        v[FOREIGN_FEATURES] = np.maximum(draw, floor)
        basis[k] = v / v.sum()
    return basis


def build_foreign_signatures(sig_true: np.ndarray, foreign_basis: np.ndarray,
                             omega: float) -> np.ndarray:
    """sig_foreign[k] = normalize((1 - omega) * foreign_basis[k] + omega * sig_true[k]).

    omega = 1 is the human family exactly; omega = 0 is fully foreign; in between is the
    partial-overlap regime where E20 predicts confident fabrication peaks.
    """
    w = float(omega)
    assert 0.0 <= w <= 1.0, f"omega must be in [0, 1], got {w}"
    mixed = (1.0 - w) * np.asarray(foreign_basis, dtype=float) + w * np.asarray(sig_true, float)
    mixed = np.clip(mixed, _LOG_FLOOR, None)
    return mixed / mixed.sum(axis=1, keepdims=True)


def build_v4_signatures(cfg: Config, omega: float, include_explore: bool,
                        foreign_seed: int | None = None) -> V4Signatures:
    """Assemble the whole family and run every C1/C2 assertion on it."""
    seed = int(cfg.get("v4.foreign.draw_seed", 41) if foreign_seed is None else foreign_seed)
    sig_true = build_human_signatures(cfg)
    sig_explore = build_explore_signature(sig_true)
    basis = build_foreign_basis(cfg, seed)
    sig_foreign = build_foreign_signatures(sig_true, basis, omega)

    sig_observer = (np.vstack([sig_true, sig_explore[None, :]]) if include_explore
                    else sig_true.copy())

    sigs = V4Signatures(sig_observer=sig_observer, sig_true=sig_true,
                        sig_explore=sig_explore, sig_foreign=sig_foreign,
                        foreign_basis=basis, omega=float(omega),
                        include_explore=bool(include_explore))
    sigs.diagnostics = assert_c1_properties(cfg, sigs)
    assert_explore_is_not_globally_uniform(sig_explore)
    return sigs


# --------------------------------------------------------------------------- #
# Assertions. C1 names three properties and says each gets one.
# --------------------------------------------------------------------------- #
def mi_features_goal(likelihoods: np.ndarray) -> float:
    """MI(features; goal) in nats for a family ``p(feature | goal)`` under a uniform goal
    prior. This is the quantity all three C1 properties are stated in."""
    L = np.asarray(likelihoods, dtype=float)
    g = L.shape[0]
    pg = 1.0 / g
    pf = L.mean(axis=0)
    mi = 0.0
    for k in range(g):
        row = L[k]
        nz = row > 0
        mi += pg * float(np.sum(row[nz] * np.log(row[nz] / np.clip(pf[nz], _LOG_FLOOR, None))))
    return mi


def mi_observer_reading_foreign(sig_observer: np.ndarray) -> float:
    """C1 property 2 — MI(features; observer_goal) evaluated ON foreign content.

    Operationalised as: restrict each of the observer's own goal likelihoods to the foreign
    feature block and renormalise, then ask how much a feature drawn from that block tells you
    about which observer-goal produced it. If the observer's signatures are identical over the
    foreign block, which they are by construction since all of them sit at floor there, this
    is zero and the observer genuinely cannot read foreign content.

    Measuring it rather than asserting it from the construction is the point: a later change
    that leaks human structure into the foreign block would show up here as a nonzero value.
    """
    sub = np.asarray(sig_observer, dtype=float)[:, FOREIGN_FEATURES]
    sub = np.clip(sub, _LOG_FLOOR, None)
    sub = sub / sub.sum(axis=1, keepdims=True)
    return mi_features_goal(sub)


def assert_c1_properties(cfg: Config, sigs: V4Signatures) -> dict:
    """The three properties C1 requires, each with its own assertion and its own floor.

    Property 1 is the one that distinguishes V4 from V3, and §6 says a construction that fails
    it "has reverted to the previous model". It is checked at the omega = 0 limit, because
    that is where the claim "foreign content is goal-directed" has to hold; at omega = 1 the
    foreign family IS the human family and high MI is trivial.
    """
    mi_observer_ceiling = float(cfg.get("v4.foreign.observer_mi_ceiling", 0.05))
    entropy_ceiling = float(cfg.get("v4.foreign.entropy_ceiling", 2.0))

    # 1. Foreign content is GOAL-DIRECTED. This is the whole reframe.
    #
    # The floor is a FRACTION of the human family's own MI rather than a bare number. That
    # makes it a statement of the claim under test — foreign content is goal-directed, about
    # as much as human content is — instead of a threshold that could be slid to whatever the
    # construction happens to produce. It also self-calibrates if the human family is ever
    # rebuilt: a weaker human family cannot quietly lower the bar for the foreign one.
    mi_human = mi_features_goal(sigs.sig_true)
    frac = float(cfg.get("v4.foreign.mi_floor_fraction_of_human", 0.85))
    mi_foreign_floor = max(frac * mi_human,
                           float(cfg.get("v4.foreign.mi_floor_absolute", 0.50)))
    mi_foreign = mi_features_goal(sigs.foreign_basis)
    assert mi_foreign >= mi_foreign_floor, (
        f"N18/C1 property 1: MI(features; foreign_goal) = {mi_foreign:.4f} nats is below the "
        f"floor {mi_foreign_floor:.4f} ({frac:.0%} of the human family's {mi_human:.4f}). "
        f"Foreign content that is not itself goal-informative has silently reverted to V3's "
        f"goal-empty model, and every V4 result would be V3's result in new vocabulary.")

    # 2. The OBSERVER cannot read it.
    mi_obs = mi_observer_reading_foreign(sigs.sig_observer)
    assert mi_obs <= mi_observer_ceiling, (
        f"C1 property 2: MI(features; observer_goal) on foreign content = {mi_obs:.4f} nats "
        f"exceeds {mi_observer_ceiling}. The foreign block is readable in-family, so this is "
        f"partial overlap mislabelled as foreign.")

    # 3. Foreign content is STRUCTURED, not noise. V1 §9 N6 carried forward.
    h_foreign = [shannon_entropy(row) for row in sigs.foreign_basis]
    h_max = float(np.max(h_foreign))
    assert h_max < entropy_ceiling, (
        f"C1 property 3: max H(foreign signature) = {h_max:.4f} nats exceeds "
        f"{entropy_ceiling} (uniform over {NUM_FEATURES_V4} features would be "
        f"{np.log(NUM_FEATURES_V4):.4f}). Foreign content must be structured, not noise, "
        f"or the crash is the N6 strawman rather than the reframe.")

    # Guard the disjointness the whole construction rests on.
    human_mass_in_foreign_block = float(sigs.sig_true[:, FOREIGN_FEATURES].sum(axis=1).max())
    foreign_mass_in_human_block = float(sigs.foreign_basis[:, HUMAN_FEATURES].sum(axis=1).max())
    assert human_mass_in_foreign_block < 0.10, (
        f"human signatures carry {human_mass_in_foreign_block:.3f} mass in the foreign block; "
        f"the partition is not disjoint and pre-mortem failure #1 is live")
    assert foreign_mass_in_human_block < 0.10, (
        f"foreign signatures carry {foreign_mass_in_human_block:.3f} mass in the human block; "
        f"the partition is not disjoint and pre-mortem failure #1 is live")

    return {
        "mi_features_foreign_goal": mi_foreign,
        "mi_features_human_goal": mi_human,
        "mi_floor_applied": mi_foreign_floor,
        "foreign_over_human_mi_ratio": mi_foreign / mi_human if mi_human > 0 else float("nan"),
        "mi_features_observer_goal_on_foreign": mi_obs,
        "max_foreign_signature_entropy": h_max,
        "uniform_entropy_16": float(np.log(NUM_FEATURES_V4)),
        "human_mass_in_foreign_block": human_mass_in_foreign_block,
        "foreign_mass_in_human_block": foreign_mass_in_human_block,
    }


def assert_explore_is_not_globally_uniform(sig_explore: np.ndarray) -> dict:
    """C2's own stated requirement, made executable.

    The spec says EXPLORE must not be uniform over features. At V1-V3 cardinality the
    prescribed construction produced exactly that, to 1.4e-17, and it would have decided E19
    on its own. This assertion is the reason the failure cannot recur silently: any future
    change to the feature partition that re-flattens EXPLORE fails here rather than in a
    results file.
    """
    ex = np.asarray(sig_explore, dtype=float)
    n = ex.size
    uniform = np.full(n, 1.0 / n)
    linf = float(np.max(np.abs(ex - uniform)))
    kl = float(np.sum(ex * np.log(np.clip(ex, _LOG_FLOOR, None) / uniform)))
    js_vs_uniform = js_divergence(ex, uniform)
    assert linf > 1.0e-3, (
        f"C2: sig_EXPLORE is uniform over all {n} features (Linf from uniform = {linf:.2e}). "
        f"The spec forbids this in as many words: a uniform-over-features EXPLORE absorbs "
        f"anything, including noise, and would decide E19 by construction rather than by "
        f"measurement. See this module's docstring for how V1-V3 cardinality produced it.")
    # It must still be flat WITHIN the human block: that is what makes it goal-agnostic
    # rather than just another goal.
    human = ex[HUMAN_FEATURES]
    spread = float(human.max() - human.min())
    assert spread < 1.0e-6, (
        f"sig_EXPLORE is not flat across the human block (spread {spread:.2e}); it has become "
        f"a fifth specific goal rather than the goal-agnostic human-shaped signature C2 asks "
        f"for")
    return {"linf_from_uniform": linf, "kl_from_uniform": kl,
            "js_from_uniform": js_vs_uniform,
            "human_block_mass": float(ex[HUMAN_FEATURES].sum()),
            "foreign_block_mass": float(ex[FOREIGN_FEATURES].sum())}
