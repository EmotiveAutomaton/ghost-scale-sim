"""Construction of the observer's generative model: A, B, C, D — plus every
construction-time assertion.

This is the load-bearing module. The single most important decision here (Spec §3.3)
is that ``noise_free_synth`` — the structure that synthetic artifacts carry — is a
peaked, NON-uniform, goal-INDEPENDENT distribution. Synthetic content is structured;
what it lacks is any dependence of that structure on a goal. Making it uniform would
model static, not generative AI, and turn the whole result into an artifact of surprise.

Two objects are kept strictly separate (Spec §3.3):
    * A1_observer   — the observer's *belief* about signal emission; its precision is κ.
    * A1_generative — the true emission process (lives in ``environment.py``).

Load-bearing constraint (Spec §14): the observer has NO preference over provenance or
signal. ``C[0]`` and ``C[1]`` are exactly zero and asserted here at every construction
(null N7). Disengagement must arise from the epistemic term going to zero, never from a
distaste written into C.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

from . import constants as K
from .config import Config
from .metrics import (
    normalize,
    shannon_entropy,
    js_divergence,
    mutual_information_features_goal,
)


# --------------------------------------------------------------------------- #
# Cardinality helper.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cards:
    provenance: int
    goals: int
    attention: int
    features: int
    signals: int
    effort: int

    @property
    def num_states(self) -> list[int]:
        return [self.provenance, self.goals, self.attention]

    @property
    def num_obs(self) -> list[int]:
        return [self.features, self.signals, self.effort]


def cards(cfg: Config) -> Cards:
    c = cfg.cardinalities
    return Cards(c.num_provenance, c.num_goals, c.num_attention,
                 c.num_features, c.num_signals, c.num_effort)


# --------------------------------------------------------------------------- #
# Building blocks of A[0].
# --------------------------------------------------------------------------- #
def build_goal_signatures(cfg: Config) -> np.ndarray:
    """sig[g] — a deterministic peaked categorical over features, mass on the goal's
    feature pair. Shape (num_goals, num_features).

    Distinguishable by construction; the pairwise Jensen-Shannon check lives in
    ``check_signature_invariants`` / the invariant tests.
    """
    cd = cards(cfg)
    pairs = cfg.artifact_model.goal_feature_pairs
    peak = float(cfg.artifact_model.sig_peak_mass)
    floor = float(cfg.artifact_model.sig_floor)
    assert len(pairs) == cd.goals, "goal_feature_pairs must have one entry per goal"

    sig = np.zeros((cd.goals, cd.features))
    for g, pair in enumerate(pairs):
        a, b = pair
        v = np.full(cd.features, floor)
        v[a] += peak / 2.0
        v[b] += peak / 2.0
        sig[g] = v / v.sum()
    return sig


def build_noise_free_synth(cfg: Config, uniform_override: bool = False) -> np.ndarray:
    """noise_free_synth — ONE peaked, non-uniform, goal-independent distribution over
    features, drawn once from a Dirichlet(concentration < 1) with a dedicated seed and
    then frozen. Shape (num_features,).

    ``uniform_override=True`` replaces it with a uniform distribution — the deliberate
    NOISE STRAWMAN of null N6, used only to show the MI/entropy diagnostic separates the
    two cases. It is NOT the default and must never be the default.
    """
    cd = cards(cfg)
    if uniform_override:
        return np.full(cd.features, 1.0 / cd.features)
    rng = np.random.default_rng(int(cfg.artifact_model.noise_free_synth_seed))
    conc = float(cfg.artifact_model.noise_free_synth_concentration)
    synth = rng.dirichlet(np.full(cd.features, conc))
    return synth


def alpha_by_provenance(cfg: Config,
                        overrides: dict | None = None,
                        permutation: list[int] | None = None) -> np.ndarray:
    """Intent-transmission coefficients α[p], indexed by provenance (Spec §3.3).

    ``overrides`` maps tier name -> value (null N1: {'GHOST': 1.0}).
    ``permutation`` reorders the provenance→α mapping (null N4: permutation control).
    """
    raw = dict(cfg.artifact_model.alpha.raw)
    if overrides:
        raw.update(overrides)
    alpha = np.array([raw[name] for name in K.PROVENANCE_NAMES], dtype=float)
    if permutation is not None:
        alpha = alpha[np.asarray(permutation, dtype=int)]
    return alpha


# --------------------------------------------------------------------------- #
# A, B, C, D.
# --------------------------------------------------------------------------- #
def build_A0(cfg: Config, sig: np.ndarray, noise_free_synth: np.ndarray,
             alpha: np.ndarray) -> np.ndarray:
    """A[0] artifact_features, shape (features, provenance, goal, attention).

        DEEP: A0[:, p, g, DEEP] = α[p]·sig[g] + (1-α[p])·noise_free_synth
        SKIM: A0[:, p, g, SKIM] = noise_free_synth   (skim never resolves goals)
    """
    cd = cards(cfg)
    A0 = np.zeros((cd.features, cd.provenance, cd.goals, cd.attention))
    for p in range(cd.provenance):
        a = alpha[p]
        for g in range(cd.goals):
            A0[:, p, g, K.DEEP] = a * sig[g] + (1.0 - a) * noise_free_synth
            A0[:, p, g, K.SKIM] = noise_free_synth
    return A0


def build_A1_observer(cfg: Config, kappa: float) -> np.ndarray:
    """A[1] ghost_signal as the OBSERVER BELIEVES it, shape (signals, prov, goal, att).

        A1_observer[:, p, :, :] = κ·truthful_row(p) + (1-κ)·uniform_over_5

    κ = precision of THIS one likelihood mapping, not a global temperature (Spec §3.3;
    E5 proves κ ≠ γ). Constant across goal and attention.
    """
    cd = cards(cfg)
    uniform = np.full(cd.signals, 1.0 / cd.signals)
    A1 = np.zeros((cd.signals, cd.provenance, cd.goals, cd.attention))
    for p in range(cd.provenance):
        truthful = np.zeros(cd.signals)
        truthful[K.TRUTHFUL_SIGNAL[p]] = 1.0
        row = kappa * truthful + (1.0 - kappa) * uniform
        for g in range(cd.goals):
            for att in range(cd.attention):
                A1[:, p, g, att] = row
    return A1


def build_A2(cfg: Config) -> np.ndarray:
    """A[2] effort, deterministic, shape (effort, prov, goal, att).
    LOW_COST↔SKIM, HIGH_COST↔DEEP; constant across provenance and goal (Spec §3.3)."""
    cd = cards(cfg)
    A2 = np.zeros((cd.effort, cd.provenance, cd.goals, cd.attention))
    for p in range(cd.provenance):
        for g in range(cd.goals):
            A2[K.HIGH_COST, p, g, K.DEEP] = 1.0
            A2[K.LOW_COST, p, g, K.SKIM] = 1.0
    return A2


def build_B(cfg: Config) -> object:
    """B transitions. B[0],B[1] identity (provenance & goal fixed within an episode).
    B[2] fully controllable: action a sets next attention to a deterministically for
    both actions (Spec §3.4)."""
    cd = cards(cfg)
    B = utils.obj_array(3)
    B[0] = np.eye(cd.provenance)[:, :, None]
    B[1] = np.eye(cd.goals)[:, :, None]
    B2 = np.zeros((cd.attention, cd.attention, cd.attention))  # (next, current, action)
    for a in range(cd.attention):
        B2[a, :, a] = 1.0  # action a -> next attention = a, regardless of current
    B[2] = B2
    return B


def build_C(cfg: Config) -> object:
    """C preferences (Spec §3.5). Only effort carries a preference.

    C[0] = zeros over features, C[1] = zeros over signals  — asserted zero (N7).
    C[2] = [+c_effort, -c_effort]: prefers LOW_COST over HIGH_COST.
    """
    cd = cards(cfg)
    ce = float(cfg.preferences.c_effort)
    C = utils.obj_array(3)
    C[0] = np.zeros(cd.features)
    C[1] = np.zeros(cd.signals)
    C[2] = np.array([+ce, -ce])  # index 0 = LOW_COST (preferred), index 1 = HIGH_COST
    return C


def build_D(cfg: Config, rng: np.random.Generator) -> object:
    """D priors (Spec §3.6). Observers are HETEROGENEOUS — this is not cosmetic; the H2
    result is a between-observer claim (null N3).

        D[1] = normalize(uniform + eps · dirichlet(ones))     # goal prior
        D[0] = normalize(uniform + eps · dirichlet(ones))     # provenance prior (if enabled)
        D[2] = uniform over attention (belief only; pinned by the deterministic effort obs)
    """
    cd = cards(cfg)
    eps = float(cfg.priors.eps)
    D = utils.obj_array(3)

    uni_p = np.full(cd.provenance, 1.0 / cd.provenance)
    if bool(cfg.priors.perturb_provenance):
        D[0] = normalize(uni_p + eps * rng.dirichlet(np.ones(cd.provenance)))
    else:
        D[0] = uni_p

    uni_g = np.full(cd.goals, 1.0 / cd.goals)
    D[1] = normalize(uni_g + eps * rng.dirichlet(np.ones(cd.goals)))

    D[2] = np.full(cd.attention, 1.0 / cd.attention)
    return D


# --------------------------------------------------------------------------- #
# Assembled model + agent.
# --------------------------------------------------------------------------- #
@dataclass
class GenerativeModel:
    """The observer's shared model pieces (A, B, C) and the blocks used to build A."""
    A: object
    B: object
    C: object
    sig: np.ndarray
    noise_free_synth: np.ndarray
    alpha: np.ndarray
    kappa: float
    cfg: Config = field(repr=False)


def build_shared_model(cfg: Config,
                       kappa: float | None = None,
                       alpha_overrides: dict | None = None,
                       alpha_permutation: list[int] | None = None,
                       uniform_synth: bool = False,
                       run_assertions: bool = True) -> GenerativeModel:
    """Build the observer's shared A, B, C (everything except the per-observer D).

    Null-condition hooks: ``alpha_overrides`` (N1), ``alpha_permutation`` (N4),
    ``uniform_synth`` (N6). κ defaults to ``cfg.signal_model.kappa``.
    """
    kappa = float(cfg.signal_model.kappa) if kappa is None else float(kappa)
    sig = build_goal_signatures(cfg)
    noise_free_synth = build_noise_free_synth(cfg, uniform_override=uniform_synth)
    alpha = alpha_by_provenance(cfg, overrides=alpha_overrides, permutation=alpha_permutation)

    A = utils.obj_array(3)
    A[0] = build_A0(cfg, sig, noise_free_synth, alpha)
    A[1] = build_A1_observer(cfg, kappa)
    A[2] = build_A2(cfg)
    B = build_B(cfg)
    C = build_C(cfg)

    gm = GenerativeModel(A=A, B=B, C=C, sig=sig, noise_free_synth=noise_free_synth,
                         alpha=alpha, kappa=kappa, cfg=cfg)
    if run_assertions:
        assert_construction_invariants(gm, cfg)
    return gm


def make_agent(gm: GenerativeModel, D: object, cfg: Config,
               kappa: float | None = None) -> Agent:
    """Construct a pymdp legacy Agent for one observer (Spec §3.7)."""
    a = cfg.agent
    return Agent(
        A=gm.A, B=gm.B, C=gm.C, D=D,
        control_fac_idx=[K.F_ATTENTION],
        policy_len=int(a.policy_len),
        inference_horizon=int(a.inference_horizon),
        use_utility=bool(a.use_utility),
        use_states_info_gain=bool(a.use_states_info_gain),
        action_selection=str(a.action_selection),
        gamma=float(a.gamma),
    )


# --------------------------------------------------------------------------- #
# Assertions.
# --------------------------------------------------------------------------- #
def assert_preferences_zero(C: object) -> None:
    """Null N7 — run at construction in EVERY experiment. Guards against the single most
    likely way this build silently becomes worthless (a smuggled preference)."""
    assert np.all(np.asarray(C[0]) == 0.0), "C[0] (features) must be exactly zero (N7)"
    assert np.all(np.asarray(C[1]) == 0.0), "C[1] (signals) must be exactly zero (N7)"


def assert_construction_invariants(gm: GenerativeModel, cfg: Config) -> None:
    """Cheap invariants run on every build: shapes, column-stochasticity, C-zero."""
    cd = cards(cfg)
    atol = float(cfg.tolerances.prob_sum_atol)

    assert np.asarray(gm.A[0]).shape == (cd.features, cd.provenance, cd.goals, cd.attention)
    assert np.asarray(gm.A[1]).shape == (cd.signals, cd.provenance, cd.goals, cd.attention)
    assert np.asarray(gm.A[2]).shape == (cd.effort, cd.provenance, cd.goals, cd.attention)

    for m in range(3):
        col_sums = np.asarray(gm.A[m]).sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=atol), f"A[{m}] columns must sum to 1"
    for f in range(3):
        col_sums = np.asarray(gm.B[f]).sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=atol), f"B[{f}] columns must sum to 1"

    assert_preferences_zero(gm.C)


def check_signature_invariants(gm: GenerativeModel, cfg: Config) -> dict:
    """Heavier invariants (Spec §10): pairwise JS(sig) above threshold, H(synth) below
    the structured ceiling, and MI(features;goal) monotonically decreasing across tiers.

    Returns a dict of the measured quantities (used by tests and reporting). Raises on
    violation of the JS / entropy / monotonicity requirements.
    """
    cd = cards(cfg)
    am = cfg.artifact_model

    # Pairwise JS divergence between goal signatures.
    js_thresh = float(am.js_threshold)
    js_pairs = {}
    for g1 in range(cd.goals):
        for g2 in range(g1 + 1, cd.goals):
            d = js_divergence(gm.sig[g1], gm.sig[g2])
            js_pairs[(g1, g2)] = d
            assert d > js_thresh, (
                f"sig[{g1}] vs sig[{g2}] JS={d:.4f} <= threshold {js_thresh}")

    # Synthetic content is structured, NOT high-entropy.
    h_synth = shannon_entropy(gm.noise_free_synth)
    ceiling = float(am.structured_ceiling)
    assert h_synth < ceiling, (
        f"H(noise_free_synth)={h_synth:.4f} nats must be < structured_ceiling {ceiling} "
        f"(uniform would be ln({cd.features})={np.log(cd.features):.4f})")

    # MI(features; goal | provenance) monotonically decreasing CREATOR>POLISHED>CURATOR>GHOST.
    mi_by_tier = [mutual_information_features_goal(gm.A[0], p, K.DEEP)
                  for p in range(cd.provenance)]
    for p in range(cd.provenance - 1):
        assert mi_by_tier[p] >= mi_by_tier[p + 1] - 1e-9, (
            f"MI must be non-increasing across tiers; "
            f"MI[{K.PROVENANCE_NAMES[p]}]={mi_by_tier[p]:.4f} < "
            f"MI[{K.PROVENANCE_NAMES[p+1]}]={mi_by_tier[p+1]:.4f}")

    return {"js_pairs": js_pairs, "h_noise_free_synth": h_synth, "mi_by_tier": mi_by_tier}
