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


def build_noise_free_synth(cfg: Config, uniform_override: bool = False,
                           goal_symmetric: bool | None = None,
                           synth_draw_seed: int | None = None) -> np.ndarray:
    """noise_free_synth — ONE peaked, non-uniform, goal-independent distribution over
    features, drawn once from a Dirichlet(concentration < 1) with a dedicated seed and
    then frozen. Shape (num_features,).

    GOAL-SYMMETRY (config ``goal_symmetric``). V2 C2 makes this an EXPERIMENT-LEVEL AXIS
    rather than a global default; both arms are named conditions and neither is "correct".

    * ``True`` (V1 behaviour, retained as the named control): the raw Dirichlet draw is
      symmetrized across the goal-pair group so the frozen synth is EQUIDISTANT from every
      goal signature. An un-symmetrized draw generically resembles ONE goal by chance, which
      would turn the H2 result from *arbitrary, observer-specific* hallucination (confident
      DISAGREEMENT — the theoretical claim) into spurious CONSENSUS on that one goal.
      Symmetrizing removes that confound while keeping the distribution structured.
    * ``False`` (V2 C2): the raw draw is kept and generically leans toward some goal ``k``.
      Symmetric synthetic content is NOISE and averages out over a corpus; biased synthetic
      content is an ATTRACTOR and accumulates linearly in sample size. That distinction is
      the axis V2 exists to measure — see ``synth_lean``.

    SUPPORT FLOOR (``synth_floor``, decision D3b). The raw Dirichlet(0.03) draw contains hard
    zeros. A zero mass makes those features IMPOSSIBLE under GHOST, so observing one would be
    a *proof* of non-GHOST provenance — an unintended perfect provenance channel that bypasses
    the Ghost Scale entirely (and it would bias AGAINST the C2 effect, by letting an
    aggregator exclude synthetics for free). It also makes the per-column KL diagnostics of E9
    infinite. The floor is applied in BOTH arms and then renormalized; it is a no-op for the
    symmetrized V1 draw (whose minimum mass is 0.007 >> 1e-3), so N8/N9 are unaffected.

    ``uniform_override=True`` replaces it with a uniform distribution — the deliberate
    NOISE STRAWMAN of null N6. It is NOT the default and must never be the default.
    """
    cd = cards(cfg)
    if uniform_override:
        return np.full(cd.features, 1.0 / cd.features)
    seed = (int(cfg.artifact_model.noise_free_synth_seed)
            if synth_draw_seed is None else int(synth_draw_seed))
    rng = np.random.default_rng(seed)
    conc = float(cfg.artifact_model.noise_free_synth_concentration)
    synth = rng.dirichlet(np.full(cd.features, conc))
    symmetric = (bool(cfg.artifact_model.get("goal_symmetric", True))
                 if goal_symmetric is None else bool(goal_symmetric))
    if symmetric:
        synth = _symmetrize_over_goal_pairs(synth, cfg.artifact_model.goal_feature_pairs)
    floor = float(cfg.artifact_model.get("synth_floor", 1.0e-3))
    if floor > 0.0:
        synth = np.maximum(synth, floor)
    return synth / synth.sum()


def synth_lean(synth: np.ndarray, pairs: list[list[int]]) -> tuple[int, float]:
    """Which goal does a synthetic distribution lean toward, and by how much (C2).

    Returns ``(k, lean)`` where ``k`` is the goal whose feature pair carries the most
    synthetic mass and ``lean`` is that mass. Because the goal pairs tile the feature space,
    a perfectly symmetric synth gives ``lean = 1/num_goals`` (no lean) and a synth
    concentrated entirely on one goal's pair gives ``lean = 1.0``.

    This is the quantity that sets E6b's pre-registered bound: contamination at fraction f
    pulls the recovered preference distribution toward ``onehot(k)``, so ``k`` decides
    whether that bound is small (k is a common goal) or large (k is a rare one).
    """
    synth = np.asarray(synth, dtype=float)
    pair_mass = np.array([float(synth[a] + synth[b]) for a, b in pairs])
    k = int(np.argmax(pair_mass))
    return k, float(pair_mass[k])


def build_observer_signature(sig_true: np.ndarray, d_i: float,
                             rng: np.random.Generator) -> np.ndarray:
    """C1 — the observer's OWN goal signatures, a perturbation of the shared latent one:

        sig_i[g] = normalize( (1 - d_i) * sig_true[g] + d_i * rng.dirichlet(ones(F)) )

    ``d_i`` is INEXPERTISE. d=0 is a perfect reader of this domain; d->1 is a novice whose
    intent-reading template is essentially arbitrary. Expertise is ``1 - d_i``.

    This is C1's central claim: observers share a *family* of likelihoods because they share
    a body plan, but each individual's mapping from artifact to inferred goal is built from
    their own sensory history. V1 gave observers heterogeneous priors and one shared
    likelihood, which is the opposite of what the theory says.

    N8 HAZARD — READ BEFORE EDITING. At ``d_i == 0`` this returns ``sig_true`` and consumes
    NO variate from ``rng``. That is deliberate and load-bearing: N8 requires that V2 with
    d=0 reproduce V1 exactly, and any variate consumed here would shift every downstream draw
    in the stream. The caller must ALSO pass a dedicated RNG stream (see
    ``observer.observer_sig_rng``) so the D-prior stream is untouched regardless.
    """
    d_i = float(d_i)
    if d_i <= 0.0:
        return np.asarray(sig_true, dtype=float).copy()
    sig_true = np.asarray(sig_true, dtype=float)
    num_goals, num_features = sig_true.shape
    out = np.empty_like(sig_true)
    for g in range(num_goals):
        out[g] = normalize((1.0 - d_i) * sig_true[g]
                           + d_i * rng.dirichlet(np.ones(num_features)))
    return out


def _symmetrize_over_goal_pairs(dist: np.ndarray, pairs: list[list[int]]) -> np.ndarray:
    """Make ``dist`` invariant under relabeling of goals (hence equidistant from every goal
    signature) WHILE PRESERVING its structure.

    Each pair is assigned the pooled *larger* mass on its first feature and the pooled
    *smaller* mass on its second feature (both pooled as means across pairs). Because the
    same (high, low) shape is applied to every pair, the result is goal-symmetric; because
    high > low (the raw draw is peaked), it stays structured — unlike a plain average over
    permutations, which would collapse toward uniform.
    """
    dist = np.asarray(dist, dtype=float)
    highs = np.mean([max(dist[a], dist[b]) for a, b in pairs])
    lows = np.mean([min(dist[a], dist[b]) for a, b in pairs])
    out = np.zeros_like(dist)
    for a, b in pairs:
        out[a] = highs
        out[b] = lows
    covered = {i for pair in pairs for i in pair}
    for i in range(len(dist)):
        if i not in covered:
            out[i] = dist[i]
    return out / out.sum()


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


def build_A1_regime_aware(cfg: Config, kappa: float,
                          signing_rate_by_provenance: dict[int, float]) -> np.ndarray:
    """A[1] for an observer who KNOWS THE LABELLING REGIME (E16).

        A1[:, p, :, :] = s_p · (κ·truthful_row(p) + (1-κ)·uniform_over_the_4_signal_types)
                         + (1 - s_p) · onehot(UNSIGNED)

    -------------------------------------------------------------------------------------
    WHY THIS EXISTS, AND WHY IT IS A SEPARATE FUNCTION RATHER THAN A FIX.

    ``build_A1_observer`` gives ``P(UNSIGNED | provenance) = (1-κ)/5`` — the SAME value for
    every provenance. Under that likelihood the absence of a label carries exactly zero
    information about where an artifact came from, by construction and at any coverage.

    That is fine for V1/V2, where labelling is uniform across tiers so absence really is
    uninformative. It is fatal to the question "how much AI content must be labelled to
    protect the corpus", because the entire policy mechanism is the CONTRAPOSITIVE: if
    synthetic work is disclosed at rate p, an unlabelled artifact is evidence of humanity,
    and that inference is unavailable to an observer holding the V1/V2 likelihood.

    So this is not a bug fix. It is a **different observer**, carrying an extra piece of
    knowledge — the coverage of the regime it lives under — and the difference between the
    two is itself the finding E16 reports. A reader who does not know a disclosure mandate
    exists cannot benefit from it, however good the mandate is.

    ``s_p`` is the observer's BELIEF about coverage. E16 gives it the true value, which is
    the most generous assumption available and therefore an upper bound on what disclosure
    can buy; a miscalibrated reader can only do worse.
    -------------------------------------------------------------------------------------
    """
    cd = cards(cfg)
    signal_types = [s for s in range(cd.signals) if s != K.UNSIGNED]
    uniform_signed = np.zeros(cd.signals)
    for s in signal_types:
        uniform_signed[s] = 1.0 / len(signal_types)

    A1 = np.zeros((cd.signals, cd.provenance, cd.goals, cd.attention))
    for p in range(cd.provenance):
        s_p = float(signing_rate_by_provenance.get(int(p), 0.0))
        truthful = np.zeros(cd.signals)
        truthful[K.TRUTHFUL_SIGNAL[p]] = 1.0
        signed_row = kappa * truthful + (1.0 - kappa) * uniform_signed
        row = s_p * signed_row
        row[K.UNSIGNED] += (1.0 - s_p)
        row = row / row.sum()
        for g in range(cd.goals):
            for att in range(cd.attention):
                A1[:, p, g, att] = row
    assert np.allclose(A1.sum(axis=0), 1.0, atol=1e-10), \
        "A1_regime_aware must stay column-stochastic"
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
    """A, B, C plus the blocks used to build A.

    Serves two roles, and V2 keeps them conceptually distinct even though they share a type:

    * a **world model** (``build_shared_model``) — ``sig`` is the shared latent signature
      ``sig_true``, and this is what ``Environment``/``creators`` consume as ground truth.
      The world is NOT affected by observer expertise.
    * an **observer model** (``build_observer_model``) — ``sig`` is that observer's own
      ``sig_i`` (C1) and ``A[0]`` is rebuilt from it. ``d`` records its inexpertise and
      ``sig_true`` retains the latent signature it was perturbed from.

    At ``d == 0`` the two coincide exactly, which is what null N8 checks.
    """
    A: object
    B: object
    C: object
    sig: np.ndarray
    noise_free_synth: np.ndarray
    alpha: np.ndarray
    kappa: float
    cfg: Config = field(repr=False)
    d: float = 0.0                       # observer inexpertise (0 for a world model)
    sig_true: np.ndarray = None          # the shared latent signature sig was drawn from
    synth_k: int = -1                    # goal the synthetic distribution leans toward (C2)
    synth_lean: float = 0.0              # mass on that goal's feature pair
    goal_symmetric: bool = True          # which C2 arm this model was built in

    def __post_init__(self):
        if self.sig_true is None:
            self.sig_true = self.sig


def build_shared_model(cfg: Config,
                       kappa: float | None = None,
                       alpha_overrides: dict | None = None,
                       alpha_permutation: list[int] | None = None,
                       uniform_synth: bool = False,
                       goal_symmetric: bool | None = None,
                       synth_draw_seed: int | None = None,
                       run_assertions: bool = True) -> GenerativeModel:
    """Build the WORLD model: the shared latent A, B, C (everything except the per-observer D).

    This is ground truth. ``Environment`` and ``creators`` consume it; observers derive their
    own models from it via ``build_observer_model``. At ``d=0`` an observer model is identical
    to this one, so V1 call sites keep V1 behaviour with no edit.

    Null-condition hooks: ``alpha_overrides`` (N1), ``alpha_permutation`` (N4),
    ``uniform_synth`` (N6). C2 hooks: ``goal_symmetric``, ``synth_draw_seed``.
    κ defaults to ``cfg.signal_model.kappa``.
    """
    kappa = float(cfg.signal_model.kappa) if kappa is None else float(kappa)
    sig = build_goal_signatures(cfg)
    noise_free_synth = build_noise_free_synth(cfg, uniform_override=uniform_synth,
                                              goal_symmetric=goal_symmetric,
                                              synth_draw_seed=synth_draw_seed)
    alpha = alpha_by_provenance(cfg, overrides=alpha_overrides, permutation=alpha_permutation)
    k, lean = synth_lean(noise_free_synth, cfg.artifact_model.goal_feature_pairs)

    A = utils.obj_array(3)
    A[0] = build_A0(cfg, sig, noise_free_synth, alpha)
    A[1] = build_A1_observer(cfg, kappa)
    A[2] = build_A2(cfg)
    B = build_B(cfg)
    C = build_C(cfg)

    symmetric = (bool(cfg.artifact_model.get("goal_symmetric", True))
                 if goal_symmetric is None else bool(goal_symmetric))
    gm = GenerativeModel(A=A, B=B, C=C, sig=sig, noise_free_synth=noise_free_synth,
                         alpha=alpha, kappa=kappa, cfg=cfg, d=0.0, sig_true=sig,
                         synth_k=k, synth_lean=lean, goal_symmetric=symmetric)
    if run_assertions:
        assert_construction_invariants(gm, cfg)
    return gm


def build_observer_model(world: GenerativeModel, cfg: Config, d_i: float,
                         rng_sig: np.random.Generator | None = None,
                         kappa: float | None = None) -> GenerativeModel:
    """Derive ONE observer's model from the world model (C1).

    The observer gets its own ``sig_i`` and an ``A[0]`` rebuilt from it; ``A[1]`` (κ) and
    ``A[2]`` (effort) are shared with the world, since those are not what expertise is about.

    FAST PATH — load-bearing for N8. At ``d_i == 0`` (and no κ override) the observer model
    IS the world model: the same object is returned, no array is rebuilt, and no variate is
    drawn from ``rng_sig``. V2 at d=0 is therefore V1, bit for bit.
    """
    d_i = float(d_i)
    kappa_eff = world.kappa if kappa is None else float(kappa)
    if d_i <= 0.0 and kappa_eff == world.kappa:
        return world

    sig_i = build_observer_signature(world.sig_true, d_i,
                                     rng_sig if rng_sig is not None
                                     else np.random.default_rng(0))
    A = utils.obj_array(3)
    A[0] = build_A0(cfg, sig_i, world.noise_free_synth, world.alpha)
    A[1] = world.A[1] if kappa_eff == world.kappa else build_A1_observer(cfg, kappa_eff)
    A[2] = world.A[2]
    return GenerativeModel(A=A, B=world.B, C=world.C, sig=sig_i,
                           noise_free_synth=world.noise_free_synth, alpha=world.alpha,
                           kappa=kappa_eff, cfg=cfg, d=d_i, sig_true=world.sig_true,
                           synth_k=world.synth_k, synth_lean=world.synth_lean,
                           goal_symmetric=world.goal_symmetric)


def make_agent(gm: GenerativeModel, D: object, cfg: Config,
               kappa: float | None = None) -> Agent:
    """Construct the agent for one observer (Spec §3.7).

    THE SOLVER IS A CONFIG SWITCH, and that is the whole design of validation item V-1.
    ``inference.exact: true`` returns an ``ExactAgent`` — the same A, B, C, D, the same policy
    set, the same gamma, the same action rule, and a belief over the full joint instead of a
    factorised one. Every experiment therefore re-runs under exact inference through its own
    unmodified code path, so a number that moves was moved by the factorisation and not by a
    second implementation of the experiment. Defaults to false, so nothing already committed
    changes meaning.
    """
    if bool(cfg.get("inference.exact", False)):
        from .exact import make_exact_agent
        return make_exact_agent(gm, D, cfg)
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


def assert_A_column_stochastic(A: object, cfg: Config, where: str = "") -> None:
    """V2 §3 invariant: EVERY pA update must leave A column-stochastic. Asserted after each
    ``update_A`` in ``learning.learn_step`` — unconditionally, not debug-gated, because a
    silently un-normalized A would corrupt every learned-model diagnostic downstream."""
    atol = float(cfg.tolerances.prob_sum_atol)
    for m in range(len(A)):
        col_sums = np.asarray(A[m]).sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=atol), (
            f"A[{m}] lost column-stochasticity{' at ' + where if where else ''}; "
            f"max deviation {float(np.max(np.abs(col_sums - 1.0))):.3e}")


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
