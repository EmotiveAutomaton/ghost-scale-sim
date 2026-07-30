"""V4.5 — the three-gate observer: kappa_p, beta, theta, plus social influence on the prior.

WHAT IS WRONG WITH THE MODEL THIS REPLACES (V4.5 §3.1). kappa is a single scalar doing three
jobs, and the canonical trust literature says they are separate constructs. Mayer, Davis &
Schoorman's ability / benevolence / integrity and Lee & See's performance / process / purpose
both split trustworthiness into a COMPETENCE dimension and a VALUES dimension. The Psi
equation has neither cleanly:

  * value alignment is DOUBLE-COUNTED. It already sits in theta as
    theta_base(E) + lambda * D_KL(Q(R|tau) || P_c(R)). If kappa also carries value
    convergence, the same quantity enters twice, once multiplicatively and once through the
    gate, and redundant parameters are unidentifiable.
  * competence is ABSENT. MaxEnt IRL's demonstrator likelihood is properly
    P(tau|R) ~ exp(beta * R(tau)). The preprint's Appendix A.1 writes it with beta = 1 — the
    creator modelled as perfectly rational, always. That elision is the missing dimension.
  * social influence has nowhere to sit, because it is not a gate on integration at all. It is
    a shift on the prior, upstream of every gate.

-----------------------------------------------------------------------------------------
THE ARCHITECTURE. Three gates in series, each answering a different question, each acting at
a DIFFERENT POINT in the pipeline. This is E5's kappa-versus-gamma finding generalised: things
that look like one knob act at different positions and produce different signatures.

    gate     question                                    acts on
    kappa_p  is there anything here to read?             the likelihood
    beta     how much does this trajectory constrain R?  the demonstrator model
    theta    should I write down what I found?           the update

-----------------------------------------------------------------------------------------
BETA AND EXPLORE ARE THE SAME AXIS (V4.5 §3.3), AND THAT IS THE IMPLEMENTATION SHORTCUT.

Under low beta the observer models the creator as not optimizing hard, so the expected output
moves away from the goal-specific signature toward the creator's POLICY MARGINAL:

    A[:, p, g, DEEP, b] ~ normalize( beta_b * sig[g] + (1 - beta_b) * mean_g(sig[g]) )

and ``mean_g(sig[g])`` IS ``sig_EXPLORE``, which V4 already built, already pre-registered,
already asserted against global uniformity, and already validated by E19's positive control.
So EXPLORE is beta = 0 as a discrete hypothesis and beta is its continuous generalisation.

That identification is a CLAIM, not a convenience, and V4.5 §3.3 makes it falsifiable: at
beta = 0, E28 must reproduce E19's exploratory-human cell. If a continuous beta near zero does
not recover the discrete EXPLORE result, the two are not the same axis and the shortcut is
wrong.

-----------------------------------------------------------------------------------------
WHERE BETA SITS RELATIVE TO ALPHA, AND WHY IT IS INSIDE RATHER THAN BESIDE IT.

V4.5 §3.3 writes the construction without alpha because it is describing the demonstrator
model in isolation. The pipeline already has alpha[p] mixing the goal signature against
``noise_free_synth``, and the two do different jobs: alpha is how much of the creator's
structure SURVIVES TRANSMISSION, beta is how hard the creator was OPTIMISING in the first
place. Composed in the only order that makes sense, beta acts first, on the creator, and
alpha acts second, on the channel:

    A0[:, p, g, DEEP, b] = alpha[p] * (beta_b * sig[g] + (1 - beta_b) * sig_EXPLORE)
                           + (1 - alpha[p]) * noise_free_synth

At beta = 1 the inner bracket is ``sig[g]`` exactly and the whole expression is V4's A0,
element for element. That is the N-series check V4.5 §7 demands — "at beta = 1 across all
conditions, V4.5 must reproduce V4 within tolerance; if it does not, beta has been wired into
the wrong pipeline position" — and it is asserted here rather than left to a test to notice.

-----------------------------------------------------------------------------------------
WHAT THIS MODULE DOES NOT DO, DELIBERATELY.

* **Integrity is not added as a construct.** Under revealed preference — which is what IRL is
  — values are defined by what behaviour reveals, so integrity cannot diverge from values by
  construction. What CAN diverge is STATED values from revealed ones, and that gap is already
  V4's C4 (``declared_tier`` versus ``omega_true``). V4.5 §3.2: the framework already contains
  integrity under a better name, so nothing is added here.
* **Social influence is never a multiplier on integration.** It shifts P_0(R) and the
  provenance prior before any gate runs. ``assert_no_social_term_in_update_path`` makes that
  executable rather than aspirational.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

from . import constants as K
from . import foreign as FN
from . import metrics
from .config import Config
from .generative_model import (GenerativeModel, alpha_by_provenance, assert_preferences_zero,
                               build_A2, build_C, cards)
from .metrics import normalize
from .v4_model import build_v4_synth, load_v4_config

# The beta factor is appended AFTER attention, so provenance/goal/attention keep the indices
# every V1-V4 call site assumes and ``control_fac_idx=[K.F_ATTENTION]`` stays correct.
F_BETA = 3

_EPS = 1e-12


@dataclass
class V45World:
    gm: GenerativeModel
    sigs: FN.V4Signatures
    cfg: Config
    beta_levels: np.ndarray
    social_shift: np.ndarray = None
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The demonstrator model.
# --------------------------------------------------------------------------- #
def demonstrator_signature(sig_true: np.ndarray, sig_explore: np.ndarray,
                           beta: float) -> np.ndarray:
    """What a creator at rationality ``beta`` is expected to emit, per goal.

        normalize( beta * sig[g] + (1 - beta) * sig_EXPLORE )

    beta = 1 is the perfectly-optimising creator the preprint's Appendix A.1 assumes by
    writing the MaxEnt likelihood with beta = 1. beta = 0 is a creator whose output is its own
    policy marginal — goal-agnostic but still human-shaped, which is exactly the object C2
    built and E19 validated.

    Used for BOTH sides of E28: the observer's likelihood over what a creator would emit, and
    the world's actual generator when producing content at a known true beta. Sharing one
    function is deliberate — an observer whose demonstrator model and whose world disagreed
    about the definition of beta would make the recovery measurement meaningless.
    """
    b = float(beta)
    assert 0.0 <= b <= 1.0, f"beta must be in [0, 1], got {b}"
    mix = b * np.asarray(sig_true, dtype=float) + (1.0 - b) * np.asarray(sig_explore, float)
    mix = np.clip(mix, _EPS, None)
    return mix / mix.sum(axis=-1, keepdims=True)


def build_v45_A0(cfg: Config, sig_observer: np.ndarray, sig_explore: np.ndarray,
                 noise_free_synth: np.ndarray, alpha: np.ndarray,
                 beta_levels: np.ndarray) -> np.ndarray:
    """A[0], shape (features, provenance, goal, attention, beta).

    beta acts on the CREATOR (inside), alpha on the CHANNEL (outside). See the module
    docstring for why that order and not the other.
    """
    nf = int(cfg.cardinalities.num_features)
    ng = int(cfg.cardinalities.num_goals)
    npv = int(cfg.cardinalities.num_provenance)
    na = int(cfg.cardinalities.num_attention)
    nb = int(len(beta_levels))
    A0 = np.zeros((nf, npv, ng, na, nb))
    for bi, b in enumerate(beta_levels):
        demo = demonstrator_signature(sig_observer, sig_explore, float(b))
        for p in range(npv):
            a = float(alpha[p])
            for g in range(ng):
                A0[:, p, g, K.DEEP, bi] = a * demo[g] + (1.0 - a) * noise_free_synth
                A0[:, p, g, K.SKIM, bi] = noise_free_synth
    return A0


def build_v45_A1(cfg: Config, kappa: float, n_beta: int) -> np.ndarray:
    """A[1] ghost_signal, V4's construction with a beta axis that it is constant over.

    beta is a fact about the creator's optimisation, not about the honesty of the provenance
    channel, so the signal likelihood must not depend on it. Constant across beta is the
    statement that beta and kappa are different quantities; if the signal channel varied with
    beta they would be partly the same knob and E29's dissociation would be contaminated at
    construction.
    """
    cd = cards(cfg)
    uniform = np.full(cd.signals, 1.0 / cd.signals)
    A1 = np.zeros((cd.signals, cd.provenance, cd.goals, cd.attention, int(n_beta)))
    for p in range(cd.provenance):
        truthful = np.zeros(cd.signals)
        truthful[K.TRUTHFUL_SIGNAL[p]] = 1.0
        row = kappa * truthful + (1.0 - kappa) * uniform
        A1[:, p, :, :, :] = row[:, None, None, None]
    return A1


def build_v45_A2(cfg: Config, n_beta: int) -> np.ndarray:
    """A[2] effort, deterministic, with a beta axis it is constant over."""
    cd = cards(cfg)
    base = build_A2(cfg)                       # (effort, prov, goal, att)
    return np.repeat(np.asarray(base)[..., None], int(n_beta), axis=-1)


def build_v45_B(cfg: Config, n_beta: int) -> object:
    """B transitions over four factors. beta is a fixed property of the artifact's creator
    within an episode, so B[3] is the identity — the same treatment provenance and goal get,
    and for the same reason: these are things to be INFERRED, not things that change."""
    cd = cards(cfg)
    B = utils.obj_array(4)
    B[0] = np.eye(cd.provenance)[:, :, None]
    B[1] = np.eye(cd.goals)[:, :, None]
    B2 = np.zeros((cd.attention, cd.attention, cd.attention))
    for a in range(cd.attention):
        B2[a, :, a] = 1.0
    B[2] = B2
    B[3] = np.eye(int(n_beta))[:, :, None]
    return B


def build_v45_D(cfg: Config, rng: np.random.Generator, n_beta: int,
                social_shift: np.ndarray | None = None,
                social_weight: float = 0.0) -> object:
    """D priors over four factors, with SOCIAL INFLUENCE entering here and only here.

    V4.5 §3.2 is explicit that social influence is not a gate: it shifts P_0(R) and the prior
    over provenance BEFORE any gate runs, with a per-observer weight. So it is a perturbation
    of D and it appears nowhere else in this module. ``assert_no_social_term_in_update_path``
    is the executable form of that constraint.

    The beta prior is UNIFORM and stays uniform. It is the one thing E28 measures the
    recovery of, and an informative prior over it would be putting the answer in by hand.
    """
    cd = cards(cfg)
    eps = float(cfg.priors.eps)
    D = utils.obj_array(4)

    uni_p = np.full(cd.provenance, 1.0 / cd.provenance)
    D[0] = (normalize(uni_p + eps * rng.dirichlet(np.ones(cd.provenance)))
            if bool(cfg.priors.perturb_provenance) else uni_p)

    uni_g = np.full(cd.goals, 1.0 / cd.goals)
    D[1] = normalize(uni_g + eps * rng.dirichlet(np.ones(cd.goals)))

    D[2] = np.full(cd.attention, 1.0 / cd.attention)
    D[3] = np.full(int(n_beta), 1.0 / int(n_beta))

    if social_shift is not None and float(social_weight) > 0.0:
        w = float(social_weight)
        s = np.asarray(social_shift, dtype=float)
        s = s / s.sum()
        D[1] = normalize((1.0 - w) * np.asarray(D[1], dtype=float) + w * s)
    return D


def build_v45_world(cfg: Config, omega: float = 0.0, kappa: float | None = None,
                    beta_levels=None, foreign_seed: int | None = None,
                    run_assertions: bool = True) -> V45World:
    """The V4.5 world model. Runs V4's C1/C2 assertions plus the beta-position check."""
    kappa = float(cfg.signal_model.kappa) if kappa is None else float(kappa)
    beta_levels = np.asarray(
        cfg.get("v4_5.beta.levels", [0.0, 0.25, 0.5, 0.75, 1.0])
        if beta_levels is None else beta_levels, dtype=float)

    # EXPLORE is not in the observer's goal space here: beta subsumes it (§3.3). Holding both
    # would make the model degenerate, because EXPLORE and beta = 0 are the same hypothesis.
    sigs = FN.build_v4_signatures(cfg, omega=omega, include_explore=False,
                                  foreign_seed=foreign_seed)
    synth = build_v4_synth(cfg)
    alpha = alpha_by_provenance(cfg)

    A = utils.obj_array(3)
    A[0] = build_v45_A0(cfg, sigs.sig_true, sigs.sig_explore, synth, alpha, beta_levels)
    A[1] = build_v45_A1(cfg, kappa, len(beta_levels))
    A[2] = build_v45_A2(cfg, len(beta_levels))

    gm = GenerativeModel(A=A, B=build_v45_B(cfg, len(beta_levels)), C=build_C(cfg),
                         sig=sigs.sig_true, noise_free_synth=synth, alpha=alpha,
                         kappa=kappa, cfg=cfg, d=0.0, sig_true=sigs.sig_true,
                         synth_k=-1, synth_lean=0.0, goal_symmetric=True)

    world = V45World(gm=gm, sigs=sigs, cfg=cfg, beta_levels=beta_levels)
    if run_assertions:
        assert_preferences_zero(gm.C)
        world.diagnostics = assert_beta_position(cfg, sigs, synth, alpha, beta_levels)
        assert_no_social_term_in_update_path()
    return world


def make_v45_observer(world: V45World, rng: np.random.Generator,
                      social_shift: np.ndarray | None = None,
                      social_weight: float = 0.0) -> Agent:
    D = build_v45_D(world.cfg, rng, len(world.beta_levels),
                    social_shift=social_shift, social_weight=social_weight)
    a = world.cfg.agent
    # The solver switch, exactly as in ``generative_model.make_agent``; see validation item V-1.
    if bool(world.cfg.get("inference.exact", False)):
        from .exact import ExactAgent
        return ExactAgent(
            A=world.gm.A, B=world.gm.B, C=world.gm.C, D=D,
            control_fac_idx=[K.F_ATTENTION],
            policy_len=int(a.policy_len), gamma=float(a.gamma),
            action_selection=str(a.action_selection),
            use_utility=bool(a.use_utility),
            use_states_info_gain=bool(a.use_states_info_gain), rng=rng)
    return Agent(
        A=world.gm.A, B=world.gm.B, C=world.gm.C, D=D,
        control_fac_idx=[K.F_ATTENTION],
        policy_len=int(a.policy_len),
        inference_horizon=int(a.inference_horizon),
        use_utility=bool(a.use_utility),
        use_states_info_gain=bool(a.use_states_info_gain),
        action_selection=str(a.action_selection),
        gamma=float(a.gamma),
    )


# --------------------------------------------------------------------------- #
# Reading beta back out.
# --------------------------------------------------------------------------- #
def recovered_beta(beta_posterior: np.ndarray, beta_levels: np.ndarray) -> float:
    """E[beta] under the observer's posterior. The quantity E28 regresses against true beta."""
    q = np.asarray(beta_posterior, dtype=float)
    return float(np.dot(q / q.sum(), np.asarray(beta_levels, dtype=float)))


# --------------------------------------------------------------------------- #
# The theta gate.
# --------------------------------------------------------------------------- #
def value_divergence(goal_posterior: np.ndarray, value_prior: np.ndarray) -> float:
    """D_KL( Q(R|tau) || P_c(R) ) — how far the recovered goal is from what the observer
    values. This is the quantity the Psi equation's theta term is written in.

    Note what it is NOT. It is not a distaste over provenance or signal: C[0] and C[1] stay
    exactly zero (null N7) and nothing here touches them. The observer has no preference about
    where an artifact came from. It has a preference about what it is willing to ADOPT, which
    is a different object acting at a different point, and keeping them separate is what
    prevents theta from leaking into the engagement decision.
    """
    return metrics.kl_divergence(np.asarray(goal_posterior, dtype=float),
                                 np.asarray(value_prior, dtype=float))


def theta_gate(goal_posterior: np.ndarray, value_prior: np.ndarray,
               theta_base: float, lam: float) -> float:
    """The acceptance gate, in [0, 1]:  sigmoid( theta_base - lambda * D_KL(Q || P_c) ).

    theta_base is how open the observer is by default; lambda is how sharply divergence from
    its own values closes it. An OPEN gate (large theta_base, small lambda) integrates
    whatever it recovered. A CLOSED gate recovers the goal perfectly well and declines to
    write it down.

    WHERE THIS ACTS, WHICH IS THE DESIGN DECISION E29 RESTS ON. theta gates the update that is
    CARRIED FORWARD — what the observer's prior becomes for the next encounter — and it does
    not touch inference on the current artifact. That placement is what makes theta a gate on
    integration rather than on perception, and it is the only placement consistent with V4.5
    §3.2's table, where theta acts on "the update" while kappa_p acts on "the likelihood".

    The honest consequence, stated before E29 ran: within a SINGLE artifact, a closed theta and
    a low beta produce the same engagement and the same resolution. They differ on update
    magnitude and on value divergence, and the divergence spike is the sharper of the two
    because it is a positive prediction rather than a difference of degree. E29's decisive
    contrast rests on exactly that, and if it fails, §5 says to report the decomposition as
    not earning its parameters.

    ``lam == 0`` returns EXACTLY 1.0 rather than sigmoid(theta_base). The docstring above
    promises that an open gate reduces the update to V1's psi_analogue, and "open" means the
    gate ignores divergence entirely — which is what lambda = 0 says. Leaving it as
    sigmoid(10) = 0.99995 would have made every V1-V4 quantity reachable through this path
    differ in the fifth decimal for no reason anyone could later reconstruct. An exact
    identity is worth more than a parameterisation that is uniform on paper.
    """
    if float(lam) == 0.0:
        return 1.0
    z = float(theta_base) - float(lam) * value_divergence(goal_posterior, value_prior)
    return float(1.0 / (1.0 + np.exp(-z)))


def gated_update(goal_posterior: np.ndarray, goal_prior: np.ndarray, kappa: float,
                 engaged: bool, value_prior: np.ndarray | None = None,
                 theta_base: float = 10.0, lam: float = 0.0) -> dict:
    """Psi with the theta gate applied, plus the pieces needed to read E29's table.

    At ``lam = 0`` the gate is open regardless of divergence and this reduces to V1's
    ``metrics.psi_analogue`` exactly, which is what keeps every V1-V4 number reachable from
    the V4.5 code path.
    """
    psi = metrics.psi_analogue(goal_posterior, goal_prior, kappa, engaged)
    if value_prior is None:
        return {"psi_raw": psi, "theta": 1.0, "psi_gated": psi, "value_divergence": 0.0}
    div = value_divergence(goal_posterior, value_prior)
    th = theta_gate(goal_posterior, value_prior, theta_base, lam)
    return {"psi_raw": psi, "theta": th, "psi_gated": th * psi, "value_divergence": div}


# --------------------------------------------------------------------------- #
# Assertions.
# --------------------------------------------------------------------------- #
def assert_beta_position(cfg: Config, sigs: FN.V4Signatures, synth: np.ndarray,
                         alpha: np.ndarray, beta_levels: np.ndarray) -> dict:
    """The N-series check V4.5 §7 requires: at beta = 1, V4.5 must reproduce V4.

    "If it does not, beta has been wired into the wrong pipeline position." Checked at
    construction rather than only in a test, because a mis-positioned beta would not fail
    loudly — it would produce a plausible E28 curve for the wrong reason, and E29's whole
    design assumes beta acts on the demonstrator model and not on the update.
    """
    from .v4_model import build_v4_A0

    b1 = [i for i, b in enumerate(beta_levels) if abs(float(b) - 1.0) < 1e-12]
    if not b1:
        return {"beta_1_present": False,
                "note": "no beta = 1 level in the grid, so the V4 boundary is unreachable"}
    a45 = build_v45_A0(cfg, sigs.sig_true, sigs.sig_explore, synth, alpha, beta_levels)
    a4 = build_v4_A0(cfg, sigs.sig_true, synth, alpha)
    delta = float(np.max(np.abs(a45[:, :, :, :, b1[0]] - a4)))
    tol = float(cfg.get("v4_5.beta.v4_boundary_tol", 1e-12))
    assert delta <= tol, (
        f"N-series: at beta = 1 the V4.5 likelihood differs from V4's by {delta:.3e} "
        f"(tolerance {tol}). V4.5 §7: 'at beta = 1 across all conditions, V4.5 must reproduce "
        f"V4 within tolerance. If it does not, beta has been wired into the wrong pipeline "
        f"position.' Check the composition order: beta acts on the CREATOR, inside the alpha "
        f"channel mixture, not beside it.")
    return {"beta_1_present": True, "max_abs_delta_from_v4_A0": delta, "tolerance": tol}


def assert_no_social_term_in_update_path() -> dict:
    """V4.5 §3.2 and §7: social influence enters the prior, never the integration multiplier.

    Made executable by reading the source of the two functions on the update path and
    refusing to let a social term appear in either. This is a crude check and it is meant to
    be: the failure it guards against is not subtle, it is somebody adding a plausible-looking
    ``social`` multiplier to the update because that is where a trust term intuitively wants
    to go. The architecture's whole claim is that it does not go there.
    """
    import inspect
    for fn in (gated_update, theta_gate, value_divergence):
        src = inspect.getsource(fn).lower()
        body = src.split('"""')[-1] if '"""' in src else src
        for token in ("social_shift", "social_weight", "social_prior"):
            assert token not in body, (
                f"V4.5 §3.2/§7: '{token}' appears in the body of {fn.__name__}, which is on "
                f"the update-magnitude path. Social influence shifts P_0(R) and the "
                f"provenance prior BEFORE any gate runs; it is never a multiplier on "
                f"integration. Move it into build_v45_D.")
    return {"checked": ["gated_update", "theta_gate", "value_divergence"], "clean": True}


def load_v4_5_config(quick: bool = False, overrides: dict | None = None) -> Config:
    """A config at V4 cardinality with the four real goals and no EXPLORE (beta subsumes it)."""
    base = load_v4_config(quick=quick, include_explore=False)
    if overrides:
        for k, v in overrides.items():
            base.set(k, v)
    return base
