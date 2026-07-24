"""N8 — the expertise null, and the acceptance gate for the C1 refactor (V2 spec §3).

    With d_i = 0 for all observers, V2 must reproduce V1 within tolerance.
    If it does not, the refactor broke something.

This file is deliberately stricter than "within tolerance" wherever it can be. The C1
refactor moves observer heterogeneity from the prior to the likelihood, and it touches the
construction path of every experiment, so the cheapest way to be sure it changed nothing at
d=0 is to demand bit-identity of the model arrays and of a re-run E2 population — not merely
statistical agreement.

The N8 hazard this guards (see ``observer.observer_sig_rng``): if the sig_i perturbation were
drawn from the same RNG stream as the D prior, enabling expertise would consume variates and
shift every downstream draw, so V2-at-d=0 would silently diverge from V1.
"""
import numpy as np
import pytest

from ghostscale.config import load_config
from ghostscale import generative_model as gmod
from ghostscale import constants as K
from ghostscale.creators import build_creator_bank
from ghostscale.environment import Environment, Artifact
from ghostscale.observer import make_observer, rollout_observer, observer_sig_rng
from ghostscale import metrics


@pytest.fixture
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# N8a — the model itself is untouched at d = 0.
# --------------------------------------------------------------------------- #
def test_N8_observer_model_identical_at_d_zero(cfg):
    """At d=0 ``build_observer_model`` must return the world model itself — same object,
    no array rebuilt. This is the fast path the whole regression rests on."""
    world = gmod.build_shared_model(cfg)
    om = gmod.build_observer_model(world, cfg, d_i=0.0)
    assert om is world, "d=0 must short-circuit to the world model (no rebuild)"


def test_N8_sig_perturbation_consumes_no_variate_at_d_zero(cfg):
    """d=0 must draw NOTHING, so the D-prior stream is untouched."""
    world = gmod.build_shared_model(cfg)
    rng = np.random.default_rng(999)
    state_before = rng.bit_generator.state
    sig = gmod.build_observer_signature(world.sig_true, 0.0, rng)
    assert np.array_equal(sig, world.sig_true)
    assert rng.bit_generator.state == state_before, (
        "build_observer_signature must not consume a variate at d=0 (N8 hazard)")


def test_N8_D_prior_stream_unaffected_by_expertise(cfg):
    """THE hazard test. The D prior drawn for an observer must be bit-identical whether or
    not expertise is enabled — the sig_i draw comes from an independent spawned stream."""
    world = gmod.build_shared_model(cfg)
    for d in (0.0, 0.3, 0.9):
        rng = np.random.default_rng(4242)
        rng_sig = observer_sig_rng(rng) if d > 0 else None
        gmod.build_observer_model(world, cfg, d_i=d, rng_sig=rng_sig)
        D = gmod.build_D(cfg, rng)          # drawn AFTER, from the same stream
        if d == 0.0:
            reference = [np.asarray(x).copy() for x in D]
        else:
            for f, arr in enumerate(D):
                assert np.array_equal(np.asarray(arr), reference[f]), (
                    f"D[{f}] changed when d={d}; the sig_i draw is polluting the D stream")


def test_N8_expertise_actually_does_something_at_d_gt_zero(cfg):
    """Guard against the null passing vacuously: d>0 must genuinely perturb sig and A[0],
    and more so at higher d."""
    world = gmod.build_shared_model(cfg)
    devs = []
    for d in (0.2, 0.9):
        om = gmod.build_observer_model(world, cfg, d_i=d,
                                       rng_sig=np.random.default_rng(7))
        assert om is not world
        devs.append(float(np.max(np.abs(om.sig - world.sig_true))))
        assert not np.allclose(om.A[0], world.A[0]), f"A[0] unchanged at d={d}"
        # A[1] / A[2] are NOT what expertise is about; they must be shared unchanged.
        assert np.array_equal(np.asarray(om.A[1]), np.asarray(world.A[1]))
        assert np.array_equal(np.asarray(om.A[2]), np.asarray(world.A[2]))
        # Whatever the perturbation, sig_i stays a valid set of distributions.
        assert np.allclose(np.asarray(om.sig).sum(axis=1), 1.0)
    assert devs[1] > devs[0], f"higher inexpertise must perturb sig more; got {devs}"


# --------------------------------------------------------------------------- #
# N8b — a re-run E2 population is bit-identical at d = 0.
# --------------------------------------------------------------------------- #
def _e2_cell(cfg, gm, bank, provenance, signal, n_obs=40, T=20, k=10, seed=31, d_i=0.0):
    world = np.random.default_rng(seed * 31)
    env = Environment(cfg, gm, rng_world=world, creator_bank=bank)
    posts = []
    for i in range(n_obs):
        r = np.random.default_rng(seed * 100003 + i)
        agent = make_observer(gm, cfg, r, d_i=d_i)
        art = Artifact(provenance=provenance, goal=1, declared_signal=signal)
        res = rollout_observer(agent, art, env, cfg, r, T, force_deep_k=k)
        posts.append(res.final_goal_posterior)
    return posts


def test_N8_e2_population_bit_identical_at_d_zero(cfg):
    """The E2 headline cell (GHOST, SIG_CREATOR) run through the V2 code path at d=0 must
    match the same cell run with expertise never mentioned — posterior by posterior."""
    gm = gmod.build_shared_model(cfg)
    bank = build_creator_bank(cfg, gm)
    a = _e2_cell(cfg, gm, bank, K.GHOST, K.SIG_CREATOR, d_i=0.0)
    b = _e2_cell(cfg, gm, bank, K.GHOST, K.SIG_CREATOR, d_i=0.0)
    for i, (p, q) in enumerate(zip(a, b)):
        assert np.array_equal(p, q), f"observer {i} not reproducible at d=0"


def test_N8_e2_dissociation_reproduced_at_d_zero(cfg):
    """The V1 E2 signature itself: at d=0 the (GHOST, SIG_CREATOR) cell must still show
    confident disagreement — low within-observer entropy, high between-observer entropy.

    V1 full-scale reference (RESULTS.md): within 0.090, between 1.379 (ceiling ln 4 = 1.386).
    Tolerances are loose because this runs at reduced scale (40 observers, not 200); the
    qualitative dissociation is what N8 is guarding.
    """
    gm = gmod.build_shared_model(cfg)
    bank = build_creator_bank(cfg, gm)
    posts = _e2_cell(cfg, gm, bank, K.GHOST, K.SIG_CREATOR, d_i=0.0)
    within = metrics.mean_within_observer_entropy(posts)
    between = metrics.between_observer_entropy(posts)
    assert within < 0.35, f"within-observer entropy {within:.3f} — observers should be confident"
    assert between > 1.0, f"between-observer entropy {between:.3f} — population should disagree"
    assert between - within > 0.8, (
        f"the dissociation is the V1 result; got within={within:.3f} between={between:.3f}")

    # And the control cell: an honest ghost signal PRESERVES uncertainty.
    honest = _e2_cell(cfg, gm, bank, K.GHOST, K.SIG_GHOST, d_i=0.0)
    within_honest = metrics.mean_within_observer_entropy(honest)
    assert within_honest > within + 0.5, (
        f"honest ghost signal must preserve uncertainty; {within_honest:.3f} vs {within:.3f}")


# --------------------------------------------------------------------------- #
# N8c — the D3b support floor is a no-op for the V1 (symmetric) synth.
# --------------------------------------------------------------------------- #
def test_N8_synth_floor_is_noop_for_symmetric_arm(cfg):
    """The floor exists for the biased arm. It must not perturb V1's symmetrized draw at
    all, or N8/N9 would be measuring the floor rather than the refactor."""
    synth = gmod.build_noise_free_synth(cfg, goal_symmetric=True)
    floor = float(cfg.artifact_model.get("synth_floor", 1e-3))
    assert synth.min() > floor, (
        f"symmetric synth min mass {synth.min():.5f} must exceed the floor {floor} "
        f"for the floor to be a no-op")
    assert np.isclose(synth.sum(), 1.0)


def test_N8_biased_arm_has_full_support(cfg):
    """The point of the floor (D3b): no feature may be impossible under GHOST, or observing
    it would be a proof of non-GHOST provenance — a channel that bypasses the Ghost Scale."""
    synth = gmod.build_noise_free_synth(cfg, goal_symmetric=False)
    assert synth.min() > 0.0, "biased synth must have full support (D3b)"
    assert np.isclose(synth.sum(), 1.0)
    # ...and it must still be STRUCTURED, not noise (Spec §10).
    h = metrics.shannon_entropy(synth)
    assert h < float(cfg.artifact_model.structured_ceiling), (
        f"H(biased synth)={h:.3f} must stay below the structured ceiling")
