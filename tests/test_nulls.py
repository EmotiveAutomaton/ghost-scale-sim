"""Null conditions (Spec §9) — mandatory.

Each test runs a reduced version of the relevant experiment and asserts the effect is
ABSENT. A null that fails to be null is a finding: it would surface here as a failing test
and be recorded in RESULTS.md rather than tuned away.

Reduced scale (small populations / short horizons) keeps the suite fast; the effects these
guard against are qualitative and appear well before full scale.
"""
import numpy as np
import pytest

from ghostscale.config import load_config
from ghostscale import generative_model as gmod
from ghostscale.creators import build_creator_bank
from ghostscale.environment import Environment, Artifact
from ghostscale.observer import make_observer, rollout_observer
from ghostscale import metrics, constants as K


# --------------------------------------------------------------------------- #
# Small reusable runners (direct, not via the CSV-writing experiment modules).
# --------------------------------------------------------------------------- #
def _run_tier_population(cfg, gm, bank, provenance, n_obs=40, T=25, seed=7,
                         honesty=1.0, signing_rate=1.0, kappa=None, force_deep_k=0,
                         signal=None, true_goal=1):
    kappa = float(cfg.signal_model.kappa) if kappa is None else kappa
    world = np.random.default_rng(seed * 31)
    env = Environment(cfg, gm, rng_world=world, honesty=honesty,
                      signing_rate=signing_rate, creator_bank=bank)
    cum_deep, final_ent, posts = [], [], []
    for i in range(n_obs):
        r = np.random.default_rng(seed * 100003 + i)
        agent = make_observer(gm, cfg, r, kappa=kappa)
        if signal is None:
            artifact = env.make_artifact(provenance=provenance, goal=true_goal, rng=r)
        else:
            artifact = Artifact(provenance=provenance, goal=true_goal, declared_signal=signal)
        res = rollout_observer(agent, artifact, env, cfg, r, T,
                               force_deep_k=force_deep_k, kappa=kappa)
        cum_deep.append(res.cum_deep)
        final_ent.append(float(res.within_entropy[-1]))
        posts.append(res.final_goal_posterior)
    return {"cum_deep": float(np.mean(cum_deep)),
            "final_ent": float(np.mean(final_ent)),
            "between": metrics.between_observer_entropy(posts),
            "within": metrics.mean_within_observer_entropy(posts)}


# --------------------------------------------------------------------------- #
# N1 — Intent restoration.  alpha[GHOST]=1.0 => the crash must vanish.
# --------------------------------------------------------------------------- #
def test_N1_intent_restoration():
    cfg = load_config()
    gm = gmod.build_shared_model(cfg, alpha_overrides={"GHOST": 1.0})
    bank = build_creator_bank(cfg, gm)
    # Environment must also carry the restored intent (true world alpha).
    def pop(p):
        world = np.random.default_rng(11 * 31)
        env = Environment(cfg, gm, rng_world=world, alpha_overrides={"GHOST": 1.0},
                          creator_bank=bank)
        ents, deeps = [], []
        for i in range(40):
            r = np.random.default_rng(11 * 100003 + i)
            ag = make_observer(gm, cfg, r)
            art = env.make_artifact(provenance=p, goal=1, rng=r)
            res = rollout_observer(ag, art, env, cfg, r, 25)
            ents.append(float(res.within_entropy[-1])); deeps.append(res.cum_deep)
        return np.mean(ents), np.mean(deeps)
    ghost_ent, ghost_deep = pop(K.GHOST)
    creator_ent, creator_deep = pop(K.CREATOR)
    # With full intent restored, GHOST recovers the goal like CREATOR: low, comparable entropy.
    assert ghost_ent < 0.4, f"restored-GHOST entropy {ghost_ent:.3f} should be low (crash gone)"
    assert abs(ghost_ent - creator_ent) < 0.3, (
        f"restored-GHOST entropy {ghost_ent:.3f} should match CREATOR {creator_ent:.3f}")
    assert ghost_deep > 0.5, "restored-GHOST should engage DEEP (it now carries intent)"


# --------------------------------------------------------------------------- #
# N2 — Free cognition.  c_effort=0 => disengagement must vanish across all tiers.
# --------------------------------------------------------------------------- #
def test_N2_free_cognition():
    T = 20
    # Baseline (with effort): GHOST selectively disengages (the crash).
    cfg_base = load_config()
    gm_b = gmod.build_shared_model(cfg_base)
    bank_b = build_creator_bank(cfg_base, gm_b)
    ghost_base = _run_tier_population(cfg_base, gm_b, bank_b, K.GHOST, n_obs=30, T=T)
    assert ghost_base["cum_deep"] < 1.0, "sanity: GHOST crashes under the default effort cost"

    # Free cognition: with c_effort=0 the metabolic trade-off is removed, so the selective
    # disengagement from GHOST vanishes — GHOST no longer crashes (it engages fully).
    cfg = load_config(overrides={"preferences.c_effort": 0.0})
    gm = gmod.build_shared_model(cfg)
    bank = build_creator_bank(cfg, gm)
    ghost = _run_tier_population(cfg, gm, bank, K.GHOST, n_obs=30, T=T)
    assert ghost["cum_deep"] >= 0.7 * T, (
        f"with c_effort=0 GHOST disengagement must vanish; cum_deep={ghost['cum_deep']:.1f} "
        f"(crash is a metabolic trade-off, not a distaste for synthetic content)")


# --------------------------------------------------------------------------- #
# N3 — Homogeneous observers.  eps=0 + shared seed => between-observer disagreement ~0
# in EVERY E2 cell, including (GHOST, SIG_CREATOR).
# --------------------------------------------------------------------------- #
def test_N3_homogeneous_observers():
    cfg = load_config(overrides={"priors.eps": 0.0, "priors.perturb_provenance": False})
    gm = gmod.build_shared_model(cfg)
    bank = build_creator_bank(cfg, gm)
    world = np.random.default_rng(3 * 31)
    env = Environment(cfg, gm, rng_world=world, creator_bank=bank)
    SHARED = 123456  # one RNG seed shared across observers
    for prov, sig in [(K.GHOST, K.SIG_CREATOR), (K.GHOST, K.SIG_GHOST), (K.CREATOR, K.SIG_CREATOR)]:
        posts = []
        for i in range(30):
            r = np.random.default_rng(SHARED)  # identical seed => identical observers
            ag = make_observer(gm, cfg, r)
            art = Artifact(provenance=prov, goal=1, declared_signal=sig)
            res = rollout_observer(ag, art, env, cfg, np.random.default_rng(SHARED), 20,
                                   force_deep_k=10)
            posts.append(res.final_goal_posterior)
        between = metrics.between_observer_entropy(posts)
        assert between < 1e-6, (
            f"cell ({K.PROVENANCE_NAMES[prov]},{K.SIGNAL_NAMES[sig]}) between={between:.4f}; "
            f"homogeneous observers must produce ~zero disagreement (H2 needs heterogeneity)")


# --------------------------------------------------------------------------- #
# N4 — Label shuffle.  Permuting provenance->alpha destroys the tier ordering of
# recoverability: MI is no longer monotonic in CREATOR>POLISHED>CURATOR>GHOST.
# --------------------------------------------------------------------------- #
def test_N4_label_shuffle():
    cfg = load_config()
    base = gmod.build_shared_model(cfg)
    mi_base = [metrics.mutual_information_features_goal(base.A[0], p, K.DEEP) for p in range(4)]
    assert all(mi_base[i] >= mi_base[i + 1] - 1e-9 for i in range(3)), "baseline must be monotonic"

    # A non-identity permutation of the provenance->alpha mapping.
    perm = [3, 1, 2, 0]  # swap CREATOR<->GHOST alpha assignment
    shuffled = gmod.build_shared_model(cfg, alpha_permutation=perm, run_assertions=False)
    mi_shuf = [metrics.mutual_information_features_goal(shuffled.A[0], p, K.DEEP) for p in range(4)]
    monotonic = all(mi_shuf[i] >= mi_shuf[i + 1] - 1e-9 for i in range(3))
    assert not monotonic, (
        f"permuted alpha must break the tier-ordered recoverability effect; MI={mi_shuf}")


# --------------------------------------------------------------------------- #
# N5 — Signal blackout.  signing_rate=0 => calibrated (kappa=0.9) and naive (kappa=0.1)
# observers must behave identically (the Ghost Scale effect requires the Ghost Scale).
# --------------------------------------------------------------------------- #
def test_N5_signal_blackout():
    cfg = load_config()
    gm = gmod.build_shared_model(cfg)
    bank = build_creator_bank(cfg, gm)

    def selectivity(kappa):
        # P(engage) difference between CREATOR and GHOST content, with NO signal emitted.
        creator = _run_tier_population(cfg, gm, bank, K.CREATOR, n_obs=30, T=20,
                                       signing_rate=0.0, kappa=kappa)
        ghost = _run_tier_population(cfg, gm, bank, K.GHOST, n_obs=30, T=20,
                                     signing_rate=0.0, kappa=kappa)
        return creator["cum_deep"], ghost["cum_deep"]

    cal_c, cal_g = selectivity(0.9)
    naive_c, naive_g = selectivity(0.1)
    # With no signal, kappa is inert: calibrated and naive engagement must match.
    assert abs(cal_c - naive_c) < 1.0 and abs(cal_g - naive_g) < 1.0, (
        f"blackout: calibrated {cal_c:.1f}/{cal_g:.1f} vs naive {naive_c:.1f}/{naive_g:.1f} "
        f"must coincide when no signal is emitted")


# --------------------------------------------------------------------------- #
# N6 — The noise strawman, run deliberately.  Uniform synth => crash still occurs, but
# the MI/entropy diagnostic separates structured-unidentifiable from uniform-noise.
# --------------------------------------------------------------------------- #
def test_N6_noise_strawman_diagnostic():
    cfg = load_config()
    structured = gmod.build_shared_model(cfg)                       # default
    noise = gmod.build_shared_model(cfg, uniform_synth=True)        # strawman

    mi_struct = metrics.mutual_information_features_goal(structured.A[0], K.GHOST, K.DEEP)
    mi_noise = metrics.mutual_information_features_goal(noise.A[0], K.GHOST, K.DEEP)
    h_struct = metrics.feature_entropy_given_provenance(structured.A[0], K.GHOST, K.DEEP)
    h_noise = metrics.feature_entropy_given_provenance(noise.A[0], K.GHOST, K.DEEP)

    # Both are unidentifiable (low MI)...
    assert mi_struct < 0.1 and mi_noise < 0.1, f"both should be low MI: {mi_struct:.3f}, {mi_noise:.3f}"
    # ...but only the strawman is high-entropy. The diagnostic separates them.
    assert h_struct < float(cfg.artifact_model.structured_ceiling) < h_noise, (
        f"diagnostic must separate: H_struct={h_struct:.3f} < ceiling "
        f"{cfg.artifact_model.structured_ceiling} < H_noise={h_noise:.3f}")

    # And the crash still happens under the strawman (GHOST still disengages / fails to recover).
    bank = build_creator_bank(cfg, noise)
    out = _run_tier_population(cfg, noise, bank, K.GHOST, n_obs=30, T=20)
    assert out["final_ent"] > 1.0, f"crash should persist under strawman; final_ent={out['final_ent']:.3f}"


# --------------------------------------------------------------------------- #
# N7 — Preference leakage audit.  C[0] and C[1] are exactly zero at construction.
# --------------------------------------------------------------------------- #
def test_N7_preference_leakage():
    cfg = load_config()
    gm = gmod.build_shared_model(cfg)   # build_shared_model asserts this internally too
    gmod.assert_preferences_zero(gm.C)
    assert np.all(np.asarray(gm.C[0]) == 0.0)
    assert np.all(np.asarray(gm.C[1]) == 0.0)
