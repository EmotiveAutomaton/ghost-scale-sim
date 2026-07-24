"""V2 null conditions (V2 spec §3) — mandatory, same discipline as V1's N1-N7.

    N8  — expertise null            -> tests/test_v1_regression.py
    N9  — symmetric control         -> here (added with E6b)
    N10 — learning null             -> here
    N11 — zero-contamination recursion -> here (added with the generation loop)
    N12 — clean-corpus expertise    -> here

Plus the V2 §3 standing invariant: every pA update leaves A column-stochastic.
"""
import numpy as np
import pandas as pd
import pytest

from ghostscale.config import load_config
from ghostscale import generative_model as gmod
from ghostscale import learning as L
from ghostscale import constants as K
from ghostscale.creators import build_creator_bank
from ghostscale.environment import Environment
from ghostscale.observer import make_observer, rollout_observer


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def gm(cfg):
    return gmod.build_shared_model(cfg)


# --------------------------------------------------------------------------- #
# N10 — Learning null.  lr_pA = 0 => the learner must behave IDENTICALLY to a
# fixed-A observer with the same starting A.  Proves learning is doing the work in E7.
# --------------------------------------------------------------------------- #
def test_N10_zero_learning_rate_is_exactly_inert(cfg, gm):
    """The strong form: with lr_pA=0 and an oracle-seeded pA, the learner's A must stay
    bit-identical to the true A through many update_A calls. pymdp supports this exactly,
    so this is asserted at machine precision rather than to a tolerance."""
    D = gmod.build_D(cfg, np.random.default_rng(0))
    agent = L.make_learner_agent(gm, D, cfg, lr_pA=0.0, oracle_A0=True)
    A0_start = np.asarray(agent.A[0]).copy()

    bank = build_creator_bank(cfg, gm)
    world = np.random.default_rng(5)
    env = Environment(cfg, gm, rng_world=world, creator_bank=bank)
    art = env.make_artifact(provenance=K.CREATOR, goal=1, rng=world)
    for _ in range(20):
        agent.infer_policies()
        a = agent.sample_action()
        obs = env.observation(art, int(a[K.F_ATTENTION]), world)
        agent.infer_states(obs)
        L.learn_step(agent, obs, cfg)

    assert np.array_equal(np.asarray(agent.A[0]), A0_start), (
        "lr_pA=0 must leave A[0] bit-identical (N10)")
    assert np.allclose(np.asarray(agent.A[0]), np.asarray(gm.A[0]), atol=1e-12), (
        "an oracle-seeded, non-learning learner IS the fixed-A observer (N10)")


def test_N10_learner_matches_fixed_A_observer_behaviour(cfg, gm):
    """The behavioural form: same starting A, no learning => same actions and same
    posteriors as a plain V1 observer."""
    bank = build_creator_bank(cfg, gm)

    def run(make_agent_fn, seed=11):
        world = np.random.default_rng(seed * 31)
        env = Environment(cfg, gm, rng_world=world, creator_bank=bank)
        r = np.random.default_rng(seed)
        agent = make_agent_fn(r)
        art = env.make_artifact(provenance=K.CREATOR, goal=1, rng=r)
        return rollout_observer(agent, art, env, cfg, np.random.default_rng(seed * 7), 20,
                                early_stop=False)

    fixed = run(lambda r: make_observer(gm, cfg, r))
    learner = run(lambda r: L.make_learner_agent(gm, gmod.build_D(cfg, r), cfg,
                                                 lr_pA=0.0, oracle_A0=True,
                                                 use_param_info_gain=False))
    assert np.array_equal(fixed.attention, learner.attention), (
        "with lr_pA=0 the learner must choose the same actions as a fixed-A observer")
    assert np.allclose(fixed.goal_posterior, learner.goal_posterior, atol=1e-10), (
        "with lr_pA=0 the learner must reach the same posteriors as a fixed-A observer")


def test_N10_guard_learning_is_not_vacuous(cfg, gm):
    """Guard against N10 passing because learning does nothing at all: with lr_pA>0 the
    learner's A MUST move."""
    D = gmod.build_D(cfg, np.random.default_rng(0))
    agent = L.make_learner_agent(gm, D, cfg, lr_pA=1.0, oracle_A0=True)
    A0_start = np.asarray(agent.A[0]).copy()
    bank = build_creator_bank(cfg, gm)
    world = np.random.default_rng(5)
    env = Environment(cfg, gm, rng_world=world, creator_bank=bank)
    art = env.make_artifact(provenance=K.CREATOR, goal=1, rng=world)
    for _ in range(20):
        agent.infer_policies(); a = agent.sample_action()
        obs = env.observation(art, int(a[K.F_ATTENTION]), world)
        agent.infer_states(obs); L.learn_step(agent, obs, cfg)
    assert not np.array_equal(np.asarray(agent.A[0]), A0_start), (
        "lr_pA>0 must actually change A[0], or N10 is vacuous")


# --------------------------------------------------------------------------- #
# V2 §3 standing invariant: every pA update leaves A column-stochastic.
# --------------------------------------------------------------------------- #
def test_pA_update_preserves_column_stochasticity(cfg, gm):
    """Asserted inside ``learn_step`` on every call; this test proves the assertion is
    reachable and holds across a long, mixed-provenance run."""
    D = gmod.build_D(cfg, np.random.default_rng(1))
    agent = L.make_learner_agent(gm, D, cfg)
    bank = build_creator_bank(cfg, gm)
    world = np.random.default_rng(9)
    env = Environment(cfg, gm, rng_world=world, creator_bank=bank)
    for p in (K.CREATOR, K.POLISHED, K.CURATOR, K.GHOST):
        art = env.make_artifact(provenance=p, goal=int(world.integers(4)), rng=world)
        agent.reset()
        for _ in range(10):
            agent.infer_policies(); a = agent.sample_action()
            obs = env.observation(art, int(a[K.F_ATTENTION]), world)
            agent.infer_states(obs)
            L.learn_step(agent, obs, cfg)   # asserts internally
    gmod.assert_A_column_stochastic(agent.A, cfg, where="end of mixed run")


def test_learner_pins_non_learned_modalities(cfg, gm):
    """pymdp's ``reset()`` rebuilds A from pA for EVERY modality. A[1] carries kappa and
    A[2] is the deterministic effort mapping; both must survive resets and learning
    untouched, or the observer's precision would be silently destroyed."""
    D = gmod.build_D(cfg, np.random.default_rng(2))
    agent = L.make_learner_agent(gm, D, cfg)
    bank = build_creator_bank(cfg, gm)
    world = np.random.default_rng(3)
    env = Environment(cfg, gm, rng_world=world, creator_bank=bank)
    art = env.make_artifact(provenance=K.GHOST, goal=0, rng=world)
    for _ in range(15):
        agent.infer_policies(); a = agent.sample_action()
        obs = env.observation(art, int(a[K.F_ATTENTION]), world)
        agent.infer_states(obs); L.learn_step(agent, obs, cfg)
    agent.reset()
    assert np.allclose(np.asarray(agent.A[1]), np.asarray(gm.A[1]), atol=1e-9), \
        "A[1] (kappa) must be pinned through learning and reset"
    assert np.allclose(np.asarray(agent.A[2]), np.asarray(gm.A[2]), atol=1e-9), \
        "A[2] (effort) must be pinned through learning and reset"


def test_learning_survives_reset(cfg, gm):
    """Learning accumulates in pA across artifacts. ``rollout_observer`` resets per
    artifact, so if reset discarded learning, E7/E8/E9 would measure nothing."""
    D = gmod.build_D(cfg, np.random.default_rng(4))
    agent = L.make_learner_agent(gm, D, cfg)
    bank = build_creator_bank(cfg, gm)
    world = np.random.default_rng(6)
    env = Environment(cfg, gm, rng_world=world, creator_bank=bank)
    art = env.make_artifact(provenance=K.CREATOR, goal=2, rng=world)
    for _ in range(10):
        agent.infer_policies(); a = agent.sample_action()
        obs = env.observation(art, int(a[K.F_ATTENTION]), world)
        agent.infer_states(obs); L.learn_step(agent, obs, cfg)
    before = np.asarray(agent.A[0]).copy()
    agent.reset()
    assert np.allclose(before, np.asarray(agent.A[0])), \
        "learning must survive the per-artifact reset"


# --------------------------------------------------------------------------- #
# N9 — Symmetric control.  With goal_symmetric: true, E6b must reproduce V1's E6.
# Proves the new effect comes from the bias axis and nothing else.
# --------------------------------------------------------------------------- #
def test_N9_symmetric_arm_reproduces_v1_regime(cfg, tmp_path):
    """Run E6b's symmetric arm at reduced scale and check it lands in V1's regime rather
    than the biased arm's. V1 measured naive KL = 0.066 at f=0.8 with 2000 artifacts; at
    test scale we require only that the symmetric arm stays small and clearly below its own
    pre-registered bound."""
    from ghostscale.config import load_config
    from ghostscale.experiments import e6b_corpus_biased as e6b

    cfg_q = load_config(quick=True)
    cfg_q.set("experiments.e6b.contamination_sweep", [0.8])
    cfg_q.set("experiments.e6b.kappa_levels", [0.9])
    cfg_q.set("experiments.e6b.signing_rate_levels", [1.0])
    cfg_q.set("experiments.e6b.n_replications", 2)
    cfg_q.set("experiments.e6b.n_artifacts", 400)
    cfg_q.set("experiments.e6b.seed_scan_n", 30)
    e6b.run(cfg_q, out_dir=tmp_path, workers=1, make_fig=False)

    df = pd.read_csv(tmp_path / "e6b_raw.csv")
    sym = df[df.arm == "symmetric"]
    assert len(sym) > 0, "the symmetric control arm must actually be run (N9)"
    # It must stay below its bound and must not blow up: symmetric synth is NOISE.
    assert sym.kl_naive.mean() < sym.bound.mean(), (
        f"symmetric arm KL {sym.kl_naive.mean():.3f} must stay below its bound "
        f"{sym.bound.mean():.3f}")
    assert sym.kl_naive.mean() < 0.5, (
        f"symmetric arm KL {sym.kl_naive.mean():.3f} is far above V1's regime; the control "
        f"is not behaving like V1 and the bias-axis attribution is unsafe")


def test_N9_prereg_is_locked_against_tampering(cfg, tmp_path):
    """The pre-registration mechanism itself (V2 spec §6): a bound edited after the fact
    must be detected, and a differing bound must not silently overwrite the committed one."""
    import json
    from ghostscale.config import load_config
    from ghostscale import preregistration as P

    cfg_q = load_config(quick=True)
    path = tmp_path / "prereg.json"
    payload = P.write_preregistration(cfg_q, path)
    assert P.assert_prereg_locked(path)["content_hash"] == payload["content_hash"]

    # Tampering is detected.
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["selected_draws"][0]["bounds"]["0.8"] = 0.0001
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified since it was written"):
        P.assert_prereg_locked(path)

    # A genuinely different pre-registration refuses to overwrite silently.
    path2 = tmp_path / "prereg2.json"
    P.write_preregistration(cfg_q, path2)
    cfg_b = load_config(quick=True)
    cfg_b.set("experiments.e6b.contamination_sweep", [0.0, 0.9])
    with pytest.raises(RuntimeError, match="DIFFERENT content hash"):
        P.write_preregistration(cfg_b, path2)


# --------------------------------------------------------------------------- #
# N11 — Zero-contamination recursion.  THE most important new null.
# "If generational decay appears without any contamination, the recursion loop itself is
#  lossy and every E8 result is an artifact of the implementation rather than a finding."
# --------------------------------------------------------------------------- #
def test_N11_zero_contamination_recursion_is_not_lossy(cfg):
    """At f=0 the generation chain must not degrade.

    Tested as "no SIGNIFICANT trend" rather than "exactly zero": each generation estimates A
    from finitely many artifacts, so some estimation noise is unavoidable and strict zero is
    not achievable. A structurally lossy loop fails this; a merely finite one does not.
    """
    from ghostscale.config import load_config
    from ghostscale.generations import run_chain, chain_trend
    from ghostscale.preregistration import POP_GOAL_DIST

    cfg_q = load_config(quick=True)
    gm = gmod.build_shared_model(cfg_q, goal_symmetric=False, synth_draw_seed=17)
    results = run_chain(cfg_q, gm, POP_GOAL_DIST, contamination=0.0, signing_rate=1.0,
                        honesty=1.0, g_max=3, n_creators=20, n_artifacts=60,
                        n_observers=2, infer_steps=4, d_i=0.0, base_seed=4242)
    assert len(results) == 3
    trend = chain_trend(results, "kl_payload")
    assert abs(trend["t"]) < 4.0, (
        f"N11: significant payload degradation at f=0 (slope={trend['slope']:.4f}, "
        f"t={trend['t']:.2f}). The recursion loop is lossy and every E8 result would be an "
        f"implementation artifact.")
    kls = [r.kl_payload for r in results]
    assert max(kls) < 0.25, f"N11: f=0 payload KL should stay small; got {kls}"


def test_N11_guard_contamination_does_degrade(cfg):
    """Guard against N11 passing vacuously: at f>0 the chain MUST degrade more than at f=0,
    or the loop transmits nothing and N11 is meaningless."""
    from ghostscale.config import load_config
    from ghostscale.generations import run_chain
    from ghostscale.preregistration import POP_GOAL_DIST

    cfg_q = load_config(quick=True)
    gm = gmod.build_shared_model(cfg_q, goal_symmetric=False, synth_draw_seed=17)
    kw = dict(signing_rate=0.0, honesty=1.0, g_max=3, n_creators=20, n_artifacts=60,
              n_observers=2, infer_steps=4, d_i=0.0, base_seed=4242)
    clean = run_chain(cfg_q, gm, POP_GOAL_DIST, contamination=0.0, **kw)
    dirty = run_chain(cfg_q, gm, POP_GOAL_DIST, contamination=0.8, **kw)
    kl_clean = float(np.mean([r.kl_payload for r in clean]))
    kl_dirty = float(np.mean([r.kl_payload for r in dirty]))
    assert kl_dirty > kl_clean, (
        f"contamination must degrade the payload more than a clean corpus; "
        f"f=0.8 gave {kl_dirty:.4f} vs f=0 {kl_clean:.4f}")


def test_N11_seeded_creator_is_lossless_given_a_perfect_model():
    """The fixed-point property the whole loop rests on (D5): a creator seeded from a
    PERFECTLY learned A reproduces sig_true exactly, so f=0 cannot drift structurally."""
    from ghostscale.config import load_config
    from ghostscale.generations import SeededCreator

    cfg_q = load_config()
    gm = gmod.build_shared_model(cfg_q)
    for g in range(cfg_q.cardinalities.num_goals):
        target = np.asarray(gm.A[0])[:, K.CREATOR, g, K.DEEP]   # a perfect learned column
        creator = SeededCreator(cfg_q, target, g)
        emitted = creator.emission_distribution()
        assert np.allclose(emitted, target, atol=1e-6), (
            f"goal {g}: seeded creator emits {emitted} but should reproduce {target}")


# --------------------------------------------------------------------------- #
# N12 — Clean-corpus expertise.  E10's gradient must persist with zero synthetic content.
# --------------------------------------------------------------------------- #
def test_N12_e10_corpus_contains_no_ghost(cfg, tmp_path):
    """Asserted in E10's worker itself; this test proves the assertion is reachable and that
    a full E10 cell really does run on a corpus with no synthetic content."""
    from ghostscale.config import load_config
    from ghostscale.experiments import e10_expertise as e10

    cfg_q = load_config(quick=True)
    cfg_q.set("experiments.e10.d_sweep", [0.0, 0.9])
    cfg_q.set("experiments.e10.n_replications", 1)
    cfg_q.set("experiments.e10.n_observers", 2)
    cfg_q.set("experiments.e10.n_artifacts", 40)
    e10.run(cfg_q, out_dir=tmp_path, workers=1, make_fig=False)
    df = pd.read_csv(tmp_path / "e10_raw.csv")
    assert (df.n_ghost_in_corpus == 0).all(), "N12: E10 corpus must contain no GHOST artifacts"
    assert set(df.d.unique()) == {0.0, 0.9}


# --------------------------------------------------------------------------- #
# D1 evidence, kept as a live test: the literal uniform prior is unidentifiable.
# --------------------------------------------------------------------------- #
def test_D1_uniform_prior_is_unidentifiable(cfg, gm):
    """The measurement that decided D1, run at reduced scale so it stays in the suite.

    With DEEP forced — disengagement impossible by construction — a uniform-pA[0] learner
    still cannot separate the goal columns: the goal posterior never leaves the prior, so
    each observation deposits ~1/G of a count into every goal column and they converge to a
    common marginal. If this test ever FAILS, the D1 deviation should be revisited.
    """
    cd = gmod.cards(cfg)
    from pymdp.legacy import utils
    from pymdp.legacy.agent import Agent

    pA = utils.obj_array(3)
    pA[0] = np.full((cd.features, cd.provenance, cd.goals, cd.attention), 1.0 / cd.features)
    pA[1] = L.FIXED_SCALE * np.asarray(gm.A[1])
    pA[2] = L.FIXED_SCALE * np.asarray(gm.A[2])
    agent = Agent(A=gm.A, pA=pA, B=gm.B, C=gm.C, D=gmod.build_D(cfg, np.random.default_rng(0)),
                  control_fac_idx=[K.F_ATTENTION], policy_len=2, inference_horizon=1,
                  use_utility=True, use_states_info_gain=True, use_param_info_gain=True,
                  lr_pA=1.0, modalities_to_learn=[0],
                  action_selection="deterministic", gamma=16.0)
    agent.reset()

    bank = build_creator_bank(cfg, gm)
    world = np.random.default_rng(4242)
    env = Environment(cfg, gm, rng_world=world, creator_bank=bank)
    for _ in range(60):
        g = int(world.integers(cd.goals))
        art = env.make_artifact(provenance=K.CREATOR, goal=g, rng=world)
        agent.reset()
        for _ in range(6):
            agent.infer_policies()
            action = np.zeros(len(agent.num_controls)); action[K.F_ATTENTION] = K.DEEP
            agent.action = action                      # FORCE DEEP
            obs = env.observation(art, K.DEEP, world)
            agent.infer_states(obs); L.learn_step(agent, obs, cfg)

    cols = np.asarray(agent.A[0])[:, K.CREATOR, :, K.DEEP]
    spread = float(np.max(np.abs(cols - cols.mean(axis=1, keepdims=True))))
    mi = L.human_column_mi(agent.A[0], K.CREATOR)
    assert spread < 1e-9, f"uniform-prior goal columns should stay identical; spread={spread:.2e}"
    assert mi < 1e-9, f"uniform-prior MI should stay at zero; got {mi:.6f}"

    # ...and the D1 seeding does NOT have this problem.
    learner = L.make_learner_agent(gm, gmod.build_D(cfg, np.random.default_rng(0)), cfg)
    world2 = np.random.default_rng(4242)
    env2 = Environment(cfg, gm, rng_world=world2, creator_bank=bank)
    for _ in range(60):
        g = int(world2.integers(cd.goals))
        art = env2.make_artifact(provenance=K.CREATOR, goal=g, rng=world2)
        learner.reset()
        for _ in range(6):
            learner.infer_policies()
            action = np.zeros(len(learner.num_controls)); action[K.F_ATTENTION] = K.DEEP
            learner.action = action
            obs = env2.observation(art, K.DEEP, world2)
            learner.infer_states(obs); L.learn_step(learner, obs, cfg)
    mi_d1 = L.human_column_mi(learner.A[0], K.CREATOR)
    assert mi_d1 > 0.5, f"D1-seeded learner must retain goal-discriminability; MI={mi_d1:.3f}"
