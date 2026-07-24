"""V2 null conditions (V2 spec §3) — mandatory, same discipline as V1's N1-N7.

    N8  — expertise null            -> tests/test_v1_regression.py
    N9  — symmetric control         -> here (added with E6b)
    N10 — learning null             -> here
    N11 — zero-contamination recursion -> here (added with the generation loop)
    N12 — clean-corpus expertise    -> here

Plus the V2 §3 standing invariant: every pA update leaves A column-stochastic.
"""
import numpy as np
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
