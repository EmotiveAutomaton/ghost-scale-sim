"""Tests for the exact-inference observer and the validation pass's own machinery.

These are not experiments and they make no claim about the world. They exist because V-1's whole
argument is "the only difference between the two arms is the factorisation", and that argument is
worth exactly as much as the evidence that the exact agent is correct. If ``exact.py`` were subtly
wrong, V-1 would report a difference and the difference would be the bug.

The strategy is the one the codebase already uses for its boundary regressions: find the cases
where the answer is known independently, and check those.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale import constants as K
from ghostscale.config import load_config
from ghostscale.environment import Artifact, Environment
from ghostscale.exact import (ExactAgent, assert_agrees_on_factorised_case, make_exact_agent)
from ghostscale.generative_model import (build_D, build_shared_model, make_agent)
from ghostscale.observer import rollout_observer


def test_exact_matches_mean_field_where_mean_field_is_exact():
    """The one construction in this model family where the joint posterior factorises.

    With alpha = 1 for every tier, A[0] depends on the goal and not on provenance, A[1] depends on
    provenance and not on the goal, and A[2] pins attention. No modality couples two uncertain
    factors, so mean field is exact and the two agents MUST agree. A failure here is a bug in
    exact.py rather than a finding about the model, and V-1 runs this same check before it compares
    anything.
    """
    got = assert_agrees_on_factorised_case()
    assert got["max_goal_discrepancy"] < 1e-9
    assert got["max_provenance_discrepancy"] < 1e-9


def test_exact_posterior_is_a_distribution_over_the_joint():
    cfg = load_config()
    gm = build_shared_model(cfg)
    rng = np.random.default_rng(3)
    agent = make_exact_agent(gm, build_D(cfg, rng), cfg)
    assert isinstance(agent, ExactAgent)
    assert np.isclose(agent.qs_joint.sum(), 1.0)
    assert agent.qs_joint.size == np.prod(agent.num_states)

    env = Environment(cfg, gm, np.random.default_rng(4), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.GHOST, goal=2, declared_signal=K.SIG_CREATOR)
    obs_rng = np.random.default_rng(5)
    for _ in range(6):
        agent.infer_states(env.observation(art, K.DEEP, obs_rng))
        assert np.isclose(agent.qs_joint.sum(), 1.0)
        assert np.all(agent.qs_joint >= 0.0)
        for f in range(agent.num_factors):
            assert np.isclose(np.asarray(agent.qs[f]).sum(), 1.0)


def test_marginals_are_consistent_with_the_joint():
    """The reported marginals must be marginals OF the joint, not a separate belief.

    This is the property that makes the exact agent exact. If the two ever came apart, the agent
    would be reporting one thing and inferring with another, which is the failure it exists to
    remove.
    """
    cfg = load_config()
    gm = build_shared_model(cfg)
    rng = np.random.default_rng(11)
    agent = make_exact_agent(gm, build_D(cfg, rng), cfg)
    env = Environment(cfg, gm, np.random.default_rng(12), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.CURATOR, goal=0, declared_signal=K.SIG_CURATOR)
    obs_rng = np.random.default_rng(13)
    for _ in range(4):
        agent.infer_states(env.observation(art, K.DEEP, obs_rng))
    cube = agent.qs_joint.reshape(agent.num_states)
    assert np.allclose(np.asarray(agent.qs[K.F_GOAL]), cube.sum(axis=(0, 2)))
    assert np.allclose(np.asarray(agent.qs[K.F_PROVENANCE]), cube.sum(axis=(1, 2)))


def test_effort_observation_pins_attention_exactly():
    """A[2] is deterministic, so the attention marginal must be a point mass on what was chosen.

    True under either solver, and it is the cheapest check that the joint update is wired to the
    right axes: get the reshape order wrong and this is the first thing that breaks.
    """
    cfg = load_config()
    gm = build_shared_model(cfg)
    rng = np.random.default_rng(21)
    agent = make_exact_agent(gm, build_D(cfg, rng), cfg)
    env = Environment(cfg, gm, np.random.default_rng(22), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.CREATOR, goal=1, declared_signal=K.SIG_CREATOR)
    agent.infer_states(env.observation(art, K.DEEP, np.random.default_rng(23)))
    assert np.isclose(np.asarray(agent.qs[K.F_ATTENTION])[K.DEEP], 1.0, atol=1e-12)


def test_a_confident_true_label_recovers_the_true_provenance():
    """A sanity check with an answer known from outside the model.

    With a perfectly precise provenance channel and an honest label, the reader's provenance
    posterior must go to the truth. It is the one place in this model where the correct answer is
    known without running anything, and any indexing error in the joint update fails it.
    """
    cfg = load_config()
    cfg.set("signal_model.kappa", 1.0)
    gm = build_shared_model(cfg, kappa=1.0)
    rng = np.random.default_rng(31)
    agent = make_exact_agent(gm, build_D(cfg, rng), cfg)
    env = Environment(cfg, gm, np.random.default_rng(32), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.CURATOR, goal=3,
                   declared_signal=K.TRUTHFUL_SIGNAL[K.CURATOR])
    agent.infer_states(env.observation(art, K.DEEP, np.random.default_rng(33)))
    assert np.argmax(np.asarray(agent.qs[K.F_PROVENANCE])) == K.CURATOR
    assert np.asarray(agent.qs[K.F_PROVENANCE])[K.CURATOR] > 0.99


def test_the_solver_switch_is_off_by_default_and_reachable():
    """``inference.exact`` must default to false, or every committed number changes meaning."""
    cfg = load_config()
    assert cfg.get("inference.exact", None) is False
    gm = build_shared_model(cfg)
    rng = np.random.default_rng(41)
    D = build_D(cfg, rng)
    assert not isinstance(make_agent(gm, D, cfg), ExactAgent)
    cfg.set("inference.exact", True)
    assert isinstance(make_agent(gm, D, cfg), ExactAgent)


def test_exact_agent_completes_a_rollout_through_the_shared_loop():
    """The exact agent must work through ``rollout_observer`` unmodified.

    V-1's argument depends on the experiments running their own code with one substitution, so the
    agent has to satisfy the rollout loop's whole interface — reset, policies, action, early stop —
    and not just the inference calls.
    """
    cfg = load_config()
    gm = build_shared_model(cfg)
    rng = np.random.default_rng(51)
    cfg.set("inference.exact", True)
    agent = make_agent(gm, build_D(cfg, rng), cfg)
    env = Environment(cfg, gm, np.random.default_rng(52), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.GHOST, goal=1, declared_signal=K.SIG_CREATOR)
    res = rollout_observer(agent, art, env, cfg, rng, n_timesteps=12, force_deep_k=4)
    assert res.goal_posterior.shape == (12, int(cfg.cardinalities.num_goals))
    assert np.allclose(res.goal_posterior.sum(axis=1), 1.0)
    assert res.cum_deep >= 4


def test_learning_is_refused_rather_than_approximated():
    """A half-exact agent that quietly falls back is worse than no exact agent at all."""
    cfg = load_config()
    gm = build_shared_model(cfg)
    agent = make_exact_agent(gm, build_D(cfg, np.random.default_rng(61)), cfg)
    with pytest.raises(NotImplementedError):
        agent.update_A()


def test_validation_criteria_are_hash_locked(tmp_path):
    """A pre-registration that can be edited without detection is not one."""
    import json

    from ghostscale.validation import criteria as CR
    path = tmp_path / "criteria.json"
    written = CR.write_criteria(load_config(), path)
    assert CR.ensure_criteria(load_config(), path)["content_hash"] == written["content_hash"]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["v1"]["spread_standard_errors"] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified since it was written"):
        CR.ensure_criteria(load_config(), path)


def test_the_independent_reimplementation_imports_nothing_from_the_package():
    """V-8's entire value rests on this, so it is checked mechanically rather than promised."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "independent_two_gates.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "ghostscale" not in stripped, (
                f"scripts/independent_two_gates.py imports from the package it is supposed to be "
                f"independent of: {stripped!r}. V-8 is worthless if this is allowed.")
            assert "pymdp" not in stripped, (
                f"the reimplementation must not use the original's inference engine: {stripped!r}")
