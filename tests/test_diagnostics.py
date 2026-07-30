"""Tests for the diagnostics machinery, and regressions for the two bugs it caught in itself.

The two regressions are the point of this file. P-1's first run reported a parameter as perfectly
recovered when its estimator returned the same number at every grid point, and reported a second
parameter as unidentifiable when its estimator was exact. Both were defects in the SCORING rather
than in the model, both produced entirely plausible output, and both are the class of failure this
project has been bitten by repeatedly. They are pinned here so they cannot come back quietly.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale import constants as K
from ghostscale.config import load_config
from ghostscale.diagnostics import criteria as CR


# --------------------------------------------------------------------------- #
# Regression: rank correlation on tied data.
# --------------------------------------------------------------------------- #
def test_spearman_returns_nan_for_a_constant_estimator():
    """A CONSTANT estimate against an ascending truth must not score as perfect recovery.

    `argsort(argsort(v))` assigns distinct ranks to tied values in arrival order, so a constant
    vector ranks 0, 1, 2, ... and correlates perfectly with anything ascending. P-1's first run hit
    exactly this: trust's estimator returned the same value at all nine grid points and scored a rank
    correlation of 1.00, which reads as flawless recovery and means the opposite.
    """
    assert np.isnan(CR.spearman([1, 2, 3, 4, 5], [7, 7, 7, 7, 7]))
    assert np.isnan(CR.spearman([7, 7, 7, 7], [1, 2, 3, 4]))


def test_spearman_handles_ties_and_matches_the_textbook_values():
    assert CR.spearman([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert CR.spearman([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    # One tie, mid-ranked. Value cross-checked against scipy.stats.spearmanr.
    assert CR.spearman([1, 2, 3, 4], [1, 2, 2, 4]) == pytest.approx(0.948683, abs=1e-5)
    assert CR.spearman([1, 2, 3, 4], [3, 1, 4, 2]) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Regression: usable range with a zero standard error.
# --------------------------------------------------------------------------- #
def test_noiseless_recovery_counts_as_distinguishable():
    """A step measured with NO uncertainty is the most distinguishable case, not an excluded one.

    The first version required a strictly positive pooled standard error, so a noiselessly recovered
    parameter scored a usable range of zero and was classified unidentifiable for being too precise.
    The value gate is exact over half its range and was misclassified on exactly this.
    """
    frac, flags = CR.usable_range_fraction([0, 1, 2, 3], [0, 1, 2, 3], [0, 0, 0, 0])
    assert frac == pytest.approx(1.0)
    assert all(f["distinguishable"] for f in flags)


def test_noiseless_but_constant_is_not_distinguishable():
    """The other half of the same branch: zero error AND zero gap is genuinely indistinguishable."""
    frac, flags = CR.usable_range_fraction([0, 1, 2, 3], [5, 5, 5, 5], [0, 0, 0, 0])
    assert frac == pytest.approx(0.0)
    assert not any(f["distinguishable"] for f in flags)


def test_usable_range_charges_a_flat_region_its_own_width():
    frac, _ = CR.usable_range_fraction([0, 1, 2, 3], [0, 1, 1, 1], [0, 0, 0, 0])
    assert frac == pytest.approx(1.0 / 3.0)


def test_classification_is_applied_mechanically():
    assert CR.classify_recovery(1.0, 1.0, 1.0) == "RECOVERED"
    assert CR.classify_recovery(1.0, 0.3, 1.0) == "COMPRESSED"
    assert CR.classify_recovery(float("nan"), 0.0, 0.0) == "UNIDENTIFIABLE"
    assert CR.classify_recovery(0.5, 1.0, 1.0) == "UNIDENTIFIABLE"
    assert CR.classify_recovery(1.0, 1.0, 0.5) == "PARTIALLY_RECOVERED"


def test_diagnostics_criteria_are_hash_locked(tmp_path):
    import json
    path = tmp_path / "criteria.json"
    written = CR.write_criteria(load_config(), path)
    assert CR.ensure_criteria(load_config(), path)["content_hash"] == written["content_hash"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["p1"]["min_grid_points"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified since it was written"):
        CR.ensure_criteria(load_config(), path)


# --------------------------------------------------------------------------- #
# D-4's fix: the exact agent's own expected-free-energy accessors.
# --------------------------------------------------------------------------- #
def _fresh_exact_agent():
    from ghostscale.exact import make_exact_agent
    from ghostscale.generative_model import build_D, build_shared_model
    cfg = load_config()
    cfg.set("inference.exact", True)
    gm = build_shared_model(cfg)
    agent = make_exact_agent(gm, build_D(cfg, np.random.default_rng(1)), cfg)
    agent.reset()
    return cfg, gm, agent


def test_pymdp_efe_helpers_run_on_the_exact_agent():
    """`policy_efe_terms` raised AttributeError before B became an object array.

    It has to run, because the alternative is that anybody reaching for the shipped helper on an
    exact agent gets a crash in the middle of an experiment rather than a number.
    """
    from ghostscale import metrics
    from ghostscale.observer import find_named_policies
    _, _, agent = _fresh_exact_agent()
    deep, _ = find_named_policies(agent)
    prag, epi = metrics.policy_efe_terms(agent, deep)
    assert np.isfinite(prag) and np.isfinite(epi)


def test_exact_and_meanfield_efe_agree_before_any_evidence():
    """At t = 0 the joint IS a product of its marginals, so the two estimators must agree exactly.

    This is the anchor for the drift measurement: if they disagreed here, the difference measured
    later would not be attributable to the factorisation.
    """
    from ghostscale import metrics
    from ghostscale.observer import find_named_policies
    _, _, agent = _fresh_exact_agent()
    deep, _ = find_named_policies(agent)
    mf = metrics.policy_efe_terms(agent, deep)
    ex = agent.efe_terms(deep)
    assert mf[0] == pytest.approx(ex[0], abs=1e-9)
    assert mf[1] == pytest.approx(ex[1], abs=1e-4)
    assert metrics.epistemic_value(agent, deep) == pytest.approx(
        agent.epistemic_value_about(deep), abs=1e-4)


def test_exact_and_meanfield_efe_diverge_once_the_joint_couples():
    """And they must NOT agree forever, or the exact accessor is not doing anything.

    A few observations on a mislabelled artifact make the joint depart from a product of its
    marginals, and the two estimators then differ. A test that only checked agreement would pass on
    an implementation that just called the mean-field version.
    """
    from ghostscale import metrics
    from ghostscale.environment import Artifact, Environment
    from ghostscale.observer import find_named_policies
    cfg, gm, agent = _fresh_exact_agent()
    deep, _ = find_named_policies(agent)
    env = Environment(cfg, gm, np.random.default_rng(3), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.GHOST, goal=1, declared_signal=K.SIG_CREATOR)
    obs_rng = np.random.default_rng(9)
    worst = 0.0
    for _ in range(4):
        agent.infer_states(env.observation(art, K.DEEP, obs_rng))
        agent.action = np.zeros(len(agent.num_controls))
        agent.action[K.F_ATTENTION] = K.DEEP
        worst = max(worst, abs(metrics.policy_efe_terms(agent, deep)[1]
                               - agent.efe_terms(deep)[1]))
    assert worst > 1e-3, ("the exact and mean-field expected free energies never diverged, which "
                          "means the exact accessor is not computing anything the shipped helper "
                          "does not already compute")


# --------------------------------------------------------------------------- #
# Sequence log-evidence, which is what makes fitting possible.
# --------------------------------------------------------------------------- #
def test_log_evidence_is_a_proper_sequence_likelihood():
    """It must equal log P(o_1..o_T) computed the long way, by brute force over the joint."""
    from ghostscale.environment import Artifact, Environment
    cfg, gm, agent = _fresh_exact_agent()
    env = Environment(cfg, gm, np.random.default_rng(4), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.CURATOR, goal=2, declared_signal=K.SIG_CURATOR)
    obs_rng = np.random.default_rng(6)
    obs_list = [env.observation(art, K.DEEP, obs_rng) for _ in range(3)]

    # The filter's running total.
    action = np.zeros(len(agent.num_controls))
    action[K.F_ATTENTION] = K.DEEP
    for obs in obs_list:
        agent.infer_states(obs)
        agent.action = action.copy()
    from_filter = agent.log_evidence

    # The same thing by brute force: provenance and goal are fixed within an episode and attention
    # is pinned by the action, so the sequence likelihood is a sum over (provenance, goal) of the
    # prior times the product of per-step likelihoods.
    _, _, fresh = _fresh_exact_agent()
    prior = fresh._joint_prior()
    lik = np.ones_like(prior)
    for obs in obs_list:
        step = np.ones_like(prior)
        for m, o in enumerate(obs):
            step = step * fresh._L[m][int(o)]
        lik = lik * step
    # Attention is set by the action from step 2 onward; the first step uses D. Restrict to the DEEP
    # slice, which is what the imposed action forces.
    brute = float(np.log(np.sum(prior * lik)))
    assert from_filter == pytest.approx(brute, abs=1e-6)


def test_replay_imposes_the_recorded_actions():
    """Replay must not let the candidate model choose, or two models score different datasets."""
    from ghostscale.environment import Artifact, Environment
    from ghostscale import fitting as F
    cfg, gm, _ = _fresh_exact_agent()
    env = Environment(cfg, gm, np.random.default_rng(7), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.GHOST, goal=0, declared_signal=K.SIG_GHOST)
    tape = F.record_tape(env, art, cfg, np.random.default_rng(8), 5)
    assert len(tape.steps) == 5
    assert all(s[1] is not None and s[1][K.F_ATTENTION] == K.DEEP for s in tape.steps)

    from ghostscale.exact import make_exact_agent
    from ghostscale.generative_model import build_D
    D = build_D(cfg, np.random.default_rng(0))
    a = make_exact_agent(gm, D, cfg)
    b = make_exact_agent(gm, D, cfg)
    assert a.replay(tape.steps) == pytest.approx(b.replay(tape.steps), abs=1e-12)


# --------------------------------------------------------------------------- #
# The seed replacement D-6 offers.
# --------------------------------------------------------------------------- #
def test_the_offered_seed_replacement_is_collision_free():
    from ghostscale.diagnostics.d5_d6_power_and_seeds import collision_free_observer_seed
    seen = set()
    for c in range(8):
        for s in range(30):
            for i in range(120):
                seen.add(collision_free_observer_seed(20240719, c, s, i))
    assert len(seen) == 8 * 30 * 120


def test_the_shipped_seed_function_still_collides_and_only_across_cells():
    """Pins the DIRECTION of the known defect, which is what makes it benign.

    If a future change moved collisions inside a (cell, seed) group this would fail, because that is
    the unit the between-reader statistic assumes independence over.
    """
    from ghostscale.diagnostics.d5_d6_power_and_seeds import _audit
    from ghostscale.experiments._common import observer_seed
    a = _audit(observer_seed, 4, 20, 200)
    assert a["collisions_total"] > 0, "the documented defect has been fixed without updating D-6"
    assert a["collisions_within_a_cell_and_seed"] == 0
    assert a["collisions_across_seeds_within_a_cell"] == 0


# --------------------------------------------------------------------------- #
# The alternative disagreement statistic.
# --------------------------------------------------------------------------- #
def test_pairwise_divergence_separates_what_modal_entropy_conflates():
    """The whole case for the replacement, as a unit test on two synthetic populations.

    Both populations produce the same modal-goal entropy. One is readers who are each certain of a
    different answer; the other is readers who are all equally unsure. The shipped statistic cannot
    tell them apart and the replacement must.
    """
    from ghostscale import metrics
    from ghostscale.diagnostics.d3_disagreement import mean_pairwise_js

    k, n = 4, 200
    confident = np.full((n, k), 0.01 / (k - 1))
    for i in range(n):
        confident[i, i % k] = 0.99
    unsure = np.full((n, k), 1.0 / k)
    unsure += np.random.default_rng(0).normal(0, 1e-4, unsure.shape)
    unsure = np.abs(unsure)
    unsure /= unsure.sum(axis=1, keepdims=True)

    h_conf = metrics.between_observer_entropy(list(confident))
    h_unsure = metrics.between_observer_entropy(list(unsure))
    assert abs(h_conf - h_unsure) < 0.05, ("the two populations should be indistinguishable to the "
                                           "shipped statistic; if they are not, the fixture no "
                                           "longer poses the problem")
    js_conf = mean_pairwise_js(list(confident))
    js_unsure = mean_pairwise_js(list(unsure))
    assert js_conf > 10 * max(js_unsure, 1e-9), (
        "the replacement must separate confident-and-different from all-equally-unsure")
