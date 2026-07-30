"""Tests for the repair pass: the corrected measures, the seed schemes, and exact learning.

The first group pins the three corrections this pass made to its own specification. All three were
found by checking a formula or an estimand before running it, and all three would have produced
plausible output if left alone, which is the failure mode this project keeps meeting.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale import constants as K
from ghostscale import metrics
from ghostscale.config import load_config


# --------------------------------------------------------------------------- #
# The corrected error-reduction measure.
# --------------------------------------------------------------------------- #
def test_error_reduction_is_signed_and_finite():
    """The specification's own form is infinite; this one is not, and it goes negative.

    Written as KL(prior || truth) - KL(posterior || truth) against a point mass on the truth, every
    term diverges, and floored at an epsilon it returns a number that tracks the epsilon rather than
    the data. The reversed form is the reduction in the surprisal of the truth.
    """
    prior = np.full(4, 0.25)
    right = np.array([0.01, 0.97, 0.01, 0.01])
    wrong = np.array([0.97, 0.01, 0.01, 0.01])
    good = metrics.error_reduction(right, prior, 1)
    bad = metrics.error_reduction(wrong, prior, 1)
    assert np.isfinite(good) and np.isfinite(bad)
    assert good > 0 > bad, "a reader moved toward the truth must score above zero and one moved away below"
    assert good == pytest.approx(np.log(right[1]) - np.log(prior[1]))


def test_error_reduction_is_monotone_in_confidence_in_the_truth():
    """Unlike the distance it replaces, it never rewards being confidently wrong."""
    prior = np.full(4, 0.25)
    last = -np.inf
    for p_true in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
        rest = (1.0 - p_true) / 3.0
        post = np.array([rest, p_true, rest, rest])
        v = metrics.error_reduction(post, prior, 1)
        assert v > last
        last = v


def test_movement_is_not_monotone_but_error_reduction_is():
    """The whole reason for the decomposition, as a unit test.

    A confidently wrong reader has moved as far from its prior as a confidently right one. The
    distance cannot tell them apart; the signed measure must.
    """
    prior = np.full(4, 0.25)
    right = np.array([0.01, 0.97, 0.01, 0.01])
    wrong = np.array([0.97, 0.01, 0.01, 0.01])
    assert metrics.kl_divergence(right, prior) == pytest.approx(
        metrics.kl_divergence(wrong, prior), rel=1e-9), (
        "the fixture no longer poses the problem: the two posteriors must be equidistant from the "
        "prior for the point to hold")
    assert metrics.error_reduction(right, prior, 1) != pytest.approx(
        metrics.error_reduction(wrong, prior, 1))


def test_trust_factor_is_reported_separately_and_spans_a_decade():
    lo, hi = metrics.trust_factor(0.1), metrics.trust_factor(0.99)
    assert hi / lo > 40, ("the factor's range is the reason it is pulled out of the product; if it "
                          "were small there would be no point separating it")


# --------------------------------------------------------------------------- #
# The second disagreement statistic and the bias correction.
# --------------------------------------------------------------------------- #
def test_pairwise_divergence_separates_what_modal_entropy_conflates():
    k, n = 4, 200
    confident = np.full((n, k), 0.01 / (k - 1))
    for i in range(n):
        confident[i, i % k] = 0.99
    unsure = np.full((n, k), 1.0 / k)
    unsure += np.random.default_rng(0).normal(0, 1e-4, unsure.shape)
    unsure = np.abs(unsure)
    unsure /= unsure.sum(axis=1, keepdims=True)

    assert abs(metrics.between_observer_entropy(list(confident))
               - metrics.between_observer_entropy(list(unsure))) < 0.05
    assert metrics.mean_pairwise_js(list(confident)) > 10 * max(
        metrics.mean_pairwise_js(list(unsure)), 1e-9)


def test_bias_correction_moves_toward_the_ceiling_and_vanishes_at_scale():
    rng = np.random.default_rng(1)
    k = 4
    for n, tol in ((16, 0.15), (4000, 0.005)):
        modal = rng.integers(0, k, n)
        P = np.zeros((n, k))
        P[np.arange(n), modal] = 1.0
        plug = metrics.between_observer_entropy(list(P))
        corr = metrics.between_observer_entropy_corrected(list(P))
        assert corr >= plug, "the correction is upward; the plug-in estimator is biased low"
        assert abs(corr - np.log(k)) < tol


def test_a_unanimous_population_is_not_corrected():
    """The correction uses the OBSERVED support, so unanimity is left alone rather than inflated."""
    P = np.tile(np.array([0.97, 0.01, 0.01, 0.01]), (100, 1))
    assert metrics.between_observer_entropy_corrected(list(P)) == pytest.approx(
        metrics.between_observer_entropy(list(P)))


# --------------------------------------------------------------------------- #
# The estimand correction behind R-1.
# --------------------------------------------------------------------------- #
def test_pooling_readers_is_a_different_estimand_from_the_cell_correlation():
    """Pooling attenuates when the within-cell correlation is zero, which is the R-1 correction.

    Synthetic fixture with the same structure the committed data has: cells whose means are
    perfectly ordered, and readers within a cell whose two quantities are independent.
    """
    from ghostscale.diagnostics.criteria import spearman
    rng = np.random.default_rng(3)
    means = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0), (6.0, 6.0)]
    cell_x, cell_y, all_x, all_y = [], [], [], []
    for mx, my in means:
        x = mx + rng.normal(0, 2.0, 400)
        y = my + rng.normal(0, 2.0, 400)
        cell_x.append(x.mean())
        cell_y.append(y.mean())
        all_x.extend(x)
        all_y.extend(y)
    cell_rho = spearman(cell_x, cell_y)
    pooled_rho = spearman(all_x, all_y)
    assert cell_rho == pytest.approx(1.0)
    assert pooled_rho < cell_rho - 0.2, (
        "pooling readers must attenuate the correlation when the within-cell association is zero; "
        "if it does not, the fixture no longer poses the problem R-1 corrects")


# --------------------------------------------------------------------------- #
# The seed schemes.
# --------------------------------------------------------------------------- #
def test_the_default_seed_scheme_is_the_collision_free_one():
    from ghostscale.experiments import _common as C
    assert C.seed_scheme() == "hash"
    assert C.observer_seed(1, 2, 3, 4) == C.hashed_observer_seed(1, 2, 3, 4)


def test_the_legacy_scheme_is_still_reachable_and_still_reproduces_itself(monkeypatch):
    """The historical record has to remain regenerable by the code that produced it."""
    from ghostscale.experiments import _common as C
    monkeypatch.setenv(C.SEED_SCHEME_ENV, "legacy")
    assert C.seed_scheme() == "legacy"
    assert C.observer_seed(20240719, 0, 10, 0) == C.legacy_observer_seed(20240719, 0, 10, 0)
    # The documented collision, pinned so the historical structure stays checkable.
    assert C.legacy_observer_seed(20240719, 0, 10, 0) == \
        C.legacy_observer_seed(20240719, 1, 0, 67)


def test_the_new_scheme_has_no_collisions_on_the_envelopes_the_project_runs():
    from ghostscale.experiments import _common as C
    seen = set()
    for c in range(8):
        for s in range(30):
            for i in range(120):
                seen.add(C.hashed_observer_seed(20240719, c, s, i))
    assert len(seen) == 8 * 30 * 120


# --------------------------------------------------------------------------- #
# Exact-posterior Dirichlet learning.
# --------------------------------------------------------------------------- #
def _learner(exact: bool):
    from ghostscale.generative_model import build_D, build_shared_model
    from ghostscale.learning import make_learner_agent
    cfg = load_config()
    cfg.set("inference.exact", bool(exact))
    gm = build_shared_model(cfg)
    agent = make_learner_agent(gm, build_D(cfg, np.random.default_rng(5)), cfg)
    return cfg, gm, agent


def test_the_solver_switch_reaches_the_learner():
    from ghostscale.exact import ExactAgent
    from ghostscale.learning import is_learner
    _, _, approx = _learner(False)
    _, _, exact = _learner(True)
    assert not isinstance(approx, ExactAgent)
    assert isinstance(exact, ExactAgent)
    assert is_learner(approx) and is_learner(exact), (
        "both must be recognised as learners, or the rollout loop will not call the update at all")


def test_exact_learning_actually_learns_and_keeps_the_likelihood_normalised():
    from ghostscale.environment import Artifact, Environment
    from ghostscale.observer import rollout_observer
    cfg, gm, agent = _learner(True)
    env = Environment(cfg, gm, np.random.default_rng(6), honesty=1.0, signing_rate=1.0)
    before = metrics.mutual_information_features_goal(np.asarray(agent.A[0]), K.CREATOR, K.DEEP)
    for j in range(15):
        art = Artifact(provenance=K.CREATOR, goal=j % 4, declared_signal=K.SIG_CREATOR)
        rollout_observer(agent, art, env, cfg, np.random.default_rng(700 + j), 8,
                         force_deep_k=8, learn=True)
    after = metrics.mutual_information_features_goal(np.asarray(agent.A[0]), K.CREATOR, K.DEEP)
    assert after > before, "the exact learner must actually acquire structure"
    assert np.allclose(np.asarray(agent.A[0]).sum(axis=0), 1.0, atol=1e-10)


def test_the_exact_learner_rebuilds_its_own_likelihood_cache():
    """A silent failure if missed: inference would carry on using the pre-update model.

    The flattened likelihood the filter uses is derived from A, so an update that does not rebuild
    it leaves the agent learning into an array nothing reads.
    """
    from ghostscale.environment import Artifact, Environment
    cfg, gm, agent = _learner(True)
    env = Environment(cfg, gm, np.random.default_rng(8), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.CREATOR, goal=2, declared_signal=K.SIG_CREATOR)
    obs = env.observation(art, K.DEEP, np.random.default_rng(9))
    before = agent._L[0].copy()
    agent.update_A(obs)
    assert not np.allclose(before, agent._L[0]), (
        "update_A must rebuild the flattened likelihood, or the filter keeps using the old model "
        "and everything the agent learns is silently discarded")


def test_a_learner_without_dirichlet_parameters_refuses_rather_than_guessing():
    from ghostscale.exact import make_exact_agent
    from ghostscale.generative_model import build_D, build_shared_model
    cfg = load_config()
    cfg.set("inference.exact", True)
    gm = build_shared_model(cfg)
    agent = make_exact_agent(gm, build_D(cfg, np.random.default_rng(11)), cfg)
    with pytest.raises(NotImplementedError):
        agent.update_A([0, 0, 0])


# --------------------------------------------------------------------------- #
# The repair criteria lock and its scorers.
# --------------------------------------------------------------------------- #
def test_repair_criteria_are_hash_locked(tmp_path):
    import json

    from ghostscale.repair import criteria as CR
    path = tmp_path / "criteria.json"
    written = CR.write_criteria(load_config(), path)
    assert CR.ensure_criteria(load_config(), path)["content_hash"] == written["content_hash"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["r1_r3"]["bootstrap_draws"] = 7
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified since it was written"):
        CR.ensure_criteria(load_config(), path)


def test_map_label_needs_both_a_signal_and_a_noise_term():
    """The third distinguishability failure, pinned.

    A statistically separable estimate that tracks almost none of the true variation must not be
    called measurable. The learned-trust check found one recovering 0.7% of the signal and scoring
    100% identifiable, because its readers were near-deterministic so the noise was smaller still.
    """
    from ghostscale.repair import criteria as CR
    assert CR.map_label(1.0) == "measurable across the range"
    assert CR.map_label(1.0, slope=0.9) == "measurable across the range"
    assert "flat" in CR.map_label(1.0, slope=0.007)
    assert "flat" in CR.map_label(1.0, slope=-0.01), "the floor is on the magnitude, not the sign"


def test_determined_against_reports_both_directions():
    from ghostscale.repair import criteria as CR
    assert CR.determined_against((0.8, 0.9), 0.7) == "determined_meets"
    assert CR.determined_against((0.1, 0.2), 0.7) == "determined_fails"
    assert CR.determined_against((0.5, 0.9), 0.7) == "undetermined"


def test_identifiable_fraction_still_handles_a_zero_standard_error():
    """Carried over from the diagnostics regression: noiseless recovery is the best case."""
    from ghostscale.repair import criteria as CR
    frac, _ = CR.identifiable_fraction([0, 1, 2, 3], [0, 1, 2, 3], [0, 0, 0, 0])
    assert frac == pytest.approx(1.0)
    frac, _ = CR.identifiable_fraction([0, 1, 2, 3], [5, 5, 5, 5], [0, 0, 0, 0])
    assert frac == pytest.approx(0.0)
