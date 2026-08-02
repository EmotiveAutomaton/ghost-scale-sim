"""V10 tests — the seven nulls, the pre-registration lock, and the bugs that actually happened.

The last block exists because those things genuinely broke while building. A test that pins a bug
you never had is decoration; one that pins a bug you did is a record.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ghostscale import constants as K
from ghostscale import prereg_v10
from ghostscale.v10 import v10_dir
from ghostscale.v10.e55_intent_gate import INTENT_READERS, _gate_for, _resolved
from ghostscale.v10.e56_selective_gate import _uptake


@pytest.fixture(scope="module")
def summary():
    p = v10_dir() / "summary.json"
    if not p.exists():
        pytest.skip("run_v10.py has not been run")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The pre-registration
# --------------------------------------------------------------------------- #
def test_prereg_hash_verifies():
    prereg_v10.write_card()
    assert prereg_v10.verify()


def test_prereg_records_the_authors_stated_responses():
    """The thing that makes a null expensive is having written down what it would mean."""
    priors = prereg_v10.CARD["authors_recorded_priors"]
    assert "H10.6" in priors and "N45" in priors
    assert "E8" in priors["N45"]


def test_prereg_forbids_the_attribution_claim():
    assert "no causal attribution" in prereg_v10.CARD["may_not_claim"]


def test_prereg_makes_the_severity_rule_binding():
    assert "every V10 headline" in prereg_v10.CARD["severity_rule"]


# --------------------------------------------------------------------------- #
# N45-N51
# --------------------------------------------------------------------------- #
def test_n45_reported_and_the_failure_is_attributed(summary):
    """N45 FAILS as stated, and it fails because of the rider arm alone.

    This is the null the author recorded himself as most expecting to fail. It did -- and it did
    its job: the failure is confined to reader 7, whose clean-corpus cost is catastrophic, which is
    what disqualifies that arm rather than the headline. The three actual gates cost nothing.
    """
    cost = summary["E55"]["null_n45"]["retention_cost_by_reader"]
    gates = {k: v for k, v in cost.items() if k != "intent_plus_rider"}
    assert all(abs(float(v)) <= 0.05 for v in gates.values()), gates
    assert float(cost["intent_plus_rider"]) > 0.05, "the rider was expected to fail this"
    assert summary["E55"]["null_n45"]["passed"] is False


def test_n46_the_surface_filter_is_a_fair_comparison(summary):
    assert summary["E55"]["null_n46"]["passed"]


def test_n47_the_intent_gate_never_reads_the_provenance_signal():
    """Otherwise it is a label filter with extra steps."""
    from ghostscale import v6_model as V6
    from ghostscale.environment import Artifact

    ng = 4
    vm = V6.build_values_map(ng, n_values=2)
    vp = np.array([0.85, 0.15])
    post = np.array([0.6, 0.2, 0.1, 0.1])
    for reader in INTENT_READERS:
        gates = set()
        for sig in (K.SIG_CREATOR, K.SIG_GHOST, K.UNSIGNED):
            art = Artifact(provenance=K.GHOST, goal=0, declared_signal=sig)
            gates.add(round(_gate_for(reader, post, art, ng, vm, vp, False,
                                      0.7, 0.25, 0.35, 8.0), 12))
        assert len(gates) == 1, f"{reader} changed its gate when only the LABEL changed"


def test_n48_process_varies_more_than_goal_readability(summary):
    assert summary["E56"]["null_n48"]["passed"]


def test_n49_with_evasion_off_e53_reproduces(summary):
    assert summary["E57"]["null_n49"]["passed"]


def test_n50_reported_as_failing_and_the_column_is_withheld(summary):
    """Value drift is non-zero on a CLEAN corpus, so the measure is contaminated by ordinary
    learning and cannot carry H10.3 or H10.4. Recorded rather than quietly dropped."""
    assert summary["E55"]["null_n50"]["passed"] is False


def test_n51_the_riders_value_gate_is_the_same_gate(summary):
    """H10.4's whole claim is that values arrive DESPITE the guard. If the rider's guard were
    weaker than the gated reader's, nothing would be shown."""
    from ghostscale import v6_model as V6
    from ghostscale.environment import Artifact

    ng = 4
    vm = V6.build_values_map(ng, n_values=2)
    vp = np.array([0.85, 0.15])
    post = np.array([0.55, 0.25, 0.1, 0.1])
    art = Artifact(provenance=K.GHOST, goal=0, declared_signal=K.SIG_CREATOR)
    a = _gate_for("intent_plus_rider", post, art, ng, vm, vp, False, 0.7, 0.25, 0.35, 8.0)
    b = _gate_for("intent_handset_values", post, art, ng, vm, vp, False, 0.7, 0.25, 0.35, 8.0)
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# The severity rule
# --------------------------------------------------------------------------- #
def test_every_headline_has_a_severity_rate(summary):
    """SPEC_V10 made this binding: no headline gets a sentence without its false-positive rate."""
    rates = summary["S2"]["rates"]
    assert len(rates) >= 3
    for name, r in rates.items():
        assert np.isfinite(float(r["false_positive_rate"])), name


# --------------------------------------------------------------------------- #
# The bugs that actually happened
# --------------------------------------------------------------------------- #
def test_goal_posterior_is_per_step_and_must_be_reduced():
    """The bug this pins, and it had two faces from one cause.

    ``rollout_observer`` returns ``goal_posterior`` with shape (T, n_goals). Passing the whole
    matrix made the values readers crash outright -- and, far worse because it was silent, made the
    reconstructibility gate compute entropy over the entire matrix and sit at exactly 0.0 on every
    corpus. A gate stuck shut reads as perfect retention: the learner had simply never learned.
    """
    ng = 4
    per_step = np.tile(np.array([0.7, 0.1, 0.1, 0.1]), (6, 1))
    assert per_step.ndim == 2
    bad = _resolved(per_step, ng)
    good = _resolved(per_step[-1], ng)
    assert bad != pytest.approx(good), "reducing to the final row must change the answer"
    assert good > 0.1, "a sharp posterior must give a meaningfully open gate"


def test_uptake_channels_all_accrue_per_step():
    """The bug this pins: scoring process per step but goal and values against the FINAL gate.

    By the end of a reading the sympathetic reader's running divergence has reached the value the
    adversarial reader anticipated from the start, so both arms share a final gate BY CONSTRUCTION.
    Goal and value ratios came back at exactly 1.000 -- an erased measurement, not a null.
    """
    class E:
        goal_posteriors_by_step = [np.array([0.4, 0.3, 0.2, 0.1]),
                                   np.array([0.7, 0.1, 0.1, 0.1])]
        goal_prior = np.full(4, 0.25)
        goal_posterior = np.array([0.7, 0.1, 0.1, 0.1])
        subgoal_posteriors = [np.array([0.6, 0.4]), np.array([0.8, 0.2])]
        true_modes = [0, 0]
        true_goal = 0
        process = {"process_error_reduction": 0.3}

    from ghostscale import v6_model as V6
    vm = V6.build_values_map(4, n_values=2)
    open_gate = _uptake(E(), [1.0, 1.0], vm, 4, 2)
    early_shut = _uptake(E(), [0.0, 1.0], vm, 4, 2)
    assert early_shut["goal_uptake"] < open_gate["goal_uptake"], (
        "a gate shut early must reduce goal uptake; if it does not, the channels are being "
        "weighted by the final gate again")


def test_local_surface_detector_lives_in_the_learners_own_world():
    """The bug this pins: E55 first reached into a V5 world for its signature families while
    learning in the V1-style one, which silently mismatches cardinality. A filter scoring content
    from a different model than the learner reads is not a comparison."""
    import inspect

    from ghostscale.v10 import e55_intent_gate as E55
    src = inspect.getsource(E55)
    assert "class LocalSurfaceDetector" in src
    assert "build_world_and_config" not in src, (
        "E55 must not build a V5 world; its learner lives in the V1-style model")
