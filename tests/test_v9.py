"""V9 tests — the four nulls, the pre-registration lock, and the pieces that broke while building.

The last three tests exist because those three things actually went wrong. A test that pins a bug
you never had is decoration; a test that pins one you did is a record.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ghostscale import constants as K
from ghostscale import prereg_v9
from ghostscale.config import load_config
from ghostscale.v6 import harness as H
from ghostscale.v9 import v9_dir
from ghostscale.v9.e53_e54 import SurfaceDetector, _paired_diff, _value_prior
from ghostscale.v9.minimal_models import ABLATIONS


@pytest.fixture(scope="module")
def summary():
    p = v9_dir() / "summary.json"
    if not p.exists():
        pytest.skip("run_v9.py has not been run")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The pre-registration
# --------------------------------------------------------------------------- #
def test_prereg_hash_verifies():
    prereg_v9.write_card()
    assert prereg_v9.verify()


def test_prereg_records_the_predictions_that_failed():
    """H9.2 and H9.4 did not survive. The card is what makes saying so checkable."""
    assert "H9.2" in prereg_v9.CARD["hypotheses"]
    assert "H9.4" in prereg_v9.CARD["hypotheses"]
    assert "H9.4" in prereg_v9.CARD["fails_if"]


# --------------------------------------------------------------------------- #
# N41 and N42 — the minimal-model programme
# --------------------------------------------------------------------------- #
def test_n41_the_label_effect_survives_at_least_one_ablation(summary):
    """Otherwise the ablations are too destructive and this measures whether the model runs."""
    assert summary["MIN"]["null_n41"]["passed"]


def test_n42_at_least_one_finding_dies(summary):
    """Otherwise the ablations never reached the mechanism and the programme has not been run."""
    assert summary["MIN"]["null_n42"]["passed"]


def test_a_finding_that_does_not_hold_in_the_baseline_reports_no_load_bearing_set(summary):
    """The bug this pins: a row that never fired was reporting as maximally fragile.

    A finding that does not reproduce in the ablation harness's own baseline cannot have a
    load-bearing set. Reporting one made 'sustained futile attention' read as dying to all six
    commitments when in fact it never fired once.
    """
    for name, m in summary["MIN"]["minimal_models"].items():
        if not m["holds_intact"]:
            assert m["load_bearing"] is None, f"{name} claims a load-bearing set it cannot have"
            assert "UNINFORMATIVE" in m["note"]


def test_every_ablation_column_is_a_named_commitment(summary):
    grid = summary["MIN"]["grid"] if "grid" in summary["MIN"] else None
    assert set(ABLATIONS[1:]) <= set(ABLATIONS)
    assert ABLATIONS[0] == "none", "the intact baseline must be the first column"


# --------------------------------------------------------------------------- #
# N43 and N44
# --------------------------------------------------------------------------- #
def test_n43_the_detector_carries_no_goal_information(summary):
    """Otherwise it is a second legibility channel rather than an origin heuristic."""
    assert summary["E53E54"]["E53"]["null_n43"]["passed"]


def test_n44_matched_engagement_is_enforced_not_hoped_for(summary):
    """Both stances replay the same attention trace, so any drift difference is the gate."""
    n44 = summary["E53E54"]["E54"]["null_n44"]
    assert n44["passed"]
    assert n44["engagement_sympathetic"] == pytest.approx(n44["engagement_adversarial"])
    assert "ENFORCED" in n44["how"]


# --------------------------------------------------------------------------- #
# The three things that actually broke
# --------------------------------------------------------------------------- #
def test_the_value_prior_leaves_the_gate_room_to_move():
    """The bug this pins: a hard reversal of a near-one-hot implied-values vector made the
    divergence the log of the clipping epsilon (13.8155 = ln 1e6), which pinned the gate at its
    leak floor in every arm. The experiment then reported 'stance makes no difference' when in
    fact no gate had moved."""
    from ghostscale import v6_model as V6
    values_map = V6.build_values_map(6, n_values=2)
    one_hot = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vp = _value_prior(one_hot, values_map)
    d = V6.value_divergence_via_values(one_hot, vp, values_map)
    assert d < 5.0, f"divergence {d} is back in the clipping-artifact regime"
    assert np.isfinite(d)


def test_the_glance_carries_no_origin_information():
    """The finding that forced E53's declared deviation: a skim emits the SAME feature
    distribution for machine and human work, so a glance-level detector cannot exist here."""
    cfg = load_config(quick=True)
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    nf = int(cfg_b.cardinalities.num_features)
    dists = {}
    for prov, beta, name in ((K.GHOST, 0.0, "machine"), (K.CREATOR, 1.0, "human")):
        counts = np.zeros(nf)
        for i in range(300):
            rng = np.random.default_rng(4242 + i)
            g = int(rng.integers(ng))
            _, art, env = H.make_artifact_and_env(world, cfg_r, g, 2, beta, 4, rng,
                                                  provenance=prov)
            counts[int(env.sample_feature(art, K.SKIM, rng))] += 1
        dists[name] = counts / counts.sum()
    assert np.allclose(dists["machine"], dists["human"], atol=1e-9)


def test_paired_bootstrap_resamples_units_not_arms():
    """The bug this pins: E54's label ordering flipped between two runs at different sample sizes.
    A few-percent difference read as a result until the pairs were resampled."""
    import pandas as pd
    rng = np.random.default_rng(0)
    base = rng.normal(size=40)
    df = pd.DataFrame({"reader": list(range(40)) * 2,
                       "label": ["a"] * 40 + ["b"] * 40,
                       "drift": np.concatenate([base, base + 0.5])})
    d = _paired_diff(df, "drift", "label", "b", "a", n_boot=800)
    assert d["difference"] == pytest.approx(0.5, abs=1e-9)
    assert d["separated_from_zero"]

    same = pd.DataFrame({"reader": list(range(40)) * 2,
                         "label": ["a"] * 40 + ["b"] * 40,
                         "drift": np.concatenate([base, base])})
    assert not _paired_diff(same, "drift", "label", "b", "a", n_boot=800)["separated_from_zero"]


def test_an_untrained_detector_never_fires():
    det = SurfaceDetector(None, None, 16, 0, 6, np.random.default_rng(0))
    assert not det.fires([0, 1, 2])
    assert det.score([0, 1, 2]) == 0.0
