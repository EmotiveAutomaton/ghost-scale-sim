"""Invariant tests (Spec §10).

Model-structure invariants live here. Two further §10 invariants are tested where their
machinery lives:
    * empirical creator feature distributions match A[0] columns  -> test in this file
      (``test_creator_empirical_matches_A0``), using ghostscale.creators.
    * reproducibility (identical seed -> identical CSV)            -> test in this file
      (``test_reproducibility_identical_csv``), using a small experiment run.
"""
import numpy as np
import pytest

from ghostscale.config import load_config
from ghostscale import generative_model as gmod
from ghostscale import constants as K
from ghostscale.metrics import js_divergence, shannon_entropy, mutual_information_features_goal


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def gm(cfg):
    return gmod.build_shared_model(cfg, run_assertions=True)


def test_A_columns_sum_to_one(gm, cfg):
    atol = float(cfg.tolerances.prob_sum_atol)
    for m in range(3):
        col_sums = np.asarray(gm.A[m]).sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=atol), f"A[{m}] columns must sum to 1"


def test_B_columns_sum_to_one(gm, cfg):
    atol = float(cfg.tolerances.prob_sum_atol)
    for f in range(3):
        col_sums = np.asarray(gm.B[f]).sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=atol), f"B[{f}] columns must sum to 1"


def test_shapes_match_spec(gm, cfg):
    cd = gmod.cards(cfg)
    assert np.asarray(gm.A[0]).shape == (cd.features, cd.provenance, cd.goals, cd.attention)
    assert np.asarray(gm.A[1]).shape == (cd.signals, cd.provenance, cd.goals, cd.attention)
    assert np.asarray(gm.A[2]).shape == (cd.effort, cd.provenance, cd.goals, cd.attention)
    assert np.asarray(gm.B[0]).shape == (cd.provenance, cd.provenance, 1)
    assert np.asarray(gm.B[1]).shape == (cd.goals, cd.goals, 1)
    assert np.asarray(gm.B[2]).shape == (cd.attention, cd.attention, cd.attention)


def test_pairwise_js_above_threshold(gm, cfg):
    thresh = float(cfg.artifact_model.js_threshold)
    cd = gmod.cards(cfg)
    for g1 in range(cd.goals):
        for g2 in range(g1 + 1, cd.goals):
            d = js_divergence(gm.sig[g1], gm.sig[g2])
            assert d > thresh, f"JS(sig[{g1}], sig[{g2}])={d:.4f} <= {thresh}"


def test_synth_is_structured_not_high_entropy(gm, cfg):
    """H(noise_free_synth) must be below the structured ceiling AND below uniform —
    asserting explicitly that synthetic content is not high-entropy (Spec §10)."""
    h = shannon_entropy(gm.noise_free_synth)
    ceiling = float(cfg.artifact_model.structured_ceiling)
    cd = gmod.cards(cfg)
    assert h < ceiling, f"H(synth)={h:.4f} not below ceiling {ceiling}"
    assert h < np.log(cd.features), "synth entropy must be below uniform entropy"


def test_mi_monotonic_decreasing_in_tier(gm, cfg):
    cd = gmod.cards(cfg)
    mi = [mutual_information_features_goal(gm.A[0], p, K.DEEP) for p in range(cd.provenance)]
    for p in range(cd.provenance - 1):
        assert mi[p] >= mi[p + 1] - 1e-9, (
            f"MI must be non-increasing CREATOR>POLISHED>CURATOR>GHOST; got {mi}")
    # GHOST is effectively unidentifiable.
    assert mi[K.GHOST] < 0.1, f"MI[GHOST]={mi[K.GHOST]:.4f} should be near zero"


def test_preferences_zero_N7(gm):
    gmod.assert_preferences_zero(gm.C)


def test_skim_column_is_synth_for_all_goals(gm, cfg):
    """DEEP is epistemically valuable, SKIM is not: A0[:,p,g,SKIM] == noise_free_synth
    for every provenance and goal (Spec §3.3)."""
    cd = gmod.cards(cfg)
    for p in range(cd.provenance):
        for g in range(cd.goals):
            assert np.allclose(gm.A[0][:, p, g, K.SKIM], gm.noise_free_synth)


def test_creator_empirical_matches_A0(cfg):
    """§10: empirical feature distributions from creators.py match the corresponding
    A[0] columns within tolerance."""
    from ghostscale import creators
    gm = gmod.build_shared_model(cfg, run_assertions=True)
    atol = float(cfg.tolerances.empirical_match_atol)
    rng = np.random.default_rng(7)
    report = creators.verify_empirical_matches_A0(gm, cfg, rng, n_samples=20000)
    for tier, err in report.items():
        assert err < atol, f"empirical feature dist for {tier} off by {err:.4f} > {atol}"


def test_reproducibility_identical_csv(cfg, tmp_path):
    """§10: identical seed produces identical CSV output."""
    from ghostscale.experiments import e1_crash
    cfg_q = load_config(quick=True)
    df1 = e1_crash.run(cfg_q, out_dir=tmp_path / "a", seed=cfg_q.run.base_seed)
    df2 = e1_crash.run(cfg_q, out_dir=tmp_path / "b", seed=cfg_q.run.base_seed)
    import pandas as pd
    pd.testing.assert_frame_equal(df1, df2)
