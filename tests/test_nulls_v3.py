"""V3 nulls and invariants (V3 spec §3).

    N13 — Convergence scaling      -> here (guards, and TESTS, the finite-sample diagnosis)
    N14 — Averaging null           -> here (proves the fix is the averaging, not the refactor)
    N15 — Probe-set purity         -> here (encoder divergence must measure the observer)

Plus the V3 §3 standing invariant: every population-averaged seed column remains a valid
categorical distribution after averaging and renormalization.

N11's repair lives in ``test_nulls_v2.py`` alongside the null it replaces, so that the
history of that gate stays in one file.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ghostscale.config import load_config
from ghostscale import generative_model as gmod
from ghostscale import constants as K
from ghostscale import prereg_v3 as P3
from ghostscale.generations import (average_seed_column, creator_deep_column, run_chain,
                                    SeededCreator)
from ghostscale.preregistration import POP_GOAL_DIST

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def cfg_q():
    return load_config(quick=True)


# --------------------------------------------------------------------------- #
# N13 — Convergence scaling.
#
# V3 §3 states this as "the without-averaging leak slope must be monotonically non-increasing
# in sample size". DECISION D8 replaces monotonicity with the 1/N exponent, and the reason is
# scientific rather than statistical convenience:
#
#   * Monotonicity says only "the leak did not get worse". A leak shrinking as 1/log N passes
#     it, and 1/log N is not the finite-sample claim.
#   * The DIAGNOSIS is that the leak is finite-sample estimation error, whose signature is KL
#     error falling as 1/N. Regressing log|slope| on log N tests THAT — so this null now
#     confirms or refutes the V3 story rather than merely guarding it.
#
# It also stops failing by luck: five noisy slope estimates break strict monotonicity often,
# and N13 blocks E8.
# --------------------------------------------------------------------------- #
def test_N13_criterion_rejects_a_leak_that_does_not_shrink():
    """A flat leak must REFUTE the diagnosis. The gate has to be able to fail."""
    flat = P3.n13_verdict(P3.loglog_slope_fit([100, 300, 1000, 3000, 10000],
                                              [0.012, 0.012, 0.012, 0.012, 0.012]))
    assert not flat["passed"], "a leak that does not shrink with N must refute the diagnosis"

    growing = P3.n13_verdict(P3.loglog_slope_fit([100, 300, 1000, 3000],
                                                 [0.004, 0.008, 0.016, 0.032]))
    assert not growing["passed"], "a leak that GROWS with N must refute the diagnosis"


def test_N13_criterion_accepts_one_over_N_and_identifies_it():
    """A textbook 1/N leak passes the gate AND is flagged as consistent with 1/N."""
    n = np.array([100, 300, 1000, 3000, 10000], dtype=float)
    v = P3.n13_verdict(P3.loglog_slope_fit(n, 1.2 / n))
    assert v["passed"]
    assert v["consistent_with_one_over_N"]
    assert abs(v["b"] - (-1.0)) < 0.05, f"exponent should recover -1, got {v['b']}"


def test_N13_distinguishes_one_over_N_from_one_over_sqrt_N():
    """The point of D8: 1/sqrt(N) shrinks, so monotonicity would pass it — but it is NOT the
    finite-sample signature, and the band must say so while the gate still passes."""
    n = np.array([100, 300, 1000, 3000, 10000], dtype=float)
    v = P3.n13_verdict(P3.loglog_slope_fit(n, 0.12 / np.sqrt(n)))
    assert v["passed"], "a declining leak still clears the gate"
    assert not v["consistent_with_one_over_N"], (
        "1/sqrt(N) must be reported as INCONSISTENT with the finite-sample diagnosis — this "
        "is exactly the distinction a monotonicity null cannot make")


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "N13 FAILS ON THE FULL-SCALE E12 RUN, and that failure is THE REPORTED V3 RESULT, not a "
    "bug to tune away — see RESULTS_V3.md. The f=0 honest-signal leak does not shrink with "
    "per-generation sample size: log|slope| ~ log N gives b = -0.017 (t = -0.28) across a 100x "
    "range, against a predicted -1. Per V3 spec §1 C2 the finite-sample diagnosis is therefore "
    "wrong, E8 was not run, and it remains unreported. Marked xfail(strict) for exactly the "
    "reason V2 marked N11 that way: the failure stays VISIBLE in the suite rather than being "
    "deleted or weakened, and if a future change ever makes the leak shrink with data this "
    "XPASSes and forces the marker off."))
def test_N13_on_the_real_e12_output():
    """The real gate, on E12's actual output. Skips until E12 has been run."""
    path = RESULTS / "e12_threshold.json"
    if not path.exists():
        pytest.skip("E12 has not been run; nothing to gate")
    verdict = json.loads(path.read_text(encoding="utf-8"))
    n13 = verdict["N13"]
    assert n13["passed"], (
        f"N13: the without-averaging leak does not shrink with sample size "
        f"(log-log exponent b = {n13.get('b')}, t = {n13.get('t')}). "
        + verdict["if_e8_may_not_run"])


# --------------------------------------------------------------------------- #
# N14 — Averaging null.
#
# "With a single observer per generation (M = 1), C1 averaging must reduce EXACTLY to the V2
#  single-observer seeding, reproducing the V2 leak. Proves the fix is the averaging over
#  M > 1 and not an incidental change to the seeding code."
#
# Tested at two levels, because the spec's sentence contains two different claims:
#   (a) EXACT reduction at M=1 — a bit-exactness property of the seeding function;
#   (b) reproducing the V2 leak — a property of the whole chain, which is what
#       ``restore_v2_e8.py`` checks against the reported +0.0119 / t=3.75.
# --------------------------------------------------------------------------- #
def test_N14_averaging_at_M1_is_bit_identical_to_v2_seeding(cfg):
    """(a) At M = 1 the averaged column IS the single observer's column, exactly."""
    rng = np.random.default_rng(11)
    gm = gmod.build_shared_model(cfg, goal_symmetric=False, synth_draw_seed=17)
    A0 = np.asarray(gm.A[0]) * (1.0 + 0.05 * rng.random(np.asarray(gm.A[0]).shape))
    A0 = A0 / A0.sum(axis=0, keepdims=True)
    col = creator_deep_column(A0)

    averaged = average_seed_column([col])
    assert np.array_equal(averaged, average_seed_column([col])), "averaging must be deterministic"
    assert np.allclose(averaged, col, atol=0, rtol=0) or np.max(np.abs(averaged - col)) < 1e-15, (
        "at M=1 the averaged seed column must be the single observer's column bit-for-bit; "
        "any difference means the fix is confounded with an incidental seeding change")


def test_N14_seeded_creator_is_identical_under_both_paths(cfg):
    """(a) continued: the creator built from the M=1 averaged column emits identically to one
    built from the raw column. The seeding rule is the ONLY thing C1 changes."""
    gm = gmod.build_shared_model(cfg, goal_symmetric=False, synth_draw_seed=17)
    col = creator_deep_column(np.asarray(gm.A[0]))
    for g in range(int(cfg.cardinalities.num_goals)):
        direct = SeededCreator(cfg, col[:, g], g).emission_distribution()
        viaavg = SeededCreator(cfg, average_seed_column([col])[:, g], g).emission_distribution()
        assert np.max(np.abs(direct - viaavg)) < 1e-12


def test_N14_averaging_actually_changes_the_seed_for_M_greater_than_1(cfg_q):
    """Guard against N14 passing vacuously. If averaging were a no-op at M > 1 the whole of
    C1 would be inert, and the M=1 equality above would be trivially satisfied by a fix that
    does nothing. The two seeding rules must produce DIFFERENT columns with real observers."""
    gm = gmod.build_shared_model(cfg_q, goal_symmetric=False, synth_draw_seed=17)
    kw = dict(contamination=0.0, signing_rate=1.0, honesty=1.0, g_max=2, n_creators=6,
              n_artifacts=60, n_observers=4, infer_steps=4, d_i=0.0, base_seed=4242)
    avg = run_chain(cfg_q, gm, POP_GOAL_DIST, population_average_seed=True, **kw)
    one = run_chain(cfg_q, gm, POP_GOAL_DIST, population_average_seed=False, **kw)
    delta = float(np.max(np.abs(avg[0].seed_column - one[0].seed_column)))
    assert delta > 1e-9, (
        "the averaged and single-observer seed columns are identical at M=4, so C1 is inert "
        "and N14 would pass vacuously")


# --------------------------------------------------------------------------- #
# N15 — Probe-set purity.
#
# "E8/E13's encoder-divergence probe set must contain ZERO contaminated or synthetic
#  artifacts, asserted in the worker and verified in the CSV. Otherwise encoder divergence
#  would be measuring the probe corpus, not the observer."
# --------------------------------------------------------------------------- #
def test_N15_probe_set_is_pure(cfg_q):
    """Asserted at construction, on the real draw path."""
    from ghostscale.probes import build_probe_env, draw_probe_set
    gm = gmod.build_shared_model(cfg_q, goal_symmetric=False, synth_draw_seed=17)
    env = build_probe_env(cfg_q, gm, np.random.default_rng(3))
    goals = np.array([0, 1, 2, 3] * 3)
    probes = draw_probe_set(env, 120, goals, np.random.default_rng(5))
    assert len(probes) == 120
    assert all(a.provenance == K.CREATOR for a in probes)


def test_N15_purity_assertion_actually_fires(cfg_q):
    """The assertion must be capable of failing, or it is decoration."""
    from ghostscale.environment import Artifact
    from ghostscale.probes import assert_probe_purity
    dirty = [Artifact(provenance=K.CREATOR, goal=0, declared_signal=0),
             Artifact(provenance=K.GHOST, goal=1, declared_signal=3)]
    with pytest.raises(AssertionError, match="N15"):
        assert_probe_purity(dirty)


def test_N15_verified_in_the_csv():
    """Verified in the CSV, not only in the worker — the spec asks for both."""
    path = RESULTS / "e8_raw.csv"
    if not path.exists():
        pytest.skip("E8 has not been run")
    df = pd.read_csv(path)
    if "probe_contaminated_count" not in df.columns:
        pytest.skip("this E8 run had the encoder channel disabled (n_probes = 0)")
    assert (df["probe_contaminated_count"] == 0).all(), (
        "N15: a contaminated artifact reached the encoder-divergence probe set; encoder "
        "divergence measured the probe corpus rather than the observer")
    assert (df["probe_n"] > 0).all()


# --------------------------------------------------------------------------- #
# The V3 §3 standing invariant.
# --------------------------------------------------------------------------- #
def test_averaged_seed_column_is_a_valid_distribution(cfg_q):
    """"Every population-averaged seed column remains a valid categorical distribution
    (non-negative, sums to 1) after averaging and renormalization. Assert it." """
    gm = gmod.build_shared_model(cfg_q, goal_symmetric=False, synth_draw_seed=17)
    results = run_chain(cfg_q, gm, POP_GOAL_DIST, contamination=0.3, signing_rate=1.0,
                        honesty=1.0, g_max=2, n_creators=6, n_artifacts=60, n_observers=3,
                        infer_steps=4, d_i=0.0, base_seed=99, population_average_seed=True)
    for r in results:
        col = r.seed_column
        assert col is not None and np.all(np.isfinite(col))
        assert np.all(col >= 0.0), "a seed column went negative"
        assert np.allclose(col.sum(axis=0), 1.0, atol=1e-10), "a seed column is not normalized"


def test_averaging_rejects_a_degenerate_column():
    """The invariant assertion must fire on a column that cannot be renormalized."""
    with pytest.raises(AssertionError):
        average_seed_column([np.full((8, 4), np.nan)])
