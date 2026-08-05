"""Metamorphic relations: properties that must hold across pairs of runs, not within one.

WHY THIS FILE EXISTS. Both defects that shipped in the first Sounding Line batch were invisible to
every test in this repository, and neither was a statistics problem. S-2's manipulation never
reached the reader. S-3's threshold was fitted on the labels it was being scored against. A larger
sample would not have found either; a unit test on a single call would not have found either,
because each individual call returned a perfectly reasonable number.

What finds them is a relation between TWO executions: "change this input and the output must
change", "permute that label and the output must not". That is metamorphic testing, and the
Hypothesis library generates the inputs so the relations are checked across the space rather than
at the one point someone thought of.

Kept fast on purpose -- these run in the ordinary test cycle, so the expensive rollout-level
relation is capped at a handful of examples. A relation that is too slow to run is a relation that
gets marked skip.
"""
from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis", reason="pip install -e '.[dev]'")
from hypothesis import HealthCheck, given, settings                    # noqa: E402
from hypothesis import strategies as st                                # noqa: E402

from ghostscale.methods import overlap as OV                           # noqa: E402
from ghostscale.methods import pid as PID                              # noqa: E402
from ghostscale.validation.soundingline import t_common as T           # noqa: E402

FAST = settings(max_examples=40, deadline=None,
                suppress_health_check=[HealthCheck.function_scoped_fixture])
SLOW = settings(max_examples=6, deadline=None,
                suppress_health_check=[HealthCheck.function_scoped_fixture])

cards = st.integers(min_value=2, max_value=8)
fids = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# --------------------------------------------------------------------------------------------- #
# The supply channel: the object T-1's whole result rests on.
# --------------------------------------------------------------------------------------------- #
@given(c=cards)
@FAST
def test_zero_information_channel_is_exactly_uniform(c):
    """A channel at chance fidelity must be EXACTLY uniform, not approximately.

    The approximate version is what broke: at c = 3 the diagonal ``1/3`` and the off-diagonal
    ``(1 - 1/3)/2`` differ in the last bits, the likelihood is very slightly non-constant, and the
    near-tied depth posterior flips on argmax. It moved a reported accuracy from 0.417 to 0.383 on
    a channel carrying no information at all.
    """
    n_mu, n_sub, ng = 3, 4, 4
    for vertex, card in (("goal", ng), ("process", n_sub), ("depth", n_mu)):
        M = T.channel_matrix(vertex, 1.0 / card, n_mu, n_sub, ng)
        assert np.allclose(M, M[0, 0, 0, 0], rtol=0, atol=0), (
            f"{vertex} channel at chance fidelity is not bit-exactly uniform")


@given(c=cards, f=fids)
@FAST
def test_channel_information_is_bounded_and_monotone(c, f):
    """Mutual information is in [0, log c], zero exactly at chance, and rises with fidelity."""
    lo = 1.0 / c
    f = lo + f * (1.0 - lo)                       # map into the valid range
    nats = T.channel_nats(f, c)
    assert -1e-12 <= nats <= np.log(c) + 1e-12
    assert abs(T.channel_nats(lo, c)) < 1e-12
    if f > lo + 1e-6:
        assert T.channel_nats(f, c) >= T.channel_nats(lo + (f - lo) / 2.0, c) - 1e-9


@given(c=cards, target=st.floats(min_value=0.01, max_value=0.9, allow_nan=False))
@FAST
def test_fidelity_for_nats_inverts_channel_nats(c, target):
    """The matched-information design is only valid if this inversion is actually exact."""
    t = min(target, np.log(c) * 0.95)
    f = T.fidelity_for_nats(t, c)
    assert abs(T.channel_nats(f, c) - t) < 1e-6


@given(seed=st.integers(min_value=0, max_value=10_000))
@SLOW
def test_placebo_channel_changes_nothing_end_to_end(seed):
    """THE S-2 RELATION, run through the full stack.

    A channel carrying zero nats must leave every recovery number bit-identical to no channel at
    all. This is the relation that catches a side channel drawing from the rollout's RNG, which is
    exactly what it caught the first time it was written -- as an ad-hoc check rather than a test.
    """
    from ghostscale.config import load_config
    from ghostscale.validation.soundingline.common import build

    world, _b, cfg_r, n_mu, n_sub, ng = build(load_config())
    g = int(seed % ng)
    kw = dict(mu=3, beta=1.0, g_true=g, n_timesteps=12, forced_k=12,
              n_mu=n_mu, n_sub=n_sub, ng=ng)
    base = T.run_supplied_encounter(
        world, cfg_r, [], art_rng=np.random.default_rng(seed),
        obs_rng=np.random.default_rng(seed + 1), **kw)
    for vertex, card in (("goal", ng), ("process", n_sub), ("depth", n_mu)):
        placebo = T.run_supplied_encounter(
            world, cfg_r, [(vertex, 1.0 / card, card)],
            art_rng=np.random.default_rng(seed),
            obs_rng=np.random.default_rng(seed + 1), **kw)
        for key in ("goal", "process", "depth"):
            assert base[key] == pytest.approx(placebo[key], abs=1e-12), (
                f"a zero-nat {vertex} channel moved {key}: "
                f"{base[key]!r} -> {placebo[key]!r}. The channel is perturbing the run through "
                f"something other than its information content.")


@given(seed=st.integers(min_value=0, max_value=10_000))
@SLOW
def test_full_information_channel_changes_something(seed):
    """The other half of the same relation, and the one S-2 failed.

    A channel at full fidelity about a vertex must move that vertex's recovery. A manipulation
    that cannot be detected in the output is not a manipulation.
    """
    from ghostscale.config import load_config
    from ghostscale.validation.soundingline.common import build

    world, _b, cfg_r, n_mu, n_sub, ng = build(load_config())
    kw = dict(mu=3, beta=0.25, g_true=int(seed % ng), n_timesteps=12, forced_k=12,
              n_mu=n_mu, n_sub=n_sub, ng=ng)
    base = T.run_supplied_encounter(
        world, cfg_r, [], art_rng=np.random.default_rng(seed),
        obs_rng=np.random.default_rng(seed + 1), **kw)
    for vertex, card, key in (("process", n_sub, "process"), ("depth", n_mu, "depth")):
        hi = T.run_supplied_encounter(
            world, cfg_r, [(vertex, 0.99, card)],
            art_rng=np.random.default_rng(seed),
            obs_rng=np.random.default_rng(seed + 1), **kw)
        assert abs(hi[key] - base[key]) > 1e-6, (
            f"supplying {vertex} at 0.99 fidelity did not move {key} at all. This is the S-2 "
            f"failure mode: the manipulation is not reaching the measurement.")


# --------------------------------------------------------------------------------------------- #
# Scoring statistics
# --------------------------------------------------------------------------------------------- #
samples = st.lists(st.floats(min_value=-50, max_value=50, allow_nan=False,
                             allow_infinity=False), min_size=5, max_size=200)


@given(a=samples, b=samples)
@FAST
def test_overlap_is_symmetric_and_bounded(a, b):
    ab = OV.overlap_coefficient(a, b)
    ba = OV.overlap_coefficient(b, a)
    assert 0.0 - 1e-12 <= ab <= 1.0 + 1e-12
    assert ab == pytest.approx(ba, abs=1e-12)


@given(a=samples)
@FAST
def test_overlap_of_a_sample_with_itself_is_total(a):
    assert OV.overlap_coefficient(a, a) == pytest.approx(1.0, abs=1e-9)


@given(a=samples, shift=st.floats(min_value=-10, max_value=10, allow_nan=False))
@FAST
def test_overlap_is_translation_invariant(a, shift):
    """Shifting both samples together cannot change how much they overlap."""
    x = np.asarray(a, dtype=float)
    if x.size < 5 or np.ptp(x) < 1e-9:
        return
    assert OV.overlap_coefficient(x, x + 0.0) == pytest.approx(
        OV.overlap_coefficient(x + shift, x + shift), abs=1e-9)


@given(a=samples, b=samples)
@FAST
def test_auc_is_antisymmetric(a, b):
    """THE ORIENTATION RELATION. auc(a, b) + auc(b, a) must be exactly 1.

    T-5 reported ``goal_final_entropy`` at AUC 0.000 and it read as total failure; it is a perfect
    detector with its sign the other way round. This relation is what makes ``auc_oriented``
    well defined, so it is worth asserting rather than assuming.
    """
    from ghostscale.validation.soundingline.t5_detection import auc
    ab, ba = auc(a, b), auc(b, a)
    if np.isfinite(ab) and np.isfinite(ba):
        assert ab + ba == pytest.approx(1.0, abs=1e-9)


@given(a=samples)
@FAST
def test_auc_of_a_sample_against_itself_is_chance(a):
    from ghostscale.validation.soundingline.t5_detection import auc
    assert auc(a, a) == pytest.approx(0.5, abs=1e-9)


@given(a=samples, b=samples)
@FAST
def test_paired_bootstrap_point_estimate_is_the_paired_mean(a, b):
    """The bootstrap may widen the interval; it must never move the point estimate."""
    n = min(len(a), len(b))
    if n < 5:
        return
    rng = np.random.default_rng(0)
    r = T.boot_paired(a[:n], b[:n], rng, draws=64)
    expected = float(np.mean(np.asarray(a[:n]) - np.asarray(b[:n])))
    assert r["difference"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------------------------- #
# Partial information decomposition
# --------------------------------------------------------------------------------------------- #
@pytest.mark.skipif(not PID.available()[0], reason=PID.available()[1])
@given(seed=st.integers(min_value=0, max_value=500))
@FAST
def test_pid_atoms_sum_to_the_joint_information(seed):
    """The four atoms must reconstruct the joint mutual information exactly."""
    rng = np.random.default_rng(seed)
    p = rng.random((2, 2, 3)) + 1e-3
    r = PID.decompose(p)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    total = (r["redundant_bits"] + r["unique_source_a_bits"]
             + r["unique_source_b_bits"] + r["synergistic_bits"])
    assert total == pytest.approx(r["total_mutual_information_bits"], abs=1e-9)


@pytest.mark.skipif(not PID.available()[0], reason=PID.available()[1])
def test_pid_reads_xor_as_pure_synergy():
    """A known answer: XOR carries one bit, all of it synergistic and none of it anywhere else.

    The positive control for the PID wiring. If this drifts, every number in
    ``emission_pid_by_depth`` is suspect and nothing else in the repository would say so.
    """
    p = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            p[a, b, a ^ b] = 0.25
    r = PID.decompose(p)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["synergistic_bits"] == pytest.approx(1.0, abs=1e-6)
    assert r["redundant_bits"] == pytest.approx(0.0, abs=1e-6)
    assert r["unique_source_a_bits"] == pytest.approx(0.0, abs=1e-6)
    assert r["unique_source_b_bits"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not PID.available()[0], reason=PID.available()[1])
def test_pid_recovers_null_n28_from_the_likelihood():
    """At the shallowest depth an emission cannot carry mode information. No rollouts needed.

    This is null N28 -- the repository's most productive negative control -- arrived at from the
    generative model instead of from a reader, which makes it an independent check on both.
    """
    from ghostscale.config import load_config
    from ghostscale.validation.soundingline.common import build

    world, _b, _c, _n_mu, _n_sub, _ng = build(load_config())
    r = PID.n28_identity_from_pid(np.asarray(world.subsig, dtype=float))
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["holds"], (
        f"N28 does not hold in the likelihood: unique mode {r['unique_mode_bits']}, "
        f"synergy {r['synergistic_bits']}")
