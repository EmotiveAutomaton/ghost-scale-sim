"""V4 nulls and invariants (V4 spec §3).

    N17 — EXPLORE is not a free parameter   -> here
    N18 — foreign content is goal-directed  -> here
    C1  — the three construction properties -> here (each one C1 names gets a test)

N16 (boundary regression against V2/V3) belongs to stage 3 and is not implemented yet, because
it needs E20's omega sweep to have something to regress. It is named in RESULTS_V4.md as
outstanding rather than quietly dropped.

N19 and N20 guard C4 and C3, neither of which stage 1 builds.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ghostscale import foreign as FN
from ghostscale import prereg_v4 as P4
from ghostscale.v4_model import build_v4_world, load_v4_config, real_goal_posterior

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture
def cfg():
    return load_v4_config(include_explore=True)


# --------------------------------------------------------------------------- #
# C2 / N17 — EXPLORE.
#
# The V4 spec forbids a uniform-over-features EXPLORE in as many words, and at V1-V3
# cardinality the construction it prescribes produced exactly that. These tests are the reason
# that cannot recur silently.
# --------------------------------------------------------------------------- #
def test_explore_would_have_been_uniform_at_v1_v3_cardinality():
    """The measurement that forced the feature space to double, kept as a live test.

    This is not a test of V4 code. It is the record of WHY V4 does not run at F = 8, executable
    so that anyone who proposes reverting the partition sees the consequence immediately.
    """
    from ghostscale.config import load_config
    from ghostscale import generative_model as gmod
    v3 = load_config()
    sig = gmod.build_goal_signatures(v3)          # 4 goals tiling 8 features
    explore = sig.mean(axis=0)
    explore = explore / explore.sum()
    assert np.max(np.abs(explore - 1.0 / 8.0)) < 1e-12, (
        "the V1-V3 goal partition no longer produces an exactly-uniform mean; the argument "
        "in foreign.py for doubling the feature space needs rechecking")
    synth = gmod.build_noise_free_synth(v3)
    kl_to_explore = float(np.sum(synth * np.log(synth / explore)))
    kl_to_goal = float(np.sum(synth * np.log(synth / sig[0])))
    assert kl_to_explore < kl_to_goal, (
        "at V1-V3 cardinality EXPLORE sits CLOSER to synthetic content than a real goal does, "
        "which is why E19 could only ever have returned CRASH_IS_AN_ARTIFACT there")


def test_N17_explore_is_not_globally_uniform(cfg):
    """C2's own requirement, at V4 cardinality. Must pass, and must be capable of failing."""
    sig_true = FN.build_human_signatures(cfg)
    ex = FN.build_explore_signature(sig_true)
    stats = FN.assert_explore_is_not_globally_uniform(ex)
    assert stats["linf_from_uniform"] > 1e-3
    # Human-shaped: nearly all its mass is in the block the observer's goals live in.
    assert stats["human_block_mass"] > 0.85
    with pytest.raises(AssertionError, match="uniform over all"):
        FN.assert_explore_is_not_globally_uniform(np.full(FN.NUM_FEATURES_V4,
                                                          1.0 / FN.NUM_FEATURES_V4))


def test_N17_explore_is_flat_within_the_human_block(cfg):
    """Goal-agnostic, not a fifth specific goal. Under the FEP the exploring agent flattens
    within its own policy space, which is exactly this."""
    ex = FN.build_explore_signature(FN.build_human_signatures(cfg))
    human = ex[FN.HUMAN_FEATURES]
    assert float(human.max() - human.min()) < 1e-9
    # A signature peaked on one human pair must be rejected as EXPLORE.
    fake = np.full(FN.NUM_FEATURES_V4, 0.01)
    fake[0] = fake[1] = 0.46
    fake = fake / fake.sum()
    with pytest.raises(AssertionError, match="not flat across the human block"):
        FN.assert_explore_is_not_globally_uniform(fake)


def test_N17_matches_the_preregistered_array(cfg, tmp_path):
    """The lock itself. Writing, verifying, and then detecting a change."""
    path = tmp_path / "v4_preregistration.json"
    P4.write_preregistration_v4(cfg, path)
    stats = P4.assert_explore_matches_prereg(cfg, path)
    assert stats["max_abs_delta"] <= P4.EXPLORE_ATOL

    # Tampering with the recorded signature must be caught.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["C2_explore"]["signature"][0] += 0.01
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified since it was written"):
        P4.assert_explore_matches_prereg(cfg, path)


def test_N17_prereg_refuses_a_conflicting_rewrite(cfg, tmp_path):
    """Re-registering DIFFERENT criteria over an existing file must raise, not overwrite.
    V4 spec §7 names this as the most likely way the build goes wrong."""
    path = tmp_path / "v4_preregistration.json"
    P4.write_preregistration_v4(cfg, path)
    moved = load_v4_config(include_explore=True,
                           overrides={"v4.foreign.draw_seed": 999})
    with pytest.raises(RuntimeError, match="DIFFERENT content hash"):
        P4.write_preregistration_v4(moved, path)


# --------------------------------------------------------------------------- #
# N18 / C1 — the foreign family.
# --------------------------------------------------------------------------- #
def test_N18_foreign_content_is_goal_directed(cfg):
    """C1 property 1, the one that distinguishes V4 from V3. Foreign content must be about as
    goal-informative as human content, not less."""
    sigs = FN.build_v4_signatures(cfg, omega=0.0, include_explore=True)
    d = sigs.diagnostics
    assert d["mi_features_foreign_goal"] >= d["mi_floor_applied"]
    assert d["foreign_over_human_mi_ratio"] > 0.85, (
        "foreign content is materially less goal-directed than human content, which is the "
        "silent reversion to V3's goal-empty model that V4 spec §6 forbids")


def test_N18_fires_on_a_goal_empty_foreign_family(cfg):
    """The guard must be capable of failing. A V3-style goal-EMPTY family (every foreign goal
    emitting the same distribution) has to be rejected."""
    sigs = FN.build_v4_signatures(cfg, omega=0.0, include_explore=True)
    flat = np.repeat(sigs.foreign_basis[0][None, :], FN.NUM_REAL_GOALS, axis=0)
    sigs.foreign_basis = flat
    with pytest.raises(AssertionError, match="N18/C1 property 1"):
        FN.assert_c1_properties(cfg, sigs)


def test_C1_property2_observer_cannot_read_foreign_content(cfg):
    """The observer's own likelihoods are identical over the foreign block, so a foreign
    feature carries no information about which observer-goal produced it."""
    sigs = FN.build_v4_signatures(cfg, omega=0.0, include_explore=True)
    assert sigs.diagnostics["mi_features_observer_goal_on_foreign"] < 1e-6


def test_C1_property3_foreign_content_is_structured_not_noise(cfg):
    """Carried forward from V1 §9 N6. Low MI with HIGH entropy is the noise strawman; the
    reframe needs low observer-MI with LOW entropy."""
    sigs = FN.build_v4_signatures(cfg, omega=0.0, include_explore=True)
    d = sigs.diagnostics
    assert d["max_foreign_signature_entropy"] < d["uniform_entropy_16"] * 0.6


def test_C1_partition_is_disjoint(cfg):
    """Pre-mortem failure #1: if the two blocks overlap, foreign content is unidentifiable
    rather than foreign and V4 reports V3's results in new vocabulary."""
    sigs = FN.build_v4_signatures(cfg, omega=0.0, include_explore=True)
    assert sigs.diagnostics["human_mass_in_foreign_block"] < 0.10
    assert sigs.diagnostics["foreign_mass_in_human_block"] < 0.10


def test_omega_1_recovers_the_human_family(cfg):
    """The overlap parameter has to mean what it says at its endpoint. This is the seed of
    N16: at omega = 1 there is no foreign content, only human content."""
    sigs = FN.build_v4_signatures(cfg, omega=1.0, include_explore=True)
    assert np.allclose(sigs.sig_foreign, sigs.sig_true, atol=1e-9)


def test_omega_interpolates_monotonically(cfg):
    """Between the endpoints, more overlap must mean more readable. Guards against an omega
    that does nothing, which would make E20 vacuous."""
    prev = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        sigs = FN.build_v4_signatures(cfg, omega=w, include_explore=True)
        # Mass the foreign family places in the block the observer can actually read.
        readable = float(sigs.sig_foreign[:, FN.HUMAN_FEATURES].sum(axis=1).mean())
        if prev is not None:
            assert readable > prev, f"omega={w} did not increase in-family mass"
        prev = readable


# --------------------------------------------------------------------------- #
# The E19 criteria themselves. They must be able to return every outcome.
# --------------------------------------------------------------------------- #
def test_e19_absorption_is_conjunctive():
    """Mass alone is not absorption. An observer that parks probability on the vaguest
    hypothesis available while staying uncertain has not explained anything."""
    assert P4.explore_absorbed(0.9, 0.1)["absorbed"]
    assert not P4.explore_absorbed(0.9, 1.4)["absorbed"], "high mass, but uncertain"
    assert not P4.explore_absorbed(0.2, 0.1)["absorbed"], "flat-baseline mass"


def test_e19_absorption_ignores_engagement():
    """Deviation 2. Disengagement is what SUCCESS looks like in this model, so it must not be
    a clause of absorption. The canonical success case (human_directed) resolves the goal
    completely and then correctly stops paying, scoring 0.000 on engagement."""
    resolved_and_stopped = P4.explore_absorbed(0.9, 0.1, 0.0)
    still_looking = P4.explore_absorbed(0.9, 0.1, 1.0)
    assert resolved_and_stopped["absorbed"] and still_looking["absorbed"]
    assert resolved_and_stopped["absorbed"] == still_looking["absorbed"], (
        "engagement must not change the absorption verdict; if it does, the criterion has "
        "reacquired the clause that failed E19's own positive control")


def test_crash_signature_separates_success_from_giving_up():
    """The crash was never 'disengages'. It is 'disengages WITHOUT having resolved anything'."""
    # Resolved the goal, then stopped paying. This is success, not a crash.
    assert not P4.crash_signature(0.05, 0.0)["crashed"]
    # Stopped paying while still having no idea. This is the crash.
    assert P4.crash_signature(1.5, 0.0)["crashed"]
    # Still looking, still lost. Not a crash: it has not given up.
    assert not P4.crash_signature(1.5, 0.9)["crashed"]


def test_e19_verdict_can_return_each_outcome():
    """The decisive experiment has to be able to say the framework is wrong (V4 spec §6)."""
    absorbed = P4.explore_absorbed(0.9, 0.1)
    rejected = P4.explore_absorbed(0.1, 1.5)

    survives = P4.e19_verdict({"human_exploratory/explore_on": absorbed,
                               "foreign/explore_on": rejected})
    assert survives["verdict"] == "CRASH_SURVIVES"

    artifact = P4.e19_verdict({"human_exploratory/explore_on": absorbed,
                               "foreign/explore_on": absorbed})
    assert artifact["verdict"] == "CRASH_IS_AN_ARTIFACT"

    inconclusive = P4.e19_verdict({"human_exploratory/explore_on": rejected,
                                   "foreign/explore_on": rejected})
    assert inconclusive["verdict"] == "INCONCLUSIVE", (
        "with a failed positive control, EXPLORE explains nothing at all, so its failure on "
        "foreign content carries no information and must not be credited as a win")


# --------------------------------------------------------------------------- #
# D4 — the posterior arity rule, fixed before E19 rather than after.
# --------------------------------------------------------------------------- #
def test_D4_real_goal_posterior_marginalises_explore():
    q = np.array([0.1, 0.2, 0.1, 0.1, 0.5])       # 0.5 on EXPLORE
    real = real_goal_posterior(q)
    assert real.size == FN.NUM_REAL_GOALS
    assert np.isclose(real.sum(), 1.0)
    assert np.allclose(real, np.array([0.2, 0.4, 0.2, 0.2]))


def test_D4_is_a_no_op_without_explore():
    q = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(real_goal_posterior(q), q)


# --------------------------------------------------------------------------- #
# The assembled world.
# --------------------------------------------------------------------------- #
def test_v4_world_builds_and_preferences_are_zero(cfg):
    """Null N7 carried into V4. A smuggled preference over features or provenance would make
    every V4 result worthless in exactly the way it would a V1 one."""
    for include_explore in (False, True):
        c = load_v4_config(include_explore=include_explore)
        w = build_v4_world(c, omega=0.0, include_explore=include_explore)
        A0 = np.asarray(w.gm.A[0])
        assert A0.shape == (16, 4, 4 + int(include_explore), 2)
        assert np.all(np.asarray(w.gm.C[0]) == 0.0)
        assert np.all(np.asarray(w.gm.C[1]) == 0.0)
        assert np.allclose(A0.sum(axis=0), 1.0)


def test_v1_v3_artifacts_are_untouched_by_the_foreign_path():
    """The Environment change is additive. An artifact built the V1-V3 way must take the
    V1-V3 branch, and it does because ``foreign_goal`` defaults to -1."""
    from ghostscale.environment import Artifact
    a = Artifact(provenance=0, goal=1, declared_signal=0)
    assert a.foreign_goal == -1
