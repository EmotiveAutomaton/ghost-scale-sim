"""V5 null conditions — the C1 depth construction.

Every one of these guards a way the depth construction could silently stop meaning what it
says. Three of them exist because the failure they check for ACTUALLY HAPPENED during the
build and was caught by an assertion rather than by a result: a mode family whose members were
near-duplicates, mode mass landing in the foreign block, and mode permutations that left one
goal unable to tell two depths apart.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale import foreign as FN
from ghostscale.v5_model import (MU_LEVELS, build_execution_modes, build_subgoal_chains,
                                 build_v5_world, goal_mode_permutations, load_v5_config,
                                 mode_separation, stationary)


@pytest.fixture(scope="module")
def world():
    cfg = load_v5_config()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))
    return build_v5_world(cfg)


# --------------------------------------------------------------------------- #
# N21a — mu = 1 IS V4. The boundary the whole construction hangs from.
# --------------------------------------------------------------------------- #
def test_mu1_reproduces_v4_elementwise(world):
    """V5 decision 4. Not 'approximately': the construction reduces to V4 algebraically, so
    any nonzero difference is a wiring error rather than a numerical one."""
    d = world.diagnostics["mu1_vs_v4"]
    assert d["max_abs_delta_from_v4_A0"] == 0.0
    assert d["checked_subgoal_slices"] == 64


def test_beta_reduces_to_v4_5_sig_explore(world):
    """V5's beta generalises V4.5's rather than replacing it, exactly."""
    assert world.diagnostics["beta_vs_v4_5"]["max_abs_delta_from_sig_explore"] == 0.0


# --------------------------------------------------------------------------- #
# N21b — depth is CORRELATIONAL. A counting reader must not be able to see it.
# --------------------------------------------------------------------------- #
def test_execution_modes_average_exactly_to_sig(world):
    """If the modes did not average back to sig[g], a deep artifact's feature histogram would
    differ from a shallow one's and depth would be readable off a bare feature count — which
    is legibility, the thing kappa_p already measures."""
    for mi in range(len(MU_LEVELS)):
        for g in range(FN.NUM_REAL_GOALS):
            assert np.allclose(world.subsig[mi, g].mean(axis=0),
                               world.sigs.sig_true[g], atol=1e-12)


def test_depth_invisible_in_the_flat_likelihood(world):
    dm = world.diagnostics["depth_marginal"]
    assert dm["subgoal_marginal_deviation"] < 1e-12
    assert dm["full_marginal_deviation"] < 1e-12


def test_depth_is_visible_in_the_goal_marginal(world):
    """The E28 analogue, and the one quantity here that is a MEASUREMENT rather than a design
    invariant. For beta it was zero BY CONSTRUCTION, which left finite-sample luck as the only
    evidence channel and produced E28's compression at both ends. It must not be zero here."""
    assert world.diagnostics["depth_marginal"]["goal_marginal_deviation"] > 0.01


# --------------------------------------------------------------------------- #
# N21c — the plan must not leak into a time-averaged histogram.
# --------------------------------------------------------------------------- #
def test_chains_are_doubly_stochastic_with_uniform_stationary(world):
    for g in range(world.chains.shape[0]):
        assert np.allclose(world.chains[g].sum(axis=0), 1.0, atol=1e-12)
        assert np.allclose(world.chains[g].sum(axis=1), 1.0, atol=1e-12)
        assert np.allclose(stationary(world.chains[g]),
                           1.0 / world.n_subgoals, atol=1e-9)


def test_goal_never_moves_within_an_episode(world):
    """B[1] is block diagonal by (mu, goal). V1-V4's identity B on the goal, preserved: no
    probability mass may cross between goals or between depths."""
    B = np.asarray(world.gm.B[1])[:, :, 0]
    n_sub, ng, n_mu = world.n_subgoals, FN.NUM_REAL_GOALS, len(MU_LEVELS)
    for blk in range(n_mu * ng):
        lo = blk * n_sub
        block = B[lo:lo + n_sub, lo:lo + n_sub]
        assert np.isclose(block.sum(), n_sub, atol=1e-12), "mass escaped its (mu, goal) block"


# --------------------------------------------------------------------------- #
# N21d — the three faults that actually occurred during the build.
# --------------------------------------------------------------------------- #
def test_modes_are_separated(world):
    """The first mode family built cleared every other assertion at 0.019 nats and carried
    almost no recoverable depth."""
    sep = world.diagnostics["mode_separation"]
    assert sep["min_pairwise_js_across_goals"] >= sep["floor"]


def test_modes_stay_out_of_the_foreign_block(world):
    """assert_c1_properties only inspects sig_true, so without this the depth construction
    could break V4's partition and every C1 property would still report clean."""
    assert world.diagnostics["mode_separation"]["max_mode_mass_in_foreign_block"] < 0.10


def test_no_goal_gets_the_identity_permutation():
    """Cyclic shifts gave goal 0 a shift of zero, so its mu = 2 and mu = 3 emissions were
    identical and that goal could not discriminate the top two depths at all."""
    perms = goal_mode_permutations(FN.NUM_REAL_GOALS, 4)
    for g, p in enumerate(perms):
        assert not np.array_equal(p, np.arange(4)), f"goal {g} kept the identity permutation"
    assert len({tuple(p) for p in perms}) == len(perms)


def test_mu2_and_mu3_emissions_differ_for_every_goal(world):
    """The observable consequence of the permutation fault, checked directly."""
    for g in range(FN.NUM_REAL_GOALS):
        gap = float(np.max(np.abs(world.subsig[1, g] - world.subsig[2, g])))
        assert gap > 0.1, f"goal {g} cannot tell mu = 2 from mu = 3 (gap {gap:.4f})"


# --------------------------------------------------------------------------- #
# N21e — the block geometry E30's result would otherwise be measuring.
# --------------------------------------------------------------------------- #
def test_three_to_four_blocks_fit_an_artifact(world):
    d = world.diagnostics["dwell"]
    assert d["window"][0] <= d["expected_blocks"] <= d["window"][1]


def test_delta_zero_collapses_the_mode_family():
    """E30's negative control has to BE a negative control: at delta = 0 a deep artifact must
    be identical to a shallow one, or the control cannot detect a hypothesis-space artifact."""
    cfg = load_v5_config()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))
    cfg.set("v5.depth.delta", 0.0)
    w = build_v5_world(cfg, enforce_mode_separation=False)
    for mi in range(len(MU_LEVELS)):
        for g in range(FN.NUM_REAL_GOALS):
            for s in range(w.n_subgoals):
                assert np.allclose(w.subsig[mi, g, s], w.sigs.sig_true[g], atol=1e-12)


def test_mode_family_is_normalised_and_nonnegative():
    cfg = load_v5_config()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))
    sig = FN.build_human_signatures(cfg)
    modes = build_execution_modes(sig[0], 4, support=np.asarray(FN.HUMAN_FEATURES))
    assert np.all(modes >= 0.0)
    assert np.allclose(modes.sum(axis=1), 1.0, atol=1e-10)
    assert mode_separation(modes) > 0.0


def test_chain_dwell_matches_its_parameter():
    chains = build_subgoal_chains(4, 4, dwell=9.2)
    for g in range(4):
        assert np.isclose(chains[g].diagonal().mean(), 1.0 - 1.0 / 9.2, atol=1e-12)
