"""V5-5: the original chain builder collides goals 0 and 3; the v5b builder must not.

The first test DOCUMENTS the defect rather than fixing it: closed versions keep the original
builder, so the collision is part of the committed record and this test is what stops anyone
"fixing" it in place and silently changing what v5–v10 mean. If it ever fails, somebody edited
the closed builder, and that is the emergency it looks like.
"""
from __future__ import annotations

import numpy as np

from ghostscale.v5_model import (build_subgoal_chains, build_subgoal_chains_v5b,
                                 goal_mode_permutations, stationary)

NG, NS, DWELL = 4, 4, 9.2


def test_v5_collision_is_still_there_and_documented():
    old = build_subgoal_chains(NG, NS, DWELL)
    assert np.allclose(old[0], old[3]), (
        "goals 0 and 3 no longer share a chain in the ORIGINAL builder. The closed versions' "
        "worlds were built with the collision; changing the original builder rewrites them. "
        "Deviation V5-5 documents it; build_subgoal_chains_v5b is the fix for new work.")


def test_v5b_chains_are_pairwise_distinct_and_stochastic():
    new = build_subgoal_chains_v5b(NG, NS, DWELL)
    for g1 in range(NG):
        for g2 in range(g1 + 1, NG):
            assert not np.allclose(new[g1], new[g2])
    assert np.allclose(new.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(new.sum(axis=2), 1.0, atol=1e-12)


def test_v5b_stationary_is_uniform():
    new = build_subgoal_chains_v5b(NG, NS, DWELL)
    for g in range(NG):
        assert np.allclose(stationary(new[g]), np.full(NS, 1.0 / NS), atol=1e-9), (
            "a doubly stochastic chain must keep the uniform stationary distribution, or the "
            "sub-goal marginal leaks the goal into a time-averaged histogram")


def test_v5b_successors_are_disjoint_from_emission_derangements():
    new = build_subgoal_chains_v5b(NG, NS, DWELL)
    emission = {tuple(p) for p in goal_mode_permutations(NG, NS)}
    for g in range(NG):
        succ = tuple(int(np.argmax(np.where(np.eye(NS)[s] == 0, new[g][:, s], -1)))
                     for s in range(NS))
        assert succ not in emission, (
            "a goal's order channel reuses its emission derangement; the two channels must stay "
            "uncoupled or order evidence and emission evidence stop being independent")
