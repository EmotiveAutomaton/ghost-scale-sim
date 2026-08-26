"""Identity and relation tests for the V13 machinery: the invariants the world, priors,
attention, trust and hierarchy must satisfy before any card is believed.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale.validation.soundingline.v13 import common as C
from ghostscale.validation.soundingline.v13 import exact as X
from ghostscale.validation.soundingline.v13 import goals_trust as GT
from ghostscale.validation.soundingline.v13 import hierarchy as H
from ghostscale.validation.soundingline.v13 import priors as P
from ghostscale.validation.soundingline.v13 import world as W


@pytest.fixture(scope="module")
def world():
    return W.make_world(0, "discovery")


@pytest.fixture(scope="module")
def model(world):
    rd = W.make_maker(world, "rd", np.random.default_rng(1), family=0, k=0.05)
    return X.reader_model(world, rd, families=[0])


def test_seeds_are_stable():
    assert C.seed("card:x") == C.seed("card:x")
    assert C.seed("a") != C.seed("b")


def test_lane_lineages_disjoint():
    ids = {lane: C.lane_ids(lane, {"discovery_worlds": 512, "transfer_worlds": 128, "confirmation_worlds": 96, "pilot_worlds": 4})
           for lane in ("discovery", "transfer", "confirmation", "pilot")}
    assert C.lineage_disjoint(ids)


def test_world_is_reproducible(world):
    w2 = W.make_world(0, "discovery")
    for f1, f2 in zip(world.families, w2.families):
        assert np.array_equal(f1.sig, f2.sig)
        assert np.array_equal(f1.methods, f2.methods)


def test_neutral_attention_is_the_identity(world, model):
    rng = np.random.default_rng(3)
    m = W.make_maker(world, "m", rng, family=0, k=0.2)
    arts = W.stream(world, m, 0, rng, 3, n_steps=8)
    chans = ("surface", "common_structure", "group_convention", "goal_consequences")
    prior = X.uniform_prior(model)
    a = model.posterior(prior, arts, chans)
    b = model.posterior(prior, arts, chans, {c: 1.0 for c in chans})
    assert np.abs(a - b).max() == 0.0


def test_posterior_is_a_distribution(world, model):
    rng = np.random.default_rng(4)
    m = W.make_maker(world, "m", rng, family=0, k=0.2)
    arts = W.stream(world, m, 0, rng, 4)
    q = model.posterior(X.uniform_prior(model), arts, ("surface",))
    assert abs(q.sum() - 1.0) < 1e-9 and (q >= 0).all()


def test_matched_priors_match_entropy(world, model):
    rng = np.random.default_rng(5)
    makers = W.population(world, 16, rng, family=0)
    readers = [W.make_maker(world, f"r{i}", rng, family=0, k=0.05) for i in range(3)]
    selfs = [(r, P.measure_self(world, r, X.reader_model(world, r, families=[0]), np.random.default_rng(6 + i))) for i, r in enumerate(readers)]
    pri, rep = P.routes_for(model, readers[0], selfs[0][1], makers, selfs, rng, makers[0])
    H0 = C.entropy(pri["self"])
    for route in ("equal_local", "generic_local", "random_local", "permuted_self"):
        assert abs(C.entropy(pri[route]) - H0) < 1e-5, route


def test_realization_matches_surface(world):
    fam = world.family(0)
    base = fam.sig[0]
    reals = [W.realization(fam, base, s, 0.35) for s in range(fam.tail.size)]
    block = fam.blocks[0]
    masses = [r[block].sum() for r in reals]
    ents = [C.entropy(r) for r in reals]
    assert max(masses) - min(masses) < 1e-12
    assert max(ents) - min(ents) < 1e-9


def test_goal_kind_distributions_are_mirror_images():
    d0, d1 = GT.kind_dists(np.random.default_rng(7))
    assert abs(C.entropy(d0) - C.entropy(d1)) < 1e-12
    half = d0.size // 2
    assert np.allclose(d0[:half], d1[half:]) and np.allclose(d0[half:], d1[:half])


def test_uptake_channels_only_move_on_declared_edges():
    up_a = GT.uptake_decision({"accurate": 0.9, "misleading": 0.05}, 0.8, 1.0, 1.0, 0.9)
    up_b = GT.uptake_decision({"accurate": 0.9, "misleading": 0.05}, 0.8, 0.2, 1.0, 0.9)
    assert up_a["belief_update"] == up_b["belief_update"]
    assert up_a["prediction_use"] == up_b["prediction_use"]
    assert up_a["process_imitation"] != up_b["process_imitation"]


def test_central_and_shared_brief_artifacts_are_twins(world):
    a1 = H.make_team(world, np.random.default_rng(11), "central", n_subs=3, family=0)
    a2 = H.make_team(world, np.random.default_rng(11), "shared_brief", n_subs=3, family=0)
    p1 = H.produce_team(world, "central", a1, np.random.default_rng(12), n_parts=6, steps=8)
    p2 = H.produce_team(world, "shared_brief", a2, np.random.default_rng(12), n_parts=6, steps=8)
    assert np.array_equal(p1["features"], p2["features"])
    f1, f2 = H.interaction_features(p1), H.interaction_features(p2)
    assert f1["n_corrections"] == f2["n_corrections"]
    assert f1["fraction_other_actor_corrections"] > f2["fraction_other_actor_corrections"]


def test_cost_unit_invariance(world):
    from ghostscale.validation.soundingline.v13 import costs as CO
    fam = world.family(0)
    actor = CO.Actor(fam.grid[1], weights={"time": 0.7})
    recs = CO.stream(actor, np.random.default_rng(13), 5, fam.ng)
    ll0 = sum(CO.loglik(actor, r) for r in recs)
    scale = 2.5
    actor_s = CO.Actor(fam.grid[1], weights={d: w / scale for d, w in zip(CO.COST_DIMS, actor.dim_weights())},
                       risk_tolerance=0.5, social_obligation=0.5)
    ll1 = 0.0
    for r in recs:
        rr = dict(r, cost=np.asarray(r["cost"]) * scale)
        ll1 += CO.loglik(actor_s, rr)
    assert abs(ll0 - ll1) < 1e-9
