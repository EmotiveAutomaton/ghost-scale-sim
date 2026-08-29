"""Identity and relation tests for the V14 machinery (spec §17): the invariants the world, the
joint reader, the routes, the history/skill layer, the communication layer, the hierarchy and
the foraging layer must satisfy before any card is believed. Nothing here runs a card.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale.validation.soundingline.v14 import common as C
from ghostscale.validation.soundingline.v14 import communication as CM
from ghostscale.validation.soundingline.v14 import foraging as F
from ghostscale.validation.soundingline.v14 import hierarchy as H
from ghostscale.validation.soundingline.v14 import history_skill as HS
from ghostscale.validation.soundingline.v14 import joint as J
from ghostscale.validation.soundingline.v14 import routes as R
from ghostscale.validation.soundingline.v14 import world as W
from ghostscale.validation.soundingline.v14.cards import world_for
from ghostscale.validation.soundingline.v14.world import PLAN_DIRECT, PLAN_HABIT

NF = ("action", "semantic", "context")


@pytest.fixture(scope="module")
def world():
    return world_for({"wid": 0, "lane": "discovery", "cfg": {}, "smoke": False})


@pytest.fixture(scope="module")
def reader(world):
    return J.Reader(world, 0, 0.75, 0.8)


def test_seeds_are_stable():
    assert C.seed("card:x") == C.seed("card:x")
    assert C.seed("a") != C.seed("b")


def test_lane_lineages_disjoint():
    ids = {lane: C.lane_ids(lane, {"discovery_worlds": 256, "transfer_worlds": 128, "confirmation_worlds": 128, "pilot_worlds": 4})
           for lane in ("discovery", "transfer", "confirmation", "pilot")}
    assert C.lineage_disjoint(ids)


def test_world_is_reproducible(world):
    w2 = world_for({"wid": 0, "lane": "discovery", "cfg": {}, "smoke": False})
    for f1, f2 in zip(world.families, w2.families):
        assert np.array_equal(f1.plan, f2.plan)
        assert np.array_equal(f1.vocab, f2.vocab)


def test_posterior_is_a_distribution(world, reader):
    r = np.random.default_rng(4)
    m = W.make_maker(world, "m", r, family=0, competence="mid")
    eps = [W.episode(world, m, r, index=k) for k in range(3)]
    q = J.joint(J.uniform_prior(), reader.route_tables(eps, NF))
    assert abs(q.sum() - 1.0) < 1e-9 and (q >= 0).all()


def test_equal_weights_are_the_identity(world, reader):
    r = np.random.default_rng(5)
    m = W.make_maker(world, "m", r, family=0, competence="mid")
    eps = [W.episode(world, m, r, index=k) for k in range(2)]
    tabs = reader.route_tables(eps, NF)
    a = J.joint(J.uniform_prior(), tabs)
    b = J.joint(J.uniform_prior(), tabs, {k: 1.0 for k in NF})
    assert np.abs(a - b).max() < 1e-12


def test_equifinal_pair_is_exactly_equifinal_off_forensic(world, reader):
    r = np.random.default_rng(6)
    m = W.make_maker(world, "h", r, family=0, pref=2, plan=PLAN_HABIT, competence="mid")
    ep = W.episode(world, m, r, goal=0)
    ll = J.combined(reader.episode_tables(ep, NF))
    for pr in range(6):
        assert abs(ll[J.state_index(PLAN_DIRECT, 0, pr)] - ll[J.state_index(PLAN_HABIT, 0, pr)]) < 1e-9
    # the forensic route separates the pair in expectation (its signature is right 90% of the time;
    # a wrong signature that names a third plan scores both members equally)
    seps = []
    for _ in range(20):
        ep = W.episode(world, m, r, goal=0)
        tf = reader.ll_forensic(ep)
        seps.append(abs(tf[J.state_index(PLAN_DIRECT, 0, 2)] - tf[J.state_index(PLAN_HABIT, 0, 2)]))
    assert np.mean(seps) > 0.5


def test_template_blur_zero_is_the_family_template(world):
    rd = J.Reader(world, 0, 0.75, 0.8, template_blur=0.0)
    assert np.array_equal(rd.plan, world.family(0).plan)
    rb = J.Reader(world, 0, 0.75, 0.8, template_blur=0.5)
    assert C.entropy(rb.plan[0, 0, 0]) > C.entropy(rd.plan[0, 0, 0]) - 1e-12


def test_reliability_weights_temper_and_never_amplify(world, reader):
    r = np.random.default_rng(7)
    training = R.make_training(world, r, 6)
    w, gains = R.learn_reliability(reader, training, J.uniform_prior())
    assert max(w.values()) == 1.0 and min(w.values()) >= 0.0
    assert set(gains) == set(R.ROUTES)


def test_route_ease_is_planted_not_measured():
    for rt in R.ROUTES:
        assert R.ease(rt) == pytest.approx(R.ROUTE_COST[rt])
    assert R.ease("action", {"action": 3.0}) == pytest.approx(3.0)


def test_history_decays_by_the_planted_law(world):
    r = np.random.default_rng(8)
    m = HS.agent(world, "m", r, 0, "mid", "strong")
    HS.reverse_reward(m, 4)
    h = [W.effective_h(m, 4 + k, 0.7) for k in range(9)]
    assert h[8] == pytest.approx(h[0] * 0.7 ** 8)


def test_no_history_gives_no_early_signal(world):
    r = np.random.default_rng(9)
    m = HS.agent(world, "m", r, 0, "mid", "none")
    eps = [W.episode(world, m, r, index=k) for k in range(200)]
    assert abs(HS.history_signal(eps, m, world.family(0))) <= 0.2
    strong = HS.agent(world, "s", r, 0, "mid", "strong", h_feat=m.h_feat)
    eps_s = [W.episode(world, strong, r, index=k) for k in range(200)]
    assert HS.history_signal(eps_s, strong, world.family(0)) > 0.8


def test_matched_regions_collide_on_the_artifact():
    assert np.allclose(CM.emission_policy("contradicts", "cherry_pick"), CM.emission_policy("supports", "full"))
    assert np.allclose(CM.emission_policy("contradicts", "fabricate"), CM.emission_policy("supports", "full"))
    fan, prop = CM.REGIONS["sincere_fanatic"], CM.REGIONS["strategic_propagandist"]
    assert CM.assertion_of(fan) == CM.assertion_of(prop) == "threat"


def test_region_prior_and_posterior_are_distributions():
    p = CM.region_prior()
    assert abs(p.sum() - 1.0) < 1e-12 and (p > 0).all()
    r = np.random.default_rng(10)
    art = CM.speak(CM.source(r, "honest_warning"), r)
    post = CM.posterior(CM.loglik_artifact(art), p)
    assert abs(post.sum() - 1.0) < 1e-9
    assert sum(CM.region_posterior(post).values()) <= 1.0 + 1e-9


def test_uptake_gates_are_factored():
    post = CM.region_prior()
    fac = CM.uptake(post, 0.5, 0.8, gate="factored")
    assert fac["belief_update"] == pytest.approx(0.8)
    assert fac["policy_uptake"] == pytest.approx(0.4)
    sup = CM.uptake(post, 0.5, 0.8, gate="suppress")
    assert sup["belief_update"] == 0.0 and sup["policy_uptake"] == 0.0


def test_private_action_reveals_the_belief_when_consistent():
    r = np.random.default_rng(11)
    s = CM.source(r, "honest_warning")
    acts = [CM.private_action(s, r) for _ in range(400)]
    hits = np.mean([a["acted"] == CM.BELIEFS.index(s["belief"]) for a in acts if a["consistent"] == 1])
    assert hits > 0.8


def test_shaped_reward_is_policy_invariant_until_intervened():
    r = np.random.default_rng(12)
    reward, potential = r.normal(0, 1, H.N_PRIM), r.normal(0, 1, H.N_PRIM)
    assert np.allclose(H.policy_from_reward(H.shaped(reward, potential)), H.policy_from_reward(reward))
    pp, ps = H.resolving_intervention(reward, potential, r)
    assert np.abs(pp - ps).max() > 1e-3


def test_information_gain_falls_with_evidence_and_explained_items_go_flat():
    r = np.random.default_rng(13)
    it = F.make_item(r, "structured_learnable")
    g0 = F.expected_information_gain(it, r)
    for _ in range(30):
        F.observe(it, r)
    assert F.expected_information_gain(it, r) < g0
    ex = F.make_item(r, "novel_explained")
    F.observe(ex, r)
    assert F.expected_information_gain(ex, r) < F.ABSTAIN_FLOOR["eig_per_cost"]


def test_unlearnable_noise_shows_no_progress():
    r = np.random.default_rng(14)
    noise, learn = F.make_item(r, "unlearnable_noise"), F.make_item(r, "structured_learnable")
    for _ in range(12):
        F.observe(noise, r)
        F.observe(learn, r)
    assert F.expected_learning_progress(noise, r) <= F.expected_learning_progress(learn, r) + 0.05


def test_foraging_abstains_when_nothing_teaches():
    r = np.random.default_rng(15)
    items = [F.make_item(r, "novel_explained") for _ in range(3)]
    for it in items:
        F.observe(it, r)
    out = F.forage(items, "eig_per_cost", 8.0, r)
    assert out["spent"] == 0.0 and out["picks"] == []
