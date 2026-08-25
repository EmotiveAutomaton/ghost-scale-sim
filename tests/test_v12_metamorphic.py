"""Identity and relation tests for the V12 machinery (spec section 21).

These are the invariants the world, priors, bridge, and solvers must satisfy before any card is
believed: the default world is V11's world to the bit; information matching matches; the bridge
is the identity at zero weight; the exact posterior is a distribution; and the legacy PyMDP
reader converges on the closed-form posterior under a fixed probe.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale.config import load_config
from ghostscale.validation.soundingline.v12 import exact as X
from ghostscale.validation.soundingline.v12 import opportunities as OP
from ghostscale.validation.soundingline.v12 import pymdp_reader as PR
from ghostscale.validation.soundingline.v12 import self_other as SO
from ghostscale.validation.soundingline.v12 import uptake as U
from ghostscale.validation.soundingline.v12 import world as W
from ghostscale.validation.soundingline.v12.common import seed


@pytest.fixture(scope="module")
def world():
    return W.make_world(load_config(), rng=np.random.default_rng(1))


def test_default_world_is_v11_world(world):
    from ghostscale.v11.maker import build_maker_world, profile_family
    mw = build_maker_world(load_config())
    assert np.array_equal(world.sig, mw.sig)
    assert np.array_equal(world.synth, mw.synth)
    fam = profile_family(4)
    for n, w in fam.items():
        assert np.array_equal(world.family[n], w)


def test_parametric_default_matches_v11_signatures(world):
    p = W.WorldParams(synth_seed=world.params.synth_seed, synth_conc=world.params.synth_conc)
    assert np.allclose(W.build_signatures(p), world.sig, atol=1e-12)


def test_seeds_are_stable():
    assert seed("v12:card:x") == seed("v12:card:x")
    assert seed("a") != seed("b")


def test_information_matched_priors_match_entropy(world):
    rng = np.random.default_rng(3)
    makers = W.population(world, 12, rng)
    sp = SO.self_first_prior(world, world.family["peaked_1"])
    gp = SO.information_matched_generic(world, sp, makers)
    pp = SO.permuted_self_prior(sp, rng)
    assert abs(SO.entropy_of(sp) - SO.entropy_of(gp)) < 1e-6
    assert abs(SO.entropy_of(sp) - SO.entropy_of(pp)) < 1e-12
    assert sorted(sp.values()) == pytest.approx(sorted(pp.values()))


def test_bridge_identity_at_zero_weight(world):
    rng = np.random.default_rng(4)
    out = U.task(rng, world.ng)
    c_self = np.array([0.4, 0.3, 0.2, 0.1])
    a = U.policy(c_self, out)
    b = U.policy(U.bridge(c_self, world.family["peaked_3"], 0.0), out)
    assert np.array_equal(a, b)


def test_uniform_posterior_equals_population_prior_update(world):
    """A uniform posterior carries no direction beyond the family itself: its mean representation
    is the family mean, so the bridge update equals the population-prior update exactly."""
    rng = np.random.default_rng(5)
    c_self = np.full(world.ng, 0.25)
    w_uniform = U.representation({n: 1 / 6 for n in world.family_names}, world.family, "mean")
    fam_mean = np.mean([world.family[n] for n in world.family_names], axis=0)
    assert np.allclose(w_uniform, fam_mean / fam_mean.sum(), atol=1e-12)
    assert np.allclose(U.bridge(c_self, w_uniform, 1.0), U.bridge(c_self, fam_mean, 1.0), atol=1e-12)


def test_exact_posterior_is_a_distribution(world):
    rng = np.random.default_rng(6)
    m = W.make_maker(world, "m", "bimodal", rng)
    arts = W.stream(world, m, 0, rng, 5)
    cum = X.profile_loglik_cumulative(world, world.sig, None, arts, "CREATOR", "plain")
    post = X.posterior(cum, 5)
    assert abs(sum(post.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in post.values())


def test_pymdp_reader_matches_exact_under_fixed_probe(world):
    rng = np.random.default_rng(7)
    K = len(world.family_names)
    ems = np.stack([X.reader_emission(world, world.sig, None, world.family[n], 0, "CREATOR",
                                      "plain", n) for n in world.family_names])
    E = np.stack([ems, ems])
    ag = PR.build_reader(E, np.full(K, 1 / K), probe_costs=np.zeros(2))
    m = W.make_maker(world, "m", "peaked_0", rng)
    feats = W.artifact(world, m, 0, rng)["features"][:8]
    q = PR.observe_sequence(ag, feats, 0)
    ex = PR.exact_sequence_posterior(E, np.full(K, 1 / K), feats, 0)
    assert np.abs(q - ex).max() < 1e-6


def test_choice_posterior_prefers_planted_profile_with_enough_evidence(world):
    rng = np.random.default_rng(8)
    cw = OP.ChoiceWorld(world.ng, world.family, world.family_names)
    recs = OP.stream_choices(cw, world.family["peaked_2"], rng, 80)
    post = OP.profile_posterior_from_choices(cw, recs)
    assert max(post, key=post.get) == "peaked_2"


def test_regime_construction_matches_surface(world):
    """Bard, neutral, and concealer realizations share pair mass and entropy exactly."""
    from ghostscale.validation.soundingline.v12.world import realization
    base = world.sig[0]
    pair = np.argsort(base)[-2:]
    reals = [realization(world, base, 0, s) for s in range(len(world.family_names))]
    masses = [r[pair].sum() for r in reals]
    ents = [float(-(r[r > 0] * np.log(r[r > 0])).sum()) for r in reals]
    assert np.allclose(masses, masses[0], atol=1e-12)
    assert np.allclose(ents, ents[0], atol=1e-9)
