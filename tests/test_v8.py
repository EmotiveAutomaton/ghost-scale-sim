"""V8 nulls N35-N40, and the reductions."""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale import v8_model as V8
from ghostscale.prereg_v8 import (assert_prereg_locked_v8, build_preregistration_v8,
                                  write_preregistration_v8)
from ghostscale.v5_model import load_v5_config


@pytest.fixture(scope="module")
def cfg():
    c = load_v5_config()
    c.set("inference.exact", True)
    return c


def test_n35_every_addition_off_by_default(cfg):
    assert V8.V8Switches.from_config(cfg).all_off()


def test_n36_forgetting_never_erases():
    """Associations weaken toward a floor. They do not vanish, per the author's specification."""
    b = V8.DecayingBelief(belief=np.array([0.85, 0.05, 0.05, 0.05]),
                          baseline=np.full(4, 0.25), rate=0.2, floor=0.15)
    for _ in range(500):
        b.decay()
    assert b.drift() > 1e-4, "forgetting must leave a permanent trace, not reach zero"
    assert b.drift() < 0.05, "and it must actually fade"


def test_forgetting_is_monotone():
    b = V8.DecayingBelief(belief=np.array([0.9, 0.04, 0.03, 0.03]), baseline=np.full(4, 0.25))
    prev = b.drift()
    for _ in range(20):
        b.decay()
        now = b.drift()
        assert now <= prev + 1e-12
        prev = now


def test_n37_reader_depth_buys_nothing_where_there_is_no_hierarchy():
    """The gate on the whole reader-depth claim: no hierarchy, no advantage."""
    readings = {rd: V8.depth_reading(1, V8.ReaderHierarchy(levels=rd)) for rd in (1, 2, 3)}
    assert max(readings.values()) - min(readings.values()) == pytest.approx(0.0, abs=1e-9)


def test_reader_depth_caps_a_deep_work():
    shallow = V8.depth_reading(3, V8.ReaderHierarchy(levels=1))
    deep = V8.depth_reading(3, V8.ReaderHierarchy(levels=3))
    assert shallow < deep
    assert deep == pytest.approx(3.0)


def test_a_deep_reader_gets_no_bonus_on_shallow_work():
    """A ceiling, not a scaling: depth you did not need is not rewarded."""
    assert V8.depth_reading(1, V8.ReaderHierarchy(levels=3)) == pytest.approx(
        V8.depth_reading(1, V8.ReaderHierarchy(levels=1)))


def test_growth_requires_both_a_shortfall_and_resolution():
    """You do not acquire a master's hierarchy by staring at a master."""
    unresolved = V8.ReaderHierarchy(levels=1)
    unresolved.observe(3, resolved=0.0)
    assert unresolved.acquired == pytest.approx(0.0)

    matched = V8.ReaderHierarchy(levels=3)
    matched.observe(3, resolved=1.0)
    assert matched.acquired == pytest.approx(0.0), "no shortfall, no growth"

    learning = V8.ReaderHierarchy(levels=1)
    learning.observe(3, resolved=1.0)
    assert learning.acquired > 0.0


def test_n38_integration_cost_is_not_a_preference_over_provenance():
    """It charges for MOVEMENT, and knows nothing about who made the thing."""
    a = V8.integration_cost(np.array([0.9, 0.1]), np.array([0.5, 0.5]))
    b = V8.integration_cost(np.array([0.1, 0.9]), np.array([0.5, 0.5]))
    assert a == pytest.approx(b), "the cost must be symmetric in which way you moved"
    assert abs(V8.integration_cost(np.array([0.5, 0.5]), np.array([0.5, 0.5]))) < 1e-9


def test_n39_density_does_not_reward_an_empty_short_artifact():
    """Brevity helps a ratio, and that is stated. The numerator still has to be there."""
    readymade = V8.density(3.0, 2)
    short_empty = V8.density(0.0, 2)
    long_deep = V8.density(3.0, 24)
    assert short_empty == pytest.approx(0.0)
    assert readymade > long_deep
    assert readymade > short_empty


def test_bimodality_separates_a_split_population_from_a_spread_one():
    rng = np.random.default_rng(0)
    split = np.concatenate([rng.normal(0.1, 0.03, 60), rng.normal(2.7, 0.03, 60)])
    spread = rng.normal(1.4, 0.9, 120)
    assert V8.bimodality(split)["split"] is True
    assert V8.bimodality(spread)["split"] is False


def test_two_stage_attention_separates_shock_art_from_slop():
    t = V8.TwoStageAttention()
    shock = t.trace(1.0, 0.0, 0.02)
    slop = t.trace(0.1, 0.0, 0.02)
    assert shock["captured"] and not shock["sustained"]
    assert not slop["captured"] and not slop["sustained"]
    assert shock["peak"] - slop["peak"] > 0.3


def test_prereg_v8_hash_is_stable_and_detects_tampering(cfg, tmp_path):
    import json
    p = tmp_path / "prereg.json"
    a = write_preregistration_v8(cfg, p)
    assert a["content_hash"] == build_preregistration_v8(cfg)["content_hash"]
    assert assert_prereg_locked_v8(p)["content_hash"] == a["content_hash"]
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["H8.1"]["interaction"] = 0.0001
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_prereg_locked_v8(p)
