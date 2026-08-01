"""V7 nulls N31-N34, and the reductions.

The reduction tests matter because V7 adds a term to the gate that, if switched on by default,
would change every result in the repository. It is not switched on, and that is tested rather
than asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale import v6_model as V6
from ghostscale.prereg_v7 import (assert_prereg_locked_v7, build_preregistration_v7,
                                  write_preregistration_v7)
from ghostscale.v5_model import load_v5_config


@pytest.fixture(scope="module")
def cfg():
    c = load_v5_config()
    c.set("inference.exact", True)
    return c


# =========================================================================== #
# N31 — at leak zero, V7 reproduces the V6 gate exactly.
# =========================================================================== #
def test_n31_zero_leak_reproduces_the_v6_gate():
    for omega, theta in ((0.8, 0.3), (0.2, 0.7), (0.5, 0.5), (0.05, 0.95)):
        assert V6.gate(omega, theta, 8.0, leak=0.0) == pytest.approx(
            V6._sigmoid_gate(omega, theta, 8.0), abs=1e-15)


def test_leak_is_off_by_default():
    """The term exists and is not adopted. An addition that silently rewrites the record is the
    accretion problem the repair pass was written against."""
    import inspect
    sig = inspect.signature(V6.gate)
    assert sig.parameters["leak"].default == 0.0


# =========================================================================== #
# N32 — the leak PASSES rather than MANUFACTURES.
# =========================================================================== #
def test_n32_leak_is_a_floor_not_a_source():
    """A leak raises the floor of the gate. It cannot exceed one, and it cannot invert the
    gate's response to its own arguments."""
    for leak in (0.0, 0.05, 0.2, 0.5):
        closed = V6.gate(0.0, 5.0, 8.0, leak=leak)
        open_ = V6.gate(1.0, 0.0, 8.0, leak=leak)
        assert closed == pytest.approx(leak, abs=1e-6)
        assert open_ >= closed
        assert 0.0 <= closed <= open_ <= 1.0


def test_leak_is_monotone_in_itself():
    vals = [V6.gate(0.2, 0.9, 8.0, leak=lk) for lk in (0.0, 0.02, 0.05, 0.1, 0.2)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_a_fully_open_gate_is_unaffected_by_the_leak():
    """Where everything already gets through, a leak adds nothing. The leak is a floor."""
    for leak in (0.0, 0.1, 0.3):
        assert V6.gate(1.0, -5.0, 8.0, leak=leak) == pytest.approx(1.0, abs=1e-6)


# =========================================================================== #
# The graded gate already leaked, which V7 found by measuring it.
# =========================================================================== #
def test_the_sigmoid_gate_never_actually_reaches_zero():
    """Recorded as a test because it changes what V1-V5 versus V6 could claim.

    Version 6 replaced the binary engagement decision with a sigmoid, and a sigmoid has no zero.
    So the graded gate already leaked a little and nobody noticed -- E42 reported integration as
    0.00 because that is what 3e-06 looks like at two decimal places. The only versions of this
    model that could protect a reader perfectly were the ones with the binary gate.
    """
    tiny = V6._sigmoid_gate(0.0, 5.0, 8.0)
    assert 0.0 < tiny < 1e-4


# =========================================================================== #
# N33 / N34 — the E45 comparison is fair.
# =========================================================================== #
def test_n34_held_out_intentions_get_no_training_examples():
    from ghostscale.config import load_config
    from ghostscale.generative_model import build_shared_model
    from ghostscale.environment import Environment
    from ghostscale.v7.e45_tom_efficiency import _train_on

    c = load_config()
    gm = build_shared_model(c)
    ng = int(c.cardinalities.num_goals)
    rng = np.random.default_rng(7)
    env = Environment(c, gm, rng, honesty=1.0, signing_rate=0.0)

    held = [ng - 1, ng - 2]
    clf = _train_on(c, env, ng, [g for g in range(ng) if g not in held], 256, rng)
    # An untrained row is the smoothing prior alone, so it is exactly uniform.
    for g in held:
        row = np.exp(clf.log_lik[g])
        assert np.allclose(row, row[0]), "a held-out intention must have seen zero examples"
    for g in range(ng):
        if g not in held:
            row = np.exp(clf.log_lik[g])
            assert not np.allclose(row, row[0]), "a trained intention must have learned something"


def test_n33_both_readers_can_share_one_tape():
    """The fairness condition, which the first version of E45 did not meet."""
    from ghostscale.config import load_config
    from ghostscale.generative_model import build_shared_model
    from ghostscale.environment import Artifact, Environment
    from ghostscale.baselines import ObservationTape, TapedEnvironment
    from ghostscale import constants as K

    c = load_config()
    gm = build_shared_model(c)
    rng = np.random.default_rng(11)
    env = Environment(c, gm, rng, honesty=1.0, signing_rate=0.0)
    art = Artifact(provenance=K.CREATOR, goal=1, declared_signal=K.UNSIGNED)
    tape = ObservationTape(env, art, rng, 6)

    te = TapedEnvironment(tape)
    te.sample_feature(art, K.SKIM, rng)              # the free glance
    served = [te.observation(art, K.DEEP, rng)[0] for _ in range(6)]
    assert served == [tape.feature(t, K.DEEP) for t in range(6)]


# =========================================================================== #
# The lock.
# =========================================================================== #
def test_prereg_v7_hash_is_stable_and_verifies(cfg, tmp_path):
    p = tmp_path / "prereg.json"
    a = write_preregistration_v7(cfg, p)
    assert a["content_hash"] == build_preregistration_v7(cfg)["content_hash"]
    assert assert_prereg_locked_v7(p)["content_hash"] == a["content_hash"]


def test_prereg_v7_detects_tampering(cfg, tmp_path):
    import json
    p = tmp_path / "prereg.json"
    write_preregistration_v7(cfg, p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["C-1"]["bar"] = 0.01
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_prereg_locked_v7(p)
