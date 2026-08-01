"""V6 nulls N22-N30, the V5 reduction, and the gate-gain limit.

The reduction tests matter more here than in any previous version. V6 adds more at once than
anything before it, and the whole defence against "too many changes to attribute anything" is
that every addition is independently switchable and OFF by default. If that is not actually true,
the defence is a story rather than a property, so it is tested rather than asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale import foreign as FN
from ghostscale import v6_model as V6
from ghostscale.prereg_v6 import (assert_prereg_locked_v6, build_preregistration_v6, spearman)
from ghostscale.v4_model import build_v4_synth
from ghostscale.v5_model import load_v5_config
from ghostscale.v6 import harness as H


@pytest.fixture(scope="module")
def cfg():
    c = load_v5_config()
    c.set("inference.exact", True)
    return c


@pytest.fixture(scope="module")
def v4cfg(cfg):
    c = cfg.copy()
    c.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    c.set("cardinalities.num_features", FN.NUM_FEATURES_V4)
    return c


# =========================================================================== #
# N22 — depletion must not accumulate on a fully resolvable corpus.
# =========================================================================== #
def test_n22_reserve_does_not_drift_when_everything_resolves():
    """The null the generational experiment never passed, written first on purpose.

    A depletion mechanism that drifts on content it should not touch would produce E35's
    headline for free, and no amount of downstream care would recover the result.
    """
    r = V6.MetabolicReserve()
    for _ in range(200):
        r.update(engaged_fraction=1.0, resolved=1.0)
    assert r.e == pytest.approx(1.0, abs=1e-9), (
        "a reader that looks hard and always succeeds must not be depleted by it")


def test_n22_reserve_does_not_drift_when_nothing_is_looked_at():
    r = V6.MetabolicReserve()
    for _ in range(200):
        r.update(engaged_fraction=0.0, resolved=0.0)
    assert r.e == pytest.approx(1.0, abs=1e-9), (
        "a reader that never looks cannot be worn down by what it did not read")


def test_depletion_requires_both_looking_and_failing():
    """The product form is the mechanism, not a convenience: only disappointment costs."""
    spent = V6.MetabolicReserve()
    for _ in range(30):
        spent.update(engaged_fraction=1.0, resolved=0.0)
    assert spent.e < 0.5
    assert spent.theta_base() > 0.0, "a depleted reader must face a higher threshold"


# =========================================================================== #
# N23 — with every switch off, V6 is V5.
# =========================================================================== #
def test_n23_switches_all_off_by_default(cfg):
    sw = V6.V6Switches.from_config(cfg)
    assert sw.all_off(), (
        "every V6 addition must be off by default, or a V6 run is not attributable to any one "
        "of them")


def test_n23_gate_with_coupling_off_ignores_kappa():
    """At c = 0 the threshold is exactly V1-V5's: kappa does not enter it at all."""
    a = V6.disgust_threshold(1.3, kappa=0.1, reserve=None, lam=1.0, theta_0=0.35, coupling=0.0)
    b = V6.disgust_threshold(1.3, kappa=0.99, reserve=None, lam=1.0, theta_0=0.35, coupling=0.0)
    assert a == pytest.approx(b, abs=1e-12)


def test_coupling_on_suppresses_the_threshold_as_trust_rises():
    """The preprint's stated behaviour: kappa -> 1 drives theta_EC -> 0."""
    vals = [V6.disgust_threshold(1.3, kappa=k, reserve=None, lam=1.0, theta_0=0.35, coupling=1.0)
            for k in (0.0, 0.3, 0.6, 0.9, 0.99)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    assert vals[-1] < 0.05


# =========================================================================== #
# N24 — as k grows, the graded gate reproduces the binary decision.
# =========================================================================== #
def test_n24_gate_approaches_the_binary_decision():
    for omega, theta in ((0.8, 0.3), (0.2, 0.7), (0.51, 0.49)):
        hard = 1.0 if omega > theta else 0.0
        assert V6.gate(omega, theta, k_gain=5000.0) == pytest.approx(hard, abs=1e-6)


def test_gate_is_graded_at_finite_gain():
    """The thing V1-V5 could not express: a gate that is nearly, but not quite, closed."""
    g = V6.gate(0.45, 0.50, k_gain=8.0)
    assert 0.0 < g < 0.5


# =========================================================================== #
# N25 — no preference over provenance, still.
# =========================================================================== #
def test_n25_alt_worlds_carry_zero_preferences(cfg, v4cfg):
    from ghostscale.generative_model import assert_preferences_zero
    sigs = FN.build_v4_signatures(v4cfg, omega=0.0, include_explore=False, foreign_seed=7)
    world, _, _, _, _, _ = H.build_alt_world(cfg, np.asarray(sigs.sig_foreign, dtype=float))
    assert_preferences_zero(world.gm.C)


# =========================================================================== #
# N26 — the values map must be non-injective, or it is the goal renamed.
# =========================================================================== #
def test_n26_values_map_is_non_injective():
    M = V6.build_values_map(4, n_values=2)
    assert M.shape == (2, 4)
    cols = {tuple(M[:, g]) for g in range(4)}
    assert len(cols) < 4, "two goals must be able to imply the same values"


def test_n26_rejects_a_bijective_map():
    with pytest.raises(AssertionError):
        V6.build_values_map(4, n_values=4)


def test_values_layer_lets_different_goals_share_a_gate_state():
    """What the layer buys: 'I disagree with what you did but we want the same things'."""
    M = V6.build_values_map(4, n_values=2)
    g0 = np.array([1.0, 0.0, 0.0, 0.0])
    g2 = np.array([0.0, 0.0, 1.0, 0.0])
    assert np.allclose(V6.implied_values(g0, M), V6.implied_values(g2, M))


# =========================================================================== #
# N27 — NO_MAKER must not absorb human work. The failure V4 caught with EXPLORE.
# =========================================================================== #
def test_n27_no_maker_does_not_absorb_human_goals(v4cfg):
    sigs = FN.build_v4_signatures(v4cfg, omega=0.0, include_explore=False, foreign_seed=7)
    nm = V6.build_no_maker_signature(build_v4_synth(v4cfg))
    out = V6.assert_no_maker_does_not_absorb(nm, sigs.sig_true)
    assert out["min_js_to_human_goal"] >= out["floor"]


def test_n27_rejects_a_hypothesis_that_sits_on_a_human_goal(v4cfg):
    """The check must fire on a hypothesis close to the goals it would absorb.

    NOTE ON WHAT IS *NOT* THE DANGER HERE, because the first version of this test got it wrong
    and the reason is informative. A hypothesis uniform over the WHOLE feature space does not
    absorb at V4 cardinality: the human goal signatures are concentrated on two features each,
    so a global uniform sits a long way from all of them and the check correctly passes it.
    That is precisely what doubling the feature space bought -- V4's own note records that at
    V1-V3 cardinality the averaged fallback WAS globally flat and would have absorbed
    everything. So the absorbing shape at this cardinality is one that sits ON the human block,
    not one that is spread thin across everything.
    """
    sigs = FN.build_v4_signatures(v4cfg, omega=0.0, include_explore=False, foreign_seed=7)
    impostor = np.asarray(sigs.sig_true, dtype=float)[0]
    with pytest.raises(AssertionError):
        V6.assert_no_maker_does_not_absorb(impostor, sigs.sig_true, floor=0.20)


def test_a_globally_uniform_hypothesis_is_safe_at_v4_cardinality(v4cfg):
    """The other half of the same point, measured rather than asserted."""
    sigs = FN.build_v4_signatures(v4cfg, omega=0.0, include_explore=False, foreign_seed=7)
    flat = np.full(sigs.sig_true.shape[1], 1.0 / sigs.sig_true.shape[1])
    out = V6.assert_no_maker_does_not_absorb(flat, sigs.sig_true, floor=0.20)
    assert out["min_js_to_human_goal"] > 0.20


# =========================================================================== #
# N28 — at the shallowest depth there is no process, so recovery carries no information.
# =========================================================================== #
def test_n28_flat_posterior_scores_zero_information():
    """The reason the null is decided on information rather than on accuracy.

    An unmoved posterior must score exactly zero however the truth was distributed. Accuracy
    does not have that property: the argmax of a uniform is a tie broken to index zero, and the
    true mode is autocorrelated, so accuracy on an uninformative posterior can land ANYWHERE --
    including below nominal chance, which is what it did.
    """
    n_sub = 4
    flat = [np.full(n_sub, 1.0 / n_sub) for _ in range(24)]
    for truth in (0, 1, 2, 3):
        out = V6.process_recovery(flat, [truth] * 24, n_sub)
        assert out["process_error_reduction"] == pytest.approx(0.0, abs=1e-9)

    # ...and here is accuracy failing to have it, which is the whole reason for the change.
    accs = {V6.process_recovery(flat, [t] * 24, n_sub)["process_accuracy"] for t in range(4)}
    assert accs == {0.0, 1.0}, (
        "accuracy on a flat posterior is decided entirely by which mode the truth sat in")


def test_process_recovery_rewards_a_correct_confident_posterior():
    n_sub = 4
    sharp = []
    truth = []
    for t in range(24):
        s = t % n_sub
        p = np.full(n_sub, 0.01)
        p[s] = 0.97
        sharp.append(p / p.sum())
        truth.append(s)
    out = V6.process_recovery(sharp, truth, n_sub)
    assert out["process_accuracy"] == pytest.approx(1.0)
    assert out["process_error_reduction"] > 1.0


# =========================================================================== #
# N29 / N30 — the cue channels and the wall.
# =========================================================================== #
def test_n29_cue_channels_do_not_touch_the_goal():
    """The cues enter the ENGAGEMENT decision and nothing else; they cannot name a goal."""
    c = V6.CueChannels(0.5, 0.5, "additive")
    assert c.combine(1.0, 1.0, 0.0) == pytest.approx(1.5)
    assert c.combine(1.0, 0.0, 1.0) == pytest.approx(1.5)


def test_additive_and_multiplicative_disagree_where_information_gain_is_zero():
    """The corner that IS the test: a cue driving engagement on content offering nothing."""
    add = V6.CueChannels(0.5, 0.5, "additive")
    mul = V6.CueChannels(0.5, 0.5, "multiplicative")
    assert add.combine(0.0, 0.0, 1.0) > 0.0
    assert mul.combine(0.0, 0.0, 1.0) == pytest.approx(0.0)


def test_n30_noninvertible_family_stays_on_the_human_block(v4cfg):
    sigs = FN.build_v4_signatures(v4cfg, omega=0.0, include_explore=False, foreign_seed=7)
    fam = V6.build_noninvertible_family(sigs.sig_true, n_states=4, collapse_to=2)
    assert fam["max_foreign_mass"] < 0.10
    assert fam["invertible"] is False


def test_noninvertible_family_really_is_many_to_one(v4cfg):
    sigs = FN.build_v4_signatures(v4cfg, omega=0.0, include_explore=False, foreign_seed=7)
    fam = V6.build_noninvertible_family(sigs.sig_true, n_states=4, collapse_to=2)
    surfaces = {fam["state_to_surface"][s] for s in range(fam["n_states"])}
    assert len(surfaces) < fam["n_states"]


def test_a_bijective_generator_is_rejected(v4cfg):
    sigs = FN.build_v4_signatures(v4cfg, omega=0.0, include_explore=False, foreign_seed=7)
    with pytest.raises(AssertionError):
        V6.build_noninvertible_family(sigs.sig_true, n_states=4, collapse_to=4)


# =========================================================================== #
# Graded self-report, scale invariance, and the lock.
# =========================================================================== #
def test_self_report_falls_with_depth():
    """The author's correction: the subconscious holds the PRACTISED goals, not all of them."""
    vals = [V6.self_report_accuracy(m) for m in (1, 2, 3)]
    assert vals[0] > vals[1] > vals[2]


def test_subwindow_recovery_tracks_the_whole_artifact():
    n_sub = 4
    posts, truth = [], []
    for t in range(24):
        s = (t // 6) % n_sub
        p = np.full(n_sub, 0.05)
        p[s] = 0.85
        posts.append(p / p.sum())
        truth.append(s)
    out = V6.subwindow_recovery(posts, truth, n_sub, fraction=0.25)
    assert out["n_windows"] > 1
    assert abs(out["whole_accuracy"] - out["window_accuracy_mean"]) < 0.35


def test_prereg_hash_is_stable_and_verifies(cfg, tmp_path):
    from ghostscale.prereg_v6 import write_preregistration_v6
    p = tmp_path / "prereg.json"
    a = write_preregistration_v6(cfg, p)
    b = build_preregistration_v6(cfg)
    assert a["content_hash"] == b["content_hash"]
    assert assert_prereg_locked_v6(p)["content_hash"] == a["content_hash"]


def test_prereg_detects_tampering(cfg, tmp_path):
    import json
    from ghostscale.prereg_v6 import write_preregistration_v6
    p = tmp_path / "prereg.json"
    write_preregistration_v6(cfg, p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["H6.1"]["probe_drop"] = 0.0001
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_prereg_locked_v6(p)


def test_spearman_is_tie_aware_and_refuses_a_constant():
    """Pinned because the diagnostics pass found the naive version scoring a CONSTANT at 1.00."""
    assert np.isnan(spearman([1, 2, 3], [5, 5, 5]))
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


# =========================================================================== #
# The V5 fallback path V6 found by crashing it.
# =========================================================================== #
def test_execution_modes_survive_a_flat_signature_family(v4cfg):
    """A latent crash in shipped V5 code, unreachable with the human family.

    ``build_execution_modes``' shrink fallback indexed a (1, F) array with an (S, F) mask, which
    raises rather than broadcasting. The human goal signatures are concentrated enough that the
    projection never goes negative, so the path was never taken until V6 built a world over a
    flatter family. Pinned so it cannot come back.
    """
    from ghostscale.v5_model import build_execution_modes
    nf = int(v4cfg.cardinalities.num_features)
    flat = np.full(nf, 1.0 / nf)
    modes = build_execution_modes(flat, 4, support=np.asarray(FN.HUMAN_FEATURES, dtype=int))
    assert modes.shape == (4, nf)
    assert np.allclose(modes.sum(axis=1), 1.0)
    assert np.allclose(modes.mean(axis=0), flat, atol=1e-10)
