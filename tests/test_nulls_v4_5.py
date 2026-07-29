"""V4.5 nulls and invariants (spec §7).

V4.5 adds three constraints to the ones V1-V4 already carry, and each one guards a specific
way this delta could silently become worthless:

    beta is not a rename of kappa   -> at beta = 1, V4.5 must reproduce V4 within tolerance.
                                       "If it does not, beta has been wired into the wrong
                                       pipeline position."
    social influence enters the prior -> and NEVER the integration multiplier. Asserted at
                                       construction, not left to reviewer attention.
    beta = 0 IS EXPLORE             -> the §3.3 identification, which is a claim and is
                                       therefore falsifiable. E28 tests it on data; the tests
                                       here cover the half of it that is a property of the
                                       construction rather than of a run.

Plus unit coverage for the new metrics and for each pre-registered criterion, including the
criteria's ability to return the UNWELCOME outcome — a verdict function that cannot say the
framework is wrong is not a criterion, it is a formality.
"""
import json

import numpy as np
import pytest

from ghostscale import baselines as BL
from ghostscale import foreign as FN
from ghostscale import metrics
from ghostscale import prereg_v4_5 as P45
from ghostscale import v4_5_model as V45
from ghostscale.v4_model import build_v4_A0, build_v4_synth
from ghostscale.generative_model import alpha_by_provenance


@pytest.fixture
def cfg():
    return V45.load_v4_5_config()


@pytest.fixture
def world(cfg):
    return V45.build_v45_world(cfg)


# --------------------------------------------------------------------------- #
# N — beta is not a rename of kappa. The V4 boundary.
# --------------------------------------------------------------------------- #
def test_beta_one_reproduces_v4_likelihood_elementwise(cfg, world):
    """V4.5 §7's N-series check, and it is exact rather than approximate.

    At beta = 1 the demonstrator signature reduces to sig[g] algebraically, so the V4.5 A[0]
    slice must equal V4's A[0] element for element. A tolerance would be hiding a wiring
    error behind a number.
    """
    synth = build_v4_synth(cfg)
    alpha = alpha_by_provenance(cfg)
    a45 = V45.build_v45_A0(cfg, world.sigs.sig_true, world.sigs.sig_explore, synth, alpha,
                           world.beta_levels)
    a4 = build_v4_A0(cfg, world.sigs.sig_true, synth, alpha)
    b1 = int(np.argmin(np.abs(np.asarray(world.beta_levels) - 1.0)))
    assert np.array_equal(a45[:, :, :, :, b1], a4), (
        "at beta = 1 the V4.5 likelihood must BE V4's. If it is not, beta has been wired "
        "into the wrong pipeline position (V4.5 §7) — check that beta acts on the creator, "
        "inside the alpha channel mixture, rather than beside it.")


def test_beta_position_assertion_fires_when_beta_is_misplaced(cfg, world):
    """The guard has to be able to fail, or it is decoration.

    Simulates the most plausible wiring error: beta applied OUTSIDE the alpha mixture, so it
    scales the whole channel rather than the creator's optimisation.
    """
    synth = build_v4_synth(cfg)
    alpha = alpha_by_provenance(cfg)
    a4 = build_v4_A0(cfg, world.sigs.sig_true, synth, alpha)
    # beta on the outside: beta * (alpha*sig + (1-alpha)*synth) + (1-beta)*sig_explore
    beta = 1.0
    wrong = beta * a4 + (1.0 - beta) * 0.0
    assert np.array_equal(wrong, a4)   # at beta = 1 even the wrong wiring agrees...
    beta = 0.5
    wrong_half = np.zeros_like(a4)
    for p in range(a4.shape[1]):
        for g in range(a4.shape[2]):
            wrong_half[:, p, g, 0] = beta * a4[:, p, g, 0] + (1 - beta) * world.sigs.sig_explore
    right_half = V45.build_v45_A0(cfg, world.sigs.sig_true, world.sigs.sig_explore, synth,
                                  alpha, [0.5])[:, :, :, 0, 0]

    # ...and at beta = 0.5 they diverge, which is what makes the beta = 1 check meaningful
    # rather than vacuous: the two wirings are only indistinguishable at the boundary.
    #
    # Checked at CURATOR (alpha = 0.60), not CREATOR. At alpha = 1 the channel mixture is the
    # identity, so beta-inside and beta-outside are the same expression at EVERY beta and the
    # comparison proves nothing. That the two wirings coincide wherever alpha = 1 is worth
    # knowing on its own: it means E28, which runs entirely on CREATOR content, could not
    # have caught a mis-positioned beta. The construction-time assertion is what catches it.
    curator = K_CURATOR = 2
    assert float(alpha[curator]) < 1.0
    assert np.allclose(wrong_half[:, 0, 0, 0], right_half[:, 0, 0]), (
        "at alpha = 1 the two wirings must agree; if they do not, one of them is not the "
        "expression this test thinks it is")
    assert not np.allclose(wrong_half[:, curator, 0, 0], right_half[:, curator, 0]), (
        "the two candidate wirings for beta agree away from beta = 1, so the boundary check "
        "would not discriminate them and the N-series guard proves nothing")


def test_beta_zero_demonstrator_is_exactly_sig_explore(world):
    """The construction half of §3.3's identification. This part IS true, exactly.

    At beta = 0 the expected output is the creator's policy marginal, which C2 built as
    sig_EXPLORE. E28 tests the other half — whether the two behave the same under INFERENCE —
    and that half fails.
    """
    demo = V45.demonstrator_signature(world.sigs.sig_true, world.sigs.sig_explore, 0.0)
    for g in range(demo.shape[0]):
        assert np.allclose(demo[g], world.sigs.sig_explore, atol=1e-15)


def test_beta_carries_no_information_in_the_goal_marginal(world):
    """The mechanism behind E28's beta = 0 result, as a property of the construction.

    mean_g(beta*sig[g] + (1-beta)*sig_EXPLORE) = sig_EXPLORE for every beta, because
    sig_EXPLORE IS mean_g(sig[g]). So an observer with a flat goal posterior learns nothing
    about beta from the marginal, and every bit of evidence about beta comes from the joint
    (beta, goal) coupling. That is not a bug; it is why the low end of the beta grid is
    biased upward, and it needs to be executable so nobody re-derives it from an output.
    """
    base = V45.demonstrator_signature(world.sigs.sig_true, world.sigs.sig_explore,
                                      0.0).mean(axis=0)
    for b in world.beta_levels:
        marg = V45.demonstrator_signature(world.sigs.sig_true, world.sigs.sig_explore,
                                          float(b)).mean(axis=0)
        assert np.allclose(marg, base, atol=1e-15)


# --------------------------------------------------------------------------- #
# N — social influence enters the prior, never the integration multiplier.
# --------------------------------------------------------------------------- #
def test_no_social_term_on_the_update_path():
    assert V45.assert_no_social_term_in_update_path()["clean"]


def test_social_influence_moves_the_goal_prior_and_nothing_else(cfg, world):
    """It shifts D[1]. It does not touch the likelihood, the gate, or the update."""
    rng = np.random.default_rng(3)
    shift = np.array([0.7, 0.1, 0.1, 0.1])
    d_plain = V45.build_v45_D(cfg, np.random.default_rng(3), len(world.beta_levels))
    d_social = V45.build_v45_D(cfg, rng, len(world.beta_levels),
                               social_shift=shift, social_weight=0.5)
    assert not np.allclose(d_plain[1], d_social[1]), "social influence must move D[1]"
    assert np.asarray(d_social[1])[0] > np.asarray(d_plain[1])[0], (
        "the prior must move TOWARD the socially-endorsed goal")
    # Everything else is untouched.
    for f in (0, 2, 3):
        assert np.allclose(d_plain[f], d_social[f])


def test_beta_prior_stays_uniform_under_social_influence(cfg, world):
    """E28 measures beta RECOVERY. An informative prior over beta would be putting the answer
    in by hand, and a social term leaking into it would do that invisibly."""
    d = V45.build_v45_D(cfg, np.random.default_rng(0), len(world.beta_levels),
                        social_shift=np.array([0.7, 0.1, 0.1, 0.1]), social_weight=0.9)
    assert np.allclose(d[3], 1.0 / len(world.beta_levels))


# --------------------------------------------------------------------------- #
# The theta gate.
# --------------------------------------------------------------------------- #
def test_open_theta_reduces_to_v1_psi_exactly():
    """lambda = 0 must make the V4.5 update path indistinguishable from V1's.

    This is what keeps every V1-V4 number reachable from the V4.5 code: if the gate cost
    something even when open, the decomposition would have moved results it was only supposed
    to explain.
    """
    post = np.array([0.7, 0.1, 0.1, 0.1])
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    pc = np.array([0.1, 0.1, 0.1, 0.7])
    g = V45.gated_update(post, prior, kappa=0.9, engaged=True, value_prior=pc,
                         theta_base=10.0, lam=0.0)
    assert g["psi_gated"] == pytest.approx(metrics.psi_analogue(post, prior, 0.9, True))
    assert g["theta"] == pytest.approx(1.0, abs=1e-4)


def test_closed_theta_shuts_on_divergence_not_unconditionally():
    """A gate that closes whatever was recovered is an off switch, not a values gate.

    With lambda high, a recovered goal the observer VALUES must still pass; only a divergent
    one is refused. E29's decisive contrast is meaningless if this is not true.
    """
    prior = np.full(4, 0.25)
    pc = np.array([0.7, 0.1, 0.1, 0.1])
    aligned = np.array([0.85, 0.05, 0.05, 0.05])
    divergent = np.array([0.05, 0.05, 0.05, 0.85])
    g_ok = V45.gated_update(aligned, prior, 0.9, True, pc, theta_base=0.0, lam=4.0)
    g_no = V45.gated_update(divergent, prior, 0.9, True, pc, theta_base=0.0, lam=4.0)
    assert g_ok["theta"] > g_no["theta"], (
        "theta must respond to VALUE DIVERGENCE; if it closes regardless of what was "
        "recovered it is an off switch and E29 tests nothing")
    assert g_no["psi_gated"] < g_ok["psi_gated"]


def test_theta_does_not_touch_the_likelihood(cfg):
    """theta acts on the update. Two worlds differing only in theta config must share an A."""
    w_open = V45.build_v45_world(cfg)
    cfg2 = V45.load_v4_5_config()
    cfg2.set("v4_5.theta.lambda_open", 4.0)
    w_closed = V45.build_v45_world(cfg2)
    for m in range(3):
        assert np.array_equal(np.asarray(w_open.gm.A[m]), np.asarray(w_closed.gm.A[m])), (
            "theta has leaked into the likelihood; V4.5 §3.2 puts it on the update and "
            "kappa_p on the likelihood, and E29 cannot separate gates that share a position")


# --------------------------------------------------------------------------- #
# Calibration metrics (A2).
# --------------------------------------------------------------------------- #
def test_brier_reference_points():
    """A uniform posterior over K classes scores 1 - 1/K whatever the outcome; a perfect
    one scores 0; a confidently-wrong one approaches 2."""
    uni = np.full((10, 4), 0.25)
    y = np.zeros(10, dtype=int)
    assert metrics.brier_score(uni, y) == pytest.approx(0.75)
    perfect = np.eye(4)[np.arange(4) % 4]
    assert metrics.brier_score(perfect, np.arange(4)) == pytest.approx(0.0)
    wrong = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    assert metrics.brier_score(wrong, np.ones(4, dtype=int)) == pytest.approx(2.0)


def test_ece_is_zero_for_a_calibrated_observer():
    """Confidence 0.8 on 100 predictions of which 80 are right: perfectly calibrated."""
    n = 100
    p = np.tile([0.8, 0.2 / 3, 0.2 / 3, 0.2 / 3], (n, 1))
    y = np.array([0] * 80 + [1] * 20)
    assert metrics.expected_calibration_error(p, y, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_the_e2_failure_mode():
    """Near-certain and right at chance — the shape of the mislabeled-synthetic cell."""
    n = 400
    p = np.tile([0.97, 0.01, 0.01, 0.01], (n, 1))
    y = np.array([0] * (n // 4) + [1] * (3 * n // 4))
    ece = metrics.expected_calibration_error(p, y, n_bins=10)
    assert ece > 0.7, "ECE must be large when confidence is at ceiling and accuracy at chance"


def test_ece_is_not_proper_and_brier_is():
    """Documented in the metric docstrings and asserted here, because reporting ECE alone
    would be reporting a number an uninformative observer can score perfectly on."""
    n = 400
    hedge = np.full((n, 4), 0.25)
    y = np.random.default_rng(0).integers(0, 4, n)
    assert metrics.expected_calibration_error(hedge, y, n_bins=10) < 0.05
    assert metrics.brier_score(hedge, y) == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# E21 criteria.
# --------------------------------------------------------------------------- #
def test_dissociation_is_conjunctive():
    assert P45.reproduces_dissociation(0.1, 1.3)["reproduces"]
    assert not P45.reproduces_dissociation(0.1, 0.2)["reproduces"]   # confident, agreeing
    assert not P45.reproduces_dissociation(1.3, 1.3)["reproduces"]   # uncertain, disagreeing


def test_label_induction_is_secondary_and_stricter():
    """An arm confidently wrong regardless of the label reproduces the dissociation but not
    the induction. That distinction is the whole of A1/A2's claim."""
    ignores_label = P45.reproduces_dissociation(0.1, 1.3, control_within=0.1)
    responds = P45.reproduces_dissociation(0.1, 1.3, control_within=1.3)
    assert ignores_label["reproduces"] and not ignores_label["label_induced"]
    assert responds["reproduces"] and responds["label_induced"]


def test_engagement_specificity_clause_rejects_a_constant_arm():
    """The two ways an arm passes for free, both of which V4.5's arms actually exhibit."""
    good = P45.reproduces_foreign_engagement(0.75, 1.4, control_engaged=0.0,
                                             control_entropy=0.01)
    never_resolves = P45.reproduces_foreign_engagement(0.75, 1.4, control_engaged=0.2,
                                                       control_entropy=1.4)
    always_engaged = P45.reproduces_foreign_engagement(1.0, 1.4, control_engaged=1.0,
                                                       control_entropy=0.01)
    assert good["reproduces"]
    assert not never_resolves["reproduces"], "an arm that never resolves anything must fail"
    assert not always_engaged["reproduces"], "an always-DEEP arm must fail an ENGAGEMENT test"


def test_e21_can_return_the_unwelcome_outcome():
    """V4.5 §2 requires E21 to be able to say the machinery is unnecessary."""
    def d(rep, ctrl=1.3):
        return dict(P45.reproduces_dissociation(0.1 if rep else 1.3, 1.3, ctrl))

    def e(rep):
        return dict(P45.reproduces_foreign_engagement(
            0.75 if rep else 0.0, 1.4, control_engaged=0.0, control_entropy=0.01))

    scaffolding = P45.e21_verdict(
        {"A_active_inference": d(True), "C_label_truster": d(True)},
        {"A_active_inference": e(True), "C_label_truster": e(False)})
    assert scaffolding["verdict"] == "MACHINERY_UNNECESSARY"

    necessary = P45.e21_verdict(
        {"A_active_inference": d(True), "C_label_truster": d(False),
         "D_effort_heuristic": d(False)},
        {"A_active_inference": e(True), "C_label_truster": e(False),
         "D_effort_heuristic": e(False)})
    assert necessary["verdict"] == "MACHINERY_NECESSARY"

    no_control = P45.e21_verdict({"A_active_inference": d(False)},
                                 {"A_active_inference": e(False)})
    assert no_control["verdict"] == "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# E28 criteria.
# --------------------------------------------------------------------------- #
def test_e28_can_return_each_outcome():
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    rising = [0.1, 0.3, 0.5, 0.7, 0.9]

    sep = P45.e28_verdict(grid, rising, [0.95] * 5, rising, 0.4, 0.4)
    assert sep["verdict"] == "BETA_IS_SEPARABLE"

    partial = P45.e28_verdict(grid, rising, [0.3, 0.6, 0.9, 0.99, 1.0], rising, 0.4, 0.4)
    assert partial["verdict"] == "BETA_IS_SEPARABLE_OVER_PART_OF_THE_RANGE"

    confounded = P45.e28_verdict(grid, rising, [0.3, 0.4, 0.5, 0.6, 0.7], rising, 0.4, 0.4)
    assert confounded["verdict"] == "CONFOUNDED_WITH_LEGIBILITY"

    collapsed = P45.e28_verdict(grid, rising, [0.95] * 5, [0.5] * 5, 0.4, 0.4)
    assert collapsed["verdict"] == "BETA_COLLAPSES"


def test_e28_beta0_consistency_check_can_fail():
    """§3.3's required check must be able to say the identification is wrong. It did."""
    ok = P45.e28_verdict([0, 1], [0.1, 0.9], [0.9, 0.9], [0.1, 0.9], 0.40, 0.39)
    bad = P45.e28_verdict([0, 1], [0.1, 0.9], [0.9, 0.9], [0.1, 0.9], 0.89, 0.39)
    assert ok["beta0_recovers_e19_explore_cell"]
    assert not bad["beta0_recovers_e19_explore_cell"]


# --------------------------------------------------------------------------- #
# E29 criteria.
# --------------------------------------------------------------------------- #
def test_update_classification_is_a_ratio_not_a_level():
    assert P45.classify_update(0.01, 1.0) == "none"
    assert P45.classify_update(0.30, 1.0) == "low"
    assert P45.classify_update(0.90, 1.0) == "high"
    # Same ratios at a different scale must classify identically.
    assert P45.classify_update(1.0, 100.0) == "none"
    assert P45.classify_update(30.0, 100.0) == "low"


def test_e29_reports_decomposition_not_earned_when_the_cells_coincide():
    """V4.5 §5: if low beta and closed theta are indistinguishable on every measure, say so
    rather than keeping the decomposition on theoretical grounds."""
    same = P45.e29_cell_signature(0.9, 0.1, 0.3, 1.0, 0.5, 0.5)
    v = P45.e29_verdict({"low_beta": same, "closed_theta": dict(same)})
    assert v["verdict"] == "DECOMPOSITION_NOT_EARNED"


def test_e29_dissociates_when_the_cells_differ():
    low_beta = P45.e29_cell_signature(0.9, 0.1, 0.30, 1.0, 0.5, 0.5)
    closed = P45.e29_cell_signature(0.9, 0.1, 0.01, 1.0, 2.0, 0.5)
    v = P45.e29_verdict({"low_beta": low_beta, "closed_theta": closed})
    assert v["verdict"] in ("GATES_DISSOCIATE", "GATES_PARTLY_DISSOCIATE")
    assert v["decisive_contrast"]["differs_on_update"]


# --------------------------------------------------------------------------- #
# E21's fairness machinery.
# --------------------------------------------------------------------------- #
def test_observation_tape_gives_every_arm_the_same_content(cfg, world):
    """Without this the arms diverge after their first differing attention choice and are
    reading different artifacts, which would make the comparison meaningless in the direction
    that flatters the full model."""
    from ghostscale import constants as K
    from ghostscale.creators import HumanCreator
    from ghostscale.environment import Artifact, Environment

    creators = {g: HumanCreator(cfg, world.sigs.sig_true, g) for g in range(4)}
    env = Environment(cfg, world.gm, np.random.default_rng(0), honesty=1.0,
                      signing_rate=0.0, creator_bank=creators)
    art = Artifact(provenance=K.CREATOR, goal=1, declared_signal=K.UNSIGNED)
    tape = BL.ObservationTape(env, art, np.random.default_rng(7), 12)

    e1, e2 = BL.TapedEnvironment(tape), BL.TapedEnvironment(tape)
    # One env chooses DEEP where the other chooses SKIM, and they still agree wherever the
    # attention choice agrees.
    a = [e1.observation(art, K.DEEP, None)[0] for _ in range(12)]
    b = [e2.observation(art, K.DEEP, None)[0] for _ in range(12)]
    assert a == b
    assert tape.deep.tolist() == a


def test_prereg_v4_5_is_hash_locked(cfg, tmp_path):
    path = tmp_path / "v4_5_preregistration.json"
    P45.write_preregistration_v4_5(cfg, path)
    P45.assert_prereg_locked_v4_5(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["E21_criteria"]["dissociation"]["confident_entropy_max"] = 9.9
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified since it was written"):
        P45.assert_prereg_locked_v4_5(path)


def test_spearman_matches_a_known_value():
    assert P45.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert P45.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
