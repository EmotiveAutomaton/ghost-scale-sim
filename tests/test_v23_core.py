"""Known answers for the new instruments; never invoke a campaign runner."""
from copy import deepcopy
from fractions import Fraction as F
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import adaptive, evidence, registry, repairs


def test_flooring_counterexample_and_valid_support():
    p, m, floor = F(1, 10), F(1, 50), F(-1, 10)
    assert p*(-1)+(1-p)*F(2, 15) == m
    invalid_fraction = 1/(2*(m-floor))
    assert 1+invalid_fraction*(p*floor+(1-p)*F(2, 15)-m) == F(11, 8)
    valid_fraction = 1/(2*(m+1))
    assert 1+valid_fraction*(p*(-1)+(1-p)*F(2, 15)-m) == 1
    assert 1+valid_fraction*(-1-m) == F(1, 2)


def test_confirmation_preserves_negative_gain_and_rejects_bad_bound():
    out = evidence.betting_evalue([-1]*16, -1, independent_balanced_blocks=True)
    assert out['e_value'] == pytest.approx(.5)
    assert out['block_means'] == [-1]
    with pytest.raises(ValueError, match='support violation'):
        evidence.betting_evalue([-1]*16, -.1, independent_balanced_blocks=True)
    with pytest.raises(ValueError, match='balanced'):
        evidence.betting_evalue([0]*17, -1, independent_balanced_blocks=True)
    assert evidence.betting_evalue([float('inf')]*16, -1,
        independent_balanced_blocks=True)['state'] == 'CONFIRMATION_UNAVAILABLE'


def test_elond_previous_discoveries_and_unavailable_not_confirmed():
    out = evidence.elond([math.log(20), None, math.log(60)])
    assert [r['confirmed'] for r in out] == [True, False, True]
    assert out[2]['earlier_discoveries'] == 1
    assert math.exp(out[2]['log_threshold']) == pytest.approx(60)


def test_finite_grids_and_primary_orientation():
    assert sum(len(registry.variants(f)) for f in registry.EXPECTED) == 1234
    for family, expected in registry.EXPECTED.items():
        assert len(registry.variants(family)) == expected
        assert sum(v['late_effect'] for v in registry.variants(family)) <= math.ceil(expected/3)
    assert registry.PRIMARY['R2'][3] == 'recognition_cost'


def test_direct_features_do_not_reveal_marginalized_generator_knobs():
    v = deepcopy(registry.variants('N4')[-1]); a = registry.variant_features(v)
    v['settings'].update(mechanism='transition')
    assert registry.variant_features(v) == a


def test_posterior_quadrature_and_burn_in():
    post = adaptive.checked_posterior([0]*27, [.02]*27)
    assert np.max(np.abs(post['mean'])) < 1e-12
    assert post['q'] < .5
    estimates = [[dict(mean=-.1, se=.01)] for _ in range(27)]
    assert adaptive.decide(1, estimates, list(range(27)), [])['state'] == 'BURN_IN'
    assert adaptive.decide(2, estimates, list(range(27)), [])['state'] == 'STOPPED_BELOW_MARGIN'


def test_r1_vanishing_signature_identity_all_prevalences():
    for v in registry.variants('R1'):
        if v['settings']['strength'] != .5:
            continue
        public, truth = repairs.sample(v, 'engineering-control', 4, 0)
        predictions, _ = repairs.evaluate(v, public, truth)
        np.testing.assert_allclose(predictions['masked'], predictions['law-marginal'], atol=1e-12)


def test_r1_informative_signature_changes_goal_belief():
    v = deepcopy(registry.variants('R1')[-1])
    p = dict(own=1, claim=0, selection=[1]*32, operations=[1]*8)
    visible = repairs.r1_posterior(v, p)
    masked = repairs.r1_posterior(v, p, masked=True)
    assert visible[1, 1] > .999
    assert masked[1, 1] < .9


def test_r2_zero_attention_freezes_all_purpose_channels():
    v = deepcopy(registry.variants('R2')[-1]); v['settings']['benign_share'] = .5
    p = dict(observations=[[1, 1, 1]]*32, flags=[1]*32)
    _, path = repairs.r2_trajectory(v, p, 'withdraw-all-0-0.5-none')
    assert all(x['flag_precision'] == 0 for x in path)
    assert all(x['purpose'] == pytest.approx(.5) for x in path)


def test_r3_choice_independent_cost_labels_are_algebraic_null():
    v = next(v for v in registry.variants('R3') if v['settings'] == dict(
        cost_role='irrelevant', records=2, dimensions=2, shift=False))
    p, truth = repairs.sample(v, 'engineering-control', 0, 0)
    changed = deepcopy(p)
    for row in changed['history']+[changed['current']]:
        row['costs'] = [1000., 2000., 3000.]
    a = repairs.r3_infer(v, p, role=0.)[0]
    b = repairs.r3_infer(v, changed, role=0.)[0]
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_r4_root_cue_closed_form_and_neutral_cue_identity():
    v = deepcopy(registry.variants('R4')[0]); v['settings'].update(cue_reliability=.9, coverage=0)
    parent, _ = repairs.tree(2)
    p = dict(evidence=[-1]*len(parent), cue=1, cue_node=0)
    joint = repairs.r4_infer(v, p)
    assert joint[0, 1] == pytest.approx(.9)
    np.testing.assert_allclose(joint, repairs.r4_infer(v, p, mode='top-down'), atol=1e-12)
    v['settings']['cue_reliability'] = .5
    np.testing.assert_allclose(repairs.r4_infer(v, p), repairs.r4_infer(v, p, cue=False), atol=1e-12)
