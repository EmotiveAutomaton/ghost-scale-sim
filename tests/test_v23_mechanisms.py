from copy import deepcopy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import (
    registry, makers, corrections, projection, uptake, statements, play, values)


def test_shared_recipe_count_and_additive_alias():
    recipes, weights, _ = makers.recipe_library()
    assert weights.shape == (200, 10)
    options = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0]])
    pair = recipes.index((1, 2)); single1 = recipes.index((1,)); single2 = recipes.index((2,))
    sat = makers.satisfaction(options, recipes, 'additive')
    np.testing.assert_allclose(sat[:, pair], (sat[:, single1]+sat[:, single2])/2)


def test_hmm_zero_drift_matches_static_likelihood():
    recipes, weights, _ = makers.recipe_library()
    menus = np.array([[[0,0,1,1],[1,0,0,1],[1,1,0,0]], [[0,0,0,0],[1,1,1,1],[0,1,0,1]]])
    choices = np.array([1, 2]); prior = np.full(200, 1/200)
    post, evidence = makers.type_filter(menus, choices, 2, recipes, weights, prior, 0)
    from ghostscale.validation.soundingline.v23.common import normalize
    np.testing.assert_allclose(post[0], normalize(makers.record_loglik(menus, choices, 2, recipes, weights)), atol=1e-12)


def test_n1_empty_records_hierarchical_prior_and_identity():
    v = deepcopy(registry.variants('N1')[0]); v['settings'].update(makers=2, records=0)
    p, t = makers.n1_world(v, 'engineering-control', 0)
    prediction, detail = makers.n1_infer(v, p)
    np.testing.assert_allclose(prediction['law-marginal'], prediction['partial-pooling'], atol=1e-12)
    for x in prediction.values():
        np.testing.assert_allclose(x.sum(-1), 1, atol=1e-12)


def test_n3_softmax_can_produce_negative_utility_correction():
    w = np.array([1., 0., 0.]); edits = np.array([[0.,0.,0.],[-1.,0.,0.],[1.,0.,0.]])
    from ghostscale.validation.soundingline.v23.common import normalize
    probability = normalize(4*(edits@w))
    assert probability[1] > 0
    assert edits[1]@w < 0
    assert len(corrections.directions(3)) == 26 and len(corrections.directions(5)) == 242


def test_n3_predictions_normalized_without_truth():
    v = registry.variants('N3')[0]
    p, _ = corrections.sample(v, 'engineering-control', 0, 0)
    pred, detail = corrections.evaluate(v, p)
    for name, x in pred.items():
        if x is None:
            assert name == 'half-space' and detail[name]['state'] == 'EMPTY_HARD_CONSTRAINT_SUPPORT'
        else: np.testing.assert_allclose(x.sum(-1), 1, atol=1e-12)


def test_n3_opposed_hard_constraints_preserve_failure():
    v = registry.variants('N3')[0]
    row = dict(x0=[0, 0, 0], edits=[[0,0,0],[1,0,0],[-1,0,0]], costs=[0,.1,.1],
               context=[1,0,0], picked=1, menu=[[1,0,0],[-1,0,0],[0,1,0],[0,0,1]], choice=0)
    other = deepcopy(row); other['picked'] = 2
    p = dict(history=[row, other], query=row['menu'])
    hard, detail = corrections.posterior(v, p, 'half-space')
    assert hard is None and not detail['prediction_available']
    assert np.isfinite(corrections.posterior(v, p)[0]).all()


def test_n3_all_local_fix_negative_control_is_direction_free():
    v=deepcopy(registry.variants('N3')[0]);v['settings']['local_share']=1.
    p,_=corrections.sample(v,'all-local-known-answer',0,0)
    _,detail=corrections.posterior(v,p,'corrections-only')
    np.testing.assert_allclose(detail['direction_posterior'],np.full(26,1/26),atol=1e-12)


def test_n1_law_blocks_cover_all_cultures_without_replacement():
    v = deepcopy(registry.variants('N1')[0]); v['settings'].update(makers=1, records=0)
    laws = [makers.n1_world(v, 'law-control', i)[1]['law'] for i in range(32)]
    assert len(set(laws[:16])) == len(set(laws[16:])) == 16
    assert any(x >= 16 for x in laws)


def test_n5_zero_offset_calibration_identity():
    v = registry.variants('N5')[0]
    p, truth = projection.sample(v, 'engineering-control', 0, 0)
    pred, detail = projection.evaluate(v, p, truth)
    np.testing.assert_allclose(pred['projection'], pred['calibrated-projection'], atol=1e-12)
    assert detail['headroom_eligible'] is False


def test_n6_zero_stakes_is_not_forecast_information_null():
    v = deepcopy(registry.variants('N6')[0]); v['settings'].update(records=32, threshold=.7)
    p = dict(history=[dict(claim=0, procedure=0)]*32)
    a, ad = uptake.infer(v, p); b, bd = uptake.infer(v, p, withdraw=True)
    assert bd['updates'] < ad['updates']
    assert np.max(np.abs(a-b)) > .01
    v['settings']['stakes'] = 1
    np.testing.assert_allclose(a, uptake.infer(v, p)[0], atol=1e-12)


def test_n7_native_forecast_normalized_and_no_truth_input():
    v = registry.variants('N7')[0]
    p, _ = statements.sample(v, 'engineering-control', 0, 0)
    pred, detail = statements.infer(v, p)
    np.testing.assert_allclose(pred.sum(-1), 1, atol=1e-12)
    assert len(detail['recipe_presence']) == 10


def test_n8_no_play_neutralizes_mode_signal():
    v = registry.variants('N8')[0]
    p, _ = play.sample(v, 'engineering-control', 0, 0)
    pred, _ = play.evaluate(v, p)
    np.testing.assert_allclose(pred['law-marginal'], pred['mode-blind'], atol=1e-12)


def test_n4_additive_linear_identity_and_no_abstention_leak():
    v = deepcopy(registry.variants('N4')[0]); v['settings']['records'] = 2
    p, truth = values.sample(v, 'engineering-control', 0, 0)
    for row in p['history']:
        if not row['acted']:
            assert row['choice'] is None
    exact = values.infer(v, p)[0]; linear = values.infer(v, p, linear=True)[0]
    np.testing.assert_allclose(exact, linear, atol=1e-12)
