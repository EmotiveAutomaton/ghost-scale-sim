import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import mismatch_review as V, reliability_mismatch as M
from ghostscale.validation.soundingline.v18_3.io import read, write
from test_v19_reliability_mismatch import fixture, LEGAL


def materialize(tmp_path):
    root, design = fixture(tmp_path)
    summary = M.run(root, dict(design=design), lambda **kw: None)
    write(root/'PLAN.json', dict(design=design)); write(root/'SUMMARY.json', summary)
    out = tmp_path/'review'; out.mkdir()
    return root, out, dict(bootstrap_seed=190981, bootstrap_resamples=20)


def test_scalar_odds_identity_and_both_zero_support_fallbacks():
    for accuracy in V.ACCURACIES:
        d = V.N.conditional(LEGAL, [0,1,2,3], [.1,.2,.3,.4], accuracy)
        weights = np.array([.1*accuracy,.2*accuracy,.3*(1-accuracy),.4*(1-accuracy)])
        assert np.allclose(d['tables']['skill'][0][:4], weights/math.fsum(weights), atol=1e-15, rtol=0)
        if accuracy == .5:
            for table in d['tables'].values():
                for p in table.values(): assert np.array_equal(p, d['prior'])
    d = V.N.conditional(LEGAL[:2], [0,1], [1.,0.], 1.)
    assert d['fallbacks']['belief'][1] == 'uniform-likelihood-supported-legal'
    assert d['fallbacks']['skill'][1] == 'uniform-whole-legal-group'


def test_actual_channel_infinite_loss_and_constant_endpoint_null():
    pred = np.zeros((1,2,8)); pred[0,0,0] = 1; pred[0,1,1] = 1
    for actual in V.ACCURACIES:
        s = V.N.scores(pred, [[actual,1-actual]], [0], [1.])
        assert s['infinite_loss_mass'] == 1-actual and s['finite_loss_contribution'] == 0
    pred[0,1] = pred[0,0]
    assert V.N.scores(pred, [[.5,.5]], [0], [1.])['squared_error'] == 0


def test_complete_fixture_and_all_paired_contrasts(tmp_path):
    root, out, cfg = materialize(tmp_path); result = V.review(root, out, cfg)
    assert result['passed'] and result['cells'] == 648 and result['max_error'] < 1e-12
    estimates = read(out/'INDEPENDENT_REGROUP.json')['estimates']
    assert len(estimates) == 13608
    assert {r['contrast'] for r in estimates} == {'mean','minus-none','minus-calibrated','minus-ignore-reply'}
    assert {r['actual_accuracy'] for r in estimates} == {.5,.75,1.}
    assert {r['assumed_accuracy'] for r in estimates} == {.5,.75,1.}
    assert {r['weighting'] for r in estimates} == {'native','equal-query'}


@pytest.mark.parametrize('mutation', ['forecast','actual','target','strings','law','modeled-mass','fallback','reader','cost','denominator','extra-row','missing-file'])
def test_corruption_rejected(tmp_path, mutation):
    root, out, cfg = materialize(tmp_path)
    if mutation in ('forecast','actual','target','strings'):
        p = next((root/'forecasts').glob('*uniform-legal-assumed-0.75*'))
        with np.load(p) as z: data = {k:z[k].copy() for k in z.files}
        if mutation == 'forecast': data['none'][0,0] = np.roll(data['none'][0,0], 1)
        elif mutation == 'actual': data['skill-actual-0.75'][0] = data['skill-actual-0.75'][0][::-1]
        elif mutation == 'target': data['targets'][0] = (data['targets'][0]+1)%8
        else: data['strings'] = data['strings'][::-1]
        np.savez_compressed(p, **data)
    elif mutation == 'missing-file':
        next((root/'forecasts').glob('*.npz')).unlink()
    else:
        p = root/('inputs/DISCLOSURE_LAWS.json' if mutation == 'law' else 'reader/REPLIES.json' if mutation == 'reader'
                  else 'evaluator/REPLY_LAWS.json' if mutation in ('modeled-mass','fallback','extra-row') else 'SUMMARY.json')
        data = read(p)
        if mutation == 'law': data[0]['conditional_weights'][0] += .1
        elif mutation == 'modeled-mass': data[0]['modeled_reply_mass'][0] += .1
        elif mutation == 'fallback': data[0]['fallbacks'][0] = 'wrong'
        elif mutation == 'extra-row': data.append(data[0])
        elif mutation == 'reader': next(iter(data.values()))['actual_accuracy'] = .75
        elif mutation == 'cost': data['cells'][0]['net_finite_loss'] += .1
        else: data['reader_packets'] += 1
        p.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError)):
        V.review(root, out, cfg)
