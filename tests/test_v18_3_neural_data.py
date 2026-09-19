import json
from collections import Counter
from itertools import combinations
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_3 import world as W,neural_data as D


def test_features_reject_truth_and_ignore_evaluator_changes():
    w=W.make_world(0,98765);h=D.history(w,1,W.rng('feature-control'),8)
    payload=W.packet(w,h);x,n,wf=D.features(payload)
    assert x.shape[0]==32 and n==8 and np.all(x[8:]==0)
    bad=json.loads(payload);bad['future']=[1,2,3]
    with pytest.raises(ValueError):D.features(W.canonical(bad))
    assert np.array_equal(x,D.features(payload)[0])


def test_balanced_withheld_combinations_and_predictive_alias():
    states=[W.STATES[i] for i in D.NEURAL_STATES if D.parity(i)==0]
    assert len(states)==8
    for a,b in combinations(range(4),2):assert set(Counter((s[a],s[b]) for s in states).values())=={2}
    checks=D.predictive_check(W.make_world(0,98765))
    assert checks[0]['new_query_span_residual']>1e-5
    assert not checks[0]['sufficient_for_declared_updates']


def test_proper_score_zero_mass_is_infinite():
    from ghostscale.validation.soundingline.v18_3.neural_runtime import proper_scores
    loss,brier,tv=proper_scores(np.array([[1.,0.]]),np.array([[0.,1.]]))
    assert np.isinf(loss[0]) and brier[0]==2 and tv[0]==1


def test_third_rule_native_reference_and_identity():
    from ghostscale.validation.soundingline.v18_3 import active,dynamics,verify
    for unit in (active.unit(98765,rule='lexicographic',split='pilot'),dynamics.unit(98765,rule='lexicographic',split='pilot')):
        assert unit['public']['world']['rule']==unit['rule']=='lexicographic'
        assert verify.check_unit(unit,True)['reference']


def test_passive_summary_alias_and_proper_decoder():
    w=W.make_world(0,97653)
    a=W.STATES.index((1,0,0,0));b=W.STATES.index((1,1,1,0))
    pa,meta=D.summary_state(w,np.eye(len(W.STATES))[a],'passive-summary')
    pb,_=D.summary_state(w,np.eye(len(W.STATES))[b],'passive-summary')
    assert np.allclose(pa,pb,atol=1e-10,rtol=0)
    assert np.all(pa>=0) and abs(pa.sum()-1)<1e-12 and meta['storage_floats']<len(W.STATES)
    future=W.artifact_matrix(w,W.context(signal=0))
    assert not np.allclose(future[a],future[b])
