from copy import deepcopy
import math
import pytest
from ghostscale.validation.soundingline.v18_4 import adaptation as A, matched_prefix as M, runtime as R
from ghostscale.validation.soundingline.v18_3.io import write


def test_live_switch_and_placebo_prefix_identity():
    w, short, st = M.stream(271, 2, 'skill', 32, 3, 12)
    _, long, lt = M.stream(271, 2, 'skill', 96, 3, 12)
    _, stationary, nt = M.stream(271, 2, 'stationary', 96, 3, 12)
    assert short == long[:32]
    assert st['states'] == lt['states'][:32]
    assert long[:12] == stationary[:12]
    assert lt['targets'][:12] == nt['targets'][:12]
    assert lt['states'][12] != lt['states'][11]
    assert lt['worlds'] == nt['worlds']
    assert long[0] == long[1] == long[2]
    # Scoring a longer future cannot alter a reader's earlier forecasts.
    a = A.evaluate_stream(271, 2, 'skill', 32, 3, w, short, st)
    b = A.evaluate_stream(271, 2, 'skill', 96, 3, w, long, lt)
    for x, y in zip(a['rows'], b['rows']): assert x['trace'] == y['trace'][:32]


def test_no_future_switch_leak_and_independent_window_answers():
    early = M.unit(272, 0, 'skill', change_at=12)
    late = M.unit(272, 0, 'skill', change_at=28)
    null = M.unit(272, 0, 'stationary', change_at=12)
    for a, b, c in zip(early['rows'], late['rows'], null['rows']):
        assert a['trace'][:12] == b['trace'][:12] == c['trace'][:12]
        assert b['trace'][:28] == c['trace'][:28]
    assert M.verify(early)
    # A deterministic arithmetic fixture detects wrong boundaries/denominators.
    trace=[dict(expected_loss=float(t),expected_match=.25) for t in range(96)]
    score=M.window_metrics(trace,28)
    assert score['prefix32_loss']==15.5
    assert score['prefix64_loss']==31.5
    assert score['prefix96_loss']==47.5
    assert score['age0_16_loss']==35.5
    assert score['age16_32_loss']==51.5
    assert score['age48_64_loss']==83.5
    assert score['prefix96_match']==.25
    broken=deepcopy(early);broken['rows'][0]['prefix32_loss']+=.1
    with pytest.raises(ValueError,match='window score'): M.verify(broken)


def test_u2_retained_summary_and_independent_replay(tmp_path):
    from runners.replay_v18_4 import finite
    specs=[dict(family='U2',index=280,cell=2,condition='skill',length=96,copy_span=3,change_at=12),
           dict(family='U2',index=281,cell=4,condition='rule',length=96,copy_span=1,change_at=28)]
    write(tmp_path/'PLAN.json',dict(design=dict(units=specs,block_size=2)))
    units=[R.dispatch(s) for s in specs]
    R.keep(tmp_path,'block-000000',units,0.,0.)
    summary=R.aggregate(tmp_path)
    assert summary['units']==2 and len(summary['cells'])==14
    write(tmp_path/'COMPLETE.json',dict(blocks=['block-000000']))
    proof=finite(tmp_path)
    assert proof['independent_means']==14*len(M.METRICS)
    assert proof['whole_unit_replays']==[0,1]


def test_old_u1_dispatch_is_unchanged():
    old=A.unit(182,cell=3,length=12,copy_span=3)
    w,history,truth=A.stream(182,3,'goal',12,3)
    assert old==A.evaluate_stream(182,3,'goal',12,3,w,history,truth)
    with pytest.raises(ValueError): M.unit(0,cell=1)
    with pytest.raises(ValueError): M.unit(0,change_at=20)
