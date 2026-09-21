import numpy as np
from ghostscale.validation.soundingline.v19 import forward_support as S,rollout as R,local_world as L
from ghostscale.validation.soundingline.v18_3.io import write,file_digest


def test_controls_and_sample_count_smoothing():
    assert all(S.controls().values())
    assert np.array_equal(S.smooth(S.empirical([3])),S.smooth(S.empirical([3]*16)))
    assert S.smooth(np.eye(8)[3])[3]==1-S.EPSILON+S.EPSILON/8


def test_holdout_uses_only_declared_operation_prefix():
    q=(1,0,2,3,1,4)
    assert S.withheld(q) and S.withheld((1,1,5,3,1,5))
    assert not S.withheld((1,0,2,1,3,4))
    assert not S.withheld((1,0,2,4,3,1))


def test_primitive_support_uses_full_undo_state():
    q=(1,0,2,3,1,5);counts=np.ones((2,2,8,8,6,8));a=b=q[2]
    for op in q[3:]:
        nxt=R.code(L.execute(R.ARTIFACTS[a],R.ARTIFACTS[b],L.OPERATIONS[op],(0,1,0,0)))
        counts[1,0,a,b,op,nxt]+=1;b,a=a,nxt
    result=S.support(counts,{},q)
    assert result['all_primitives_seen'] and result['visited_primitives']==3 and not result['query_seen']
    counts[1,0,6,6,5]=1
    assert not S.support(counts,{},q)['all_primitives_seen']


def test_complete_support_fixture_preserves_frozen_model(tmp_path,monkeypatch):
    rr=L.enumerate_world(L.law(190966))
    selected=[dict(next(r for r in rr if S.withheld(R.query(r))),probability=.5),
              dict(next(r for r in rr if not S.withheld(R.query(r))),probability=.5)]
    monkeypatch.setattr(L,'enumerate_world',lambda world:selected)
    inputs=tmp_path/'inputs';(inputs/'models').mkdir(parents=True)
    write(inputs/'auxiliary/selection-190301.json',dict(paths=selected))
    counts,direct=R.fit(selected);keys=sorted(direct)
    model=inputs/'models/190301-2048.npz'
    np.savez(model,transition_counts=counts,direct_keys=np.array(keys),direct_counts=np.array([direct[q] for q in keys]))
    pins={p.relative_to(inputs).as_posix():file_digest(p) for p in inputs.rglob('*') if p.is_file()}
    cfg=dict(arms=list(S.ARMS),epsilon=S.EPSILON,lineages=[190967],training_draws=[190301],input_files=pins)
    result=S.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert result['fits']==1 and result['rows']==28 and result['queries']==2
    assert all(file_digest(inputs/n)==h for n,h in pins.items())
    import json
    records=json.loads((tmp_path/'forecasts/190301-composition-holdout-support.json').read_bytes())
    assert all(not r['query_seen'] for r in records if r['withheld_composition'])
