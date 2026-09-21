import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import cue_correction as C

def test_analytic_controls():
    assert all(C.controls().values())

def test_duplicate_provenance_and_naive_overconfidence():
    p=np.array([.3,.7]);r=np.array([False,True]);reports=C.slots('duplicate-aware',True)
    aware=C.posterior(p,r,reports,.9);naive=C.posterior(p,r,reports,.9,True)
    assert np.allclose(aware,[.27/.34,.07/.34])
    assert naive[1]<aware[1]
    assert np.allclose(C.posterior(p,r,C.slots('independent-conflict',True),.9),p)

@pytest.mark.parametrize('report',[{'operation':'report','source':'a','assertion':1},
    {'operation':'report','assertion':True},{'operation':'mystery'},{'operation':'noop','truth':True}])
def test_invalid_reports_rejected(report):
    with pytest.raises(ValueError):C.posterior([.5,.5],[False,True],[report],.9)

def test_no_support_and_uninformative_reports():
    for arm in C.ARMS:
        assert not len(C.posterior([],[],C.slots(arm,True),.9))
        assert np.array_equal(C.posterior([.3,.7],[False,True],C.slots(arm,True),.5),[.3,.7])

def test_full_pipeline_on_constructed_fixture(tmp_path,monkeypatch):
    from ghostscale.validation.soundingline.v18_3.io import write,canonical,digest,file_digest
    import gzip
    packet=dict(schema='fixture',inputs=dict(artifact=[0,0,0]))
    records=[dict(probability=.3,maker=[0,0,0,0],steps=[dict(goal=g,operation='inspect') for g in ('a','a','b')]),
             dict(probability=.7,maker=[1,0,0,0],steps=[dict(goal=g,operation='inspect') for g in ('a','b','a')])]
    monkeypatch.setattr(C.L,'project',lambda r,tier:packet)
    monkeypatch.setattr(C.L,'validate_public',lambda p:None)
    monkeypatch.setattr(C.L,'infer',lambda p,r:dict(unknown=True))
    root=tmp_path;inp=root/'inputs';(inp/'evaluator').mkdir(parents=True)
    write(inp/'PUBLIC_PACKET.json',dict(cases=[dict(case_id='1-2-0-E1',packet=packet,input_sha256=digest(packet))]))
    write(inp/'EVALUATOR_ONLY.json',dict(cases=[dict(case_id='1-2-0-E1',trajectory_index=0,true_goals=['a','a','b'],true_operations=['inspect']*3)]))
    (inp/'evaluator/lineage-1_points.json.gz').write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(lineages=[1],priors=['neutral','self-like'],reliabilities=[.5,.9],input_files={p.relative_to(inp).as_posix():file_digest(p) for p in inp.rglob('*') if p.is_file()})
    result=C.run(root,dict(design=cfg),lambda **k:None)
    assert result['rows']==48 and result['fits']==0 and all(result['controls'].values())
    cells={(r['prior'],r['reliability'],r['arm']):r for r in result['cells']}
    assert cells['neutral',.9,'wrong']['loss']>cells['neutral',.9,'base']['loss']>cells['neutral',.9,'truthful']['loss']
    public=(root/'PUBLIC_PACKET.json').read_text()
    assert 'true_goals' not in public and 'trajectory_index' not in public and 'wrong-retracted' not in public
