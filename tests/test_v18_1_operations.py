from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v18_1 import runtime,g2,common
from ghostscale.validation.soundingline.v16.records import canonical
from runners import watch_v18_1


def test_progress_replace_retries_only_transient_permission(monkeypatch,tmp_path):
    attempts=[];original=runtime.write
    def flaky(path,value,**kwargs):
        attempts.append(1)
        if len(attempts)<3:raise PermissionError('transient reader')
        return original(path,value,**kwargs)
    monkeypatch.setattr(runtime,'write',flaky);monkeypatch.setattr(runtime.time,'sleep',lambda _:None)
    runtime.live_write(tmp_path/'status.json',{'state':'running'})
    assert len(attempts)==3
    monkeypatch.setattr(runtime,'write',lambda *a,**k:(_ for _ in ()).throw(ValueError('not a permission error')))
    with pytest.raises(ValueError):runtime.live_write(tmp_path/'status.json',{})


def test_native_supervisor_empty_queue_and_plan_refusal(tmp_path):
    manifest=tmp_path/'queue.json';manifest.write_text('{"jobs":[]}')
    assert watch_v18_1.watch(manifest,'unused-interpreter')==0
    assert (tmp_path/'queue-EVENT.json').exists()
    root=tmp_path/'run';root.mkdir();(root/'PLAN.json').write_text('{}')
    import json
    manifest.write_text(json.dumps({'jobs':[dict(root=str(root),source=str(root),plan_sha256='wrong')]}))
    with pytest.raises(ValueError,match='plan changed'):watch_v18_1.watch(manifest,'unused-interpreter')


def test_query_transfer_truth_separation_and_native_rule_reuse():
    cases=g2.transfer_cases('development-transfer-contract',per_stratum=1,histories=1,sizes=(5,),families=('fork',))
    for case in cases:
        p=case['public'];truth=case['private']['true_world']
        assert len(p['models'])==12 and truth in p['models'] and len(p['observations'])==3
        assert p['models']==sorted(p['models'],key=lambda m:m['parents'])
        for observation in p['observations']:
            assert observation['outcome']==g2.observed(truth,observation['query'])
        work=common.Work(1000)
        query=g2.select_query(canonical(p),p['observations'],[],'fixed',work)
        assert p['menu'][query]==dict(kind='context',part=4)
    rows=g2.evaluate(cases[0],budgets=(512,),query_counts=(0,))
    assert all(r['forecast_probe_count']==32 and r['costs']['total_online']<=512 for r in rows)
