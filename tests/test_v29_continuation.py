"""New continuation behavior only; no primary redraws or broad regression."""
import json
from pathlib import Path
import numpy as np
import pytest
from ghostscale.validation.soundingline.v29 import ledger,q2,q2_reference,b2
from ghostscale.validation.soundingline.v29.continuation import Adapter
from ghostscale.validation.soundingline.v29.common import write,read,digest
from runners import run_v29_continuation as runner

def test_watched_failure_updates_without_owned_choice():
    s=dict(part='test',truth=3,order='FU',length=12,worlds=8,blocks=8)
    raw=q2.generate(s,'excluded-continuation-test')
    assert np.all(raw['watching'][:12]==1)
    assert np.all(raw['successes'][:,:12]==0) and np.all(raw['actions'][:,:12]==0)
    a=q2.infer(raw,s);b=q2_reference.infer(raw,s)
    for key in a:assert np.allclose(a[key],b[key],rtol=0,atol=1e-9)

def test_mild_hub_exact_reader_and_independent_path(tmp_path):
    s=dict(truth=[2,1.5,0],hub_weights=[1.,1.5,2.,4.],regime='PROBE',environment='scarce',buy_share=.5,steps=64,worlds=8,blocks=8)
    p=ledger.candidates(hub_weights=s['hub_weights'])
    assert p[p[:,0]==2,1].tolist()==s['hub_weights']
    assert Adapter(dict(module='x1',roster=[s],cpu_cap_seconds=300)).controls(tmp_path)['passed']

def test_extinction_reminder_is_paired_and_independently_read(tmp_path):
    s=dict(truth=4,rho=.5,steps=640,worlds=8,blocks=8,extinction_start=256,reminder_step=None,paired_reminder=True)
    assert Adapter(dict(module='b2',roster=[s,s|dict(reminder_step=512)],cpu_cap_seconds=300)).controls(tmp_path)['passed']

def test_b3_adapter_refits_intervals_and_reconstructs(tmp_path):
    s=dict(part='B3b',mu=.5,chi=.5,population='independent',steps=32,worlds=8,blocks=8)
    adapter=Adapter(dict(module='b3',roster=[s],cpu_cap_seconds=300))
    original=tmp_path/'original';reference=tmp_path/'reference'
    a=adapter.run(original,selected=[s],lane='excluded-continuation-test')
    b=adapter.run(reference,selected=[s],mode='reconstruct',original=original,lane='excluded-continuation-test')
    runner.compare(a,b)
    assert 'B3b_uncertainty_correction' in a
    assert all('refitting both readers' in r['interval_method'] for r in a['rows'] if r['measure']=='heldout_gain_per_decision')

def test_prepare_carries_costs_selects_long_records_and_pins_plans(tmp_path,monkeypatch):
    primary=tmp_path/'primary';c=tmp_path/'continuation';source=tmp_path/'source'
    for p in (primary/'accounting',c/'accounting',c/'families',c/'jobs',source/'runners'):p.mkdir(parents=True)
    write(primary/'ACCEPTANCE.json',dict(immutable=True))
    write(primary/'QUEUE_COMPLETE.json',dict(verified=['Q2']))
    write(primary/'QUEUE_STATUS.json',dict(pid=123))
    write(primary/'accounting/old.json',dict(family='reserve',cpu_seconds=17))
    write(primary/'families/Q2/VERIFIED.json',dict(passed=True))
    setting=dict(part='Q2',truth=0,order='UC',length=25,worlds=300,blocks=12)
    write(primary/'jobs/Q2-original/output/SUMMARY.json',dict(settings=[setting],rows=[dict(setting=setting,measure='separation',unresolved=dict(estimate=.31))]))
    for stage in runner.STAGES:write(primary/f'jobs/Q2-{stage}/COMPLETE.json',dict(cpu_seconds=1))
    write(c/'ACCEPTANCE.json',dict(primary_campaign=str(primary),primary_acceptance_sha256=digest(primary/'ACCEPTANCE.json'),interpreter='fixture',new_starts_close='2099-01-01T00:00:00+00:00'))
    write(c/'CATALOGUE.json',[dict(id='Q2-long',parent='Q2',module='q2',selection='long')]);write(c/'ENVIRONMENT.json',{})
    write(source/'SOURCE_MANIFEST.json',{});(source/'SOURCE.zip').write_bytes(b'fixture')
    monkeypatch.setattr(runner,'primary_alive',lambda pid:False);monkeypatch.setattr(runner,'__file__',str(source/'runners/run_v29_continuation.py'))
    runner.prepare(c)
    assert read(c/'accounting/primary-reserve-carried.json')['cpu_seconds']==17
    plan=read(c/'jobs/Q2-long-original/PLAN.json')
    assert plan['roster'][0]['length']==50
    assert read(c/'PREPARED_MANIFEST.json')['Q2-long-original']==digest(c/'jobs/Q2-long-original/PLAN.json')
