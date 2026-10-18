from datetime import datetime,timezone,timedelta
from ghostscale.validation.soundingline.v23.common import write
from runners.watch_v23_events import deferred_events

def test_exact_deferred_event_does_not_hide_new_failure_or_deadline(tmp_path):
    campaign=tmp_path/'campaign';state=tmp_path/'state';current=datetime(2026,10,18,tzinfo=timezone.utc)
    write(campaign/'ACCEPTANCE.json',dict(report_start=(current+timedelta(days=3)).isoformat()))
    events=[dict(id='dependency',kind='dependency_ready',event_sha256='a'),
        dict(id='failed',kind='instrument_failure',event_sha256='b'),
        dict(id='deadline',kind='final_deadline',event_sha256='c')]
    item=dict(event_sha256='a',due=(current+timedelta(minutes=30)).isoformat(),reason='bounded pilot running',next_action='inspect terminal pilot and land disposition')
    write(state/'DEFERRED_EVENTS.json',dict(events={'dependency':item,'deadline':dict(item,event_sha256='c')}))
    assert deferred_events(campaign,state,events,current)=={'dependency'}
    assert not deferred_events(campaign,state,events,current+timedelta(minutes=31))
    events[0]['event_sha256']='changed'
    assert not deferred_events(campaign,state,events,current)

def test_directed_order_and_shedding_are_fixed():
    from runners.queue_v23_directed import ORDER,designs
    assert ORDER==('N16','N17','N16b','N18')
    assert designs('N16')[1]['budgets']==[1,2,4,6]
    assert designs('N16')[2]['reliability_misspecification'] is False
    assert len(designs('N16')[3]['variant_ids'])==42
    assert designs('N17')[-1]['worlds']==8

def test_family_accounting_accepts_setup_packets_without_a_family(monkeypatch,tmp_path):
    from runners import queue_v23_directed as q
    monkeypatch.setattr(q.queue,'inventory',lambda _: [(None,{},dict(cpu_seconds=20)),
        (None,dict(family='N16'),dict(cpu_seconds=7)),
        (None,dict(family='N16',logical_parent='parent'),dict(cpu_seconds=7))])
    assert q.charged(tmp_path,'N16')==7

def test_retained_review_accepts_integer_readout_keys(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from ghostscale.validation.soundingline.v23 import directed_common as c
    from ghostscale.validation.soundingline.v23.common import records
    model=SimpleNamespace(FAMILY='fixture',READERS=['exact'],variants=lambda:[dict(id='cell')],
        check=lambda *args:dict(primary=.1),study_readout=lambda rows:dict(budgets={2:dict(mean=.1)}))
    monkeypatch.setattr(c,'model_for',lambda _:model)
    monkeypatch.setattr(c.ledger,'canonical_schema',lambda *args:(None,None))
    monkeypatch.setattr(c.ledger,'validate',lambda *args:None)
    fit=tmp_path/'fits';write(fit/'FROZEN.json',{})
    plan=dict(family='fixture',worlds=2,cases=64,fit_directory=str(fit),fit_hashes={},schema_directory='',schema_pin={})
    root=tmp_path/'original';folder=root/'cell';folder.mkdir(parents=True)
    for kind in ('public','truth','evidence','ledger'):
        payload={'public':dict(public={}), 'truth':dict(truth={}),
            'evidence':dict(evidence={},metrics=dict(primary=.1)),
            'ledger':dict(ledger={'rows':{'Decision':[{'context':{'observed':{}}}]}})}[kind]
        records(folder/(kind+'_points.jsonl.gz'),[dict(identity=['fixture','cell','discovery',w,i],**payload) for w in range(2) for i in range(64)])
    write(root/'SUMMARY.json',c.summarize(model,plan,{'cell':{0:dict(primary=.1),1:dict(primary=.1)}}))
    assert c.reconstruct(root,plan)['passed']

def test_completed_replay_with_live_independent_review_is_still_active(tmp_path,monkeypatch):
    from runners import watch_v23_events as m
    packet=tmp_path/'packets/job'
    write(packet/'PLAN.json',{});write(packet/'COMPLETE.json',dict(state='verified'))
    write(packet/'STATUS.json',dict(state='reviewing',pid=123))
    monkeypatch.setattr(m,'alive',lambda _:True)
    assert m.observe(tmp_path)==['job']
