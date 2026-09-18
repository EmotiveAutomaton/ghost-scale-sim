import json
from pathlib import Path
from datetime import datetime,timezone
from runners import watch_v18_1_events as events


def test_no_healthy_wake_and_only_commissioned_times(tmp_path,monkeypatch):
    (tmp_path/'ACCEPTANCE.json').write_text(json.dumps({'started_at':'2026-09-18T16:17:22Z'}))
    (tmp_path/'Q-STATUS.json').write_text(json.dumps({'state':'running','pid':123}))
    monkeypatch.setattr(events,'alive',lambda _:True)
    assert events.observed(tmp_path,'Q',datetime(2026,9,18,17,tzinfo=timezone.utc))==[]
    found=events.observed(tmp_path,'Q',datetime(2026,9,18,21,tzinfo=timezone.utc))
    assert len(found)==1 and found[0]['payload']['elapsed_hour']==4
    monkeypatch.setattr(events,'alive',lambda _:False)
    assert events.observed(tmp_path,'Q',datetime(2026,9,18,17,tzinfo=timezone.utc))[0]['kind']=='supervisor_disappeared'


def test_event_delivery_stays_disarmed_and_preserves_live_owner(tmp_path,monkeypatch):
    delivery=events.Delivery(tmp_path,{'command':['unused']})
    delivery.tick([dict(id='event',kind='queue_complete')]);assert delivery.state['attempts']==[]
    delivery.state['attempts']=[dict(pid=123,events=['event'])];delivery.save()
    monkeypatch.setattr(events,'alive',lambda _:True)
    (tmp_path/'ARMED').touch()
    restarted=events.Delivery(tmp_path,{'command':['unused']});restarted.tick([dict(id='event')])
    assert 'exit_code' not in restarted.state['attempts'][0]


def test_changed_review_binary_retains_failure(tmp_path):
    executable=tmp_path/'fake.exe';executable.write_bytes(b'not executed')
    delivery=events.Delivery(tmp_path/'state',{'command':[str(executable),'exec'],'executable_sha256':'wrong'})
    (delivery.root/'ARMED').touch();delivery.tick([dict(id='event',kind='queue_complete')])
    assert delivery.state['failed']==['event'] and delivery.child is None


def test_native_transition_delivery_stdin_and_single_owner(tmp_path):
    import sys,hashlib
    executable=sys.executable
    command=[executable,'-B','-c',
             "import sys,json; p=sys.stdin.read(); print(json.dumps({'received': 'V18.1' in p, 'event': 'queue_complete' in p}))"]
    delivery=events.Delivery(tmp_path,dict(command=command,executable_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                                        cwd=str(tmp_path),campaign_root=str(tmp_path)))
    (tmp_path/'ARMED').touch();payload=[dict(id='fixture-completion',kind='queue_complete',payload={'fixture':True})]
    delivery.tick(payload);assert delivery.child is not None and not (tmp_path/'ARMED').exists()
    delivery.child.wait(timeout=10);delivery.tick(payload)
    assert delivery.state['delivered']==['fixture-completion'] and len(delivery.state['attempts'])==1
    assert json.loads((tmp_path/'review-1.jsonl').read_text())=={'received':True,'event':True}
