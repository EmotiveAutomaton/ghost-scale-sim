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


def test_startup_accepts_venv_child_but_rejects_stale_or_unrelated_owner(monkeypatch):
    monkeypatch.setattr(events,'alive',lambda pid:pid in (101,202))
    status=dict(pid=202,launcher_pid=101,instance_id='unique-launch',queue='Q',
                heartbeat='2026-09-19T09:00:01+00:00',active_review_pid=None)
    started=datetime(2026,9,19,9,tzinfo=timezone.utc).timestamp()
    assert events.startup_ready(status,101,'unique-launch','Q',started)
    assert events.startup_ready(dict(status,pid=101),101,'unique-launch','Q',started)
    for changes in (dict(instance_id='old'),dict(queue='old'),dict(launcher_pid=999),
                    dict(pid=303),dict(active_review_pid=404),
                    dict(heartbeat='2026-09-19T08:59:59+00:00')):
        assert not events.startup_ready(dict(status,**changes),101,'unique-launch','Q',started)
    assert not events.startup_ready({},101,'unique-launch','Q',started)


def test_native_watcher_startup_handshake_and_graceful_stop(tmp_path):
    import subprocess,sys,time,uuid
    campaign=tmp_path/'campaign';campaign.mkdir()
    state=tmp_path/'state';state.mkdir()
    (campaign/'ACCEPTANCE.json').write_text(json.dumps({'started_at':datetime.now(timezone.utc).isoformat()}))
    config=tmp_path/'config.json';config.write_text(json.dumps({'command':['unused']}))
    instance=uuid.uuid4().hex;started=time.time()
    child=subprocess.Popen([sys.executable,'-B','-m','runners.watch_v18_1_events',
        '--campaign',str(campaign),'--queue','Q','--state',str(state),'--config',str(config),
        '--instance-id',instance],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        deadline=time.monotonic()+10;ready=False
        while time.monotonic()<deadline and child.poll() is None:
            try:status=json.loads((state/'STATUS.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(.05);continue
            ready=events.startup_ready(status,child.pid,instance,'Q',started)
            if ready:break
            time.sleep(.05)
        assert ready and not status['armed'] and status['pending']==0
    finally:
        (state/'STOP').touch()
        child.wait(timeout=15)
    assert child.returncode==0
