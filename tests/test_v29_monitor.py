from datetime import datetime,timezone
from ghostscale.validation.soundingline.v29.common import write,read,digest
from runners import watch_v29 as w

def test_only_unacknowledged_primary_events_are_mirrored(tmp_path,monkeypatch):
    primary=tmp_path/'primary';c=tmp_path/'continuation';state=c/'notifications'
    write(primary/'events/landed.json',dict(id='landed',kind='batch_complete',evidence={}))
    write(primary/'events/new.json',dict(id='new',kind='batch_complete',evidence={'family':'X2'}))
    write(primary/'notifications/acks/landed.json',dict(event_sha256=digest(primary/'events/landed.json')))
    w.observe(primary,c,state,datetime.now(timezone.utc))
    assert not (c/'events/primary-landed.json').exists()
    assert read(c/'events/primary-new.json')['evidence']['primary_event_sha256']==digest(primary/'events/new.json')

def test_waiter_failure_and_single_due_plan_are_recorded(tmp_path,monkeypatch):
    primary=tmp_path/'primary';c=tmp_path/'continuation';state=c/'notifications'
    write(c/'NATIVE_OWNER.json',dict(pid=123));monkeypatch.setattr(w,'alive',lambda p:False)
    write(c/'WAKE_PLAN.json',dict(id='next-boundary',due='2026-01-01T00:00:00+00:00',reason='finish'))
    now=datetime(2026,11,13,tzinfo=timezone.utc);w.observe(primary,c,state,now);w.observe(primary,c,state,now)
    assert len(list((c/'events').glob('*.json')))==2

def test_completions_batch_but_failed_or_due_reviews_wake():
    assert not w.review_ready([])
    assert not w.review_ready([dict(kind='batch_complete')])
    for kind in ('planned_review','batch_failed','service_failed','queue_depleted'):
        assert w.review_ready([dict(kind='batch_complete'),dict(kind=kind)])
