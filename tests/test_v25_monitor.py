from datetime import datetime,timezone
from ghostscale.validation.soundingline.v23.common import write
from runners.watch_v23_events import deferred_events
from runners.watch_v25 import actionable


def test_live_prior_review_preserved_without_killing_monitor(tmp_path,monkeypatch):
    from runners import watch_v25 as monitor
    from runners.watch_v23_events import ReviewOwnerBusyError
    from ghostscale.validation.soundingline.v23.common import read
    state=tmp_path/'notifications';state.mkdir();(state/'ARMED').touch()
    calls=[]
    def busy(*args):
        calls.append(args)
        raise ReviewOwnerBusyError('preserve live review owner')
    monkeypatch.setattr(monitor,'deliver',busy)
    assert monitor.deliver_when_available(tmp_path,state,{},[]) is False
    assert (state/'ARMED').exists()
    assert read(state/'STATUS.json')['state']=='waiting_for_existing_review'
    monkeypatch.setattr(monitor,'deliver',lambda *args:calls.append(args))
    assert monitor.deliver_when_available(tmp_path,state,{},[]) is True
    assert len(calls)==2


def test_delivery_validation_errors_still_fail(tmp_path,monkeypatch):
    import pytest
    from runners import watch_v25 as monitor
    def invalid(*args):raise ValueError('review executable changed')
    monkeypatch.setattr(monitor,'deliver',invalid)
    with pytest.raises(ValueError,match='executable changed'):
        monitor.deliver_when_available(tmp_path,tmp_path,{},[])

def test_routine_blocks_batch_but_failures_and_health_wake():
    block=dict(id='b',kind='batch_complete',evidence=dict(packet='v24-n16-confirmation-block-002'))
    assert actionable([block],['next'])==[]
    assert actionable([block],[])==[]
    for kind in ('packet_failure','twelve_hour_health','production_boundary'):
        urgent=dict(id='u',kind=kind)
        assert actionable([block,urgent],['next'])==[block,urgent]

def test_bounded_deferral_accepts_v25_clock_and_expires(tmp_path):
    state=tmp_path/'notifications';state.mkdir()
    write(tmp_path/'ACCEPTANCE.json',dict(new_starts_close='2026-11-02T04:00:00+00:00',drain='2026-11-02T06:00:00+00:00'))
    write(state/'DEFERRED_EVENTS.json',dict(events={'done':dict(event_sha256='bound',due='2026-10-26T05:00:00+00:00',reason='awaiting complete record',next_action='land')}))
    event=dict(id='done',kind='batch_complete',event_sha256='bound')
    assert deferred_events(tmp_path,state,[event],datetime(2026,10,26,4,tzinfo=timezone.utc))=={'done'}
    assert deferred_events(tmp_path,state,[event],datetime(2026,10,26,5,tzinfo=timezone.utc))==set()


def test_v25_deferral_preserves_backlog_but_never_hides_new_failure(tmp_path):
    from runners.watch_v25 import reviewable
    state=tmp_path/'notifications';state.mkdir()
    write(tmp_path/'ACCEPTANCE.json',dict(drain='2026-11-02T06:00:00+00:00'))
    write(state/'DEFERRED_EVENTS.json',dict(events={'old':dict(event_sha256='exact',due='2026-10-30T07:57:34+00:00',reason='recorded batch awaiting bounded review',next_action='verify landing')}))
    old=dict(id='old',kind='batch_complete',event_sha256='exact');new=dict(id='new',kind='packet_failure',event_sha256='new')
    assert reviewable(tmp_path,state,[old,new],datetime(2026,10,29,20,tzinfo=timezone.utc))==[new]
    changed=dict(old,event_sha256='changed')
    assert reviewable(tmp_path,state,[changed],datetime(2026,10,29,20,tzinfo=timezone.utc))==[changed]
    assert reviewable(tmp_path,state,[old],datetime(2026,10,30,8,tzinfo=timezone.utc))==[old]
