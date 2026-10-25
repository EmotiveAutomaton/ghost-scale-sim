from datetime import datetime,timezone
from ghostscale.validation.soundingline.v23.common import write
from runners.watch_v23_events import deferred_events
from runners.watch_v25 import actionable

def test_routine_blocks_batch_but_failures_and_health_wake():
    block=dict(id='b',kind='batch_complete',evidence=dict(packet='v24-n16-confirmation-block-002'))
    assert actionable([block],['next'])==[]
    assert actionable([block],[])==[block]
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
