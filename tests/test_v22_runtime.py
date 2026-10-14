from datetime import datetime, timezone
from pathlib import Path
from ghostscale.validation.soundingline.v18_3.io import write
from runners import watch_v22, watch_v22_events


def test_priority_respects_pending_work_stale_heartbeat_and_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(watch_v22, 'alive', lambda pid: True)
    stamp = datetime.now(timezone.utc)
    state = dict(heartbeat=stamp.isoformat(), pid=123, active=[], pending=0)
    write(tmp_path/'QUEUE-STATUS.json', state)
    assert not watch_v22.priority_clear(tmp_path, stamp)
    (tmp_path/'HOLD').touch()
    assert watch_v22.priority_clear(tmp_path, stamp)
    for changed in [dict(pending=1), dict(active=[dict(id='owned')]), dict(heartbeat='2000-01-01T00:00:00+00:00')]:
        write(tmp_path/'QUEUE-STATUS.json', dict(state, **changed), immutable=False)
        assert not watch_v22.priority_clear(tmp_path, stamp)


def test_peer_delivery_in_flight_blocks_competing_review(tmp_path, monkeypatch):
    peer = tmp_path/'campaign/notifications'
    monkeypatch.setattr(watch_v22_events, 'alive', lambda pid: True)
    write(peer/'STATUS.json', dict(heartbeat=datetime.now(timezone.utc).isoformat(), active_review_pid=None))
    assert not watch_v22_events.peer_busy(peer)
    write(peer/'DELIVERY.json', dict(attempts=[dict(started_at='now')]))
    assert watch_v22_events.peer_busy(peer)
    write(peer/'DELIVERY.json', dict(attempts=[dict(exit_code=0)]), immutable=False)
    assert not watch_v22_events.peer_busy(peer)


def test_peer_arm_escrow_never_rearms_closed_campaign(tmp_path):
    peer, state = tmp_path/'old/notifications', tmp_path/'new/notifications'
    peer.mkdir(parents=True)
    write(state/'PEER_ARM_ESCROW.json', dict(was_armed=True))
    watch_v22_events.release_peer(state, peer)
    assert (peer/'ARMED').exists()
    (peer/'ARMED').unlink()
    write(peer.parent/'CLOSEOUT.json', dict(state='closed'))
    write(state/'PEER_ARM_ESCROW.json', dict(was_armed=True))
    watch_v22_events.release_peer(state, peer)
    assert not (peer/'ARMED').exists()
