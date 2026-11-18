import pytest
from ghostscale.validation.soundingline.v31.common import write,digest
from runners.watch_v31 import PendingInventory


def test_event_completion_ack_and_new_event(tmp_path,monkeypatch):
    c=tmp_path/'c';state=tmp_path/'state';event=c/'events/one.json'
    write(event,dict(id='one',kind='batch_complete'))
    inv=PendingInventory();assert len(inv.pending(c,state))==1
    write(state/'acks/one.json',dict(event_sha256=digest(event)))
    assert not inv.pending(c,state)
    write(c/'events/two.json',dict(id='two',kind='batch_failed'))
    assert [x['id'] for x in inv.pending(c,state)]==['two']
    monkeypatch.setattr('runners.watch_v31.read',lambda p:pytest.fail('unchanged bytes reread'))
    monkeypatch.setattr('runners.watch_v31.digest',lambda p:pytest.fail('unchanged bytes rehashed'))
    assert [x['id'] for x in inv.pending(c,state)]==['two']


def test_changed_event_does_not_keep_stale_ack(tmp_path):
    c=tmp_path/'c';state=tmp_path/'state';event=c/'events/one.json'
    write(event,dict(id='one',kind='batch_complete'))
    write(state/'acks/one.json',dict(event_sha256=digest(event)))
    inv=PendingInventory();assert not inv.pending(c,state)
    write(event,dict(id='one',kind='changed_record_after_ack'),immutable=False)
    with pytest.raises(ValueError,match='acknowledgement changed'):inv.pending(c,state)


def test_ack_removal_restores_pending_event(tmp_path):
    c=tmp_path/'c';state=tmp_path/'state';event=c/'events/one.json';ack=state/'acks/one.json'
    write(event,dict(id='one',kind='batch_complete'));write(ack,dict(event_sha256=digest(event)))
    inv=PendingInventory();assert not inv.pending(c,state)
    ack.unlink()
    assert len(inv.pending(c,state))==1
