"""Meaningful monitor regressions: no routine wake storm, finite due, native owner loss."""
from datetime import datetime,timezone,timedelta
from runners import watch_v30 as watch
from ghostscale.validation.soundingline.v30.common import write,read

def fixture(tmp_path):
    c=tmp_path/"campaign";s=c/"notifications";s.mkdir(parents=True)
    now=datetime.now(timezone.utc)
    write(c/"STATUS.json",dict(pid=123,at=now.isoformat(),active=["cell"],waiting=["next"],
        counts={"VERIFIED":3},blocked={},build_complete=False))
    return c,s,now

def test_routine_completion_waits_for_one_boundary():
    assert not watch.review_ready([dict(kind="batch_complete")]*500)
    assert watch.review_ready([dict(kind="planned_review")])
    assert watch.review_ready([dict(kind="batch_failed")])

def test_due_is_fixed_and_idempotent(tmp_path,monkeypatch):
    c,s,now=fixture(tmp_path);monkeypatch.setattr(watch,"alive",lambda _:True)
    write(c/"WAKE_PLAN.json",dict(id="next-boundary",due=(now+timedelta(minutes=30)).isoformat(),reason="validation"))
    watch.observe(c,s,now);assert not list((c/"events").glob("*.json"))
    for _ in range(3):watch.observe(c,s,now+timedelta(minutes=31))
    # Stale heartbeat and due are each emitted once, never once per poll.
    assert len(list((c/"events").glob("*.json")))==2

def test_disappearance_is_immediate(tmp_path,monkeypatch):
    c,s,now=fixture(tmp_path);monkeypatch.setattr(watch,"alive",lambda _:False)
    watch.observe(c,s,now)
    assert read(c/"events/v30-controller-disappeared.json")["kind"]=="service_failed"

def test_empty_unbuilt_boundary_and_v30_pool(tmp_path,monkeypatch):
    c,s,now=fixture(tmp_path);monkeypatch.setattr(watch,"alive",lambda _:True)
    status=read(c/"STATUS.json");status.update(active=[],waiting=[])
    write(c/"STATUS.json",status,immutable=False)
    watch.observe(c,s,now);watch.observe(c,s,now)
    assert len(list((c/"events").glob("*.json")))==1
    watch.charge(c,"RUNNING")
    record=read(next((c/"accounting").glob("*.json")))
    assert record["pool"]=="reserve" and record["reservation"]==0
    assert "family" not in record

def test_rearm_drops_already_acknowledged_snapshot(tmp_path,monkeypatch):
    c,s,now=fixture(tmp_path)
    (s/'ARMED').touch()
    delivered=[]
    monkeypatch.setattr(watch,'pending',lambda *_:[])
    monkeypatch.setattr(watch,'deferred_events',lambda *args:{})
    monkeypatch.setattr(watch,'deliver',lambda *args:delivered.append(args[-1]))
    assert not watch.deliver_current(c,s,{})
    assert not delivered

def test_rearm_respects_new_deferral_and_operator_disarm(tmp_path,monkeypatch):
    c,s,now=fixture(tmp_path);(s/'ARMED').touch()
    events=[dict(id='failed',kind='batch_failed')]
    monkeypatch.setattr(watch,'pending',lambda *_:events)
    monkeypatch.setattr(watch,'deferred_events',lambda *args:{'failed':True})
    monkeypatch.setattr(watch,'deliver',lambda *args:(_ for _ in ()).throw(AssertionError('unexpected delivery')))
    assert not watch.deliver_current(c,s,{})
    def disarm(*args):
        (s/'ARMED').unlink()
        return {}
    monkeypatch.setattr(watch,'deferred_events',disarm)
    assert not watch.deliver_current(c,s,{})

def test_fresh_failure_is_delivered(tmp_path,monkeypatch):
    c,s,now=fixture(tmp_path);(s/'ARMED').touch()
    events=[dict(id='new',kind='batch_failed')];delivered=[]
    monkeypatch.setattr(watch,'pending',lambda *_:events)
    monkeypatch.setattr(watch,'deferred_events',lambda *args:{})
    monkeypatch.setattr(watch,'deliver',lambda *args:delivered.append(args[-1]))
    assert watch.deliver_current(c,s,{})
    assert delivered==[events]
