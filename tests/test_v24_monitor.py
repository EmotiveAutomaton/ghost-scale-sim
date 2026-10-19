from datetime import datetime
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners import watch_v24_events as watcher


def test_v24_fixed_health_clock_deduplicates_and_stops_at_immutable_end(tmp_path):
    watcher.scheduled(tmp_path,datetime.fromisoformat('2026-10-22T11:59:59-07:00'))
    assert not list((tmp_path/'events').glob('*.json'))
    watcher.scheduled(tmp_path,datetime.fromisoformat('2026-10-23T01:00:00-07:00'))
    first={p.name:digest(p) for p in (tmp_path/'events').glob('*.json')}
    assert len(first)==2
    watcher.scheduled(tmp_path,datetime.fromisoformat('2026-10-23T01:01:00-07:00'))
    assert {p.name:digest(p) for p in (tmp_path/'events').glob('*.json')}==first
    watcher.scheduled(tmp_path,datetime.fromisoformat(watcher.contract.CLOSE))
    assert (tmp_path/'STOP').exists()
    assert read(tmp_path/'events/v24-pilot-window-end.json')['kind']=='final_deadline'


def test_v24_monitor_emits_only_observed_terminal_or_owner_events(tmp_path):
    prepared=tmp_path/'packets/prepared';running=tmp_path/'packets/running';done=tmp_path/'packets/done'
    write(prepared/'STATUS.json',dict(state='PREPARED_NOT_ADMITTED'))
    write(running/'STATUS.json',dict(state='RUNNING',parent_pid=123))
    write(done/'STATUS.json',dict(state='COMPLETE'));write(done/'COMPLETE.json',dict(state='PILOT_VERIFIED'))
    assert watcher.observe(tmp_path,is_alive=lambda p:True)==['running']
    assert [p.name for p in (tmp_path/'events').glob('*.json')]==['done-complete.json']
    assert watcher.observe(tmp_path,is_alive=lambda p:False)==[]
    assert read(tmp_path/'events/running-owner-missing.json')['kind']=='owner_missing'
    assert not (prepared/'original').exists()
