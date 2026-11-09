from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v27.common import write, digest, read
from runners.watch_v27 import scheduled
from runners.review_shutdown import completed_shutdown


def campaign(tmp_path):
    c=tmp_path/'campaign';c.mkdir()
    anchor=datetime(2026,11,9,14,tzinfo=timezone.utc)
    write(c/'ACCEPTANCE.json',dict(opened='2026-11-07T18:18:53+00:00',
        new_starts_close='2026-11-16T04:00:00+00:00',drain='2026-11-16T06:00:00+00:00'))
    write(c/'MONITOR_START.json',dict(at=anchor.isoformat()))
    write(c/'HEALTH_CADENCE.json',dict(hours=6,anchor=anchor.isoformat()))
    write(c/'QUEUE_STATUS.json',dict(state='RUNNING',active=['science'],prepared=[],build_work_remaining=False))
    return c,anchor


def test_six_hour_schedule_is_fixed_and_does_not_duplicate_twelve_hour_events(tmp_path):
    c,t=campaign(tmp_path)
    scheduled(c,t+timedelta(hours=5,minutes=59))
    assert not list((c/'events').glob('*health*'))
    scheduled(c,t+timedelta(hours=6))
    scheduled(c,t+timedelta(hours=6))
    assert len(list((c/'events').glob('*health*')))==1
    scheduled(c,t+timedelta(hours=12))
    assert len(list((c/'events').glob('*health6*')))==2
    assert not list((c/'events').glob('*health12*'))


def test_completion_boundary_and_deadline_still_wake_early(tmp_path):
    c,t=campaign(tmp_path)
    write(c/'WAKE_PLAN.json',dict(id='early',due=(t+timedelta(minutes=20)).isoformat(),reason='turnover'))
    scheduled(c,t+timedelta(minutes=20))
    assert read(c/'events/early.json')['kind']=='planned_review'
    scheduled(c,datetime(2026,11,16,6,tzinfo=timezone.utc))
    assert (c/'STOP').exists()
    assert (c/'events/v27-drain.json').exists()


def test_only_unchanged_reviewed_holds_suspend_hourly_build_checks(tmp_path):
    c,t=campaign(tmp_path)
    write(c/'QUEUE_STATUS.json',dict(state='DRAINED',active=[],prepared=[],unbuilt=['K3'],build_work_remaining=True),immutable=False)
    write(c/'cards/K3.json',dict(id='K3'))
    (c/'protocols').mkdir()
    write(c/'REVIEWED_FRONTIER.json',dict(held_unbuilt=['K3'],card_count=1,protocol_count=0,
        input_hashes={'cards/K3.json':digest(c/'cards/K3.json')}))
    scheduled(c,t)
    assert not (c/'BUILD_WAKE.json').exists()
    write(c/'cards/new.json',dict(id='new'))
    scheduled(c,t)
    assert (c/'BUILD_WAKE.json').exists()
    scheduled(c,t+timedelta(minutes=51))
    assert list((c/'events').glob('v27-build-*'))


def review(tmp_path, *, ending='task_complete', token='review-owner-unique'):
    t=datetime(2026,11,9,14,tzinfo=timezone.utc)
    cfg=dict(review_user_profile=str(tmp_path),cwd=str(tmp_path/'repo'),completed_review_shutdown_grace_seconds=120)
    entry=dict(at=t.isoformat(),pid=1234,review_token=token)
    directory=tmp_path/'.codex/sessions/2026/11/09';directory.mkdir(parents=True)
    path=directory/'rollout-test.jsonl'
    rows=[dict(type='session_meta',payload=dict(timestamp=(t+timedelta(seconds=1)).isoformat(),cwd=cfg['cwd'],source='exec')),
          dict(type='response_item',payload=dict(role='user',text=token)),
          dict(type='event_msg',timestamp=(t+timedelta(minutes=5)).isoformat(),payload=dict(type=ending))]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    return cfg,entry,t,path


def test_owned_completed_review_requires_shutdown_grace(tmp_path):
    cfg,e,t,path=review(tmp_path)
    assert completed_shutdown(cfg,e,t+timedelta(minutes=6)) is None
    proof=completed_shutdown(cfg,e,t+timedelta(minutes=8))
    assert proof['review_pid']==1234 and proof['active_model_timeout'] is False


@pytest.mark.parametrize('ending',['token_count','task_started','error'])
def test_active_or_failed_review_is_never_timed_out(tmp_path,ending):
    cfg,e,t,path=review(tmp_path,ending=ending)
    assert completed_shutdown(cfg,e,t+timedelta(days=1)) is None


def test_ambiguous_unowned_and_partial_transcripts_do_not_release_owner(tmp_path):
    cfg,e,t,path=review(tmp_path)
    other=path.with_name('rollout-other.jsonl');other.write_bytes(path.read_bytes())
    assert completed_shutdown(cfg,e,t+timedelta(minutes=8)) is None
    other.unlink()
    assert completed_shutdown(cfg,e|dict(review_token='wrong'),t+timedelta(minutes=8)) is None
    assert completed_shutdown(cfg|dict(cwd='unrelated'),e,t+timedelta(minutes=8)) is None
    path.write_bytes(path.read_bytes().rstrip(b'\n'))
    assert completed_shutdown(cfg,e,t+timedelta(minutes=8)) is None
