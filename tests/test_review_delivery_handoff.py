from datetime import datetime,timezone,timedelta
import json
import pytest
from ghostscale.validation.soundingline.v23.common import read,write
from runners.watch_v23_events import event,pending,defer_review_events,finish_review_delivery,deferred_events
from runners.watch_v23_events import record_review_report


def fixture(tmp_path):
    c=tmp_path/'campaign';s=c/'notifications';s.mkdir(parents=True)
    write(c/'ACCEPTANCE.json',dict(drain=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat()))
    event(c,'done','batch_complete',dict(packet='study'))
    return c,s,pending(c,s)


def test_incomplete_review_cannot_force_immediate_redelivery(tmp_path):
    c,s,events=fixture(tmp_path);(s/'ARMED').touch()
    entry=dict(events=['done'])
    assert finish_review_delivery(c,s,entry)=={'done'}
    assert not (s/'ARMED').exists()
    assert entry['acknowledged']==[]
    assert read(s/'DELIVERY_UNRESOLVED.json')['events']==['done']


def test_canonical_handoff_rearms_without_acknowledging_unpublished_events(tmp_path):
    c,s,events=fixture(tmp_path);due=datetime.now(timezone.utc)+timedelta(hours=1)
    defer_review_events(c,s,events,due,'Verified locally; awaiting publication','Inspect next turnover')
    entry=dict(events=['done'])
    assert not finish_review_delivery(c,s,entry)
    assert (s/'ARMED').exists() and entry['acknowledged']==[]
    assert deferred_events(c,s,events,due+timedelta(seconds=1))==set()
    event(c,'new-failure','instrument_failure',dict(packet='different'))
    assert deferred_events(c,s,pending(c,s))=={'done'}
    write(c/'events/done.json',dict(id='done',kind='batch_complete',evidence='changed'),immutable=False)
    assert deferred_events(c,s,pending(c,s))==set()


def test_stop_does_not_rearm_and_deadlines_cannot_be_deferred(tmp_path):
    c,s,events=fixture(tmp_path);(s/'STOP').touch();(s/'ARMED').touch()
    due=datetime.now(timezone.utc)+timedelta(hours=1)
    defer_review_events(c,s,events,due,'Inspected','Wait')
    finish_review_delivery(c,s,dict(events=['done']))
    assert not (s/'ARMED').exists()
    event(c,'deadline','final_deadline',{})
    defer_review_events(c,s,pending(c,s),due,'Inspected','Close')
    assert 'deadline' not in deferred_events(c,s,pending(c,s))
    with pytest.raises(ValueError):defer_review_events(c,s,events,due+timedelta(days=3),'reason','action')


def test_actual_cli_report_is_retained_without_claiming_chat_delivery(tmp_path):
    entry=dict(at='2026-11-09T14:45:00+00:00',finished='2026-11-09T15:25:00+00:00',exit_code=0)
    rows=[dict(type='item.completed',item=dict(type='agent_message',text='Earlier progress')),
          dict(type='item.completed',item=dict(type='agent_message',text='Final queue report')),
          dict(type='turn.completed')]
    (tmp_path/'review-1.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    receipt=record_review_report(tmp_path,1,entry)
    assert receipt['turn_completed'] and not receipt['interactive_chat_delivery']
    assert 'Final queue report' in (tmp_path/'LATEST_REVIEW.md').read_text()
    assert 'Earlier progress' not in (tmp_path/'LATEST_REVIEW.md').read_text()
    assert record_review_report(tmp_path,1,entry)==receipt


def test_missing_report_is_explicit_even_after_successful_process_exit(tmp_path):
    receipt=record_review_report(tmp_path,1,dict(at='start',finished='finish',exit_code=0))
    assert receipt['state']=='REPORT_MISSING' and not receipt['turn_completed']
    assert 'report missing' in (tmp_path/'LATEST_REVIEW.md').read_text()


def test_progress_without_completed_turn_cannot_rearm_delivery(tmp_path):
    c,s,events=fixture(tmp_path)
    (s/'review-1.jsonl').write_text(json.dumps(dict(type='item.completed',item=dict(type='agent_message',text='Still working')))+'\n')
    entry=dict(at='start',finished='finish',exit_code=0,events=['done'])
    entry['report_record']=record_review_report(s,1,entry)
    assert entry['report_record']['state']=='REPORT_INCOMPLETE'
    defer_review_events(c,s,events,datetime.now(timezone.utc)+timedelta(hours=1),'Inspected','Check')
    (s/'ARMED').touch();finish_review_delivery(c,s,entry)
    assert not (s/'ARMED').exists()
