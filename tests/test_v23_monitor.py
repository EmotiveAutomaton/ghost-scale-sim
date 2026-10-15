from datetime import datetime,timezone
from runners import watch_v23_events as monitor
from ghostscale.validation.soundingline.v23.common import read,write,digest


def test_health_is_fixed_to_opening_and_finite(tmp_path):
    acceptance=dict(opened='2026-10-15T05:04:15+00:00',deadline='2026-10-22T05:04:15+00:00',report_start='2026-10-22T03:04:15+00:00')
    monitor.scheduled(tmp_path,acceptance,datetime(2026,10,16,5,5,tzinfo=timezone.utc))
    assert len(list((tmp_path/'events').glob('*.json')))==2
    monitor.scheduled(tmp_path,acceptance,datetime(2026,11,1,tzinfo=timezone.utc))
    assert len(list((tmp_path/'events').glob('health*.json')))==13
    assert (tmp_path/'STOP').exists()


def test_exact_ack_and_routine_batching(tmp_path):
    state=tmp_path/'notifications'
    monitor.event(tmp_path,'one','batch_complete',{'hash':'a'})
    events=monitor.pending(tmp_path,state)
    assert not monitor.delivery_ready(events,['running'],{})
    assert monitor.delivery_ready(events,[],{})
    write(state/'acks/one.json',dict(event_sha256=digest(tmp_path/'events/one.json')))
    assert not monitor.pending(tmp_path,state)


def test_failure_is_urgent_and_closure_needs_report(tmp_path):
    monitor.event(tmp_path,'failure','instrument_failure',{})
    assert monitor.delivery_ready(monitor.pending(tmp_path,tmp_path/'notifications'),['running'],{})
    report=tmp_path/'report.txt';report.write_text('Final report')
    write(tmp_path/'CLOSEOUT.json',dict(state='closed',owned_scientific_services_stopped=True,
        final_report=str(report),final_report_sha256=digest(report)))
    assert monitor.closed(tmp_path)
