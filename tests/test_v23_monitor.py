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
def test_disappeared_preparation_owner_requests_one_health_event(tmp_path,monkeypatch):
    from datetime import datetime,timezone
    from runners import watch_v23_events as watcher
    from ghostscale.validation.soundingline.v23.common import write
    service=tmp_path/'preparation-queue-3'
    write(service/'STATUS.json',dict(state='RUNNING',pid=123,heartbeat=datetime.now(timezone.utc).isoformat()))
    monkeypatch.setattr(watcher,'alive',lambda pid:False)
    config=dict(owned_service_states=[str(service)])
    watcher.observe_services(tmp_path,config);watcher.observe_services(tmp_path,config)
    assert len(list((tmp_path/'events').glob('*owner-unhealthy.json')))==1


def test_notifier_restart_preserves_prior_accounting(tmp_path):
    import pytest
    write(tmp_path/'attempts/notifier.json',dict(cpu_seconds=123,state='stopped'))
    with pytest.raises(ValueError,match='preserve existing'):
        monitor.service_cost_path(tmp_path,{})
    assert monitor.service_cost_path(tmp_path,dict(cost_identity='notifier-next')).name=='notifier-next.json'
    assert read(tmp_path/'attempts/notifier.json')['cpu_seconds']==123
    with pytest.raises(ValueError,match='invalid notifier'):
        monitor.service_cost_path(tmp_path,dict(cost_identity='../escape'))


def test_review_environment_pins_existing_owner_configuration(tmp_path,monkeypatch):
    import pytest
    with pytest.raises(ValueError,match='owner configuration'):
        monitor.review_environment(dict(review_user_profile=str(tmp_path)))
    (tmp_path/'.codex').mkdir();(tmp_path/'.codex/config.toml').write_text('approval_policy="on-request"')
    monkeypatch.setenv('CODEX_HOME','incorrect-inherited-home')
    env=monitor.review_environment(dict(review_user_profile=str(tmp_path)))
    assert env['CODEX_HOME']==str(tmp_path/'.codex')
    assert env['HOME']==str(tmp_path)
    assert env['USERPROFILE']==str(tmp_path)


def test_bounded_backlog_delay_never_hides_new_failure_or_deadline():
    from datetime import timedelta
    config=dict(defer_events_until=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat(),
        deferred_event_hashes={'old':'a','deadline':'d'})
    old=dict(id='old',kind='instrument_failure',event_sha256='a')
    assert not monitor.delivery_ready([old],['running'],config)
    changed=dict(old,event_sha256='b')
    assert monitor.delivery_ready([changed],['running'],config)==[changed]
    deadline=dict(id='deadline',kind='final_deadline',event_sha256='d')
    assert monitor.delivery_ready([deadline],['running'],config)==[deadline]
    config['defer_events_until']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    assert monitor.delivery_ready([old],['running'],config)==[old]


def test_one_actionable_checkpoint_does_not_change_fixed_cadence(tmp_path):
    acceptance=dict(opened='2026-10-15T05:04:15+00:00',deadline='2026-10-22T05:04:15+00:00',report_start='2026-10-22T03:04:15+00:00')
    config=dict(operator_checkpoints=[dict(id='recovery-review',due='2026-10-16T20:05:00+00:00',reason='Review expected recovered partition assembly')])
    moment=datetime(2026,10,16,20,6,tzinfo=timezone.utc)
    monitor.scheduled(tmp_path,acceptance,moment,config);monitor.scheduled(tmp_path,acceptance,moment,config)
    assert len(list((tmp_path/'events').glob('operator-*.json')))==1
    assert len(list((tmp_path/'events').glob('health12-*.json')))==3


def test_later_committed_dependencies_are_observed_after_initial_schema(tmp_path,monkeypatch):
    from types import SimpleNamespace
    write(tmp_path/'events/canonical-schema-committed.json',dict(id='canonical-schema-committed'))
    calls=[]
    def run(argv,**kwargs):calls.append(argv);return SimpleNamespace(returncode=0,stdout='b'*40+'\n')
    monkeypatch.setattr(monitor.subprocess,'run',run)
    config=dict(watched_committed_dependencies=[dict(repository=str(tmp_path),baseline_commit='a'*40)])
    monitor.dependency(tmp_path,config);monitor.dependency(tmp_path,config)
    assert len(list((tmp_path/'events').glob('committed-dependency-*.json')))==1
    assert calls==[['git','rev-parse','HEAD']]*2
