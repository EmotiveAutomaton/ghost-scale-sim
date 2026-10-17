from runners import queue_v23_twins as q
from ghostscale.validation.soundingline.v23.common import write


def test_delivered_reference_carries_both_independent_prerequisites(tmp_path,monkeypatch):
    write(tmp_path/'ACCEPTANCE.json',dict(report_start='2999-01-01T00:00:00+00:00'))
    calls=[]
    def task(campaign,family,identity,kind,cap,extras):
        calls.append((family,extras))
        return campaign/identity,'RUNNING_OR_ADMITTED'
    monkeypatch.setattr(q,'task',task)
    config=dict(families={name:dict(empirical_reference='reference',empirical_reference_sha256='a'*64,
        empirical_source='source',profile_cap=900) for name in ('T1','T4','T3','T2')},
        implementation_controls=dict(control_receipt='controls',control_receipt_sha256='b'*64),
        engine_dependency=dict(engine_reference_packet='engine',engine_reference_complete_sha256='c'*64))
    result=q.step(tmp_path,tmp_path/'state',config)
    assert result['state']=='RUNNING' and len(calls)==4
    assert [name for name,_ in calls]==['T1','T4','T3','T2']
    for _,extras in calls:
        assert extras['control_receipt']=='controls'
        assert extras['engine_reference_packet']=='engine'
        assert extras['empirical_source']=='source'


def test_missing_input_is_a_dependency_hold_without_dispatch(tmp_path,monkeypatch):
    write(tmp_path/'ACCEPTANCE.json',dict(report_start='2999-01-01T00:00:00+00:00'))
    monkeypatch.setattr(q,'task',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not dispatch')))
    result=q.step(tmp_path,tmp_path/'state',dict(families={name:{} for name in ('T1','T4','T3','T2')}))
    assert result['state']=='REVIEW_BOUNDARY'
    assert all(row['state']=='DEPENDENCY_HELD' for row in result['families'].values())
