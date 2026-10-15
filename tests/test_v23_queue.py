from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from runners import queue_v23_families as queue
from ghostscale.validation.soundingline.v23.common import write,read,digest


def test_unstarted_admissions_and_burnin_reservations_count_toward_resources(tmp_path):
    write(tmp_path/'ACCEPTANCE.json',dict(max_scientific_workers=3,cpu_seconds=90000,protected_cpu_seconds=18000))
    (tmp_path/'attempts').mkdir()
    plan=dict(cpu_cap_seconds=1000,budget_group='repairs',family='R1',second_burnin_reservation_seconds=4000)
    entries=[(tmp_path/'a',plan,{})]*3
    assert queue.capacity(tmp_path,entries,300)=='NO_SLOT'
    entries=[(tmp_path/'a',plan,dict(state='complete',cpu_seconds=2000))]
    assert queue.capacity(tmp_path,entries,47000)=='REPAIRS_GROUP_CAP'
    assert queue.capacity(tmp_path,entries,47000,second_family='R1')=='AVAILABLE'


def test_terminal_family_card_cannot_launch_or_reinterpret_results(tmp_path,monkeypatch):
    now=datetime.now(timezone.utc)
    write(tmp_path/'ACCEPTANCE.json',dict(report_start=(now+timedelta(days=1)).isoformat()))
    summary=tmp_path/'cal.json';write(summary,dict(kill_threshold=.05,prior=[.08,.05]))
    write(tmp_path/'CALIBRATION_ACCEPTANCE.json',dict(summary=str(summary),summary_sha256=digest(summary),complete_sha256='a'))
    root=tmp_path/'packets/first'
    write(root/'COMPLETE.json',dict(state='verified'));write(root/'NUMERICAL_REVIEW.json',dict(state='passed'))
    write(root/'STATUS.json',dict(state='complete'))
    write(root/'MACHINE_CARD.json',dict(state='NOMINATED',complete_sha256=digest(root/'COMPLETE.json'),
        numerical_review_sha256=digest(root/'NUMERICAL_REVIEW.json'),calibration_complete_sha256='a'))
    def forbidden(*args,**kwargs):raise AssertionError('terminal family was rerun')
    monkeypatch.setattr(queue,'launch',forbidden);monkeypatch.setattr(queue.family,'decide',forbidden)
    result=queue.step(tmp_path,dict(families={'R1':dict(first_packet=str(root))}))
    assert result['state']=='REVIEW_BOUNDARY'
    assert read(tmp_path/'events/r1-decision-1.json')['kind']=='family_decision'
    write(root/'NUMERICAL_REVIEW.json',dict(state='changed'),immutable=False)
    with pytest.raises(ValueError,match='bindings'):queue.step(tmp_path,dict(families={'R1':dict(first_packet=str(root))}))
