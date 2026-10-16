from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import twin_admission
from runners import queue_v23_confirmation as confirmation,queue_v23_twins as twins,queue_v23_extensions as extension


def test_twin_profile_refuses_controlled_source_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(twin_admission.twin_pipeline,'reference',lambda plan:{})
    sources={}
    for name in twin_admission.MODULES:
        path=tmp_path/'ghostscale/validation/soundingline/v23'/(name+'.py');path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
        sources[path.relative_to(tmp_path).as_posix()]=digest(path)
    receipt=tmp_path/'CONTROL.json';write(receipt,dict(exit_code=0,implementation_source_hashes=sources))
    extras=dict(control_receipt=str(receipt),control_receipt_sha256=digest(receipt))
    assert twin_admission.validate('twin-profile','T1',extras,tmp_path)==(0.,0.)
    path.write_text('# changed\n')
    with pytest.raises(ValueError,match='source changed'):twin_admission.validate('twin-profile','T1',extras,tmp_path)


def test_missing_twin_inputs_are_explicit_terminal_dependencies(tmp_path,monkeypatch):
    write(tmp_path/'ACCEPTANCE.json',dict(report_start='9999-12-31'))
    outcome=twins.step(tmp_path,tmp_path/'service',dict(families={name:{} for name in ('T1','T2','T3','T4')}))
    assert outcome['state']=='REVIEW_BOUNDARY'
    assert all(row['state']=='DEPENDENCY_HELD' and row['missing_measured_fields'] for row in outcome['families'].values())
    assert not (tmp_path/'packets').exists()


def test_confirmation_does_not_infer_or_freeze_an_order(tmp_path):
    write(tmp_path/'ACCEPTANCE.json',dict(report_start='9999-12-31'))
    outcome=confirmation.step(tmp_path,tmp_path/'service',dict(order=str(tmp_path/'absent-order')))
    assert outcome['state']=='REVIEW_BOUNDARY' and not (tmp_path/'absent-order').exists()


def test_population_precedence_waits_only_for_prior_admission(tmp_path,monkeypatch):
    write(tmp_path/'ACCEPTANCE.json',dict(report_start='9999-12-31'))
    config=dict(admission_precedence=dict(packet=str(tmp_path/'other-packet'),service_state=str(tmp_path/'other-controller')))
    assert extension.step(tmp_path,config)['phase']=='WAITING_FOR_PRIOR_AUTHORIZED_ADMISSION'
    write(tmp_path/'other-controller/STATUS.json',dict(state='REVIEW_BOUNDARY'))
    config['families']={};monkeypatch.setattr(extension.queue,'inventory',lambda campaign:[])
    assert extension.step(tmp_path,config)['state']=='REVIEW_BOUNDARY'
