from datetime import datetime,timezone,timedelta
import pytest
from runners.admit_v23_repair import admit
from ghostscale.validation.soundingline.v23.common import write,digest


def test_missing_committed_schema_never_creates_scientific_plan(tmp_path):
    now=datetime.now(timezone.utc)
    write(tmp_path/'ACCEPTANCE.json',dict(opened=(now-timedelta(hours=1)).isoformat(),report_start=(now+timedelta(days=1)).isoformat()))
    summary=tmp_path/'summary.json';complete=tmp_path/'complete.json'
    write(summary,dict(state='CALIBRATION_PASSED'));write(complete,dict(state='verified'))
    write(tmp_path/'CALIBRATION_ACCEPTANCE.json',dict(summary=str(summary),summary_sha256=digest(summary),
        complete=str(complete),complete_sha256=digest(complete)))
    with pytest.raises(FileNotFoundError):admit(tmp_path,'R1','pilot',300)
    assert not (tmp_path/'packets').exists()


def test_unbuilt_family_cannot_enter_repair_queue(tmp_path):
    write(tmp_path/'ACCEPTANCE.json',{})
    with pytest.raises(ValueError,match='independently implemented'):admit(tmp_path,'T1','pilot',300)
    with pytest.raises(ValueError,match='independently implemented'):admit(tmp_path,'N1','family-batch',300)


def test_new_profile_timer_uncertainty_is_per_measured_group():
    from runners.admit_v23_repair import profile_estimate
    from ghostscale.validation.soundingline.v23.registry import variants
    costs=[dict(profile=[dict(variant=v['id'],cases=16,cpu_seconds=0.) for v in variants('N5')],
                auxiliary_cpu_seconds=0.)]*2
    assert profile_estimate(costs,'N5',True)==384.
    costs[0]['profile'].pop()
    with pytest.raises(ValueError,match='complete family grid'):profile_estimate(costs,'N5',True)


def test_calibration_paths_survive_frozen_service_working_directory(tmp_path,monkeypatch):
    from runners.admit_v23_repair import calibration_paths
    campaign=tmp_path/'v23';service=campaign/'services/controller';service.mkdir(parents=True)
    monkeypatch.chdir(service)
    paths=calibration_paths(campaign,dict(summary='../v23/packets/cal/original/SUMMARY.json',complete='../v23/packets/cal/COMPLETE.json'))
    assert paths==(campaign/'packets/cal/original/SUMMARY.json',campaign/'packets/cal/COMPLETE.json')


def test_resource_reallocation_preserves_total_reserve_and_gear(tmp_path):
    from runners.admit_v23_repair import budget_limits
    record=dict(authority='owner-approved-2026-10-15',total_cpu_hours=90,protected_cpu_hours=18,
                max_scientific_workers=3,group_caps_cpu_hours=dict(n2=20,n3=24,n6=8,repairs=20))
    write(tmp_path/'RESOURCE_REALLOCATION.json',record)
    assert budget_limits(tmp_path)['n2']==20
    record['protected_cpu_hours']=17;write(tmp_path/'RESOURCE_REALLOCATION.json',record,immutable=False)
    with pytest.raises(ValueError,match='unchanged total'):budget_limits(tmp_path)


def test_family_source_compatibility_rejects_changed_registered_contrast(tmp_path):
    from pathlib import Path
    import shutil
    from runners.admit_v23_repair import verify_profile_sources
    source=Path(__file__).resolve().parents[1];frozen=tmp_path/'frozen';current=tmp_path/'current';hashes={}
    for module in ('common','registry','ledger','checker','repairs'):
        relative='ghostscale/validation/soundingline/v23/'+module+'.py'
        for root in (frozen,current):
            target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,target)
        hashes[relative]=digest(frozen/relative)
    plan=dict(source_archive=str(frozen.with_suffix('.zip')),sources=hashes)
    registry_path=current/'ghostscale/validation/soundingline/v23/registry.py'
    original=registry_path.read_text(encoding='utf-8')
    registry_path.write_text(original+'\n# Unrelated documentation change.\n',encoding='utf-8')
    verify_profile_sources(plan,'R3',current)
    registry_path.write_text(original.replace("'R3': ('unit', 'effort-weighted'","'R3': ('effort-weighted', 'unit'"),encoding='utf-8')
    with pytest.raises(ValueError,match='registry.py'):verify_profile_sources(plan,'R3',current)


def test_optional_admission_requires_exhaustive_bound_mandatory_dispositions(tmp_path):
    from runners.admit_v23_repair import optional_readiness
    from ghostscale.validation.soundingline.v23.registry import ORDER
    with pytest.raises(ValueError,match='waits for mandatory'):optional_readiness(tmp_path)
    evidence=tmp_path/'held.json';write(evidence,dict(state='RESOURCE_DEFERRED'))
    rows={name:dict(state='RESOURCE_DEFERRED',reason='measured cap',evidence=str(evidence),
                   evidence_sha256=digest(evidence)) for name in ORDER if name!='N8'}
    write(tmp_path/'MANDATORY_DISPOSITIONS.json',dict(families=rows));(tmp_path/'packets').mkdir()
    assert optional_readiness(tmp_path)==digest(tmp_path/'MANDATORY_DISPOSITIONS.json')
    write(tmp_path/'packets/unfinished/PLAN.json',dict(family='R3'))
    with pytest.raises(ValueError,match='unfinished'):optional_readiness(tmp_path)
    write(tmp_path/'packets/unfinished/STATUS.json',dict(state='failed'))
    rows.pop('T1');write(tmp_path/'MANDATORY_DISPOSITIONS.json',dict(families=rows),immutable=False)
    with pytest.raises(ValueError,match='all mandatory'):optional_readiness(tmp_path)
