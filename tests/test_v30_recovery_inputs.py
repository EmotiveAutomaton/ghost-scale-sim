import pytest
from ghostscale.validation.soundingline.v30.common import write,digest
from runners.prepare_v30_followons import k_primary_jobs

def fixture(c,old_state="FAILED",new_setting=1):
    old=c/"jobs/K-CORE-R1-0135";new=c/"jobs/K-CORE-RECOVERY1-0135"
    write(old/"PLAN.json",dict(family="original",admission_sha256="a",setting=1,cpu_cap_seconds=900))
    write(new/"PLAN.json",dict(family="recovery",admission_sha256="b",setting=new_setting,cpu_cap_seconds=900))
    write(old/"COMPLETE.json",dict(state=old_state))
    write(c/"K_RESTART_RECOVERY.json",dict(replacements=[dict(original=old.name,successor=new.name,
        original_plan_sha256=digest(old/"PLAN.json"),successor_plan_sha256=digest(new/"PLAN.json"))]))
    return new

def test_recovery_keeps_full_trigger_grid_without_duplicate_settings(tmp_path):
    fixture(tmp_path)
    names=k_primary_jobs(tmp_path)
    assert len(names)==len(set(names))==156
    assert names[134]=="K-CORE-R1-0134"
    assert names[135]=="K-CORE-RECOVERY1-0135"
    assert names[136]=="K-CORE-R1-0136"

@pytest.mark.parametrize("state,setting",[("VERIFIED",1),("FAILED",2)])
def test_recovery_rejects_repeated_valid_science_or_changed_settings(tmp_path,state,setting):
    fixture(tmp_path,state,setting)
    with pytest.raises(ValueError):k_primary_jobs(tmp_path)

def test_recovery_checks_exact_plan_binding(tmp_path):
    new=fixture(tmp_path)
    (new/"PLAN.json").write_text('{}',encoding="utf-8")
    with pytest.raises(ValueError,match="plan changed"):k_primary_jobs(tmp_path)
