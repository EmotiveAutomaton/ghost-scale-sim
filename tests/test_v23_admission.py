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
