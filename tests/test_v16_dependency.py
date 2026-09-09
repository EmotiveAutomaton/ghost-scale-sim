import copy
import pytest
from ghostscale.validation.soundingline.v16.dependency_monitor import DESIGN,EXTENSION
from ghostscale.validation.soundingline.v16.dependency_study import execute_unit
from ghostscale.validation.soundingline.v16.dependency_gates import run
from ghostscale.validation.soundingline.v16.audit_dependency import audit_unit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess


def test_actual_dependency_and_no_false_alarm_controls():
    assert run()["instrument_state"]=="valid"


def test_paid_inspection_and_committed_repairs_regenerate(tmp_path):
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            root=tmp_path/condition["id"]
            row=execute_unit(root,condition,0,namespace="dependency-fixture",packet={"packet_hash":"fixture"},reader=reader,scope="fixture")
            audit_unit(root,row)
            assert execute_unit(root,condition,0,namespace="dependency-fixture",packet={"packet_hash":"fixture"},reader=reader)==row
            altered=copy.deepcopy(row)
            altered["arms"]["inspect-now"]["outcomes"]["inspection_queries"]=0.0
            with pytest.raises(ValueError,match="outcome"):
                audit_unit(root,altered)
