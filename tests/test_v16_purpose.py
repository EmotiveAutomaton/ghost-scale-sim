import copy
import pytest
from ghostscale.validation.soundingline.v16.purpose_craft import DESIGN,EXTENSION
from ghostscale.validation.soundingline.v16.purpose_study import execute_unit
from ghostscale.validation.soundingline.v16.purpose_gates import run
from ghostscale.validation.soundingline.v16.audit_purpose import audit_unit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess


def test_purpose_relation_and_inhibition_are_measured_before_discovery():
    assert run()["instrument_state"]=="valid"


def test_adaptation_executes_and_independent_checker_detects_cost_tampering(tmp_path):
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            root=tmp_path/condition["id"]
            row=execute_unit(root,condition,0,namespace="purpose-fixture",packet={"packet_hash":"fixture"},reader=reader,scope="fixture")
            audit_unit(root,row)
            assert execute_unit(root,condition,0,namespace="purpose-fixture",packet={"packet_hash":"fixture"},reader=reader)==row
            altered=copy.deepcopy(row)
            altered["arms"]["inhibited"]["outcomes"]["checking_cost"]+=1
            with pytest.raises(ValueError,match="cost"):
                audit_unit(root,altered)
