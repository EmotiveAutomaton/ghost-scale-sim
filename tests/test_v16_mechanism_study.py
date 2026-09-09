import copy
import pytest
from ghostscale.validation.soundingline.v16.mechanism_study import execute_unit,DESIGN,EXTENSION
from ghostscale.validation.soundingline.v16.audit_mechanism import audit_unit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.mechanism_gates import run


def test_all_mechanism_admission_controls_pass():
    assert run()["instrument_state"]=="valid"


def test_distinct_maker_mechanisms_and_independent_physics_are_retained(tmp_path):
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION],timeout=30) as reader:
        for condition in DESIGN["conditions"]:
            root=tmp_path/condition["id"]
            row=execute_unit(root,condition,0,namespace="mechanism-fixture",packet={"packet_hash":"fixture"},reader=reader,scope="fixture")
            audit_unit(root,row)
            assert execute_unit(root,condition,0,namespace="mechanism-fixture",packet={"packet_hash":"fixture"},reader=reader)==row
            altered=copy.deepcopy(row)
            altered["arms"]["mixture"]["outcomes"]["future_log_score"]+=0.2
            with pytest.raises(ValueError,match="scoring"):
                audit_unit(root,altered)
            if condition["world"]=="W2":
                assert row["diagnostics"]["options-model"]["state"]=="not_admitted"
