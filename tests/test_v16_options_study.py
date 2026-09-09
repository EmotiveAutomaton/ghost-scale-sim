import copy
import pytest
from ghostscale.validation.soundingline.v16.options_study import execute_unit,DESIGN,EXTENSION
from ghostscale.validation.soundingline.v16.options_gates import run
from ghostscale.validation.soundingline.v16.audit_options import audit_unit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess


def test_option_controls_and_independent_execution_costs(tmp_path):
    assert run()["instrument_state"]=="valid"
    packet={"packet_hash":"fixture"}
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            root=tmp_path/condition["id"]
            row=execute_unit(root,condition,0,namespace="options-fixture",packet=packet,reader=reader,scope="fixture")
            audit_unit(root,row)
            changed=copy.deepcopy(row)
            changed["arms"]["options"]["outcomes"]["search_cost"]+=1
            with pytest.raises(ValueError,match="cost"):
                audit_unit(root,changed)
