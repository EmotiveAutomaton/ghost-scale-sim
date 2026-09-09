import copy
import pytest
from ghostscale.validation.soundingline.v16.attention_craft import DESIGN,EXTENSION,LEARNING_EXTENSION,public_learn
from ghostscale.validation.soundingline.v16.attention_study import execute_unit
from ghostscale.validation.soundingline.v16.attention_gates import run
from ghostscale.validation.soundingline.v16.audit_attention import audit_unit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.records import read,canonical


def test_attention_attempts_and_information_controls():
    assert run()["instrument_state"]=="valid"


def test_acquisition_is_committed_before_task_reveal_and_costs_regenerate(tmp_path):
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION,LEARNING_EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            root=tmp_path/condition["id"]
            row=execute_unit(root,condition,0,namespace="attention-fixture",packet={"packet_hash":"fixture"},reader=reader,scope="fixture")
            audit_unit(root,row)
            assert execute_unit(root,condition,0,namespace="attention-fixture",packet={"packet_hash":"fixture"},reader=reader)==row
            observation=read(root/"public"/f"{row['unit_id']}.json")["learning"]
            observation["future_target"]=7
            with pytest.raises(ValueError,match="schema"):
                public_learn(canonical(observation))
            altered=copy.deepcopy(row)
            altered["arms"]["focal-effort"]["outcomes"]["training_primitives"]+=2
            with pytest.raises(ValueError,match="outcome"):
                audit_unit(root,altered)
