import copy
import pytest
from ghostscale.validation.soundingline.v16.inquiry_study import execute_unit
from ghostscale.validation.soundingline.v16.inquiry_designs import DESIGNS
from ghostscale.validation.soundingline.v16.inquiry_gates import run
from ghostscale.validation.soundingline.v16.audit_inquiry import audit_unit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.records import read


def test_admission_known_answers():
    assert run()["instrument_state"]=="valid"


def test_delayed_feedback_skill_loss_and_stop_are_actual_episodes(tmp_path):
    packet={"packet_hash":"fixture","identity":{"commission_hash":"fixture"}}
    with ReaderProcess(tmp_path/"reader") as reader:
        for condition in DESIGNS["R03"]["conditions"]:
            root=tmp_path/condition["id"]
            row=execute_unit(root,"R03",condition,0,namespace="fixture-inquiry",packet=packet,reader=reader,scope="fixture")
            audit_unit(root,row)
            assert row["arms"]["decline"]["outcomes"]["queries"]==0
            assert execute_unit(root,"R03",condition,0,namespace="fixture-inquiry",packet=packet,reader=reader)==row
            altered=copy.deepcopy(row)
            altered["arms"]["uniform"]["outcomes"]["success"]+=0.25
            with pytest.raises(ValueError,match="numeric"):
                audit_unit(root,altered)


def test_equal_examples_and_zero_practice_do_not_invent_enactment_benefit(tmp_path):
    packet={"packet_hash":"fixture","identity":{"commission_hash":"fixture"}}
    with ReaderProcess(tmp_path/"reader") as reader:
        for condition in DESIGNS["R05"]["conditions"]:
            root=tmp_path/condition["id"]
            row=execute_unit(root,"R05",condition,0,namespace="fixture-practice",packet=packet,reader=reader,scope="fixture")
            audit_unit(root,row)
            for metric in ["success","future_log_score"]:
                assert row["arms"]["enact"]["outcomes"][metric]==row["arms"]["observe"]["outcomes"][metric]
                if condition["practice_examples"]==0:
                    assert row["arms"]["enact"]["outcomes"][metric]==row["arms"]["unrelated"]["outcomes"][metric]
