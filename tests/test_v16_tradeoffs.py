import copy
import json
import pytest
from ghostscale.validation.soundingline.v16.tradeoffs import design,EXTENSION,public_reader,POLICIES
from ghostscale.validation.soundingline.v16.tradeoffs_gates import run,fixture
from ghostscale.validation.soundingline.v16.tradeoffs_reference import prediction
from ghostscale.validation.soundingline.v16.tradeoffs_study import execute_unit,summarize
from ghostscale.validation.soundingline.v16.audit_tradeoffs import audit_unit,audit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess

def test_tradeoff_physics_null_and_selection_have_known_answers():
    assert run()["instrument_state"]=="valid",run()

def test_tradeoff_public_reader_matches_independent_and_denies_hidden_fields():
    public=fixture()
    for policy in POLICIES:
        actual=public_reader(json.dumps(public).encode(),policy)
        reference=prediction(public,policy)
        assert actual["probabilities"]==pytest.approx(reference["probabilities"],abs=1e-12)
        assert actual["profile_posterior"]==pytest.approx(reference["profile_posterior"],abs=1e-12)
        assert actual["costs"]==reference["costs"]
    altered=copy.deepcopy(public);altered["profile"]="changing-up"
    with pytest.raises(ValueError,match="schema"):
        public_reader(json.dumps(altered).encode(),"chronological")
    altered=copy.deepcopy(public);altered["history"][0]["private_program"]=[0,1,2,9]
    with pytest.raises(ValueError,match="hidden"):
        public_reader(json.dumps(altered).encode(),"chronological")
    altered=copy.deepcopy(public);altered["history"][0]["state"]=[-1,1,1]
    assert public_reader(json.dumps(altered).encode(),"chronological")["model_mismatch"]

def test_tradeoff_actual_dated_choices_rejections_and_all_scores_regenerate(tmp_path):
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        for card in ["V01","V02"]:
            root=tmp_path/card;rows=[]
            for condition in design(card)["conditions"]:
                row=execute_unit(root,condition,0,namespace=f"tradeoff-fixture-{card}",packet={"packet_hash":"fixture"},
                    reader=reader,scope="fixture",card=card)
                rows.append(row);audit_unit(root,row)
                assert execute_unit(root,condition,0,namespace=f"tradeoff-fixture-{card}",packet={"packet_hash":"fixture"},
                    reader=reader,scope="fixture",card=card)==row
            altered=copy.deepcopy(rows[0])
            altered["arms"]["chronological"]["outcomes"]["original_production_primitives"]+=1
            with pytest.raises(ValueError,match="outcomes"):
                audit_unit(root,altered)
            assert audit(root,summarize(rows,card))["all_reported_aggregates_reproduced"]

def test_tradeoff_dates_identify_a_trajectory_beyond_undated_scores():
    public=fixture()
    forward=public_reader(json.dumps(public).encode(),"chronological")
    swapped=copy.deepcopy(public)
    for event in swapped["history"]:
        event["context"]["date"]=7-event["context"]["date"]
    backward=public_reader(json.dumps(swapped).encode(),"chronological")
    assert forward["profile_posterior"]["changing-up"]>forward["profile_posterior"]["changing-down"]
    assert forward["profile_posterior"]["changing-up"]==pytest.approx(backward["profile_posterior"]["changing-down"],abs=1e-12)
    assert forward["profile_posterior"]["stable-low"]==pytest.approx(backward["profile_posterior"]["stable-low"],abs=1e-12)
    assert public_reader(json.dumps(public).encode(),"stable")["probabilities"]==pytest.approx(
        public_reader(json.dumps(swapped).encode(),"stable")["probabilities"],abs=1e-12)
