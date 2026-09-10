import copy
import json
import pytest
from ghostscale.validation.soundingline.v16.audience import DESIGN,EXTENSION,public_reader
from ghostscale.validation.soundingline.v16.audience_gates import run,fixture
from ghostscale.validation.soundingline.v16.audience_study import execute_unit,summarize
from ghostscale.validation.soundingline.v16.audit_audience import audit_unit,audit,independent_prediction
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess

def test_audience_changes_execution_without_rewriting_known_history():
    assert run()["instrument_state"]=="valid"

def test_audience_reader_requires_paid_answers_and_matches_independent_joint():
    for policy in DESIGN["arms"]:
        public=fixture(policy)
        actual=public_reader(json.dumps(public).encode(),policy)
        reference=independent_prediction(public)
        for key in ["audience","historical_core","historical_vector","future_core","future_style","acquired_core","acquired_style"]:
            assert actual[key]==pytest.approx(reference[key],abs=1e-12)
        assert actual["new_composition"]==reference["new_composition"]
        assert actual["costs"]==reference["costs"]
    altered=fixture("both")
    altered["phase"]=1
    with pytest.raises(ValueError,match="before paid requests"):
        public_reader(json.dumps(altered).encode(),"both")
    altered=fixture("none")
    altered["audience"]=1
    with pytest.raises(ValueError,match="unpurchased"):
        public_reader(json.dumps(altered).encode(),"none")

def test_audience_queries_old_history_and_new_work_independently_regenerate(tmp_path):
    rows=[]
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            row=execute_unit(tmp_path,condition,0,namespace="audience-fixture",packet={"packet_hash":"fixture"},reader=reader,scope="fixture")
            rows.append(row)
            audit_unit(tmp_path,row)
            assert execute_unit(tmp_path,condition,0,namespace="audience-fixture",packet={"packet_hash":"fixture"},reader=reader)==row
    altered=copy.deepcopy(rows[0])
    altered["arms"]["both"]["outcomes"]["audience_queries"]=0.0
    with pytest.raises(ValueError,match="outcomes"):
        audit_unit(tmp_path,altered)
    assert audit(tmp_path,summarize(rows))["all_reported_aggregates_reproduced"]
