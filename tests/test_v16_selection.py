import copy
import json
import pytest
from ghostscale.validation.soundingline.v16.selection import DESIGN,EXTENSION,public_reader
from ghostscale.validation.soundingline.v16.selection_gates import run,fixture
from ghostscale.validation.soundingline.v16.selection_study import execute_unit,summarize
from ghostscale.validation.soundingline.v16.audit_selection import audit_unit,audit,independent_prediction
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess

def test_actual_selection_law_and_paid_evidence_controls():
    assert run()["instrument_state"]=="valid"

def test_selected_batch_likelihood_matches_independent_enumeration():
    for count in [1,4]:
        for full in [False,True]:
            for rule in ["all","random","selected"]:
                public=fixture(count=count,full=full,retention=rule)
                for audience in [None,0]:
                    public["audience"]=audience
                    for method in DESIGN["arms"]:
                        actual=public_reader(json.dumps(public).encode(),method)
                        reference=independent_prediction(public,method)
                        for key in ["acquired_style","acquired_core","audience","future_raw_style","future_raw_core","future_release_style"]:
                            assert actual[key]==pytest.approx(reference[key],abs=1e-12)
                        assert actual["costs"]==reference["costs"]
    polluted=fixture()
    polluted["history"][0]["rejected_truth"]=[0,1]
    with pytest.raises(ValueError,match="batch"):
        public_reader(json.dumps(polluted).encode(),"selection-aware")

def test_selection_retains_all_production_and_scores_fresh_outcomes(tmp_path):
    rows=[]
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            row=execute_unit(tmp_path,condition,0,namespace="selection-fixture",packet={"packet_hash":"fixture"},reader=reader,scope="fixture")
            rows.append(row)
            audit_unit(tmp_path,row)
            assert execute_unit(tmp_path,condition,0,namespace="selection-fixture",packet={"packet_hash":"fixture"},reader=reader)==row
    altered=copy.deepcopy(rows[-1])
    altered["arms"]["selection-aware"]["outcomes"]["rejected_works"]=0.0
    with pytest.raises(ValueError,match="outcome"):
        audit_unit(tmp_path,altered)
    assert audit(tmp_path,summarize(rows))["all_reported_aggregates_reproduced"]

