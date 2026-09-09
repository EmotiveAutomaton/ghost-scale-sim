import copy
import json
import pytest
from ghostscale.validation.soundingline.v16.recognition import DESIGN,EXTENSION,public_reader
from ghostscale.validation.soundingline.v16.recognition_study import execute_unit,summarize
from ghostscale.validation.soundingline.v16.recognition_gates import run,fixture
from ghostscale.validation.soundingline.v16.audit_recognition import audit_unit,audit,independent_prediction
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.resource_accounting import MeasuredReader

def test_acquired_core_style_separation_and_exact_prior_controls():
    assert run()["instrument_state"]=="valid"

def test_recognition_public_boundary_and_independent_prediction():
    for process in [False,True]:
        public=fixture(process=process)
        for method in DESIGN["arms"]:
            native=public_reader(json.dumps(public).encode(),method)
            reference=independent_prediction(public,method)
            for key in ["identity","future_core","future_decoration","historical_core","acquired_core","acquired_decoration"]:
                assert native[key]==pytest.approx(reference[key],abs=1e-12)
            assert native["identity_choice"]==reference["identity_choice"]
        polluted=copy.deepcopy(public)
        polluted["anonymous"][0]["true_library"]=[0,1]
        with pytest.raises(ValueError,match="observation"):
            public_reader(json.dumps(polluted).encode(),"craft")
        polluted=copy.deepcopy(public)
        polluted["world"]["maker_identity"]=0
        with pytest.raises(ValueError,match="metadata"):
            public_reader(json.dumps(polluted).encode(),"craft")

def test_larger_maker_production_and_scores_independently_regenerate(tmp_path):
    rows=[]
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            row=execute_unit(tmp_path,condition,0,namespace="recognition-fixture",packet={"packet_hash":"fixture"},reader=reader,scope="fixture")
            rows.append(row)
            audit_unit(tmp_path,row)
            assert execute_unit(tmp_path,condition,0,namespace="recognition-fixture",packet={"packet_hash":"fixture"},reader=reader)==row
        altered=copy.deepcopy(rows[0])
        altered["arms"]["craft"]["outcomes"]["future_core_log_score"]+=0.1
        with pytest.raises(ValueError,match="outcome"):
            audit_unit(tmp_path,altered)
    assert audit(tmp_path,summarize(rows))["all_reported_aggregates_reproduced"]

def test_owned_reader_resource_measurement_preserves_public_predictions(tmp_path):
    public=fixture(process=True)
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        measured=MeasuredReader(reader)
        result=measured.request(EXTENSION,public,method="craft")
        assert result==public_reader(json.dumps(public).encode(),"craft")
        sample=measured.samples[0]
        assert sample["wall_seconds"]>0
        assert sample["parent_cpu_seconds"]>=0
        if sample["resource_state"]=="measured":
            assert sample["reader_cpu_seconds"]>=0 and sample["peak_resident_bytes"]>0
        else:
            assert sample["reader_cpu_seconds"] is None and sample["measurement"]
