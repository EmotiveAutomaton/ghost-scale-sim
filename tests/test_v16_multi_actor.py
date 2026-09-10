import copy
import json
import pytest
from ghostscale.validation.soundingline.v16.multi_actor import DESIGN,EXTENSION,public_reader
from ghostscale.validation.soundingline.v16.multi_actor_gates import run,fixture
from ghostscale.validation.soundingline.v16.multi_actor_reference import predict
from ghostscale.validation.soundingline.v16.multi_actor_study import execute_unit,summarize
from ghostscale.validation.soundingline.v16.audit_multi_actor import audit_unit,audit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess

def test_multi_actor_real_collisions_and_role_probes():
    assert run()["instrument_state"]=="valid"

def test_role_posterior_matches_independent_factorization_and_guards_paid_views():
    for revision in ["none","self","other"]:
        for policy in DESIGN["arms"]:
            public=fixture(policy,revision=revision)
            actual=public_reader(json.dumps(public).encode(),policy)
            reference=predict(public)
            for key in ["producer_core","producer_style","revision","release","brief","topology","historical_core","selector","shared_brief","revision_relation"]:
                assert actual[key]==pytest.approx(reference[key],abs=1e-12)
            assert actual["costs"]==reference["costs"]
    polluted=fixture()
    polluted["revision_view"]["routine_owner"]="editor"
    with pytest.raises(ValueError,match="revision fields"):
        public_reader(json.dumps(polluted).encode(),"all")
    polluted=fixture()
    polluted["phase"]=1
    with pytest.raises(ValueError,match="before request"):
        public_reader(json.dumps(polluted).encode(),"all")

def test_multi_actor_all_candidate_and_role_outcomes_regenerate(tmp_path):
    rows=[]
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            row=execute_unit(tmp_path,condition,0,namespace="multi-actor-fixture",packet={"packet_hash":"fixture"},reader=reader,scope="fixture")
            rows.append(row)
            audit_unit(tmp_path,row)
            assert execute_unit(tmp_path,condition,0,namespace="multi-actor-fixture",packet={"packet_hash":"fixture"},reader=reader)==row
    altered=copy.deepcopy(rows[-1])
    altered["arms"]["all"]["outcomes"]["future_revision_log_score"]+=0.1
    with pytest.raises(ValueError,match="outcome"):
        audit_unit(tmp_path,altered)
    assert audit(tmp_path,summarize(rows))["all_reported_aggregates_reproduced"]
