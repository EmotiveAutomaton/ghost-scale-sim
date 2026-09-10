import copy
import json
import pytest
from ghostscale.validation.soundingline.v16.preference_probe import DESIGN,EXTENSION,POLICIES,public_reader
from ghostscale.validation.soundingline.v16.preference_probe_gates import fixture,run
from ghostscale.validation.soundingline.v16.preference_probe_reference import choice,prediction
from ghostscale.validation.soundingline.v16.preference_probe_world import produce
from ghostscale.validation.soundingline.v16.preference_probe_study import execute_unit,summarize
from ghostscale.validation.soundingline.v16.audit_preference_probe import audit_unit,audit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
import random

def test_preference_probes_preserve_collisions_and_execute_real_interventions():
    assert run()["instrument_state"]=="valid",run()

def test_preference_probe_information_uses_possible_answers_and_public_bytes_only():
    public=fixture()
    for policy in POLICIES:
        first=public_reader(json.dumps(public).encode(),policy)
        reference=choice(public,policy)
        assert first["query"]==reference["query"]
        assert first["costs"]==reference["costs"]
        query=first["query"]
        event=produce(public["world"],public["task_context"],public["law"],"changed-profile",query,[0,6,1],[1,0,-1],random.Random(91)) if query else None
        visible={**public,"phase":2,"answer":{"query":query,"state":event["execution"]["state"]} if query else None}
        actual=public_reader(json.dumps(visible).encode(),policy)
        independent=prediction(visible,policy)
        assert actual["probabilities"]==pytest.approx(independent["probabilities"],abs=1e-12)
        assert actual["cause_posterior"]==pytest.approx(independent["cause_posterior"],abs=1e-12)
        assert actual["costs"]==independent["costs"]
    altered=copy.deepcopy(public);altered["answer"]={"query":"all","state":[1,0,0]}
    with pytest.raises(ValueError,match="before"):
        public_reader(json.dumps(altered).encode(),"all")
    altered["phase"]=2
    with pytest.raises(ValueError,match="unpurchased"):
        public_reader(json.dumps(altered).encode(),"none")
    altered=copy.deepcopy(public);altered["cause"]="changed-profile"
    with pytest.raises(ValueError,match="schema"):
        public_reader(json.dumps(altered).encode(),"all")

def test_preference_probe_paid_joins_future_and_all_scores_regenerate(tmp_path):
    rows=[]
    with ReaderProcess(tmp_path/"reader",extensions=[EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            row=execute_unit(tmp_path,condition,0,namespace="preference-probe-fixture",packet={"packet_hash":"fixture"},
                reader=reader,scope="fixture")
            rows.append(row);audit_unit(tmp_path,row)
            assert execute_unit(tmp_path,condition,0,namespace="preference-probe-fixture",packet={"packet_hash":"fixture"},
                reader=reader,scope="fixture")==row
    altered=copy.deepcopy(rows[0]);altered["arms"]["all"]["outcomes"]["query_fee"]+=1
    with pytest.raises(ValueError,match="scores"):
        audit_unit(tmp_path,altered)
    assert audit(tmp_path,summarize(rows))["all_reported_aggregates_reproduced"]
