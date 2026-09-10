import json
from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v16.whole_replay import compare_unit, Contents, compare
from ghostscale.validation.soundingline.v16.study import unit
from ghostscale.validation.soundingline.v16.craft import DESIGN
from ghostscale.validation.soundingline.v16.expansion_adapter import execute_unit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.expansion_runner import EXTENSIONS

def test_whole_replay_regenerates_world_and_guarded_reader_with_new_timestamps(tmp_path):
    original,replay=tmp_path/"original",tmp_path/"replay"
    condition=DESIGN["conditions"][1]
    row=unit(original,condition,3,packet_hash="whole-replay-fixture",
             namespace="whole-replay-known-world",constructors=8,evidence_scope="fixture")
    with ReaderProcess(tmp_path/"reader",extensions=EXTENSIONS) as reader:
        execute_unit("K01",replay,condition,3,packet={"packet_hash":"whole-replay-fixture"},
                     namespace="whole-replay-known-world",constructors=8,scope="fixture",reader=reader)
    result=compare_unit(original,replay,row["unit_id"])
    assert result["world_reader_and_score_match"]
    path=replay/"private"/(row["unit_id"]+".json")
    payload=json.loads(path.read_bytes())
    payload["deliberately_changed_world"]=True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="replay"):
        compare_unit(original,replay,row["unit_id"])

def test_replay_alias_and_timestamp_exclusions_keep_scientific_history():
    left=Contents({})
    right=Contents({})
    a={"task_id":"old","submitted_at":"yesterday","lineage":"scientific-A",
       "acquisition_record":{"dates":[1,2,3]},"world":{"goal":3}}
    b={"task_id":"new","submitted_at":"today","lineage":"scientific-A",
       "acquisition_record":{"dates":[1,2,3]},"world":{"goal":3}}
    compare(left.clean(a),right.clean(b))
    b["acquisition_record"]["dates"][0]=0
    with pytest.raises(ValueError,match="deterministic value"):
        compare(left.clean(a),right.clean(b))
    b["acquisition_record"]["dates"][0]=1
    b["lineage"]="scientific-B"
    with pytest.raises(ValueError,match="deterministic value"):
        compare(left.clean(a),right.clean(b))


def test_inquiry_replay_restores_recorded_independent_entropy_and_rebuilds_outcomes(tmp_path):
    from ghostscale.validation.soundingline.v16.whole_replay import preserve_random_inputs
    from ghostscale.validation.soundingline.v16.expansion_adapter import design
    original,replay=tmp_path/"original-inquiry",tmp_path/"replay-inquiry"
    condition=design("R02")["conditions"][0]
    packet={"packet_hash":"known-inquiry-replay","identity":{"commission_hash":"fixture"}}
    with ReaderProcess(tmp_path/"reader-first",extensions=EXTENSIONS) as reader:
        row=execute_unit("R02",original,condition,0,namespace="known-independent-entropy",
                         packet=packet,reader=reader,constructors=8,scope="fixture")
    inputs=preserve_random_inputs(original,replay,row)
    assert len(inputs)==2
    assert not (replay/"units").exists()
    assert not (replay/"predictions").exists()
    with ReaderProcess(tmp_path/"reader-second",extensions=EXTENSIONS) as reader:
        execute_unit("R02",replay,condition,0,namespace="known-independent-entropy",
                     packet=packet,reader=reader,constructors=8,scope="fixture")
    assert compare_unit(original,replay,row["unit_id"])["world_reader_and_score_match"]
