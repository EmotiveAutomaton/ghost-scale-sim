"""An independent verifier must reject a self-consistent packet with wrong scores."""
import json
from pathlib import Path
import pytest
from runners.verify_v17_screen import physics,verify
from ghostscale.validation.soundingline.v17.queue_runtime import packet
from ghostscale.validation.soundingline.v16.records import write,read,file_digest

def test_independent_assembly_physics_known_answers():
    p=dict(initial=[0,0,0],world=dict(parents=[-1,0,0],defaults=[0,0,0]))
    assert physics(p,[6])[1] is False
    assert physics(p,[4,5,6,1,2,9])==([1,0,0],True,6)
    assert physics(p,[9,4])[1] is False

def test_self_consistent_wrong_score_is_rejected(tmp_path):
    design=dict(family="B",namespace="v17-independent-control",constructors=1,histories_per_constructor=1,
        regimes=["wrong"],cases_per_chunk=1,claim_status="discarded_development")
    packet(tmp_path,design)
    assert verify(tmp_path)["passed"]
    index=read(tmp_path/"INDEX.json")
    chunk=tmp_path/index["chunks"][0]["path"]
    item=json.loads(chunk.read_bytes().splitlines()[0])
    item["rows"][0]["brier_score"]+=.25
    chunk.write_text(json.dumps(item)+"\n")
    index["chunks"][0]["sha256"]=file_digest(chunk)
    write(tmp_path/"INDEX.json",index,immutable=False)
    completion=read(tmp_path/"COMPLETION.json")
    completion["index_sha256"]=file_digest(tmp_path/"INDEX.json")
    write(tmp_path/"COMPLETION.json",completion,immutable=False)
    with pytest.raises(ValueError,match="arithmetic"):
        verify(tmp_path)


@pytest.mark.parametrize("constructor", range(4))
@pytest.mark.parametrize("history", range(4))
def test_revision_creation_trace_reaches_declared_initial(constructor, history):
    from ghostscale.validation.soundingline.v17.revision import make_case
    case=make_case("v17-trace-validity",constructor,history,"stale")
    trace=case["private"]["realized_history"]
    assert trace[0]["before"]==[-1,-1,-1]
    assert all(step["legal"] for step in trace)
    assert all(left["after"]==right["before"] for left,right in zip(trace,trace[1:]))
    assert trace[-1]["after"]==case["public"]["initial"]


def test_json_key_sorting_preserves_declared_tie_break(tmp_path):
    design=dict(family="B",namespace="v17-b-screen-1",constructors=2,histories_per_constructor=4,
        regimes=["correct"],cases_per_chunk=8,claim_status="discarded_development")
    packet(tmp_path,design)
    assert verify(tmp_path)["passed"]
