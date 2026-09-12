"""Known answers and literal-command recovery, before new V17 scientific packets."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from ghostscale.validation.soundingline.v16.records import read,write
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v17 import recipient as b
from ghostscale.validation.soundingline.v17.queue_runtime import source_identity,packet

REPO=Path(__file__).resolve().parents[1]

def design():
    return dict(family="B",namespace="v17-b-queue-fixture",constructors=2,histories_per_constructor=2,
        regimes=["correct","wrong"],cases_per_chunk=2,claim_status="discarded_development")

def command(tmp_path,root,*extra):
    path=tmp_path/"design.json"
    if not path.exists(): write(path,design())
    return subprocess.run([sys.executable,"-B","-m","runners.run_v17_queue","packet",
        "--root",str(root),"--design",str(path),*extra],cwd=REPO,capture_output=True,text=True,timeout=90)

def test_actual_recipient_interventions_both_directions():
    world=dict(templates=[1,2],visible=3)
    model=dict(order=[0,1],precision=3,prior=[.5,.5])
    assert b.recipient(1,world,model)[0] > .99
    assert b.recipient(2,world,model)[1] > .99
    reversed_model=dict(model,order=[1,0])
    assert b.recipient(1,world,reversed_model)[1] > .99
    assert b.recipient(2,world,reversed_model)[0] > .99

def test_no_information_uniform_and_distribution_goal():
    world=dict(templates=[1,2],visible=3)
    model=dict(order=[0,1],precision=0,prior=[.5,.5])
    assert b.recipient(1,world,model)==[.5,.5]
    assert b.outcome_value([.5,.5],[.5,.5]) > b.outcome_value([1,0],[.5,.5])

@pytest.mark.parametrize("regime",b.REGIMES)
def test_public_is_not_truth_and_all_outputs_normalize(regime):
    case=b.make_case("known",1,2,regime)
    snapshot=copy.deepcopy(case["public"])
    for row in b.evaluate_case(case):
        assert sum(row["probabilities"].values())==pytest.approx(1)
        assert 0<=row["brier_score"]<=2
        cost=row["costs"]
        assert cost["cold_total"] >= cost["repeat_online"] >= 0
    assert case["public"]==snapshot
    bad=copy.deepcopy(snapshot)
    bad["recipient_state"]=case["private"]["recipient_state"]
    with pytest.raises(ValueError,match="schema"):
        b.predict(bad,"inferred_recipient")
    bad=copy.deepcopy(snapshot)
    bad["candidates"][1]["program"]=[999]
    with pytest.raises(ValueError,match="candidate"):
        b.predict(bad,"inferred_recipient")

def test_support_varies_and_real_trace_executes():
    a=b.make_case("known",0,0,"correct")
    c=b.make_case("known",1,0,"correct")
    assert len(a["public"]["answer_support"])!=len(c["public"]["answer_support"])
    for case in (a,c):
        trace=case["private"]["realized_history"]
        run=b.execute([step["action"] for step in trace])
        assert run["legal"] and run["artifact"]==case["public"]["artifact"]

def test_literal_resume_rows_and_aggregates_equal(tmp_path):
    partial,whole=tmp_path/"partial",tmp_path/"whole"
    first=command(tmp_path,partial,"--stop-after-chunks","1")
    assert first.returncode==0,first.stderr
    preserved=(partial/"raw/chunk-00000_points.jsonl").read_bytes()
    resumed=command(tmp_path,partial)
    complete=command(tmp_path,whole)
    assert resumed.returncode==complete.returncode==0,(resumed.stderr,complete.stderr)
    assert preserved==(partial/"raw/chunk-00000_points.jsonl").read_bytes()
    assert read(partial/"COMPARISONS.json")==read(whole/"COMPARISONS.json")
    for chunk in read(partial/"INDEX.json")["chunks"]:
        assert (partial/chunk["path"]).read_bytes()==(whole/chunk["path"]).read_bytes()
    # An interrupted index commit recovers the existing raw bytes by exact replay.
    index=read(partial/"INDEX.json")
    index["chunks"].pop()
    write(partial/"INDEX.json",index,immutable=False)
    (partial/"COMPLETION.json").unlink()
    repaired=command(tmp_path,partial)
    assert repaired.returncode==0,repaired.stderr

def test_literal_corruption_and_duplicate_owner_refused(tmp_path):
    root=tmp_path/"corrupt"
    assert command(tmp_path,root,"--stop-after-chunks","1").returncode==0
    chunk=root/"raw/chunk-00000_points.jsonl"
    chunk.write_bytes(chunk.read_bytes()+b" ")
    failed=command(tmp_path,root)
    assert failed.returncode!=0 and "hash mismatch" in failed.stderr
    locked=tmp_path/"locked"
    with local_owner(locked):
        refused=command(tmp_path,locked)
    assert refused.returncode!=0 and "another supervisor" in refused.stderr
    assert not (locked/"STATUS.json").exists()

def test_source_change_and_science_without_admission_refused(tmp_path):
    root=tmp_path/"changed"
    packet(root,design(),stop_after_chunks=1)
    lock=read(root/"LOCK.json")
    lock["source_files"]["runners/run_v17_queue.py"]="deliberately-invalid"
    write(root/"LOCK.json",lock,immutable=False)
    with pytest.raises(ValueError,match="frozen"):
        packet(root,design())
    scientific=dict(design(),claim_status="descriptive")
    with pytest.raises(ValueError,match="admission"):
        packet(tmp_path/"not-admitted",scientific)


def test_queue_failure_is_local_and_terminal_resume_preserves_completion(tmp_path):
    from ghostscale.validation.soundingline.v17.queue_runtime import run_queue,watch_queue
    root=tmp_path/"queue"
    source=source_identity()
    write(root/"ADMISSION.json",dict(passed=True,source_files=source,scope="queue lifecycle fixture"))
    bad=dict(design(),regimes=["deliberate-invalid-regime"])
    jobs=[dict(id="deliberate-failure",design=bad,depends_on=[]),
          dict(id="dependent",design=design(),depends_on=["deliberate-failure"]),
          dict(id="independent",design=design(),depends_on=[])]
    manifest=dict(workers=1,gpu_used=False,source_files=source,jobs=jobs,campaign_target="2026-09-17T18:16:41.531700+00:00")
    write(tmp_path/"manifest.json",manifest)
    assert run_queue(root,manifest)==1
    result=read(root/"QUEUE_COMPLETION.json")
    assert result["completed"]==["independent"]
    assert result["failed_or_blocked"]==["deliberate-failure","dependent"]
    before=(root/"QUEUE_COMPLETION.json").read_bytes()
    assert watch_queue(root,tmp_path/"manifest.json")==1
    assert (root/"QUEUE_COMPLETION.json").read_bytes()==before
