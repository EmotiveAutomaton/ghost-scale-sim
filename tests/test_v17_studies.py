"""Local A2/C/D/E known answers and evidence contracts, independent of scientific scores."""
import copy
from pathlib import Path
import os
import pytest
from ghostscale.validation.soundingline.v17 import adaptive as c,revision as d,observer as e,craft_extension as a,stitch_adapter as s
from ghostscale.validation.soundingline.v17.contracts import Costs,COST_KINDS
from ghostscale.validation.soundingline.v16 import assembly

@pytest.fixture(scope="module")
def training():
    return c.prepare(dict(namespace="v17-controller-validity"))

def test_controller_development_is_separate_and_actual_effort_bounded(training):
    assert "controller-development" in training["namespace"]
    case=c.make_case("evaluation-distinct",1,2,"false_belief")
    before=copy.deepcopy(case["public"])
    rows=c.evaluate_case(case,dict(prepared=training))
    assert len(rows)==21
    assert all(r["hypotheses"]<=case["public"]["resource_context"]["hypothesis_limit"] for r in rows)
    assert case["public"]==before
    identities={r["case_id"] for r in training["records"]}
    assert case["case_id"] not in identities
    assert training["learning_operations"]==sum(r["cost"]+1 for r in training["records"])
    c3=c.option_prediction(before,"inverse_3")
    c9=c.option_prediction(before,"inverse_9")
    assert c3["hypotheses"]==3 and c9["hypotheses"]==9
    assert c9["costs"]["hypothetical_execution"]>c3["costs"]["hypothetical_execution"]

def test_access_is_different_from_accidentally_correct_belief(training):
    true=c.make_case("epistemic",0,0,"true_belief")
    false=c.make_case("epistemic",0,0,"false_belief")
    ignorant=c.make_case("epistemic",0,0,"ignorance")
    lucky=c.make_case("epistemic",0,0,"accidentally_correct")
    assert true["private"]["actual_belief_before"][0]>.99
    assert false["private"]["actual_belief_before"][1]>.99
    assert ignorant["private"]["actual_belief_before"]==pytest.approx([.5,.5])
    assert lucky["private"]["actual_belief_before"][0]>.5
    assert not lucky["public"]["resource_context"]["has_current_access"]

def test_invalid_assembly_rotation_and_valid_dependency_repair():
    world=assembly.World()
    assert not assembly.execute(world,[6],initial=(0,0,0))["legal"]
    repaired=assembly.execute(world,[4,5,6,1,2,9],initial=(0,0,0))
    assert repaired["legal"] and repaired["state"]==[1,0,0]

def test_stop_always_fails_and_appropriate_stopping_exists():
    regrets=[]
    for h in range(4):
        for regime in ("stable","stale"):
            case=d.make_case("stop-validity",0,h,regime)
            row=next(r for r in d.evaluate_case(case) if r["method"]=="stop_always")
            regrets.append(row["stop_regret"])
    assert max(regrets)>0
    assert min(regrets)==0

def test_feedback_and_internal_pass_have_matched_cognitive_cost():
    case=d.make_case("feedback-validity",0,0,"stale")
    rows={r["method"]:r for r in d.evaluate_case(case)}
    a1=rows["internal_reconsideration"]["costs"]
    b1=rows["recipient_feedback"]["costs"]
    assert a1["cold_total"]-a1["actual_execution"]==b1["cold_total"]-b1["actual_execution"]
    assert len(rows["recipient_feedback"]["feedback_queries"])==1
    assert rows["complete_error_monitor"]["decision_regret"]==pytest.approx(0)

@pytest.mark.parametrize("family",e.REGIMES)
def test_observer_projection_truth_hidden_and_process_explains_true_family(family):
    case=e.make_case("observer-validity",0,0,family)
    for tier in e.TIERS:
        request=e.project(case["public"],tier)
        assert "private" not in request and "true_hypothesis" not in str(request)
        result=e.predict(request,"inverse_maker")
        assert sum(result["probabilities"].values())==pytest.approx(1)
        assert result["probabilities"][case["private"]["future_choice"]]>0
        assert result["compatible_histories"]>=1
        if tier=="artifact":
            assert "earlier_artifacts" not in request and "recorded_process" not in request
        request["private"]=case["private"]
        with pytest.raises(ValueError,match="schema"):
            e.predict(request,"inverse_maker")

def test_typed_stitch_binding_cost_physics_and_rejection():
    body=s.sexpr("(seq (place #0) (place #1))")
    assert s.expand(body,{},(2,5))==[2,5]
    with pytest.raises(ValueError): s.expand(body,{},(2,16))
    with pytest.raises(ValueError): s.expand(s.sexpr("(#0 c0)"),{},(2,))
    defs={"fn_0":(s.sexpr("(seq (attach c0) (attach #0))"),1)}
    program=s.expand(s.sexpr("(fn_0 c1)"),defs,world="assembly")
    run=assembly.execute(assembly.World(),program+[9])
    assert run["legal"] and run["primitive_cost"]==3

@pytest.mark.parametrize("regime",a.REGIMES)
def test_construction_targets_are_reachable_and_withheld(regime):
    for index in range(3):
        case=a.make_case("reachability",index,0,regime)
        assert case["public"]["target"] not in [t["target"] for t in case["public"]["training"]]
        assert case["private"]["realized_history"]
        for method in a.METHODS[:3]:
            rep=a.acquire(case["public"],method,32,{})
            run=a.solve(case["public"],rep,8192)
            assert run["costs"]["search_total"]<=8192
            assert run["costs"]["definition_storage"]<=32
            assert not run["invalid_program"]

@pytest.mark.skipif("GS_V17_STITCH_EXE" not in os.environ,reason="pinned optional standalone executable not installed in this environment")
def test_real_stitch_standalone_learns_parameter_and_validates_cache():
    result=s.learn([dict(program=[0,cell]) for cell in (1,2,3)]*4)
    assert result["pin"]==s.PIN
    assert any(d["arity"]==1 for d in result["accepted"])
    assert result["learning_operations"]>0
    assert result==s.learn([dict(program=[0,cell]) for cell in (1,2,3)]*4)


@pytest.mark.parametrize("family,regimes",[
    ("C",["true_belief","ignorance"]),("D",["stable","stale"]),("E",["A","B","C","D"]),
    pytest.param("A2",["graphic:familiar_combinations","assembly:changed_constraint"],
        marks=pytest.mark.skipif("GS_V17_STITCH_EXE" not in os.environ,reason="optional pinned Stitch executable absent"))])
def test_each_consumer_literal_score_and_resume(tmp_path,family,regimes):
    from ghostscale.validation.soundingline.v16.records import write,read
    from ghostscale.validation.soundingline.v17.queue_runtime import packet
    import subprocess,sys
    repo=Path(__file__).resolve().parents[1]
    design=dict(family=family,namespace="v17-local-"+family,constructors=1,histories_per_constructor=1,
        regimes=regimes,cases_per_chunk=1,claim_status="discarded_development",budgets=[128],memory_caps=[32])
    write(tmp_path/"design.json",design)
    root=tmp_path/"partial"
    command=[sys.executable,"-B","-m","runners.run_v17_queue","packet","--root",str(root),"--design",str(tmp_path/"design.json")]
    first=subprocess.run(command+["--stop-after-chunks","1"],cwd=repo,capture_output=True,text=True,timeout=90)
    assert first.returncode==0,first.stderr
    original=(root/"raw/chunk-00000_points.jsonl").read_bytes()
    resumed=subprocess.run(command,cwd=repo,capture_output=True,text=True,timeout=90)
    assert resumed.returncode==0,resumed.stderr
    packet(tmp_path/"whole",design)
    assert original==(root/"raw/chunk-00000_points.jsonl").read_bytes()
    assert read(root/"COMPARISONS.json")==read(tmp_path/"whole/COMPARISONS.json")

def test_reader_bundle_extracted_prediction_and_private_read_refusal(tmp_path):
    from ghostscale.validation.soundingline.v17.queue_runtime import packet
    from ghostscale.validation.soundingline.v17.packets import load_cases
    from ghostscale.validation.soundingline.v16.records import read
    from runners.package_v17_observers import package
    import zipfile,subprocess,sys,json
    design=dict(family="E",namespace="v17-export-validity",constructors=1,histories_per_constructor=1,
        regimes=list(e.REGIMES),cases_per_chunk=4,claim_status="discarded_development")
    source=tmp_path/"source"
    packet(source,design)
    output=tmp_path/"export"
    receipt=package(source,output,1)
    extracted=tmp_path/"extracted"
    with zipfile.ZipFile(output/"reader.zip") as z:
        assert not any("private" in name.lower() or "answers" in name.lower() for name in z.namelist())
        z.extractall(extracted)
    expected={}
    with zipfile.ZipFile(output/"evaluator.zip") as z:
        for line in z.read("answers.jsonl").splitlines():
            record=json.loads(line)
            expected[record["task_id"]]=next(r["probabilities"] for r in record["reference_rows"] if r["method"]=="inverse_maker")
    request_data=(extracted/"requests.jsonl").read_text()
    executed=subprocess.run([sys.executable,"-B","consumer.py"],cwd=extracted,input=request_data,
        capture_output=True,text=True,timeout=90,env=dict(os.environ,PYTHONPATH=""))
    assert executed.returncode==0,executed.stderr
    results=[json.loads(line) for line in executed.stdout.splitlines()]
    assert len(results)==receipt["requests"]
    assert all(r["probabilities"]==pytest.approx(expected[r["task_id"]]) for r in results)
    (extracted/"PRIVATE-canary.txt").write_text("must not be read")
    probe=subprocess.run([sys.executable,"-B","consumer.py","--guard-probe"],cwd=extracted,
        capture_output=True,text=True,timeout=30,env=dict(os.environ,PYTHONPATH=""))
    assert probe.returncode==0 and json.loads(probe.stdout)["private_read_denied"]
