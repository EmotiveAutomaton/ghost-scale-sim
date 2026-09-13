"""Known answers, transaction recovery, frozen confirmation and transition-only delivery."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest
from ghostscale.validation.soundingline.v17 import continuation_cases as cc
from ghostscale.validation.soundingline.v17.continuation_analysis import verify_item,confirmation,holm
from ghostscale.validation.soundingline.v17.continuation_store import Store
from ghostscale.validation.soundingline.v17.continuation_plan import build_plan,rank_branches
from ghostscale.validation.soundingline.v17.continuation_runtime import source_identity
from ghostscale.validation.soundingline.v17.contracts import Costs
from ghostscale.validation.soundingline.v16.records import digest
from runners.watch_v17_continuation import Delivery,ALLOWED_EVENTS

@pytest.mark.parametrize("regime",cc.b.REGIMES)
def test_real_assembly_recipient_physics_and_boundary(regime):
    case=cc.make_assembly_recipient("v17-ba-validity",1,3,regime)
    assert case["private"]["realized_history"][-1]["after"]==case["public"]["initial"]
    item=dict(case=case,rows=cc.evaluate_assembly(case,{}))
    assert verify_item(item)
    bad=copy.deepcopy(case["public"]);bad["candidates"][0]["program"]=[999]
    with pytest.raises(ValueError,match="opportunity"):cc.predict_assembly(bad,"inferred_recipient")
    bad=copy.deepcopy(case["public"]);bad["recipient_state"]=case["private"]["recipient_state"]
    with pytest.raises(ValueError,match="boundary"):cc.predict_assembly(bad,"inferred_recipient")

@pytest.mark.parametrize("regime",("expert_partner","unfamiliar_partner"))
def test_sharing_uses_data_and_separates_actual_execution(regime):
    case=cc.make_sharing("v17-sharing-validity",1,2,regime)
    rows=cc.evaluate_sharing(case,{})
    assert verify_item(dict(case=case,rows=rows))
    assert {r["target"] for r in rows}=={"personal_execution","partner_prediction"}
    assert all(r["costs"]["actual_execution"]>0 for r in rows)
    p=copy.deepcopy(case["public"]);p["own_training"]=[];p["partner_training"]=[]
    for method in cc.F_METHODS:
        own,other=cc.sharing_posteriors(p,method,Costs())
        assert [x[1] for x in own]==pytest.approx([1/len(own)]*len(own))
        assert [x[1] for x in other]==pytest.approx([1/len(other)]*len(other))
    p["own_model"]=case["private"]["own_model"]
    with pytest.raises(ValueError,match="boundary"):cc.sharing_posteriors(p,"shared",Costs())

@pytest.mark.parametrize("world",("graphic","assembly"))
def test_personal_and_pooled_acquisition_withhold_target(world):
    case=cc.make_pooled("v17-pool-validity",2,1,world+":new_combinations")
    p=case["public"]
    assert len(p["training"])==len(p["pooled_training"])
    assert all(x["target"]!=p["target"] for x in p["training"]+p["pooled_training"])
    assert all(cc.a.execute(p,x["program"],x["initial"])["legal"] for x in p["pooled_training"])

def test_actual_stitch_pooled_arm(tmp_path,monkeypatch):
    if not os.environ.get("GS_V17_STITCH_EXE"):pytest.skip("optional isolated Stitch executable unavailable")
    monkeypatch.setenv("GS_V17_STITCH_CACHE",str(tmp_path/"stitch"))
    case=cc.make_pooled("v17-pool-actual",0,0,"graphic:familiar_combinations")
    rows=cc.evaluate_pooled(case,dict(memory_caps=[32],budgets=[512]))
    assert len(rows)==10
    assert verify_item(dict(case=case,rows=rows))

def test_known_confirmation_and_holm_outcomes():
    contrast=dict(id="known",constructors=10000,low=-1,high=1,minimum_gain=.05,planned_radius=.04,direction="known positive")
    positive=confirmation([.3]*10000,contrast)
    null=confirmation([.05]*10000,contrast)
    reversal=confirmation([-.3]*10000,contrast)
    assert positive["simultaneous_lower_bound"]>.05 and positive["p_value"]<.05/3
    assert null["p_value"]==reversal["p_value"]==1
    values=holm([dict(id="a",p_value=.01),dict(id="b",p_value=.03),dict(id="c",p_value=.2)])
    assert [x["holm_adjusted_p"] for x in values]==pytest.approx([.03,.06,.2])
    assert [x["holm_rejected"] for x in values]==[True,False,False]
    with pytest.raises(ValueError,match="fixed sample"):confirmation([.3],contrast)

def test_branch_rule_retains_reversals_without_using_fresh_data():
    def entry(identity,spread):
        return dict(id=identity,family="C",comparisons=[
            dict(target="x",mean_brier_or_failure=.8,between_constructor_variance=.01,constructor_clusters=32),
            dict(target="x",mean_brier_or_failure=.8-spread,between_constructor_variance=.01,constructor_clusters=32)])
    selection=rank_branches([entry("one",.4),entry("two",.2),entry("three",.01)])
    assert [x["id"] for x in selection["selected"]]==["one","two"]
    assert selection["confirmation_selection_independent"]

def item():
    case=cc.b.make_case("v17-store-validity",0,0,"correct")
    return dict(case=case,rows=cc.b.evaluate_case(case))

def test_compressed_store_refuses_corruption_and_replacement(tmp_path):
    store=Store(tmp_path/"raw.sqlite")
    design=dict(id="b")
    value=item()
    assert store.save("expansion",design,0,[value],cpu=0,wall=0)
    assert not store.save("expansion",design,0,[value],cpu=99,wall=99)
    bad=copy.deepcopy(value);bad["rows"][0]["brier_score"]+=.5
    with pytest.raises(ValueError,match="arithmetic"):store.save("expansion",design,1,[bad],cpu=0,wall=0)
    assert store.counts()["constructor_units"]==1
    store.db.execute("UPDATE units SET digest='corrupt'");store.db.commit()
    with pytest.raises(ValueError,match="hash mismatch"):list(store.iter_items())
    store.close()

def fixture_plan():
    plan=build_plan(dict(discovery="discarded test input"))
    plan.update(mode="discarded_development",run_hours=1,min_free_disk_bytes=0,max_database_bytes=100000000)
    plan["initial_expansions"]=[dict(id="b",family="B",regime="correct",namespace="v17-resume-literal-fixture",
        constructors=2,histories_per_constructor=1,claim_status="discarded_development")]
    plan["confirmation"]=[]
    plan["robustness_max_constructor_index"]=0
    plan["source_files"]=source_identity()
    return plan

def command(tmp_path,root,units=None):
    plan=tmp_path/"plan.json";admission=tmp_path/"admission.json"
    if not plan.exists():plan.write_text(json.dumps(fixture_plan()))
    if not admission.exists():admission.write_text(json.dumps(dict(passed=True,source_files=source_identity())))
    argv=[sys.executable,"-B","-m","runners.run_v17_continuation","--root",str(root),"--plan",str(plan),"--admission",str(admission)]
    if units:argv+=["--fixture-units",str(units)]
    return subprocess.run(argv,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=180)

def test_literal_resume_preserves_window_rows_and_no_duplicate(tmp_path):
    partial=tmp_path/"partial";whole=tmp_path/"whole"
    first=command(tmp_path,partial,1);assert first.returncode==0,first.stderr
    clock=(partial/"WINDOW.json").read_bytes()
    resumed=command(tmp_path,partial,1);assert resumed.returncode==0,resumed.stderr
    uninterrupted=command(tmp_path,whole,2);assert uninterrupted.returncode==0,uninterrupted.stderr
    assert (partial/"WINDOW.json").read_bytes()==clock
    a,b=Store(partial/"records.sqlite"),Store(whole/"records.sqlite")
    assert a.counts()["constructor_units"]==b.counts()["constructor_units"]==2
    assert list(a.iter_items())==list(b.iter_items())
    a.close();b.close()
    p=tmp_path/"admission.json";wrong=json.loads(p.read_bytes());wrong["source_files"]["bad"]="bad";p.write_text(json.dumps(wrong))
    refused=command(tmp_path,partial,1)
    assert refused.returncode!=0 and "source/admission mismatch" in refused.stderr
    assert (partial/"WINDOW.json").read_bytes()==clock

def test_transition_delivery_runs_once_and_never_for_healthy_state(tmp_path):
    (tmp_path/"supervisor").mkdir()
    config=dict(command=[sys.executable,"-c","import sys; print(len(sys.stdin.read()))"],cwd=str(tmp_path))
    delivery=Delivery(tmp_path,config)
    rows=[dict(id="healthy",kind="heartbeat",payload={},at="now")]
    (tmp_path/"AGENT_DELIVERY_ARMED").write_text("explicit launch handoff")
    delivery.tick(rows);assert delivery.process is None
    rows.append(dict(id="done",kind="run_complete",payload={},at="now"))
    delivery.tick(rows);assert delivery.process is not None
    delivery.process.wait(timeout=10)
    delivery.tick(rows);assert delivery.process is None
    delivery.tick(rows)
    assert len(delivery.state["attempts"])==1 and delivery.state["delivered"]==["done"]
    assert "work_remains" not in ALLOWED_EVENTS and "agent_completed" not in ALLOWED_EVENTS

def test_final_reader_packaging_from_real_completed_expansion(tmp_path):
    from ghostscale.validation.soundingline.v17.continuation_report import choose_examples,reader_bundle
    store=Store(tmp_path/"records.sqlite")
    for family in cc.e.REGIMES:
        case=cc.e.make_case("v17-final-export-fixture",0,0,family)
        rows=cc.e.evaluate_case(case)
        store.save("expansion",dict(id="e_"+family),0,[dict(case=case,rows=rows)],cpu=0,wall=0)
    selection=choose_examples(store)
    proof=reader_bundle(tmp_path/"export",selection,source_identity())
    assert proof["passed"] and proof["requests"]>=12 and proof["private_read_denied"]
    store.close()


def test_full_literal_phase_sequence_and_export(tmp_path):
    plan=fixture_plan()
    plan["initial_expansions"].append(dict(id="e",family="E",regime="B",namespace="v17-full-phase-export-fixture",constructors=1,histories_per_constructor=1,claim_status="discarded_development"))
    plan["confirmation"]=[dict(id="known_b",family="B",regime="correct",namespace="v17-full-phase-confirm-fixture",constructors=2,histories_per_constructor=1,method="inferred_recipient",rival="retrieval",target="recipient_outcome",metric="brier",low=-2.,high=2.,minimum_gain=.02,planned_radius=.04,direction="discarded fixture only")]
    (tmp_path/"plan.json").write_text(json.dumps(plan))
    result=command(tmp_path,tmp_path/"complete")
    assert result.returncode==0,result.stderr
    root=tmp_path/"complete"
    completion=json.loads((root/"RUN_COMPLETE.json").read_bytes())
    assert completion["execution_complete"] and completion["reader"]["passed"]
    assert (root/"BRANCH_SELECTION.json").exists() and (root/"CONFIRMATION.json").exists()
    from runners.watch_v17_continuation import events
    assert {x["kind"] for x in events(root)}=={"expansion_selection_frozen","confirmation_complete","run_complete"}
    original=(root/"RUN_COMPLETE.json").read_bytes()
    again=command(tmp_path,root)
    assert again.returncode==0,again.stderr
    assert (root/"RUN_COMPLETE.json").read_bytes()==original


def test_existing_live_owner_refuses_duplicate_worker(tmp_path):
    from ghostscale.validation.soundingline.v16.runtime import local_owner
    root=tmp_path/"owned"
    with local_owner(root/"worker-owner"):
        result=command(tmp_path,root,1)
    assert result.returncode!=0
    assert not (root/"WINDOW.json").exists()


def test_missing_stitch_keeps_native_representations_running(tmp_path,monkeypatch):
    monkeypatch.setenv("GS_V17_STITCH_EXE",str(tmp_path/"missing.exe"))
    monkeypatch.setenv("GS_V17_STITCH_CACHE",str(tmp_path/"cache"))
    case=cc.a.make_case("v17-stitch-missing-control",0,0,"graphic:new_combinations")
    rows=cc.evaluate_craft(case,dict(memory_caps=[32],budgets=[2048]))
    assert verify_item(dict(case=case,rows=rows))
    native=[r for r in rows if not r.get("apparatus_failure")]
    skipped=[r for r in rows if r.get("apparatus_failure")]
    assert len(native)==3 and len(skipped)==2
    assert any(r["task_success"] for r in native)
    assert all(r["missing_output"] and not r["task_success"] for r in skipped)


def test_delivery_restart_preserves_active_review_and_missing_cli(tmp_path):
    from ghostscale.validation.soundingline.v16.records import write
    (tmp_path/"supervisor").mkdir()
    (tmp_path/"AGENT_DELIVERY_ARMED").write_text("explicit launch handoff")
    config=dict(command=[str(tmp_path/"absent.exe")],cwd=str(tmp_path))
    active=dict(delivered=[],failed_attempts={},terminal_failures=[],attempts=[
        dict(event_ids=["done"],pid=os.getpid(),started_epoch=0)])
    write(tmp_path/"supervisor/DELIVERY.json",active)
    delivery=Delivery(tmp_path,config)
    rows=[dict(id="done",kind="run_complete",payload={},at="now")]
    delivery.tick(rows)
    assert delivery.process is None and len(delivery.state["attempts"])==1
    delivery.state["attempts"]=[];delivery.save()
    delivery.tick(rows);delivery.tick(rows)
    assert delivery.process is None and delivery.state["terminal_failures"]==["done"]
    assert len(delivery.state["attempts"])==1
