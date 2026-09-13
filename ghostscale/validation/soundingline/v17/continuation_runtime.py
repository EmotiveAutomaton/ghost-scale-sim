"""Native V17 phase progression and sustained independent robustness sampling."""
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
import base64
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from ..v16.records import canonical,digest,file_digest,read,write
from ..v16.runtime import local_owner
from .queue_runtime import heartbeat
from .continuation_cases import make_case,evaluate_case
from .continuation_store import Store,utc
from .continuation_plan import rank_branches
from .continuation_analysis import confirmation,holm
REPO=Path(__file__).resolve().parents[4]
SOURCES=[
    *["ghostscale/validation/soundingline/v17/"+p.name for p in sorted(Path(__file__).parent.glob("*.py"))],
    *["ghostscale/validation/soundingline/v16/"+n for n in ("__init__.py","records.py","runtime.py","campaign_ownership.py","graphic_world.py","assembly.py")],
    "ghostscale/__init__.py","ghostscale/validation/__init__.py","ghostscale/validation/soundingline/__init__.py",
    "runners/run_v17_continuation.py","runners/watch_v17_continuation.py","runners/package_v17_observers.py",
    "runners/verify_v17_screen.py","runners/replay_v17_screen.py",
    *["tests/"+p.name for p in sorted((REPO/"tests").glob("test_v17*.py"))],
    "docs/versions/v17-adaptive-appreciation/CODING_PACKAGE.md",
]
def source_identity(): return {p:file_digest(REPO/p) for p in SOURCES}

def verify_admission(plan,admission):
    if plan["workers"]!=1 or plan["gpu_used"] is not False: raise ValueError("unauthorized allocation")
    actual=source_identity()
    if plan["source_files"]!=actual or not admission.get("passed") or admission["source_files"]!=actual:
        raise ValueError("continuation source/admission mismatch")
    unavailable=[]
    for name,expected in plan.get("external_files",{}).items():
        path=Path(os.environ["GS_V17_STITCH_EXE"])
        if name=="stitch_runtime": path=path.with_name("libunwind.dll")
        if not path.is_file() or file_digest(path)!=expected: unavailable.append(name)
    if unavailable:
        # Pause the unavailable optional learner while the native representations
        # continue and retain explicit apparatus-failure rows. Never execute an
        # unverified replacement binary.
        os.environ["GS_V17_STITCH_EXE"]=str(REPO/"__unavailable_stitch__"/"compress.exe")

def capture_cache(cache):
    files={}
    for directory in sorted(cache.iterdir()) if cache.exists() else []:
        if not directory.is_dir(): continue
        for p in sorted(directory.rglob("*")):
            if p.is_file(): files[str(p.relative_to(cache)).replace("\\","/")]=base64.b64encode(p.read_bytes()).decode()
    return files

def clear_committed_cache(cache,run_root):
    resolved=cache.resolve();parent=(run_root/"unit-cache").resolve()
    if resolved.parent!=parent: raise ValueError("cache cleanup outside owned unit cache")
    if resolved.exists(): shutil.rmtree(resolved)

def prepared_controller(root,store,plan):
    """Freeze once on separate development data before fresh draws or long sampling."""
    from .adaptive import prepare
    path=root/"CONTROLLER.json"
    if path.exists(): result=read(path)
    else:
        result=prepare(dict(namespace="v17-continuation-controller-development"))
        write(path,result)
    if plan.get("controller_sha256") and file_digest(path)!=plan["controller_sha256"]: raise ValueError("frozen controller changed")
    store.lock("controller_sha256",file_digest(path))
    return result

def one_unit(root,store,stage,design,ci,controller,emit,contrast=None):
    if store.has(stage,design["id"],ci): return
    if store.failed(stage,design["id"]): return
    cached=root/"unit-cache"/(stage+"-"+design["id"]+"-"+str(ci))
    cached.mkdir(parents=True,exist_ok=True)
    os.environ["GS_V17_STITCH_CACHE"]=str(cached)
    start,cpu=time.monotonic(),time.process_time()
    emit(stage=stage,branch=design["id"],constructor_index=ci,**store.counts())
    options=dict(design)
    if options["family"]=="C": options["prepared"]=controller
    try:
        items=[]
        for hi in range(design["histories_per_constructor"]):
            case=make_case(options,ci,hi)
            items.append(dict(case=case,rows=evaluate_case(case,options)))
        dependency_files=capture_cache(cached)
        store.save(stage,design,ci,items,cpu=time.process_time()-cpu,wall=time.monotonic()-start,
            dependency_files=dependency_files,contrast=contrast)
        for reason in sorted({r["apparatus_failure"] for i in items for r in i["rows"] if r.get("apparatus_failure")}):
            store.event("dependency_failure",dict(component="Stitch",reason=reason,native_arms_continue=True))
        clear_committed_cache(cached,root)
    except Exception as exc:
        reason=type(exc).__name__+": "+str(exc)
        event=store.failure(stage,design["id"],ci,reason)
        store.event("scientific_failure",dict(failure_id=event,stage=stage,branch=design["id"],constructor_index=ci,reason=reason))
        emit(last_failure=reason)
        # The failed unit and any dependency evidence remain; other branches proceed.

def run(root,plan,admission,*,fixture_units=None):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    verify_admission(plan,admission)
    with local_owner(root/"worker-owner"),heartbeat(root,scope="V17 sustained scientific queue") as emit:
        store=Store(root/"records.sqlite")
        try:
            store.lock("plan",plan);store.lock("admission",admission)
            window=store.metadata("window")
            if window is None:
                start=datetime.now(timezone.utc)
                window=dict(started_at=start.isoformat(),ends_at=(start+timedelta(hours=plan["run_hours"])).isoformat(),
                    run_hours=plan["run_hours"],original_implementation_start=plan["original_implementation_start"],
                    original_soft_target=plan["original_soft_target"],timing_amendment=plan["timing_amendment"])
                store.lock("window",window);write(root/"WINDOW.json",window)
            deadline=datetime.fromisoformat(window["ends_at"])
            if (root/"RUN_COMPLETE.json").exists():
                emit(state="completed",**store.counts());return 0
            controller=prepared_controller(root,store,plan)
            initial_count=store.counts()["constructor_units"]
            def stop_fixture():
                return fixture_units is not None and store.counts()["constructor_units"]-initial_count>=fixture_units
            def can_continue():
                if datetime.now(timezone.utc)>=deadline:return False
                if store.path.stat().st_size>plan["max_database_bytes"] or shutil.disk_usage(root).free<plan["min_free_disk_bytes"]:
                    store.event("resource_checkpoint",dict(reason="predefined disk allocation reached",**store.counts()))
                    raise RuntimeError("predefined disk allocation reached; retained data preserved")
                return True
            jobs=plan["initial_expansions"]
            # Balanced initial allocation; a slow family cannot monopolize setup progression.
            for ci in range(max(j["constructors"] for j in jobs)):
                for design in jobs:
                    if ci<design["constructors"] and can_continue():
                        one_unit(root,store,"expansion",design,ci,controller,emit)
                        if stop_fixture(): emit(state="fixture_checkpoint",**store.counts());return 0
            selection=store.get_snapshot("BRANCH_SELECTION")
            if selection is None:
                surfaces=[store.surface("expansion",j) for j in jobs]
                selection=rank_branches(surfaces)
                store.snapshot("INITIAL_EXPANSIONS",surfaces);store.snapshot("BRANCH_SELECTION",selection)
                write(root/"BRANCH_SELECTION.json",selection)
                store.event("expansion_selection_frozen",dict(selection_sha256=digest(selection),**store.counts()))
            # The primaries, directions, sample sizes and controllers were frozen
            # before the first new fresh draw. Failures do not receive replacements.
            contrasts=plan["confirmation"]
            for contrast in contrasts:
                if not can_continue(): break
                if source_identity()!=plan["source_files"]:raise ValueError("frozen source changed at phase boundary")
                design=dict(contrast,claim_status="frozen_confirmation")
                for ci in range(contrast["constructors"]):
                    if not can_continue(): break
                    if store.failed("confirmation",design["id"]):break
                    one_unit(root,store,"confirmation",design,ci,controller,emit,contrast)
                    if stop_fixture(): emit(state="fixture_checkpoint",**store.counts());return 0
            confirmed=store.get_snapshot("CONFIRMATION")
            if confirmed is None:
                entries=[];incomplete=[]
                for contrast in contrasts:
                    values=store.pairs(contrast["id"])
                    if len(values)==contrast["constructors"]: entries.append(confirmation(values,contrast))
                    else: incomplete.append(dict(id=contrast["id"],completed=len(values),planned=contrast["constructors"]))
                # Incomplete primaries retain p=1 in the same Holm family.
                for row in incomplete: entries.append(dict(id=row["id"],p_value=1.,claim_status="incomplete",incomplete=row))
                family=holm(entries)
                for row in family:
                    if "incomplete" in row: row["claim_status"]="incomplete; no confirmatory claim"
                confirmed=dict(schema="v17.confirmation-family.1",entries=family,incomplete=incomplete,
                    frozen_selection_sha256=digest(contrasts),controller_sha256=file_digest(root/"CONTROLLER.json"))
                store.snapshot("CONFIRMATION",confirmed);write(root/"CONFIRMATION.json",confirmed)
                store.event("confirmation_complete",dict(confirmation_sha256=digest(confirmed),**store.counts()))
            # This is scientific job selection, not a recursive agent rule.
            # Fresh confirmation never receives more observations based on its result.
            promoted={x["id"] for x in selection["selected"]}
            chosen=[j for j in jobs if j["id"] in promoted]
            if not chosen: chosen=list(jobs)
            schedule=chosen*3+jobs
            position=store.metadata("robustness_origin") or dict(start_index=0)
            store.lock("robustness_origin",position)
            total=0
            for ci in range(plan["robustness_max_constructor_index"]):
                if not can_continue(): break
                if all(store.failed("robustness",j["id"]) for j in schedule):raise RuntimeError("all robustness branches failed; agent intervention required")
                if ci%64==0 and source_identity()!=plan["source_files"]:raise ValueError("frozen source changed during run")
                for original in schedule:
                    if not can_continue():break
                    design=dict(original,namespace=original["namespace"]+"-robustness",constructors=plan["robustness_max_constructor_index"])
                    # Repeated selected entries get independent index lanes, not aliases.
                    offset=total%len(schedule)
                    actual_ci=ci*len(schedule)+offset
                    total+=1
                    one_unit(root,store,"robustness",design,actual_ci,controller,emit)
                    count=store.counts()["constructor_units"]
                    if count>=32768 and store.get_snapshot("ROBUSTNESS_32768") is None:
                        checkpoint=dict(**store.counts(),window=window)
                        store.snapshot("ROBUSTNESS_32768",checkpoint)
                        store.event("robustness_32768_constructor_units",checkpoint)
                    if stop_fixture(): emit(state="fixture_checkpoint",**store.counts());return 0
            from .continuation_report import finish
            result=finish(root,store,plan,window)
            write(root/"RUN_COMPLETE.json",result)
            store.event("run_complete",dict(completion_sha256=digest(result),**store.counts()))
            emit(state="completed",**store.counts())
            return 0
        finally: store.close()
