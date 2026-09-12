"""Finite one-worker V17 queue with native ownership, immutable inputs and resume.

No old campaign stage is called. A failure blocks its dependent jobs only.
The scheduler does not generate scientific jobs, refill work or pad occupancy.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from ..v16.records import canonical, digest, file_digest, now, read, write
from ..v16.runtime import local_owner
from .packets import write_bytes_once, load_cases
REPO = Path(__file__).resolve().parents[4]
FAMILIES = {"B": "recipient", "C": "adaptive", "D": "revision", "A2": "craft_extension", "E": "observer"}
EXTRA_SOURCES = ["queue_runtime.py", "recipient.py", "adaptive.py", "revision.py", "craft_extension.py", "stitch_adapter.py", "observer.py"]
SOURCE_PATHS = [
    *["ghostscale/validation/soundingline/v17/"+s for s in EXTRA_SOURCES],
    *["ghostscale/validation/soundingline/v17/"+s for s in ("__init__.py","contracts.py","programs.py","packets.py","craft.py")],
    *["ghostscale/validation/soundingline/v16/"+s for s in ("__init__.py","records.py","runtime.py","campaign_ownership.py","graphic_world.py","assembly.py")],
    "runners/run_v17_queue.py", "runners/package_v17_observers.py", "tests/test_v17_queue.py", "tests/test_v17_studies.py",
    "docs/versions/v17-adaptive-appreciation/CODING_PACKAGE.md",
    "ghostscale/__init__.py", "ghostscale/validation/__init__.py", "ghostscale/validation/soundingline/__init__.py",
]


def source_identity():
    return {name:file_digest(REPO/name) for name in SOURCE_PATHS}


@contextmanager
def heartbeat(root, **initial):
    state = dict(schema="v17.live.2", pid=os.getpid(), started_at=now(), state="running", **initial)
    guard, stop = threading.RLock(), threading.Event()
    errors = []
    def emit(**updates):
        with guard:
            if errors:
                raise RuntimeError("heartbeat writer failed") from errors[0]
            state.update(updates)
            state["heartbeat"] = now()
            write(root/"STATUS.json", state, immutable=False)
    def pulse():
        while not stop.wait(30):
            try:
                emit()
            except BaseException as exc:
                errors.append(exc)
                return
    emit()
    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield emit
    except BaseException as exc:
        emit(state="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        stop.set()
        thread.join()


def validate_design(design):
    if design["family"] not in FAMILIES:
        raise ValueError("family is not implemented")
    if any(type(design[k]) is not int or design[k] < 1 for k in ("constructors","histories_per_constructor","cases_per_chunk")):
        raise ValueError("invalid finite allocation")
    if design["namespace"].endswith("-fresh-1"):
        raise ValueError("fresh evaluation requires a separately frozen confirmation consumer")
    if design["claim_status"] not in ("discarded_development","descriptive"):
        raise ValueError("screen engine cannot assert confirmation")


def cases(design):
    module = importlib.import_module("."+FAMILIES[design["family"]], __package__)
    for c in range(design["constructors"]):
        for h in range(design["histories_per_constructor"]):
            for regime in design["regimes"]:
                yield module.make_case(design["namespace"], c, h, regime)


def summarize(items, design):
    from collections import defaultdict
    import random
    groups, identities = defaultdict(list), defaultdict(set)
    for item in items:
        case = item["case"]
        for name in ("case_id","constructor_id","history_id","public_problem_sha256","private_construction_sha256","structural_family"):
            identities[name].add(case[name])
        for row in item["rows"]:
            groups[(case["regime"], row["method"], row.get("target","construction"))].append((case,row))
    table = []
    for (regime, method, target), entries in sorted(groups.items()):
        clusters = defaultdict(list)
        for case,row in entries:
            clusters[case["constructor_id"]].append(row.get("brier_score", float(not row["task_success"])))
        means = [sum(xs)/len(xs) for _,xs in sorted(clusters.items())]
        rng = random.Random(int(digest([design["namespace"],regime,method,target])[:16],16))
        boot = sorted(sum(rng.choices(means,k=len(means)))/len(means) for _ in range(2000))
        rows = [row for _,row in entries]
        infinite = sum(r.get("log_loss_infinite",False) for r in rows)
        losses = [r["log_loss_nats"] for r in rows if r.get("log_loss_nats") is not None]
        table.append(dict(family=design["family"],regime=regime,method=method,target=target,
            attempted=len(rows),valid=sum(not r["missing_output"] and not r["invalid_program"] for r in rows),
            unique_public_problems=len({c["public_problem_sha256"] for c,_ in entries}),
            unique_private_constructions=len({c["private_construction_sha256"] for c,_ in entries}),
            constructor_clusters=len(clusters),mean_brier_or_failure=sum(means)/len(means),
            descriptive_95_interval=[boot[49],boot[1949]],interval_method="constructor percentile bootstrap, 2000 draws",
            accuracy_or_success=sum(r["task_success"] for r in rows)/len(rows),
            log_loss_infinite_count=infinite,mean_log_loss_nats=None if infinite or not losses else sum(losses)/len(losses),
            mean_cold_cost=sum(r["costs"]["cold_total"] for r in rows)/len(rows),
            mean_repeat_cost=sum(r["costs"]["repeat_online"] for r in rows)/len(rows),
            evidence_tiers=sorted({r["evidence_tier"] for r in rows}),claim_status=design["claim_status"]))
    return dict(schema="v17.comparisons.2", comparison_table=table,
        uniqueness={k:len(v) for k,v in identities.items()}, campaign_complete=False,
        interpretation="constructed mechanisms; descriptive; miniature - architecture untested; no human inference")


def packet(root, design, *, stop_after_chunks=None, admission=None):
    root = Path(root)
    validate_design(design)
    if design["claim_status"] == "descriptive":
        if admission is None or not admission.get("passed") or admission.get("source_files") != source_identity():
            raise ValueError("scientific packet requires source-bound admission")
    with local_owner(root), heartbeat(root, family=design["family"]) as emit:
        source = source_identity()
        lock_path = root/"LOCK.json"
        if lock_path.exists():
            lock = read(lock_path)
            if lock["source_files"] != source or lock["design"] != design:
                raise ValueError("frozen source/design differs; old evidence retained")
        else:
            write(lock_path, dict(schema="v17.lock.2",design=design,source_files=source,frozen_at=now()))
        index_path = root/"INDEX.json"
        index = read(index_path) if index_path.exists() else dict(schema="v17.index.2",lock_sha256=file_digest(lock_path),chunks=[])
        if index["lock_sha256"] != file_digest(lock_path):
            raise ValueError("index lock mismatch")
        list(load_cases(root,index))
        module = importlib.import_module("."+FAMILIES[design["family"]],__package__)
        begin,cpu = time.monotonic(),time.process_time()
        case_list = list(cases(design))
        added = 0
        try:
            evaluation_design=dict(design)
            if hasattr(module,"prepare"):
                prepared=module.prepare(design)
                write(root/"TRAINING.json",prepared)
                evaluation_design["prepared"]=prepared
            size = design["cases_per_chunk"]
            for chunk_index,start in enumerate(range(0,len(case_list),size)):
                part = case_list[start:start+size]
                relative = f"raw/chunk-{chunk_index:05d}_points.jsonl"
                if chunk_index < len(index["chunks"]):
                    record = index["chunks"][chunk_index]
                    if record["path"] != relative or record["case_ids"] != [c["case_id"] for c in part]:
                        raise ValueError("retained chunk identity mismatch")
                    continue
                evaluated = []
                for case in part:
                    rows = module.evaluate_case(case,evaluation_design)
                    if not rows:
                        raise ValueError("silently empty scientific row set")
                    evaluated.append(dict(case=case,rows=rows))
                payload = b"".join(canonical(item)+b"\n" for item in evaluated)
                path = root/relative
                if path.exists():
                    if path.read_bytes() != payload:
                        raise ValueError("unindexed chunk fails deterministic recovery")
                else:
                    write_bytes_once(path,payload)
                index["chunks"].append(dict(path=relative,sha256=file_digest(path),bytes=len(payload),
                    cases=len(part),rows=sum(len(i["rows"]) for i in evaluated),case_ids=[c["case_id"] for c in part]))
                write(index_path,index,immutable=False)
                added += 1
                emit(completed_cases=start+len(part),planned_cases=len(case_list),completed_chunks=len(index["chunks"]))
                if stop_after_chunks and added >= stop_after_chunks:
                    emit(state="paused_at_chunk_boundary")
                    return None
            summary = summarize(load_cases(root,index),design)
            write(root/"COMPARISONS.json",summary)
            completion = dict(schema="v17.completion.2",state="completed",campaign_complete=False,
                lock_sha256=file_digest(lock_path),index_sha256=file_digest(index_path),
                summary_sha256=file_digest(root/"COMPARISONS.json"),cases=len(case_list),rows=sum(c["rows"] for c in index["chunks"]))
            write(root/"COMPLETION.json",completion)
            emit(state="completed")
            return completion
        except BaseException as exc:
            write(root/f"FAILURE-{time.time_ns()}.json",dict(at=now(),error=type(exc).__name__,message=str(exc)))
            emit(state="failed",error=str(exc))
            raise
        finally:
            write(root/f"TIME-{time.time_ns()}.json",dict(wall_seconds=time.monotonic()-begin,
                process_cpu_seconds=time.process_time()-cpu,new_chunks=added,workers=1,gpu_used=False))


def run_queue(root, manifest, *, python=None):
    root = Path(root)
    if manifest["workers"] != 1 or manifest["gpu_used"] is not False:
        raise ValueError("this launch allocation authorizes one CPU worker")
    for name,expected in manifest.get("external_files",{}).items():
        executable=Path(os.environ["GS_V17_STITCH_EXE"])
        actual=executable if name=="stitch_executable" else executable.with_name("libunwind.dll")
        if file_digest(actual)!=expected: raise ValueError("external executable/runtime identity changed")
    with local_owner(root), heartbeat(root, scope="finite commissioned queue") as emit:
        write(root/"MANIFEST.json",manifest)
        if manifest["source_files"] != source_identity():
            raise ValueError("queue source changed")
        admission = read(root/"ADMISSION.json")
        if not admission.get("passed") or admission.get("source_files") != source_identity():
            raise ValueError("queue requires current source-bound local admission")
        jobs = manifest["jobs"]
        identifiers = [j["id"] for j in jobs]
        if len(set(identifiers)) != len(jobs):
            raise ValueError("duplicate queue identity")
        done,failed = set(),set()
        for job in jobs:
            if not set(job.get("depends_on",[])) <= done:
                failed.add(job["id"])
                write(root/(job["id"]+"-BLOCKED.json"),dict(reason="dependency not completed",dependencies=job.get("depends_on",[])))
                continue
            output = root/"packets"/job["id"]
            design=dict(job["design"])
            if job.get("budget_pilot"):
                from itertools import combinations
                pilot=root/"packets"/job["budget_pilot"]
                completion=read(pilot/"COMPLETION.json")
                for filename,keyname in (("LOCK.json","lock_sha256"),("INDEX.json","index_sha256"),("COMPARISONS.json","summary_sha256")):
                    if file_digest(pilot/filename)!=completion[keyname]: raise ValueError("pilot receipt mismatch")
                grouped={}
                for item in load_cases(pilot,read(pilot/"INDEX.json")):
                    world=item["case"]["public"]["world_kind"]
                    for row in item["rows"]:
                        grouped.setdefault(world,{}).setdefault(row["budget"],[]).append(row["task_success"])
                design["budgets_by_world"]={world:list(min(combinations(sorted(values),3),key=lambda bs:(sum((sum(values[b])/len(values[b])-t)**2 for b,t in zip(bs,(.2,.5,.8))),bs))) for world,values in grouped.items()}
                design["pilot_inputs"]={name:file_digest(pilot/name) for name in ("LOCK.json","INDEX.json","COMPLETION.json")}
            validate_design(design)
            write(root/"designs"/(job["id"]+".json"),design)
            # If an earlier supervisor exited while its child survived, wait for
            # that owned packet lock before dispatching a resumptive verifier.
            while True:
                try:
                    with local_owner(output):
                        pass
                    break
                except RuntimeError:
                    emit(job=job["id"], state="waiting_for_surviving_owned_worker")
                    time.sleep(30)
            command = [python or sys.executable,"-B","-m","runners.run_v17_queue","packet","--root",str(output),
                       "--design",str(root/"designs"/(job["id"]+".json")),"--admission",str(root/"ADMISSION.json")]
            # Every job enters the normal resume path even if a completion file exists.
            # It revalidates all retained hashes and produces identical aggregates.
            env = dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
            if source_identity() != manifest["source_files"]:
                raise ValueError("live source changed before job dispatch")
            attempt = root/"logs"/(job["id"]+"-"+str(time.time_ns()))
            attempt.parent.mkdir(parents=True,exist_ok=True)
            with attempt.with_suffix(".out.log").open("wb") as out, attempt.with_suffix(".err.log").open("wb") as err:
                process = subprocess.Popen(command,cwd=REPO,env=env,stdout=out,stderr=err)
                emit(job=job["id"],worker_pid=process.pid,completed_jobs=len(done),planned_jobs=len(jobs))
                code = process.wait()
            if code == 0 and (output/"COMPLETION.json").exists():
                done.add(job["id"])
            else:
                failed.add(job["id"])
            write(root/"PROGRESS.json",dict(at=now(),completed=sorted(done),failed_or_blocked=sorted(failed),
                planned=identifiers,campaign_complete=False),immutable=False)
        emit(state="failed_jobs_need_attention" if failed else "queue_completed",worker_pid=None,
             completed_jobs=len(done),failed_or_blocked=sorted(failed))
        completion=dict(completed=sorted(done),failed_or_blocked=sorted(failed),
            campaign_complete=False,next_action="analyze completed screens and select informative expansion; no automatic confirmation")
        write(root/"QUEUE_COMPLETION.json",completion)
        return 1 if failed else 0


def watch_queue(root,manifest_path):
    """Cheap bounded recovery. Native locks prevent duplicate queue/worker owners."""
    root=Path(root).resolve()
    manifest_path=Path(manifest_path).resolve()
    manifest=read(manifest_path)
    watch=root/"watchdog"
    with local_owner(watch),heartbeat(watch,scope="bounded owner recovery") as emit:
        attempts=0
        last_progress=-1
        while True:
            busy=False
            try:
                with local_owner(root): pass
            except RuntimeError:
                busy=True
            if busy:
                emit(state="waiting_for_existing_queue_owner")
                time.sleep(30)
                continue
            if (root/"QUEUE_COMPLETION.json").exists():
                completion=read(root/"QUEUE_COMPLETION.json")
                emit(state="terminal",completion=completion)
                return 1 if completion["failed_or_blocked"] else 0
            if attempts>=3:
                emit(state="recovery_limit_reached")
                return 1
            if attempts and datetime.now(timezone.utc)>datetime.fromisoformat(manifest["campaign_target"]):
                emit(state="planning_target_reached_needs_review")
                return 1
            progress=read(root/"PROGRESS.json") if (root/"PROGRESS.json").exists() else {}
            completed=len(progress.get("completed",[]))
            if attempts>=2 and completed<=last_progress:
                emit(state="no_progress_after_recovery")
                return 1
            last_progress=completed
            attempts+=1
            command=[sys.executable,"-B","-m","runners.run_v17_queue","queue",
                     "--root",str(root),"--manifest",str(manifest_path)]
            log=root/"logs"/("supervisor-"+str(time.time_ns()))
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.with_suffix(".out.log").open("wb") as out,log.with_suffix(".err.log").open("wb") as err:
                child=subprocess.Popen(command,cwd=REPO,stdout=out,stderr=err)
                emit(state="supervising",supervisor_pid=child.pid,attempt=attempts)
                code=child.wait()
            emit(supervisor_exit=code,supervisor_pid=None)
            # Terminal scientific failures are never rerun by this watchdog.
            # A terminated supervisor without a completion gets bounded resume.
