"""Finite source-bound V31 cell queue. A cell lands only after complete replay/reconstruction."""
import argparse,ctypes,importlib,math,os,shutil,subprocess,time,traceback
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v31.common import read,write,digest,run_cell
from ghostscale.validation.soundingline.v27.contract import load_limit
from runners.run_v27 import env_for
from runners.run_k3_redesign import environment,host_sample
from runners.run_v29 import compare
from runners.watch_v23_events import owner,event

def now():return datetime.now(timezone.utc)
def resource_contract(c):
    """Apply only an explicit, conserved pool supplement; original pins stay intact."""
    original=read(c/"ACCEPTANCE.json")
    path=c/"RESOURCE_REALLOCATION.json"
    if not path.exists():return original
    supplement=read(path)
    if supplement.get("approved") is not True or not supplement.get("owner_authorization"):
        raise ValueError("resource supplement lacks owner authorization")
    if supplement.get("acceptance_sha256")!=digest(c/"ACCEPTANCE.json"):
        raise ValueError("resource supplement acceptance mismatch")
    limits=supplement["cpu_limits"]
    if set(limits)!=set(original["cpu_limits"]):raise ValueError("pool roster changed")
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in limits.values()):
        raise ValueError("invalid pool limit")
    if sum(limits.values())!=sum(original["cpu_limits"].values()):raise ValueError("pool total changed")
    if supplement.get("total_cpu_seconds")!=original["total_cpu_seconds"]:raise ValueError("total ceiling changed")
    later=c/"RESOURCE_REALLOCATION_20261118.json"
    if later.exists():
        next_supplement=read(later)
        if next_supplement.get("approved") is not True or not next_supplement.get("owner_authorization"):
            raise ValueError("second resource supplement lacks owner authorization")
        if next_supplement.get("parent_supplement_sha256")!=digest(path) or next_supplement.get("acceptance_sha256")!=digest(c/"ACCEPTANCE.json"):
            raise ValueError("second resource supplement binding mismatch")
        proposed=next_supplement["cpu_limits"]
        expected=dict(limits,D=limits["D"]-7200,reserve=limits["reserve"]+7200)
        if proposed!=expected or next_supplement.get("total_cpu_seconds")!=original["total_cpu_seconds"]:
            raise ValueError("second resource supplement exceeds approved transfer")
        limits=proposed
    # Nothing else in the supplement can override clocks, gear or scientific gates.
    return dict(original,cpu_limits=limits)
def accounting(c):
    a=read(c/"ACCEPTANCE.json");totals={k:0. for k in a["cpu_limits"]}
    for path in (c/"accounting").glob("*.json"):
        row=read(path);totals[row["pool"]]+=max(row["cpu_seconds"],row.get("reservation",0.))
    return totals

class QueueInventory:
    """Cache immutable plans/receipts; discover newly admitted jobs once a minute.

    Dispatch still reopens and verifies the complete source/admission binding.
    Newly finished jobs are inspected every cycle, before any dispatch decision.
    """
    def __init__(self):
        self.plans={};self.completed={};self.next_discovery=0.
    def snapshot(self,c,monotonic):
        if monotonic>=self.next_discovery:
            for path in (c/"jobs").glob("*/PLAN.json"):
                if path.parent.name not in self.plans:self.plans[path.parent.name]=(path,read(path))
            self.next_discovery=monotonic+60.
        for name,(path,plan) in self.plans.items():
            receipt=path.parent/"COMPLETE.json"
            if name not in self.completed and receipt.exists():self.completed[name]=read(receipt)["state"]
        return sorted(self.plans.values(),key=lambda item:item[1]["order"]),dict(self.completed)

class AccountingInventory:
    """Reuse unchanged charges, but observe mutable service/reservation updates."""
    def __init__(self):self.rows={}
    def totals(self,c):
        result={k:0. for k in read(c/"ACCEPTANCE.json")["cpu_limits"]};seen=set()
        with os.scandir(c/"accounting") as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):continue
                seen.add(entry.name);stat=entry.stat();stamp=(stat.st_mtime_ns,stat.st_size)
                old=self.rows.get(entry.name)
                if old is None or old[0]!=stamp:self.rows[entry.name]=(stamp,read(Path(entry.path)))
                row=self.rows[entry.name][1]
                result[row["pool"]]+=max(row["cpu_seconds"],row.get("reservation",0.))
        self.rows={name:row for name,row in self.rows.items() if name in seen}
        return result
def charge(c,key,pool,cpu,state,reservation=0.):
    write(c/"accounting"/(key+".json"),dict(pool=pool,cpu_seconds=cpu,state=state,reservation=reservation),immutable=False)
def admissible(a,totals,plan,current):
    return (totals[plan["pool"]]+plan["cpu_cap_seconds"]<=a["cpu_limits"][plan["pool"]]
        and sum(totals.values())+plan["cpu_cap_seconds"]<=a["total_cpu_seconds"]
        and current+timedelta(seconds=plan["wall_cap_seconds"])<datetime.fromisoformat(a["drain"])
        and current<datetime.fromisoformat(a["new_starts_close"]))

def output_volume_root(job):
    output=job/"output"
    if output.is_symlink():return output.resolve(strict=True)
    return output.resolve() if output.is_dir() else job.resolve()

def storage_admissible(c,job,plan,active):
    """Reserve worst-case new output before starting; never delete evidence.

    The recovery's raw bound includes original and replay, pilot and study.
    Already-running jobs retain their entire bound until completion, so this
    deliberately double-counts any bytes they have already written.
    """
    required=plan.get("storage_required_bytes",0)
    floor=plan.get("storage_floor_bytes",5*1024**3)
    if required<0 or floor<0:raise ValueError("invalid storage reservation")
    try:
        root=output_volume_root(job);volume=root.anchor.lower()
        held=sum(item["plan"].get("storage_required_bytes",0) for item in active.values()
                 if output_volume_root(item["job"]).anchor.lower()==volume)
        return (shutil.disk_usage(c).free>=5*1024**3
                and shutil.disk_usage(root).free>=required+held+floor)
    except OSError:
        # An unavailable archive volume holds new starts; it does not kill peers.
        return False
def verify(c,plan):
    if digest(c/"ACCEPTANCE.json")!=plan["acceptance_sha256"]:raise ValueError("acceptance changed")
    if environment()!=read(c/"ENVIRONMENT.json"):raise ValueError("environment changed")
    source=Path(plan["source_root"])
    if digest(source/"SOURCE_MANIFEST.json")!=plan["source_manifest_sha256"]:raise ValueError("source manifest changed")
    for path,sha in read(source/"SOURCE_MANIFEST.json").items():
        if digest(source/path)!=sha:raise ValueError("source changed: "+path)
    gate=c/"families"/plan["family"]/"ADMISSION.json"
    if digest(gate)!=plan["admission_sha256"] or not read(gate)["passed"]:raise ValueError("family not admitted")
    if plan.get("recovery_parent"):
        verify_recovery_parent(c,plan)

def verify_recovery_parent(c,plan,completed_job=None):
    parent=c/"jobs"/plan["recovery_parent"]
    if digest(parent/"PLAN.json")!=plan["recovery_parent_plan_sha256"]:raise ValueError("recovery parent changed")
    original=read(parent/"PLAN.json")
    for key in ("setting","module","source_root","source_manifest_sha256","acceptance_sha256","cpu_cap_seconds","wall_cap_seconds","interpreter"):
        if plan[key]!=original[key]:raise ValueError("recovery changes scientific contract: "+key)
    expected=plan.get("recovery_parent_completion_sha256")
    if (parent/"COMPLETE.json").exists():
        if digest(parent/"COMPLETE.json")!=expected or read(parent/"COMPLETE.json")["state"]!="FAILED":
            raise ValueError("recovery parent is not the retained failed attempt")
    elif expected is not None:raise ValueError("recovery failure receipt missing")
    for relative,sha in plan.get("retained_parent_files",{}).items():
        if digest(parent/relative)!=sha:raise ValueError("retained recovery evidence changed")
        if completed_job is not None and digest(completed_job/relative)!=sha:
            raise ArithmeticError("recovery did not reproduce retained parent evidence")
def equivalent(original,other,replay=False):
    if replay:
        if {p.name:digest(p) for p in original.iterdir()}!={p.name:digest(p) for p in other.iterdir()}:
            raise ArithmeticError("full seed replay differs")
    else:
        compare(read(original/"SUMMARY.json"),read(other/"SUMMARY.json"))
        with np.load(original/"cell_measures.npz",allow_pickle=False) as a,np.load(other/"cell_measures.npz",allow_pickle=False) as b:
            if set(a.files)!=set(b.files):raise ArithmeticError("measure roster differs")
            for key in a.files:
                if not np.allclose(a[key],b[key],rtol=1e-10,atol=1e-8,equal_nan=True):
                    raise ArithmeticError("independent metric differs: "+key)
def worker(c,job):
    runtime.below_normal();runtime.child_handshake(job,"worker")
    plan=read(job/"PLAN.json");verify(c,plan);a=read(c/"ACCEPTANCE.json");last=[0.]
    def pulse(**progress):
        if time.process_time()>plan["cpu_cap_seconds"] or now()>=datetime.fromisoformat(a["drain"]) or (c/"STOP").exists():
            raise TimeoutError("cell CPU/stop/drain boundary")
        if time.monotonic()-last[0]>5:
            write(job/"PROGRESS.json",dict(at=now().isoformat(),pid=os.getpid(),cpu_seconds=time.process_time(),**progress),immutable=False)
            last[0]=time.monotonic()
    try:
        module=importlib.import_module(plan["module"]);setting=plan["setting"]
        pilot=dict(setting,worlds=40,blocks=8)
        root=job/"output";root.mkdir()
        if plan.get("kind")=="validation":
            result=module.validate(setting,root,pulse)
            write(root/"VALIDATION.json",result)
            if not result["passed"]:raise ArithmeticError("excluded validation failed; retain all evidence")
            write(job/"WORKER_COMPLETE.json",dict(state="VERIFIED",scientific=False,
                plan_sha256=digest(job/"PLAN.json"),validation_sha256=digest(root/"VALIDATION.json"),
                cpu_seconds=time.process_time()))
            return
        controls=module.controls(root/"controls",pulse)
        if not controls["passed"]:raise ArithmeticError("known-answer failure")
        reason=None
        exact=bool(getattr(module,"EXACT_MAP",False))
        if exact:write(root/"PILOT.json",dict(worlds=0,excluded=False,reason="exact finite enumeration; no sampled pilot",replay=True,independent=True))
        phases=(("study",setting),) if exact else (("pilot",pilot),("study",setting))
        for lane,s in phases:
            pulse(phase=lane)
            original=root/(lane+"-original")
            run_cell(module,s,original,lane=lane,pulse=pulse)
            replay=root/(lane+"-replay");run_cell(module,s,replay,lane=lane,pulse=pulse)
            equivalent(original,replay,True)
            for name,sha in read(original/"RAW_MANIFEST.json").items():
                if digest(original/name)!=sha:raise ArithmeticError("retained data changed")
            reference=root/(lane+"-reconstruct")
            run_cell(module,s,reference,mode="reconstruct",original=original,lane=lane,pulse=pulse)
            equivalent(original,reference)
            if hasattr(module,"validate_measures"):
                with np.load(original/"cell_measures.npz",allow_pickle=False) as z:
                    module.validate_measures(s,{k:z[k] for k in z.files})
            if lane=="pilot":
                with np.load(original/"cell_measures.npz",allow_pickle=False) as z:
                    measures={k:z[k] for k in z.files}
                reason=module.pilot_exclusion(setting,measures) if hasattr(module,"pilot_exclusion") else None
                write(root/"PILOT.json",dict(worlds=40,excluded=bool(reason),reason=reason,
                    replay=True,independent=True))
                if reason:break
        result=dict(state="EXCLUDED" if reason else "VERIFIED",setting=setting,
            plan_sha256=digest(job/"PLAN.json"),pilot_sha256=digest(root/"PILOT.json"),
            summary_sha256=digest(root/("pilot-original" if reason else "study-original")/"SUMMARY.json"),
            cpu_seconds=time.process_time(),no_registered_claim=True)
        write(job/"WORKER_COMPLETE.json",result)
    except BaseException as exc:
        write(job/"FAILURE.json",dict(error=str(exc),traceback=traceback.format_exc(),cpu_seconds=time.process_time()))
        raise

def start(c,job):
    plan=read(job/"PLAN.json");verify(c,plan)
    write(job/"STARTED.json",dict(at=now().isoformat(),plan_sha256=digest(job/"PLAN.json")))
    charge(c,job.name,plan["pool"],0.,"RUNNING",plan["cpu_cap_seconds"])
    log=(job/"worker.log").open("xb");env=env_for(plan["source_root"]);env["NUMBA_NUM_THREADS"]="1"
    env["NUMBA_CACHE_DIR"]=str(c/"numba-cache")
    child=subprocess.Popen([plan["interpreter"],"-B","-m","runners.run_v31","--campaign",str(c),"--worker",str(job)],
        cwd=plan["source_root"],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    try:handle=runtime.attach_child(child,job,"worker")
    except BaseException:
        if child.poll() is None:child.terminate()
        child.wait(timeout=10);log.close();charge(c,job.name,plan["pool"],plan["cpu_cap_seconds"],"FAILED")
        write(job/"COMPLETE.json",dict(state="FAILED",error="native handshake failed; cap charged"))
        raise
    return dict(child=child,handle=handle,log=log,plan=plan,job=job,opened=time.monotonic())
def finish(c,item,error=None):
    child=item["child"];job=item["job"];plan=item["plan"]
    if error and child.poll() is None:
        ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(item["handle"]),1)
        ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(item["handle"]),10000);child.wait(timeout=10)
    used=runtime.child_cpu(item["handle"]);receipt=None
    if error is None:
        try:
            receipt=read(job/"WORKER_COMPLETE.json")
            if child.returncode or receipt["plan_sha256"]!=digest(job/"PLAN.json"):raise ValueError("exit/binding mismatch")
            if plan.get("recovery_parent"):verify_recovery_parent(c,plan,job)
        except (OSError,ValueError,KeyError,ArithmeticError) as exc:error=str(exc)
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(item["handle"]));item["log"].close()
    state="FAILED" if error else receipt["state"]
    charge(c,job.name,plan["pool"],used,state)
    write(job/"COMPLETE.json",dict(state=state,error=error,cpu_seconds=used,
        plan_sha256=digest(job/"PLAN.json"),worker_receipt_sha256=digest(job/"WORKER_COMPLETE.json") if receipt else None))
    event(c,job.name+"-"+state.lower(),"batch_failed" if error else "batch_complete",
          dict(cell=job.name,family=plan["family"],state=state))
    return state

def run(c):
    runtime.below_normal();a=resource_contract(c);active={};blocked={};previous=None;samples=[];limit=4
    inventory=QueueInventory();charges=AccountingInventory()
    with owner(c/"controller"):
        if (c/"QUEUE_COMPLETE.json").exists():raise ValueError("finite queue already closed")
        for job in (c/"jobs").iterdir():
            if (job/"STARTED.json").exists() and not (job/"COMPLETE.json").exists():raise ValueError("interrupted cell requires explicit recovery")
        try:
            while True:
                current=now();charge(c,"controller-"+str(os.getpid()),"reserve",time.process_time(),"RUNNING")
                load,previous=host_sample(previous)
                if load is not None:samples.append((time.monotonic(),load));samples=samples[-180:];limit=load_limit(samples,limit)
                for name,item in list(active.items()):
                    error=None
                    if runtime.child_cpu(item["handle"])>item["plan"]["cpu_cap_seconds"]:error="cell CPU cap"
                    if time.monotonic()-item["opened"]>item["plan"]["wall_cap_seconds"]:error="cell wall cap"
                    if current>=datetime.fromisoformat(a["drain"]) or (c/"STOP").exists():error="drain or explicit stop"
                    if runtime.memory()["available"]<2*1024**3:error="host memory protection"
                    if error or item["child"].poll() is not None:
                        state=finish(c,item,error);del active[name]
                        if state=="FAILED":blocked[item["plan"]["family"]]="retained cell failure"
                for name,item in list(active.items()):
                    if item["plan"]["family"] in blocked:
                        finish(c,item,"family stopped after retained failure");del active[name]
                jobs,completed=inventory.snapshot(c,time.monotonic())
                totals=charges.totals(c);waiting=[];counts={};storage_held=[]
                for name,state in completed.items():
                    counts[state]=counts.get(state,0)+1
                    if state=="FAILED":blocked[inventory.plans[name][1]["family"]]="retained cell failure"
                for path,plan in jobs:
                    job=path.parent;family=plan["family"]
                    if job.name in completed:continue
                    if (job/"STARTED.json").exists():continue
                    if family in blocked:continue
                    if not admissible(a,totals,plan,current):
                        waiting.append(job.name);continue
                    if len(active)>=limit:waiting.append(job.name);continue
                    if not storage_admissible(c,job,plan,active):
                        waiting.append(job.name);storage_held.append(job.name);continue
                    active[job.name]=start(c,job);totals[plan["pool"]]+=plan["cpu_cap_seconds"]
                write(c/"STATUS.json",dict(at=current.isoformat(),pid=os.getpid(),gear=2,worker_limit=limit,
                    active=list(active),waiting=waiting,counts=counts,blocked=blocked,
                    accounting=totals,storage_held=storage_held,build_complete=(c/"BUILD_COMPLETE.json").exists()),immutable=False)
                if storage_held and not active:
                    event(c,f"v31-storage-held-{sum(counts.values())}","storage_needed",
                          dict(unstarted=len(storage_held),scientific_evidence_preserved=True))
                terminal=(c/"BUILD_COMPLETE.json").exists() and not active and not waiting
                exhausted=current>=datetime.fromisoformat(a["new_starts_close"]) or (c/"STOP").exists()
                # Resource-held pending cells become terminal once no active reservation can release capacity.
                resource_end=(c/"BUILD_COMPLETE.json").exists() and not active and waiting and all(
                    not admissible(a,totals,inventory.plans[name][1],current) for name in waiting)
                if not active and (terminal or exhausted or resource_end):
                    write(c/"QUEUE_COMPLETE.json",dict(at=current.isoformat(),counts=counts,blocked=blocked,
                         unrun=waiting,reason="finite/resource/time endpoint"))
                    event(c,"v31-finite-end","queue_depleted",dict(counts=counts,blocked=blocked,unrun=waiting));break
                # A blocked queue needs a heartbeat, not a full directory scan every five seconds.
                time.sleep(5 if active else 60)
        finally:
            for item in active.values():finish(c,item,"controller exit")
            charge(c,"controller-"+str(os.getpid()),"reserve",time.process_time(),"STOPPED")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--campaign",type=Path,required=True);parser.add_argument("--worker",type=Path)
    args=parser.parse_args()
    worker(args.campaign.resolve(),args.worker.resolve()) if args.worker else run(args.campaign.resolve())
