"""Finite source-bound V30 cell queue. A cell lands only after complete replay/reconstruction."""
import argparse,ctypes,importlib,os,subprocess,time,traceback
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v30.common import read,write,digest,run_cell
from ghostscale.validation.soundingline.v27.contract import load_limit
from runners.run_v27 import env_for
from runners.run_k3_redesign import environment,host_sample
from runners.run_v29 import compare
from runners.watch_v23_events import owner,event

def now():return datetime.now(timezone.utc)
def accounting(c):
    a=read(c/"ACCEPTANCE.json");totals={k:0. for k in a["cpu_limits"]}
    for path in (c/"accounting").glob("*.json"):
        row=read(path);totals[row["pool"]]+=max(row["cpu_seconds"],row.get("reservation",0.))
    return totals
def charge(c,key,pool,cpu,state,reservation=0.):
    write(c/"accounting"/(key+".json"),dict(pool=pool,cpu_seconds=cpu,state=state,reservation=reservation),immutable=False)
def admissible(a,totals,plan,current):
    return (totals[plan["pool"]]+plan["cpu_cap_seconds"]<=a["cpu_limits"][plan["pool"]]
        and sum(totals.values())+plan["cpu_cap_seconds"]<=a["total_cpu_seconds"]
        and current+timedelta(seconds=plan["wall_cap_seconds"])<datetime.fromisoformat(a["drain"])
        and current<datetime.fromisoformat(a["new_starts_close"]))
def verify(c,plan):
    if digest(c/"ACCEPTANCE.json")!=plan["acceptance_sha256"]:raise ValueError("acceptance changed")
    if environment()!=read(c/"ENVIRONMENT.json"):raise ValueError("environment changed")
    source=Path(plan["source_root"])
    if digest(source/"SOURCE_MANIFEST.json")!=plan["source_manifest_sha256"]:raise ValueError("source manifest changed")
    for path,sha in read(source/"SOURCE_MANIFEST.json").items():
        if digest(source/path)!=sha:raise ValueError("source changed: "+path)
    gate=c/"families"/plan["family"]/"ADMISSION.json"
    if digest(gate)!=plan["admission_sha256"] or not read(gate)["passed"]:raise ValueError("family not admitted")
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
    child=subprocess.Popen([plan["interpreter"],"-B","-m","runners.run_v30","--campaign",str(c),"--worker",str(job)],
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
        except (OSError,ValueError,KeyError) as exc:error=str(exc)
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(item["handle"]));item["log"].close()
    state="FAILED" if error else receipt["state"]
    charge(c,job.name,plan["pool"],used,state)
    write(job/"COMPLETE.json",dict(state=state,error=error,cpu_seconds=used,
        plan_sha256=digest(job/"PLAN.json"),worker_receipt_sha256=digest(job/"WORKER_COMPLETE.json") if receipt else None))
    event(c,job.name+"-"+state.lower(),"batch_failed" if error else "batch_complete",
          dict(cell=job.name,family=plan["family"],state=state))
    return state

def run(c):
    runtime.below_normal();a=read(c/"ACCEPTANCE.json");active={};blocked={};previous=None;samples=[];limit=4
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
                jobs=sorted((c/"jobs").glob("*/PLAN.json"),key=lambda x:read(x)["order"])
                totals=accounting(c);waiting=[];counts={}
                for path in jobs:
                    job=path.parent;plan=read(path);family=plan["family"]
                    if (job/"COMPLETE.json").exists():
                        state=read(job/"COMPLETE.json")["state"];counts[state]=counts.get(state,0)+1
                        if state=="FAILED":blocked[family]="retained cell failure"
                        continue
                    if (job/"STARTED.json").exists():continue
                    if family in blocked:continue
                    if not admissible(a,totals,plan,current):
                        waiting.append(job.name);continue
                    if len(active)>=limit:waiting.append(job.name);continue
                    active[job.name]=start(c,job);totals[plan["pool"]]+=plan["cpu_cap_seconds"]
                write(c/"STATUS.json",dict(at=current.isoformat(),pid=os.getpid(),gear=2,worker_limit=limit,
                    active=list(active),waiting=waiting,counts=counts,blocked=blocked,
                    accounting=accounting(c),build_complete=(c/"BUILD_COMPLETE.json").exists()),immutable=False)
                terminal=(c/"BUILD_COMPLETE.json").exists() and not active and not waiting
                exhausted=current>=datetime.fromisoformat(a["new_starts_close"]) or (c/"STOP").exists()
                # Resource-held pending cells become terminal once no active reservation can release capacity.
                resource_end=(c/"BUILD_COMPLETE.json").exists() and not active and waiting and all(
                    not admissible(a,accounting(c),read(c/"jobs"/name/"PLAN.json"),current) for name in waiting)
                if not active and (terminal or exhausted or resource_end):
                    write(c/"QUEUE_COMPLETE.json",dict(at=current.isoformat(),counts=counts,blocked=blocked,
                         unrun=waiting,reason="finite/resource/time endpoint"))
                    event(c,"v30-finite-end","queue_depleted",dict(counts=counts,blocked=blocked,unrun=waiting));break
                time.sleep(5)
        finally:
            for item in active.values():finish(c,item,"controller exit")
            charge(c,"controller-"+str(os.getpid()),"reserve",time.process_time(),"STOPPED")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--campaign",type=Path,required=True);parser.add_argument("--worker",type=Path)
    args=parser.parse_args()
    worker(args.campaign.resolve(),args.worker.resolve()) if args.worker else run(args.campaign.resolve())
