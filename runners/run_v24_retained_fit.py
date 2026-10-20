"""Bounded native retained-input engineering, never scientific admission."""
import argparse
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from ghostscale.validation.soundingline.v23.common import read, write, digest
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v24 import fit_fixture
from runners.run_v24_pilot import accounting

REPO = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def verify(packet):
    plan = read(packet/"PLAN.json")
    if REPO.resolve() != (packet/"source").resolve():
        raise ValueError("immutable engineering source required")
    if digest(packet/"PLAN.json") != read(packet/"PREPARATION.json")["plan_sha256"]:
        raise ValueError("engineering plan changed")
    if digest(packet/"SOURCE.zip") != plan["source_archive_sha256"]:
        raise ValueError("source archive changed")
    for name, expected in plan["source_files"].items():
        if digest(packet/"source"/name) != expected:
            raise ValueError("engineering source changed")
    actual = {p.relative_to(packet/"source").as_posix() for p in (packet/"source").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual != set(plan["source_files"]):
        raise ValueError("source roster changed")
    if runtime.environment() != plan["environment"]:
        raise ValueError("resident environment changed")
    if (plan["kind"] != "retained-engineering" or plan["cpu_cap_seconds"] > 3000
            or plan["new_scientific_worlds"] != 0 or plan["numerical_threads"] != 1):
        raise ValueError("bounded retained-input preparation only")
    if digest(packet.parent.parent/"ACCEPTANCE.json") != plan["acceptance_sha256"]:
        raise ValueError("V24 acceptance changed")
    fit_fixture.verify_inputs(plan)
    return plan


def child(packet,label):
    runtime.child_handshake(packet,label)
    runtime.below_normal()
    plan=verify(packet);start=time.process_time();last=[0.]
    def pulse(**values):
        elapsed=time.process_time()-start
        if time.monotonic()-last[0]>5:
            write(packet/(label+"-progress.json"),dict(pid=os.getpid(),at=now(),
                  cpu_seconds=elapsed,**values),immutable=False);last[0]=time.monotonic()
        if elapsed>plan["pass_cpu_seconds"][label]:
            raise TimeoutError("retained engineering pass CPU cap")
    if label=="review":
        write(packet/"INDEPENDENT_REVIEW.json",fit_fixture.reconstruct(packet/"original",plan,pulse))
    else:
        fit_fixture.execute(packet/label,plan,pulse)
    pulse(phase="complete")


def execute(packet):
    if os.name!="nt":
        raise RuntimeError("native Windows accounting required")
    packet=packet.resolve();campaign=packet.parent.parent;plan=verify(packet)
    attempt=campaign/"accounting"/(packet.name+".json")
    if attempt.exists() or (packet/"COMPLETE.json").exists():
        raise ValueError("retained attempt cannot be restarted")
    acceptance=read(campaign/"ACCEPTANCE.json")
    lock=campaign/"OWNER.lock"
    with lock.open("x",encoding="utf-8") as f:f.write(str(os.getpid()))
    started=time.process_time();cpu=0.;active=None;handle=None
    wall=time.monotonic();state="FAILED";error=None
    try:
        charged,_=accounting(campaign,None)
        prep=0.
        for path in (campaign/"accounting").glob("*.json"):
            row=read(path)
            if row.get("kind")=="preparation":
                prep+=max(row.get("cpu_seconds",0),row.get("reserved_cpu_seconds",0))
        if (charged+plan["cpu_cap_seconds"]>acceptance["total_cpu_seconds"]-acceptance["verification_reserve_seconds"]
                or prep+plan["cpu_cap_seconds"]>acceptance["preparation_seconds"]):
            raise ValueError("preparation or protected budget exceeded")
        write(attempt,dict(kind="preparation",state="RESERVED",reserved_cpu_seconds=plan["cpu_cap_seconds"]))
        for label in ("original","replay","review"):
            env=dict(os.environ,PYTHONPATH=str(packet/"source"),PYTHONNOUSERSITE="1")
            for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
                env[name]="1"
            with (packet/(label+".log")).open("xb") as log:
                active=subprocess.Popen([plan["interpreter"],"-m","runners.run_v24_retained_fit",
                    "--packet",str(packet),"--child",label],cwd=packet/"source",env=env,
                    stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                handle=runtime.attach_child(active,packet,label)
                while active.poll() is None:
                    current=runtime.child_cpu(handle)
                    write(packet/"STATUS.json",dict(state="RUNNING",phase=label,parent_pid=os.getpid(),
                        child_pid=read(packet/(label+"-owner.json"))["pid"],heartbeat=now(),
                        cpu_seconds=cpu+current+time.process_time()-started),immutable=False)
                    if (cpu+current+time.process_time()-started>plan["cpu_cap_seconds"]
                            or current>plan["pass_cpu_seconds"][label]
                            or time.monotonic()-wall>plan["wall_cap_seconds"]
                            or runtime.memory()["available"]<2*1024**3 or (campaign/"STOP").exists()):
                        raise TimeoutError("retained engineering resource boundary")
                    time.sleep(2)
                cpu+=runtime.child_cpu(handle)
                ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle));handle=None
                if active.returncode:
                    raise RuntimeError(f"{label} failed with exit {active.returncode}")
                active=None
            if label=="replay":
                original={p.name:digest(p) for p in (packet/"original").iterdir() if p.is_file()}
                replay={p.name:digest(p) for p in (packet/"replay").iterdir() if p.is_file()}
                if original!=replay:
                    raise ValueError("complete retained fixture replay mismatch")
                write(packet/"REPLAY.json",dict(state="PASSED",files=original))
        write(packet/"COMPLETE.json",dict(state="RETAINED_ENGINEERING_VERIFIED",new_scientific_worlds=0,
            plan_sha256=digest(packet/"PLAN.json"),replay_sha256=digest(packet/"REPLAY.json"),
            review_sha256=digest(packet/"INDEPENDENT_REVIEW.json"),scientific_admission=False))
        state="COMPLETE"
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if handle is not None:
            if active is not None and active.poll() is None:
                ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(handle),1)
                ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(handle),10000)
            cpu+=runtime.child_cpu(handle)
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))
        if active is not None and active.poll() is None:
            active.terminate();active.wait(timeout=10)
        total=cpu+time.process_time()-started
        write(attempt,dict(kind="preparation",state=state,cpu_seconds=total,
              reserved_cpu_seconds=0 if state=="COMPLETE" else plan["cpu_cap_seconds"],error=error),immutable=False)
        write(packet/"STATUS.json",dict(state=state,at=now(),cpu_seconds=total,error=error),immutable=False)
        lock.unlink()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--packet",type=Path,required=True)
    parser.add_argument("--child",choices=("original","replay","review"))
    args=parser.parse_args()
    child(args.packet.resolve(),args.child) if args.child else execute(args.packet.resolve())
