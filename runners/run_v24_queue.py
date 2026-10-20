"""Finite source-bound V24 pilot queue; no retries, new studies or production."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v24 import contract
from runners.watch_v23_events import event


def validate_queue(campaign,manifest,current):
    if current.tzinfo is None:
        raise ValueError("aware clock required")
    if not datetime.fromisoformat(contract.OPEN)<=current<datetime.fromisoformat(contract.CLOSE):
        raise ValueError("outside the V24 pilot window")
    if read(campaign/"ACCEPTANCE.json")!=contract.acceptance():
        raise ValueError("campaign acceptance changed")
    if digest(campaign/"ACCEPTANCE.json")!=manifest["acceptance_sha256"]:
        raise ValueError("queue acceptance pin differs")
    names=[item["packet"] for item in manifest["packets"]]
    if names!=["v24-n16-pilot-1","v24-d5-pilot-1-r1"]:
        raise ValueError("only the two prepared independent pilots are commissioned here")
    for item in manifest["packets"]:
        packet=(campaign/"packets"/item["packet"]).resolve()
        if not packet.is_relative_to((campaign/"packets").resolve()):
            raise ValueError("packet outside campaign")
        if digest(packet/"PLAN.json")!=item["plan_sha256"]:
            raise ValueError("queued plan changed")
        plan=read(packet/"PLAN.json")
        if plan["kind"]!="pilot" or plan["family"]!=item["family"]:
            raise ValueError("queue cannot create scientific admissions")
    return True


def completion(packet):
    complete=read(packet/"COMPLETE.json")
    if (complete["state"]!="PILOT_VERIFIED" or complete["plan_sha256"]!=digest(packet/"PLAN.json")
            or complete["replay_sha256"]!=digest(packet/"REPLAY.json")
            or complete["review_sha256"]!=digest(packet/"INDEPENDENT_REVIEW.json")
            or complete["summary_sha256"]!=digest(packet/"original/SUMMARY.json")):
        raise ValueError("complete pilot receipts do not bind")


def run(campaign,queue):
    if os.name!="nt":
        raise RuntimeError("native Windows queue required")
    manifest=read(queue/"MANIFEST.json")
    pin=read(queue/"PIN.json")
    if digest(queue/"MANIFEST.json")!=pin["manifest_sha256"]:
        raise ValueError("queue manifest changed")
    source=Path(__file__).resolve().parents[1]
    if source!=(queue/"source").resolve():
        raise ValueError("immutable queue source required")
    actual={p.relative_to(source).as_posix() for p in source.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"}
    if actual!=set(pin["source_files"]):
        raise ValueError("queue source roster changed")
    for name,expected in pin["source_files"].items():
        if digest(source/name)!=expected:
            raise ValueError("queue source changed")
    validate_queue(campaign,manifest,datetime.now(timezone.utc))
    if (queue/"STATUS.json").exists():
        raise ValueError("queue already attempted; retain state and reconcile rather than retry")
    started=time.process_time();state="FAILED";active=None;error=None;finished=[]
    with (queue/"OWNER.lock").open("x",encoding="utf-8") as f:f.write(str(os.getpid()))
    try:
        for item in manifest["packets"]:
            validate_queue(campaign,manifest,datetime.now(timezone.utc))
            packet=campaign/"packets"/item["packet"];plan=read(packet/"PLAN.json")
            if (campaign/"STOP").exists():
                raise ValueError("campaign stopped")
            if (packet/"COMPLETE.json").exists():
                completion(packet);finished.append(packet.name);continue
            if (campaign/"accounting"/(packet.name+".json")).exists():
                raise ValueError("reserved or failed packet requires an operator disposition")
            env=dict(os.environ,PYTHONPATH=str(packet/"source"),PYTHONNOUSERSITE="1")
            for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
                env[name]="1"
            with (queue/(packet.name+".log")).open("xb") as log:
                active=subprocess.Popen([plan["interpreter"],"-B","-m","runners.run_v24_pilot",
                    "--packet",str(packet)],cwd=packet/"source",env=env,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                while active.poll() is None:
                    write(queue/"STATUS.json",dict(state="RUNNING",pid=os.getpid(),packet=packet.name,
                        launcher_pid=active.pid,heartbeat=datetime.now(timezone.utc).isoformat(),
                        finished=finished),immutable=False)
                    # Packet parents enforce native child caps and the immutable
                    # end. The queue never kills a different owner's process.
                    time.sleep(5)
                if active.returncode:
                    raise RuntimeError(f"{packet.name} exited {active.returncode}")
                active=None
            completion(packet);finished.append(packet.name)
        state="COMPLETE"
        event(campaign,"v24-prepared-queue-drained","review_boundary",dict(packets=finished))
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        event(campaign,"v24-prepared-queue-failed","queue_failure",dict(error=error,finished=finished))
        raise
    finally:
        write(campaign/"accounting/prepared-queue.json",dict(kind="controller",state=state,
              cpu_seconds=time.process_time()-started),immutable=False)
        write(queue/"STATUS.json",dict(state=state,pid=os.getpid(),finished=finished,error=error,
              active_launcher_pid=active.pid if active and active.poll() is None else None),immutable=False)
        (queue/"OWNER.lock").unlink()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--queue",type=Path,required=True)
    args=parser.parse_args()
    run(args.campaign.resolve(),args.queue.resolve())
