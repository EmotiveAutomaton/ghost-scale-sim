"""Non-LLM supervisor. Only science state transitions may request an agent."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v16.records import canonical,digest,read,write
from ghostscale.validation.soundingline.v17.queue_runtime import heartbeat
ALLOWED_EVENTS={"expansion_selection_frozen","confirmation_complete","robustness_32768_constructor_units",
                "run_complete","scientific_failure","worker_disappeared","supervisor_failure","resource_checkpoint","dependency_failure"}

def process_alive(pid):
    if os.name == "nt":
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x100000,False,pid)
        if not handle:return False
        try:return kernel.WaitForSingleObject(ctypes.c_void_p(handle),0)==258
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

def events(root):
    path=root/"records.sqlite"
    if not path.exists():return []
    try:
        db=sqlite3.connect("file:"+path.as_posix()+"?mode=ro",uri=True,timeout=5)
        try:return [dict(id=i,kind=k,payload=json.loads(p),at=t) for i,k,p,t in db.execute("SELECT id,kind,payload,at FROM events ORDER BY at,id")]
        finally:db.close()
    except sqlite3.OperationalError:return []

class Delivery:
    def __init__(self,root,config):
        self.root=Path(root);self.config=config;self.process=None;self.batch=None;self.streams=[]
        self.path=self.root/"supervisor/DELIVERY.json"
        self.state=read(self.path) if self.path.exists() else dict(delivered=[],failed_attempts={},attempts=[],terminal_failures=[])
    def save(self):write(self.path,self.state,immutable=False)
    def pending(self,rows):
        done=set(self.state["delivered"])|set(self.state["terminal_failures"])
        return [r for r in rows if r["kind"] in ALLOWED_EVENTS and r["id"] not in done]
    def tick(self,rows):
        # A supervisor restart must not launch a second copy of an agent whose
        # transition review is already running. An uncertain finished delivery
        # is retained for manual inspection rather than automatically repeated.
        if self.process is None and self.state["attempts"]:
            previous=self.state["attempts"][-1]
            if "exit_code" not in previous:
                if previous.get("pid") and process_alive(previous["pid"]):return
                previous.update(exit_code="unknown after supervisor restart",finished_at=datetime.now(timezone.utc).isoformat())
                self.state["terminal_failures"].extend(previous["event_ids"])
                self.save()
        if self.process is not None:
            code=self.process.poll()
            if code is None:return
            ids=[r["id"] for r in self.batch]
            entry=self.state["attempts"][-1];entry.update(exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
            if code==0:self.state["delivered"].extend(ids)
            else:
                for identity in ids:
                    attempts=self.state["failed_attempts"].get(identity,0)+1
                    self.state["failed_attempts"][identity]=attempts
                    if attempts>=3:self.state["terminal_failures"].append(identity)
            for stream in self.streams:stream.close()
            self.streams=[];self.process=None;self.batch=None;self.save()
        if not self.config or not (self.root/"AGENT_DELIVERY_ARMED").exists():return
        pending=self.pending(rows)
        if not pending:return
        last=self.state["attempts"][-1] if self.state["attempts"] else {}
        if last.get("exit_code",0)!=0 and time.time()-last["started_epoch"]<60:return
        directory=self.root/"supervisor";directory.mkdir(exist_ok=True)
        number=len(self.state["attempts"])+1
        prompt=("A non-LLM Ghost Scale supervisor detected the following predefined V17 scientific state transitions. "
          "This is the user's authorized continuation of the same task. Read AGENTS.md and the exact event receipts. "
          "Perform the bounded review, result filing, or fault repair made ready by these events. "
          "Do not poll a healthy external simulation through model turns. Do not restart valid work, reset clocks, "
          "edit the frozen worker checkout, create a Stop-hook continuation, or continue merely because work remains. "
          "If science is healthy after your event-specific actions, record its PID and expected outputs and end the turn. "
          "A completed run requires final evidence review, archival, scientific write-through and the already-authorized ordinary push. "
          "Do not affect Sounding Line. Run root: "+str(self.root)+". Events: "+json.dumps(pending,sort_keys=True))
        command=self.config["command"]
        if not isinstance(command,list) or not all(isinstance(x,str) for x in command):raise ValueError("invalid event command")
        executable=Path(command[0])
        try:
            if self.config.get("executable_sha256") and hashlib.sha256(executable.read_bytes()).hexdigest()!=self.config["executable_sha256"]:
                raise ValueError("event agent executable changed")
        except (OSError,ValueError) as exc:
            self.state["attempts"].append(dict(event_ids=[r["id"] for r in pending],exit_code="delivery apparatus unavailable",
                reason=str(exc),started_epoch=time.time(),started_at=datetime.now(timezone.utc).isoformat()))
            self.state["terminal_failures"].extend(r["id"] for r in pending);self.save();return
        out=(directory/("agent-"+str(number)+".jsonl")).open("wb")
        err=(directory/("agent-"+str(number)+".stderr.log")).open("wb")
        self.streams=[out,err]
        try:
            self.process=subprocess.Popen(command,cwd=self.config["cwd"],stdin=subprocess.PIPE,stdout=out,stderr=err,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        except OSError as exc:
            out.close();err.close();self.streams=[]
            self.state["attempts"].append(dict(event_ids=[r["id"] for r in pending],exit_code="delivery launch failed",
                reason=str(exc),started_epoch=time.time(),started_at=datetime.now(timezone.utc).isoformat()))
            self.state["terminal_failures"].extend(r["id"] for r in pending);self.save();return
        self.batch=pending
        self.state["attempts"].append(dict(event_ids=[r["id"] for r in pending],pid=self.process.pid,
            started_at=datetime.now(timezone.utc).isoformat(),started_epoch=time.time()))
        self.save()
        try:self.process.stdin.write(prompt.encode("utf-8"));self.process.stdin.close()
        except (BrokenPipeError,OSError):pass  # The exit is recorded by the next native tick.

def watch(root,plan_path,admission_path,event_config):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    own=root/"supervisor";own.mkdir(exist_ok=True)
    config=read(event_config) if event_config else None
    if config and config.get("stitch_executable"):os.environ["GS_V17_STITCH_EXE"]=config["stitch_executable"]
    delivery=Delivery(root,config)
    plan=read(plan_path)
    with local_owner(own),heartbeat(own,scope="native transition supervisor") as emit:
        attempts=read(own/"RECOVERY.json") if (own/"RECOVERY.json").exists() else dict(launches=0,last_progress=-1,no_progress=0)
        extra_events=read(own/"EVENTS.json") if (own/"EVENTS.json").exists() else []
        child=None;streams=[]
        def extra(kind,payload):
            entry=dict(kind=kind,payload=payload,at=datetime.now(timezone.utc).isoformat(),id=digest([kind,payload]))
            if entry["id"] not in {r["id"] for r in extra_events}:extra_events.append(entry);write(own/"EVENTS.json",extra_events,immutable=False)
        while True:
            observed=events(root)+extra_events
            delivery.tick(observed)
            terminal=(root/"RUN_COMPLETE.json").exists() or (root/"FATAL.json").exists()
            if child is not None and child.poll() is not None:
                code=child.returncode
                for stream in streams:stream.close()
                streams=[];child=None
                if not terminal:
                    extra("worker_disappeared",dict(exit_code=code,launch=attempts["launches"]))
                    current=read(root/"STATUS.json").get("constructor_units",0) if (root/"STATUS.json").exists() else 0
                    attempts["no_progress"]=attempts["no_progress"]+1 if current<=attempts["last_progress"] else 0
                    attempts["last_progress"]=current
                    write(own/"RECOVERY.json",attempts,immutable=False)
                    if attempts["no_progress"]>=2 or attempts["launches"]>=8:
                        write(root/"FATAL.json",dict(reason="bounded recovery exhausted",attempts=attempts))
                        extra("supervisor_failure",dict(reason="bounded recovery exhausted",attempts=attempts))
                        terminal=True
                emit(worker_exit=code,worker_pid=None)
            if terminal:
                observed=events(root)+extra_events
                delivery.tick(observed)
                emit(state="terminal",completion_present=(root/"RUN_COMPLETE.json").exists(),pending_event_delivery=len(delivery.pending(observed)))
                # Native delivery may finish while no scientific work is active;
                # it does not create a new event from the agent's own completion.
                if not config or (delivery.process is None and not delivery.pending(observed)):return 0 if (root/"RUN_COMPLETE.json").exists() else 1
                if not (root/"AGENT_DELIVERY_ARMED").exists():return 0
                time.sleep(10);continue
            if child is None:
                try:
                    with local_owner(root/"worker-owner"):pass
                except RuntimeError:
                    emit(state="existing_worker_preserved")
                    time.sleep(10);continue
                attempts["launches"]+=1;write(own/"RECOVERY.json",attempts,immutable=False)
                log=own/("worker-"+str(attempts["launches"]))
                streams=[log.with_suffix(".stdout.log").open("wb"),log.with_suffix(".stderr.log").open("wb")]
                command=[sys.executable,"-B","-m","runners.run_v17_continuation","--root",str(root),
                         "--plan",str(Path(plan_path).resolve()),"--admission",str(Path(admission_path).resolve())]
                env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
                child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],stdout=streams[0],stderr=streams[1],env=env,
                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                emit(state="supervising",worker_pid=child.pid,worker_started_at=datetime.now(timezone.utc).isoformat(),
                    attempt=attempts["launches"],window=(read(root/"WINDOW.json") if (root/"WINDOW.json").exists() else "assigned once by first worker"))
            time.sleep(10)

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True,type=Path);p.add_argument("--plan",required=True,type=Path)
    p.add_argument("--admission",required=True,type=Path);p.add_argument("--event-config",type=Path)
    a=p.parse_args();raise SystemExit(watch(a.root,a.plan,a.admission,a.event_config))
if __name__=="__main__":main()
