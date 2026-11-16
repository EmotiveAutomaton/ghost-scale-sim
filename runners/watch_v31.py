"""One finite V31 expected-boundary monitor; native reviews produce local reports."""
import argparse,os,time
from datetime import datetime,timezone
from pathlib import Path
from ghostscale.validation.soundingline.v31.common import read,write
from runners.watch_v23_events import (owner,event,pending,deliver,closed,
    require_native_review_host,ReviewOwnerBusyError,deferred_events,alive)

def observe(c,state,current):
    path=c/"STATUS.json"
    if path.exists() and not (c/"QUEUE_COMPLETE.json").exists():
        status=read(path)
        if not alive(status["pid"]):
            event(c,"v31-controller-disappeared","service_failed",dict(pid=status["pid"],last_heartbeat=status["at"]))
        elif (current-datetime.fromisoformat(status["at"])).total_seconds()>600:
            event(c,"v31-controller-heartbeat-stale","service_failed",dict(pid=status["pid"],last_heartbeat=status["at"]))
        if not status["active"] and not status["waiting"] and not status.get("build_complete"):
            # Fixed identity per completed-receipt count; cheap observations do not
            # manufacture repeated model work from an unchanged empty queue.
            total=sum(status["counts"].values())
            event(c,f"v31-build-boundary-{total}","build_needed",
                dict(completed_receipts=total,blocked=status["blocked"]))
    if (c/"WAKE_PLAN.json").exists() and not (c/"CLOSEOUT.json").exists():
        plan=read(c/"WAKE_PLAN.json")
        if current>=datetime.fromisoformat(plan["due"]):
            event(c,plan["id"],"planned_review",dict(due=plan["due"],reason=plan["reason"]))

def review_ready(events):
    return any(e["kind"]!="batch_complete" for e in events)

def deliver_current(c,state,cfg):
    """Re-read acknowledgements after arming; never deliver an earlier snapshot."""
    if not (state/"ARMED").exists():return False
    current=datetime.now(timezone.utc)
    events=pending(c,state)
    deferred=deferred_events(c,state,events,current)
    ready=[e for e in events if e["id"] not in deferred]
    if not review_ready(ready) or not (state/"ARMED").exists():return False
    deliver(c,state,cfg,ready)
    return True

def charge(c,state):
    write(c/"accounting"/("monitor-"+str(os.getpid())+".json"),
        dict(pool="reserve",state=state,cpu_seconds=time.process_time(),reservation=0.),immutable=False)

def run(c,state,config):
    require_native_review_host();cfg=read(config)
    with owner(state):
        try:
            while not closed(c) and not (state/"STOP").exists():
                current=datetime.now(timezone.utc);observe(c,state,current)
                events=pending(c,state);deferred=deferred_events(c,state,events,current)
                ready=[x for x in events if x["id"] not in deferred]
                write(state/"STATUS.json",dict(state="monitoring",pid=os.getpid(),heartbeat=current.isoformat(),
                    armed=(state/"ARMED").exists(),pending_events=len(ready),
                    interactive_chat_delivery=False),immutable=False)
                charge(c,"RUNNING")
                if review_ready(ready) and (state/"ARMED").exists():
                    try:deliver_current(c,state,cfg)
                    except ReviewOwnerBusyError:pass
                time.sleep(30)
        finally:
            (state/"ARMED").unlink(missing_ok=True)
            write(state/"STATUS.json",dict(state="stopped",pid=os.getpid(),
                at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)
            charge(c,"STOPPED")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("campaign","state","config"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
