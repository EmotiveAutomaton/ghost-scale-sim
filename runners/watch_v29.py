"""One finite V29 monitor for primary and successor; local review reports only."""
import argparse,os,time
from datetime import datetime,timezone
from pathlib import Path
from ghostscale.validation.soundingline.v29.common import read,write,digest
from runners.watch_v23_events import owner,event,pending,deliver,closed,require_native_review_host,ReviewOwnerBusyError,deferred_events,alive

def observe(primary,c,state,current):
    for item in pending(primary,primary/'notifications'):
        event(c,'primary-'+item['id'],item['kind'],dict(primary_event_id=item['id'],primary_event_sha256=item['event_sha256'],evidence=item['evidence']))
    for label,root in [('primary',primary),('continuation',c)]:
        if (root/'QUEUE_COMPLETE.json').exists():continue
        path=root/'QUEUE_STATUS.json'
        if path.exists():
            status=read(path)
            if not alive(status['pid']):event(c,label+'-controller-disappeared','service_failed',dict(pid=status['pid']))
        elif label=='continuation' and (root/'NATIVE_OWNER.json').exists():
            pid=read(root/'NATIVE_OWNER.json')['pid']
            if not alive(pid):event(c,'continuation-waiter-disappeared','service_failed',dict(pid=pid))
    if (c/'WAKE_PLAN.json').exists() and not (c/'QUEUE_COMPLETE.json').exists():
        plan=read(c/'WAKE_PLAN.json')
        if current>=datetime.fromisoformat(plan['due']):event(c,plan['id'],'planned_review',dict(due=plan['due'],reason=plan['reason']))

def review_ready(events):
    # Routine completions accumulate until the one planned check; failures,
    # owner loss and a drained queue need action immediately.
    return any(e['kind']!='batch_complete' for e in events)


def run(primary,c,state,config):
    require_native_review_host();cfg=read(config)
    with owner(state):
        try:
            while not closed(c) and not (state/'STOP').exists():
                current=datetime.now(timezone.utc);observe(primary,c,state,current)
                events=pending(c,state);deferred=deferred_events(c,state,events,current);ready=[x for x in events if x['id'] not in deferred]
                write(state/'STATUS.json',dict(state='monitoring',pid=os.getpid(),heartbeat=current.isoformat(),armed=(state/'ARMED').exists(),pending_events=len(ready),interactive_chat_delivery=False),immutable=False)
                write(c/'accounting'/('monitor-'+str(os.getpid())+'.json'),dict(family='reserve',state='RUNNING',cpu_seconds=time.process_time()),immutable=False)
                if review_ready(ready) and (state/'ARMED').exists():
                    try:deliver(c,state,cfg,ready)
                    except ReviewOwnerBusyError:pass
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(state/'STATUS.json',dict(state='stopped',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)
            write(c/'accounting'/('monitor-'+str(os.getpid())+'.json'),dict(family='reserve',state='STOPPED',cpu_seconds=time.process_time()),immutable=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('primary','campaign','state','config'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.primary.resolve(),a.campaign.resolve(),a.state.resolve(),a.config.resolve())
