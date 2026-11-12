"""V28 finite reviews at expected end or an at-most-four-hour build boundary."""
import argparse,os,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from ghostscale.validation.soundingline.v28.common import read,write
from runners.watch_v23_events import owner,event,pending,deliver,closed,require_native_review_host,ReviewOwnerBusyError,deferred_events,alive


def run(c,state,config):
    require_native_review_host();cfg=read(config)
    with owner(state):
        try:
            while not closed(c) and not (state/'STOP').exists():
                current=datetime.now(timezone.utc)
                if not (c/'QUEUE_COMPLETE.json').exists():
                    if (c/'WAKE_PLAN.json').exists():
                        plan=read(c/'WAKE_PLAN.json')
                        if current>=datetime.fromisoformat(plan['due']):event(c,plan['id'],'planned_review',dict(due=plan['due'],reason=plan['reason']))
                    if (c/'QUEUE_STATUS.json').exists():
                        queue=read(c/'QUEUE_STATUS.json')
                        if not alive(queue['pid']):event(c,'controller-disappeared','service_failed',dict(pid=queue['pid']))
                events=pending(c,state);deferred=deferred_events(c,state,events,current)
                ready=[e for e in events if e['id'] not in deferred]
                write(state/'STATUS.json',dict(state='monitoring',pid=os.getpid(),heartbeat=current.isoformat(),
                    armed=(state/'ARMED').exists(),pending_events=len(ready),interactive_chat_delivery=False),immutable=False)
                write(c/'accounting'/('monitor-'+str(os.getpid())+'.json'),dict(family='G1',state='RUNNING',cpu_seconds=time.process_time()),immutable=False)
                if ready and (state/'ARMED').exists():
                    try:deliver(c,state,cfg,ready)
                    except ReviewOwnerBusyError:pass
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(state/'STATUS.json',dict(state='stopped',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)
            write(c/'accounting'/('monitor-'+str(os.getpid())+'.json'),dict(family='G1',state='STOPPED',cpu_seconds=time.process_time()),immutable=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('campaign','state','config'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
