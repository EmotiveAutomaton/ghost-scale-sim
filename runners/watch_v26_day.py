"""Finite day monitor: terminal events and fixed twelve-hour checkpoints only."""
import argparse,os,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
from ghostscale.validation.soundingline.v26.common import read,write
from runners.watch_v23_events import owner,event,pending,deliver,closed,require_native_review_host,ReviewOwnerBusyError,deferred_events
from runners.watch_v26 import observe

def scheduled(campaign,current):
    a=read(campaign/'ACCEPTANCE.json');opened=datetime.fromisoformat(a['opened']);drain=datetime.fromisoformat(a['drain']);point=opened+timedelta(hours=12);i=1
    while point<drain:
        if current>=point:event(campaign,f'day-health12-{i:02d}','twelve_hour_health',dict(due=point.isoformat()))
        point+=timedelta(hours=12);i+=1
    for key in ('new_starts_close','drain'):
        if current>=datetime.fromisoformat(a[key]):event(campaign,'day-'+key,'final_deadline',dict(due=a[key]))
    if current>=drain:(campaign/'STOP').touch(exist_ok=True)
    wake=campaign/'WAKE_PLAN.json'
    if wake.exists():
        plan=read(wake)
        if current>=datetime.fromisoformat(plan['due']):event(campaign,plan['id'],'planned_review',dict(due=plan['due'],reason=plan['reason']))

def run(campaign,state,config):
    require_native_review_host();cfg=read(config);started=time.process_time();account=campaign/'accounting'/f'monitor-{os.getpid()}-{time.time_ns()}.json'
    with owner(state):
        try:
            while not closed(campaign) and not (state/'STOP').exists():
                current=datetime.now(timezone.utc);active=observe(campaign);scheduled(campaign,current);events=pending(campaign,state);deferred=deferred_events(campaign,state,events,current);events=[e for e in events if e['id'] not in deferred]
                write(state/'STATUS.json',dict(state='monitoring',pid=os.getpid(),heartbeat=current.isoformat(),armed=(state/'ARMED').exists(),active_packets=active,pending_events=len(events)),immutable=False)
                write(account,dict(pool='preparation',kind='monitoring',cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
                if events and any(e['kind']!='batch_complete' for e in events) and (state/'ARMED').exists():
                    try:deliver(campaign,state,cfg,events)
                    except ReviewOwnerBusyError:pass
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True);write(state/'STATUS.json',dict(state='stopped',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)
            write(account,dict(pool='preparation',kind='monitoring',cpu_seconds=time.process_time()-started,state='STOPPED'),immutable=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('campaign','state','config'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
