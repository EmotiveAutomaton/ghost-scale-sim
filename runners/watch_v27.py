"""V27 finite native monitoring: build wakes, twelve-hour health and daily line."""
import argparse,os,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
from ghostscale.validation.soundingline.v27.common import read,write
from runners.watch_v23_events import owner,event,pending,deliver,closed,require_native_review_host,ReviewOwnerBusyError,deferred_events
from runners.watch_v26 import observe


def scheduled(c,current):
    a=read(c/'ACCEPTANCE.json');opened=datetime.fromisoformat(a['opened']);drain=datetime.fromisoformat(a['drain'])
    floor=datetime.fromisoformat(read(c/'MONITOR_START.json')['at']);point=opened+timedelta(hours=12);i=1
    while point<drain:
        if current>=point>=floor:event(c,f'v27-health12-{i:02d}','twelve_hour_health',dict(due=point.isoformat()))
        point+=timedelta(hours=12);i+=1
    for day in range(8,17):
        point=datetime(2026,11,day,4,tzinfo=timezone.utc)
        if current>=point>=floor:event(c,f'v27-daily-{day-1:02d}','daily_report',dict(due=point.isoformat()))
    q=read(c/'QUEUE_STATUS.json') if (c/'QUEUE_STATUS.json').exists() else dict(build_work_remaining=True)
    wake=c/'BUILD_WAKE.json'
    if q.get('build_work_remaining') or q.get('prepared') or (not q.get('active') and q.get('eligible_ready')):
        plan=read(wake) if wake.exists() else dict(due=(current+timedelta(minutes=50)).isoformat())
        if current>=datetime.fromisoformat(plan['due']):
            event(c,'v27-build-'+plan['due'].replace(':','-'),'build_required',dict(due=plan['due']))
            plan=dict(due=(current+timedelta(minutes=50)).isoformat())
        write(wake,plan,immutable=False)
    elif wake.exists():wake.unlink()
    plan=c/'WAKE_PLAN.json'
    if plan.exists():
        p=read(plan)
        if current>=datetime.fromisoformat(p['due']):event(c,p['id'],'planned_review',dict(due=p['due'],reason=p['reason']))
    for key in ('new_starts_close','drain'):
        if current>=datetime.fromisoformat(a[key]):event(c,'v27-'+key,'final_deadline',dict(due=a[key]))
    if current>=drain:(c/'STOP').touch(exist_ok=True)


def run(c,state,config):
    require_native_review_host();cfg=read(config);start=time.process_time();account=c/'accounting'/f'monitor-{os.getpid()}-{time.time_ns()}.json'
    with owner(state):
        try:
            while not closed(c) and not (state/'STOP').exists():
                current=datetime.now(timezone.utc);active=observe(c);scheduled(c,current)
                events=pending(c,state);deferred=deferred_events(c,state,events,current);events=[e for e in events if e['id'] not in deferred]
                write(state/'STATUS.json',dict(state='monitoring',pid=os.getpid(),heartbeat=current.isoformat(),armed=(state/'ARMED').exists(),active_packets=active,pending_events=len(events)),immutable=False)
                write(account,dict(pool='preparation',kind='monitoring',cpu_seconds=time.process_time()-start,state='RUNNING'),immutable=False)
                if events and any(e['kind']!='batch_complete' for e in events) and (state/'ARMED').exists():
                    try:deliver(c,state,cfg,events)
                    except ReviewOwnerBusyError:pass
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(state/'STATUS.json',dict(state='stopped',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)
            write(account,dict(pool='preparation',kind='monitoring',cpu_seconds=time.process_time()-start,state='STOPPED'),immutable=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('campaign','state','config'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
