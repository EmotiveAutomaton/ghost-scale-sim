"""One V26 terminal-event / twelve-hour monitor, with declared daily report events."""
import argparse,os,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.watch_v23_events import owner,event,pending,deliver,closed,alive,require_native_review_host,ReviewOwnerBusyError,deferred_events

def observe(campaign):
    active=[]
    for p in (campaign/'packets').glob('*'):
        if not (p/'STATUS.json').exists():continue
        s=read(p/'STATUS.json');plan=read(p/'PLAN.json')
        if (p/'COMPLETE.json').exists() and plan.get('scientific_kind')!='probe':event(campaign,p.name+'-complete','batch_complete',dict(packet=p.name,complete_sha256=digest(p/'COMPLETE.json')))
        elif s['state'] in ('FAILED','ADMISSION_HELD'):event(campaign,p.name+'-'+s['state'].lower(),'packet_failure',dict(packet=p.name,status_sha256=digest(p/'STATUS.json')))
        elif s['state']=='RUNNING':
            if alive(s['parent_pid']):active.append(p.name)
            else:event(campaign,p.name+'-owner-missing','owner_missing',dict(packet=p.name,status_sha256=digest(p/'STATUS.json')))
    q=campaign/'QUEUE_STATUS.json'
    if q.exists():
        status=read(q)
        # The build receipt can precede the controller's first owner heartbeat.
        # Read once: a simultaneous atomic status replacement must not mix states.
        if status['state'] not in ('BUILDING','DRAINED','CLOSED'):
            pid=status.get('pid')
            if not pid or not alive(pid):event(campaign,'queue-owner-missing-'+digest(q)[:12],'owner_missing',dict(queue_sha256=digest(q)))
    return active

def scheduled(campaign,current):
    a=read(campaign/'ACCEPTANCE.json');point=datetime.fromisoformat(a['opened'])+timedelta(hours=12);i=1
    # A separately authorized follow-up retains the original acceptance clock,
    # but must not wake on historical checkpoints from the closed primary queue.
    start=campaign/'MONITOR_START.json'
    floor=datetime.fromisoformat(read(start)['at']) if start.exists() else datetime.fromisoformat(a['opened'])
    while point<datetime.fromisoformat(a['drain']):
        if current>=point>=floor:event(campaign,f'v26-health12-{i:02d}','twelve_hour_health',dict(due=point.isoformat()))
        point+=timedelta(hours=12);i+=1
    for day in range(3,10):
        point=datetime(2026,11,day,4,tzinfo=timezone.utc) # preceding20:00 PST
        if current>=point>=floor:event(campaign,f'v26-daily-{day-1:02d}','daily_report',dict(due=point.isoformat()))
    for key in ('new_starts_close','drain'):
        if current>=datetime.fromisoformat(a[key]):event(campaign,'v26-'+key,'final_deadline',dict(due=a[key]))
    supplement=campaign/'SUPPLEMENT.json'
    if supplement.exists() and read(supplement).get('review_target'):
        target=read(supplement)['review_target']
        if current>=datetime.fromisoformat(target):event(campaign,'v26-friday-theory-review','review_boundary',dict(due=target,hold_new_admissions=True))
    if current>=datetime.fromisoformat(a['drain']):(campaign/'STOP').touch(exist_ok=True)

def run(campaign,state,config):
    require_native_review_host();cfg=read(config);started=time.process_time();account=campaign/'accounting'/f'monitor-{os.getpid()}-{time.time_ns()}.json'
    with owner(state):
        try:
            while not closed(campaign) and not (state/'STOP').exists():
                current=datetime.now(timezone.utc);active=observe(campaign);scheduled(campaign,current);events=pending(campaign,state)
                deferred=deferred_events(campaign,state,events,current);events=[e for e in events if e['id'] not in deferred]
                write(state/'STATUS.json',dict(state='monitoring',pid=os.getpid(),heartbeat=current.isoformat(),armed=(state/'ARMED').exists(),active_packets=active,pending_events=len(events)),immutable=False)
                write(account,dict(pool='preparation',kind='monitoring',cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
                # Routine results are already landed locally by the finite
                # controller. Batch model review at the twelve-hour/daily checks;
                # failures, depletion and deadline boundaries wake earlier.
                actionable=any(e['kind']!='batch_complete' for e in events)
                if events and actionable and (state/'ARMED').exists():
                    try:deliver(campaign,state,cfg,events)
                    except ReviewOwnerBusyError:pass
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(state/'STATUS.json',dict(state='stopped',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)
            write(account,dict(pool='preparation',kind='monitoring',cpu_seconds=time.process_time()-started,state='STOPPED'),immutable=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('campaign','state','config'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
