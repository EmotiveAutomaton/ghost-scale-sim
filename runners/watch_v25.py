"""One V25 terminal-event / twelve-hour monitor; never owns scientists."""
import argparse,os,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.watch_v23_events import owner,event,pending,deliver,closed,alive,require_native_review_host,ReviewOwnerBusyError,deferred_events

def observe(campaign):
    active=[];prepared=[]
    for p in (campaign/'packets').glob('*'):
        if not (p/'STATUS.json').exists():continue
        s=read(p/'STATUS.json')
        if (p/'COMPLETE.json').exists():event(campaign,p.name+'-complete','batch_complete',dict(packet=p.name,complete_sha256=digest(p/'COMPLETE.json')))
        elif s['state']=='FAILED':event(campaign,p.name+'-failed','packet_failure',dict(packet=p.name,status_sha256=digest(p/'STATUS.json')))
        elif s['state']=='RUNNING':
            if alive(s['parent_pid']):active.append(p.name)
            else:event(campaign,p.name+'-owner-missing','owner_missing',dict(packet=p.name,status_sha256=digest(p/'STATUS.json')))
        elif s['state']=='PREPARED':prepared.append(p.name)
    status=campaign/'QUEUE_STATUS.json'
    if prepared and status.exists() and not alive(read(status)['pid']):
        event(campaign,'queue-owner-missing-'+digest(status)[:12],'owner_missing',dict(queue_sha256=digest(status),prepared=prepared))
    return active

def scheduled(campaign,current):
    a=read(campaign/'ACCEPTANCE.json');point=datetime.fromisoformat(a['opened'])+timedelta(hours=12);i=1
    while point<datetime.fromisoformat(a['drain']):
        if current>=point:event(campaign,f'v25-health12-{i:02d}','twelve_hour_health',dict(due=point.isoformat()))
        point+=timedelta(hours=12);i+=1
    for name,key in [('new-starts-close','new_starts_close'),('drain','drain')]:
        if current>=datetime.fromisoformat(a[key]):event(campaign,'v25-'+name,'final_deadline',dict(due=a[key]))
    if current>=datetime.fromisoformat(a['drain']):(campaign/'STOP').touch(exist_ok=True)
    for identity,due in [('v24-administrative-close','2026-10-26T05:00:00+00:00'),('wb1-delivery-cutoff','2026-10-28T19:00:00+00:00')]:
        if current>=datetime.fromisoformat(due):event(campaign,identity,'review_boundary',dict(due=due))

def actionable(events,active):
    """Batch routine N16 blocks/probes at health checks; wake for decisions."""
    for item in events:
        if item['kind']!='batch_complete':return events
        name=item.get('evidence',{}).get('packet',item['id'])
        if not name.startswith('v24-n16-confirmation-block-') and '-probe-' not in name:return events
    return []


def reviewable(campaign,state,events,current):
    """Keep exact inspected backlog pending until its bounded next review."""
    deferred=deferred_events(campaign,state,events,current)
    return [item for item in events if item['id'] not in deferred]

def observe_production(campaign):
    root=campaign.parent/'v24-n16-production'
    for name in ('TERMINAL.json','RESOURCE_REPAIR_EVALUATION.json','RESOURCE_REPAIR_EVALUATION_R3.json','RESOURCE_REPAIR_ACCEPTED.json','TERMINAL_AFTER_REPAIR.json','THROUGHPUT_24H.json'):
        path=root/name
        if path.exists():event(campaign,'n16-'+name.lower().replace('.json',''),'production_boundary',dict(file=name,sha256=digest(path)))

def deliver_when_available(campaign,state,config,events):
    try:
        deliver(campaign,state,config,events)
    except ReviewOwnerBusyError:
        # A review can outlive the service that launched it. Preserve its
        # ownership without terminating the replacement monitor.
        write(state/'STATUS.json',dict(state='waiting_for_existing_review',pid=os.getpid(),
            heartbeat=datetime.now(timezone.utc).isoformat(),armed=(state/'ARMED').exists()),immutable=False)
        return False
    return True


def run(campaign,state,config):
    require_native_review_host();cfg=read(config);started=time.process_time();prior=read(campaign/'accounting/monitor.json')['cpu_seconds'] if (campaign/'accounting/monitor.json').exists() else 0
    with owner(state):
        try:
            while not closed(campaign) and not (state/'STOP').exists():
                current=datetime.now(timezone.utc);active=observe(campaign);scheduled(campaign,current);observe_production(campaign);events=pending(campaign,state)
                write(state/'STATUS.json',dict(state='monitoring',pid=os.getpid(),heartbeat=current.isoformat(),armed=(state/'ARMED').exists(),active_packets=active,pending_events=len(events)),immutable=False)
                write(campaign/'accounting/monitor.json',dict(pool='preparation',kind='monitoring',cpu_seconds=prior+time.process_time()-started,state='RUNNING'),immutable=False)
                ready=actionable(reviewable(campaign,state,events,current),active)
                if ready and (state/'ARMED').exists():deliver_when_available(campaign,state,cfg,ready)
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(state/'STATUS.json',dict(state='stopped',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)
            write(campaign/'accounting/monitor.json',dict(pool='preparation',kind='monitoring',cpu_seconds=prior+time.process_time()-started,state='STOPPED'),immutable=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('campaign','state','config'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
