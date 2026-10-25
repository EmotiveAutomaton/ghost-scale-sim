"""One V25 terminal-event / twelve-hour monitor; never owns scientists."""
import argparse,os,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.watch_v23_events import owner,event,pending,deliver,closed,alive,require_native_review_host

def observe(campaign):
    active=[]
    for p in (campaign/'packets').glob('*'):
        if not (p/'STATUS.json').exists():continue
        s=read(p/'STATUS.json')
        if (p/'COMPLETE.json').exists():event(campaign,p.name+'-complete','batch_complete',dict(packet=p.name,complete_sha256=digest(p/'COMPLETE.json')))
        elif s['state']=='FAILED':event(campaign,p.name+'-failed','packet_failure',dict(packet=p.name,status_sha256=digest(p/'STATUS.json')))
        elif s['state']=='RUNNING':
            if alive(s['parent_pid']):active.append(p.name)
            else:event(campaign,p.name+'-owner-missing','owner_missing',dict(packet=p.name,status_sha256=digest(p/'STATUS.json')))
    return active

def scheduled(campaign,current):
    a=read(campaign/'ACCEPTANCE.json');point=datetime.fromisoformat(a['opened'])+timedelta(hours=12);i=1
    while point<datetime.fromisoformat(a['drain']):
        if current>=point:event(campaign,f'v25-health12-{i:02d}','twelve_hour_health',dict(due=point.isoformat()))
        point+=timedelta(hours=12);i+=1
    for name,key in [('new-starts-close','new_starts_close'),('drain','drain')]:
        if current>=datetime.fromisoformat(a[key]):event(campaign,'v25-'+name,'final_deadline',dict(due=a[key]))
    if current>=datetime.fromisoformat(a['drain']):(campaign/'STOP').touch(exist_ok=True)

def run(campaign,state,config):
    require_native_review_host();cfg=read(config);started=time.process_time();prior=read(campaign/'accounting/monitor.json')['cpu_seconds'] if (campaign/'accounting/monitor.json').exists() else 0
    with owner(state):
        try:
            while not closed(campaign) and not (state/'STOP').exists():
                current=datetime.now(timezone.utc);active=observe(campaign);scheduled(campaign,current);events=pending(campaign,state)
                write(state/'STATUS.json',dict(state='monitoring',pid=os.getpid(),heartbeat=current.isoformat(),armed=(state/'ARMED').exists(),active_packets=active,pending_events=len(events)),immutable=False)
                write(campaign/'accounting/monitor.json',dict(pool='preparation',kind='monitoring',cpu_seconds=prior+time.process_time()-started,state='RUNNING'),immutable=False)
                if events and (state/'ARMED').exists():deliver(campaign,state,cfg,events)
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(state/'STATUS.json',dict(state='stopped',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)
            write(campaign/'accounting/monitor.json',dict(pool='preparation',kind='monitoring',cpu_seconds=prior+time.process_time()-started,state='STOPPED'),immutable=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('campaign','state','config'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
