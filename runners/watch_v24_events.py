"""Finite V24 completion/failure and fixed twelve-hour reviews; no dispatch."""
import argparse
from datetime import datetime,timedelta,timezone
import os
from pathlib import Path
import time
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v24 import contract
from runners.watch_v23_events import owner,event,pending,deliver,closed,alive,require_native_review_host


def scheduled(campaign,current):
    checkpoint=datetime.fromisoformat(contract.OPEN)+timedelta(hours=12);number=1
    ending=datetime.fromisoformat(contract.CLOSE)
    while checkpoint<ending:
        if current>=checkpoint:
            event(campaign,f'v24-health12-{number:02d}','twelve_hour_health',dict(due=checkpoint.isoformat()))
        checkpoint+=timedelta(hours=12);number+=1
    if current>=ending:
        (campaign/'STOP').touch(exist_ok=True)
        event(campaign,'v24-pilot-window-end','final_deadline',dict(due=contract.CLOSE))


def observe(campaign,is_alive=alive):
    active=[]
    for packet in sorted((campaign/'packets').glob('*')):
        if not packet.is_dir() or not (packet/'STATUS.json').exists():continue
        status=read(packet/'STATUS.json');state=status['state']
        if (packet/'COMPLETE.json').exists():
            event(campaign,packet.name+'-complete','batch_complete',dict(packet=packet.name,
                complete_sha256=digest(packet/'COMPLETE.json')))
        elif state=='FAILED':
            event(campaign,packet.name+'-failed','packet_failure',dict(packet=packet.name,
                status_sha256=digest(packet/'STATUS.json'),error=status.get('error')))
        elif state=='RUNNING':
            if is_alive(status['parent_pid']):active.append(packet.name)
            else:event(campaign,packet.name+'-owner-missing','owner_missing',dict(packet=packet.name,
                status_sha256=digest(packet/'STATUS.json'),recorded_parent_pid=status['parent_pid']))
    return active


def run(campaign,state,config_path):
    require_native_review_host();config=read(config_path)
    if read(campaign/'ACCEPTANCE.json')!=contract.acceptance():raise ValueError('unknown V24 acceptance')
    if datetime.now(timezone.utc)<datetime.fromisoformat(contract.OPEN):raise ValueError('V24 pilot monitoring opens with the pilot window')
    started=time.process_time();cost_path=campaign/'accounting/notifier.json'
    prior=read(cost_path).get('cpu_seconds',0) if cost_path.exists() else 0
    with owner(state):
        try:
            while not closed(campaign) and not (state/'STOP').exists():
                current=datetime.now(timezone.utc);active=observe(campaign);scheduled(campaign,current)
                events=pending(campaign,state)
                write(state/'STATUS.json',dict(state='monitoring',pid=os.getpid(),heartbeat=current.isoformat(),
                    armed=(state/'ARMED').exists(),active_packets=active,pending_events=len(events)),immutable=False)
                write(cost_path,dict(kind='monitoring',cpu_seconds=prior+time.process_time()-started,state='running'),immutable=False)
                if events and (state/'ARMED').exists():
                    # V24 only emits terminal packet, health and deadline events.
                    # A review lands/acknowledges these; it cannot defer unfinished
                    # evidence through the closed V23 campaign's deadline adapter.
                    if (state/'DEFERRED_EVENTS.json').exists():raise ValueError('V24 review must land inspected events; V23 deferred-event format is not supported')
                    deliver(campaign,state,config,events)
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(cost_path,dict(kind='monitoring',cpu_seconds=prior+time.process_time()-started,state='stopped'),immutable=False)
            write(state/'STATUS.json',dict(state='stopped',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('campaign','state','config'):parser.add_argument('--'+field,type=Path,required=True)
    args=parser.parse_args();run(args.campaign.resolve(),args.state.resolve(),args.config.resolve())
