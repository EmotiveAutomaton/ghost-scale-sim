"""One finite extension monitor: declared milestones, failures, and twelve hours."""
import argparse
from datetime import datetime,timezone,timedelta
import os
from pathlib import Path
import time
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.watch_v23_events import owner,event,pending,deliver,require_native_review_host,alive,ReviewOwnerBusyError

def run(root,state,config):
    require_native_review_host();cfg=read(config);started=time.process_time()
    account=root/'accounting'/f'monitor-{os.getpid()}.json'
    with owner(state):
        try:
            while not (state/'STOP').exists():
                current=datetime.now(timezone.utc);a=read(root/'ACCEPTANCE.json')
                q=read(root/'QUEUE_STATUS.json') if (root/'QUEUE_STATUS.json').exists() else None
                if q and not (root/'TERMINAL.json').exists() and not alive(q['pid']):
                    event(root,'n16r-owner-missing','owner_missing',dict(queue_sha256=digest(root/'QUEUE_STATUS.json'),state=q['state']))
                checkpoint=datetime.fromisoformat(a['opened'])+timedelta(hours=12);i=1
                while checkpoint<datetime.fromisoformat(a['drain']) and checkpoint<=current:
                    event(root,f'n16r-health12-{i}','twelve_hour_health',dict(due=checkpoint.isoformat()));checkpoint+=timedelta(hours=12);i+=1
                if current>=datetime.fromisoformat(a['drain']):(root/'STOP').touch(exist_ok=True)
                events=pending(root,state)
                write(state/'STATUS.json',dict(state='monitoring',pid=os.getpid(),heartbeat=current.isoformat(),armed=(state/'ARMED').exists(),pending=len(events)),immutable=False)
                write(account,dict(kind='monitor',state='RUNNING',cpu_seconds=time.process_time()-started),immutable=False)
                if events and (state/'ARMED').exists():
                    try:deliver(root,state,cfg,events)
                    except ReviewOwnerBusyError:pass
                if (root/'TERMINAL.json').exists() and not pending(root,state):break
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(state/'STATUS.json',dict(state='stopped',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),armed=False),immutable=False)
            write(account,dict(kind='monitor',state='STOPPED',cpu_seconds=time.process_time()-started),immutable=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','state','config'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.state.resolve(),a.config.resolve())
