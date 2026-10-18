"""One authorized future design/pilot handoff; no scientific dispatch or polling agent."""
import argparse,time,os
from pathlib import Path
from datetime import datetime,timezone
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.watch_v23_events import owner,require_native_review_host,event,deliver,pending,alive

def eligible(current,config):
    return datetime.fromisoformat(config['due'])<=current<datetime.fromisoformat(config['expires'])

def run(root,config_path):
    require_native_review_host();config=read(config_path);state=root/'notifications';started=time.process_time()
    if digest(config['authority_packet'])!=config['authority_packet_sha256']:raise ValueError('future handoff authority changed')
    with owner(state):
        try:
            while True:
                now=datetime.now(timezone.utc)
                if (root/'STOP').exists() or now>=datetime.fromisoformat(config['expires']):
                    write(root/'STATUS.json',dict(state='STOPPED_WITHOUT_DELIVERY',at=now.isoformat(),pid=os.getpid()),immutable=False);return
                write(root/'CPU.json',dict(cpu_seconds=time.process_time()-started),immutable=False)
                if eligible(now,config):
                    active=False
                    for other in config.get('prior_notification_states',[]):
                        path=Path(other)/'STATUS.json'
                        if path.exists():
                            pid=read(path).get('active_review_pid');active=active or bool(pid and alive(pid))
                    if not active:
                        event(root,'v24-authorized-design-handoff','authorized_future_handoff',dict(authority_sha256=config['authority_packet_sha256'],
                            pilot_cpu_hours=40,scientific_workers=2,production_not_before='2026-10-26T08:00:00-07:00'))
                        (state/'ARMED').touch();write(root/'STATUS.json',dict(state='DELIVERING_ONCE',at=now.isoformat(),pid=os.getpid()),immutable=False)
                        deliver(root,state,config,pending(root,state))
                        write(root/'STATUS.json',dict(state='ONE_HANDOFF_FINISHED',at=datetime.now(timezone.utc).isoformat(),pid=os.getpid()),immutable=False)
                        return
                write(root/'STATUS.json',dict(state='WAITING_FOR_AUTHORIZED_WINDOW',due=config['due'],expires=config['expires'],pid=os.getpid(),heartbeat=now.isoformat()),immutable=False)
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(root/'CPU.json',dict(cpu_seconds=time.process_time()-started,terminal=True),immutable=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.config.resolve())
