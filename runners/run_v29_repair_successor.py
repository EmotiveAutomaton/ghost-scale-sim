"""Prepared finite repair: wait for actual predecessor exit, then carry costs once."""
import argparse,os,time
from pathlib import Path
from datetime import datetime,timezone
from ghostscale.validation.soundingline.v29.common import read,write,digest
from runners import run_v29_continuation as queue
from runners.watch_v23_events import owner,alive,event

def activate(c):
    a=read(c/'ACCEPTANCE.json');previous=Path(a['primary_campaign'])
    if digest(previous/'ACCEPTANCE.json')!=a['primary_acceptance_sha256']:
        raise ValueError('predecessor acceptance changed')
    with owner(c/'preparation-owner'):
        if (c/'ACTIVATED.json').exists():raise ValueError('repair already activated; inspect prior ownership')
        while not (previous/'QUEUE_COMPLETE.json').exists() or alive(read(previous/'QUEUE_STATUS.json')['pid']):
            current=datetime.now(timezone.utc)
            if current>=datetime.fromisoformat(a['new_starts_close']):raise TimeoutError('original last-start boundary')
            write(c/'QUEUE_STATUS.json',dict(state='WAITING_FOR_PREDECESSOR_EXIT',pid=os.getpid(),heartbeat=current.isoformat(),active=[],gear=2,worker_limit=4),immutable=False)
            write(c/'accounting/preparation-owner.json',dict(family='reserve',state='WAITING',cpu_seconds=time.process_time()),immutable=False)
            time.sleep(10)
        reservation=previous/'accounting/q2-h2-reservation.json';held=read(reservation)
        if held['reserved_cpu_seconds']!=2400:raise ValueError('repair reservation changed')
        write(c/'RESERVATION_TRANSFER.json',dict(source_sha256=digest(reservation),reserved_seconds=2400))
        write(reservation,dict(family='reserve',state='TRANSFERRED_TO_Q2_H2',cpu_seconds=0,reserved_cpu_seconds=0,transfer_sha256=digest(c/'RESERVATION_TRANSFER.json')),immutable=False)
        carried=sum(max(read(p).get('cpu_seconds',0),read(p).get('reserved_cpu_seconds',0)) for p in (previous/'accounting').glob('*.json'))
        write(c/'accounting/prior-protected.json',dict(family='reserve',state='COMPLETE',cpu_seconds=carried))
        write(c/'accounting/preparation-owner.json',dict(family='reserve',state='COMPLETE',cpu_seconds=time.process_time()),immutable=False)
        write(c/'ACTIVATED.json',dict(at=datetime.now(timezone.utc).isoformat(),predecessor_closeout_sha256=digest(previous/'QUEUE_COMPLETE.json'),carried_cpu_seconds=carried))
    queue.LIMITS=a['cpu_limits'];queue.run(c)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();c=a.campaign.resolve()
    try:activate(c)
    except BaseException as exc:
        event(c,'repair-successor-failed','service_failed',dict(error=str(exc)))
        raise
