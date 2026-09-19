"""Manually commissioned finite queue; terminal events only, never auto-expansion."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import time
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_3.native import ProcessClock


def supervise(campaign,queue,python):
    with local_owner(campaign/'supervisor'):
        plan=read(queue);acceptance=read(campaign/'ACCEPTANCE.json');failures=[];cutoff=False
        if len({j['root'] for j in plan['jobs']})!=len(plan['jobs']):raise ValueError('duplicate produce namespace')
        for job in plan['jobs']:
            if (campaign/'STOP').exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start']):cutoff=True;break
            root=Path(job['root']);source=Path(job['source'])
            if file_digest(root/'PLAN.json')!=job['plan_sha256']:raise ValueError('queued plan changed')
            if (root/'COMPLETE.json').exists():continue
            with (root/'worker.log').open('ab',buffering=0) as log:
                started=time.time();clock=None
                child=subprocess.Popen([str(python),'-B','-m','runners.run_v18_3','--root',str(root),'--campaign',str(campaign)],
                    cwd=source,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while child.poll() is None:
                    status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else None
                    if status and datetime.fromisoformat(status['heartbeat']).timestamp()>=started:
                        if clock is None and os.name=='nt' and (status['pid']==child.pid or status['parent_pid']==child.pid):
                            try:clock=ProcessClock(status['pid'])
                            except OSError:pass # final self accounting or explicit conservative failure charge
                    write(campaign/'QUEUE-STATUS.json',dict(state='running',pid=os.getpid(),child_pid=child.pid,
                          job=root.name,heartbeat=now()),immutable=False)
                    time.sleep(5)
                if (root/'STATUS.json').exists():
                    status=read(root/'STATUS.json')
                    if datetime.fromisoformat(status['heartbeat']).timestamp()>=started:
                        path=campaign/'attempts'/f"{status['attempt']}.json";receipt=read(path)
                        if clock is not None:receipt['native_cpu_seconds']=clock.seconds()
                        if receipt['state']=='running':
                            receipt['interrupted_state']='running';receipt['state']='native_failed'
                            if clock is None:receipt['uncertainty_cpu_seconds']=time.time()-started
                        write(path,receipt,immutable=False)
                if clock is not None:clock.close()
            complete=(root/'COMPLETE.json').exists()
            write(campaign/'events'/f'{root.name}.json',dict(kind='batch_complete' if complete else 'batch_failed_or_cutoff',
                  job=root.name,exit_code=child.returncode,at=now(),plan_sha256=job['plan_sha256']))
            if not complete:
                failures.append(root.name)
                write(root/'DISPOSITION.json',dict(state='failed_or_cutoff',exit_code=child.returncode,at=now()))
        state='resource_cutoff' if cutoff else ('drained_with_failures' if failures else 'drained')
        write(campaign/'QUEUE-STATUS.json',dict(state=state,pid=os.getpid(),heartbeat=now(),failed_packets=failures),immutable=False)
        write(campaign/'events'/f'{queue.stem}-drained.json',dict(kind=state,failed_packets=failures,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True);a=p.parse_args();supervise(a.campaign,a.queue,a.python)
