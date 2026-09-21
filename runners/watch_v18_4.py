"""Execute agent-admitted immutable packets; queue drain requests review/refill."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import time
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now,digest
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_3.native import ProcessClock


def event(campaign,key,payload):
    path=campaign/'events'/(key+'.json')
    if not path.exists():write(path,dict(id=key,at=now(),**payload))


def supervise(campaign,queue,python,runner_module='runners.run_v18_4'):
    from ghostscale.validation.soundingline.v18_4.priority import below_normal
    below_normal()
    with local_owner(campaign/'supervisor'):
        acceptance=read(campaign/'ACCEPTANCE.json')
        while True:
            state=dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=now())
            if (campaign/'STOP').exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start']):
                write(campaign/'QUEUE-STATUS.json',dict(state,state='resource_cutoff'),immutable=False)
                event(campaign,'absolute-cutoff',dict(kind='resource_cutoff'));return
            plan=read(queue)
            if len({j['root'] for j in plan['jobs']})!=len(plan['jobs']):raise ValueError('duplicate produce namespace')
            pending=[j for j in plan['jobs'] if not (Path(j['root'])/'COMPLETE.json').exists() and not (Path(j['root'])/'DISPOSITION.json').exists()]
            if not pending:
                event(campaign,'refill-'+digest(plan)[:16],dict(kind='queue_needs_refill',queue_sha256=file_digest(queue)))
                write(campaign/'QUEUE-STATUS.json',dict(state,state='awaiting_review_refill'),immutable=False)
                time.sleep(10);continue
            job=pending[0];root=Path(job['root']);source=Path(job['source'])
            if file_digest(root/'PLAN.json')!=job['plan_sha256']:raise ValueError('admitted plan changed')
            with (root/'worker.log').open('ab',buffering=0) as log:
                started=time.time();clock=None
                child=subprocess.Popen([str(python),'-B','-m',runner_module,'--root',str(root),'--campaign',str(campaign)],
                    cwd=source,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
                while child.poll() is None:
                    status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else None
                    if status and datetime.fromisoformat(status['heartbeat']).timestamp()>=started:
                        if clock is None and os.name=='nt' and (status['pid']==child.pid or status['parent_pid']==child.pid):
                            try:clock=ProcessClock(status['pid'])
                            except OSError:pass
                    write(campaign/'QUEUE-STATUS.json',dict(state='running',pid=os.getpid(),parent_pid=os.getppid(),child_pid=child.pid,
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
            event(campaign,root.name,dict(kind='batch_complete' if complete else 'batch_failed_or_cutoff',
                job=root.name,exit_code=child.returncode,plan_sha256=job['plan_sha256']))
            if not complete:
                write(root/'DISPOSITION.json',dict(state='failed_or_cutoff',exit_code=child.returncode,at=now()))
                # Stop admission until the operator reconciles a possible surviving child.
                write(campaign/'QUEUE-STATUS.json',dict(state='needs_failure_review',pid=os.getpid(),heartbeat=now(),job=root.name),immutable=False)
                return


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True);a=p.parse_args();supervise(a.campaign,a.queue,a.python)
