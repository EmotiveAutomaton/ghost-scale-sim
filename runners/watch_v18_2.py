"""Finite native queue: one worker, immutable deadline, transitions only."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from ghostscale.validation.soundingline.v16.records import write,read,file_digest,now
from ghostscale.validation.soundingline.v16.runtime import local_owner


def supervise(campaign,queue,python):
    with local_owner(campaign/'supervisor'):
        plan=read(queue);acceptance=read(campaign/'ACCEPTANCE.json')
        for job in plan['jobs']:
            root=Path(job['root']);source=Path(job['source'])
            if file_digest(root/'PLAN.json')!=job['plan_sha256']:raise ValueError('queued plan changed')
            if (root/'COMPLETE.json').exists():continue
            if datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start']):break
            with (root/'worker.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([str(python),'-B','-m','runners.run_v18_2','--root',str(root),'--campaign',str(campaign)],
                    cwd=source,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while child.poll() is None:
                    write(campaign/'QUEUE-STATUS.json',dict(state='running',pid=os.getpid(),child_pid=child.pid,
                        job=root.name,heartbeat=now()),immutable=False)
                    time.sleep(5)
            complete=(root/'COMPLETE.json').exists()
            event=dict(kind='batch_complete' if complete else 'batch_failed_or_cutoff',job=root.name,
                       exit_code=child.returncode,at=now(),plan_sha256=job['plan_sha256'])
            write(campaign/'events'/f'{root.name}.json',event)
            if not complete:
                # Preserve failed records, proceed with unrelated admitted branches.
                write(root/'DISPOSITION.json',dict(state='failed_or_cutoff',exit_code=child.returncode,
                    needs_bounded_repair=True,at=now()),immutable=False)
        write(campaign/'QUEUE-STATUS.json',dict(state='drained',pid=os.getpid(),heartbeat=now()),immutable=False)
        write(campaign/'events'/'queue-drained.json',dict(kind='queue_drained',at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--python',type=Path,required=True)
    a=parser.parse_args();supervise(a.campaign,a.queue,a.python)
