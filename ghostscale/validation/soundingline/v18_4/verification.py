"""Metered verification occupies the same single-worker queue as science."""
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from ..v18_3.io import read,write,file_digest,now
from ..v18_3.native import ProcessClock
from ..v16.runtime import local_owner


def run(root,campaign):
    from .runtime import REPO
    with local_owner(campaign/'scientific-worker-owner'),local_owner(root):
        plan=read(root/'PLAN.json');design=plan['design'];parent=campaign/design['parent']
        if any(file_digest(REPO/n)!=v for n,v in plan['sources'].items()):raise ValueError('verification source changed')
        if file_digest(parent/'PLAN.json')!=design['parent_plan_sha256']:raise ValueError('verification parent changed')
        if (root/'COMPLETE.json').exists():return 'complete'
        acceptance=read(campaign/'ACCEPTANCE.json');previous=[read(p) for p in campaign.glob('attempts/*.json')]
        if any(p['state']=='running' for p in previous):raise ValueError('unreconciled owner')
        old=acceptance['prior_cpu_seconds']+sum(max(p['cpu_seconds'],p.get('native_cpu_seconds',0),p.get('uncertainty_cpu_seconds',0))+p.get('child_cpu_seconds',0) for p in previous)
        attempt=uuid.uuid4().hex;start=time.monotonic();cpu=time.process_time();child_cpu=0.;child=None
        def emit(state,**kw):
            write(root/'STATUS.json',dict(state=state,pid=os.getpid(),parent_pid=os.getppid(),attempt=attempt,heartbeat=now(),**kw),immutable=False)
            write(campaign/'attempts'/f'{attempt}.json',dict(packet=root.name,state=state,cpu_seconds=time.process_time()-cpu,child_cpu_seconds=child_cpu,wall_seconds=time.monotonic()-start),immutable=False)
        emit('running')
        try:
            if old>=acceptance['cumulative_cpu_ceiling_seconds']-60 or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start']):
                emit('resource_cutoff');return 'resource_cutoff'
            extracted=root/'extracted';extracted.mkdir(exist_ok=True)
            with zipfile.ZipFile(parent/'SOURCE.zip') as z:
                if any(not (extracted/n).resolve().is_relative_to(extracted.resolve()) for n in z.namelist()):raise ValueError('source path escapes')
                z.extractall(extracted)
            driver='runners/replay_v18_4.py';target=extracted/driver
            if target.exists() and file_digest(target)!=file_digest(REPO/driver):raise ValueError('frozen replay driver differs')
            shutil.copyfile(REPO/driver,target)
            python=os.environ['GHOST_V18_TORCH_PYTHON'] if read(parent/'PLAN.json')['design']['engine']=='neural' else sys.executable
            env=os.environ.copy();env['GHOST_VERIFY_DEADLINE']=acceptance['report_start']
            env['GHOST_VERIFY_CPU_LIMIT']=str(max(0,acceptance['cumulative_cpu_ceiling_seconds']-old-(time.process_time()-cpu)-5))
            with (root/'replay.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([str(python),'-B','-m','runners.replay_v18_4','--root',str(parent),'--output',str(root/'PROOF.json')],
                    cwd=extracted,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                clock=None
                while child.poll() is None:
                    status=root/'REPLAY-STATUS.json'
                    if clock is None and status.exists() and os.name=='nt':
                        s=read(status)
                        if s['pid']==child.pid or s['parent_pid']==child.pid:
                            try:clock=ProcessClock(s['pid'])
                            except OSError:pass
                    if clock:child_cpu=max(child_cpu,clock.seconds())
                    emit('running',child_pid=child.pid);time.sleep(1)
                if clock:child_cpu=max(child_cpu,clock.seconds());clock.close()
            if child.returncode:child_cpu=max(child_cpu,time.monotonic()-start);raise RuntimeError('retained verification failed')
            proof=read(root/'PROOF.json');child_cpu=max(child_cpu,proof['cpu_seconds'])
            write(parent/'INDEPENDENT_REPLAY.json',proof)
            write(root/'COMPLETE.json',dict(parent=design['parent'],proof_sha256=file_digest(root/'PROOF.json'),completed_at=now()))
            emit('complete');return 'complete'
        except BaseException as exc:
            emit('running' if child is not None and child.poll() is None else 'failed',error=repr(exc));raise
