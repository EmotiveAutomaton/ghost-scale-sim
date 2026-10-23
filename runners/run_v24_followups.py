"""One finite serial queue; record a failed family, never retry or invent work."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import time
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v24 import contract
from runners.watch_v23_events import event


def run(campaign,queue,manifest_sha256):
    if os.name!='nt':raise RuntimeError('native Windows process ownership required')
    if digest(queue/'MANIFEST.json')!=manifest_sha256:raise ValueError('queue manifest changed')
    manifest=read(queue/'MANIFEST.json')
    if digest(campaign/'ACCEPTANCE.json')!=manifest['acceptance_sha256']:raise ValueError('acceptance changed')
    if not datetime.fromisoformat(contract.OPEN)<=datetime.now(timezone.utc)<datetime.fromisoformat(contract.CLOSE):raise ValueError('outside pilot window')
    if (queue/'STATUS.json').exists():raise ValueError('queue already attempted; no automatic retry')
    # Controller also runs from a verified member capsule, not a mutable checkout.
    source=Path(__file__).resolve().parents[1];member=source.parent
    plan=read(member/'PLAN.json')
    from runners.run_v24_pilot import verify_sources
    verify_sources(member,plan)
    if digest(Path(__file__))!=plan['source_files']['runners/run_v24_followups.py']:raise ValueError('controller source changed')
    with (queue/'OWNER.lock').open('x') as f:f.write(str(os.getpid()))
    started=time.process_time();finished=[];failed=[];active=None;state='FAILED'
    try:
        for item in manifest['packets']:
            name=item['packet']
            if Path(name).name!=name:raise ValueError('unsafe packet path')
            packet=campaign/'packets'/name
            if digest(packet/'PLAN.json')!=item['plan_sha256']:raise ValueError('queued plan changed')
            if (campaign/'STOP').exists() or (campaign/'OWNER.lock').exists():raise ValueError('stop or competing scientific owner')
            p=read(packet/'PLAN.json')
            env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(packet/'source'),str(packet/'source/dependencies')]),PYTHONNOUSERSITE='1')
            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
            with (queue/(name+'.log')).open('xb') as log:
                active=subprocess.Popen([p['interpreter'],'-B','-m','runners.run_v24_pilot','--packet',str(packet)],cwd=packet/'source',env=env,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                while active.poll() is None:
                    write(queue/'STATUS.json',dict(state='RUNNING',pid=os.getpid(),packet=name,launcher_pid=active.pid,
                        heartbeat=datetime.now(timezone.utc).isoformat(),finished=finished,failed=failed),immutable=False)
                    time.sleep(5)
                code=active.returncode;active=None
            if code:
                failed.append(name)
                event(campaign,name+'-failed','packet_failure',dict(packet=name,status_sha256=digest(packet/'STATUS.json')))
                if (campaign/'OWNER.lock').exists():raise ValueError('failed owner did not release science lock')
            else:
                complete=read(packet/'COMPLETE.json')
                for key,filename in [('plan_sha256','PLAN.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json'),('summary_sha256','original/SUMMARY.json')]:
                    if complete[key]!=digest(packet/filename):raise ValueError('completed packet binding differs')
                finished.append(name)
                event(campaign,name+'-complete','batch_complete',dict(packet=name,complete_sha256=digest(packet/'COMPLETE.json')))
        state='COMPLETE_WITH_FAILURES' if failed else 'COMPLETE'
        event(campaign,'v24-oct23-followups-drained','review_boundary',dict(finished=finished,failed=failed))
    finally:
        write(queue/'STATUS.json',dict(state=state,pid=os.getpid(),finished=finished,failed=failed,
            active_launcher_pid=active.pid if active and active.poll() is None else None),immutable=False)
        write(campaign/'accounting/oct23-followup-controller.json',dict(kind='controller',state=state,cpu_seconds=time.process_time()-started),immutable=False)
        (queue/'OWNER.lock').unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--queue',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    args=p.parse_args();run(args.campaign.resolve(),args.queue.resolve(),args.manifest_sha256)
