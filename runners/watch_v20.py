"""V20 queue adapter: complete-block timing, dependencies and protected reserve.

Uses the prior native process/lock/event contract. Never invents scientific cards.
"""
import argparse,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from runners.watch_v18_4 import event
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now,digest
from ghostscale.validation.soundingline.v18_3.native import ProcessClock
from ghostscale.validation.soundingline.v18_4.priority import below_normal
from ghostscale.validation.soundingline.v20.runtime import consumed

def eligible(job,complete,current):
    return set(job.get('requires',[]))<=complete and current>=datetime.fromisoformat(job.get('not_before','2000-01-01T00:00:00+00:00'))

def supervise(campaign,queue,python):
    below_normal()
    with local_owner(campaign/'supervisor'):
        a=read(campaign/'ACCEPTANCE.json');last_idle=time.monotonic();idle_reason=None
        while True:
            current=datetime.now(timezone.utc);state=dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=now())
            if (campaign/'STOP').exists() or current>=datetime.fromisoformat(a['report_start']):
                write(campaign/'QUEUE-STATUS.json',dict(state,state='resource_cutoff'),immutable=False);event(campaign,'absolute-cutoff',dict(kind='resource_cutoff'));return
            if (campaign/'HOLD').exists():
                if idle_reason!='required_result_review':last_idle=time.monotonic();idle_reason='required_result_review'
                write(campaign/'QUEUE-STATUS.json',dict(state,state='awaiting_review_refill',idle_reason=idle_reason),immutable=False);time.sleep(10);continue
            plan=read(queue);jobs=plan['jobs']
            if len({j['root'] for j in jobs})!=len(jobs):raise ValueError('duplicate output namespace')
            complete={j['id'] for j in jobs if (Path(j['root'])/'COMPLETE.json').exists()}
            pending=[j for j in jobs if j['id'] not in complete and not (Path(j['root'])/'DISPOSITION.json').exists()]
            ready=[j for j in pending if eligible(j,complete,current)]
            # Once the protected confirmation window begins, confirmation owns first place.
            if current>=datetime.fromisoformat(a['confirmation_start']):ready.sort(key=lambda j:not j.get('reserve_eligible',False))
            if not ready:
                reason='awaiting_review_refill' if not pending else 'waiting_for_dependency_or_confirmation'
                if idle_reason!=reason:last_idle=time.monotonic();idle_reason=reason
                event(campaign,'refill-'+digest(plan)[:16],dict(kind='queue_needs_refill',queue_sha256=file_digest(queue),reason=reason))
                write(campaign/'QUEUE-STATUS.json',dict(state,state='awaiting_review_refill',idle_since_seconds=last_idle,idle_reason=reason),immutable=False)
                time.sleep(10);continue
            job=ready[0];root=Path(job['root']);source=Path(job['source']);used=consumed(campaign)
            cap=a['cumulative_cpu_ceiling_seconds'] if job.get('reserve_eligible') else a['exploratory_cpu_ceiling_seconds']
            remaining=(datetime.fromisoformat(a['report_start'])-current).total_seconds()
            if job['wall_estimate_seconds']>remaining or used+job['cpu_estimate_seconds']>cap:
                write(root/'DISPOSITION.json',dict(state='time_or_cpu_deferred',at=now(),remaining_wall_seconds=remaining,used_cpu_seconds=used));continue
            if file_digest(root/'PLAN.json')!=job['plan_sha256']:raise ValueError('admitted plan changed')
            if idle_reason:
                write(campaign/'idle'/f'{time.time_ns()}.json',dict(reason=idle_reason,wall_seconds=time.monotonic()-last_idle))
                idle_reason=None
            with (root/'worker.log').open('ab',buffering=0) as log:
                started=time.time();clock=None
                child=subprocess.Popen([str(python),'-B','-m','runners.run_v20','--root',str(root),'--campaign',str(campaign)],cwd=source,
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
                while child.poll() is None:
                    status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else None
                    if status and datetime.fromisoformat(status['heartbeat']).timestamp()>=started and clock is None and os.name=='nt':
                        if status['pid']==child.pid or status['parent_pid']==child.pid:
                            try:clock=ProcessClock(status['pid'])
                            except OSError:pass
                    write(campaign/'QUEUE-STATUS.json',dict(state,state='running',child_pid=child.pid,job=job['id'],heartbeat=now()),immutable=False)
                    time.sleep(5)
                if (root/'STATUS.json').exists():
                    status=read(root/'STATUS.json');path=campaign/'attempts'/f"{status['attempt']}.json";receipt=read(path)
                    if clock is not None:receipt['native_cpu_seconds']=clock.seconds()
                    if receipt['state']=='running':receipt.update(state='native_failed',uncertainty_cpu_seconds=time.time()-started)
                    write(path,receipt,immutable=False)
                if clock is not None:clock.close()
            done=(root/'COMPLETE.json').exists()
            payload=dict(kind='batch_complete' if done else 'batch_failed_or_cutoff',job=job['id'],exit_code=child.returncode,plan_sha256=job['plan_sha256'])
            write(campaign/'completions'/(job['id']+'.json'),dict(payload,at=now()))
            if not done or job.get('review_boundary',True):
                (campaign/'HOLD').touch();event(campaign,job['id'],payload)
            if not done:
                write(root/'DISPOSITION.json',dict(state='failed_or_cutoff',exit_code=child.returncode,at=now()));write(campaign/'QUEUE-STATUS.json',dict(state,state='needs_failure_review',job=job['id']),immutable=False);return
            last_idle=time.monotonic();idle_reason='between_admitted_blocks'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--queue',type=Path,required=True);p.add_argument('--python',type=Path,required=True);a=p.parse_args()
    try:supervise(a.campaign,a.queue,a.python)
    except BaseException as e:
        event(a.campaign,'supervisor-failure-'+str(os.getpid()),dict(kind='supervisor_failed',error=type(e).__name__+': '+str(e)))
        raise
