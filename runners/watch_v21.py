"""One Gear-2 owner, at most three one-thread children, bounded reserved CPU."""
import argparse,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now,digest
from ghostscale.validation.soundingline.v18_4.priority import below_normal
from ghostscale.validation.soundingline.v18_3.native import ProcessClock
from ghostscale.validation.soundingline.v21.runtime import spent
from ghostscale.validation.soundingline.v20.runtime import available_memory
from runners.release_v21_resource_hold import native_identity,matches

def artifact_size(root):
    """Atomic writers may remove a temporary file between enumeration and stat."""
    total=0
    for path in root.rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:
            continue
    return total

class AdoptedChild(ProcessClock):
    """Retain the exact existing wrapper handle; never relaunch its packet."""
    def __init__(self,identity):
        super().__init__(identity['pid'])
        if self.created!=identity['creation']:
            self.close();raise ValueError('adopted wrapper creation differs')
        self.returncode=None
        self.kernel.GetExitCodeProcess.argtypes=[self.ctypes.c_void_p,self.ctypes.POINTER(self.ctypes.c_ulong)]
    def poll(self):
        code=self.ctypes.c_ulong()
        if not self.kernel.GetExitCodeProcess(self.handle,self.ctypes.byref(code)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        self.returncode=None if code.value==259 else int(code.value)
        return self.returncode

def adopt_jobs(campaign,contract,jobs):
    """Requires an inspected, hash-bound contract and absent old owner."""
    previous=native_identity(contract['previous_owner_pid'])
    if previous and previous['alive']:raise ValueError('previous owner PID still exists')
    if file_digest(campaign/'QUEUE.json')!=contract['queue_sha256']:raise ValueError('recovery queue changed')
    active={};mapping={j['id']:j for j in jobs}
    try:
        for record in contract['workers']:
            job=mapping[record['job']];root=Path(job['root']);status=read(root/'STATUS.json')
            assert job['plan_sha256']==record['plan_sha256']==file_digest(root/'PLAN.json')
            assert status['pid']==record['worker']['pid'] and status['parent_pid']==record['wrapper']['pid']
            assert matches(native_identity(status['pid']),record['worker'])
            assert matches(native_identity(record['wrapper']['pid']),record['wrapper'])
            child=AdoptedChild(record['wrapper']);clock=ProcessClock(status['pid'])
            if clock.created!=record['worker']['creation']:raise ValueError('adopted worker creation differs')
            # Preserve the original elapsed limit, not time since adoption.
            started=datetime.fromisoformat(record['started_at']);elapsed=(datetime.now(timezone.utc)-started).total_seconds()
            active[job['id']]=dict(child=child,job=job,log=(root/'worker.log').open('ab',buffering=0),clock=clock,started=time.monotonic()-elapsed,adopted=True)
        for job in jobs:
            root=Path(job['root'])
            if (root/'STATUS.json').exists() and not (root/'COMPLETE.json').exists() and not (root/'DISPOSITION.json').exists() and job['id'] not in active:
                raise ValueError('unreconciled executed identity: '+job['id'])
        for record in contract.get('completed_after_owner_exit',[]):
            job=mapping[record['job']];root=Path(job['root'])
            assert file_digest(root/'COMPLETE.json')==record['complete_sha256']
            write(campaign/'completions'/(job['id']+'.json'),dict(at=now(),exit_code=None,executed=True,recovered_after_owner_exit=True,complete_sha256=record['complete_sha256'],limit='Original wrapper exit code not retained; immutable COMPLETE and worker final status establish execution.'))
            if job.get('review_boundary'):event(campaign,job['id'],'batch_complete',job=job['id'])
            if job.get('hold_after'):(campaign/'HOLD').touch()
        return active
    except BaseException:
        for item in active.values():
            item['child'].close();item['clock'].close();item['log'].close()
        raise

def event(campaign,identity,kind,**more):
    p=campaign/'events'/(identity+'.json')
    if not p.exists():write(p,dict(id=identity,kind=kind,at=now(),**more))

def budget_fits(used,reserved,cap,job):return used+reserved+job['cpu_cap_seconds']<=cap

def supervise(campaign,python,recovery=None):
    below_normal();active={};service_start=time.process_time();last_artifact_check=0;artifact_bytes=0
    with local_owner(campaign/'supervisor'):
        a=read(campaign/'ACCEPTANCE.json');a_hash=file_digest(campaign/'ACCEPTANCE.json')
        service_base=read(campaign/'SERVICE_COST.json').get('cpu_seconds',0) if (campaign/'SERVICE_COST.json').exists() else 0
        jobs=read(campaign/'QUEUE.json')['jobs']
        if recovery:
            contract=read(recovery);active=adopt_jobs(campaign,contract,jobs)
            write(recovery.parent/'ADOPTED.json',dict(at=now(),pid=os.getpid(),parent_pid=os.getppid(),identity=native_identity(os.getpid()),contract_sha256=file_digest(recovery),jobs=list(active),previous_service_cpu_seconds=service_base))
        else:
            for job in jobs:
                root=Path(job['root'])
                if (root/'STATUS.json').exists() and not (root/'COMPLETE.json').exists() and not (root/'DISPOSITION.json').exists():raise ValueError('executed identity requires explicit recovery contract')
        while True:
            current=datetime.now(timezone.utc)
            if file_digest(campaign/'ACCEPTANCE.json')!=a_hash:raise ValueError('acceptance changed')
            cutoff=(campaign/'STOP').exists() or current>=datetime.fromisoformat(a['report_start'])
            if time.monotonic()-last_artifact_check>60:
                artifact_bytes=artifact_size(campaign/'packets')
                last_artifact_check=time.monotonic()
            for identity,item in list(active.items()):
                child=item['child'];job=item['job'];root=Path(job['root'])
                status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else None
                if status and item['clock'] is None and os.name=='nt' and status['parent_pid'] in (child.pid,os.getpid()):
                    try:item['clock']=ProcessClock(status['pid'])
                    except OSError:pass
                if child.poll() is None:
                    seconds=item['clock'].seconds() if item['clock'] else 0
                    if seconds>job['cpu_cap_seconds']+5 or time.monotonic()-item['started']>job['wall_cap_seconds']:
                        # Only this owned child tree; retain every partial output.
                        if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                        else:child.terminate()
                    continue
                item['log'].close()
                if item.get('adopted'):child.close()
                if status:
                    p=campaign/'attempts'/(status['attempt']+'.json');receipt=read(p)
                    if item['clock']:receipt['native_cpu_seconds']=item['clock'].seconds();item['clock'].close()
                    if receipt['state']=='running':receipt.update(state='native_failed',uncertainty_cpu_seconds=min(job['cpu_cap_seconds'],time.monotonic()-item['started']))
                    write(p,receipt,immutable=False)
                done=(root/'COMPLETE.json').exists()
                write(campaign/'completions'/(identity+'.json'),dict(at=now(),exit_code=child.returncode,executed=done))
                if not done:
                    write(root/'DISPOSITION.json',dict(state='INSTRUMENT FAILED',reason='worker failed or resource limit',exit_code=child.returncode))
                    event(campaign,identity,'batch_failed_or_cutoff',job=identity)
                elif job.get('review_boundary'):event(campaign,identity,'batch_complete',job=identity)
                if job.get('hold_after'):(campaign/'HOLD').touch()
                del active[identity]
            write(campaign/'SERVICE_COST.json',dict(cpu_seconds=service_base+time.process_time()-service_start,segment_cpu_seconds=time.process_time()-service_start,previous_service_cpu_seconds=service_base,pid=os.getpid(),at=now()),immutable=False)
            if (campaign/'SUPERVISOR_STOP').exists() and not active:
                write(campaign/'QUEUE-STATUS.json',dict(pid=os.getpid(),heartbeat=now(),state='owner_stopped_for_handoff',active=[]),immutable=False);return
            if cutoff and not active:
                event(campaign,'science-cutoff','resource_cutoff')
                write(campaign/'QUEUE-STATUS.json',dict(pid=os.getpid(),heartbeat=now(),state='science_closed',active=[]),immutable=False);return
            queue=read(campaign/'QUEUE.json');jobs=queue['jobs']
            if len({j['root'] for j in jobs})!=len(jobs):raise ValueError('duplicate packet root')
            complete={j['id'] for j in jobs if (Path(j['root'])/'COMPLETE.json').exists()}
            pending=[j for j in jobs if j['id'] not in complete and j['id'] not in active and not (Path(j['root'])/'DISPOSITION.json').exists()]
            ready=[j for j in pending if set(j.get('requires',[]))<=complete and current>=datetime.fromisoformat(j['not_before'])]
            ready.sort(key=lambda j: not j.get('reserve_eligible',False) if current>=datetime.fromisoformat(a['confirmation_start']) else False)
            if not cutoff and not (campaign/'HOLD').exists():
                for job in ready:
                    if (campaign/'SUPERVISOR_STOP').exists():break
                    if not job.get('reserve_eligible') and current>=datetime.fromisoformat(a['discovery_freeze']):continue
                    if len(active)>=a['workers']:break
                    if artifact_bytes+(len(active)+1)*4*2**30>a['artifact_ceiling_bytes']:break
                    if available_memory()<a['free_memory_reserve_bytes']+a['worker_memory_ceiling_bytes']:break
                    used=spent(campaign)+a['service_setup_cpu_bound_seconds'];reserved=sum(v['job']['cpu_cap_seconds'] for v in active.values())
                    cap=a['cumulative_cpu_ceiling_seconds'] if job.get('reserve_eligible') else a['exploratory_cpu_ceiling_seconds']
                    if not budget_fits(used,reserved,cap,job):continue
                    if current.timestamp()+job['wall_cap_seconds']>datetime.fromisoformat(a['report_start']).timestamp():continue
                    root=Path(job['root'])
                    if file_digest(root/'PLAN.json')!=job['plan_sha256']:raise ValueError('plan changed')
                    log=(root/'worker.log').open('ab',buffering=0)
                    child=subprocess.Popen([str(python),'-B','-m','runners.run_v21','--root',str(root),'--campaign',str(campaign)],cwd=job['source'],
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
                    active[job['id']]=dict(child=child,job=job,log=log,clock=None,started=time.monotonic())
            state='running' if active else 'awaiting_review_refill'
            write(campaign/'QUEUE-STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=now(),state=state,
                gear=2,workers=a['workers'],active=[dict(id=k,child_pid=v['child'].pid) for k,v in active.items()],
                completed=len(complete),pending=len(pending),ready=len(ready)),immutable=False)
            if not active and not (campaign/'HOLD').exists():event(campaign,'refill-'+digest(queue)[:16],'queue_needs_refill')
            time.sleep(5)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--python',type=Path,required=True);p.add_argument('--recover',type=Path);a=p.parse_args()
    try:supervise(a.campaign,a.python,a.recover)
    except BaseException as e:
        event(a.campaign,'supervisor-failure-'+str(os.getpid()),'supervisor_failed',error=str(e));raise
