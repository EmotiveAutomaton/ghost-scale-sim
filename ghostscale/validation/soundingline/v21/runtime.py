"""Source-bound V21 packets with per-attempt accounting and cooperative limits."""
import os,time,uuid,shutil,threading
from pathlib import Path
from datetime import datetime,timezone
from ..v16.runtime import local_owner
from ..v18_3.io import read,write,file_digest,now
from ..v18_4.priority import below_normal
from ..v20.runtime import memory,available_memory,fingerprint

REPO=Path(__file__).resolve().parents[4]

def spent(campaign):
    return sum(max(r.get('cpu_seconds',0),r.get('native_cpu_seconds',0),r.get('uncertainty_cpu_seconds',0))
        for r in (read(p) for p in (campaign/'attempts').glob('*.json')))

def scientific_files(root):
    return {p.relative_to(root).as_posix():file_digest(p) for p in sorted(root.rglob('*'))
        if p.is_file() and p.name not in ('PLAN.json','SOURCE.zip','STATUS.json','COMPLETE.json','worker.log','OWNER.lock','COST.json')}

def run(root,campaign):
    with local_owner(root):
        if (root/'COMPLETE.json').exists() or (root/'STATUS.json').exists():raise ValueError('never overwrite an executed identity')
        plan=read(root/'PLAN.json');a=read(campaign/'ACCEPTANCE.json');d=plan['design']
        from .checker import validate_design
        validate_design(d)
        if file_digest(campaign/'ACCEPTANCE.json')!=plan['acceptance_sha256']:raise ValueError('acceptance changed')
        if plan['environment']!=fingerprint():raise ValueError('environment changed')
        for name,h in plan['sources'].items():
            if file_digest(REPO/name)!=h:raise ValueError('frozen source changed: '+name)
        if file_digest(root/'SOURCE.zip')!=plan['archive_sha256']:raise ValueError('capsule changed')
        below_normal();start=time.monotonic();cpu=time.process_time();attempt=uuid.uuid4().hex
        lock=threading.RLock();stop=threading.Event();state={'phase':'starting'}
        def pulse(**updates):
            with lock:
                state.update(updates);m=memory();used=time.process_time()-cpu
                write(root/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=now(),attempt=attempt,**m,**state),immutable=False)
                write(campaign/'attempts'/f'{attempt}.json',dict(job=root.name,state=state.get('state','running'),
                    cpu_seconds=used,wall_seconds=time.monotonic()-start,peak_bytes=m['peak_bytes']),immutable=False)
                if state.get('state') in ('complete','failed'):return
                if (campaign/'STOP').exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(a['report_start']):raise TimeoutError('science cutoff')
                if used>d['cpu_cap_seconds']:raise TimeoutError('packet CPU cap')
                if m['working_bytes']>a['worker_memory_ceiling_bytes'] or available_memory()<a['free_memory_reserve_bytes']:raise MemoryError('memory reserve')
                if shutil.disk_usage(campaign).free<a['minimum_free_disk_bytes']:raise OSError('disk reserve')
                if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>d.get('artifact_cap_bytes',4*2**30):raise OSError('packet artifact allowance')
        # The supervisor also checks hard elapsed limits; this heartbeat proves liveness
        # through long numerical calls without asserting scientific progress.
        def beat():
            while not stop.wait(30):
                try:pulse()
                except BaseException:return
        pulse();timer=threading.Thread(target=beat,daemon=True);timer.start()
        try:
            if d['handler']=='retained-audit':
                from .audit import opening
                summary=opening(root,d,pulse)
            else:
                from .studies import run_study
                summary=run_study(root,d,pulse)
            if not all(summary['controls'].values()):raise ValueError('instrument controls failed')
            write(root/'SUMMARY.json',summary)
            write(root/'COMPLETE.json',dict(at=now(),plan_sha256=file_digest(root/'PLAN.json'),files=scientific_files(root),
                state='executed; numerical acceptance and scientific interpretation require review'))
            pulse(state='complete')
        except BaseException as e:
            pulse(state='failed',error=type(e).__name__+': '+str(e));raise
        finally:stop.set();timer.join()
