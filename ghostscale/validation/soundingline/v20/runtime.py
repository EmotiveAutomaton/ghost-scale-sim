"""V20 adapter for the established immutable-packet native queue contract."""
from datetime import datetime,timezone
from pathlib import Path
import ctypes,os,platform,shutil,time,uuid,zipfile
import numpy as np
from ..v16.runtime import local_owner
from ..v18_3.io import read,write,file_digest,now
from ..v18_4.priority import below_normal

REPO=Path(__file__).resolve().parents[4]
def fingerprint():return dict(python=platform.python_version(),numpy=np.__version__)
def source_files():
    from ..v18_4.runtime import source_files as inherited
    names=inherited()+['runners/watch_v18_4_events.py','runners/run_v20.py','runners/watch_v20.py','runners/replay_v20.py']
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'ghostscale/validation/soundingline/v20').glob('*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'tests').glob('test_v20*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'runners').glob('*v20*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'ghostscale/validation/soundingline/v19').glob('*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'docs/versions/v20-contribution-reconstruction').glob('*') if p.is_file()]
    return sorted(set(names))

def freeze(source,archive,admission):
    files={n:file_digest(REPO/n) for n in source_files()}
    if not admission['passed'] or admission['sources']!=files:raise ValueError('current controls required')
    source.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in files:
            p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/name).read_bytes());z.write(p,name)
    return files

def consumed(campaign):
    records=[read(p) for p in (campaign/'attempts').glob('*.json')]
    if any(r['state']=='running' for r in records):raise ValueError('unreconciled live attempt')
    return sum(max(r['cpu_seconds'],r.get('native_cpu_seconds',0),r.get('uncertainty_cpu_seconds',0))+r.get('child_cpu_seconds',0) for r in records)

def memory():
    if os.name!='nt':return dict(working_bytes=0,peak_bytes=0)
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['peak','working','paged_peak','paged','nonpaged_peak','nonpaged','pagefile','pagefile_peak','private']]
    c=Counters();c.cb=ctypes.sizeof(c)
    ps=ctypes.WinDLL('psapi');ps.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p
    if not ps.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):raise OSError('cannot measure worker memory')
    return dict(working_bytes=c.working,peak_bytes=c.peak)

def available_memory():
    if os.name!='nt':return 2**63
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    value=Status();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):raise OSError('cannot check host memory')
    return value.available

def scientific_files(root):
    return {p.relative_to(root).as_posix():file_digest(p) for p in sorted(root.rglob('*')) if p.is_file() and
        p.name not in ('PLAN.json','SOURCE.zip','STATUS.json','COMPLETE.json','worker.log','OWNER.lock','FIT_COST.json') and 'cost' not in p.parts}

def verify_complete(root):
    record=read(root/'COMPLETE.json')
    if record['plan_sha256']!=file_digest(root/'PLAN.json'):raise ValueError('completion plan differs')
    for name,h in record['files'].items():
        if file_digest(root/name)!=h:raise ValueError('completed evidence changed')
    return record

def run(root,campaign):
    with local_owner(campaign/'scientific-worker-owner'),local_owner(root):
        plan=read(root/'PLAN.json');a=read(campaign/'ACCEPTANCE.json');d=plan['design']
        if (root/'COMPLETE.json').exists():return verify_complete(root)
        if (root/'STATUS.json').exists():raise ValueError('retained attempt requires disposition; never overwrite partial evidence')
        if plan['environment']!=fingerprint():raise ValueError('environment changed')
        if any(file_digest(REPO/n)!=h for n,h in plan['sources'].items()):raise ValueError('frozen source changed')
        if file_digest(root/'SOURCE.zip')!=plan['source_archive_sha256']:raise ValueError('source capsule changed')
        validate_design(d)
        below_normal();prior=consumed(campaign);start=time.monotonic();cpu=time.process_time();attempt=uuid.uuid4().hex
        prior_bytes=sum(p.stat().st_size for p in Path(a['artifact_root']).rglob('*') if p.is_file())
        cap=a['cumulative_cpu_ceiling_seconds'] if d.get('reserve_eligible') else a['exploratory_cpu_ceiling_seconds']
        def emit(state,**more):
            m=memory();elapsed=time.monotonic()-start
            write(root/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),attempt=attempt,heartbeat=now(),state=state,**m,**more),immutable=False)
            write(campaign/'attempts'/f'{attempt}.json',dict(packet=root.name,accounting_card=root.name,state=state,
                cpu_seconds=time.process_time()-cpu,wall_seconds=elapsed,peak_bytes=m['peak_bytes']),immutable=False)
        def pulse(**more):
            used=time.process_time()-cpu;m=memory()
            if (campaign/'STOP').exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(a['report_start']):raise TimeoutError('fixed science cutoff')
            if prior+used>=cap or used>=d['cpu_cap_seconds']:raise TimeoutError('CPU admission exhausted')
            if m['working_bytes']>a['worker_memory_ceiling_bytes']:raise MemoryError('worker memory ceiling')
            if available_memory()<16*2**30:raise MemoryError('host memory reserve')
            if shutil.disk_usage(campaign).free<a['minimum_free_disk_bytes']:raise OSError('free-space reserve')
            if prior_bytes+sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>a['artifact_ceiling_bytes']:raise OSError('artifact budget')
            emit('running',**more)
        emit('running')
        try:
            pulse(phase='begin')
            from .studies import run_study
            summary=run_study(root,d,pulse)
            if not all(summary['controls'].values()):raise ValueError('failed ruler control')
            write(root/'SUMMARY.json',summary)
            write(root/'EVIDENCE_ROLES.json',dict(reader='reader/',training='training/',evaluator=['evaluator/','raw/','CASEBOOK_EVALUATOR.json','SUMMARY.json','RETENTION_ROWS.json'],source='SOURCE.zip',
                warning='Only reader/ is a blind input; forecast/truth archives and casebook are evaluator evidence'))
            pulse(phase='complete')
            write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),files=scientific_files(root),
                state='executed-with-independent-arithmetic; scientific review pending',at=now()))
            emit('complete')
        except BaseException as e:
            emit('failed',error=type(e).__name__+': '+str(e));raise

def validate_design(d):
    if d['branch'] not in ('G0','G1','G2','G3','G4','G5','G6','G7'):raise ValueError('unimplemented branch')
    if set(d['lineages'])&set(d.get('training_lineages',[])):raise ValueError('lineage leakage')
    if len(d['lineages'])!=len(set(d['lineages'])):raise ValueError('duplicate lineages')
    if d['branch']=='G7' and d.get('split')!='confirmation':raise ValueError('confirmation split required')
    if d.get('epochs',120)!=120:raise ValueError('scientific epoch setting changed')
    if d.get('history',0) not in (0,1,4,16,64,256):raise ValueError('unregistered history')
    if d.get('train_cases',128)<128 or d.get('test_cases',32)<32:raise ValueError('undersized declared block')
    if d['cpu_cap_seconds']<=0:raise ValueError('missing cost cap')
    return True
