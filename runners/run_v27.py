"""V27 native source-bound three-pass execution; allocation is enforced by queue_v27."""
import argparse,ctypes,hashlib,json,math,os,shutil,subprocess,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v27 import contract,registry,study
from runners.watch_v23_events import event
REPO=Path(__file__).resolve().parents[1]
def now():return datetime.now(timezone.utc)

def env_for(source):
    env=dict(os.environ,PYTHONPATH=str(source),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    return env

def frozen_source(campaign,extra_sources=()):
    names=sorted([p.relative_to(REPO).as_posix() for p in (REPO/'ghostscale').rglob('*.py') if '__pycache__' not in p.parts]+['runners/run_v27.py','runners/watch_v23_events.py','runners/launch_background.py']+list(extra_sources))
    hashes={name:digest(REPO/name) for name in names};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    destination=campaign/'sources'/key
    if not destination.exists():
        destination.mkdir(parents=True)
        for name in names:
            p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,p)
        (destination/'runners/__init__.py').write_text('',encoding='utf-8');hashes['runners/__init__.py']=digest(destination/'runners/__init__.py')
        write(destination/'SOURCE_MANIFEST.json',hashes)
        with zipfile.ZipFile(destination/'SOURCE.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in sorted(hashes):z.write(destination/name,name)
    return destination

def verify_source(plan):
    source=Path(plan['source_root'])
    if digest(source/'SOURCE.zip')!=plan['source_archive_sha256'] or digest(source/'SOURCE_MANIFEST.json')!=plan['source_manifest_sha256']:raise ValueError('source capsule changed')
    for name,bound in read(source/'SOURCE_MANIFEST.json').items():
        if digest(source/name)!=bound:raise ValueError('source byte changed: '+name)

def child(packet,label):
    plan=read(packet/'PLAN.json');campaign=Path(plan.get('accounting_root',packet.parent.parent))
    if digest(campaign/'ACCEPTANCE.json')!=plan['acceptance_sha256']:raise ValueError('acceptance changed')
    runtime.below_normal();runtime.child_handshake(packet,label);verify_source(plan)
    if digest(campaign/'ENVIRONMENT.json')!=plan['environment_sha256'] or runtime.environment()!=read(campaign/'ENVIRONMENT.json'):raise ValueError('scientific environment changed')
    started=time.process_time();last=[0.];setting_costs=[];setting_start=[started];last_setting=[None];timed=[];world_start=[started]
    def pulse(**progress):
        used=time.process_time()-started
        if progress.get('phase')=='world_complete':
            timed.append(dict(setting=progress['setting'],cpu_seconds=time.process_time()-world_start[0]));world_start[0]=time.process_time()
        if used>plan['pass_caps'][label] or now()>=datetime.fromisoformat(contract.DRAIN) or (campaign/'STOP').exists():raise TimeoutError('pass CPU or immutable drain')
        setting=progress.get('setting')
        if setting is not None and last_setting[0]!=setting:
            if last_setting[0] is not None:setting_costs.append(time.process_time()-setting_start[0])
            last_setting[0]=setting;setting_start[0]=time.process_time()
        if time.monotonic()-last[0]>5:
            write(packet/(label+'-progress.json'),dict(pid=os.getpid(),heartbeat=now().isoformat(),cpu_seconds=used,**progress),immutable=False);last[0]=time.monotonic()
    study.verify_extra_environment(campaign,plan)
    implementation=study
    # Timing boundaries surround complete settings, including their first block.
    if label=='review':write(packet/'INDEPENDENT_REVIEW.json',implementation.reconstruct(packet/'original',plan,pulse))
    else:implementation.execute(packet/label,plan,pulse)
    write(packet/(label+'-cost.json'),dict(cpu_seconds=time.process_time()-started,blocks=timed))

def execute(packet):
    if os.name!='nt':raise RuntimeError('native Windows accounting required')
    packet=packet.resolve();plan=read(packet/'PLAN.json');campaign=Path(plan.get('accounting_root',packet.parent.parent))
    if not (packet/'ADMISSION.json').exists():raise ValueError('no owner admission')
    if digest(packet/'PLAN.json')!=read(packet/'ADMISSION.json')['plan_sha256']:raise ValueError('admitted plan changed')
    if (packet/'COMPLETE.json').exists():raise ValueError('completed packet cannot repeat')
    verify_source(plan);started=time.process_time();cpu=0.;pass_cpu={};active=None;handle=None;state='FAILED';error=None;wall=time.monotonic()
    verification_parent=plan.get('verification_healing_parent')
    labels=('replay','review') if verification_parent else ('original','replay','review')
    if verification_parent:
        parent=packet.parent/verification_parent
        if read(parent/'STATUS.json')['state']!='FAILED':raise ValueError('verification healing parent is not failed')
        for key,name in [('retained_original_raw_index_sha256','original/RAW_INDEX.json'),
                         ('retained_original_summary_sha256','original/SUMMARY.json'),
                         ('retained_original_cost_sha256','original-cost.json'),
                         ('failed_status_sha256','STATUS.json')]:
            if digest(parent/name)!=plan[key]:raise ValueError('verification healing parent changed: '+name)
        if digest(packet/'original/RAW_INDEX.json')!=plan['retained_original_raw_index_sha256'] or digest(packet/'original/SUMMARY.json')!=plan['retained_original_summary_sha256'] or digest(packet/'original-cost.json')!=plan['retained_original_cost_sha256']:
            raise ValueError('retained original copy changed')
    try:
        for label in labels:
            env=dict(os.environ,PYTHONPATH=str(plan['source_root']),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
            with (packet/(label+'.log')).open('xb') as log:
                active=subprocess.Popen([plan['interpreter'],'-B','-m','runners.run_v27','--packet',str(packet),'--child',label],cwd=plan['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                handle=runtime.attach_child(active,packet,label)
                while active.poll() is None:
                    used=runtime.child_cpu(handle)
                    write(packet/'STATUS.json',dict(state='RUNNING',phase=label,parent_pid=os.getpid(),child_pid=read(packet/(label+'-owner.json'))['pid'],heartbeat=now().isoformat(),cpu_seconds=cpu+used),immutable=False)
                    if used>plan['pass_caps'][label] or cpu+used+time.process_time()-started>plan['cpu_cap_seconds'] or time.monotonic()-wall>plan['wall_cap_seconds'] or now()>=datetime.fromisoformat(contract.DRAIN) or (campaign/'STOP').exists() or runtime.memory()['available']<2*1024**3:raise TimeoutError('pass/packet resource or drain boundary')
                    time.sleep(2)
                pass_cpu[label]=runtime.child_cpu(handle);cpu+=pass_cpu[label]
                ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle));handle=None
                if active.returncode:raise RuntimeError(f'{label} exit {active.returncode}')
                active=None
            if label=='replay':write(packet/'REPLAY.json',runtime.compare(packet/'original',packet/'replay'))
        receipt_pass_cpu=dict(pass_cpu)
        if verification_parent:receipt_pass_cpu['original']=read(packet/'original-cost.json')['cpu_seconds']
        timing={}
        for label in receipt_pass_cpu:
            blocks=read(packet/(label+'-cost.json'))['blocks'];timing[label]=[sum(b['cpu_seconds'] for b in blocks if b['setting']==s) for s in range(len(plan['settings']))]
        write(packet/'PASS_COSTS.json',dict(pass_cpu_seconds=receipt_pass_cpu,setting_seconds=timing,includes_world_io=True,
            retained_original_unrepeated=bool(verification_parent)))
        if plan.get('healing_parent'):
            parent=packet.parent/plan['healing_parent'];retained={}
            for path in sorted((parent/'original').glob('*_points.json*')):
                bound=digest(path)
                if plan.get('repair_observations_only'):
                    import gzip
                    with gzip.open(path,'rt',encoding='utf-8') as f:before=json.load(f)
                    with gzip.open(packet/'original'/path.name,'rt',encoding='utf-8') as f:after=json.load(f)
                    for value in (before,after):
                        value['data'].pop('measures' if plan.get('analysis_contract')=='v27-block-v1' else 'readouts')
                        for field in plan.get('repair_public_metadata',[]):value['data']['public'].pop(field,None)
                    if before!=after:raise ValueError('derived-measure repair changed retained observations')
                elif digest(packet/'original'/path.name)!=bound:raise ValueError('healing changed retained scientific draw')
                retained[path.name]=bound
            write(packet/'REPAIR_CONTINUITY.json',dict(retained_raw_files=retained,science_identity_unchanged=True,observations_only=bool(plan.get('repair_observations_only'))))
        write(packet/'COMPLETE.json',dict(state='VERIFIED',plan_sha256=digest(packet/'PLAN.json'),summary_sha256=digest(packet/'original/SUMMARY.json'),replay_sha256=digest(packet/'REPLAY.json'),review_sha256=digest(packet/'INDEPENDENT_REVIEW.json'),confirmation=False))
        state='COMPLETE'
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}';raise
    finally:
        if handle is not None:
            if active and active.poll() is None:
                ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(handle),1);ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(handle),10000)
            cpu+=runtime.child_cpu(handle);ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))
        if active and active.poll() is None:active.terminate();active.wait(timeout=10)
        write(campaign/'accounting'/(packet.name+'.json'),dict(pool=plan['pool'],kind=plan['kind'],family=plan['family'],state=state,cpu_seconds=cpu+time.process_time()-started,wall_seconds=time.monotonic()-wall,reserved_cpu_seconds=0 if state=='COMPLETE' else plan['cpu_cap_seconds'],pass_cpu_seconds=pass_cpu,error=error),immutable=False)
        write(packet/'STATUS.json',dict(state=state,at=now().isoformat(),cpu_seconds=cpu+time.process_time()-started,error=error),immutable=False)



def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--child',choices=('original','replay','review'))
    a=p.parse_args()
    if a.child:child(a.packet.resolve(),a.child)
    else:execute(a.packet.resolve())

if __name__=='__main__':main()
