"""V26 native execution: source-bound three-pass packets and finite two-slot queue."""
import argparse,ctypes,hashlib,json,math,os,shutil,subprocess,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v26 import contract,registry,study
from runners.watch_v23_events import event
REPO=Path(__file__).resolve().parents[1]
def now():return datetime.now(timezone.utc)

def env_for(source):
    env=dict(os.environ,PYTHONPATH=str(source),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    return env

def prepare(campaign,family,kind='probe',*,parent=None,source=None,variant=None):
    started=time.process_time();source=Path(source) if source else frozen_source(campaign)
    if kind not in ('probe','study','texture','texture_probe'):raise ValueError('new healing/followup needs separately frozen plan')
    identity=f'v26-{family.lower()}-{kind}-1' if variant is None else f'v26-{family.lower()}-texture-{variant+1:03d}'+('-probe' if kind=='texture_probe' else '')
    packet=campaign/'packets'/identity
    if packet.exists():raise ValueError('packet already exists; inspect before recovery')
    protocol=read(campaign/'protocols'/(family+'.json'));check=read(campaign/'designs'/(family+'.json'))
    if not check.get('passed'):raise ValueError('design cannot produce the case')
    controls=study.module(family).controls()
    if not controls.get('passed'):raise ValueError('known-answer failed')
    rows=registry.settings(family);blocks=1 if kind in ('probe','texture_probe') else 16;shed=0
    if kind in ('probe','texture_probe'):
        if kind=='texture_probe':
            if not parent or not (Path(parent)/'COMPLETE.json').exists():raise ValueError('texture requires verified primary')
            rows=[registry.texture_variants(family)[variant]]
        rows=[rows[0],rows[-1]] if len(rows)>1 else [rows[0],dict(rows[0],reference_offset=15)]
        if kind=='texture_probe':rows=[registry.texture_variants(family)[variant]]*2
        caps=dict(original=600.,replay=600.,review=600.);wall_projection=3600.
    else:
        parent=Path(parent)
        if kind=='texture' and read(parent/'PLAN.json').get('scientific_kind')!='probe':raise ValueError('texture requires its own measured excluded probe')
        c=read(parent/'COMPLETE.json')
        for k,n in [('plan_sha256','PLAN.json'),('summary_sha256','original/SUMMARY.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
            if c[k]!=digest(parent/n):raise ValueError('verified parent binding changed')
        # Primary studies use measured probe costs. A texture's excluded two-pass
        # probe is a separately frozen texture-probe plan (same shared pool).
        costs=read(parent/'PASS_COSTS.json');unit={k:max(max(v),1/64) for k,v in costs['setting_seconds'].items()}
        overhead={k:max(0,costs['pass_cpu_seconds'][k]-sum(costs['setting_seconds'][k])) for k in unit}
        if kind=='texture':rows=[registry.texture_variants(family)[variant]]
        def price(rows):return {k:1.5*(overhead[k]+unit[k]*len(rows)*blocks)+10 for k in unit}
        caps=price(rows)
        while kind=='study' and sum(caps.values())+charges(campaign).get('family:'+family,0)>registry.FAMILY_HOURS[family]*3600 and shed<3:
            nxt=registry.selected_settings(family,shed+1)
            if nxt==rows:break
            rows=nxt;shed+=1;caps=price(rows)
        measured=read(campaign/'accounting'/(parent.name+'.json'))
        wall_projection=1.5*measured['wall_seconds']*len(rows)*blocks/max(1,len(read(parent/'PLAN.json')['settings']))
    packet.mkdir(parents=True)
    write(packet/'CONTROLS_BEFORE_DATA.json',controls)
    plan=dict(family=family,kind='texture' if kind=='texture_probe' else kind,scientific_kind='probe' if kind in ('probe','texture_probe') else kind,pool='studies' if kind in ('probe','study') else 'followup_healing',identity=identity,science_identity=identity,settings=rows,blocks=blocks,shed_level=shed,
        source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),source_archive_sha256=digest(source/'SOURCE.zip'),
        acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),environment_sha256=digest(campaign/'ENVIRONMENT.json'),
        protocol_sha256=digest(campaign/'protocols'/(family+'.json')),design_check_sha256=digest(campaign/'designs'/(family+'.json')),
        numerical_threads=1,priority='below-normal',controls_sha256=digest(packet/'CONTROLS_BEFORE_DATA.json'),cpu_cap_seconds=sum(caps.values()),pass_caps=caps,
        wall_cap_seconds=18*3600,projected_wall_seconds=wall_projection,interpreter=str(Path(sys.executable).resolve()),parent=parent.name if parent else None,confirmation=False,texture_variant=variant)
    if parent:plan['parent_complete_sha256']=digest(parent/'COMPLETE.json')
    write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='PREPARED',at=now().isoformat()))
    write(campaign/'accounting'/(identity+'-preparation.json'),dict(pool='preparation',kind='preparation',family=family,cpu_seconds=time.process_time()-started,state='COMPLETE'))
    return packet

def verify_admission_bindings(campaign,packet,plan):
    verify_source(plan)
    for key,name in [('acceptance_sha256','ACCEPTANCE.json'),('environment_sha256','ENVIRONMENT.json'),('protocol_sha256','protocols/'+plan['family']+'.json'),('design_check_sha256','designs/'+plan['family']+'.json')]:
        if digest(campaign/name)!=plan[key]:raise ValueError('admission binding changed: '+name)
    if digest(packet/'CONTROLS_BEFORE_DATA.json')!=plan['controls_sha256'] or not read(packet/'CONTROLS_BEFORE_DATA.json')['passed']:raise ValueError('controls changed')
    protocol=read(campaign/'protocols'/(plan['family']+'.json'))
    module=Path(plan['source_root'])/'ghostscale/validation/soundingline/v26'/f'{registry.MODULES[plan["family"]]}.py'
    if digest(module)!=protocol['module_sha256']:raise ValueError('protocol implementation binding changed')

def outside_scientists(campaign):
    from runners.watch_v23_events import alive
    total=0
    for root in (campaign.parent/'v25',campaign.parent/'v25/n16-robustness-20261030-r1',campaign.parent/'v24-n16-production'):
        for p in (root/'packets').glob('*/STATUS.json'):
            s=read(p);pid=s.get('parent_pid') if s.get('state')=='RUNNING' else s.get('launcher_pid') if s.get('state')=='LAUNCHING' else None
            if pid and alive(pid):total+=1
    return total

def charges(campaign):
    totals={}
    for p in (campaign/'accounting').glob('*.json'):
        row=read(p);value=max(row.get('cpu_seconds',0),row.get('reserved_cpu_seconds',0))
        pool=row.get('pool','preparation');totals[pool]=totals.get(pool,0)+value
        if pool=='studies':
            key='family:'+row.get('family','none');totals[key]=totals.get(key,0)+value
        if row.get('kind') in ('followup','healing','texture'):totals[row['kind']]=totals.get(row['kind'],0)+value
    return totals

def frozen_source(campaign,extra_sources=()):
    names=sorted([p.relative_to(REPO).as_posix() for p in (REPO/'ghostscale').rglob('*.py') if '__pycache__' not in p.parts]+['runners/run_v26.py','runners/watch_v26.py','runners/watch_v23_events.py','runners/launch_background.py']+list(extra_sources))
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
    # Timing boundaries surround complete settings, including their first block.
    if label=='review':write(packet/'INDEPENDENT_REVIEW.json',study.reconstruct(packet/'original',plan,pulse))
    else:study.execute(packet/label,plan,pulse)
    write(packet/(label+'-cost.json'),dict(cpu_seconds=time.process_time()-started,blocks=timed))

def execute(packet):
    if os.name!='nt':raise RuntimeError('native Windows accounting required')
    packet=packet.resolve();plan=read(packet/'PLAN.json');campaign=Path(plan.get('accounting_root',packet.parent.parent))
    if not (packet/'ADMISSION.json').exists():raise ValueError('no owner admission')
    if digest(packet/'PLAN.json')!=read(packet/'ADMISSION.json')['plan_sha256']:raise ValueError('admitted plan changed')
    if (packet/'COMPLETE.json').exists():raise ValueError('completed packet cannot repeat')
    verify_source(plan);started=time.process_time();cpu=0.;pass_cpu={};active=None;handle=None;state='FAILED';error=None;wall=time.monotonic()
    try:
        for label in ('original','replay','review'):
            env=dict(os.environ,PYTHONPATH=str(plan['source_root']),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
            with (packet/(label+'.log')).open('xb') as log:
                active=subprocess.Popen([plan['interpreter'],'-B','-m','runners.run_v26','--packet',str(packet),'--child',label],cwd=plan['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
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
        timing={}
        for label in pass_cpu:
            blocks=read(packet/(label+'-cost.json'))['blocks'];timing[label]=[sum(b['cpu_seconds'] for b in blocks if b['setting']==s) for s in range(len(plan['settings']))]
        write(packet/'PASS_COSTS.json',dict(pass_cpu_seconds=pass_cpu,setting_seconds=timing,includes_world_io=True))
        if plan.get('healing_parent'):
            parent=packet.parent/plan['healing_parent'];retained={}
            for path in sorted((parent/'original').glob('*_points.json*')):
                bound=digest(path)
                if plan.get('repair_observations_only'):
                    import gzip
                    with gzip.open(path,'rt',encoding='utf-8') as f:before=json.load(f)
                    with gzip.open(packet/'original'/path.name,'rt',encoding='utf-8') as f:after=json.load(f)
                    for value in (before,after):
                        value['data'].pop('readouts')
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


def advance(campaign,packet):
    plan=read(packet/'PLAN.json');family=plan['family'];kind=plan['kind'];source=Path(plan['source_root'])
    if kind=='probe':return prepare(campaign,family,'study',parent=packet,source=source)
    if kind=='texture' and plan['scientific_kind']=='probe':return prepare(campaign,family,'texture',parent=packet,source=source,variant=plan['texture_variant'])
    from ghostscale.validation.soundingline.v26.reporting import land_local,texture_decision
    land_local(campaign,packet,Path(read(campaign/'LOCAL_PATHS.json')['repo']))
    if kind=='study':
        write(campaign/'texture'/(family+'.json'),dict(consecutive=0,completed_variants=[],state='OPEN' if registry.texture_variants(family) else 'NOT_DECLARED',primary=packet.name),immutable=False)
        if registry.texture_variants(family):return prepare(campaign,family,'texture_probe',parent=packet,source=source,variant=0)
    elif kind=='texture':
        state=texture_decision(campaign,packet)
        if state['state']=='OPEN':return prepare(campaign,family,'texture_probe',parent=campaign/'packets'/state['primary'],source=source,variant=plan['texture_variant']+1)


def queue(campaign):
    if os.name!='nt':raise RuntimeError('native Windows queue required')
    accepted=read(campaign/'ACCEPTANCE.json');started=time.process_time();active={};finished=[];cost=campaign/'accounting'/f'controller-{os.getpid()}-{time.time_ns()}.json'
    lock=campaign/'QUEUE_OWNER.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        while True:
            for name,(proc,log) in list(active.items()):
                if proc.poll() is None:continue
                log.close();del active[name];packet=campaign/'packets'/name;status=read(packet/'STATUS.json')
                if status['state']=='COMPLETE':
                    event(campaign,name+'-complete','batch_complete',dict(packet=name,complete_sha256=digest(packet/'COMPLETE.json')))
                    try:advance(campaign,packet)
                    except Exception as exc:
                        write(packet/'CONTINUATION_FAILURE.json',dict(error=f'{type(exc).__name__}: {exc}'))
                        event(campaign,name+'-continuation-failure','admission_failure',dict(packet=name,error=str(exc)))
                else:
                    if status['state']!='FAILED':write(packet/'STATUS.json',dict(state='FAILED',reason='parent exited without terminal receipt; full reservation retained',exit_code=proc.returncode),immutable=False)
                    event(campaign,name+'-failed','packet_failure',dict(packet=name,status_sha256=digest(packet/'STATUS.json')))
                finished.append(name)
            candidates=[]
            for path in (campaign/'packets').glob('*/PLAN.json'):
                packet=path.parent
                if read(packet/'STATUS.json')['state']=='PREPARED':
                    p=read(path);candidates.append((p['kind']=='texture',contract.ORDER.index(p['family']),p.get('scientific_kind')=='probe',packet,p))
            stopping=(campaign/'STOP').exists() or now()>=datetime.fromisoformat(contract.NEW_STARTS_CLOSE)
            if not stopping and not (campaign/'PAUSE_ADMISSIONS').exists() and now()>=datetime.fromisoformat(contract.SCIENCE_OPEN):
                for _,__,___,packet,plan in sorted(candidates):
                    outside=outside_scientists(campaign)
                    if len(active)+outside>=2:break
                    try:
                        contract.admission(plan,accepted,now(),charges(campaign),running=len(active)+outside);verify_admission_bindings(campaign,packet,plan)
                    except ValueError as exc:
                        write(packet/'STATUS.json',dict(state='ADMISSION_HELD',reason=str(exc),at=now().isoformat()),immutable=False)
                        event(campaign,packet.name+'-held','admission_failure',dict(packet=packet.name,reason=str(exc)));continue
                    if (packet/'ADMISSION.json').exists():raise ValueError('previous admission requires recovery review')
                    write(packet/'ADMISSION.json',dict(at=now().isoformat(),plan_sha256=digest(packet/'PLAN.json'),charges_before=charges(campaign)))
                    write(campaign/'accounting'/(packet.name+'.json'),dict(pool=plan['pool'],kind=plan['kind'],family=plan['family'],state='RESERVED',reserved_cpu_seconds=plan['cpu_cap_seconds']))
                    log=(packet/'parent.log').open('xb');proc=subprocess.Popen([plan['interpreter'],'-B','-m','runners.run_v26','--packet',str(packet)],cwd=plan['source_root'],env=env_for(plan['source_root']),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                    active[packet.name]=(proc,log);write(packet/'STATUS.json',dict(state='LAUNCHING',launcher_pid=proc.pid,at=now().isoformat()),immutable=False)
            state='RUNNING' if active else 'WAITING_FOR_OPENING' if now()<datetime.fromisoformat(contract.SCIENCE_OPEN) else 'WAITING_FOR_PREPARED_COMMISSIONED_WORK'
            write(campaign/'QUEUE_STATUS.json',dict(state=state,pid=os.getpid(),heartbeat=now().isoformat(),active=list(active),prepared=[p.name for _,__,___,p,plan in candidates],finished=finished,gear=2),immutable=False)
            write(cost,dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
            from ghostscale.validation.soundingline.v26.reporting import status_export
            status_export(campaign,Path(read(campaign/'LOCAL_PATHS.json')['repo']))
            if not active and (stopping or ((campaign/'BUILD_COMPLETE.json').exists() and not candidates)):
                event(campaign,'v26-queue-drained','review_boundary',dict(finished=finished,stop=stopping));break
            time.sleep(10)
    finally:
        write(campaign/'QUEUE_STATUS.json',dict(state='STOPPED_WITH_ACTIVE_CHILDREN' if active else 'DRAINED',pid=os.getpid(),active=list(active),at=now().isoformat()),immutable=False)
        write(cost,dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='STOPPED'),immutable=False)
        lock.unlink()


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path);p.add_argument('--prepare',choices=contract.ORDER);p.add_argument('--queue',action='store_true');p.add_argument('--packet',type=Path);p.add_argument('--child',choices=('original','replay','review'))
    a=p.parse_args()
    if a.prepare:print(prepare(a.campaign.resolve(),a.prepare))
    elif a.queue:queue(a.campaign.resolve())
    elif a.child:child(a.packet.resolve(),a.child)
    elif a.packet:execute(a.packet.resolve())
    else:p.error('choose prepare, queue or packet')

if __name__=='__main__':main()

