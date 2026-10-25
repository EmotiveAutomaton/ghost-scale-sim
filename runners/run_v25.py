"""Native two-slot V25 queue; measured reservations, source capsules and terminal events."""
import argparse,ctypes,hashlib,json,math,os,shutil,subprocess,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v25 import contract,registry,study
from runners.watch_v23_events import event

REPO=Path(__file__).resolve().parents[1]

def now():return datetime.now(timezone.utc)

def charges(campaign):
    totals={}
    for p in (campaign/'accounting').glob('*.json'):
        row=read(p);value=max(row.get('cpu_seconds',0),row.get('reserved_cpu_seconds',0))
        pool=row.get('pool','preparation');totals[pool]=totals.get(pool,0)+value
        if pool=='studies':
            key='family:'+row.get('family','none');totals[key]=totals.get(key,0)+value
        if row.get('kind')=='followup':totals['followup']=totals.get('followup',0)+value
    return totals

def frozen_source(campaign):
    names=sorted([p.relative_to(REPO).as_posix() for p in (REPO/'ghostscale').rglob('*.py') if '__pycache__' not in p.parts]+['runners/run_v25.py','runners/watch_v25.py','runners/watch_v23_events.py','runners/launch_background.py'])
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

def prepare(campaign,family,kind='probe',*,parent=None,source=None):
    started=time.process_time();family=family.upper()
    if kind!='probe' and parent is None:raise ValueError('study requires verified measured probe')
    source=Path(source) if source else frozen_source(campaign)
    accepted=read(campaign/'ACCEPTANCE.json');identity=f'v25-{family.lower()}-{kind}-1'
    packet=campaign/'packets'/identity;packet.mkdir(parents=True,exist_ok=False)
    controls=study.module(family).controls()
    if not controls.get('passed'):raise ValueError('known-answer controls failed before admission')
    write(packet/'CONTROLS_BEFORE_DATA.json',controls)
    rows=registry.settings(family);blocks=1 if kind=='probe' else registry.card(family)['blocks'];shed=0
    if kind=='probe':
        rows=[rows[0],rows[-1]];caps=dict(original=600.,replay=600.,review=600.);cap=1800.
    else:
        parent=Path(parent)
        if not (parent/'COMPLETE.json').exists():raise ValueError('probe not verified')
        for key,path in [('plan_sha256','PLAN.json'),('summary_sha256','original/SUMMARY.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
            if read(parent/'COMPLETE.json')[key]!=digest(parent/path):raise ValueError('probe completion changed')
        costs=read(campaign/'accounting'/(parent.name+'.json'))['pass_cpu_seconds']
        # Most expensive of the two sampled cells, measured separately in every pass.
        probes=read(parent/'PASS_COSTS.json')
        # Windows process CPU is quantized. Zero observed ticks are bounded by
        # one tick, not evidence that a completed probe had no measured work.
        unit={label:max(max(values),1/64) for label,values in probes['setting_seconds'].items()}
        inner={k:read(parent/(k+'-cost.json'))['cpu_seconds'] for k in costs}
        if not probes.get('includes_world_io'):
            # The whole two-world inner pass includes serialization/review;
            # use its complete cost plus one tick as an upper bound for either
            # single world, without replicating process startup per world.
            unit={k:max(unit[k],inner[k]+1/64) for k in unit}
        overhead={k:max(0.,costs[k]-(sum(probes['setting_seconds'][k]) if probes.get('includes_world_io') else inner[k])) for k in costs}
        projected=sum(overhead.values())+sum(unit.values())*len(rows)*blocks
        while projected>registry.FAMILY_HOURS[family]*3600 and shed<3:
            candidate=registry.selected_settings(family,shed+1)
            if candidate==rows:break
            rows=candidate;shed+=1
            if (family=='W4' and shed>=3) or (family=='W5' and shed>=2):blocks=8
            projected=sum(overhead.values())+sum(unit.values())*len(rows)*blocks
        # Entry/hash overhead is measured separately and occurs once per pass.
        caps={k:1.5*(overhead[k]+v*len(rows)*blocks)+10 for k,v in unit.items()};cap=sum(caps.values())
    plan=dict(family=family,kind=kind,pool='studies',identity=identity,science_identity=identity,settings=rows,blocks=blocks,
        shed_level=shed,source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),source_archive_sha256=digest(source/'SOURCE.zip'),
        acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),environment_sha256=digest(campaign/'ENVIRONMENT.json'),
        protocol_sha256=digest(campaign/'protocols'/(family+'.json')),numerical_threads=1,priority='below-normal',
        controls_sha256=digest(packet/'CONTROLS_BEFORE_DATA.json'),
        cpu_cap_seconds=cap,pass_caps=caps,wall_cap_seconds=18*3600,interpreter=str(Path(sys.executable).resolve()),
        parent=parent.name if parent else None,production=False,confirmation=False)
    if parent:plan['parent_complete_sha256']=digest(parent/'COMPLETE.json')
    write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='PREPARED',at=now().isoformat()))
    write(campaign/'accounting'/(identity+'-preparation.json'),dict(pool='preparation',kind='preparation',family=family,cpu_seconds=time.process_time()-started,state='COMPLETE'))
    return packet

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
    if plan.get('production'):
        from ghostscale.validation.soundingline.v24 import production
        tick=time.process_time()
        if label=='review':write(packet/'INDEPENDENT_REVIEW.json',production.reconstruct(packet/'original',plan,pulse))
        else:production.execute(packet/label,plan,pulse)
        write(packet/(label+'-cost.json'),dict(cpu_seconds=time.process_time()-started,blocks=[dict(setting=0,cpu_seconds=time.process_time()-tick)]))
        return
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
                active=subprocess.Popen([plan['interpreter'],'-B','-m','runners.run_v25','--packet',str(packet),'--child',label],cwd=plan['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
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

def queue(campaign):
    if os.name!='nt':raise RuntimeError('native queue required')
    accepted=read(campaign/'ACCEPTANCE.json');started=time.process_time();active={};finished=[]
    with (campaign/'QUEUE_OWNER.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        from ghostscale.validation.soundingline.v24 import production
        production.reconcile(campaign)
        while now()<datetime.fromisoformat(contract.DRAIN) and not (campaign/'STOP').exists():
            for name,(proc,log) in list(active.items()):
                if proc.poll() is None:continue
                log.close();del active[name];packet=campaign/'packets'/name;status=read(packet/'STATUS.json');plan=read(packet/'PLAN.json')
                if status['state']=='COMPLETE':
                    event(campaign,name+'-complete','batch_complete',dict(packet=name,complete_sha256=digest(packet/'COMPLETE.json')))
                    if plan.get('production'):
                        try:production.advance(campaign,packet)
                        except Exception as exc:
                            write(packet/'CONTINUATION_FAILURE.json',dict(error=f'{type(exc).__name__}: {exc}'))
                            event(campaign,name+'-continuation-failure','admission_failure',dict(packet=name,error=str(exc)))
                    elif plan['kind']=='probe' or plan.get('scientific_kind')=='probe':
                        try:
                            # New families may not exist in the controller's earlier capsule.
                            # Preparation must import the exact probe's scientific source.
                            env=dict(os.environ,PYTHONPATH=plan['source_root'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
                            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
                            code='from pathlib import Path; import sys; from runners.run_v25 import prepare; prepare(Path(sys.argv[1]),sys.argv[2],"study",parent=Path(sys.argv[3]),source=Path(sys.argv[4]))'
                            with (packet/'study-preparation.log').open('xb') as output:
                                subprocess.run([plan['interpreter'],'-B','-c',code,str(campaign),plan['family'],str(packet),plan['source_root']],cwd=plan['source_root'],env=env,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=300,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                        except Exception as exc:
                            write(packet/'STUDY_ADMISSION_FAILURE.json',dict(error=f'{type(exc).__name__}: {exc}'))
                            event(campaign,name+'-study-admission','admission_failure',dict(packet=name,error=str(exc)))
                else:
                    if status['state'] not in ('FAILED','COMPLETE'):
                        write(packet/'STATUS.json',dict(state='FAILED',at=now().isoformat(),reason='parent exited without a terminal receipt; reservation retained',exit_code=proc.returncode),immutable=False)
                    event(campaign,name+'-failed','packet_failure',dict(packet=name,status_sha256=digest(packet/'STATUS.json')))
                finished.append(name)
            candidates=[]
            for packet in (campaign/'packets').glob('*'):
                if (packet/'PLAN.json').exists() and (packet/'STATUS.json').exists() and read(packet/'STATUS.json')['state']=='PREPARED':
                    p=read(packet/'PLAN.json');candidates.append((len(contract.ORDER) if p.get('production') else contract.ORDER.index(p['family']),p['kind']!='study',packet,p))
            for _,__,packet,plan in sorted(candidates):
                if (campaign/'PAUSE_ADMISSIONS').exists():break
                if len(active)>=2:break
                if not plan.get('production') and now()>=datetime.fromisoformat(contract.NEW_STARTS_CLOSE):continue
                if plan['family']=='W5' and not (campaign/'packets/v25-w1-study-1/COMPLETE.json').exists():continue
                accounting_root=Path(plan.get('accounting_root',campaign))
                try:
                    if plan.get('production'):
                        from ghostscale.validation.soundingline.v24 import production
                        production.admission(plan,now(),len(active))
                    else:contract.admission(plan,accepted,now(),charges(campaign),running=len(active))
                except ValueError as exc:
                    write(packet/'STATUS.json',dict(state='ADMISSION_HELD',reason=str(exc),at=now().isoformat()),immutable=False)
                    event(campaign,packet.name+'-held','admission_failure',dict(packet=packet.name,reason=str(exc)));continue
                verify_source(plan)
                if digest(packet/'CONTROLS_BEFORE_DATA.json')!=plan['controls_sha256'] or not read(packet/'CONTROLS_BEFORE_DATA.json').get('passed'):raise ValueError('known-answer receipt changed')
                write(packet/'ADMISSION.json',dict(at=now().isoformat(),plan_sha256=digest(packet/'PLAN.json'),charges_before=charges(accounting_root)))
                write(accounting_root/'accounting'/(packet.name+'.json'),dict(pool=plan['pool'],kind=plan['kind'],family=plan['family'],state='RESERVED',reserved_cpu_seconds=plan['cpu_cap_seconds']))
                env=dict(os.environ,PYTHONPATH=plan['source_root'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
                for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
                log=(packet/'parent.log').open('xb')
                proc=subprocess.Popen([plan['interpreter'],'-B','-m','runners.run_v25','--packet',str(packet)],cwd=plan['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                active[packet.name]=(proc,log);write(packet/'STATUS.json',dict(state='LAUNCHING',launcher_pid=proc.pid,at=now().isoformat()),immutable=False)
            write(campaign/'QUEUE_STATUS.json',dict(state='RUNNING' if active else 'WAITING_FOR_PREPARED_COMMISSIONED_WORK',pid=os.getpid(),heartbeat=now().isoformat(),active=list(active),finished=finished),immutable=False)
            write(campaign/'accounting/controller.json',dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
            if (campaign/'BUILD_COMPLETE.json').exists() and not active and not candidates:
                event(campaign,'v25-prepared-queue-drained','review_boundary',dict(finished=finished));break
            if (campaign/'PAUSE_ADMISSIONS').exists() and not active:break
            time.sleep(5)
    finally:
        write(campaign/'QUEUE_STATUS.json',dict(state='STOPPED_WITH_ACTIVE_CHILDREN' if active else 'DRAINED',pid=os.getpid(),active=list(active),finished=finished,at=now().isoformat()),immutable=False)
        write(campaign/'accounting/controller.json',dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='STOPPED'),immutable=False)
        (campaign/'QUEUE_OWNER.lock').unlink()

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path);p.add_argument('--prepare',choices=contract.ORDER);p.add_argument('--queue',action='store_true');p.add_argument('--packet',type=Path);p.add_argument('--child',choices=('original','replay','review'))
    a=p.parse_args()
    if a.prepare:print(prepare(a.campaign.resolve(),a.prepare))
    elif a.queue:queue(a.campaign.resolve())
    elif a.child:child(a.packet.resolve(),a.child)
    elif a.packet:execute(a.packet.resolve())
    else:p.error('choose preparation, queue or packet')

if __name__=='__main__':main()
