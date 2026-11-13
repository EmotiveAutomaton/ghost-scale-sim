"""Finite declared V29 texture/follow-on successor, after primary service closure."""
import argparse,ctypes,importlib,os,subprocess,sys,time,traceback,json,math
from ghostscale.validation.soundingline.v29.continuation import Adapter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v29.common import read,write,digest
from ghostscale.validation.soundingline.v27 import contract
from runners.run_v27 import env_for
from runners.run_k3_redesign import environment,host_sample
from runners.watch_v23_events import owner,event

LIMITS={k:3600*v for k,v in dict(X1=4,Q2=3,G1b=3,Q3=3,X3=6,Q1=4,C1=1,B2=2,B1=3,B3=3,X2=4,reserve=24).items()}
STAGES=('controls','pilot','original','replay','reconstruct')


def now():return datetime.now(timezone.utc)


def charges(c):
    result=dict.fromkeys(LIMITS,0.)
    for p in (c/'accounting').glob('*.json'):
        row=read(p);result[row['family']]+=max(row.get('cpu_seconds',0.),row.get('reserved_cpu_seconds',0.))
    return result


def account(c,identity,family,used,state,reserve=0.):
    write(c/'accounting'/(identity+'.json'),dict(family=family,cpu_seconds=used,state=state,
        reserved_cpu_seconds=reserve),immutable=False)


def verify(c,p):
    if digest(c/'jobs'/p['id']/'PLAN.json')!=read(c/'PREPARED_MANIFEST.json')[p['id']]:raise ValueError('prepared plan changed')
    if digest(c/'ACCEPTANCE.json')!=p['acceptance_sha256']:raise ValueError('acceptance changed')
    if digest(c/'ENVIRONMENT.json')!=p['environment_sha256'] or environment()!=read(c/'ENVIRONMENT.json'):
        raise ValueError('environment changed')
    source=Path(p['source']['root'])
    if digest(source/'SOURCE_MANIFEST.json')!=p['source']['manifest_sha256']:raise ValueError('source manifest changed')
    for name,expected in read(source/'SOURCE_MANIFEST.json').items():
        if digest(source/name)!=expected:raise ValueError('source byte changed: '+name)


def pilot(module,output,pulse):
    roster=module.settings()
    selected=[s|dict(worlds=40,blocks=8) for s in roster]
    original=output/'pilot-original';replay=output/'pilot-replay';reference=output/'pilot-reconstruct'
    module.run(original,selected=selected,lane='pilot',pulse=pulse)
    module.run(replay,selected=selected,lane='pilot',pulse=pulse)
    module.run(reference,mode='reconstruct',original=original,selected=selected,lane='pilot',pulse=pulse)
    if {p.name:digest(p) for p in original.iterdir()}!={p.name:digest(p) for p in replay.iterdir()}:
        raise ValueError('pilot full replay differs')
    compare(read(original/'SUMMARY.json'),read(reference/'SUMMARY.json'))
    admitted=[];excluded=[]
    for i,s in enumerate(roster):
        with np.load(original/f'cell-{i:03d}_measures.npz',allow_pickle=False) as z:
            trivial='separation' in z and bool(np.all(z['separation']<=10))
            alias='same_probability' in z and bool(np.any(np.all(z['same_probability']==1,axis=0)))
            declared=module.pilot_exclusion(s,z) if hasattr(module,'pilot_exclusion') else None
        if trivial or alias or declared:excluded.append(dict(index=i,setting=s,reason=declared or ('all 40 within 10 steps' if trivial else 'exact observed-action class alias')))
        else:admitted.append(s)
    result=dict(passed=True,admitted_settings=admitted,excluded=excluded,pilot_worlds=40,
                original_summary_sha256=digest(original/'SUMMARY.json'),
                independent_summary_sha256=digest(reference/'SUMMARY.json'),replay_verified=True)
    write(output/'PILOT.json',result);return result


def worker(c,root):
    runtime.below_normal();runtime.child_handshake(root,'worker');p=read(root/'PLAN.json');verify(c,p)
    a=read(c/'ACCEPTANCE.json');last=[0.]
    def pulse(**progress):
        if time.process_time()>p['cpu_cap_seconds'] or now()>=datetime.fromisoformat(a['drain']) or (c/'STOP').exists():
            raise TimeoutError('CPU, stop or drain limit')
        if time.monotonic()-last[0]>5:
            write(root/'PROGRESS.json',dict(pid=os.getpid(),heartbeat=now().isoformat(),cpu_seconds=time.process_time(),**progress),immutable=False)
            last[0]=time.monotonic()
    output=root/'output';output.mkdir()
    try:
        module=Adapter(p)
        if p['stage']=='controls':
            result=module.controls(output,pulse)
            if not result['passed']:raise ArithmeticError('known-answer gate failed')
            name='CONTROLS.json'
        else:
            admission=read(root/'ADMITTED.json');gate=c/'families'/p['family']/'PREFLIGHT.json'
            if digest(gate)!=admission['preflight_sha256'] or not read(gate)['passed']:
                raise ValueError('preflight admission changed or absent')
            if p['stage']=='pilot':
                result=pilot(module,output,pulse);name='PILOT.json'
            else:
                original=c/'jobs'/(p['family']+'-original')/'output'
                roster_path=c/'jobs'/(p['family']+'-pilot')/'output/PILOT.json'
                if digest(roster_path)!=admission['pilot_sha256']:raise ValueError('pilot admission changed')
                selected=read(roster_path)['admitted_settings']
                if p['stage']=='reconstruct':
                    for name,expected in read(original/'RAW_MANIFEST.json').items():
                        if digest(original/name)!=expected:raise ValueError('retained raw input changed')
                result=module.run(output,mode=p['stage'],original=original,pulse=pulse,selected=selected);name='SUMMARY.json'
        write(root/'WORKER_COMPLETE.json',dict(plan_sha256=digest(root/'PLAN.json'),output_file=name,
            output_sha256=digest(output/name),cpu_seconds=time.process_time()))
    except BaseException as exc:
        write(root/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc()));raise


def start(c,root):
    p=read(root/'PLAN.json');write(root/'STARTED.json',dict(at=now().isoformat(),plan_sha256=digest(root/'PLAN.json')))
    account(c,root.name,p['family'],0.,'RUNNING',p['cpu_cap_seconds'])
    log=(root/'worker.log').open('xb');env=env_for(p['source']['root'])
    env['NUMBA_NUM_THREADS']='1';env['NUMBA_CACHE_DIR']=str(c/'numba-cache')
    child=subprocess.Popen([p['interpreter'],'-B','-m','runners.run_v29_continuation','--campaign',str(c),'--worker',str(root)],
        cwd=p['source']['root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    try:handle=runtime.attach_child(child,root,'worker')
    except BaseException:
        if child.poll() is None:child.terminate()
        child.wait(timeout=10);log.close()
        account(c,root.name,p['family'],p['cpu_cap_seconds'],'FAILED')
        write(root/'COMPLETE.json',dict(state='FAILED',error='native handshake failed; full cap charged'))
        raise
    return dict(root=root,plan=p,child=child,handle=handle,log=log,opened=time.monotonic())


def finish(c,job,error=None):
    child=job['child'];root=job['root'];p=job['plan']
    if error and child.poll() is None:
        ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(job['handle']),1)
        ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(job['handle']),10000);child.wait(timeout=10)
    used=runtime.child_cpu(job['handle'])
    if error is None:
        try:
            receipt=read(root/'WORKER_COMPLETE.json')
            if child.returncode!=0 or receipt['plan_sha256']!=digest(root/'PLAN.json') or receipt['output_sha256']!=digest(root/'output'/receipt['output_file']):
                error='worker exit or completion binding failure'
        except (OSError,ValueError,KeyError) as exc:error='missing worker completion: '+str(exc)
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(job['handle']));job['log'].close()
    state='FAILED' if error else 'COMPLETE';account(c,root.name,p['family'],used,state)
    write(root/'COMPLETE.json',dict(state=state,error=error,cpu_seconds=used,plan_sha256=digest(root/'PLAN.json'),
        worker_receipt_sha256=digest(root/'WORKER_COMPLETE.json') if (root/'WORKER_COMPLETE.json').exists() else None))
    if error:event(c,root.name+'-failed','batch_failed',dict(job=root.name,error=error))
    return state


def compare(a,b,path='root'):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('different keys: '+path)
        for key in a:compare(a[key],b[key],path+'/'+key)
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('different lengths: '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not np.isclose(a,b,rtol=1e-10,atol=1e-8):raise ValueError(f'numerical discrepancy {path}: {a} != {b}')
    elif a!=b:raise ValueError('value discrepancy: '+path)


def verify_family(c,family):
    root=c/'families'/family
    if (root/'VERIFIED.json').exists():return
    original=c/'jobs'/(family+'-original')/'output';replay=c/'jobs'/(family+'-replay')/'output'
    a={p.name:digest(p) for p in original.iterdir()};b={p.name:digest(p) for p in replay.iterdir()}
    if a!=b:raise ValueError('full raw/summary replay differs: '+family)
    reconstruction=c/'jobs'/(family+'-reconstruct')/'output'
    compare(read(original/'SUMMARY.json'),read(reconstruction/'SUMMARY.json'))
    for path in original.glob('*_measures.npz'):
        with np.load(path,allow_pickle=False) as x,np.load(reconstruction/path.name,allow_pickle=False) as y:
            if set(x.files)!=set(y.files):raise ValueError('measure roster differs')
            for key in x.files:
                if not np.allclose(x[key],y[key],rtol=0,atol=1e-8,equal_nan=True):raise ValueError('raw metric reconstruction differs: '+key)
    write(root/'VERIFIED.json',dict(family=family,passed=True,at=now().isoformat(),replay_files=a,
        original_summary_sha256=digest(original/'SUMMARY.json'),independent_summary_sha256=digest(reconstruction/'SUMMARY.json'),
        preflight_sha256=digest(root/'PREFLIGHT.json'),no_registered_claim=True))
    event(c,family+'-verified','batch_complete',dict(family=family,verification_sha256=digest(root/'VERIFIED.json')))


def run(c):
    runtime.below_normal();active={};blocked=read(c/'DISPOSITIONS.json');previous=None;samples=[];limit=4
    a=read(c/'ACCEPTANCE.json')
    families=a['families'];limits=a.get('cpu_limits',LIMITS)
    with owner(c/'controller'):
        try:
            if (c/'QUEUE_COMPLETE.json').exists():raise ValueError('closed V29 cannot restart')
            if any((p/'STARTED.json').exists() and not (p/'COMPLETE.json').exists() for p in (c/'jobs').iterdir()):
                raise ValueError('interrupted job needs a separate recovery decision')
            while True:
                current=now();account(c,'controller-'+str(os.getpid()),'reserve',time.process_time(),'RUNNING')
                load,previous=host_sample(previous)
                if load is not None:samples.append((time.monotonic(),load));samples=samples[-180:];limit=contract.load_limit(samples,limit)
                for identity,job in list(active.items()):
                    used=runtime.child_cpu(job['handle']);error=None
                    if used>job['plan']['cpu_cap_seconds'] or time.monotonic()-job['opened']>job['plan']['wall_cap_seconds']:error='job resource limit'
                    if current>=datetime.fromisoformat(a['drain']) or (c/'STOP').exists():error='original drain or explicit stop'
                    if runtime.memory()['available']<2*1024**3:error='host memory protection'
                    if error or job['child'].poll() is not None:
                        result=finish(c,job,error);del active[identity]
                        if result!='COMPLETE':blocked[job['plan']['family']]='retained job failure'
                    else:account(c,identity,job['plan']['family'],used,'RUNNING',job['plan']['cpu_cap_seconds'])
                for family in families:
                    jobs=[c/'jobs'/(family+'-'+s) for s in STAGES]
                    if any((j/'COMPLETE.json').exists() and read(j/'COMPLETE.json')['state']!='COMPLETE' for j in jobs):blocked[family]='retained job failure'
                    if family not in blocked and all((j/'COMPLETE.json').exists() for j in jobs):
                        try:verify_family(c,family)
                        except BaseException as exc:
                            blocked[family]=str(exc);write(c/'families'/family/'VERIFICATION_FAILED.json',dict(error=str(exc)))
                            event(c,family+'-verification-failed','batch_failed',dict(error=str(exc)))
                failed_modules={read(c/'jobs'/(f+'-controls')/'PLAN.json')['module'] for f,reason in blocked.items() if (c/'jobs'/(f+'-controls')/'PLAN.json').exists() and reason not in ('time limit','protected-pool admission limit')}
                for family in families:
                    planpath=c/'jobs'/(family+'-controls')/'PLAN.json'
                    if planpath.exists() and read(planpath)['module'] in failed_modules and not (c/'families'/family/'VERIFIED.json').exists():blocked.setdefault(family,'family held after variant instrument failure; no automatic extra healing')
                for family in blocked:
                    for identity,job in list(active.items()):
                        if job['plan']['family']==family:finish(c,job,'family stopped after failure');del active[identity]
                totals=charges(c);waiting=[]
                for root in sorted((c/'jobs').iterdir(),key=lambda p:(families.index(read(p/'PLAN.json')['family']),STAGES.index(read(p/'PLAN.json')['stage']))):
                    p=read(root/'PLAN.json');family=p['family'];stage=p['stage']
                    if (root/'STARTED.json').exists() or family in blocked:continue
                    if stage!='controls':
                        controls=c/'jobs'/(family+'-controls')/'COMPLETE.json';gate=c/'families'/family/'PREFLIGHT.json'
                        if not controls.exists() or not gate.exists():waiting.append(root.name);continue
                        if not read(gate)['passed']:blocked[family]='preflight comparison failed';continue
                        if stage not in ('pilot','original') and not (c/'jobs'/(family+'-original')/'COMPLETE.json').exists():waiting.append(root.name);continue
                    if stage not in ('controls','pilot'):
                        pilot_root=c/'jobs'/(family+'-pilot')
                        if not (pilot_root/'COMPLETE.json').exists():waiting.append(root.name);continue
                    if len(active)>=limit:waiting.append(root.name);continue
                    if current>=datetime.fromisoformat(a['new_starts_close']) or current+timedelta(seconds=p['wall_cap_seconds'])>=datetime.fromisoformat(a['drain']):blocked[family]='time limit';continue
                    if totals[family]+p['cpu_cap_seconds']>limits[family] or sum(totals.values())+p['cpu_cap_seconds']>a['total_cpu_seconds']:
                        if active:waiting.append(root.name)
                        else:blocked[family]='protected-pool admission limit'
                        continue
                    if stage!='controls':
                        admission=dict(at=current.isoformat(),preflight_sha256=digest(gate))
                        if stage!='pilot':admission['pilot_sha256']=digest(c/'jobs'/(family+'-pilot')/'output/PILOT.json')
                        write(root/'ADMITTED.json',admission)
                    active[root.name]=start(c,root);totals[family]+=p['cpu_cap_seconds']
                verified=[f for f in families if (c/'families'/f/'VERIFIED.json').exists()]
                status=dict(state='RUNNING' if active else 'AWAITING_ADMISSION',pid=os.getpid(),heartbeat=current.isoformat(),
                    active=list(active),waiting=waiting,blocked=blocked,verified=verified,gear=2,worker_limit=limit,charges=charges(c))
                write(c/'QUEUE_STATUS.json',status,immutable=False)
                if len(set(verified)|set(blocked))==len(families) and not active:
                    write(c/'QUEUE_COMPLETE.json',status|dict(state='FINITE_END'))
                    event(c,'v29-finite-end','queue_depleted',dict(verified=verified,blocked=blocked));break
                time.sleep(5)
        finally:
            for job in active.values():finish(c,job,'controller exited')
            account(c,'controller-'+str(os.getpid()),'reserve',time.process_time(),'STOPPED')



def primary_alive(pid):
    ctypes.windll.kernel32.OpenProcess.restype=ctypes.c_void_p
    h=ctypes.windll.kernel32.OpenProcess(0x100000,False,pid)
    if not h:return False
    value=ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(h),0)
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(h));return value==258


def prepare(c):
    a=read(c/'ACCEPTANCE.json');primary=Path(a['primary_campaign']);catalogue=read(c/'CATALOGUE.json')
    if digest(primary/'ACCEPTANCE.json')!=a['primary_acceptance_sha256']:raise ValueError('primary acceptance changed')
    while not (primary/'QUEUE_COMPLETE.json').exists() or primary_alive(read(primary/'QUEUE_STATUS.json')['pid']):
        if now()>=datetime.fromisoformat(a['new_starts_close']):raise TimeoutError('primary drain wait reached original last-start time')
        write(c/'WAITING.json',dict(state='WAITING_FOR_PRIMARY_OWNER_EXIT',heartbeat=now().isoformat()),immutable=False)
        time.sleep(10)
    if (c/'PREPARED_MANIFEST.json').exists():return
    prior=read(primary/'QUEUE_COMPLETE.json');carry=0.
    for p in (primary/'accounting').glob('*.json'):
        row=read(p)
        if row['family']=='reserve':carry+=max(row.get('cpu_seconds',0.),row.get('reserved_cpu_seconds',0.))
    account(c,'primary-reserve-carried','reserve',carry,'COMPLETE')
    source=Path(__file__).resolve().parents[1];descriptor=dict(root=str(source),manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),archive_sha256=digest(source/'SOURCE.zip'))
    dispositions={};manifest={};selection={}
    for card in catalogue:
        identity=card['id'];parent=card['parent'];module=importlib.import_module('ghostscale.validation.soundingline.v29.'+card['module'])
        if parent not in prior['verified']:
            dispositions[identity]='parent primary not verified';continue
        summary=read(primary/f'jobs/{parent}-original/output/SUMMARY.json')
        if card['selection']=='explicit':roster=card['roster']
        elif card['selection']=='long':
            found={}
            for row in summary['rows']:
                if row['measure']=='separation' and row['unresolved']['estimate']>.30:found[json.dumps(row['setting'],sort_keys=True)]=row['setting']
            roster=[]
            for setting in found.values():
                setting=setting.copy()
                if 'length' in setting:setting['length']*=2
                else:setting['steps']*=2
                if 'life' in setting:setting['life']=max(setting['life'],setting.get('start',0)+setting['steps'])
                roster.append(setting)
        else:roster=module.settings()
        if card.get('part'):roster=[x for x in roster if x.get('part')==card['part']]
        roster=[x|card.get('overrides',{})|dict(continuation=identity) for x in roster]
        if not roster:
            dispositions[identity]='no applicable cells after declared primary trigger';continue
        selection[identity]=dict(parent_summary_sha256=digest(primary/f'jobs/{parent}-original/output/SUMMARY.json'),settings=roster)
        root=c/'families'/identity;root.mkdir()
        write(root/'PREFLIGHT.json',dict(passed=True,kind='Inherited verified primary comparison plus prospective variant controls; no supplied texture numeric baseline',parent_verification_sha256=digest(primary/f'families/{parent}/VERIFIED.json'),source_manifest_sha256=descriptor['manifest_sha256'],catalogue_sha256=digest(c/'CATALOGUE.json'),roster=roster))
        # A full pilot is itself three passes. Reserve conservatively from the
        # parent's measured passes, scaled by cell count and record duration.
        scale=len(roster)/max(1,len(summary['settings']))
        if card['selection']=='long':scale*=2
        caps=[]
        for stage in STAGES:
            spent=read(primary/f'jobs/{parent}-{stage}/COMPLETE.json')['cpu_seconds']
            multiplier=3 if stage=='controls' else 2
            cap=max(180 if stage=='controls' else 300,math.ceil(spent*scale*multiplier+120))
            if card['module']=='b3':cap+=600
            caps.append(min(cap,14400))
        for stage,cap in zip(STAGES,caps):
            root=c/'jobs'/(identity+'-'+stage);root.mkdir()
            plan=dict(id=root.name,family=identity,module=card['module'],roster=roster,stage=stage,cpu_cap_seconds=cap,wall_cap_seconds=2*cap+300,source=descriptor,acceptance_sha256=digest(c/'ACCEPTANCE.json'),environment_sha256=digest(c/'ENVIRONMENT.json'),interpreter=a['interpreter'])
            write(root/'PLAN.json',plan);manifest[root.name]=digest(root/'PLAN.json')
    write(c/'SELECTIONS.json',selection);write(c/'DISPOSITIONS.json',dispositions);write(c/'PREPARED_MANIFEST.json',manifest)
    write(c/'PRIMARY_DRAIN_RECEIPT.json',dict(primary_closeout_sha256=digest(primary/'QUEUE_COMPLETE.json'),carried_protected_cpu=carry,no_primary_owner_alive=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--worker',type=Path)
    args=parser.parse_args();campaign=args.campaign.resolve();acceptance=read(campaign/'ACCEPTANCE.json');LIMITS=acceptance['cpu_limits']
    if args.worker:worker(campaign,args.worker.resolve())
    else:
        with owner(campaign/'preparation-owner'):prepare(campaign)
        run(campaign)
