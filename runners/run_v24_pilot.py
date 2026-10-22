"""Finite V24 N16/D5 pilots: admit only in the authorized pilot window.

One campaign owner serializes science, replay and review. Failed ownership and
reserved-cost receipts remain for native reconciliation, never automatic reset.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from ghostscale.validation.soundingline.v23.common import digest, read, write
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v24 import contract, n16, pilot

REPO=Path(__file__).resolve().parents[1]


def now():return datetime.now(timezone.utc)


def implementation(plan):
    if plan['family']=='N16':return pilot
    if plan['family']=='D5':
        from ghostscale.validation.soundingline.v24 import d5
        return d5
    if plan['family'] in ('T1','T3','T4'):
        from ghostscale.validation.soundingline.v24 import twin_successor
        return twin_successor
    if plan['family']=='N1':
        from ghostscale.validation.soundingline.v24 import population_successor
        return population_successor
    raise ValueError('unbuilt pilot family')


def verify_sources(packet,plan):
    if REPO.resolve()!=(packet/'source').resolve():
        raise ValueError('run preflight and pilot parent from the immutable source capsule')
    if digest(packet/'PLAN.json')!=read(packet/'PREPARATION.json')['plan_sha256']:
        raise ValueError('frozen plan changed')
    if digest(packet/'SOURCE.zip')!=plan['source_archive_sha256']:raise ValueError('source archive changed')
    actual={p.relative_to(packet/'source').as_posix() for p in (packet/'source').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if actual!=set(plan['source_files']):raise ValueError('frozen source roster changed')
    for name,sha in plan['source_files'].items():
        if digest(packet/'source'/name)!=sha:raise ValueError('frozen source changed')
    if digest(packet.parent.parent/'ACCEPTANCE.json')!=plan['acceptance_sha256']:raise ValueError('acceptance changed')
    if read(packet.parent.parent/'ACCEPTANCE.json')!=contract.acceptance():raise ValueError('unknown pilot acceptance')
    if runtime.environment()!=plan['environment']:raise ValueError('scientific environment changed; do not synchronize')
    implementation(plan).validate_design(plan)


def preflight(packet):
    """Read-only scientific preflight; known ledger fixture, no pilot sampling."""
    plan=read(packet/'PLAN.json');verify_sources(packet,plan)
    sys.path.insert(0,str(packet/'source/dependencies'))
    from ghostscale.validation.soundingline.v23 import ledger
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    pilot.emit(schema,['v24-engineering-excluded','preflight'],{'fixture':'known-observation'})
    controls=n16.model.controls() if plan['family']=='N16' else implementation(plan).preflight(plan)
    if not controls['passed']:raise ValueError('preflight controls failed')
    receipt=dict(state='PREFLIGHT_PASSED_NOT_ADMITTED',new_scientific_worlds=0,
        plan_sha256=digest(packet/'PLAN.json'),source_archive_sha256=plan['source_archive_sha256'],
        source_files_verified=len(plan['source_files']),canonical_ledger_roundtrip=True,
        controls=controls,at=now().isoformat())
    write(packet/'PREFLIGHT.json',receipt)
    return receipt


def prepare(campaign,predecessor_plan):
    """Creates an immutable, not-yet-admitted packet. Never starts a child."""
    campaign=campaign.resolve();campaign.mkdir(parents=True,exist_ok=True)
    if not (campaign/'ACCEPTANCE.json').exists():write(campaign/'ACCEPTANCE.json',contract.acceptance())
    if read(campaign/'ACCEPTANCE.json')!=contract.acceptance():raise ValueError('acceptance differs')
    packet=campaign/'packets/v24-n16-pilot-1';packet.mkdir(parents=True,exist_ok=False)
    source=packet/'source';source.mkdir();started=time.process_time()
    paths=sorted((REPO/'ghostscale').rglob('*.py'))+[Path(__file__).resolve()]
    for path in paths:
        target=source/path.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    # Namespace import in the isolated capsule does not depend on the main checkout.
    (source/'runners/__init__.py').write_text('',encoding='utf-8')
    previous=read(predecessor_plan);schema=source/'schema';schema.mkdir()
    for name,sha in previous['schema_pin']['files'].items():
        original=Path(previous['schema_directory'])/name
        if digest(original)!=sha:raise ValueError('canonical dependency changed')
        shutil.copyfile(original,schema/name)
    dependency_root=predecessor_plan.resolve().parent.parent.parent/'dependencies'
    for name,sha in previous['dependency_files'].items():
        original=Path(name).resolve();relative=original.relative_to(dependency_root)
        if digest(original)!=sha:raise ValueError('canonical runtime dependency changed')
        target=source/'dependencies'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,target)
    calibration=REPO/'results/v24/preparation/N16_DESIGN_CALIBRATION.json'
    shutil.copyfile(calibration,source/'N16_DESIGN_CALIBRATION.json')
    hashes={p.relative_to(source).as_posix():digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    with zipfile.ZipFile(packet/'SOURCE.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:archive.write(source/name,name)
    plan=dict(kind='pilot',family='N16',identity='v24-n16-pilot-1',blocks=4,cases_per_cell=5,
        epsilon=.1,variant_ids=[v['id'] for v in n16.model.variants()],
        numerical_threads=1,priority='below-normal',cpu_cap_seconds=5400,wall_cap_seconds=14400,
        source_files=hashes,source_archive_sha256=digest(packet/'SOURCE.zip'),
        acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
        schema_directory=str(schema),schema_pin={k:v for k,v in previous['schema_pin'].items() if k!='directory'},
        environment=runtime.environment(),interpreter=str(REPO/'.venv/Scripts/python.exe'),
        preparation_at=now().isoformat(),confirmation=False,
        retained_calibration_sha256=digest(REPO/'results/v24/preparation/N16_DESIGN_CALIBRATION.json'))
    write(packet/'PLAN.json',plan)
    write(packet/'PREPARATION.json',dict(state='PREPARED_NOT_ADMITTED',
        plan_sha256=digest(packet/'PLAN.json'),source_archive_sha256=plan['source_archive_sha256']))
    write(packet/'STATUS.json',dict(state='PREPARED_NOT_ADMITTED',pilot_open=contract.OPEN,
        plan_sha256=digest(packet/'PLAN.json'),at=now().isoformat()))
    write(campaign/'accounting/source-preparation.json',dict(kind='preparation',
        cpu_seconds=time.process_time()-started,state='COMPLETE'))
    return packet


def prepare_twin(campaign,predecessor_plan,family):
    """Freeze one complete excluded T1/T3/T4 cost-profile packet.

    The predecessor supplies the measured reference, canonical engine snapshot,
    schema and compiled bridge dependencies.  The V24 D5 successor supplies the
    independently verified certified backbone.  Preparation executes no world.
    """
    if family not in ('T1','T3','T4'):raise ValueError('built twin family required')
    campaign=campaign.resolve();campaign.mkdir(parents=True,exist_ok=True)
    if not (campaign/'ACCEPTANCE.json').exists():write(campaign/'ACCEPTANCE.json',contract.acceptance())
    if read(campaign/'ACCEPTANCE.json')!=contract.acceptance():raise ValueError('acceptance differs')
    previous=read(predecessor_plan)
    if previous.get('family')!=family or previous.get('task')!='twin-profile':
        raise ValueError('matching retained twin profile required')
    identity=f"v24-{family.lower()}-cost-profile-1"
    _,family_charged=accounting(campaign,family)
    preparation_headroom=60
    cpu_cap=contract.FAMILY_SECONDS[family]-family_charged-preparation_headroom
    if cpu_cap<=0:raise ValueError('no family allowance remains after preparation costs')
    packet=campaign/'packets'/identity;packet.mkdir(parents=True,exist_ok=False)
    source=packet/'source';source.mkdir();started=time.process_time()
    for path in sorted((REPO/'ghostscale').rglob('*.py'))+[
            Path(__file__).resolve(),REPO/'runners/launch_background.py']:
        target=source/path.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    (source/'runners/__init__.py').write_text('',encoding='utf-8')

    # Preserve the isolated schema and resident canonical runtime used by the
    # failed predecessor.  Every copied byte is rechecked before the archive is
    # frozen; no package installation or environment synchronization occurs.
    schema=source/'schema';schema.mkdir()
    for name,sha in previous['schema_pin']['files'].items():
        original=Path(previous['schema_directory'])/name
        if digest(original)!=sha:raise ValueError('canonical dependency changed')
        shutil.copyfile(original,schema/name)
    dependency_target=source/'dependencies';dependency_target.mkdir()
    retained_root=predecessor_plan.resolve().parents[2]
    roots=[(retained_root/name).resolve() for name in ('bridge-dependencies','dependencies')]
    for name,sha in previous['dependency_files'].items():
        original=Path(name).resolve()
        if digest(original)!=sha:raise ValueError('retained runtime dependency changed')
        matches=[]
        for root in roots:
            try:matches.append(original.relative_to(root))
            except ValueError:pass
        if len(matches)!=1:
            raise ValueError(f'dependency root is ambiguous or missing: {original}')
        target=dependency_target/matches[0];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,target)
    metadata_files=0
    for root in roots:
        for metadata in sorted(root.glob('*.dist-info')):
            for original in sorted(metadata.rglob('*')):
                if not original.is_file():continue
                target=dependency_target/original.relative_to(root)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(original,target);metadata_files+=1

    reference=source/'inputs'/f'{family}_REFERENCE.json';reference.parent.mkdir()
    empirical_source=source/'inputs'/f'{family}_SOURCE.json'
    for original,target,expected in (
        (Path(previous['empirical_reference']),reference,previous['empirical_reference_sha256']),
        (Path(previous['empirical_source']),empirical_source,
         read(previous['empirical_reference'])['source']['sha256']),
    ):
        if digest(original)!=expected:raise ValueError('retained empirical input changed')
        shutil.copyfile(original,target)

    d5_packet=campaign/'packets/v24-d5-pilot-1-r1'
    d5_plan=read(d5_packet/'PLAN.json')
    backbone_source=Path(d5_plan['backbone_source'])
    if digest(backbone_source)!=d5_plan['backbone_sha256']:
        raise ValueError('certified backbone changed')
    frozen_backbone=source/'upstream/backbone.py.txt';frozen_backbone.parent.mkdir()
    shutil.copyfile(backbone_source,frozen_backbone)

    engine_roots=[Path(value).resolve() for value in previous['extra_pythonpath']
                  if (Path(value).resolve()/'tompathy_engine').is_dir()]
    if len(engine_roots)!=1:raise ValueError('one retained engine root is required')
    engine_root=engine_roots[0]/'tompathy_engine';engine_files={};engine_text={}
    declared={Path(name).resolve().relative_to(engine_root).as_posix():sha
              for name,sha in previous['external_source_files'].items()}
    engine_sources=[path for path in sorted(engine_root.rglob('*'))
                    if path.is_file() and path.suffix in ('.py','.yaml')]
    if not set(declared).issubset({path.relative_to(engine_root).as_posix()
                                   for path in engine_sources}):
        raise ValueError('retained engine source roster changed')
    for original in engine_sources:
        relative=original.relative_to(engine_root);name=relative.as_posix()
        sha=digest(original)
        if name in declared and sha!=declared[name]:
            raise ValueError('retained engine source changed')
        target=dependency_target/'tompathy_engine'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
        engine_files[name]=sha;engine_text[name]=original.read_text(encoding='utf8')
    reference_packet=Path(previous['engine_reference_packet'])
    if digest(reference_packet/'COMPLETE.json')!=previous['engine_reference_complete_sha256']:
        raise ValueError('retained engine reference completion changed')
    conformance=read(reference_packet/'original/T1-clean/engine/CONFORMANCE.json')
    implementation_hash=conformance['implementation_hash']
    actual_implementation=hashlib.sha256(json.dumps(engine_text,sort_keys=True,
        ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(
            'utf-8',errors='backslashreplace')).hexdigest()
    if actual_implementation!=implementation_hash:
        raise ValueError('retained engine implementation hash changed')

    hashes={p.relative_to(source).as_posix():digest(p) for p in sorted(source.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    with zipfile.ZipFile(packet/'SOURCE.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:archive.write(source/name,name)
    from ghostscale.validation.soundingline.v23 import registry
    plan=dict(kind='pilot',task='v24-twin-profile',family=family,identity=identity,
        production=False,confirmation=False,variant_ids=[v['id'] for v in registry.variants(family)],
        world_indices=[0],numerical_threads=1,priority='below-normal',
        cpu_cap_seconds=cpu_cap,wall_cap_seconds=12*3600,
        family_charged_before_preparation=family_charged,
        preparation_headroom_seconds=preparation_headroom,
        dependency_metadata_files=metadata_files,
        source_files=hashes,source_archive_sha256=digest(packet/'SOURCE.zip'),
        acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
        schema_directory=str(schema),schema_pin={k:v for k,v in previous['schema_pin'].items() if k!='directory'},
        empirical_reference=str(reference),empirical_reference_sha256=digest(reference),
        empirical_source=str(empirical_source),engine_files=engine_files,
        implementation_hash=implementation_hash,
        certified_backbone_source=str(frozen_backbone),
        certified_backbone_sha256=digest(frozen_backbone),canary_key=previous['canary_key'],
        predecessor_plan_sha256=digest(predecessor_plan),environment=runtime.environment(),
        interpreter=str(REPO/'.venv/Scripts/python.exe'),preparation_at=now().isoformat())
    write(packet/'PLAN.json',plan)
    write(packet/'PREPARATION.json',dict(state='PREPARED_NOT_ADMITTED',
        plan_sha256=digest(packet/'PLAN.json'),source_archive_sha256=plan['source_archive_sha256'],
        predecessor_plan_sha256=plan['predecessor_plan_sha256']))
    write(packet/'STATUS.json',dict(state='PREPARED_NOT_ADMITTED',pilot_open=contract.OPEN,
        plan_sha256=digest(packet/'PLAN.json'),at=now().isoformat()))
    write(campaign/'accounting'/f'{identity}-preparation.json',dict(kind='preparation',
        family=family,cpu_seconds=time.process_time()-started,state='COMPLETE'))
    return packet


def accounting(campaign,family):
    charged=0.;family_cost=0.
    for path in (campaign/'accounting').glob('*.json'):
        record=read(path)
        # A reserved or interrupted packet keeps its full allowance until reconciled.
        cost=max(record.get('cpu_seconds',0),record.get('reserved_cpu_seconds',0))
        charged+=cost
        if record.get('family')==family:family_cost+=cost
    return charged,family_cost


def progress_callback(packet,label,plan,started):
    last=[-float('inf')]
    def pulse(**progress):
        elapsed=time.process_time()-started;current=now()
        # Check every call; persist routine heartbeats at most every five seconds.
        # D5 can evaluate its objective many times between target boundaries.
        if elapsed>plan['cpu_cap_seconds']/3:raise TimeoutError('pilot pass CPU allowance')
        if current>=datetime.fromisoformat(contract.CLOSE):raise TimeoutError('pilot window closed')
        tick=time.monotonic()
        if tick-last[0]>=5 or progress.get('phase') in ('starting','complete'):
            write(packet/(label+'-progress.json'),dict(pid=os.getpid(),heartbeat=current.isoformat(),
                cpu_seconds=elapsed,**progress),immutable=False)
            last[0]=tick
    return pulse


def child(packet,label):
    plan=read(packet/'PLAN.json');contract.admission(plan,now(),0)
    runtime.child_handshake(packet,label);runtime.below_normal()
    verify_sources(packet,plan);started=time.process_time()
    pulse=progress_callback(packet,label,plan,started)
    pulse(phase='starting')
    worker=implementation(plan)
    if label=='review':write(packet/'INDEPENDENT_REVIEW.json',worker.reconstruct(packet/'original',plan,pulse))
    else:worker.execute(packet/label,plan,pulse)
    pulse(phase='complete')


def execute(packet):
    if os.name!='nt':raise RuntimeError('native Windows process accounting required')
    packet=packet.resolve();campaign=packet.parent.parent;plan=read(packet/'PLAN.json')
    charged,family_cost=accounting(campaign,plan['family'])
    contract.admission(plan,now(),charged,family_charged_seconds=family_cost)
    if (packet/'COMPLETE.json').exists() or (campaign/'accounting'/f"{packet.name}.json").exists():
        raise ValueError('executed or reserved packet cannot be restarted; reconcile ownership and retain failures')
    verify_sources(packet,plan)
    lock=campaign/'OWNER.lock'
    # Exclusive creation is atomic; any stale owner requires native inspection.
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    started=time.process_time();cpu=0.;attempt=campaign/'accounting'/f'{packet.name}.json'
    active=None;handle=None;wall=time.monotonic();state='FAILED';error=None
    try:
        # Re-evaluate accounting under the sole campaign owner.
        charged,family_cost=accounting(campaign,plan['family'])
        contract.admission(plan,now(),charged,family_charged_seconds=family_cost)
        write(attempt,dict(family=plan['family'],state='RESERVED',reserved_cpu_seconds=plan['cpu_cap_seconds']))
        for label in ('original','replay','review'):
            env=dict(os.environ)
            for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
            env['PYTHONPATH']=os.pathsep.join([str(packet/'source'),str(packet/'source/dependencies')]);env['PYTHONNOUSERSITE']='1'
            command=[plan['interpreter'],'-m','runners.run_v24_pilot','--packet',str(packet),'--child',label]
            with (packet/(label+'.log')).open('xb') as log:
                active=subprocess.Popen(command,cwd=packet/'source',env=env,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                handle=runtime.attach_child(active,packet,label)
                while active.poll() is None:
                    current=runtime.child_cpu(handle)
                    write(packet/'STATUS.json',dict(state='RUNNING',phase=label,parent_pid=os.getpid(),
                        child_pid=read(packet/(label+'-owner.json'))['pid'],heartbeat=now().isoformat(),
                        cpu_seconds=cpu+current+time.process_time()-started),immutable=False)
                    if (cpu+current+time.process_time()-started>plan['cpu_cap_seconds'] or
                        time.monotonic()-wall>plan['wall_cap_seconds'] or
                        now()>=datetime.fromisoformat(contract.CLOSE) or
                        runtime.memory()['available']<2*1024**3 or (campaign/'STOP').exists()):
                        raise TimeoutError('CPU, wall, memory, stop or immutable window boundary')
                    time.sleep(2)
                cpu+=runtime.child_cpu(handle)
                ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle));handle=None
                if active.returncode:raise RuntimeError(f'{label} failed with exit {active.returncode}')
                active=None
            if label=='replay':
                originals={p.relative_to(packet/'original').as_posix():digest(p) for p in (packet/'original').rglob('*') if p.is_file()}
                repeats={p.relative_to(packet/'replay').as_posix():digest(p) for p in (packet/'replay').rglob('*') if p.is_file()}
                if originals!=repeats:raise ValueError('complete scientific replay mismatch')
                write(packet/'REPLAY.json',dict(state='PASSED',files=originals))
        write(packet/'COMPLETE.json',dict(state='PILOT_VERIFIED',confirmation=False,
            plan_sha256=digest(packet/'PLAN.json'),summary_sha256=digest(packet/'original/SUMMARY.json'),
            replay_sha256=digest(packet/'REPLAY.json'),review_sha256=digest(packet/'INDEPENDENT_REVIEW.json')))
        state='COMPLETE'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if handle is not None:
            if active is not None and active.poll() is None:
                ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(handle),1)
                ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(handle),10000)
            cpu+=runtime.child_cpu(handle)
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))
        if active is not None and active.poll() is None:
            # Before handshake this is only our just-created, owned launcher.
            active.terminate();active.wait(timeout=10)
        total=cpu+time.process_time()-started
        write(attempt,dict(family=plan['family'],state=state,cpu_seconds=total,
            reserved_cpu_seconds=0 if state=='COMPLETE' else plan['cpu_cap_seconds'],error=error),immutable=False)
        write(packet/'STATUS.json',dict(state=state,at=now().isoformat(),cpu_seconds=total,error=error),immutable=False)
        lock.unlink()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--campaign',type=Path)
    parser.add_argument('--prepare-twin',choices=('T1','T3','T4'))
    parser.add_argument('--predecessor-plan',type=Path);parser.add_argument('--packet',type=Path)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--child',choices=('original','replay','review'))
    args=parser.parse_args()
    if args.prepare:
        if not args.campaign or not args.predecessor_plan:parser.error('preparation requires campaign and predecessor plan')
        print(prepare(args.campaign,args.predecessor_plan))
    elif args.prepare_twin:
        if not args.campaign or not args.predecessor_plan:
            parser.error('twin preparation requires campaign and predecessor plan')
        print(prepare_twin(args.campaign,args.predecessor_plan,args.prepare_twin))
    elif args.preflight:
        if not args.packet:parser.error('preflight requires packet')
        print(preflight(args.packet.resolve()))
    elif args.child:
        if not args.packet:parser.error('child requires packet')
        child(args.packet.resolve(),args.child)
    elif args.packet:execute(args.packet)
    else:parser.error('choose preparation or a prepared packet')


if __name__=='__main__':main()
