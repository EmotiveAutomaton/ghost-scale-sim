"""Finite V24 N16 pilot: prepare now, admit only in the authorized pilot window.

One campaign owner serializes science, replay and review. Failed ownership and
reserved-cost receipts remain for native reconciliation, never automatic reset.
"""
import argparse
import ctypes
from datetime import datetime, timezone
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
    pilot.validate_design(plan)


def preflight(packet):
    """Read-only scientific preflight; known ledger fixture, no pilot sampling."""
    plan=read(packet/'PLAN.json');verify_sources(packet,plan)
    sys.path.insert(0,str(packet/'source/dependencies'))
    from ghostscale.validation.soundingline.v23 import ledger
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    pilot.emit(schema,['v24-engineering-excluded','preflight'],{'fixture':'known-observation'})
    controls=n16.model.controls()
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


def accounting(campaign,family):
    charged=0.;family_cost=0.
    for path in (campaign/'accounting').glob('*.json'):
        record=read(path)
        # A reserved or interrupted packet keeps its full allowance until reconciled.
        cost=max(record.get('cpu_seconds',0),record.get('reserved_cpu_seconds',0))
        charged+=cost
        if record.get('family')==family:family_cost+=cost
    return charged,family_cost


def child(packet,label):
    plan=read(packet/'PLAN.json');contract.admission(plan,now(),0)
    runtime.child_handshake(packet,label);runtime.below_normal()
    verify_sources(packet,plan);started=time.process_time()
    def pulse(**progress):
        write(packet/(label+'-progress.json'),dict(pid=os.getpid(),heartbeat=now().isoformat(),
            cpu_seconds=time.process_time()-started,**progress),immutable=False)
        if time.process_time()-started>plan['cpu_cap_seconds']/3:raise TimeoutError('pilot pass CPU allowance')
        if now()>=datetime.fromisoformat(contract.CLOSE):raise TimeoutError('pilot window closed')
    pulse(phase='starting')
    if label=='review':write(packet/'INDEPENDENT_REVIEW.json',pilot.reconstruct(packet/'original',plan,pulse))
    else:pilot.execute(packet/label,plan,pulse)
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
                originals={p.name:digest(p) for p in (packet/'original').iterdir() if p.is_file()}
                repeats={p.name:digest(p) for p in (packet/'replay').iterdir() if p.is_file()}
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
    parser.add_argument('--predecessor-plan',type=Path);parser.add_argument('--packet',type=Path)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--child',choices=('original','replay','review'))
    args=parser.parse_args()
    if args.prepare:
        if not args.campaign or not args.predecessor_plan:parser.error('preparation requires campaign and predecessor plan')
        print(prepare(args.campaign,args.predecessor_plan))
    elif args.preflight:
        if not args.packet:parser.error('preflight requires packet')
        print(preflight(args.packet.resolve()))
    elif args.child:
        if not args.packet:parser.error('child requires packet')
        child(args.packet.resolve(),args.child)
    elif args.packet:execute(args.packet)
    else:parser.error('choose preparation or a prepared packet')


if __name__=='__main__':main()
