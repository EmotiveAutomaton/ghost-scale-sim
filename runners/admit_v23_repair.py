"""Admit and optionally launch a bounded repair-family pilot or first batch.

All entry conditions are checked from receipts. A work-order row is never a job.
"""
import argparse
from pathlib import Path
import secrets,shutil,subprocess,sys,zipfile
from datetime import datetime,timezone
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import registry,runtime,ledger
from ghostscale.validation.soundingline.v23.family import verify_packet
from runners.watch_v23_events import alive


def admit(campaign,family,task,cpu_cap,launch=False):
    campaign=Path(campaign).resolve();acceptance=read(campaign/'ACCEPTANCE.json')
    if family not in ('R1','R2','R3','R4'):raise ValueError('only independently implemented repair pipelines can be admitted here')
    if not 0<cpu_cap<=14400:raise ValueError('repair packet cap must be positive and at most four CPU hours')
    if not acceptance['opened']<=datetime.now(timezone.utc).isoformat()<acceptance['report_start']:
        raise ValueError('outside discovery/setup window')
    calibration=read(campaign/'CALIBRATION_ACCEPTANCE.json')
    summary=Path(calibration['summary'])
    if digest(summary)!=calibration['summary_sha256'] or read(summary)['state']!='CALIBRATION_PASSED':
        raise ValueError('accepted calibration unavailable')
    if digest(Path(calibration['complete']))!=calibration['complete_sha256']:raise ValueError('calibration completion changed')
    schema_pin=read(campaign/'SCHEMA_PIN.json')
    if len(schema_pin['commit'])!=40 or schema_pin['committed_files_verified'] is not True:
        raise ValueError('a committed canonical dependency is required')
    ledger.canonical_schema(schema_pin['directory'],schema_pin)
    if runtime.environment()!=read(campaign/'ENVIRONMENT.json'):raise ValueError('environment changed')
    active=[];reserved=0.;group_reserved=0.;group_spent=0.
    for root in (campaign/'packets').iterdir():
        if not (root/'PLAN.json').exists():continue
        old=read(root/'PLAN.json');status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {}
        unfinished=status.get('state') not in ('complete','failed')
        if unfinished:reserved+=old['cpu_cap_seconds']
        protected_burnin=old.get('second_burnin_reservation_seconds',0.) if status.get('state')!='failed' else 0.
        second=campaign/'packets'/('v23-'+old.get('family','').lower()+'-batch-2')
        if protected_burnin and not (second/'PLAN.json').exists():reserved+=protected_burnin
        if status.get('state')=='running' and alive(status['pid']):active.append(root.name)
        if old['budget_group']==('setup' if task=='pilot' else 'repairs'):
            if unfinished:group_reserved+=old['cpu_cap_seconds']
            else:group_spent+=status.get('cpu_seconds',0.)
            if protected_burnin and not (second/'PLAN.json').exists():group_reserved+=protected_burnin
    if len(active)>=acceptance['max_scientific_workers']:raise ValueError('Gear 2 scientist slots are full')
    total=runtime.cost(campaign)
    if task=='pilot':
        for path in (campaign/'attempts').glob('*.json'):
            row=read(path)
            if 'job' not in row:group_spent+=max(row.get('cpu_seconds',0),row.get('uncertainty_cpu_seconds',0))
    if total+reserved+cpu_cap>acceptance['cpu_seconds']-acceptance['protected_cpu_seconds']:
        raise ValueError('global verification reserve would be invaded')
    group_limit=8*3600 if task=='pilot' else 14*3600
    if group_spent+group_reserved+cpu_cap>group_limit:raise ValueError('group cap exhausted')
    identity=f'v23-{family.lower()}-'+('pilot-1' if task=='pilot' else 'batch-1')
    root=campaign/'packets'/identity
    if root.exists():raise ValueError('preserve existing identity; reviewed repairs need a new identity')
    extras={}
    if task=='family-batch':
        pilot=campaign/'packets'/f'v23-{family.lower()}-pilot-1';pp,ps=verify_packet(pilot)
        if ps['state']!='PILOT_VALID':raise ValueError('family pilot not accepted')
        costs=[read(pilot/label/'pilot-cost.json') for label in ('original','replay')]
        case_cost=max(row['inference_cpu_seconds']/row['cases'] for row in costs)
        lower_cost=case_cost*len(registry.variants(family))*16*64*2
        if lower_cost>cpu_cap:raise ValueError('largest-cell extrapolation exceeds the admitted batch cap before training')
        if total+reserved+cpu_cap+lower_cost>acceptance['cpu_seconds']-acceptance['protected_cpu_seconds']:
            raise ValueError('second mandatory burn-in batch cannot be reserved')
        extras=dict(batch=1,variants=[v['id'] for v in registry.variants(family)],worlds=16,cases=64,
            canary_key=secrets.token_hex(32),pilot_complete_sha256=digest(pilot/'COMPLETE.json'),
            pilot_estimated_scientific_cpu_seconds=lower_cost,second_burnin_reservation_seconds=lower_cost)
    source=campaign/'sources'/identity;source.mkdir(parents=True)
    repository=Path(__file__).resolve().parents[1]
    names=['ghostscale/__init__.py','ghostscale/validation/__init__.py','ghostscale/validation/soundingline/__init__.py',
           'runners/run_v23.py','runners/launch_background.py']
    names += [p.relative_to(repository).as_posix() for p in sorted((repository/'ghostscale/validation/soundingline/v23').glob('*.py'))]
    for name in names:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repository/name,target)
    if digest(source/'ghostscale/validation/soundingline/v23/adaptive.py')!=calibration['adaptive_source_sha256']:
        raise ValueError('family policy differs from calibrated source')
    archive=source.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):z.write(source/name,name)
    dependency_files={str(p):digest(p) for p in (campaign/'dependencies').rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll')}
    plan=dict(task=task,family=family,budget_group='setup' if task=='pilot' else 'repairs',cpu_cap_seconds=cpu_cap,
        wall_cap_seconds=cpu_cap*4,protected=False,acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
        source_archive=str(archive),source_archive_sha256=digest(archive),sources={name:digest(source/name) for name in names},
        dependency_files=dependency_files,schema_directory=schema_pin['directory'],schema_pin=schema_pin,**extras)
    write(root/'PLAN.json',plan)
    write(root/'ADMISSION.json',dict(at=datetime.now(timezone.utc).isoformat(),plan_sha256=digest(root/'PLAN.json'),
        calibration_complete_sha256=calibration['complete_sha256'],canonical_commit=schema_pin['commit'],
        existing_active=active,charged_cpu_seconds=total,reserved_cpu_seconds=reserved,new_cpu_cap_seconds=cpu_cap))
    if launch:
        python=Path(sys.executable)
        if python.name.lower()!='python.exe' or '.venv' not in str(python):raise ValueError('explicit scientific venv interpreter required')
        env=__import__('os').environ.copy();env['PYTHONPATH']=str(campaign/'dependencies')
        argv=[str(python.with_name('pythonw.exe')),str(source/'runners/launch_background.py'),'--cwd',str(source),
              '--log',str(root/'worker.log'),'--',str(python),'-B','-m','runners.run_v23','--root',str(root),'--campaign',str(campaign)]
        child=subprocess.Popen(argv,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        write(root/'LAUNCH.json',dict(at=datetime.now(timezone.utc).isoformat(),launcher_pid=child.pid))
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True,type=Path);p.add_argument('--family',required=True)
    p.add_argument('--task',choices=['pilot','family-batch'],required=True);p.add_argument('--cpu-cap',required=True,type=float)
    p.add_argument('--launch',action='store_true');a=p.parse_args();print(admit(a.campaign,a.family,a.task,a.cpu_cap,a.launch))
