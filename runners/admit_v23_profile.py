"""Freeze a bounded exact-predictor dependency study; never read human outcomes."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from pathlib import Path
import shutil
import time
import zipfile
from ghostscale.validation.soundingline.v23 import profile_validation as study, runtime
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.queue_v23_families import inventory,capacity,launch
from runners.watch_v23_events import owner


def admit(campaign,sibling,pilot=False,start=False):
    campaign=campaign.resolve();sibling=sibling.resolve()
    identity='v23-s15-profile-'+('pilot-2' if pilot else 'validation-1')
    cap=900 if pilot else 7200
    with owner(campaign/'admission'):
        if capacity(campaign,inventory(campaign),cap,budget_group='twins')!='AVAILABLE':
            raise ValueError('profile validation lacks allocation or worker slot')
        acceptance=read(campaign/'ACCEPTANCE.json')
        if not acceptance['opened']<=runtime.now()<acceptance['report_start']:
            raise ValueError('outside allocation')
        source_plan=sibling/'results/phase_2_4_stage_15/raw/profile-features-v1/PLAN.json'
        if digest(source_plan)!=study.SL_PLAN:raise ValueError('Sounding Line frozen plan changed')
        sibling_plan=read(source_plan)
        engine=campaign/'canonical/82789fdb50d9366ab9352129b01ee37622554dfa'
        pin=read(engine/'ENGINE_PIN.json');engine_source=engine/'engine-snapshot'
        for relative,value in pin['files'].items():
            if digest(engine_source/relative)!=value:raise ValueError('engine snapshot changed')
        if runtime.environment()!=read(campaign/'ENVIRONMENT.json'):raise ValueError('environment differs')
        pilot_binding={}
        if not pilot:
            prior=campaign/'packets/v23-s15-profile-pilot-2'
            if read(prior/'COMPLETE.json')['state']!='verified':raise ValueError('cost pilot unverified')
            if read(prior/'PLAN.json')['configuration']!=study.configuration():raise ValueError('pilot configuration changed')
            relative='ghostscale/validation/soundingline/v23/profile_validation.py'
            if read(prior/'PLAN.json')['sources'][relative]!=digest(Path(__file__).resolve().parents[1]/relative):
                raise ValueError('pilot source changed')
            measured=max(read(prior/label/'w00-planted/fit-cost.json')['cpu_seconds'] for label in ('original','replay'))
            estimate=measured*20*5*2*1.5+120
            if estimate>cap:raise ValueError(f'cost pilot projects {estimate:.1f}s above {cap}s cap')
            pilot_binding=dict(cost_pilot_complete_sha256=digest(prior/'COMPLETE.json'),projected_cpu_seconds=estimate)
        root=campaign/'packets'/identity
        if root.exists():raise ValueError('packet identity exists')
        source=campaign/'sources'/identity;source.mkdir(parents=True,exist_ok=False)
        repo=Path(__file__).resolve().parents[1]
        names=['ghostscale/__init__.py','ghostscale/validation/__init__.py','ghostscale/validation/soundingline/__init__.py',
            'runners/run_v23.py','runners/launch_background.py','runners/review_v23_profile.py',
            'docs/versions/v23-research-division/S15_PROFILE_PROTOCOL.md']
        names += [p.relative_to(repo).as_posix() for p in sorted((repo/'ghostscale/validation/soundingline/v23').glob('*.py'))]
        for name in names:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
        for name,expected,filename in (
            ('runners/next_round/profile_features.py',study.PROFILE_HASH,'profile_features.py.txt'),
            ('runners/stage14_midweek/baselines.py',study.TASK_HASH,'task_features.py.txt')):
            if sibling_plan['pins'][name]!=expected or digest(sibling/name)!=expected:
                raise ValueError('sibling executable source differs from frozen plan')
            target=source/'upstream'/filename;target.parent.mkdir(exist_ok=True)
            shutil.copyfile(sibling/name,target);names.append('upstream/'+filename)
        archive=source.with_suffix('.zip')
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name in sorted(names):z.write(source/name,name)
        dependencies={str(p):digest(p) for folder in ('dependencies','bridge-dependencies')
            for p in (campaign/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll')}
        plan=dict(task='profile-pilot' if pilot else 'profile-validation',family='S15-PROFILE',budget_group='twins',
            cpu_cap_seconds=cap,wall_cap_seconds=cap*4,protected=False,acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
            source_archive=str(archive),source_archive_sha256=digest(archive),sources={name:digest(source/name) for name in names},
            dependency_files=dependencies,external_source_files={str(engine_source/name):value for name,value in pin['files'].items()},
            extra_pythonpath=[str(campaign/'bridge-dependencies'),str(engine_source/'engine')],
            profile_source=str(source/'upstream/profile_features.py.txt'),task_source=str(source/'upstream/task_features.py.txt'),
            sounding_line_plan_sha256=study.SL_PLAN,implementation_hash=study.IMPLEMENTATION,ledger_schema_hash=study.SCHEMA,
            configuration=study.configuration(),**pilot_binding)
        write(root/'PLAN.json',plan)
        write(root/'ADMISSION.json',dict(at=runtime.now(),plan_sha256=digest(root/'PLAN.json'),
            charged_cpu_seconds=runtime.cost(campaign),new_cap_seconds=cap,
            authority='owner requested Sounding Line dependency implementation, within V23 total and protected reserve',
            constructed_software_fixture=pilot,scientific_family_confirmation=False))
        if start:launch(root,campaign,source)
        return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--sibling',type=Path,required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--launch',action='store_true')
    a=p.parse_args();started=time.process_time()
    try:print(admit(a.campaign,a.sibling,a.pilot,a.launch))
    finally:write(a.campaign/'attempts'/('profile-admission-'+str(time.time_ns())+'.json'),dict(cpu_seconds=time.process_time()-started,kind='dependency-study-admission'))
