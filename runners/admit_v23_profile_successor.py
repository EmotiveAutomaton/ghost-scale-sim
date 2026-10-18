"""Admit the fixed D5 pilot or all disjoint successor partitions."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
from pathlib import Path
import shutil
import time
import zipfile
from ghostscale.validation.soundingline.v23 import profile_validation as base,profile_successor as study,runtime,resource_policy
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.queue_v23_families import inventory,capacity,launch
from runners.watch_v23_events import owner


def sources(campaign,sibling,identity):
    source=campaign/'sources'/identity;source.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    names=['ghostscale/__init__.py','ghostscale/validation/__init__.py','ghostscale/validation/soundingline/__init__.py',
        'runners/run_v23.py','runners/launch_background.py','runners/review_v23_profile_successor.py',
        'docs/versions/v23-research-division/S15_SUCCESSOR_PROTOCOL.md']
    names += [p.relative_to(repo).as_posix() for p in sorted((repo/'ghostscale/validation/soundingline/v23').glob('*.py'))]
    for name in names:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
    slplan=read(sibling/'results/phase_2_4_stage_15/raw/profile-features-v1/PLAN.json')
    for name,expected,filename in (
        ('runners/next_round/profile_features.py',base.PROFILE_HASH,'profile_features.py.txt'),
        ('runners/stage14_midweek/baselines.py',base.TASK_HASH,'task_features.py.txt')):
        if slplan['pins'][name]!=expected or digest(sibling/name)!=expected:raise ValueError('sibling source differs')
        target=source/'upstream'/filename;target.parent.mkdir(exist_ok=True);shutil.copyfile(sibling/name,target);names.append('upstream/'+filename)
    archive=source.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):z.write(source/name,name)
    return source,archive,names


def plan_for(campaign,sibling,engine,identity,task,worlds,cap,source,archive,names,extra):
    pin=read(engine/'ENGINE_PIN.json');snapshot=engine/'engine-snapshot'
    for name,value in pin['files'].items():
        if digest(snapshot/name)!=value:raise ValueError('certified snapshot changed')
    dependencies={str(p):digest(p) for folder in ('dependencies','bridge-dependencies')
        for p in (campaign/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll')}
    return dict(task=task,family='S15-PROFILE-SUCCESSOR',budget_group='repairs',world_indices=worlds,
        cpu_cap_seconds=cap,wall_cap_seconds=cap*4,protected=False,
        acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),source_archive=str(archive),source_archive_sha256=digest(archive),
        sources={name:digest(source/name) for name in names},dependency_files=dependencies,
        external_source_files={str(snapshot/name):value for name,value in pin['files'].items()},
        extra_pythonpath=[str(campaign/'bridge-dependencies'),str(snapshot/'engine')],
        profile_source=str(source/'upstream/profile_features.py.txt'),task_source=str(source/'upstream/task_features.py.txt'),
        sounding_line_plan_sha256=base.SL_PLAN,implementation_hash=study.ENGINE_IMPLEMENTATION,ledger_schema_hash=base.SCHEMA,
        certified_fit_tag=study.TAG,certified_fit_tag_object=study.TAG_OBJECT,engine_commit=study.ENGINE_COMMIT,
        certification_sha256=study.CERTIFICATION_SHA256,configuration=study.configuration(),
        **extra,**resource_policy.binding(campaign))


def admit(campaign,sibling,pilot=False,partition_size=10,start=False):
    campaign=campaign.resolve();sibling=sibling.resolve()
    acceptance=resource_policy.resolve(campaign)
    if not acceptance['opened']<=runtime.now()<acceptance['report_start']:raise ValueError('outside admission window')
    if runtime.environment()!=read(campaign/'ENVIRONMENT.json'):raise ValueError('environment differs')
    engine=campaign/'canonical'/(study.ENGINE_COMMIT+'-'+study.TAG)
    if not (engine/'ENGINE_PIN.json').exists():raise ValueError('certified engine snapshot absent')
    with owner(campaign/'admission'):
        if pilot:
            identity='v23-s15-profile-successor-pilot-1';worlds=[1000];cap=7200
            if (campaign/'packets'/identity).exists():raise ValueError('pilot identity exists')
            if capacity(campaign,inventory(campaign),cap,budget_group='repairs')!='AVAILABLE':raise ValueError('pilot capacity unavailable')
            source,archive,names=sources(campaign,sibling,identity)
            plan=plan_for(campaign,sibling,engine,identity,'profile-successor-pilot',worlds,cap,source,archive,names,{})
            root=campaign/'packets'/identity;write(root/'PLAN.json',plan)
            write(root/'ADMISSION.json',dict(at=runtime.now(),plan_sha256=digest(root/'PLAN.json'),
                authority='October 17 packet D5 cost pilot; excluded from science',constructed_software_fixture=True))
            if start:launch(root,campaign,source)
            return [root]
        pilot_root=campaign/'packets/v23-s15-profile-successor-pilot-1'
        if read(pilot_root/'COMPLETE.json')['state']!='verified':raise ValueError('successor pilot unverified')
        observed=max(read(pilot_root/(label+'-progress.json'))['cpu_seconds'] for label in ('original','replay'))
        if partition_size<1 or partition_size>40:raise ValueError('invalid partition size')
        estimated_pass=observed*partition_size*1.5+120
        if estimated_pass>7200:raise ValueError(f'partition projects {estimated_pass:.1f}s per pass above cap')
        cap=max(1800,min(14400,2*estimated_pass))
        parts=[list(range(i,min(40,i+partition_size))) for i in range(0,40,partition_size)]
        required=cap*len(parts)
        if capacity(campaign,inventory(campaign),required,budget_group='repairs',needs_slot=False)!='AVAILABLE':
            raise ValueError('complete successor reservation unavailable')
        roots=[]
        for index,worlds in enumerate(parts,1):
            identity=f'v23-s15-profile-successor-part-{index}'
            if (campaign/'packets'/identity).exists():raise ValueError('partition identity exists')
            source,archive,names=sources(campaign,sibling,identity)
            plan=plan_for(campaign,sibling,engine,identity,'profile-successor-part',worlds,cap,source,archive,names,
                dict(logical_study='S15-PROFILE-FEATURE-V2',partition=index,partitions=len(parts),
                    pilot_complete_sha256=digest(pilot_root/'COMPLETE.json'),projected_pass_cpu_seconds=estimated_pass))
            root=campaign/'packets'/identity;write(root/'PLAN.json',plan)
            write(root/'ADMISSION.json',dict(at=runtime.now(),plan_sha256=digest(root/'PLAN.json'),
                authority='October 17 packet D5 complete forty-world reservation',worlds=worlds,
                complete_design_reserved=True,all_worlds=list(range(40))))
            roots.append(root)
        if start:
            for root in roots[:acceptance['max_scientific_workers']]:launch(root,campaign,Path(read(root/'PLAN.json')['source_archive']).with_suffix(''))
        return roots


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--sibling',type=Path,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--partition-size',type=int,default=10);p.add_argument('--launch',action='store_true')
    a=p.parse_args();start=time.process_time()
    try:
        for root in admit(a.campaign,a.sibling,a.pilot,a.partition_size,a.launch):print(root)
    finally:write(a.campaign/'attempts'/('profile-successor-admission-'+str(time.time_ns())+'.json'),dict(cpu_seconds=time.process_time()-start))
