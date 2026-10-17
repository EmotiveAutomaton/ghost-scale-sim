"""Finite empirical-shape twin queue; missing measurements never become inputs."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import argparse,time
from ghostscale.validation.soundingline.v23 import runtime,twin_admission,twin_contract,twin_pipeline,resource_policy
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23.family import verify_packet
from runners import queue_v23_families as queue
from runners.admit_v23_extension import admit
from runners.queue_v23_extensions import binding
from runners.watch_v23_events import owner,event


def task(campaign,family,identity,kind,cap,extras):
    root=campaign/'packets'/identity
    if not (root/'PLAN.json').exists():
        if cap>14400 and not (kind=='twin-batch' and extras.get('partitioned_twin')):return root,'RESOURCE_DEFERRED'
        if queue.active_count(queue.inventory(campaign))>=resource_policy.worker_limit(campaign):return root,'WAITING_FOR_SLOT'
        try:admit(campaign,identity,kind,family,cap,extras,group='twins')
        except ValueError as exc:
            if any(word in str(exc) for word in ('GLOBAL_CPU_RESERVE','FAMILY_GROUP_CAP')):return root,'RESOURCE_DEFERRED'
            if 'NO_SLOT' in str(exc):return root,'WAITING_FOR_SLOT'
            raise
        return root,'RUNNING_OR_ADMITTED'
    status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {}
    if read(root/'PLAN.json').get('scheduler_only') and status.get('state') not in ('complete','failed'):
        from runners.partition_v23 import step as partition_step
        partition_step(campaign,root)
        status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {}
    if status.get('state')=='failed':return root,'INSTRUMENT_FAILED'
    if not (root/'COMPLETE.json').exists():return root,'RUNNING_OR_ADMITTED'
    review=root/'TWIN_REVIEW.json'
    if not review.exists():
        plan,summary=verify_packet(root)
        numerical=twin_pipeline.reconstruct(root/'original',plan) if kind=='twin-batch' else None
        write(review,dict(state='VERIFIED',complete_sha256=digest(root/'COMPLETE.json'),independent_reconstruction=numerical,scientific_engine_admission=False))
    elif read(review)['complete_sha256']!=digest(root/'COMPLETE.json'):raise ValueError('twin acceptance changed')
    return root,'VERIFIED'


def step(campaign,state,config):
    if runtime.now()>=read(campaign/'ACCEPTANCE.json')['report_start'] or (campaign/'STOP').exists():return dict(state='STOPPED_AT_CUTOFF')
    decisions={}
    for family in ('T1','T4','T3','T2'):
        item=config['families'][family]
        if not item.get('empirical_reference'):
            decisions[family]=dict(state='DEPENDENCY_HELD',missing_measured_fields=list(twin_contract.REQUIRED[family]),
                source_authentication='Committed source revision, source bytes SHA-256, citation and the corresponding locally retained source file')
            continue
        extras={key:item[key] for key in ('empirical_reference','empirical_reference_sha256','empirical_source')}
        extras.update(config['implementation_controls'])
        extras.update(config['engine_dependency'])
        pilot,outcome=task(campaign,family,'v23-'+family.lower()+'-profile-1','twin-profile',item['profile_cap'],extras)
        if outcome!='VERIFIED':decisions[family]=dict(state=outcome,stage='profile',packet=pilot.name);continue
        extras['partitioned_twin']=bool(resource_policy.binding(campaign))
        try:costs=twin_admission.estimate(pilot,family,partitioned=extras['partitioned_twin'])
        except ValueError as exc:
            if 'individual packet limit' not in str(exc):raise
            decisions[family]=dict(state='RESOURCE_DEFERRED',reason='One whole twin world exceeds the packet limit; makers cannot be split into worlds.')
            continue
        extras.update(binding(pilot,'pilot'))
        fits,outcome=task(campaign,family,'v23-'+family.lower()+'-fits-1','twin-fits',costs['fits'],dict(extras,burnin_reservations_seconds=[costs['batch']]))
        if outcome!='VERIFIED':decisions[family]=dict(state=outcome,stage='fitting',packet=fits.name,estimated_cpu_seconds=costs);continue
        extras.update(binding(fits,'fit'));folder=fits/'original/fits'
        extras.update(fit_directory=str(folder),fit_hashes={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()},batch=1,worlds=16)
        batch,outcome=task(campaign,family,'v23-'+family.lower()+'-batch-1','twin-batch',costs['batch'],extras)
        decisions[family]=dict(state='COMPLETE' if outcome=='VERIFIED' else outcome,stage='full-grid',packet=batch.name)
    terminal={'DEPENDENCY_HELD','RESOURCE_DEFERRED','INSTRUMENT_FAILED','COMPLETE'}
    return dict(state='REVIEW_BOUNDARY' if all(row['state'] in terminal for row in decisions.values()) else 'RUNNING',families=decisions)


def run(campaign,state,config_path):
    config=read(config_path);identity=config['service_id'];started=time.process_time();runtime.below_normal()
    if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in identity):raise ValueError('invalid service identity')
    charge=campaign/'attempts'/(identity+'.json')
    if charge.exists():raise ValueError('preserve previous service accounting')
    with owner(state):
        try:
            while True:
                result=queue.admission_step(campaign,step,campaign,state,config)
                write(state/'STATUS.json',dict(result,pid=os.getpid(),heartbeat=runtime.now(),config_sha256=digest(config_path)),immutable=False)
                write(charge,dict(cpu_seconds=time.process_time()-started,state=result['state']),immutable=False)
                if result['state']!='RUNNING':event(campaign,identity+'-boundary','queue_review_boundary',result);return
                time.sleep(30)
        except BaseException as exc:
            write(state/'STATUS.json',dict(state='failed',pid=os.getpid(),heartbeat=runtime.now(),error=repr(exc)),immutable=False)
            write(charge,dict(cpu_seconds=time.process_time()-started,state='failed'),immutable=False)
            event(campaign,identity+'-failed','instrument_failure',dict(error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('campaign','state','config'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();run(args.campaign.resolve(),args.state.resolve(),args.config.resolve())
