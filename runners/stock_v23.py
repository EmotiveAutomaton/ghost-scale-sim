"""Finite, source-frozen preparation queue for explicitly commissioned families.

Verified pilots lead to bounded rival fits, then full first batches and the
calibrated continuation. Failures/caps stop that family; no automatic repairs,
new variants, criterion changes or publication are performed here.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
import argparse,math,time
from pathlib import Path
from ghostscale.validation.soundingline.v23 import runtime,registry
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23.family import verify_packet
from runners import admit_v23_repair as admission,queue_v23_families as queue
from runners.watch_v23_events import owner,event


def admit_or_hold(campaign,state,name,task,cap,partitioned,repair=False):
    try:
        admission.admit(campaign,name,task,cap,launch=True,partitioned=partitioned,repair=repair)
        return True
    except ValueError as exc:
        message=str(exc)
        if not any(term in message for term in ('reserve would be invaded','cannot be reserved','group cap exhausted','family group cap')):raise
        write(state/(name+'-DISPOSITION.json'),dict(state='RESOURCE_DEFERRED',family=name,at=runtime.now(),
            reason=message,scientific_result=False,task=task,cpu_cap_seconds=cap))
        return False


def step(campaign,state,config):
    decisions={}
    acceptance=read(campaign/'ACCEPTANCE.json')
    if runtime.now()>=acceptance['report_start'] or (campaign/'STOP').exists():return dict(state='STOPPED_AT_CUTOFF',families={})
    if (state/'STOP').exists():return dict(state='OPERATOR_STOP',families={})
    for name in registry.ORDER:
        if name not in config['families']:continue
        item=config['families'][name]
        disposition=state/(name+'-DISPOSITION.json')
        first=campaign/'packets'/f'v23-{name.lower()}-batch-1'
        if (first/'PLAN.json').exists():
            decisions.update(queue.step(campaign,dict(families={name:dict(first_packet=str(first))}))['families']);continue
        if disposition.exists():decisions[name]=read(disposition)['state'];continue
        pilot=Path(item['pilot']);status=read(pilot/'STATUS.json') if (pilot/'STATUS.json').exists() else {}
        if not (pilot/'PLAN.json').exists() and item.get('pilot_cpu_cap'):
            if queue.active_count(queue.inventory(campaign))>=acceptance['max_scientific_workers']:
                decisions[name]='WAITING_FOR_PILOT_SLOT';continue
            admitted=admission.admit(campaign,name,'pilot',item['pilot_cpu_cap'],launch=True,repair=item.get('pilot_repair',False))
            if admitted.resolve()!=pilot.resolve():raise ValueError('configured pilot identity differs')
            decisions[name]='PILOT_RUNNING';continue
        if status.get('state')=='failed':
            result=dict(state='PILOT_FAILED',pilot=pilot.name,status_sha256=digest(pilot/'STATUS.json'),
                scientific_result=False,repair_requires_review=True)
        elif not (pilot/'COMPLETE.json').exists():decisions[name]='WAITING_FOR_PILOT';continue
        else:
            cached=state/(name+'-PILOT_ACCEPTANCE.json')
            bindings=dict(plan=digest(pilot/'PLAN.json'),complete=digest(pilot/'COMPLETE.json'),
                costs={label:digest(pilot/label/'pilot-cost.json') for label in ('original','replay')})
            if cached.exists():
                if read(cached)!=bindings:raise ValueError('previously checked pilot changed')
            else:
                if queue.active_count(queue.inventory(campaign))>=acceptance['max_scientific_workers']:
                    decisions[name]='WAITING_FOR_PILOT_REVIEW_SLOT';continue
                plan,summary=verify_packet(pilot)
                if summary.get('state')!='PILOT_VALID':raise ValueError('unexpected pilot state')
                write(cached,bindings)
            costs=[read(pilot/label/'pilot-cost.json') for label in ('original','replay')]
            new_family=name.startswith('N')
            estimate=admission.profile_estimate(costs,name,new_family)
            partitioned=item.get('partitioned',False)
            if partitioned:
                from ghostscale.validation.soundingline.v23.partition import layout
                estimate=layout(16,estimate,2*max(row['auxiliary_cpu_seconds'] for row in costs) if new_family else 0.)['total_cpu_cap']
            limit=admission.budget_limits(campaign)[name.lower() if new_family else 'repairs']*3600
            result=None
            if (estimate>14400 and not partitioned) or 2*estimate+item['fit_cpu_cap']>limit:
                result=dict(state='RESOURCE_DEFERRED',pilot=pilot.name,pilot_complete_sha256=digest(pilot/'COMPLETE.json'),
                    estimated_batch_cpu_seconds=estimate,mandatory_burnins=2,fit_cap_seconds=item['fit_cpu_cap'],
                    group_cap_seconds=limit,packet_cap_seconds=14400,scientific_result=False)
            else:
                fits=campaign/'packets'/f'v23-{name.lower()}-fits-1'
                if item.get('fit_repair'):fits=fits.with_name(fits.name+'-repair-1')
                fs=read(fits/'STATUS.json') if (fits/'STATUS.json').exists() else {}
                if fs.get('state')=='failed':
                    result=dict(state='FITS_FAILED',packet=fits.name,status_sha256=digest(fits/'STATUS.json'),
                        scientific_result=False,repair_requires_review=True)
                elif not (fits/'PLAN.json').exists():
                    entries=queue.inventory(campaign)
                    if queue.active_count(entries)>=acceptance['max_scientific_workers']:
                        decisions[name]='WAITING_FOR_SLOT';continue
                    admitted=admit_or_hold(campaign,state,name,'fits',item['fit_cpu_cap'],partitioned,repair=item.get('fit_repair',False))
                    decisions[name]='FITS_RUNNING' if admitted else 'RESOURCE_DEFERRED';continue
                elif not (fits/'COMPLETE.json').exists():decisions[name]='FITS_RUNNING';continue
                else:
                    entries=queue.inventory(campaign)
                    if queue.active_count(entries)>=acceptance['max_scientific_workers']:
                        decisions[name]='WAITING_FOR_SLOT';continue
                    if not admit_or_hold(campaign,state,name,'family-batch',math.ceil(estimate),partitioned):
                        decisions[name]='RESOURCE_DEFERRED';continue
                    decisions.update(queue.step(campaign,dict(families={name:dict(first_packet=str(first))}))['families']);continue
        if result is not None:
            write(disposition,dict(result,family=name,at=runtime.now()))
            decisions[name]=result['state']
    terminal={'PILOT_FAILED','FITS_FAILED','RESOURCE_DEFERRED','NOMINATED','STOPPED_BELOW_MARGIN',
              'INCONCLUSIVE_AT_CAP','INSTRUMENT_FAILED','PRIMARY_UNAVAILABLE','GLOBAL_CPU_RESERVE',
              'FAMILY_GROUP_CAP','REPAIRS_GROUP_CAP'}
    return dict(state='REVIEW_BOUNDARY' if decisions and all(s in terminal for s in decisions.values()) else 'RUNNING',families=decisions)


def run(campaign,state,config_path):
    config=read(config_path);identity=config['service_id']
    if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in identity):raise ValueError('invalid service identity')
    receipt=campaign/'attempts'/(identity+'.json')
    if receipt.exists():raise ValueError('preserve existing service accounting')
    runtime.below_normal();started=time.process_time()
    with owner(state):
        try:
            while True:
                result=step(campaign,state,config)
                write(state/'STATUS.json',dict(result,pid=os.getpid(),heartbeat=runtime.now(),config_sha256=digest(config_path)),immutable=False)
                write(receipt,dict(cpu_seconds=time.process_time()-started,state=result['state']),immutable=False)
                if result['state']!='RUNNING':event(campaign,identity+'-boundary','queue_review_boundary',result);return
                time.sleep(30)
        except BaseException as exc:
            write(state/'STATUS.json',dict(state='failed',pid=os.getpid(),heartbeat=runtime.now(),error=repr(exc)),immutable=False)
            write(receipt,dict(cpu_seconds=time.process_time()-started,state='failed'),immutable=False)
            event(campaign,identity+'-failed','instrument_failure',dict(error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
