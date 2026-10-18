"""Finite October 18 queue: replayed cost gates, predeclared shedding, one batch."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v23 import runtime,resource_policy,directed_common as science
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners import queue_v23_families as queue
from runners.queue_v23_extensions import binding,checked
from runners.admit_v23_extension import admit
from runners.watch_v23_events import owner,event

ORDER=('N16','N17','N16b','N18')
TERMINAL={'COMPLETE','RESOURCE_DEFERRED','INSTRUMENT_FAILED','KNOWN_ANSWER_FAILED'}
SERVICE_PULSE=lambda **kwargs:None

def designs(family):
    model=science.model_for(family);variants=model.variants();result=[{}]
    if family=='N16':
        result += [dict(budgets=[1,2,4,6]),dict(budgets=[1,2,4,6],reliability_misspecification=False)]
        # All source studies have twelve settings: ties use frozen source order.
        keep=[v['id'] for i,v in enumerate(variants) if v['source']!='N4d' or i%2==0]
        result.append(dict(result[-1],variant_ids=keep))
    if family=='N17':
        keep=[v['id'] for v in variants if v['history']!=250]
        result += [dict(variant_ids=keep),dict(variant_ids=keep,worlds=8)]
    return result

def charged(campaign,family):
    return sum(float(status.get('cpu_seconds',0)) for _,plan,status in queue.inventory(campaign)
        if plan.get('family')==family and not plan.get('logical_parent'))

def ensure(campaign,identity,task,family,cap,extras):
    packet=campaign/'packets'/identity
    if (packet/'PLAN.json').exists():
        plan=read(packet/'PLAN.json');status=read(packet/'STATUS.json') if (packet/'STATUS.json').exists() else {}
        if status.get('state')=='failed':return packet,'INSTRUMENT_FAILED'
        if plan.get('scheduler_only') and status.get('state') not in ('failed','complete'):
            from runners.partition_v23 import step
            step(campaign,packet)
        # A review is scientific CPU work too; do not exceed Gear 2 while checking.
        if (packet/'COMPLETE.json').exists() and not (packet/'EXTENSION_REVIEW.json').exists() and queue.active_count(queue.inventory(campaign))>=resource_policy.worker_limit(campaign):
            return packet,'WAITING_FOR_REVIEW_SLOT'
        if task=='conditional-batch' and (packet/'COMPLETE.json').exists() and not (packet/'EXTENSION_REVIEW.json').exists() and not (packet/'NUMERICAL_REVIEW.json').exists():
            from ghostscale.validation.soundingline.v23.family import verify_packet
            started=time.process_time();before=dict(status)
            with owner(campaign/'admission'):
                if queue.active_count(queue.inventory(campaign))>=resource_policy.worker_limit(campaign):return packet,'WAITING_FOR_REVIEW_SLOT'
                write(packet/'REVIEW_START_STATUS.json',before)
                write(packet/'STATUS.json',dict(before,state='reviewing',pid=os.getpid(),runtime_owner_pid=before.get('pid'),heartbeat=runtime.now()),immutable=False)
            def pulse():
                used=time.process_time()-started
                write(packet/'STATUS.json',dict(before,state='reviewing',phase='independent-reconstruction',pid=os.getpid(),
                    runtime_owner_pid=before.get('pid'),cpu_seconds=before.get('cpu_seconds',0)+used,heartbeat=runtime.now()),immutable=False)
                SERVICE_PULSE(review_packet=packet.name)
                if before.get('cpu_seconds',0)+used>plan['cpu_cap_seconds']:raise TimeoutError('reserved independent review cap reached')
                acceptance=resource_policy.resolve(campaign,plan)
                if runtime.cost(campaign)>=acceptance['cpu_seconds']-acceptance['protected_cpu_seconds']:raise TimeoutError('review reached protected reserve')
                if runtime.now()>=acceptance['deadline'] or (campaign/'STOP').exists():raise TimeoutError('review deadline')
            try:
                verify_packet(packet);pulse()
                review=science.reconstruct(packet/'original',plan,pulse)
                write(packet/'NUMERICAL_REVIEW.json',review)
                write(packet/'EXTENSION_REVIEW.json',dict(state='VERIFIED',complete_sha256=digest(packet/'COMPLETE.json'),task=task,
                    full_fresh_process_replay=True,independent_numerical_review=review,
                    independent_checker_sha256=digest(science.__file__)))
                write(packet/'STATUS.json',dict(before,state='complete',phase='independently-reviewed',
                    cpu_seconds=before.get('cpu_seconds',0)+time.process_time()-started,
                    independent_review_cpu_seconds=time.process_time()-started,heartbeat=runtime.now()),immutable=False)
                return packet,'VERIFIED'
            except BaseException as exc:
                write(packet/'REVIEW_FAILURE.json',dict(error=repr(exc),at=runtime.now()))
                write(packet/'STATUS.json',dict(before,state='failed',phase='independent-review',error=repr(exc),
                    cpu_seconds=before.get('cpu_seconds',0)+time.process_time()-started,heartbeat=runtime.now()),immutable=False)
                raise
        return packet,checked(packet)
    try:admit(campaign,identity,task,family,cap,extras,group='conditional')
    except ValueError as exc:
        message=str(exc)
        if 'NO_SLOT' in message:return packet,'WAITING_FOR_SLOT'
        if any(x in message for x in ('GLOBAL_CPU_RESERVE','FAMILY_GROUP_CAP','DIRECTED_FAMILY_CAP')):return packet,'RESOURCE_DEFERRED'
        raise
    return packet,'RUNNING_OR_ADMITTED'

def step_family(campaign,family):
    attempts=[]
    for index,design in enumerate(designs(family),1):
        pilot,state=ensure(campaign,f'v23-{family.lower()}-directed-profile-{index}','conditional-pilot',family,1200,dict(directed_design=design))
        if state!='VERIFIED':return dict(state=state,stage='cost-pilot',packet=pilot.name,design=design)
        costs=science.estimate(pilot,design);used=charged(campaign,family)
        attempts.append(dict(pilot=pilot.name,design=design,estimated=costs,charged_before=used))
        if used+costs['fits']+costs['batch']>science.CAPS[family]:continue
        extras=dict(binding(pilot,'pilot'),directed_design=design)
        fit,state=ensure(campaign,f'v23-{family.lower()}-directed-fits-1','conditional-fits',family,costs['fits'],
            dict(extras,burnin_reservations_seconds=[costs['batch']]))
        if state!='VERIFIED':return dict(state=state,stage='frozen-reader-definitions',packet=fit.name,costs=costs,design=design)
        extras.update(binding(fit,'fit'));fitroot=fit/'original/fits'
        extras.update(fit_directory=str(fitroot),fit_hashes={p.name:digest(p) for p in fitroot.iterdir() if p.is_file()},
            batch=1,worlds=design.get('worlds',16),cases=64,variants=[v['id'] for v in science.selected(science.model_for(family),dict(directed_design=design))],
            partitioned_directed=bool(costs['partition_layout']))
        packet,state=ensure(campaign,f'v23-{family.lower()}-batch-1','conditional-batch',family,costs['batch'],extras)
        return dict(state='COMPLETE' if state=='VERIFIED' else state,stage='reserved-conditional-batch',packet=packet.name,costs=costs,design=design)
    return dict(state='RESOURCE_DEFERRED',stage='after-prescribed-shedding',attempts=attempts,
        curator_trigger=bool(attempts and attempts[-1]['estimated']['batch']+attempts[-1]['estimated']['fits']>1.5*science.CAPS[family]))

def step(campaign):
    if runtime.now()>=read(campaign/'ACCEPTANCE.json')['report_start'] or (campaign/'STOP').exists():return dict(state='STOPPED_AT_CUTOFF')
    states={}
    # Explicit priority list: canonical JSON key sorting cannot reorder families.
    for family in ORDER:
        state=step_family(campaign,family);states[family]=state
        if state['state'] in TERMINAL:
            event(campaign,'directed-'+family.lower()+'-boundary','queue_review_boundary',dict(family=family,**state))
    terminal=all(x['state'] in TERMINAL for x in states.values())
    return dict(state='REVIEW_BOUNDARY' if terminal else 'RUNNING',families=states)

def run(campaign,state,config_path):
    started=time.process_time();runtime.below_normal();config=read(config_path);identity=config['service_id']
    charge=campaign/'attempts'/(identity+'.json')
    if charge.exists():raise ValueError('preserve prior directed controller identity')
    with owner(state):
        global SERVICE_PULSE
        def service_pulse(**fields):
            write(state/'STATUS.json',dict(state='RUNNING',pid=os.getpid(),heartbeat=runtime.now(),config_sha256=digest(config_path),**fields),immutable=False)
            write(charge,dict(cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
        SERVICE_PULSE=service_pulse
        try:
            while True:
                if (state/'STOP').exists():
                    write(state/'STATUS.json',dict(state='OPERATOR_HANDOFF',pid=os.getpid(),heartbeat=runtime.now()),immutable=False);return
                # admit() owns the admission lock; an outer lock would deadlock.
                result=queue.admission_step(campaign,step,campaign)
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
