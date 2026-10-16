"""Execute the frozen reviewed nomination order, with independent refitting.

No discovery order is inferred here. Operator-reviewed exhaustive dispositions
must already exist. Actual earlier completed evidence determines each e-LOND
threshold; elapsed time and a pending fit never count as a confirmation.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import argparse,time
from ghostscale.validation.soundingline.v23 import confirmation,runtime,family,resource_policy
from ghostscale.validation.soundingline.v23.common import read,write,digest,canonical
from runners.admit_v23_confirmation import admit
from runners import queue_v23_families as queue
from runners.watch_v23_events import owner,event


def step(campaign,state,config):
    if runtime.now()>=read(campaign/'ACCEPTANCE.json')['report_start'] or (campaign/'STOP').exists():return dict(state='STOPPED_AT_CUTOFF')
    order=Path(config['order'])
    if not order.exists():return dict(state='REVIEW_BOUNDARY',reason='Reviewed exhaustive discovery order is not yet frozen')
    if digest(order)!=config['order_sha256']:raise ValueError('confirmation order binding changed')
    entries=confirmation.validate_order(read(order));completed=[]
    for index,entry in enumerate(entries,1):
        name=entry['family'];settings=config['families'][name]
        for stage in ('fits','science'):
            root=campaign/'packets'/f'v23-{name.lower()}-confirmation-1'
            fits=root.with_name(root.name+'-fits')
            if stage=='fits':root=fits
            status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {}
            if status.get('state')=='failed':return dict(state='REVIEW_BOUNDARY',family=name,stage=stage,packet=root.name,reason='Retained failed confirmation apparatus')
            if not (root/'PLAN.json').exists():
                if queue.active_count(queue.inventory(campaign))>=resource_policy.worker_limit(campaign):return dict(state='RUNNING',phase='WAITING_FOR_SLOT')
                try:
                    admit(campaign,order,index,settings['card'],settings['review'],settings['first_packet'],settings['pilot'],None,True,
                        fit_only=stage=='fits',fit_packet=fits if stage=='science' else None,partitioned=stage=='science' and name!='N1')
                except ValueError as exc:
                    if any(term in str(exc) for term in ('reserve would be invaded','group cap','exceeds four CPU hours')):
                        write(state/(name+'-RESOURCE_DISPOSITION.json'),dict(state='RESOURCE_DEFERRED',family=name,stage=stage,reason=str(exc),order_sha256=config['order_sha256']))
                        return dict(state='REVIEW_BOUNDARY',family=name,stage=stage,reason=str(exc))
                    raise
                return dict(state='RUNNING',family=name,stage=stage,packet=root.name)
            plan=read(root/'PLAN.json')
            if plan.get('scheduler_only') and status.get('state') not in ('complete','failed'):
                from runners.partition_v23 import step as partition_step
                partition_step(campaign,root)
            if not (root/'COMPLETE.json').exists():return dict(state='RUNNING',family=name,stage=stage,packet=root.name)
        completed.append(root)
        review=root/'CONFIRMATION_REVIEW.json'
        if not review.exists():
            if queue.active_count(queue.inventory(campaign))>=resource_policy.worker_limit(campaign):return dict(state='RUNNING',phase='WAITING_FOR_REVIEW_SLOT')
            from runners.review_v23_family import replay_owners
            pp,_=family.verify_packet(root);owners=replay_owners(root,pp)
            result=confirmation.ordered_decisions(completed,order,config['order_sha256'])[-1]
            write(review,dict(state='VERIFIED_CONFIRMATION',family=name,decision=result,
                complete_sha256=digest(root/'COMPLETE.json'),evidence_sha256=digest(root/'original/EVIDENCE.json'),
                order_sha256=config['order_sha256'],original_and_replay_owners=owners))
            event(campaign,name.lower()+'-confirmation-reviewed','family_decision',dict(packet=root.name,review_sha256=digest(review)))
        elif read(review)['complete_sha256']!=digest(root/'COMPLETE.json'):raise ValueError('confirmation review changed')
    return dict(state='COMPLETE',confirmations=len(completed),order_sha256=config['order_sha256'])


def run(campaign,state,config_path):
    config=read(config_path);identity=config['service_id'];started=time.process_time();runtime.below_normal()
    if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in identity):raise ValueError('invalid service identity')
    charge=campaign/'attempts'/(identity+'.json')
    if charge.exists():raise ValueError('preserve earlier service accounting')
    with owner(state):
        try:
            while True:
                result=step(campaign,state,config)
                write(state/'STATUS.json',dict(result,pid=os.getpid(),heartbeat=runtime.now(),config_sha256=digest(config_path)),immutable=False)
                write(charge,dict(cpu_seconds=time.process_time()-started,state=result['state']),immutable=False)
                if result['state']!='RUNNING':event(campaign,identity+'-boundary','queue_review_boundary',result);return
                time.sleep(30)
        except BaseException as exc:
            write(state/'STATUS.json',dict(state='failed',pid=os.getpid(),heartbeat=runtime.now(),error=repr(exc)),immutable=False)
            write(charge,dict(cpu_seconds=time.process_time()-started,state='failed'),immutable=False)
            event(campaign,identity+'-failed','instrument_failure',dict(error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('campaign','state','config'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
