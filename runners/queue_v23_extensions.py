"""Finite continuation of commissioned N1 and N4b implementation.

Controls precede largest-cell profiles; measured full costs precede frozen fits.
Every failure, resource deferral and completed result has an explicit boundary.
This controller neither invents studies nor retries executed identities.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import argparse,time
from ghostscale.validation.soundingline.v23 import runtime,population_campaign,conditional_values,support_new
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23.extension_contract import cost_estimate
from ghostscale.validation.soundingline.v23.family import verify_packet
from runners import queue_v23_families as queue
from runners.admit_v23_extension import admit
from runners.watch_v23_events import owner,event


def binding(packet,prefix):
    return {prefix+'_packet':str(packet),prefix+'_complete_sha256':digest(packet/'COMPLETE.json')}


def checked(packet):
    status=read(packet/'STATUS.json') if (packet/'STATUS.json').exists() else {}
    if status.get('state')=='failed':return 'INSTRUMENT_FAILED'
    if not (packet/'COMPLETE.json').exists():return 'RUNNING_OR_ADMITTED'
    receipt=packet/'EXTENSION_REVIEW.json'
    if receipt.exists():
        review=read(receipt)
        if review['complete_sha256']!=digest(packet/'COMPLETE.json'):raise ValueError('extension review changed')
        return review['state']
    plan,summary=verify_packet(packet);state='VERIFIED';numerical=None
    if plan['task']=='population-controls':
        if not summary.get('convergence_passed') or not summary.get('shuffle_passed'):state='KNOWN_ANSWER_FAILED'
    if plan['task'] in ('population-batch','population-precision'):
        numerical=population_campaign.reconstruct(packet/'original',plan)
    if plan['task']=='conditional-batch':
        module=conditional_values
        if plan['family']=='N4c':
            from ghostscale.validation.soundingline.v23 import conditional_devaluation as module
        numerical=module.reconstruct(packet/'original',plan)
    write(receipt,dict(state=state,complete_sha256=digest(packet/'COMPLETE.json'),task=plan['task'],
        full_fresh_process_replay=True,independent_numerical_review=numerical))
    return state


def ensure(campaign,identity,task,family,cap,extras,group):
    packet=campaign/'packets'/identity
    if (packet/'PLAN.json').exists():return packet,checked(packet)
    if cap>14400:
        return packet,'RESOURCE_DEFERRED'
    try:admit(campaign,identity,task,family,cap,extras,group=group)
    except ValueError as exc:
        message=str(exc)
        if 'NO_SLOT' in message:return packet,'WAITING_FOR_SLOT'
        if any(tag in message for tag in ('GLOBAL_CPU_RESERVE','FAMILY_GROUP_CAP','REPAIRS_GROUP_CAP')):
            return packet,'RESOURCE_DEFERRED'
        raise
    return packet,'RUNNING_OR_ADMITTED'


def step_family(campaign,name,config):
    population=name=='N1';prefix='population' if population else 'conditional';group='n1' if population else 'n4'
    extras={};pilot_task=prefix+'-profile' if population else prefix+'-pilot'
    if population:
        controls,state=ensure(campaign,'v23-n1-controls-1','population-controls',name,config['controls_cap'],{},group)
        if state!='VERIFIED':return dict(state=state,stage='controls',packet=controls.name)
        extras.update(binding(controls,'controls'))
    pilot,state=ensure(campaign,'v23-'+name.lower()+'-completion-profile-1',pilot_task,name,config['profile_cap'],extras,group)
    if state!='VERIFIED':return dict(state=state,stage='cost-profile',packet=pilot.name)
    extras.update(binding(pilot,'pilot'));costs=cost_estimate(prefix+'-fits',pilot)
    fitextras=dict(extras,burnin_reservations_seconds=[costs['batch']]*(2 if population else 1))
    if population:fitextras['confirmation_support']=support_new.bound('N1')
    fit,state=ensure(campaign,'v23-'+name.lower()+'-completion-fits-1',prefix+'-fits',name,costs['fits'],fitextras,group)
    if state!='VERIFIED':return dict(state=state,stage='frozen-fitting',estimated_cpu_seconds=costs,packet=fit.name)
    extras.update(binding(fit,'fit'));fitroot=fit/'original/fits'
    extras.update(fit_directory=str(fitroot),fit_hashes={p.name:digest(p) for p in fitroot.iterdir() if p.is_file()})
    if not population:
        packet,state=ensure(campaign,'v23-'+name.lower()+'-batch-1','conditional-batch',name,costs['batch'],dict(extras,batch=1,worlds=16,cases=64),group)
        return dict(state='COMPLETE' if state=='VERIFIED' else state,stage='conditional-batch',packet=packet.name)
    extras['confirmation_support']=support_new.bound('N1');paths=[];summaries=[];active=None
    acceptance=read(campaign/'CALIBRATION_ACCEPTANCE.json');summary=Path(acceptance['summary'])
    if not summary.is_absolute():summary=(campaign/summary).resolve()
    if digest(summary)!=acceptance['summary_sha256']:raise ValueError('calibration changed')
    policy=read(summary.parent/'CHOSEN_BEFORE_VALIDATION.json')
    decision=None
    for batch in range(1,8):
        allocation=population_campaign.design(batch,None if batch<=2 else active)
        fields=dict(extras,batch=batch,variants=[x['variant']['id'] for x in allocation],population_design=allocation)
        from ghostscale.validation.soundingline.v23.extension_contract import validate
        cap,_=validate('population-batch','N1',fields,Path(__file__).resolve().parents[1])
        packet,state=ensure(campaign,f'v23-n1-batch-{batch}','population-batch',name,cap,fields,group)
        if state!='VERIFIED':return dict(state=state,stage='population-discovery',batch=batch,packet=packet.name)
        paths.append(packet);summaries.append(read(packet/'original/SUMMARY.json'))
        card=packet/'MACHINE_CARD.json'
        if card.exists():decision=read(card)
        else:
            decision=population_campaign.decide(summaries,policy)
            write(packet/'NUMERICAL_REVIEW.json',read(packet/'EXTENSION_REVIEW.json')['independent_numerical_review'])
            write(card,dict(decision,complete_sha256=digest(packet/'COMPLETE.json'),
                numerical_review_sha256=digest(packet/'NUMERICAL_REVIEW.json'),calibration_complete_sha256=acceptance['complete_sha256'],
                predecessor_cards=[digest(p/'MACHINE_CARD.json') for p in paths[:-1]]))
        if decision['state'] not in ('BURN_IN','CONTINUE'):break
        active=decision['active_ids']
    # Map precision is independent of the selected-family stopping decision.
    # A finite per-packet limit and remaining original campaign budget bind it.
    precision_summaries=list(summaries[:2]);precision_batch=1
    while True:
        active=population_campaign.precision_candidates(precision_summaries)
        if not active:return dict(state='COMPLETE',stage='population-and-precision',decision=decision['state'],packets=[p.name for p in paths])
        allocation=population_campaign.design(precision_batch,active,precision=True)
        fields=dict(extras,batch=precision_batch,variants=active,population_design=allocation)
        cap,_=validate('population-precision','N1',fields,Path(__file__).resolve().parents[1])
        packet,state=ensure(campaign,f'v23-n1-precision-{precision_batch}','population-precision',name,cap,fields,group)
        if state!='VERIFIED':return dict(state=state,stage='map-precision',decision=decision['state'],packet=packet.name)
        precision_summaries.append(read(packet/'original/SUMMARY.json'));precision_batch+=1


def step(campaign,config):
    if runtime.now()>=read(campaign/'ACCEPTANCE.json')['report_start'] or (campaign/'STOP').exists():return dict(state='STOPPED_AT_CUTOFF')
    precedence=config.get('admission_precedence')
    if precedence and not (Path(precedence['packet'])/'PLAN.json').exists():
        status_path=Path(precedence['service_state'])/'STATUS.json'
        if not status_path.exists() or read(status_path)['state']=='RUNNING':
            return dict(state='RUNNING',phase='WAITING_FOR_PRIOR_AUTHORIZED_ADMISSION')
    from ghostscale.validation.soundingline.v23 import resource_policy
    if config.get('families') and queue.active_count(queue.inventory(campaign))>=resource_policy.worker_limit(campaign):return dict(state='RUNNING',phase='WAITING_FOR_SLOT')
    states={}
    for name,settings in config['families'].items():
        if name not in ('N1','N4b','N4c'):raise ValueError('uncommissioned extension')
        states[name]=step_family(campaign,name,settings)
        if states[name]['state']=='RUNNING_OR_ADMITTED':break
    terminal={'COMPLETE','RESOURCE_DEFERRED','INSTRUMENT_FAILED','KNOWN_ANSWER_FAILED'}
    return dict(state='REVIEW_BOUNDARY' if len(states)==len(config['families']) and all(x['state'] in terminal for x in states.values()) else 'RUNNING',families=states)


def run(campaign,state,config_path):
    started=time.process_time();runtime.below_normal();config=read(config_path);identity=config['service_id']
    if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in identity):raise ValueError('invalid extension owner')
    charge=campaign/'attempts'/(identity+'.json')
    if charge.exists():raise ValueError('preserve previous service charge')
    with owner(state):
        try:
            while True:
                result=queue.admission_step(campaign,step,campaign,config)
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
