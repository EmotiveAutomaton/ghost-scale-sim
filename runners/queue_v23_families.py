"""Finite continuation of explicitly admitted families; stop at scientific decisions.

Never starts an unimplemented family, changes a source capsule, repairs failures,
publishes material, or chooses a new research question. Exact accepted packets
are reconstructed before the calibrated precommitted continuation is applied.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys,time
from ghostscale.validation.soundingline.v23 import family,runtime,resource_policy
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.watch_v23_events import owner,alive,event


def inventory(campaign):
    result=[]
    for path in sorted((campaign/'packets').iterdir()):
        if not (path/'PLAN.json').exists():continue
        plan=read(path/'PLAN.json');status=read(path/'STATUS.json') if (path/'STATUS.json').exists() else {}
        result.append((path,plan,status))
    return result


def capacity(campaign,entries,new_cap,second_family=None,budget_group='repairs',needs_slot=True,replacing_burnin=None,policy_plan=None):
    """Reserve unfinished admissions even during their launch/handshake interval."""
    acceptance=resource_policy.resolve(campaign,policy_plan);reserved=0.;group=0.;active=0
    from runners.admit_v23_repair import reservations_released,burnin_reservation
    for path,plan,status in entries:
        unfinished=status.get('state') not in ('complete','failed')
        if plan.get('logical_parent'):
            if unfinished:active+=1
            continue
        if unfinished:
            reserved+=max(0,plan['cpu_cap_seconds']-status.get('cpu_seconds',0.)) if plan.get('scheduler_only') else plan['cpu_cap_seconds']
            if not plan.get('scheduler_only') or status.get('uses_scientific_slot'):active+=1
            if plan['budget_group']==budget_group:group+=plan['cpu_cap_seconds']
        elif plan['budget_group']==budget_group:group+=status.get('cpu_seconds',0.)
        released=reservations_released(campaign,plan.get('family',''))
        reserve=0. if released else burnin_reservation(campaign,plan,2,plan.get('second_burnin_reservation_seconds',0.))
        if status.get('state')!='failed' and reserve and plan['family']!=second_family:
            second=campaign/'packets'/('v23-'+plan['family'].lower()+'-batch-2')
            if not (second/'PLAN.json').exists():
                reserved+=reserve
                if plan['budget_group']==budget_group:group+=reserve
        for batch,amount in enumerate(plan.get('burnin_reservations_seconds',[]),1):
            being_admitted=(plan['family']==second_family and batch==2) or replacing_burnin==(plan['family'],batch)
            target=campaign/'packets'/f"v23-{plan['family'].lower()}-batch-{batch}"
            if status.get('state')!='failed' and not released and not (target/'PLAN.json').exists() and not being_admitted:
                amount=burnin_reservation(campaign,plan,batch,amount)
                reserved+=amount
                if plan['budget_group']==budget_group:group+=amount
    if needs_slot and active>=acceptance['max_scientific_workers']:return 'NO_SLOT'
    if runtime.cost(campaign)+reserved+new_cap>acceptance['cpu_seconds']-acceptance['protected_cpu_seconds']:
        return 'GLOBAL_CPU_RESERVE'
    if group+new_cap>acceptance['group_caps_cpu_hours'][budget_group]*3600:return 'REPAIRS_GROUP_CAP' if budget_group=='repairs' else 'FAMILY_GROUP_CAP'
    return 'AVAILABLE'


def active_count(entries):
    return sum(s.get('state') not in ('complete','failed') and
        (not p.get('scheduler_only') or s.get('uses_scientific_slot',False)) for _,p,s in entries)


def launch(root,campaign,source):
    python=Path(sys.executable)
    if python.name.lower()!='python.exe' or '.venv' not in str(python):raise ValueError('scientific venv required')
    env=os.environ.copy();plan=read(root/'PLAN.json')
    env['PYTHONPATH']=os.pathsep.join([str(campaign/'dependencies')]+plan.get('extra_pythonpath',[]))
    argv=[str(python.with_name('pythonw.exe')),str(source/'runners/launch_background.py'),
        '--cwd',str(source),'--log',str(root/'worker.log'),'--',str(python),'-B','-m','runners.run_v23',
        '--root',str(root),'--campaign',str(campaign)]
    child=subprocess.Popen(argv,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    write(root/'LAUNCH.json',dict(at=runtime.now(),launcher_pid=child.pid))


def step(campaign,config):
    acceptance=resource_policy.resolve(campaign)
    if runtime.now()>=acceptance['report_start'] or (campaign/'STOP').exists():return dict(state='STOPPED_AT_CUTOFF')
    calibration=read(campaign/'CALIBRATION_ACCEPTANCE.json')
    summary=Path(calibration['summary'])
    if not summary.is_absolute():summary=(campaign/summary).resolve()
    if digest(summary)!=calibration['summary_sha256']:raise ValueError('calibration summary changed')
    policy=read(summary)['policy'] if 'policy' in read(summary) else read(summary)
    # The validated summary keeps its chosen policy in these two explicit fields.
    if 'kill_threshold' not in policy:
        chosen=summary.parent/'CHOSEN_BEFORE_VALIDATION.json';policy=read(chosen)
    states={}
    from ghostscale.validation.soundingline.v23.registry import ORDER
    order=resource_policy.ordered(ORDER) if resource_policy.binding(campaign) else ORDER
    for name in order:
        if name not in config['families']:continue
        first=Path(config['families'][name]['first_packet'])
        paths=[first]
        for batch in range(2,8):
            path=campaign/'packets'/f'v23-{name.lower()}-batch-{batch}'
            if not (path/'PLAN.json').exists():break
            paths.append(path)
        last=paths[-1];status=read(last/'STATUS.json') if (last/'STATUS.json').exists() else {}
        if read(last/'PLAN.json').get('scheduler_only') and status.get('state') not in ('complete','failed'):
            from runners.partition_v23 import step as partition_step
            partition_step(campaign,last)
            status=read(last/'STATUS.json') if (last/'STATUS.json').exists() else {}
        if status.get('state')=='failed':
            states[name]='INSTRUMENT_FAILED';continue
        if not (last/'COMPLETE.json').exists():
            states[name]='RUNNING_OR_ADMITTED';continue
        card=last/'MACHINE_CARD.json'
        if card.exists():
            decision=read(card)
            if (decision['complete_sha256']!=digest(last/'COMPLETE.json') or
                decision['numerical_review_sha256']!=digest(last/'NUMERICAL_REVIEW.json') or
                decision['calibration_complete_sha256']!=calibration['complete_sha256']):
                raise ValueError('accepted family card bindings changed')
        else:
            if active_count(inventory(campaign))>=acceptance['max_scientific_workers']:
                states[name]='WAITING_FOR_REVIEW_SLOT';continue
            for path in paths:
                if not (path/'NUMERICAL_REVIEW.json').exists():write(path/'NUMERICAL_REVIEW.json',family.reconstruct(path))
            decision=family.decide(name,paths,policy)
            write(card,dict(decision,complete_sha256=digest(last/'COMPLETE.json'),
                numerical_review_sha256=digest(last/'NUMERICAL_REVIEW.json'),
                calibration_complete_sha256=calibration['complete_sha256'],
                predecessor_cards=[digest(p/'MACHINE_CARD.json') for p in paths[:-1]]))
        states[name]=decision['state']
        if decision['state'] not in ('BURN_IN','CONTINUE'):
            release=campaign/'reservations'/(name+'-RELEASE.json')
            if not release.exists():write(release,dict(packet=last.name,machine_card_sha256=digest(card),
                state=decision['state'],at=runtime.now(),reason='terminal verified family; preserve unused reservation history'))
            event(campaign,f'{name.lower()}-decision-{len(paths)}','family_decision',
                dict(packet=last.name,machine_card_sha256=digest(card),state=decision['state']))
            continue
        previous=read(last/'PLAN.json');batch=previous['batch']+1
        new=family.successor(previous,decision,first)
        new.pop('scheduler_only',None);new.pop('partition_layout',None)
        # Full observed packet includes fitting and I/O, so carrying it forward
        # with a 1.5 multiplier conservatively includes sunk first-batch fitting.
        scale=(new['worlds']*len(new['variants']))/(previous['worlds']*len(previous['variants']))
        observed=previous.get('rescore_scientific_cpu_seconds',status.get('scientific_batch_cpu_seconds',status['cpu_seconds']))
        estimate=max(300,1.5*observed*scale);cap=estimate
        new.pop('rescore_scientific_cpu_seconds',None)
        for key in ('retained_parts','repair_parent','repair_parent_plan_sha256','repair_parent_status_sha256'):
            new.pop(key,None)
        if estimate>14400:
            if 'ghostscale/validation/soundingline/v23/partition.py' not in previous['sources']:
                states[name]='RESOURCE_DEFERRED';continue
            from ghostscale.validation.soundingline.v23.partition import layout
            split=layout(new['worlds'],estimate,min_worlds=1 if resource_policy.binding(campaign) else 4);cap=split['total_cpu_cap']
            new.update(scheduler_only=True,partition_layout=split)
            if resource_policy.binding(campaign):new['max_parallel_parts']=3
        with owner(campaign/'admission'):
            limit=capacity(campaign,inventory(campaign),cap,second_family=name if batch==2 else None,budget_group=previous['budget_group'],needs_slot=not new.get('scheduler_only'))
            if limit!='AVAILABLE':
                states[name]=limit;continue
            root=campaign/'packets'/f'v23-{name.lower()}-batch-{batch}'
            new.update(cpu_cap_seconds=cap,wall_cap_seconds=cap*4,second_burnin_reservation_seconds=0,
                predecessor_decision=digest(card),confirmation_support=previous['confirmation_support'])
            new=resource_policy.rebind_successor(campaign,root.name,new,Path(__file__).resolve().parents[1])
            source=Path(new['source_archive']).with_suffix('')
            for relative,expected_hash in new['sources'].items():
                if digest(source/relative)!=expected_hash:raise ValueError('source capsule changed')
            write(root/'PLAN.json',new)
            write(root/'ADMISSION.json',dict(at=runtime.now(),predecessor=last.name,
                predecessor_machine_card_sha256=digest(card),plan_sha256=digest(root/'PLAN.json'),
                new_cpu_cap_seconds=cap,charged_cpu_seconds=runtime.cost(campaign),
                rule='calibrated finite successor; frozen fits and exact original source retained'))
            if not new.get('scheduler_only'):launch(root,campaign,source)
            states[name]='SUCCESSOR_ADMITTED'
    terminal={'NOMINATED','STOPPED_BELOW_MARGIN','INCONCLUSIVE_AT_CAP','INSTRUMENT_FAILED',
              'RESOURCE_DEFERRED','GLOBAL_CPU_RESERVE','REPAIRS_GROUP_CAP','FAMILY_GROUP_CAP','PRIMARY_UNAVAILABLE'}
    return dict(state='REVIEW_BOUNDARY' if states and all(s in terminal for s in states.values()) else 'RUNNING',families=states)


def run(campaign,state,config_path):
    started=time.process_time();config=read(config_path);runtime.below_normal()
    # Each continuation service retains its own cumulative cost and terminal
    # event. Reusing the first owner's names would erase accounting and suppress
    # later queue-boundary delivery after that first event was acknowledged.
    service_id=config.get('service_id','repair-queue')
    if not service_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in service_id):
        raise ValueError('invalid queue service identity')
    cost_name='family-queue' if service_id=='repair-queue' else service_id
    cost_path=campaign/'attempts'/(cost_name+'.json')
    if cost_path.exists():raise ValueError('queue service identity already used')
    with owner(state):
        try:
            while True:
                if (state/'STOP').exists():
                    result=dict(state='OPERATOR_STOP_AT_BOUNDARY',reason='No live scientific packet was stopped')
                    write(state/'STATUS.json',dict(result,pid=os.getpid(),heartbeat=runtime.now()),immutable=False)
                    write(cost_path,dict(cpu_seconds=time.process_time()-started,state=result['state']),immutable=False)
                    return
                result=step(campaign,config)
                write(state/'STATUS.json',dict(result,pid=os.getpid(),heartbeat=runtime.now(),config_sha256=digest(config_path)),immutable=False)
                write(cost_path,dict(cpu_seconds=time.process_time()-started,state=result['state']),immutable=False)
                if result['state']!='RUNNING':
                    event(campaign,service_id+'-boundary','queue_review_boundary',result);return
                time.sleep(30)
        except BaseException as exc:
            write(state/'STATUS.json',dict(state='failed',pid=os.getpid(),heartbeat=runtime.now(),error=repr(exc)),immutable=False)
            write(cost_path,dict(cpu_seconds=time.process_time()-started,state='failed'),immutable=False)
            event(campaign,service_id+'-failed','instrument_failure',dict(error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);a=p.parse_args();run(a.campaign.resolve(),a.state.resolve(),a.config.resolve())
