"""Finite supplement queue: declared studies only, readiness before priority.

Existing N2/N3 owners drain independently. N4c and empirical sensitivity choices
cannot enter this controller without a separately reviewed implementation.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v23 import runtime,resource_policy
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners import stock_v23,queue_v23_families as queue
from runners.watch_v23_events import owner,event


def step(campaign,state,config):
    if resource_policy.binding(campaign)!=config['resource_binding']:
        raise ValueError('controller resource policy changed')
    if (state/'STOP').exists():return dict(state='OPERATOR_STOP_AT_BOUNDARY')
    if (campaign/'STOP').exists() or runtime.now()>=read(campaign/'ACCEPTANCE.json')['report_start']:
        return dict(state='STOPPED_AT_CUTOFF')
    if set(config['preparation']['families'])-{'N4','N7'} or set(config['continuation']['families'])-{'R3'}:
        raise ValueError('unreviewed addendum queue expansion')
    before={p.name for p,_,_ in queue.inventory(campaign)}
    # Each step checks native reservations and worker availability at admission.
    # A pending design or absent dependency never blocks a lower ready study.
    prep=stock_v23.step(campaign,state/'preparation',config['preparation'])
    continuation=queue.step(campaign,config['continuation'])
    after=queue.inventory(campaign)
    admitted=sorted({p.name for p,_,_ in after}-before)
    for name in admitted:
        path=state/'dispatch'/(name+'.json')
        if not path.exists():
            write(path,dict(at=runtime.now(),packet=name,plan_sha256=digest(campaign/'packets'/name/'PLAN.json'),
                rule='Immediately admit ready authorized work; 60-minute watchdog is a maximum idle interval, not a wait.'))
    state_= 'REVIEW_BOUNDARY' if prep['state']!='RUNNING' and continuation['state']!='RUNNING' else 'RUNNING'
    return dict(state=state_,families={**prep.get('families',{}),**continuation.get('families',{})},
        newly_admitted=admitted,active_scientists=queue.active_count(after),
        worker_limit=resource_policy.worker_limit(campaign),
        held_design=['N4c','T4 sensitivity mapping'],n1='RESOURCE_DEFERRED_NOT_READMITTED')


def idle_watchdog(previous,result,current):
    """Detect a dispatch stall, never turn a dependency hold into admissibility."""
    waiting=any(value.startswith('WAITING_FOR_') and value.endswith('SLOT')
                for value in result.get('families',{}).values())
    idle=result.get('active_scientists',0)<result.get('worker_limit',0)
    if not waiting or not idle or result.get('newly_admitted'):
        return None,False
    since=current if previous is None else previous
    return since,current-since>=3600


def run(campaign,state,config_path):
    config=read(config_path);identity=config['service_id']
    if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in identity):
        raise ValueError('invalid queue service identity')
    charge=campaign/'attempts'/(identity+'.json')
    if charge.exists():raise ValueError('preserve prior controller accounting')
    runtime.below_normal();started=time.process_time();idle_since=None
    with owner(state):
        try:
            while True:
                result=queue.admission_step(campaign,step,campaign,state,config)
                idle_since,overdue=idle_watchdog(idle_since,result,time.monotonic())
                if overdue:
                    event(campaign,identity+'-idle-watchdog','queue_review_boundary',
                        dict(reason='A ready slot has remained unfilled for 60 minutes; dispatch is retried each service step.',status=result))
                write(state/'STATUS.json',dict(result,pid=os.getpid(),heartbeat=runtime.now(),config_sha256=digest(config_path)),immutable=False)
                write(charge,dict(cpu_seconds=time.process_time()-started,state=result['state']),immutable=False)
                if result['state']!='RUNNING':
                    event(campaign,identity+'-boundary','queue_review_boundary',result);return
                time.sleep(30)
        except BaseException as exc:
            write(state/'STATUS.json',dict(state='failed',pid=os.getpid(),heartbeat=runtime.now(),error=repr(exc)),immutable=False)
            write(charge,dict(cpu_seconds=time.process_time()-started,state='failed'),immutable=False)
            event(campaign,identity+'-failed','instrument_failure',dict(error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('campaign','state','config'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();run(args.campaign.resolve(),args.state.resolve(),args.config.resolve())
