"""Finite production admission after the approved Q repair; no accuracy bypass."""
import time,itertools
from pathlib import Path
from .common import read,digest


def validate(setting,dest,pulse=lambda **kw:None):
    from . import q_rank
    from runners.v31_build import admit
    c=Path(setting['campaign']);repo=Path(setting['repo']);j=c/'jobs/Q_HIDDEN_R1_IO'
    deadline=time.monotonic()+3600
    while not (j/'COMPLETE.json').exists():
        if time.monotonic()>=deadline:
            return dict(passed=True,production_admitted=False,disposition='PARENT_NOT_TERMINAL_WITHIN_BOUNDED_WAIT')
        pulse(phase='waiting_for_approved_Q_validation');time.sleep(30)
    d=read(j/'COMPLETE.json')
    if d['state']!='VERIFIED':
        return dict(passed=True,production_admitted=False,disposition='HELD_AFTER_PARENT_VALIDATION_FAILURE',parent_completion_sha256=digest(j/'COMPLETE.json'))
    w=read(j/'WORKER_COMPLETE.json');v=read(j/'output/VALIDATION.json')
    assert d['plan_sha256']==w['plan_sha256']==digest(j/'PLAN.json')
    assert d['worker_receipt_sha256']==digest(j/'WORKER_COMPLETE.json')
    assert w['validation_sha256']==digest(j/'output/VALIDATION.json') and v['passed']
    configs=v['configurations'];assert len(configs)==12 and all(x['result']['passed'] for x in configs)
    admitted=[]
    for view in ('V1','V2'):
        selected=[x for x in configs if x['setting']['view']==view];assert len(selected)==6
        settings=[dict(weight=weight,group_rate=x['setting']['group_rate'],move_rate=x['setting']['move_rate'],
                       view=view,particles=x['result']['production_particles'],steps=2000,worlds=200,blocks=8)
                  for x,weight in itertools.product(selected,(0.,.5,1.))]
        pulse(phase='Q_production_cost_profile',view=view)
        family='Q_HIDDEN_R1_'+view+'_SCIENCE'
        admit(repo,c,family,'Q',q_rank,settings,'docs/versions/v31-four-settings/protocols/Q.md',54000,
              extra=dict(validation_sha256=digest(j/'output/VALIDATION.json'),parent_completion_sha256=digest(j/'COMPLETE.json')))
        admitted.append(dict(family=family,prepared_cells=len(settings)))
    return dict(passed=True,production_admitted=True,admissions=admitted,
                accounting='Profile CPU and this worker are conservatively both charged to Q',
                scope='Only original Q hidden settings after all unchanged gates and measured cost profiles')
