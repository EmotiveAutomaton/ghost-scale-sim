"""Cheap structural compiler: a stronger direct rival to generic state search.

Assembly part labels already form a public topological order. Resetting in reverse
order and attaching in forward order is safe for every law in that class, without
identifying the actual dependency graph. This assumption is explicit, not an oracle.
"""
import json
from ..v16.records import canonical
from .common import Work,Exhausted,step,identity,score_submission

SCHEMA='v18.1.structural-direct.1'


def predict(payload,budget):
    p=json.loads(payload)
    if set(p)!={'schema','models','initial','target','max_steps'} or p['schema']!=SCHEMA:
        raise ValueError('structural-direct public contract')
    work=Work(budget);selected=None;reason=None
    try:
        models=p['models']
        if not models:raise ValueError('public model class required')
        world=models[0];program=[]
        if world['kind']=='assembly':
            n=len(world['parents'])
            if n not in (3,5,7):raise ValueError('unsupported assembly size')
            for model in models:
                if set(model)!={'kind','parents','defaults','forbidden'}:raise ValueError('private field in model')
                for part,parent in enumerate(model['parents']):
                    work.charge('checking')
                    if model['kind']!='assembly' or parent not in range(-1,part):raise ValueError('public topological label class required')
                if model['defaults']!=world['defaults'] or model['forbidden']:raise ValueError('unshared defaults or unsupported restrictions')
            if any(v not in (0,1) for v in p['initial']+p['target']):
                reason='compiler requires fully attached initial and target states'
            else:
                for part in reversed(range(n)):
                    work.charge('proposal_generation');program.append(n+part)
                for part in range(n):
                    work.charge('checking');work.charge('proposal_generation');program.append(part)
                    if world['defaults'][part]!=p['target'][part]:
                        work.charge('proposal_generation');program.append(2*n+part)
                work.charge('proposal_generation');program.append(3*n)
        elif world['kind']=='graphic':
            if any(set(m)!={'kind','cells','forbidden'} for m in models):raise ValueError('private field in graphic model')
            for cell in range(world['cells']):
                work.charge('checking');before=bool(p['initial']&(1<<cell));after=bool(p['target']&(1<<cell))
                if before!=after:
                    work.charge('proposal_generation');program.append(cell if after else world['cells']+cell)
        else:raise ValueError('unknown world')
        if reason is None and len(program)<=p['max_steps']:
            current=identity(p['initial']);stopped=False;legal=True
            for action in program:
                work.charge('selection');work.charge('hypothetical_execution')
                current,stopped,legal=step(world,current,stopped,action)
                if not legal:break
            if legal and current==identity(p['target']) and (stopped or world['kind']=='graphic'):
                work.charge('actual_execution',len(program));selected=program
        elif reason is None:reason='constructive route exceeds allowed action depth'
    except Exhausted:reason='online work exhausted'
    return dict(program=selected,costs=work.receipt(),unsupported_reason=reason,storage_tokens=0,
        knowledge='public topological part numbering/shared defaults or independent graphic operators; no learned dependency identification')


def evaluate(case,budgets):
    return [dict(method='structural-direct',budget=budget,
        **score_submission(case['private']['true_world'],case['public']['initial'],case['public']['target'],
            predict(canonical(case['public']),budget),case['public']['max_steps'])) for budget in budgets]
