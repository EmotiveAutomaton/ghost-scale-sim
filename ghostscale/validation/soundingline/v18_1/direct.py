"""Cheap structural compiler: a stronger direct rival to generic state search.

The compiler uses the public candidate-law class to find an order safe under every
candidate. It does not identify which candidate is true. In the original condition
this recovers the already public label order; in the relabeled condition it pays to
construct the union dependency order explicitly.
"""
import json
from ..v16.records import canonical
from .common import Work,Exhausted,step,identity,score_submission,validate_world

SCHEMA='v18.1.structural-direct.1'


def shared_order(models,work):
    n=len(models[0]['parents']);children={i:set() for i in range(n)};incoming=[0]*n
    edges=set()
    for model in models:
        validate_world(model)
        if len(model['parents'])!=n:raise ValueError('mixed assembly sizes')
        for child,parent in enumerate(model['parents']):
            work.charge('checking')
            if parent>=0:edges.add((parent,child))
    for parent,child in edges:
        children[parent].add(child);incoming[child]+=1
    ready=sorted(i for i,value in enumerate(incoming) if value==0);order=[]
    while ready:
        work.charge('selection');part=ready.pop(0);order.append(part)
        for child in sorted(children[part]):
            work.charge('checking');incoming[child]-=1
            if incoming[child]==0:ready.append(child);ready.sort()
    return order if len(order)==n else None


def compile_models(models,initial,target,max_steps,work):
    """Compile one route safe under every supplied public candidate model."""
    if not models:raise ValueError('public model class required')
    world=models[0];program=[];reason=None
    if world['kind']=='assembly':
        n=len(world['parents'])
        if n not in (3,5,7):raise ValueError('unsupported assembly size')
        for model in models:
            if set(model)!={'kind','parents','defaults','forbidden'}:raise ValueError('private field in model')
            if model['kind']!='assembly':raise ValueError('mixed model kinds')
            if model['defaults']!=world['defaults'] or model['forbidden']:raise ValueError('unshared defaults or unsupported restrictions')
        order=shared_order(models,work)
        if order is None:reason='candidate-law union has no shared dependency order'
        elif any(v not in (0,1) for v in initial+target):
            reason='compiler requires fully attached initial and target states'
        else:
            for part in reversed(order):
                work.charge('proposal_generation');program.append(n+part)
            for part in order:
                work.charge('checking');work.charge('proposal_generation');program.append(part)
                if world['defaults'][part]!=target[part]:
                    work.charge('proposal_generation');program.append(2*n+part)
            work.charge('proposal_generation');program.append(3*n)
    elif world['kind']=='graphic':
        if any(set(m)!={'kind','cells','forbidden'} for m in models):raise ValueError('private field in graphic model')
        for cell in range(world['cells']):
            work.charge('checking');before=bool(initial&(1<<cell));after=bool(target&(1<<cell))
            if before!=after:
                work.charge('proposal_generation');program.append(cell if after else world['cells']+cell)
    else:raise ValueError('unknown world')
    if reason is None and len(program)<=max_steps:
        current=identity(initial);stopped=False;legal=True
        for action in program:
            work.charge('selection');work.charge('hypothetical_execution')
            current,stopped,legal=step(world,current,stopped,action)
            if not legal:break
        if legal and current==identity(target) and (stopped or world['kind']=='graphic'):
            work.charge('actual_execution',len(program));return program,None
    elif reason is None:reason='constructive route exceeds allowed action depth'
    return None,reason


def predict(payload,budget):
    p=json.loads(payload)
    if set(p)!={'schema','models','initial','target','max_steps'} or p['schema']!=SCHEMA:
        raise ValueError('structural-direct public contract')
    work=Work(budget);selected=None;reason=None
    try:
        selected,reason=compile_models(p['models'],p['initial'],p['target'],p['max_steps'],work)
    except Exhausted:reason='online work exhausted'
    return dict(program=selected,costs=work.receipt(),unsupported_reason=reason,storage_tokens=0,
        knowledge='shared order compiled from the public candidate-law union and defaults; no true-law identification')


def evaluate(case,budgets):
    return [dict(method='structural-direct',budget=budget,
        **score_submission(case['private']['true_world'],case['public']['initial'],case['public']['target'],
            predict(canonical(case['public']),budget),case['public']['max_steps'])) for budget in budgets]
