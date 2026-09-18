"""Native detours and paid gates under separately supplied dependency models."""
from collections import deque
from copy import deepcopy
from itertools import product
import json

from ..v16.records import canonical,digest
from .common import (Work,Exhausted,step,execute,actions,identity,distance,exhaustive,
                     solve,score_submission,learn_fragments)

SCHEMA='v18.1.g1.1'
GATES=('none','local','endpoint','lookahead2','lookahead4','viability')


def gate_for(name):
    if name not in GATES:
        raise ValueError('unknown gate')
    if name=='none':return None
    def check(world,state,target,fragment,remaining,work):
        current,stopped=identity(state),False
        original_error=distance(state,target)
        worsened=False
        for action in fragment:
            work.charge('checking')
            after,stop,legal=step(world,current,stopped,action)
            if not legal:return False
            worsened |= distance(after,target)>distance(current,target)
            current,stopped=after,stop
        if name=='local':return not worsened
        if name=='endpoint':return distance(current,target)<=original_error
        if name.startswith('lookahead') and distance(current,target)<=original_error:
            return True
        depth=(int(name[-1]) if name.startswith('lookahead') else remaining-len(fragment))
        depth=min(depth,remaining-len(fragment))
        queue=deque([(current,stopped,0)])
        seen={(current,stopped)}
        while queue:
            candidate,stop,used=queue.popleft()
            if name=='viability':
                if candidate==identity(target) and (stop or world['kind']=='graphic'):return True
            elif distance(candidate,target)<original_error:
                return True
            if stop or used>=depth:continue
            for action in actions(world):
                work.charge('checking')
                after,halt,legal=step(world,candidate,stop,action)
                key=(after,halt)
                if legal and key not in seen:
                    seen.add(key);queue.append((after,halt,used+1))
        return False
    return check


def laws(n=3):
    if n==3:
        parent_sets=list(product(*[range(-1,i) for i in range(n)]))
    else:
        parent_sets=[tuple([-1]+[0]*(n-1)),tuple([-1]+list(range(n-1))),
                     tuple([-1,-1]+[i%2 for i in range(2,n)])]
    for parents in parent_sets:
        for defaults in product((0,1),repeat=n):
            yield dict(kind='assembly',parents=list(parents),defaults=list(defaults),forbidden=[])


def wrong_model(world,information):
    result=deepcopy(world)
    if information=='correct':return result
    if information=='incomplete':
        result['parents']=[-1]*len(world['parents'])
    elif information=='misleading':
        result['parents']=[-1]+[i-1 if world['parents'][i]!=i-1 else -1 for i in range(1,len(world['parents']))]
    else:raise ValueError('unknown information')
    return result


def contexts(n=3):
    """Complete support of the declared rotate-one-part task family, not all tasks."""
    found=[]
    for world in laws(n):
        initial=list(world['defaults'])
        for part in range(n):
            target=initial.copy();target[part]=1-target[part]
            bound=2*n+2
            reference=exhaustive(world,initial,target,bound)
            assert reference['reachable']
            monotone=exhaustive(world,initial,target,bound,monotone=True)
            stratum='monotone' if monotone['reachable'] else 'required-detour'
            witness=reference['program']
            base=dict(true_world=world,initial=initial,target=target,max_steps=bound,
                      reference=reference,monotone_reference=monotone,proposed_routine=witness,
                      task='rotate-one-part',stratum=stratum)
            found.append(dict(base,unit_id=digest([world,initial,target,'useful'])))
            # Same reachable goal, but an intact demonstrated routine for a different goal.
            other=(part+1)%n
            wrong_target=initial.copy();wrong_target[other]=1-wrong_target[other]
            wrong=exhaustive(world,initial,wrong_target,bound)['program']
            found.append(dict(base,unit_id=digest([world,initial,target,'harmful']),stratum='harmful',
                              proposed_routine=wrong,demonstrated_target=wrong_target))
        # The deliberately impossible dependency target is separately labeled.
        # Its attempted rotate-child demonstration may itself fail on a chain;
        # that observed failure stays in training with false feedback, without replacement.
        unreachable_targets=set()
        for child,parent in enumerate(world['parents']):
            if parent<0:continue
            target=initial.copy();target[parent]=-1
            if tuple(target) in unreachable_targets:continue
            unreachable_targets.add(tuple(target))
            reference=exhaustive(world,initial,target,2*n+2)
            assert not reference['reachable']
            found.append(dict(true_world=world,initial=initial,target=target,max_steps=2*n+2,
                reference=reference,monotone_reference=reference,proposed_routine=[2*n+child,3*n],
                demonstrated_target=None,task='invalid-support-target',stratum='unreachable',
                unit_id=digest([world,initial,target,'unreachable'])))
    return sorted(found,key=lambda x:(x['stratum'],x['unit_id']))


def make_case(context,history,information,representation):
    true=context['true_world']
    program=context['proposed_routine']
    demonstration=execute(true,context['initial'],program,context['max_steps'])
    # The complete-plan condition is explicitly a demonstrated-plan check, never unseen transfer.
    training=[dict(initial=context['initial'],program=program,
                   endpoint=demonstration['state'],feedback=demonstration['legal'] and demonstration['stopped']) for _ in range(4)]
    if representation=='complete':
        rep=dict(fragments=[program] if demonstration['legal'] else [],episodes=[],storage=len(program),scanned=4*len(program))
    elif representation=='fragments':
        without_stop=[dict(t,program=[a for a in t['program'] if a!=3*len(true['parents'])]) for t in training]
        rep=learn_fragments(without_stop,capacity=4,threshold=2,storage_cap=64)
    elif representation=='primitive':
        rep=dict(fragments=[],episodes=[],storage=0,scanned=0)
    else:raise ValueError('unknown representation')
    order=actions(true)
    offset=history%len(order)
    order=order[offset:]+order[:offset]
    public=dict(schema=SCHEMA,model=wrong_model(true,information),initial=context['initial'],target=context['target'],
                representation=rep,training=training,max_steps=context['max_steps'],action_order=order)
    return dict(case_id=digest([context['unit_id'],history,information,representation]),
                structural_unit=context['unit_id'],history=history,stratum=context['stratum'],
                information=information,representation=representation,public=public,private=deepcopy(context),
                model_is_actually_correct=public['model']==true,
                exposure='demonstrated routine; its training endpoint is explicitly visible; no unseen-target claim')


def predict(payload,gate,budget):
    public=json.loads(payload)
    if set(public)!={'schema','model','initial','target','representation','training','max_steps','action_order'} or public['schema']!=SCHEMA:
        raise ValueError('G1 public schema violation')
    if sorted(public['action_order'])!=actions(public['model']):
        raise ValueError('invalid action order')
    return solve(public['model'],public['initial'],public['target'],public['representation'],budget,
                 public['max_steps'],public['action_order'],gate=gate_for(gate))


def evaluate(case,budgets=(128,512,2048)):
    rows=[]
    public=case['public'];truth=case['private']['true_world']
    for gate in GATES:
        # Explicit extra-information ceiling: both checks and planner receive the true law.
        request=deepcopy(public)
        if gate=='viability':request['model']=deepcopy(truth)
        for budget in budgets:
            result=predict(canonical(request),gate,budget)
            scored=score_submission(truth,public['initial'],public['target'],result,public['max_steps'])
            rows.append(dict(gate=gate,budget=budget,known_law_ceiling=gate=='viability',**scored))
    return rows


def gate_diagnostics(case):
    """Evaluator probe of the intact acquired routine, separate from online tasks."""
    public=case['public'];context=case['private'];truth=context['true_world']
    program=context['proposed_routine']
    actual=execute(truth,public['initial'],program,public['max_steps'])
    useful=actual['legal'] and actual['stopped'] and identity(actual['state'])==identity(public['target'])
    output=[]
    for name in GATES:
        work=Work(100000)
        gate=gate_for(name)
        admitted=True if gate is None else gate(truth if name=='viability' else public['model'],
                  public['initial'],public['target'],program,public['max_steps'],work)
        output.append(dict(gate=name,admitted=admitted,useful=useful,false_rejection=useful and not admitted,
                           acceptance_of_non_solution=not useful and admitted,checking_operations=work.spent,
                           scope='separate evaluator probe; no free result supplied to planner'))
    return output
