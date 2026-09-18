"""Memory/checking diagnostic and common-planner comparison; no private solver input."""
from collections import Counter
from itertools import product,permutations
import json
import random

from ..v16.records import canonical, digest,seed_for
from ..v16 import attention_craft
from ..v16.world import execute
from ..v16.purpose_craft import inhibition
from ..v16.craft import construct
from ..v18.study import learn_public, use_library, evaluate
from ..v18.study import offered_view,select_offers,CONDITION
from .common import solve, score_submission

SCHEMA = 'v18.1.g0.1'


def acquire(payload):
    request = json.loads(payload)
    if set(request) != {'schema', 'processed', 'capacity', 'representation'} or request['schema'] != SCHEMA:
        raise ValueError('G0 acquisition schema')
    # Reuse the original strict public-record validation and feedback contract.
    original = learn_public(canonical({'schema': 'v18.selective-acquisition.1', 'processed': request['processed']}))
    if request['capacity'] not in (1, 2):
        raise ValueError('only two distinct routines are available')
    records = request['processed']
    counts = Counter(tuple(x['program']) for x in records if x['feedback'] is True)
    ranked = sorted((p for p,n in counts.items() if n >= 3), key=lambda p: (-counts[p],p))
    if request['representation'] == 'fragments':
        library = [list(p) for p in ranked[:request['capacity']]]
    elif request['representation'] == 'episodes':
        library = [list(p) for p in dict.fromkeys(tuple(x['program']) for x in records if x['feedback'])][:request['capacity']]
    elif request['representation'] == 'primitive':
        library = []
    else:
        raise ValueError('unknown representation')
    return dict(library=library, costs=dict(original['costs'], storage_primitives=sum(map(len,library))),
                eligible_programs=[list(p) for p in ranked])


def enumeration(target, library, budget, order):
    """Original breadth-first token enumeration with an explicit primitive ordering."""
    tokens = list(range(len(library))) + [str(a) for a in order]
    spent = attempts = 0
    for length in range(4):
        for sequence in product(tokens, repeat=length):
            program = [a for token in sequence for a in (library[token] if type(token) is int else [int(token)])]
            charge = min(len(program), 3)
            if spent + charge > budget:
                return dict(program=[], search_primitives=spent, considered=attempts, search_timeout=True)
            result = execute(program)
            spent += result.primitive_cost; attempts += 1
            if result.legal and result.artifact == target:
                return dict(program=program, search_primitives=spent, considered=attempts, search_timeout=False)
    return dict(program=[], search_primitives=spent, considered=attempts, search_timeout=True)


def predict(payload):
    public = json.loads(payload)
    fields = {'schema','library','target','budget','checking','planner','action_order','ordering','representation'}
    if set(public) != fields or public['schema'] != SCHEMA:
        raise ValueError('G0 transfer schema')
    if sorted(public['action_order']) != list(range(8)):
        raise ValueError('invalid action order')
    library = public['library']
    active, checks = inhibition(library, public['target']) if public['checking'] else (library, [])
    paid_checks = sum(x['primitive_cost'] for x in checks)
    if public['planner'] == 'enumeration':
        if public['ordering'] != 'fixed':
            raise ValueError('goal ordering belongs to counted common planner')
        outcome = enumeration(public['target'], active, max(0, public['budget']-paid_checks), public['action_order'])
        actual = execute(outcome['program'])
        return dict(program=outcome['program'], success=actual.legal and actual.artifact == public['target'],
                    invalid=not actual.legal, missing_output=outcome['search_timeout'],
                    convention='original V18 check+search primitive envelope; final execution separate',
                    costs=dict(checking=paid_checks, search=outcome['search_primitives'], actual=actual.primitive_cost,
                               total_online=paid_checks+outcome['search_primitives']+actual.primitive_cost),
                    rejected_fragments=len(library)-len(active), checks=checks)
    if public['planner'] != 'state':
        raise ValueError('unknown planner')
    if paid_checks > public['budget']:
        return dict(program=None,success=False,invalid=False,missing_output=True,
                    convention='matched total declared online operations',costs={'checking':public['budget'],'total_online':public['budget']},
                    rejected_fragments=None,checks=[])
    key = 'episodes' if public['representation'] == 'episodes' else 'fragments'
    world = dict(kind='graphic',cells=4,forbidden=[])
    result = solve(world, 0, public['target'], {key:active}, public['budget']-paid_checks,
                   3, public['action_order'], ordering=public['ordering'])
    row = score_submission(world,0,public['target'],result,3)
    row['costs']['checking'] += paid_checks
    row['costs']['total_online'] += paid_checks
    row['costs']['envelope'] = public['budget']
    return dict(row, convention='matched total declared online operations',
                rejected_fragments=len(library)-len(active),checks=checks)


def original_matrix(case, budgets=(32,128)):
    rows=[]
    for allocation, acquisition in sorted(case['acquisitions'].items()):
        for capacity in (1,2):
            learned=acquire(canonical(dict(schema=SCHEMA,processed=acquisition['request']['processed'],capacity=capacity,representation='fragments')))
            for stratum,targets in sorted(case['transfer_targets'].items()):
                for target_index,target in enumerate(targets):
                    for budget in budgets:
                        for checking in (False,True):
                            request=dict(schema=SCHEMA,library=learned['library'],target=target,budget=budget,checking=checking,
                                         planner='enumeration',action_order=list(range(8)),ordering='fixed',representation='fragments')
                            row=predict(canonical(request))
                            rows.append(dict(allocation=allocation,capacity=capacity,stratum=stratum,target_index=target_index,
                                             budget=budget,checking=checking,request=request,acquisition=learned,**row))
    return rows


def balanced_cases(namespace,histories=4):
    cases=[]
    for permutation in permutations(range(4)):
        for teacher,student in ((0.6,0.7),(0.8,0.825),(1.0,0.95)):
            unit=digest([list(permutation),teacher,student])
            for history in range(histories):
                topics=[0,1]*16
                random.Random(seed_for(namespace,unit,history,'offers')).shuffle(topics)
                motifs=[list(permutation[:2]),list(permutation[2:])]
                offers=[]
                for offer,topic in enumerate(topics):
                    rng=random.Random(seed_for(namespace,unit,history,'instruction',offer))
                    instruction=motifs[topic].copy()
                    if rng.random()>teacher:instruction[rng.randrange(2)]=rng.randrange(8)
                    offers.append(dict(offer=offer,topic=topic,instruction=instruction,intended_target=sum(1<<c for c in motifs[topic])))
                prepared=dict(constructor=dict(permutation=list(permutation),teacher_reliability=teacher,student_reliability=student),
                              offers=offers,transfer_targets=[])
                offered=offered_view(prepared);prepared['allocations']=select_offers(canonical(offered))
                performed,private=attention_craft.perform(prepared,CONDITION,history,namespace+':'+unit)
                acquisitions={name:dict(request={'schema':'v18.selective-acquisition.1','processed':records})
                              for name,records in performed['processed'].items()}
                targets=dict(compatible=[sum(1<<c for c in [*motifs[0],b]) for b in motifs[1]],
                             changed=[(1<<a)|(1<<b) for a in motifs[0] for b in motifs[1]])
                cases.append(dict(case_id=digest([namespace,unit,history]),structural_unit=unit,history=history,
                    offered=offered,allocations=prepared['allocations'],acquisitions=acquisitions,transfer_targets=targets,
                    private=dict(constructor=prepared['constructor'],offers=offers,trials=private),
                    structural_scope='one four-cell law; 24 ordered motif relabelings crossed with three acquisition reliability contexts'))
    return cases


def common_matrix(case,budgets=(32,128),boundary_budgets=(64,96),balanced=True):
    rows=[]
    orders=[(f'rotation-{i}',list(range(i,8))+list(range(i)),'fixed') for i in range(8 if balanced else 1)]
    orders.append(('goal-relevance',list(range(8)),'goal'))
    representations=[]
    for allocation,acquisition in sorted(case['acquisitions'].items()):
        for capacity in (1,2):
            for representation in ('fragments','episodes'):
                learned=acquire(canonical(dict(schema=SCHEMA,processed=acquisition['request']['processed'],
                                    capacity=capacity,representation=representation)))
                for checking in (False,True):
                    representations.append((allocation,capacity,representation,checking,learned))
    representations.append(('none',0,'primitive',False,dict(library=[],costs={'storage_primitives':0})))
    for stratum,targets in sorted(case['transfer_targets'].items()):
        for target_index,target in enumerate(targets):
            for order_name,order,ordering in orders:
                caps=list(budgets)+(list(boundary_budgets) if order_name=='rotation-0' else [])
                for allocation,capacity,representation,checking,learned in representations:
                    for budget in caps:
                        request=dict(schema=SCHEMA,library=learned['library'],target=target,budget=budget,checking=checking,
                                     planner='state',action_order=order,ordering=ordering,representation=representation)
                        row=predict(canonical(request))
                        rows.append(dict(allocation=allocation,capacity=capacity,representation=representation,checking=checking,
                            stratum=stratum,target_index=target_index,budget=budget,action_order_stratum=order_name,
                            acquisition=learned,request=request,planner='state',**row))
                        # Original enumeration is a separate cost convention; episodes are
                        # a common-planner rival. All fixed order rotations remain crossed.
                        if representation!='episodes' and ordering=='fixed' and budget in budgets:
                            request=dict(request,planner='enumeration')
                            rows.append(dict(allocation=allocation,capacity=capacity,representation=representation,checking=checking,
                                stratum=stratum,target_index=target_index,budget=budget,action_order_stratum=order_name,
                                acquisition=learned,request=request,planner='enumeration',**predict(canonical(request))))
    return rows
