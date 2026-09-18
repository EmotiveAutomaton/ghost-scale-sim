"""Opaque part-label intervention over the retained G2 transfer structures."""
from copy import deepcopy
import random

from ..v16.records import canonical,digest,seed_for
from . import direct,g2


def relabel_world(world,permutation):
    n=len(permutation);parents=[None]*n;defaults=[None]*n
    for old,new in enumerate(permutation):
        parent=world['parents'][old]
        parents[new]=-1 if parent<0 else permutation[parent]
        defaults[new]=world['defaults'][old]
    return dict(kind='assembly',parents=parents,defaults=defaults,
                forbidden=[relabel_action(a,permutation) for a in world.get('forbidden',[])])


def relabel_state(state,permutation):
    result=[None]*len(permutation)
    for old,new in enumerate(permutation):result[new]=state[old]
    return result


def relabel_action(action,permutation):
    n=len(permutation)
    if action==3*n:return action
    operation,old=divmod(action,n)
    return operation*n+permutation[old]


def relabel_query(query,permutation):
    if query['kind']=='context':return dict(kind='context',part=permutation[query['part']])
    return dict(kind=query['kind'],initial=relabel_state(query['initial'],permutation),
                program=[relabel_action(a,permutation) for a in query['program']])


def permutation_for(case,namespace):
    n=len(case['public']['initial']);rng=random.Random(seed_for(namespace,case['structural_unit'],'labels'))
    truth=case['private']['true_world']
    for _ in range(1000):
        permutation=list(range(n));rng.shuffle(permutation)
        parents=relabel_world(truth,permutation)['parents']
        if permutation!=list(range(n)) and any(p>i for i,p in enumerate(parents) if p>=0):return permutation
    raise ValueError('could not hide topological label order')


def relabel_case(case,namespace):
    permutation=permutation_for(case,namespace);public=case['public'];truth=relabel_world(case['private']['true_world'],permutation)
    models=[relabel_world(model,permutation) for model in public['models']]
    menu=[relabel_query(query,permutation) for query in public['menu']]
    observations=[]
    for retained in public['observations']:
        query=relabel_query(retained['query'],permutation)
        observations.append(dict(query=query,outcome=g2.observed(truth,query),source_context=retained['source_context']))
    donor=(None if case['private']['donor_world'] is None else relabel_world(case['private']['donor_world'],permutation))
    transformed=dict(schema=g2.SCHEMA,models=models,observations=observations,menu=menu,
        initial=relabel_state(public['initial'],permutation),target=relabel_state(public['target'],permutation),
        max_steps=public['max_steps'],query_seed=public['query_seed'],
        action_order=[relabel_action(action,permutation) for action in public['action_order']])
    source_success=(None if donor is None else g2.observed(donor,observations[-1]['query'])['legal'])
    unit=digest([namespace,case['structural_unit'],permutation])
    return dict(case_id=digest([namespace,case['case_id'],permutation]),structural_unit=unit,history=case['history'],
        selection=case['selection'],donor=case['donor'],truth_excluded=case['truth_excluded'],n=case['n'],family=case['family'],
        topology_signature=case['topology_signature'],label_order='opaque-permuted',label_permutation=permutation,
        public=deepcopy(transformed),private=dict(true_world=truth,donor_world=donor,donor_source_success=source_success,
            origin_case_id=case['case_id']),coverage={**case['coverage'],'label_intervention':True})


def make_cases(cases,namespace):
    return [relabel_case(case,namespace) for case in cases]


def evaluate(case,budgets=(32768,),query_counts=(0,2)):
    rows=g2.evaluate(case,budgets,query_counts)
    public=case['public']
    direct_case=dict(public=dict(schema=direct.SCHEMA,models=public['models'],initial=public['initial'],
        target=public['target'],max_steps=public['max_steps']),private=case['private'])
    for row in direct.evaluate(direct_case,budgets):
        row.update(query_policy='none',requested_queries=0,acquired_queries=0,
            comparison_role='equally informed public-candidate-union compiler')
        rows.append(row)
    return rows
