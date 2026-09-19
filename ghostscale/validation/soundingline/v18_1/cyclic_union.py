"""Opaque-label dependency acquisition when the public candidate union is cyclic.

Every candidate law is an acyclic assembly world.  Each case contains a true law
and a reciprocal alternative whose edge directions make the union cyclic.  The
target changes the parent at that reciprocal edge, so no action sequence can be
safe under both directions until evidence separates them.
"""
from collections import deque
from copy import deepcopy
import random

from ..v16.records import canonical,digest,seed_for
from . import direct,g2,permuted
from .common import Exhausted,Work,actions,identity,score_submission,step,validate_world
from .g3 import topology,topology_signature


def _cycle_permutation(parents,rng):
    """Put a leaf edge at opaque labels n-1 -> n-2."""
    n=len(parents)
    leaves={i for i in range(n) if i not in parents}
    edges=[(parent,child) for child,parent in enumerate(parents) if parent>=0 and child in leaves]
    if not edges:raise ValueError('candidate topology has no leaf edge')
    parent,child=rng.choice(edges)
    remaining=[i for i in range(n) if i not in (parent,child)]
    labels=list(range(n-2));rng.shuffle(labels)
    permutation=[None]*n
    for old,new in zip(remaining,labels):permutation[old]=new
    permutation[parent]=n-1;permutation[child]=n-2
    return permutation,parent,child


def _reciprocal_alternative(truth):
    """Reverse the designated n-1 -> n-2 leaf edge without creating a cycle."""
    n=len(truth['parents']);parent=n-1;child=n-2
    if truth['parents'][child]!=parent:raise ValueError('designated reciprocal edge absent')
    alternative=deepcopy(truth)
    alternative['parents'][child]=-1
    alternative['parents'][parent]=child
    validate_world(alternative)
    return alternative


def _candidate_models(namespace,n,family,draw,truth):
    models=[deepcopy(truth),_reciprocal_alternative(truth)]
    seen={tuple(model['parents']) for model in models}
    true_parent=truth['parents'][n-1]
    attempt=0
    families=('fork','chain','groups')
    while len(models)<12:
        rng=random.Random(seed_for(namespace,n,family,draw,'candidate',attempt));attempt+=1
        parents=topology(n,families[attempt%len(families)],rng)
        labels=list(range(n));rng.shuffle(labels)
        candidate=permuted.relabel_world(
            dict(kind='assembly',parents=parents,defaults=[0]*n,forbidden=[]),labels)
        candidate['defaults']=list(truth['defaults'])
        key=tuple(candidate['parents'])
        # The fixed first context query must isolate the truth. Other candidates
        # therefore cannot share the true parent of the target part.
        if key in seen or candidate['parents'][n-1]==true_parent:continue
        validate_world(candidate);seen.add(key);models.append(candidate)
        if attempt>10000:raise ValueError('could not construct candidate family')
    return sorted(models,key=canonical)


def make_cases(namespace,*,per_stratum=16,histories=4,sizes=(5,7),families=('fork','chain','groups')):
    cases=[]
    for n in sizes:
        for family in families:
            seen=set();draw=0
            while len(seen)<per_stratum:
                rng=random.Random(seed_for(namespace,n,family,draw,'world'));draw+=1
                base_parents=topology(n,family,rng)
                defaults=[rng.randrange(2) for _ in range(n)]
                permutation,_,_=_cycle_permutation(base_parents,rng)
                truth=permuted.relabel_world(
                    dict(kind='assembly',parents=base_parents,defaults=defaults,forbidden=[]),permutation)
                validate_world(truth)
                target=list(truth['defaults']);target[n-1]=1-target[n-1]
                models=_candidate_models(namespace,n,family,draw,truth)
                unit=digest([namespace,truth,target,models])
                if unit in seen:continue
                seen.add(unit)
                queries=g2.menu(truth['defaults'],target,models)
                neutral=dict(kind='routine',initial=list(truth['defaults']),program=[3*n])
                outcome=g2.observed(truth,neutral)
                for model in models:assert g2.observed(model,neutral)==outcome
                for history in range(histories):
                    observations=[dict(query=deepcopy(neutral),outcome=deepcopy(outcome),
                                       source_context='uninformative-stop-demonstration') for _ in range(3)]
                    public=dict(schema=g2.SCHEMA,models=deepcopy(models),observations=observations,
                        menu=deepcopy(queries),initial=list(truth['defaults']),target=target,
                        max_steps=2*n+2,query_seed=seed_for(namespace,unit,history,'query'),
                        action_order=list(range(3*n+1)))
                    cases.append(dict(case_id=digest([namespace,unit,history]),structural_unit=unit,
                        history=history,selection='neutral-stop',donor='none',truth_excluded=False,
                        n=n,family=family,topology_signature=topology_signature(truth['parents']),
                        label_order='opaque-cyclic-union',label_permutation=permutation,
                        reciprocal_pair=[n-1,n-2],target_part=n-1,public=public,
                        private=dict(true_world=deepcopy(truth),donor_world=None,donor_source_success=None),
                        coverage=dict(demonstrations=3,distinct_queries=1,target_part_actions=0,
                                      failed_demonstrations=0,candidate_laws=len(models),
                                      candidate_union_cyclic=True)))
    return cases


def acquire(public,truth,policy,count,budget):
    """Reproduce G2 acquisition while retaining its isolated cost receipt."""
    payload=canonical(public);work=Work(budget);observations=deepcopy(public['observations'])
    used=[];exhausted=False
    try:
        for _ in range(count):
            chosen=g2.select_query(payload,observations,used,policy,work)
            if chosen is None:break
            query=deepcopy(public['menu'][chosen]);outcome=g2.observed(truth,query)
            work.charge('checking',max(1,outcome['primitive_cost']))
            observations.append(dict(query=query,outcome=outcome,source_context='paid-observation'))
            used.append(chosen)
    except Exhausted:
        exhausted=True
    return observations,used,exhausted,work


def _conditioned_direct(public,observations,acquisition):
    work=Work(acquisition.cap,dict(acquisition.counts));program=None;reason=None;posterior=[]
    try:
        posterior=g2.compatible(public['models'],observations,work)
        if not posterior:reason='public evidence is inconsistent with every candidate law'
        else:program,reason=direct.compile_models(
            posterior,public['initial'],public['target'],public['max_steps'],work)
    except Exhausted:
        reason='online work exhausted'
    return dict(program=program,costs=work.receipt(),unsupported_reason=reason,
                compatible_laws=len(posterior),storage_tokens=0,
                knowledge='candidate laws filtered only by public evidence, then their shared order compiled')


def _belief_search(public,observations,acquisition):
    """Primitive robust search over public candidate states, without a graph route."""
    work=Work(acquisition.cap,dict(acquisition.counts));program=None;posterior=[];states=0
    try:
        posterior=g2.compatible(public['models'],observations,work)
        if posterior:
            start=tuple((identity(public['initial']),False) for _ in posterior)
            queue=deque([(start,[])]);seen={start}
            while queue:
                work.charge('selection');belief,path=queue.popleft();states+=1
                if all(state==identity(public['target']) and stopped for state,stopped in belief):
                    if work.spent+len(path)<=work.cap:
                        work.charge('actual_execution',len(path));program=path
                    break
                if len(path)>=public['max_steps'] or all(stopped for _,stopped in belief):continue
                for action in public['action_order']:
                    work.charge('proposal_generation');after=[];legal=True
                    for model,(state,stopped) in zip(posterior,belief):
                        work.charge('hypothetical_execution')
                        next_state,next_stopped,ok=step(model,state,stopped,action)
                        if not ok:legal=False;break
                        after.append((identity(next_state),next_stopped))
                    if not legal:continue
                    candidate=tuple(after)
                    if candidate not in seen:
                        seen.add(candidate);queue.append((candidate,path+[action]))
    except Exhausted:
        pass
    return dict(program=program,costs=work.receipt(),compatible_laws=len(posterior),states=states,
                storage_tokens=0,knowledge='primitive robust search over public evidence-compatible candidate states')


def evaluate(case,budgets=(32768,),query_counts=(0,1,2)):
    rows=g2.evaluate(case,budgets,query_counts)
    public=case['public'];truth=case['private']['true_world']
    base={(row['query_policy'],row['requested_queries'],row['budget']):row
          for row in rows if row['method']=='dependencies'}
    for policy in ('uniform','fixed','decision'):
        for count in query_counts:
            for budget in budgets:
                observations,used,query_exhausted,acquisition=acquire(public,truth,policy,count,budget)
                reference=base[(policy,count,budget)]
                assert digest(observations)==digest(reference['observation_record'])
                assert used==reference['query_indices'] and query_exhausted==reference['query_exhausted']
                for method,runner in (('conditioned-direct',_conditioned_direct),
                                      ('candidate-set-primitive',_belief_search)):
                    result=runner(public,observations,acquisition)
                    row=score_submission(truth,public['initial'],public['target'],result,public['max_steps'])
                    rows.append(dict(method=method,query_policy=policy,requested_queries=count,
                        acquired_queries=len(used),query_indices=used,query_exhausted=query_exhausted,
                        observation_record=observations,comparison_role='equally informed action rival',
                        budget=budget,**row))
    return rows


def _structured_cached_action(public,truth,observations,acquisition,method):
    """Run the unchanged structured action method after separately paid selection."""
    work=Work(acquisition.cap,dict(acquisition.counts));program=None;posterior=[]
    model_status='not_inferred'
    try:
        posterior=[truth] if method=='known-law' else g2.compatible(public['models'],observations,work)
        model_status='inconsistent' if not posterior else 'compatible_set'
        if posterior:
            proposals=[]
            for model in posterior:
                proposal=g2.model_plan(model,public['initial'],public['target'],public['max_steps'],work)
                if proposal is not None and proposal not in proposals:proposals.append(proposal)
            for proposal in sorted(proposals,key=lambda p:(len(p),p)):
                work.charge('selection')
                outcomes=[g2.observed(model,dict(kind='routine',initial=public['initial'],program=proposal),work)
                          for model in posterior]
                if all(o['legal'] and o['stopped'] and o['state']==public['target'] for o in outcomes):
                    work.charge('actual_execution',len(proposal));program=proposal;break
    except Exhausted:
        pass
    result=dict(program=program,costs=work.receipt())
    row=score_submission(truth,public['initial'],public['target'],result,public['max_steps'])
    n=len(public['initial'])
    probe_states=[[1-v for v in public['initial']],
                  [1-v if i<2 else v for i,v in enumerate(public['initial'])]]
    probes=[dict(kind='action',initial=state,program=[action])
            for state in probe_states for action in range(3*n+1)]
    seen_actions={(tuple(t['before']),t['action']) for observation in observations
                  if observation['query']['kind']!='context' for t in observation['outcome']['trace']}
    assert all((tuple(query['initial']),query['program'][0]) not in seen_actions for query in probes)
    truths=[g2.observed(truth,query)['legal'] for query in probes]
    probabilities,forecast_work,compatible_count=g2.predictions(
        public['models'],observations,probes,method,truth)
    squared=sum((probability-int(actual))**2 for probability,actual in zip(probabilities,truths))/len(probes)
    actual_posterior=posterior if method=='dependencies' and model_status!='not_inferred' else []
    return dict(model_status=model_status,forecast_compatible_laws=compatible_count,
        task_compatible_laws=len(actual_posterior),inference_completed=method=='dependencies' and model_status!='not_inferred',
        truth_in_compatible_set=truth in actual_posterior,
        dependency_recovered=bool(actual_posterior) and all(model['parents']==truth['parents'] for model in actual_posterior),
        abstained_on_inconsistency=method=='dependencies' and model_status=='inconsistent' and program is None,
        forecast_brier=squared,forecast_probe_count=len(probes),forecast_operations=forecast_work,
        forecast_probabilities=probabilities,forecast_truths=truths,forecast_probes=probes,
        known_law_ceiling=method=='known-law',**row)


def evaluate_cached_decision(case,online_budget=32768,selector_budget=32768):
    """Separate decision-query content from its charged online selection work.

    The selector is executed and fully recorded under its own envelope.  Its chosen
    observation is then supplied to each unchanged action method, which gets a fresh
    online envelope but still pays the observation's execution/checking cost.  This
    exposed-context diagnostic is not an equal-total-work primary comparison.
    """
    public=case['public'];truth=case['private']['true_world'];payload=canonical(public)
    observations=deepcopy(public['observations']);selector=Work(selector_budget)
    chosen=None;selector_exhausted=False
    try:
        chosen=g2.select_query(payload,observations,[],'decision',selector)
    except Exhausted:
        selector_exhausted=True
    acquisition=Work(online_budget);used=[]
    if chosen is not None:
        query=deepcopy(public['menu'][chosen]);outcome=g2.observed(truth,query)
        acquisition.charge('checking',max(1,outcome['primitive_cost']))
        observations.append(dict(query=query,outcome=outcome,source_context='paid-observation'))
        used.append(chosen)
    compatible=g2.compatible(public['models'],observations)
    rows=[]
    for method in ('dependencies','known-law'):
        rows.append(dict(method=method,**_structured_cached_action(
            public,truth,observations,acquisition,method)))
    for method,runner in (('conditioned-direct',_conditioned_direct),
                          ('candidate-set-primitive',_belief_search)):
        result=runner(public,observations,acquisition)
        rows.append(dict(method=method,**score_submission(
            truth,public['initial'],public['target'],result,public['max_steps'])))
    selector_costs=selector.receipt()
    for row in rows:
        row.update(query_policy='decision-cached',requested_queries=1,
            acquired_queries=len(used),query_indices=used,query_exhausted=selector_exhausted,
            observation_record=deepcopy(observations),budget=online_budget,
            selector_budget=selector_budget,selector_costs=deepcopy(selector_costs),
            selector_operations=selector_costs['total_online'],
            combined_operations=selector_costs['total_online']+row['costs']['total_online'],
            query_compatible_laws=len(compatible),query_isolates_truth=compatible==[truth],
            comparison_role='exposed descriptive diagnostic: selected evidence cached before action',
            selector_contract='decision query computed from public candidates and recorded separately; no selector work is hidden')
    return rows
