"""Dependency version spaces, matched observation selection and episodic rivals."""
from collections import defaultdict,deque
from copy import deepcopy
from itertools import product
import json
import random

from ..v16.records import canonical,digest,seed_for
from .common import Work,Exhausted,execute,step,actions,identity,distance,solve,score_submission
from .g1 import laws

SCHEMA='v18.1.g2.1'


def observed(world,query,work=None):
    if query['kind']=='context':
        if work:work.charge('checking')
        return dict(parent=world['parents'][query['part']],primitive_cost=0)
    state=identity(query['initial']);stopped=False;trace=[]
    for action in query['program']:
        if work:work.charge('checking')
        after,stop,legal=step(world,state,stopped,action)
        trace.append(dict(before=list(state),action=action,after=list(after),legal=legal,stopped=stop))
        if not legal:
            return dict(state=list(state),legal=False,stopped=stopped,primitive_cost=len(trace),trace=trace)
        state,stopped=after,stop
    return dict(state=list(state),legal=True,stopped=stopped,primitive_cost=len(trace),trace=trace)


def compatible(hypotheses,observations,work=None):
    retained=[]
    for model in hypotheses:
        good=True
        for observation in observations:
            expected=observed(model,observation['query'],work)
            if expected!=observation['outcome']:
                good=False;break
        if good:retained.append(model)
    return retained


def model_plan(model,initial,target,max_steps,work=None):
    """Explicitly paid shortest native model plan for query decisions."""
    queue=deque([(tuple(initial),False,[])])
    seen={(tuple(initial),False)}
    while queue:
        state,stopped,program=queue.popleft()
        if work:work.charge('selection')
        if list(state)==target and stopped:return program
        if stopped or len(program)>=max_steps:continue
        for action in actions(model):
            if work:work.charge('checking')
            after,stop,legal=step(model,state,stopped,action)
            if legal and (after,stop) not in seen:
                seen.add((after,stop));queue.append((after,stop,program+[action]))
    return None


def menu(defaults,target,models):
    n=len(defaults);initial=list(defaults)
    items=[dict(kind='action',initial=[-1]*n,program=[i]) for i in range(n)]
    items += [dict(kind='action',initial=initial,program=[2*n+i]) for i in range(n)]
    items += [dict(kind='routine',initial=initial,program=[3*n])]
    programs=sorted({tuple(model_plan(model,initial,target,2*n+2)) for model in models})
    items += [dict(kind='routine',initial=initial,program=list(p)) for p in programs]
    items += [dict(kind='context',part=i) for i in range(1,n)]
    return items


def make_cases(namespace,*,histories=4,misspecified=True):
    cases=[]
    for truth in laws(3):
        models=[dict(kind='assembly',parents=list(p),defaults=truth['defaults'],forbidden=[])
                for p in product(*[range(-1,i) for i in range(3)])]
        for part in range(3):
            target=truth['defaults'].copy();target[part]=1-target[part]
            queries=menu(truth['defaults'],target,models)
            demo_pool=[q for q in queries if q['kind']!='context']
            for selection in ('uniform','task-relevant','selective-positive'):
                for donor in ('same-law','cross-law'):
                    unit=digest([truth,target,selection,donor])
                    for history in range(histories):
                        rng=random.Random(seed_for(namespace,unit,history,'training'))
                        ordered=list(demo_pool);rng.shuffle(ordered)
                        if selection=='task-relevant':
                            ordered.sort(key=lambda q: -sum(a%3==part for a in q['program']))
                        elif selection=='selective-positive':
                            ordered=[q for q in ordered if observed(truth,q)['legal']]
                        chosen=[deepcopy(ordered[i%len(ordered)]) for i in range(3)]
                        source_law=None
                        if donor=='cross-law':
                            source_law=next(m for m in models if m!=truth)
                            donor_program=model_plan(source_law,truth['defaults'],target,8)
                            chosen[-1]=dict(kind='routine',initial=truth['defaults'],program=donor_program)
                        observations=[dict(query=q,outcome=observed(truth,q),
                            source_context='cross-law-attempt' if donor=='cross-law' and i==2 else 'own-law') for i,q in enumerate(chosen)]
                        # No compatible-donor rejection/resampling. Actual failed transfer remains observed.
                        for absent in ((False,True) if misspecified else (False,)):
                            hypotheses=[m for m in models if not absent or m!=truth]
                            public=dict(schema=SCHEMA,models=hypotheses,observations=observations,menu=queries,
                                initial=truth['defaults'],target=target,max_steps=8,
                                query_seed=seed_for(namespace,unit,history,'query'),action_order=list(range(10)))
                            cases.append(dict(case_id=digest([namespace,unit,history,absent]),structural_unit=unit,
                                history=history,selection=selection,donor=donor,truth_excluded=absent,
                                public=public,private=dict(true_world=truth,donor_world=source_law,
                                    donor_source_success=None if source_law is None else observed(source_law,chosen[-1])['legal']),
                                coverage=dict(demonstrations=3,distinct_queries=len({digest(q) for q in chosen}),
                                    target_part_actions=sum(a%3==part for q in chosen for a in q['program']),
                                    failed_demonstrations=sum(not o['outcome']['legal'] for o in observations))))
    return cases


def contract(payload):
    public=json.loads(payload)
    if set(public)!={'schema','models','observations','menu','initial','target','max_steps','query_seed','action_order'} or public['schema']!=SCHEMA:
        raise ValueError('G2 public schema violation')
    if not public['models'] or any(set(o)!={'query','outcome','source_context'} for o in public['observations']):
        raise ValueError('G2 invalid public evidence')
    return public


def select_query(payload,observations,used,policy,work):
    public=contract(payload)
    available=[i for i in range(len(public['menu'])) if i not in used]
    if not available:return None
    if policy=='uniform':
        work.charge('selection')
        rng=random.Random(public['query_seed']+len(used))
        return rng.choice(available)
    if policy=='fixed':
        work.charge('selection')
        # Predeclared diagnostic sequence: inspect late-part support, then a routine.
        ranked=sorted(available,key=lambda i:(public['menu'][i]['kind']!='context',-public['menu'][i].get('part',-1),i))
        return ranked[0]
    if policy!='decision':raise ValueError('unknown query policy')
    hypotheses=compatible(public['models'],observations,work)
    if not hypotheses:return None
    # The controller is shared across downstream representations. Its full cost
    # is charged to each arm; a surface reader with this policy is a hybrid.
    plans=[model_plan(m,public['initial'],public['target'],public['max_steps'],work) for m in hypotheses]
    decisions=[digest(p) for p in plans]
    best=None
    for index in available:
        work.charge('selection')
        partitions=defaultdict(list)
        for h,model in enumerate(hypotheses):
            answer=observed(model,public['menu'][index],work)
            partitions[digest(answer)].append(decisions[h])
        # Expected disagreements between the candidate best task plans after observation.
        residual=sum(len(group)-max(group.count(x) for x in set(group)) for group in partitions.values())
        query=public['menu'][index]
        cost=1 if query['kind']=='context' else len(query['program'])
        score=(residual,cost,index)
        if best is None or score<best[0]:best=(score,index)
    return best[1]


def episode_plan(public,observations,work):
    edges=defaultdict(list)
    for observation in observations:
        if observation['query']['kind']=='context':continue
        for transition in observation['outcome']['trace']:
            work.charge('retrieval')
            if transition['legal']:
                edge=(transition['action'],tuple(transition['after']),transition['stopped'])
                if edge not in edges[tuple(transition['before'])]:edges[tuple(transition['before'])].append(edge)
    queue=deque([(tuple(public['initial']),False,[])])
    seen={(tuple(public['initial']),False)}
    while queue:
        work.charge('selection')
        state,stopped,program=queue.popleft()
        if list(state)==public['target'] and stopped:
            if work.spent+len(program)<=work.cap:
                work.charge('actual_execution',len(program));return program
            return None
        if stopped or len(program)>=public['max_steps']:continue
        for action,after,stop in edges[state]:
            work.charge('proposal_generation')
            if (after,stop) not in seen:
                seen.add((after,stop));queue.append((after,stop,program+[action]))
    return None


def predictions(models,observations,probes,method,true_world=None):
    # A distinct forecast deployment; its work is reported, not silently free or
    # charged again to construction. Probes are never added to task evidence.
    work=Work(1000000);probs=[]
    hypotheses=([true_world] if method=='known-law' else
                compatible(models,observations,work) if method=='dependencies' else [])
    for query in probes:
        if method in ('dependencies','known-law'):
            values=[observed(m,query,work)['legal'] for m in hypotheses]
            probs.append(sum(values)/len(values) if values else 0.5)
        else:
            values=[]
            for observation in observations:
                if observation['query']['kind']=='context':continue
                for transition in observation['outcome']['trace']:
                    work.charge('retrieval')
                    if transition['before']==query['initial'] and [transition['action']]==query['program']:
                        values.append(transition['legal'])
            probs.append(sum(values)/len(values) if values else 0.5)
    return probs,work.spent,len(hypotheses)


def parent_marginals(hypotheses):
    if not hypotheses:return []
    return [{str(p):sum(m['parents'][i]==p for m in hypotheses)/len(hypotheses)
             for p in range(-1,i)} for i in range(len(hypotheses[0]['parents']))]


def evaluate(case,budgets=(512,2048,8192),query_counts=(0,1,2)):
    public=case['public'];payload=canonical(public);truth=case['private']['true_world'];rows=[]
    n=len(public['initial'])
    # The same held-out forecasts for every policy/count/budget. Two or three
    # orientations differ from training; menu routines only change one orientation.
    # All parts are attached, so these states are legal under every candidate law.
    probe_states=[[1-v for v in public['initial']],
                  [1-v if i<2 else v for i,v in enumerate(public['initial'])]]
    all_probes=[dict(kind='action',initial=s,program=[a]) for s in probe_states for a in range(3*n+1)]
    before_marginals=parent_marginals(compatible(public['models'],public['observations']))
    for policy in ('uniform','fixed','decision'):
        for count in query_counts:
            for budget in budgets:
                acquisition=Work(budget);observations=deepcopy(public['observations']);used=[];query_failed=False
                try:
                    for _ in range(count):
                        chosen=select_query(payload,observations,used,policy,acquisition)
                        if chosen is None:break
                        query=deepcopy(public['menu'][chosen])
                        # The actual observation is the only evaluator response to the reader.
                        outcome=observed(truth,query)
                        acquisition.charge('checking',max(1,outcome['primitive_cost']))
                        observations.append(dict(query=query,outcome=outcome,source_context='paid-observation'))
                        used.append(chosen)
                except Exhausted:
                    query_failed=True
                seen_actions={(tuple(t['before']),t['action']) for o in observations if o['query']['kind']!='context' for t in o['outcome']['trace']}
                probes=all_probes
                assert all((tuple(q['initial']),q['program'][0]) not in seen_actions for q in probes)
                truths=[observed(truth,q)['legal'] for q in probes]
                for method in ('dependencies','episodes','known-law'):
                    work=Work(budget,dict(acquisition.counts));program=None;posterior=[];model_status='not_inferred'
                    try:
                        if method=='episodes':program=episode_plan(public,observations,work)
                        else:
                            posterior=[truth] if method=='known-law' else compatible(public['models'],observations,work)
                            model_status='inconsistent' if not posterior else 'compatible_set'
                            if posterior:
                                # No arbitrary collapse to the first hypothesis: retain plans
                                # only if every compatible law predicts the same legal outcome.
                                proposals=[]
                                for model in posterior:
                                    proposal=model_plan(model,public['initial'],public['target'],public['max_steps'],work)
                                    if proposal is not None and proposal not in proposals:proposals.append(proposal)
                                for proposal in sorted(proposals,key=lambda p:(len(p),p)):
                                    work.charge('selection')
                                    outcomes=[observed(model,dict(kind='routine',initial=public['initial'],program=proposal),work) for model in posterior]
                                    if all(o['legal'] and o['stopped'] and o['state']==public['target'] for o in outcomes):
                                        work.charge('actual_execution',len(proposal));program=proposal;break
                    except Exhausted:pass
                    result=dict(program=program,costs=work.receipt())
                    row=score_submission(truth,public['initial'],public['target'],result,public['max_steps'])
                    probabilities,forecast_work,compatible_count=predictions(public['models'],observations,probes,method,truth)
                    squared=sum((p-int(t))**2 for p,t in zip(probabilities,truths))/len(probes) if probes else None
                    # Dependency recovery is separate from correct predictions/acting.
                    actual_posterior=posterior if method=='dependencies' and model_status!='not_inferred' else []
                    recovered=bool(actual_posterior) and all(m['parents']==truth['parents'] for m in actual_posterior)
                    after_marginals=parent_marginals(actual_posterior)
                    changed=[sum(abs(before.get(k,0)-after.get(k,0)) for k in set(before)|set(after))
                             for before,after in zip(before_marginals,after_marginals)]
                    rows.append(dict(method=method,query_policy=policy,requested_queries=count,acquired_queries=len(used),
                        query_indices=used,query_exhausted=query_failed,budget=budget,model_status=model_status,
                        forecast_compatible_laws=compatible_count,task_compatible_laws=len(actual_posterior),
                        inference_completed=method=='dependencies' and model_status!='not_inferred',
                        truth_in_compatible_set=truth in actual_posterior,
                        dependency_recovered=recovered,abstained_on_inconsistency=method=='dependencies' and model_status=='inconsistent' and program is None,
                        parent_marginal_change=changed,
                        marginal_change_scope='descriptive changes; cross-dependency changes can be licensed by correlated evidence',
                        forecast_brier=squared,forecast_probe_count=len(probes),forecast_operations=forecast_work,
                        forecast_probabilities=probabilities,forecast_truths=truths,forecast_probes=probes,
                        observation_record=observations,known_law_ceiling=method=='known-law',
                        acquisition=dict(processed_demonstrations=len(public['observations']),
                            source_execution_primitives=sum(o['outcome'].get('primitive_cost',0) for o in public['observations']),
                            failed_demonstrations=sum(o['outcome'].get('legal') is False for o in public['observations'])),
                        selector_contract='shared paid dependency-model controller for decision queries; episodic arm is a labeled hybrid there',
                        **row))
    return rows


def transfer_cases(namespace,per_stratum=64,histories=4,sizes=(5,7),families=('fork','chain','groups')):
    """Transfer the unchanged query rules to a declared bounded candidate family.

    Truth is included in this transfer screen; deliberate family exclusion remains
    a separate native experiment. Twelve sorted candidates receive a uniform prior;
    their presentation order never identifies the included truth.
    """
    from .g3 import topology,topology_signature
    cases=[]
    for n in sizes:
        pool={tuple([-1]*n),tuple([-1]+list(range(n-1)))}
        for f in ('fork','groups'):
            for draw in range(256):pool.add(tuple(topology(n,f,random.Random(seed_for(namespace,'candidate-pool',n,f,draw)))))
        for family in families:
            for selection in ('uniform','task-relevant','selective-positive'):
                for donor in ('same-law','cross-law'):
                    seen=set();draw=0
                    while len(seen)<per_stratum:
                        rng=random.Random(seed_for(namespace,n,family,selection,donor,draw,'world'));draw+=1
                        parents=topology(n,family,rng);defaults=[rng.randrange(2) for _ in range(n)]
                        truth=dict(kind='assembly',parents=parents,defaults=defaults,forbidden=[])
                        part=rng.randrange(n);target=defaults.copy();target[part]=1-target[part]
                        unit=digest([truth,target,selection,donor])
                        if unit in seen:continue
                        seen.add(unit)
                        candidates=[tuple(parents)]+rng.sample(sorted(pool-{tuple(parents)}),11)
                        models=[dict(kind='assembly',parents=list(p),defaults=defaults,forbidden=[]) for p in sorted(candidates)]
                        queries=menu(defaults,target,models);demo_pool=[q for q in queries if q['kind']!='context']
                        for history in range(histories):
                            hrng=random.Random(seed_for(namespace,unit,history,'training'));ordered=deepcopy(demo_pool);hrng.shuffle(ordered)
                            if selection=='task-relevant':ordered.sort(key=lambda q:-sum(a%n==part for a in q['program']))
                            elif selection=='selective-positive':ordered=[q for q in ordered if observed(truth,q)['legal']]
                            chosen=[deepcopy(ordered[i%len(ordered)]) for i in range(3)];source_law=None
                            if donor=='cross-law':
                                source_law=next(m for m in models if m!=truth)
                                chosen[-1]=dict(kind='routine',initial=defaults,program=model_plan(source_law,defaults,target,2*n+2))
                            observations=[dict(query=q,outcome=observed(truth,q),source_context='cross-law-attempt' if donor=='cross-law' and i==2 else 'own-law') for i,q in enumerate(chosen)]
                            public=dict(schema=SCHEMA,models=models,observations=observations,menu=queries,initial=defaults,target=target,
                                max_steps=2*n+2,query_seed=seed_for(namespace,unit,history,'query'),action_order=list(range(3*n+1)))
                            cases.append(dict(case_id=digest([namespace,unit,history]),structural_unit=unit,history=history,
                                selection=selection,donor=donor,truth_excluded=False,n=n,family=family,topology_signature=topology_signature(parents),
                                public=json.loads(canonical(public)),private=dict(true_world=deepcopy(truth),donor_world=deepcopy(source_law),
                                    donor_source_success=None if source_law is None else observed(source_law,chosen[-1])['legal']),
                                coverage=dict(demonstrations=3,distinct_queries=len({digest(q) for q in chosen}),
                                    target_part_actions=sum(a%n==part for q in chosen for a in q['program']),
                                    failed_demonstrations=sum(not o['outcome']['legal'] for o in observations))))
    return cases
