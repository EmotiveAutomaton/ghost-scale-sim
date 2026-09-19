"""Opaque-label dependency acquisition when the public candidate union is cyclic.

Every candidate law is an acyclic assembly world.  Each case contains a true law
and a reciprocal alternative whose edge directions make the union cyclic.  The
target changes the parent at that reciprocal edge, so no action sequence can be
safe under both directions until evidence separates them.
"""
from collections import deque
from copy import deepcopy
from itertools import combinations
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


def _reverse_edge(world,parent,child):
    """Reverse one declared edge while detaching the old parent relation."""
    if world['parents'][child]!=parent:raise ValueError('declared reciprocal edge absent')
    alternative=deepcopy(world)
    alternative['parents'][child]=-1
    alternative['parents'][parent]=child
    validate_world(alternative)
    return alternative


def _two_reversible_edges(parents):
    """Find two vertex-disjoint edges whose four reversal combinations stay acyclic."""
    edges=[(parent,child) for child,parent in enumerate(parents) if parent>=0]
    world=dict(kind='assembly',parents=list(parents),defaults=[0]*len(parents),forbidden=[])
    for first,second in combinations(edges,2):
        if len(set(first+second))<4:continue
        try:
            one=_reverse_edge(world,*first)
            two=_reverse_edge(world,*second)
            both=_reverse_edge(one,*second)
        except ValueError:
            continue
        if len({tuple(candidate['parents']) for candidate in (world,one,two,both)})==4:
            return first,second
    return None


def _multi_target_models(truth):
    pairs=((1,3),(2,4))
    first=_reverse_edge(truth,*pairs[0])
    second=_reverse_edge(truth,*pairs[1])
    both=_reverse_edge(first,*pairs[1])
    models=sorted((deepcopy(truth),first,second,both),key=canonical)
    if len({canonical(model) for model in models})!=4:raise ValueError('multi-target candidates collapsed')
    return models


def _common_valid_state(models,state):
    """Whether an assembly state respects every public candidate dependency."""
    return all(all(value<0 or parent<0 or state[parent]>=0
                   for value,parent in zip(state,model['parents'])) for model in models)


def _target_action_probes(defaults,target,models):
    """Construct public one-step probes that split each changed target relation.

    Probe states use only the shared defaults and must be physically valid under
    every candidate law.  The observation is an executed action outcome, never a
    direct report of a parent label.
    """
    n=len(defaults);changed=[part for part,(before,after) in enumerate(zip(defaults,target))
                            if before!=after];probes=[]
    for part in changed:
        best=None
        for mask in range(1<<n):
            state=[defaults[index] if mask&(1<<index) else -1 for index in range(n)]
            if not _common_valid_state(models,state):continue
            for action in (part,n+part,2*n+part):
                query=dict(kind='action',initial=state,program=[action])
                groups={}
                for model in models:
                    answer=canonical(g2.observed(model,query))
                    groups[answer]=groups.get(answer,0)+1
                if sorted(groups.values())!=[2,2]:continue
                score=(sum(value>=0 for value in state),action//n,tuple(state),action)
                if best is None or score<best[0]:best=(score,query)
        if best is None:raise ValueError('no common-state target action separates candidates')
        probes.append(best[1])
    if len({canonical(probe) for probe in probes})!=len(changed):
        raise ValueError('target action probes collapsed')
    return probes


def complete_action_menu(models):
    """Every one-action probe at every common-valid default/absent state.

    Menu construction does not inspect a target, candidate outcome, or hidden
    truth. Context queries remain available to the two labeled control policies.
    Physical setup and menu enumeration are offline apparatus work, not free
    online actions by the reader.
    """
    if not models:raise ValueError('public candidate family required')
    defaults=models[0]['defaults'];n=len(defaults)
    if any(model['defaults']!=defaults for model in models):
        raise ValueError('shared public attachment defaults required')
    queries=[]
    for mask in range(1<<n):
        state=[defaults[index] if mask&(1<<index) else -1 for index in range(n)]
        if _common_valid_state(models,state):
            queries.extend(dict(kind='action',initial=list(state),program=[action])
                           for action in range(3*n))
    queries.extend(dict(kind='context',part=part) for part in range(1,n))
    return queries


def _menu_with_depth(defaults,target,models,max_steps,*,action_probes=False):
    n=len(defaults);initial=list(defaults)
    items=[dict(kind='action',initial=[-1]*n,program=[i]) for i in range(n)]
    items += [dict(kind='action',initial=initial,program=[2*n+i]) for i in range(n)]
    items += [dict(kind='routine',initial=initial,program=[3*n])]
    programs=[]
    for model in models:
        program=g2.model_plan(model,initial,target,max_steps)
        if program is None:raise ValueError('candidate target is not reachable inside the declared depth')
        if tuple(program) not in programs:programs.append(tuple(program))
    programs=sorted(programs)
    items += [dict(kind='routine',initial=initial,program=list(program)) for program in programs]
    if action_probes:
        existing={canonical(item) for item in items}
        items += [probe for probe in _target_action_probes(defaults,target,models)
                  if canonical(probe) not in existing]
    items += [dict(kind='context',part=i) for i in range(1,n)]
    return items


def make_multi_target_cases(namespace,*,per_stratum=64,histories=4,
                            families=('fork','chain','groups'),action_probes=False,
                            exclude_signatures=()):
    """Fresh seven-part contexts with two independent target-relevant cycles.

    Target parts occupy labels 1 and 2; their reciprocal counterparts occupy 3
    and 4.  Labels 5 and 6 are deliberately invariant across the four public
    candidates, so the historical late-label fixed sequence is a real placebo.
    """
    cases=[];n=7;target_parts=(1,2);reciprocal_pairs=((1,3),(2,4));max_steps=4*n+2
    excluded=set(exclude_signatures)
    for family in families:
        seen=set();draw=0
        while len(seen)<per_stratum:
            rng=random.Random(seed_for(namespace,n,family,draw,'world'));draw+=1
            base_parents=topology(n,family,rng);edges=_two_reversible_edges(base_parents)
            if edges is None:
                if draw>per_stratum*1000:raise ValueError('insufficient reversible multi-target support')
                continue
            (parent_a,child_a),(parent_b,child_b)=edges
            remaining=[part for part in range(n) if part not in (parent_a,parent_b,child_a,child_b)]
            filler=[0,5,6];rng.shuffle(filler)
            permutation=[None]*n
            for old,new in zip(remaining,filler):permutation[old]=new
            permutation[parent_a]=1;permutation[parent_b]=2
            permutation[child_a]=3;permutation[child_b]=4
            defaults=[rng.randrange(2) for _ in range(n)]
            truth=permuted.relabel_world(
                dict(kind='assembly',parents=base_parents,defaults=defaults,forbidden=[]),permutation)
            validate_world(truth)
            if truth['parents'][3]!=1 or truth['parents'][4]!=2:raise AssertionError('edge relabel failed')
            models=_multi_target_models(truth)
            if len({model['parents'][5] for model in models})!=1 or len({model['parents'][6] for model in models})!=1:
                raise AssertionError('fixed placebo labels changed')
            target=list(truth['defaults'])
            for part in target_parts:target[part]=1-target[part]
            physical_signature=digest([truth,target,models])
            if physical_signature in seen or physical_signature in excluded:continue
            try:
                queries=_menu_with_depth(
                    truth['defaults'],target,models,max_steps,action_probes=action_probes)
            except ValueError:
                if action_probes:continue
                raise
            seen.add(physical_signature);unit=digest([namespace,physical_signature])
            neutral=dict(kind='routine',initial=list(truth['defaults']),program=[3*n])
            outcome=g2.observed(truth,neutral)
            for model in models:assert g2.observed(model,neutral)==outcome
            for history in range(histories):
                observations=[dict(query=deepcopy(neutral),outcome=deepcopy(outcome),
                                   source_context='uninformative-stop-demonstration') for _ in range(3)]
                public=dict(schema=g2.SCHEMA,models=deepcopy(models),observations=observations,
                    menu=deepcopy(queries),initial=list(truth['defaults']),target=target,
                    max_steps=max_steps,query_seed=seed_for(namespace,unit,history,'query'),
                    action_order=list(range(3*n+1)))
                cases.append(dict(case_id=digest([namespace,unit,history]),structural_unit=unit,
                    history=history,selection='neutral-stop',donor='none',truth_excluded=False,
                    n=n,family=family,topology_signature=topology_signature(truth['parents']),
                    physical_signature=physical_signature,
                    label_order='opaque-two-target-cyclic-union',label_permutation=permutation,
                    reciprocal_pairs=[list(pair) for pair in reciprocal_pairs],target_parts=list(target_parts),
                    public=public,private=dict(true_world=deepcopy(truth),donor_world=None,
                                               donor_source_success=None),
                    coverage=dict(demonstrations=3,distinct_queries=1,target_part_actions=0,
                                  failed_demonstrations=0,candidate_laws=len(models),
                                  independent_reciprocal_cycles=2,
                                  target_action_probes=2 if action_probes else 0)))
    return cases


def make_misspecified_cases(namespace,*,per_stratum=64,histories=4,
                            families=('fork','chain','groups'),exclude_signatures=()):
    """Fresh cyclic contexts whose supplied three-law family excludes truth.

    The underlying four-law construction is made before truth is removed so the
    physical context can be checked against earlier support by its unchanged
    signature.  The public menu is then rebuilt from the three supplied laws
    alone; neither its contents nor its order can encode the excluded law.
    """
    cases=make_multi_target_cases(namespace,per_stratum=per_stratum,histories=histories,
        families=families,action_probes=False,exclude_signatures=exclude_signatures)
    for case in cases:
        public=case['public'];truth=case['private']['true_world']
        models=[deepcopy(model) for model in public['models'] if canonical(model)!=canonical(truth)]
        if len(models)!=3 or any(canonical(model)==canonical(truth) for model in models):
            raise AssertionError('misspecified family must contain three non-truth laws')
        public['models']=sorted(models,key=canonical)
        public['menu']=complete_action_menu(public['models'])
        case['truth_excluded']=True
        case['candidate_family_signature']=digest(public['models'])
        case['label_order']='opaque-two-target-cyclic-union-truth-excluded'
        case['coverage']['candidate_laws']=len(public['models'])
        case['coverage']['truth_excluded']=True
        case['coverage']['target_action_probes']=0
    return cases


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
            chosen=(select_target_aware(payload,observations,used,work)
                    if policy=='target-aware' else
                    select_misspecification_action(payload,observations,used,work)
                    if policy=='misspecification-action' else
                    select_target_action(payload,observations,used,work)
                    if policy=='target-action' else
                    g2.select_query(payload,observations,used,policy,work))
            if chosen is None:break
            query=deepcopy(public['menu'][chosen]);outcome=g2.observed(truth,query)
            work.charge('checking',max(1,outcome['primitive_cost']))
            observations.append(dict(query=query,outcome=outcome,source_context='paid-observation'))
            used.append(chosen)
    except Exhausted:
        exhausted=True
    return observations,used,exhausted,work


def select_target_aware(payload,observations,used,work):
    """Choose the most separating direct-parent query among changed target parts."""
    public=g2.contract(payload);hypotheses=g2.compatible(public['models'],observations,work)
    if not hypotheses:return None
    changed={part for part,(before,after) in enumerate(zip(public['initial'],public['target']))
             if before!=after}
    available=[index for index,query in enumerate(public['menu'])
               if index not in used and query['kind']=='context' and query['part'] in changed]
    if not available:return None
    best=None
    for index in available:
        work.charge('selection');groups={}
        for model in hypotheses:
            answer=canonical(g2.observed(model,public['menu'][index],work))
            groups[answer]=groups.get(answer,0)+1
        sizes=sorted(groups.values(),reverse=True)
        score=(max(sizes),sum(size*size for size in sizes),index)
        if best is None or score<best[0]:best=(score,index)
    return best[1]


def select_target_action(payload,observations,used,work):
    """Choose a separating executed action on a changed public target part."""
    public=g2.contract(payload);hypotheses=g2.compatible(public['models'],observations,work)
    if not hypotheses:return None
    n=len(public['initial'])
    changed={part for part,(before,after) in enumerate(zip(public['initial'],public['target']))
             if before!=after}
    available=[index for index,query in enumerate(public['menu'])
               if index not in used and query['kind']=='action' and
               len(query['program'])==1 and query['program'][0]%n in changed and
               _common_valid_state(public['models'],query['initial'])]
    if not available:return None
    best=None
    for index in available:
        work.charge('selection');groups={}
        for model in hypotheses:
            answer=canonical(g2.observed(model,public['menu'][index],work))
            groups[answer]=groups.get(answer,0)+1
        sizes=sorted(groups.values(),reverse=True)
        score=(max(sizes),sum(size*size for size in sizes),index)
        if best is None or score<best[0]:best=(score,index)
    return best[1]


def select_misspecification_action(payload,observations,used,work):
    """Choose independent target-part falsification probes without evaluator truth.

    Candidate partition quality is evaluated against the complete supplied
    family, not only its current survivors.  After one target part is queried,
    the other changed target part is preferred.  This prevents an arbitrary
    singleton survivor from turning every second query into an undirected tie.
    """
    public=g2.contract(payload);hypotheses=g2.compatible(public['models'],observations,work)
    if not hypotheses:return None
    n=len(public['initial'])
    changed={part for part,(before,after) in enumerate(zip(public['initial'],public['target']))
             if before!=after}
    used_parts={public['menu'][index]['program'][0]%n for index in used
                if public['menu'][index]['kind']=='action' and len(public['menu'][index]['program'])==1}
    preferred=changed-used_parts or changed
    available=[index for index,query in enumerate(public['menu'])
               if index not in used and query['kind']=='action' and
               len(query['program'])==1 and query['program'][0]%n in preferred and
               _common_valid_state(public['models'],query['initial'])]
    if not available:return None
    best=None
    for index in available:
        work.charge('selection');groups={}
        for model in public['models']:
            answer=canonical(g2.observed(model,public['menu'][index],work))
            groups[answer]=groups.get(answer,0)+1
        sizes=sorted(groups.values(),reverse=True)
        score=(max(sizes),sum(size*size for size in sizes),index)
        if best is None or score<best[0]:best=(score,index)
    return best[1]


def robust_setup_paths(public,hypotheses,work):
    """Shortest no-oracle setup paths from the task start to shared states.

    A path is retained only when every currently compatible candidate law accepts
    every action and reaches the same next state.  STOP is excluded: preparation
    must leave a live object on which the selected one-step query can execute.
    Search and candidate simulation are charged to the query-selection envelope.
    """
    if not hypotheses:return {}
    start=tuple(public['initial']);n=len(start)
    paths={start:[]};queue=deque([start])
    while queue:
        work.charge('selection')
        state=queue.popleft();path=paths[state]
        if len(path)>=public['max_steps']:continue
        for action in range(3*n):
            work.charge('proposal_generation')
            after=[];legal=True
            for model in hypotheses:
                work.charge('hypothetical_execution')
                next_state,stopped,ok=step(model,state,False,action)
                if not ok or stopped:
                    legal=False;break
                after.append(tuple(next_state))
            if not legal or len(set(after))!=1:continue
            next_state=after[0]
            if next_state not in paths:
                paths[next_state]=path+[action];queue.append(next_state)
    return paths


def select_physical_target_action(public,observations,used,work):
    """Choose an informative target action whose starting state can be built.

    Information partition is primary, then the shortest shared physical setup,
    then the frozen public menu order.  The selector receives candidate laws and
    prior public evidence, never evaluator truth.
    """
    hypotheses=g2.compatible(public['models'],observations,work)
    if not hypotheses:return None
    n=len(public['initial'])
    changed={part for part,(before,after) in enumerate(zip(public['initial'],public['target']))
             if before!=after}
    paths=robust_setup_paths(public,hypotheses,work)
    best=None
    for index,query in enumerate(public['menu']):
        if (index in used or query['kind']!='action' or len(query['program'])!=1 or
                query['program'][0]%n not in changed):
            continue
        setup=paths.get(tuple(query['initial']))
        if setup is None:continue
        work.charge('selection');groups={}
        for model in hypotheses:
            answer=canonical(g2.observed(model,query,work))
            groups[answer]=groups.get(answer,0)+1
        sizes=sorted(groups.values(),reverse=True)
        score=(max(sizes),sum(size*size for size in sizes),len(setup),index)
        if best is None or score<best[0]:best=(score,index,setup,len(paths))
    return None if best is None else dict(index=best[1],setup=best[2],reachable_states=best[3],
                                          compatible_laws=len(hypotheses))


def acquire_physical(public,truth,count,budget):
    """Acquire action evidence with no-oracle query-state preparation charged.

    Each query starts from a fresh object in the declared public task state.  The
    shared setup path, its physical execution and the one-step observation all use
    the same online envelope.  Supplying/resetting that fresh initial object and
    the candidate-law family remain apparatus assumptions.
    """
    work=Work(budget);observations=deepcopy(public['observations'])
    used=[];setups=[];reachable=[];exhausted=False
    try:
        for _ in range(count):
            selected=select_physical_target_action(public,observations,used,work)
            if selected is None:break
            query=deepcopy(public['menu'][selected['index']])
            state=tuple(public['initial'])
            for action in selected['setup']:
                work.charge('checking')
                state,stopped,legal=step(truth,state,False,action)
                if not legal or stopped:raise AssertionError('selected physical setup is not true-law legal')
            if list(state)!=query['initial']:
                raise AssertionError('selected physical setup misses the query state')
            # Setup is deliberately uninformative: every compatible law accepts
            # the same actions and reaches the same state.  Only the query outcome
            # enters the evidence record.
            outcome=g2.observed(truth,query,work)
            observations.append(dict(query=query,outcome=outcome,
                source_context='paid-physical-action-observation'))
            used.append(selected['index']);setups.append(list(selected['setup']))
            reachable.append(selected['reachable_states'])
    except Exhausted:
        exhausted=True
    return observations,used,setups,reachable,exhausted,work


def select_physical_misspecification_action(public,observations,used,work):
    """Choose a falsification probe that also has a shared physical setup.

    Setup feasibility is evaluated against the currently retained supplied laws,
    while information partitioning retains the complete supplied family.  The
    evaluator truth is never consulted by selection or setup search.
    """
    hypotheses=g2.compatible(public['models'],observations,work)
    if not hypotheses:return None
    n=len(public['initial'])
    changed={part for part,(before,after) in enumerate(zip(public['initial'],public['target']))
             if before!=after}
    used_parts={public['menu'][index]['program'][0]%n for index in used}
    preferred=changed-used_parts or changed
    paths=robust_setup_paths(public,hypotheses,work)
    best=None
    for index,query in enumerate(public['menu']):
        if (index in used or query['kind']!='action' or len(query['program'])!=1 or
                query['program'][0]%n not in preferred):
            continue
        setup=paths.get(tuple(query['initial']))
        if setup is None:continue
        work.charge('selection');groups={}
        for model in public['models']:
            answer=canonical(g2.observed(model,query,work))
            groups[answer]=groups.get(answer,0)+1
        sizes=sorted(groups.values(),reverse=True)
        score=(max(sizes),sum(size*size for size in sizes),len(setup),index)
        if best is None or score<best[0]:best=(score,index,setup,len(paths))
    return None if best is None else dict(index=best[1],setup=best[2],reachable_states=best[3],
                                          compatible_laws=len(hypotheses))


def acquire_physical_misspecified(public,truth,count,budget):
    """Acquire falsification evidence while physically preparing query states.

    A setup that the wrong supplied family predicts as shared may fail or diverge
    under evaluator truth.  That public physical outcome is retained as evidence;
    the intended one-step query executes only if its state was actually reached.
    """
    work=Work(budget);observations=deepcopy(public['observations'])
    used=[];setups=[];setup_records=[];reachable=[];exhausted=False
    try:
        for _ in range(count):
            selected=select_physical_misspecification_action(
                public,observations,used,work)
            if selected is None:break
            query=deepcopy(public['menu'][selected['index']])
            setup_query=dict(kind='routine',initial=deepcopy(public['initial']),
                             program=list(selected['setup']))
            setup_outcome=g2.observed(truth,setup_query,work)
            reached=(setup_outcome['legal'] and not setup_outcome['stopped'] and
                     setup_outcome['state']==query['initial'])
            setup_observation=None
            if selected['setup']:
                setup_observation=dict(query=setup_query,outcome=setup_outcome,
                    source_context='paid-physical-setup-outcome')
                observations.append(setup_observation)
            query_executed=False
            if reached:
                outcome=g2.observed(truth,query,work)
                observations.append(dict(query=query,outcome=outcome,
                    source_context='paid-physical-action-observation'))
                query_executed=True
            used.append(selected['index']);setups.append(list(selected['setup']))
            reachable.append(selected['reachable_states'])
            setup_records.append(dict(setup_observation=setup_observation,
                reached_query_state=reached,query_executed=query_executed))
            if not reached:break
    except Exhausted:
        exhausted=True
    return observations,used,setups,setup_records,reachable,exhausted,work


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


def evaluate_target_aware(case,budget=32768,query_counts=(1,2)):
    """Screen a cheap target-aware query rule against fixed and decision selection."""
    public=case['public'];truth=case['private']['true_world'];rows=[]
    for policy in ('fixed','decision','target-aware'):
        for count in query_counts:
            observations,used,query_exhausted,acquisition=acquire(
                public,truth,policy,count,budget)
            compatible=g2.compatible(public['models'],observations)
            current=[]
            for method in ('dependencies','known-law'):
                current.append(dict(method=method,**_structured_cached_action(
                    public,truth,observations,acquisition,method)))
            for method,runner in (('conditioned-direct',_conditioned_direct),
                                  ('candidate-set-primitive',_belief_search)):
                result=runner(public,observations,acquisition)
                current.append(dict(method=method,**score_submission(
                    truth,public['initial'],public['target'],result,public['max_steps'])))
            acquisition_costs=acquisition.receipt()
            for row in current:
                row.update(query_policy=policy,requested_queries=count,
                    acquired_queries=len(used),query_indices=list(used),query_exhausted=query_exhausted,
                    observation_record=deepcopy(observations),budget=budget,
                    acquisition_costs=deepcopy(acquisition_costs),
                    acquisition_operations=acquisition_costs['total_online'],
                    query_compatible_laws=len(compatible),query_isolates_truth=compatible==[truth],
                    target_changed_parts=[part for part,(before,after) in enumerate(
                        zip(public['initial'],public['target'])) if before!=after],
                    comparison_role='descriptive target-aware multi-part cyclic query screen',
                    selector_contract=('target-aware uses only changed public target parts and candidate parent outcomes; '
                                       'fixed and decision retain their existing public contracts'))
                rows.append(row)
    return rows


def evaluate_target_action(case,budget=32768,query_counts=(1,2)):
    """Compare target-action evidence with parent-query and fixed controls."""
    public=case['public'];truth=case['private']['true_world'];rows=[]
    for policy in ('fixed','target-aware','target-action'):
        for count in query_counts:
            observations,used,query_exhausted,acquisition=acquire(
                public,truth,policy,count,budget)
            compatible=g2.compatible(public['models'],observations)
            current=[]
            for method in ('dependencies','known-law'):
                current.append(dict(method=method,**_structured_cached_action(
                    public,truth,observations,acquisition,method)))
            for method,runner in (('conditioned-direct',_conditioned_direct),
                                  ('candidate-set-primitive',_belief_search)):
                result=runner(public,observations,acquisition)
                current.append(dict(method=method,**score_submission(
                    truth,public['initial'],public['target'],result,public['max_steps'])))
            acquisition_costs=acquisition.receipt()
            paid=observations[len(public['observations']):]
            target_action_parts=[observation['query']['program'][0]%len(public['initial'])
                                 for observation in paid if observation['query']['kind']=='action']
            for row in current:
                row.update(query_policy=policy,requested_queries=count,
                    acquired_queries=len(used),query_indices=list(used),query_exhausted=query_exhausted,
                    observation_record=deepcopy(observations),budget=budget,
                    acquisition_costs=deepcopy(acquisition_costs),
                    acquisition_operations=acquisition_costs['total_online'],
                    query_compatible_laws=len(compatible),query_isolates_truth=compatible==[truth],
                    target_changed_parts=[part for part,(before,after) in enumerate(
                        zip(public['initial'],public['target'])) if before!=after],
                    target_action_parts=target_action_parts,
                    direct_parent_cues_used=any(o['query']['kind']=='context' for o in paid),
                    comparison_role='descriptive target-relevant executed-action evidence screen',
                    selector_contract=('target-action uses only one-step executed actions on changed public target parts '
                                       'from states valid under every public candidate; target-aware is the direct-parent positive control'))
                rows.append(row)
    return rows


def evaluate_physical_action(case,budget=32768,query_counts=(1,2)):
    """Run the complete-menu action screen with charged physical preparation."""
    public=case['public'];truth=case['private']['true_world'];rows=[]
    for count in query_counts:
        observations,used,setups,reachable,query_exhausted,acquisition=acquire_physical(
            public,truth,count,budget)
        compatible=g2.compatible(public['models'],observations)
        current=[]
        for method in ('dependencies','known-law'):
            current.append(dict(method=method,**_structured_cached_action(
                public,truth,observations,acquisition,method)))
        for method,runner in (('conditioned-direct',_conditioned_direct),
                              ('candidate-set-primitive',_belief_search)):
            result=runner(public,observations,acquisition)
            current.append(dict(method=method,**score_submission(
                truth,public['initial'],public['target'],result,public['max_steps'])))
        acquisition_costs=acquisition.receipt()
        paid=observations[len(public['observations']):]
        for row in current:
            row.update(query_policy='target-action-physical',requested_queries=count,
                acquired_queries=len(used),query_indices=list(used),query_exhausted=query_exhausted,
                observation_record=deepcopy(observations),budget=budget,
                acquisition_costs=deepcopy(acquisition_costs),
                acquisition_operations=acquisition_costs['total_online'],
                physical_setup_paths=deepcopy(setups),
                physical_setup_operations=sum(map(len,setups)),
                reachable_setup_states=list(reachable),
                query_compatible_laws=len(compatible),query_isolates_truth=compatible==[truth],
                target_changed_parts=[part for part,(before,after) in enumerate(
                    zip(public['initial'],public['target'])) if before!=after],
                target_action_parts=[observation['query']['program'][0]%len(public['initial'])
                                     for observation in paid],
                comparison_role='descriptive charged physical query-preparation diagnostic',
                selector_contract=('query-state setup must use one action sequence legal under every '
                                   'currently compatible public candidate law; selection, candidate '
                                   'simulation, setup execution and observed action share the online envelope'),
                reset_contract=('each query receives a fresh object at the public task initial state; '
                                'provisioning that object is outside the count'),
                candidate_family_supplied=True,setup_uses_evaluator_truth=False)
            rows.append(row)
    return rows


def _episode_action(public,observations,acquisition):
    work=Work(acquisition.cap,dict(acquisition.counts));program=None
    try:program=g2.episode_plan(public,observations,work)
    except Exhausted:pass
    return dict(program=program,costs=work.receipt(),storage_tokens=0,
                knowledge='surface transition episodes only; supplied candidate laws are not consulted')


def _forced_candidate_direct(public,acquisition):
    """Negative control that ignores contradictory evidence and forces one law."""
    work=Work(acquisition.cap,dict(acquisition.counts));program=None;reason=None
    try:
        program,reason=direct.compile_models(
            [public['models'][0]],public['initial'],public['target'],public['max_steps'],work)
    except Exhausted:
        reason='online work exhausted'
    return dict(program=program,costs=work.receipt(),unsupported_reason=reason,
                compatible_laws=1,storage_tokens=0,
                knowledge='negative control: force the first canonically ordered supplied law after contradiction')


def evaluate_misspecified(case,budget=32768,query_counts=(1,2)):
    """Measure detection, abstention and unsafe forcing when truth is excluded."""
    public=case['public'];truth=case['private']['true_world'];rows=[]
    if truth in public['models'] or not case['truth_excluded']:
        raise ValueError('misspecification condition requires truth outside the supplied family')
    candidate_aware={'dependencies','conditioned-direct','candidate-set-primitive'}
    for count in query_counts:
        observations,used,query_exhausted,acquisition=acquire(
            public,truth,'misspecification-action',count,budget)
        compatible=g2.compatible(public['models'],observations)
        inconsistent=not compatible
        current=[]
        for method in ('dependencies','known-law'):
            current.append(dict(method=method,**_structured_cached_action(
                public,truth,observations,acquisition,method)))
        for method,runner in (('conditioned-direct',_conditioned_direct),
                              ('candidate-set-primitive',_belief_search),
                              ('episodes',_episode_action)):
            result=runner(public,observations,acquisition)
            current.append(dict(method=method,**score_submission(
                truth,public['initial'],public['target'],result,public['max_steps'])))
        forced=_forced_candidate_direct(public,acquisition)
        current.append(dict(method='forced-candidate-direct',**score_submission(
            truth,public['initial'],public['target'],forced,public['max_steps'])))
        acquisition_costs=acquisition.receipt()
        paid=observations[len(public['observations']):]
        for row in current:
            method=row['method'];program=row['program']
            if method=='dependencies':declared=row['model_status']=='inconsistent'
            elif method in ('conditioned-direct','candidate-set-primitive'):
                declared=inconsistent and row.get('compatible_laws')==0
            else:declared=False
            row.update(query_policy='misspecification-action',requested_queries=count,
                acquired_queries=len(used),query_indices=list(used),query_exhausted=query_exhausted,
                observation_record=deepcopy(observations),budget=budget,
                acquisition_costs=deepcopy(acquisition_costs),
                acquisition_operations=acquisition_costs['total_online'],
                candidate_compatible_laws=len(compatible),
                evidence_inconsistent_with_candidate_family=inconsistent,
                misspecification_detected=declared,
                abstained=program is None,
                abstained_on_inconsistency=declared and program is None,
                unsafe_action_attempted_after_inconsistency=(
                    inconsistent and program is not None and method!='known-law'),
                candidate_aware=method in candidate_aware,
                method_receives_evaluator_truth=method=='known-law',
                truth_in_candidate_family=False,candidate_family_supplied=True,
                target_changed_parts=[part for part,(before,after) in enumerate(
                    zip(public['initial'],public['target'])) if before!=after],
                target_action_parts=[observation['query']['program'][0]%len(public['initial'])
                                     for observation in paid],
                comparison_role=('descriptive candidate-family misspecification detection/abstention screen; '
                                 'forced-candidate-direct is a labeled unsafe negative control'),
                selector_contract=('two outcome-independent target-part action probes are chosen from the '
                                   'supplied family and public target; evaluator truth is used only to return outcomes'))
            rows.append(row)
    return rows


def evaluate_physical_misspecified(case,budget=32768,query_counts=(1,2)):
    """Measure misspecification detection with charged physical preparation."""
    public=case['public'];truth=case['private']['true_world'];rows=[]
    if truth in public['models'] or not case['truth_excluded']:
        raise ValueError('misspecification condition requires truth outside the supplied family')
    candidate_aware={'dependencies','conditioned-direct','candidate-set-primitive'}
    for count in query_counts:
        observations,used,setups,setup_records,reachable,query_exhausted,acquisition=(
            acquire_physical_misspecified(public,truth,count,budget))
        compatible=g2.compatible(public['models'],observations)
        inconsistent=not compatible
        current=[]
        for method in ('dependencies','known-law'):
            current.append(dict(method=method,**_structured_cached_action(
                public,truth,observations,acquisition,method)))
        for method,runner in (('conditioned-direct',_conditioned_direct),
                              ('candidate-set-primitive',_belief_search),
                              ('episodes',_episode_action)):
            result=runner(public,observations,acquisition)
            current.append(dict(method=method,**score_submission(
                truth,public['initial'],public['target'],result,public['max_steps'])))
        forced=_forced_candidate_direct(public,acquisition)
        current.append(dict(method='forced-candidate-direct',**score_submission(
            truth,public['initial'],public['target'],forced,public['max_steps'])))
        acquisition_costs=acquisition.receipt()
        for row in current:
            method=row['method'];program=row['program']
            if method=='dependencies':declared=row['model_status']=='inconsistent'
            elif method in ('conditioned-direct','candidate-set-primitive'):
                declared=inconsistent and row.get('compatible_laws')==0
            else:declared=False
            row.update(query_policy='misspecification-action-physical',requested_queries=count,
                acquired_queries=sum(record['query_executed'] for record in setup_records),
                physical_query_attempts=len(used),query_indices=list(used),
                query_exhausted=query_exhausted,observation_record=deepcopy(observations),
                budget=budget,acquisition_costs=deepcopy(acquisition_costs),
                acquisition_operations=acquisition_costs['total_online'],
                physical_setup_paths=deepcopy(setups),physical_setup_records=deepcopy(setup_records),
                physical_setup_operations=sum(map(len,setups)),reachable_setup_states=list(reachable),
                physical_setup_failures=sum(not record['reached_query_state'] for record in setup_records),
                candidate_compatible_laws=len(compatible),
                evidence_inconsistent_with_candidate_family=inconsistent,
                misspecification_detected=declared,abstained=program is None,
                abstained_on_inconsistency=declared and program is None,
                unsafe_action_attempted_after_inconsistency=(
                    inconsistent and program is not None and method!='known-law'),
                candidate_aware=method in candidate_aware,
                method_receives_evaluator_truth=method=='known-law',
                truth_in_candidate_family=False,candidate_family_supplied=True,
                target_changed_parts=[part for part,(before,after) in enumerate(
                    zip(public['initial'],public['target'])) if before!=after],
                target_action_parts=[public['menu'][index]['program'][0]%len(public['initial'])
                                     for index in used],
                setup_uses_evaluator_truth=False,
                comparison_role=('exposed descriptive charged physical candidate-family '
                                 'misspecification diagnostic; forced-candidate-direct is unsafe control'),
                selector_contract=('physical setup and falsification-query selection use the supplied '
                                   'family, public target and prior public outcomes, never evaluator truth'),
                reset_contract=('each attempted query receives a fresh object at the public task initial '
                                'state; provisioning that object is outside the count'))
            rows.append(row)
    return rows
