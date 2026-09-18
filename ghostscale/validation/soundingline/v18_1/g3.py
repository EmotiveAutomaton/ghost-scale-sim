"""Bounded structural breadth and representation comparisons at matched work."""
from copy import deepcopy
from itertools import product,combinations
import json
import random

from ..v16.records import canonical,digest,seed_for
from .common import (identity,step,execute,actions,learn_fragments,learn_episodes,
                     solve,score_submission)

SCHEMA='v18.1.g3.1'
METHODS=('primitive','episodes','fragments','fragments-and-exceptions')


def topology(n,family,rng):
    if family=='chain':return [-1]+list(range(n-1))
    if family=='fork':
        supports=max(1,(n-1)//3)
        return [-1]+[0]*supports+[rng.randrange(supports+1) for _ in range(n-supports-1)]
    if family=='groups':return [-1,-1]+[rng.randrange(2) for _ in range(n-2)]
    raise ValueError('unknown topology family')


def topology_signature(parents):
    def subtree(i):return '('+''.join(sorted(subtree(j) for j,p in enumerate(parents) if p==i))+')'
    return ''.join(sorted(subtree(i) for i,p in enumerate(parents) if p<0))


def revision_program(world,initial,target):
    """Evaluator/teacher's constructive witness. Never supplied as reader planning truth."""
    n=len(world['parents']);current=tuple(initial);program=[]
    def perform(action):
        nonlocal current
        after,_,legal=step(world,current,False,action)
        if not legal:raise ValueError('invalid teacher construction')
        current=after;program.append(action)
    for part in range(n):
        if current[part]<0:perform(part)
        if current[part]==target[part]:continue
        descendants=[]
        for child in range(part+1,n):
            parent=world['parents'][child]
            while parent>=0 and parent!=part:parent=world['parents'][parent]
            if parent==part:descendants.append(child)
        for child in reversed(descendants):
            if current[child]>=0:perform(n+child)
        perform(2*n+part)
        for child in descendants:
            if current[child]<0:perform(child)
            if current[child]!=target[child]:perform(2*n+child)
    program.append(3*n)
    result=execute(world,initial,program,4*n+2)
    if not result['legal'] or not result['stopped'] or identity(result['state'])!=identity(target):
        raise ValueError('teacher failed to realize target')
    return program


def make_cases(namespace,*,per_stratum=64,histories=4,sizes=(3,5,7),families=('fork','chain','groups')):
    cases=[]
    for n in sizes:
        for family in families:
            for condition in ('familiar','new-combination','changed-dependency','missing-observation'):
                seen=set();draw=0
                native_support=None
                if n==3:
                    parent_support=([[-1,0,0]] if family=='fork' else
                                    [[-1,0,1]] if family=='chain' else [[-1,-1,0],[-1,-1,1]])
                    native_support=[(p,list(d),list(v)) for p in parent_support for d in product((0,1),repeat=3)
                                    for v in combinations(range(3),2 if condition=='new-combination' else 1)]
                while len(seen)<per_stratum:
                    if native_support is not None and draw>=len(native_support):break
                    # Sampling rejection removes duplicate law/context units only, never outcomes or donors.
                    rng=random.Random(seed_for(namespace,n,family,condition,draw,'world'));draw+=1
                    if native_support is not None:
                        parents,defaults,pivots=native_support[draw-1]
                    else:
                        parents=topology(n,family,rng);defaults=[rng.randrange(2) for _ in range(n)]
                        # Sixteen offers cover four pivots; a familiar task must
                        # be among those intended demonstrations (failures stay).
                        pivots=rng.sample(range(min(n,4) if condition=='familiar' else n),
                                          2 if condition=='new-combination' else 1)
                    truth=dict(kind='assembly',parents=parents,defaults=defaults,forbidden=[])
                    initial=defaults.copy();target=defaults.copy()
                    for pivot in pivots:target[pivot]=1-target[pivot]
                    donor=deepcopy(truth)
                    if condition=='changed-dependency':
                        donor['parents']=([-1]+list(range(n-1)) if parents!=[-1]+list(range(n-1))
                                          else [-1]+[0]*(n-1))
                    unit=digest([truth,initial,target,donor,condition])
                    if unit in seen:
                        if draw>per_stratum*100:
                            raise ValueError('insufficient distinct larger contexts; do not silently claim exhaustion')
                        continue
                    seen.add(unit)
                    witness=revision_program(truth,initial,target)
                    for history in range(histories):
                        hrng=random.Random(seed_for(namespace,unit,history,'training'))
                        training=[]
                        for trial in range(16):
                            pivot=(trial//4)%n
                            intended=defaults.copy();intended[pivot]=1-intended[pivot]
                            proposal=revision_program(donor,initial,intended)
                            performed=proposal.copy()
                            if hrng.random()<0.125:
                                performed[hrng.randrange(len(performed))]=hrng.choice(actions(donor))
                            actual=execute(donor,initial,performed,4*n+2)
                            feedback=actual['legal'] and actual['stopped'] and identity(actual['state'])==identity(intended)
                            visible=performed[:actual['primitive_cost']]
                            complete=True
                            if condition=='missing-observation' and trial%2:
                                visible=visible[:max(1,len(visible)//2)];complete=False
                            training.append(dict(initial=initial,program=visible,complete=complete,
                                target=list(actual['state']),feedback=feedback,observed_actions=len(visible),
                                source_primitive_cost=actual['primitive_cost'],trial=trial))
                        order=actions(truth);hrng.shuffle(order)
                        public=dict(schema=SCHEMA,world=truth,initial=initial,target=target,training=training,
                                    max_steps=4*n+2,action_order=order)
                        cases.append(dict(case_id=digest([namespace,unit,history]),structural_unit=unit,
                            history=history,n=n,family=family,condition=condition,
                            topology_signature=topology_signature(parents),source_topology_signature=topology_signature(donor['parents']),
                            public=json.loads(canonical(public)),private=dict(true_world=deepcopy(truth),donor_world=deepcopy(donor),target_witness=witness),
                            knowledge='true current law supplied to every planner; representation comparison, not successful law inference',
                            exposures=sum(t['target']==target for t in training)))
    return cases


def representation(training,method,cap,n):
    observed=[dict(t,program=[a for a in t['program'] if a!=3*n]) for t in training]
    if method=='primitive':return dict(fragments=[],episodes=[],storage=0,scanned=0)
    if method=='fragments':return learn_fragments(observed,capacity=8,threshold=2,storage_cap=cap)
    if method=='episodes':return learn_episodes(training,storage_cap=cap)
    if method=='fragments-and-exceptions':
        rep=learn_fragments(observed,capacity=8,threshold=2,storage_cap=cap//2)
        episodes=learn_episodes(training,storage_cap=cap-rep['storage'],exceptions=rep['fragments'])
        return dict(fragments=rep['fragments'],episodes=episodes['episodes'],storage=rep['storage']+episodes['storage'],
                    scanned=rep['scanned']+episodes['scanned'])
    raise ValueError('unknown memory representation')


def contract(payload):
    public=json.loads(payload)
    if set(public)!={'schema','world','initial','target','training','max_steps','action_order'} or public['schema']!=SCHEMA:
        raise ValueError('G3 public schema violation')
    expected_world=({'kind','parents','defaults','forbidden'} if public['world']['kind']=='assembly'
                    else {'kind','cells','forbidden'})
    if set(public['world'])!=expected_world or any(set(t)!={'initial','program','complete','target',
        'feedback','observed_actions','source_primitive_cost','trial'} for t in public['training']):
        raise ValueError('G3 nested public schema violation')
    return public


def predict(payload,method,budget,storage):
    public=contract(payload)
    n=len(public['world']['parents']) if public['world']['kind']=='assembly' else public['world']['cells']
    rep=representation(public['training'],method,storage,n)
    result=solve(public['world'],public['initial'],public['target'],rep,budget,public['max_steps'],public['action_order'])
    return dict(result,representation=rep,acquisition=dict(
        processed_trials=len(public['training']),source_execution_primitives=sum(t['source_primitive_cost'] for t in public['training']),
        observed_actions=sum(t['observed_actions'] for t in public['training']),learning_scan_actions=rep['scanned'],
        storage_tokens=rep['storage'],storage_envelope=storage,
        failed_donor_trials=sum(not t['feedback'] for t in public['training']),
        censored_trials=sum(not t['complete'] for t in public['training'])))


def evaluate(case,budgets=(512,2048,8192),storage_caps=(32,128)):
    rows=[];public=case['public']
    for storage in storage_caps:
        for method in METHODS:
            for budget in budgets:
                result=predict(canonical(public),method,budget,storage)
                scored=score_submission(case['private']['true_world'],public['initial'],public['target'],result,public['max_steps'])
                rows.append(dict(method=method,budget=budget,storage_cap=storage,**scored))
    return rows


def graphic_cases(namespace,per_stratum=64,histories=4):
    cases=[]
    for condition in ('familiar','new-combination','changed-constraint','missing-observation'):
        for index in range(per_stratum):
            rng=random.Random(seed_for(namespace,condition,index,'graphic'))
            cells=rng.sample(range(16),8);motifs=[cells[i:i+2] for i in range(0,8,2)]
            target_cells=motifs[0] if condition=='familiar' else [motifs[0][0],motifs[1][1],motifs[2][0]]
            target=sum(1<<c for c in target_cells)
            initial=(1<<target_cells[0]) if condition=='changed-constraint' else 0
            world=dict(kind='graphic',cells=16,forbidden=[target_cells[0]] if condition=='changed-constraint' else [])
            unit=digest([world,initial,target,motifs,condition])
            witness=[c for c in target_cells if not initial&(1<<c)]
            assert execute(world,initial,witness,6)['state']==target
            for history in range(histories):
                hrng=random.Random(seed_for(namespace,unit,history,'training'));training=[]
                for trial in range(16):
                    program=motifs[trial//4].copy()
                    if hrng.random()<0.125:program[hrng.randrange(2)]=hrng.randrange(32)
                    donor=dict(kind='graphic',cells=16,forbidden=[])
                    actual=execute(donor,0,program,6)
                    desired=sum(1<<c for c in motifs[trial//4])
                    complete=not (condition=='missing-observation' and trial%2)
                    visible=program if complete else program[:1]
                    training.append(dict(initial=0,program=visible,complete=complete,target=actual['state'],
                        feedback=actual['legal'] and actual['state']==desired,observed_actions=len(visible),
                        source_primitive_cost=actual['primitive_cost'],trial=trial))
                order=actions(world);hrng.shuffle(order)
                cases.append(dict(case_id=digest([namespace,unit,history]),structural_unit=unit,history=history,
                    n=16,family='graphic',condition=condition,topology_signature='independent-16-cell-board',
                    source_topology_signature='independent-16-cell-board',
                    public=dict(schema=SCHEMA,world=world,initial=initial,target=target,training=training,max_steps=6,action_order=order),
                    private=dict(true_world=world,donor_world=donor,target_witness=witness),
                    knowledge='current transition law and constraints supplied',exposures=sum(t['target']==target for t in training)))
    return cases
