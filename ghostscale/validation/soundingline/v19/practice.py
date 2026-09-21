"""Persistent tabular production learning with an exact ordered-replay control."""
from itertools import product
import gzip,time
import numpy as np
from ..v18_3.io import canonical,write
from ..v18_3.world import rng
from . import local_world as W,readout_model as M

ARTIFACTS=tuple(product(range(2),repeat=3))
INDEX={a:i for i,a in enumerate(ARTIFACTS)}
ACTOR=(0,1,0,0)
INITIAL=(INDEX[(0,1,0)],INDEX[(1,0,1)])
GOALS=W.GOALS
SHAPE=(3,2,8,8,3,8)
SUCCESS=np.array([float(a[0]==a[1]==1) for a in ARTIFACTS])


def native_kernel(world):
    """Conditional do(goal) distribution, retaining the original operation law."""
    p=np.zeros(SHAPE)
    for step,context,a,previous in product(range(3),range(2),range(8),range(8)):
        ctx=W.CONTEXTS[context*2]
        options=list(W.choices(world,ACTOR,ARTIFACTS[a],step,ctx))
        for goal in range(3):
            rows=[(op,weight) for g,op,weight in options if g==GOALS[goal]]
            mass=sum(weight for op,weight in rows)
            for op,weight in rows:
                after=INDEX[W.execute(ARTIFACTS[a],ARTIFACTS[previous],op,ACTOR)]
                p[step,context,a,previous,goal,after]+=weight/mass
    return p


def policy(counts):
    p=counts/counts.sum(-1,keepdims=True);v=np.zeros((4,2,8,8));v[3]=SUCCESS[None,:,None]
    pi=np.zeros(SHAPE[:-2],int)
    for step in (2,1,0):
        # Next undo buffer is the current artifact, regardless of operation.
        future=v[step+1].transpose(0,2,1)[:, :,None,None,:]
        q=(p[step]*future).sum(-1);pi[step]=q.argmax(-1);v[step]=q.max(-1)
    return pi,v


def evaluate(kernel,pi):
    v=np.zeros((4,2,8,8));v[3]=SUCCESS[None,:,None]
    occupancy=np.zeros(SHAPE[:-1]);success=[]
    for step in (2,1,0):
        selected=np.take_along_axis(kernel[step],pi[step][...,None,None],axis=3).squeeze(3)
        v[step]=(selected*v[step+1].transpose(0,2,1)[:,:,None,:]).sum(-1)
    for context,initial in enumerate(INITIAL):
        success.append(v[0,context,initial,initial]);mass=np.zeros((8,8));mass[initial,initial]=1
        for step in range(3):
            following=np.zeros((8,8))
            for a,previous in product(range(8),repeat=2):
                goal=pi[step,context,a,previous]
                occupancy[step,context,a,previous,goal]+=mass[a,previous]/2
                following[:,a]+=mass[a,previous]*kernel[step,context,a,previous,goal]
            mass=following
    return np.asarray(success),v,occupancy


def increment(counts,row):
    step,context,before,previous,goal,after=map(int,row)
    counts[step,context,before,previous,goal,after]+=1


def controls():
    world=W.law(-9611);kernel=native_kernel(world)
    pi,_=policy(kernel);oracle,_,_=evaluate(kernel,pi)
    prior=np.full(SHAPE,.5);a=prior.copy();b=prior.copy();row=(0,0,2,2,0,5)
    increment(a,row);increment(b,row)
    no_action=np.broadcast_to(np.eye(8)[None,None,:,None,None,:],SHAPE).copy()
    p,_=policy(no_action);success,_,_=evaluate(no_action,p)
    # Fully controllable known law sends every action directly to successful artifact.
    known=np.zeros(SHAPE);known[...,INDEX[(1,1,0)]]=1
    kpi,_=policy(known);positive,_,_=evaluate(known,kpi)
    return dict(live_native_mass=bool(np.allclose(kernel.sum(-1),1)),
        positive_controlled_success=bool(np.allclose(positive,1)),
        placebo_action_inert=bool(np.allclose(success,SUCCESS[list(INITIAL)])),
        exact_replay_update=bool(np.array_equal(a,b)),
        no_update_identity=bool(np.array_equal(policy(prior)[0],policy(prior.copy())[0])),
        native_useful_control=bool(np.all(oracle>SUCCESS[list(INITIAL)])))


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();assert all(checks.values());rows=[];timing=[]
    for lineage in cfg['lineages']:
        world=W.law(lineage);kernel=native_kernel(world);oracle_pi,_=policy(kernel)
        oracle,oracle_values,oracle_occupancy=evaluate(kernel,oracle_pi)
        prior=np.full(SHAPE,.5);base_pi,_=policy(prior);baseline,_,_=evaluate(kernel,base_pi)
        # A uniform stochastic policy has the same transition law as goal averaging.
        uniform=np.repeat(kernel.mean(-2,keepdims=True),3,axis=-2);uniform_success,_,_=evaluate(uniform,base_pi)
        M.save_arrays(root/'evaluator'/f'law-{lineage}.npz',kernel=kernel,oracle_policy=oracle_pi,oracle_values=oracle_values,oracle_occupancy=oracle_occupancy,oracle_success=oracle,untrained_success=baseline,uniform_success=uniform_success)
        for draw in cfg['training_draws']:
            for seed in cfg['policy_seeds']:
                for condition in ('matched-start','restricted-start'):
                    pulse(phase='persistent-practice',lineage=lineage,draw=draw,seed=seed,condition=condition);start=time.process_time()
                    random=rng('v19-E1-practice-policy',lineage,draw,seed)
                    operation=rng('v19-E1-practice-operation',lineage,draw,seed)
                    tables={k:prior.copy() for k in ('active','replay','demonstration')};logs={k:[] for k in tables}
                    for episode in range(1,max(cfg['budgets'])+1):
                        context=(episode-1)%2 if condition=='matched-start' else 0
                        states={k:(INITIAL[context],INITIAL[context]) for k in ('active','demonstration')}
                        for step in range(3):
                            explore,choice,demo_coin=random.random(3);coin=operation.random()
                            for arm in ('active','demonstration'):
                                a,previous=states[arm]
                                if arm=='active':
                                    pi,_=policy(tables[arm]);goal=int(choice*3) if explore<cfg['epsilon'] else int(pi[step,context,a,previous])
                                else:
                                    options=list(W.choices(world,ACTOR,ARTIFACTS[a],step,W.CONTEXTS[context*2]))
                                    weights=np.array([sum(w for g,op,w in options if g==name) for name in GOALS])
                                    goal=min(2,int(np.searchsorted(weights.cumsum(),demo_coin,side='right')))
                                after=min(7,int(np.searchsorted(kernel[step,context,a,previous,goal].cumsum(),coin,side='right')))
                                row=[step,context,a,previous,goal,after];logs[arm].append(row);increment(tables[arm],row);states[arm]=(after,a)
                                if arm=='active':
                                    increment(tables['replay'],row);logs['replay'].append(row.copy())
                                    if not np.array_equal(tables['active'],tables['replay']):raise ValueError('exact replay update mismatch')
                        if episode in cfg['budgets']:
                            for arm,counts in tables.items():
                                pi,values=policy(counts);success,truth_values,occupancy=evaluate(kernel,pi)
                                visited=(counts.sum(-1)>4);coverage=float((oracle_occupancy*visited).sum()/3)
                                stem=f'{lineage}-{draw}-{seed}-{condition}-{episode}-{arm}'
                                M.save_arrays(root/'models'/f'{stem}.npz',counts=counts,policy=pi,learned_values=values)
                                M.save_arrays(root/'evaluator'/f'{stem}.npz',exact_values=truth_values,policy_occupancy=occupancy)
                                rows.append(dict(lineage=lineage,draw=draw,policy_seed=seed,condition=condition,episodes=episode,feedback=3*episode,arm=arm,success=float(success.mean()),context_success=success.tolist(),oracle_success=float(oracle.mean()),untrained_success=float(baseline.mean()),uniform_success=float(uniform_success.mean()),visited_state_goals=int(visited.sum()),oracle_occupancy_coverage=coverage))
                    for arm,log in logs.items():
                        M.save_arrays(root/'observed'/f'{lineage}-{draw}-{seed}-{condition}-{arm}.npz',transitions=np.asarray(log,int))
                    timing.append(dict(lineage=lineage,draw=draw,seed=seed,condition=condition,cpu_seconds=time.process_time()-start))
    (root/'raw').mkdir(exist_ok=True);(root/'raw/practice_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'TIMING.jsonl',dict(measurements=timing,note='within native accounting; do not add twice'))
    write(root/'INPUT_SCHEMA.json',dict(reader='observed transition arrays only; anonymous content-derived names at export; initial context, current and undo artifact, chosen goal, next artifact',evaluator='lineage/seed map, kernel, model arrays, oracle/reference scores and scientific summaries',teacher='observed after-artifact; no hidden purpose or local-goal teacher',state_columns=['step','context','artifact','undo_buffer','chosen_goal','next_artifact']))
    cells=[]
    for condition,episode,arm in product(('matched-start','restricted-start'),cfg['budgets'],('active','replay','demonstration')):
        rr=[r for r in rows if (r['condition'],r['episodes'],r['arm'])==(condition,episode,arm)]
        cells.append(dict(condition=condition,episodes=episode,arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in ('success','oracle_success','untrained_success','uniform_success','visited_state_goals','oracle_occupancy_coverage')}))
    return dict(controls=checks,cells=cells,rows=len(rows),tiny_settings=0,scope='persistent tabular do(local-goal) production learning in the admitted native transition world; no inverse-intent or neural capability claim',warrant='exploratory constructed-method comparison; miniature — architecture untested')
