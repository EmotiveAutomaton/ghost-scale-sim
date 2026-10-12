"""D06 finite, supplied-law ruler for contrasting opportunities."""
import time
from pathlib import Path
import numpy as np
from . import world as W
from .routes import retain
from ..v18_3.io import write

POLICIES=('random','diversity','target-information')
READERS=('population-prior','personal-posterior','matched-direct','context-only')
STATES=np.array([(g,s) for g in (0,1) for s in (0,1)])
CONTEXTS=np.array([[-1,-3.],[-1,-1.],[-1,1.],[-1,3.],[0,-2.],[0,2.],[1,-2.],[1,2.]])

def law(world):
    rng=np.random.default_rng(W.seed('v21-d06-law',world))
    return dict(preference=float(rng.uniform(1.5,3.5)),accuracy=[float(rng.uniform(.55,.7)),float(rng.uniform(.85,.95))])

def channel(contexts,known):
    c=np.asarray(contexts);choice=1/(1+np.exp(-(known['preference']*(2*STATES[:,0]-1)[None,:]-c[:,1,None])))
    choice=np.where(c[:,0,None]>=0,c[:,0,None],choice);acc=np.array(known['accuracy'])[STATES[:,1]]
    return choice*acc+(1-choice)*(1-acc)

def posterior(likelihood,ids,outcomes):
    p=np.full(4,.25)
    for i,y in zip(ids,outcomes):p*=likelihood[i] if y else 1-likelihood[i];p/=p.sum()
    return p

def information(p,likelihood):
    result=np.zeros(len(likelihood))
    for y in (0,1):
        joint=p[None,:]*(likelihood if y else 1-likelihood);prob=joint.sum(1)
        result+=np.sum(joint*np.log(joint/(p[None,:]*prob[:,None])),axis=1)
    return result

def select(policy,p,likelihood,contexts,chosen,order):
    remaining=[i for i in range(len(contexts)) if i not in chosen]
    if policy=='random':return next(int(i) for i in order if i not in chosen)
    if policy=='target-information':score=information(p,likelihood)
    elif policy=='diversity':
        # Context-only distance: categorical permission plus scaled cost.
        features=np.column_stack([np.asarray(contexts)[:,0,None]==np.array([-1,0,1]),np.asarray(contexts)[:,1]/3])
        score=np.ones(len(contexts)) if not chosen else ((features[:,None]-features[chosen][None,:])**2).sum(2).min(1)
    else:raise ValueError(policy)
    best=max(score[i] for i in remaining)
    return min(i for i in remaining if score[i]>=best-1e-12)

def controls():
    from .opportunities_check import reference,information as scalar_information
    known=law(71000830);lik=channel(CONTEXTS,known);ids=[0,4,7];ys=[1,0,1];q=posterior(lik,ids,ys)
    forced=channel([[0,0],[1,0]],known)
    return dict(live_independent_reference=bool(np.max(abs(q-reference(CONTEXTS[ids],ys,known)))<1e-12),
        placebo_forced_hides_preference=bool(np.array_equal(forced[:,0],forced[:,2]) and np.array_equal(forced[:,1],forced[:,3])),
        positive_forced_reveals_skill=bool(forced[1,1]>forced[1,0]),
        positive_free_separates_preference=bool(lik[1,2]>lik[1,0]),
        live_information_reference=bool(max(abs(information(np.full(4,.25),lik)[i]-scalar_information([.25]*4,c,known)) for i,c in enumerate(CONTEXTS))<1e-12))

def run(root,d,pulse):
    from .opportunities_check import reference,information as scalar_information
    rows=[];timings=[];maxerr=0.;decisions=0
    for world in d['lineages']:
        start=time.process_time();known=law(world);lik=channel(CONTEXTS,known)
        rng=np.random.default_rng(W.seed('v21-d06-cases',world));states=rng.integers(4,size=d['test_cases']);uniforms=rng.random((d['test_cases'],8))
        responses=(uniforms<lik[:,states].T).astype(int);orders=np.array([rng.permutation(8) for _ in states])
        # New costs are continuous and outside the training range, with changed permissions.
        fresh=np.column_stack([np.array([-1,0,1,-1,0,1,-1,-1]),rng.choice([-1,1],8)*rng.uniform(3.1,5.,8)])
        held=channel(fresh,known);future_u=rng.random((d['test_cases'],8));future=(future_u<held[:,states].T).astype(int)
        identity=f"case-{W.seed('v21-d06-blind-id',world):016x}";(root/'evaluator').mkdir(parents=True,exist_ok=True);(root/'reader').mkdir(exist_ok=True)
        np.savez_compressed(root/'evaluator'/f'{identity}_points.npz',world=world,state=states,uniforms=uniforms,future_uniforms=future_u,future=future)
        for policy in POLICIES:
            schedules=[];posteriors={k:[] for k in d['context_counts']};directs={k:[] for k in d['context_counts']}
            for case,state in enumerate(states):
                chosen=[];ys=[];p=np.full(4,.25)
                for step in range(8):
                    i=select(policy,p,lik,CONTEXTS,chosen,orders[case]);remaining=[j for j in range(8) if j not in chosen]
                    if policy=='target-information':
                        scores={j:scalar_information(p,CONTEXTS[j],known) for j in remaining};best=max(scores.values())
                        ref=min(j for j in remaining if scores[j]>=best-1e-12);assert ref==i;decisions+=1
                    chosen.append(i);ys.append(int(responses[case,i]));p=posterior(lik,chosen,ys)
                    if step+1 in posteriors:
                        q=np.array(reference(CONTEXTS[chosen],ys,known));maxerr=max(maxerr,float(np.max(abs(p-q))));assert np.max(abs(p-q))<1e-12
                        posteriors[step+1].append(p);directs[step+1].append(q)
                schedules.append(chosen)
            schedules=np.array(schedules)
            for k in d['context_counts']:
                # Readers receive only selected evidence, common context law and held-out contexts.
                chosen=schedules[:,:k];observed=np.take_along_axis(responses,chosen,axis=1)
                np.savez_compressed(root/'reader'/f'{identity}-{policy}-{k}_points.npz',contexts=CONTEXTS,selected=chosen,observed=observed,fresh_contexts=fresh,preference=known['preference'],accuracy=known['accuracy'])
                for reader in READERS:
                    p=np.array(posteriors[k]) if reader=='personal-posterior' else np.array(directs[k]) if reader=='matched-direct' else np.full((len(states),4),.25)
                    # Both restricted baselines receive identical inputs but ignore personal history.
                    marginal_goal=np.column_stack([p[:,:2].sum(1),p[:,2:].sum(1)]);marginal_skill=np.column_stack([p[:,::2].sum(1),p[:,1::2].sum(1)])
                    f=p@held.T;fp=np.stack([1-f,f],axis=-1).reshape(-1,2)
                    expected=-(held[:,states].T*np.log(f)+(1-held[:,states].T)*np.log(1-f)).mean(1)
                    metrics={}
                    for target,probs,truth in [('joint',p,states),('preference',marginal_goal,states//2),('skill',marginal_skill,states%2),('future',fp,future.reshape(-1))]:
                        name=f'{world}-{policy}-{k}-{reader}-{target}';extra=dict(expected_future_log_loss=expected) if target=='joint' else {}
                        metrics[target]=retain(root,name,probs,truth,**extra)
                    rows.append(dict(lineage=world,policy=policy,contexts=k,reader=reader,metrics=metrics,expected_future_log_loss=float(expected.mean())))
                pulse(phase='opportunity-consumer',lineage=world,policy=policy,contexts=k)
        timings.append(dict(lineage=world,cpu_seconds=time.process_time()-start))
    return dict(card='G21-D06',population=rows,timings=timings,controls=controls(),independent_max_probability_error=maxerr,independent_information_decisions=decisions,
        limitations=['Initial supplied-law binary preference/skill ruler; no fitted model, uncertain law or original-world headroom claim.',
        'Only final noisy binary output is observed; intended choice is hidden. Permission and cost are supplied truth.',
        'Eight fixed training opportunities, one outcome each; contexts are selected without replacement. Counts are nested and all policies have identical potential responses.',
        'Fresh context costs lie outside the training range and permissions vary, but the supplied response law is unchanged.',
        'Population-prior and context-only receive the same observations but discard personal history; they are restricted baselines. Matched direct sees and uses all the same evidence.',
        'Target information selects using the exact supplied joint preference/skill law, not a learned acquisition model. Diversity uses only context features.',
        'Eight coefficient worlds are the uncertainty units, not cases, contexts or queried counts. No human stable-value inference or fresh confirmation.',
        'Miniature — architecture untested.'])
