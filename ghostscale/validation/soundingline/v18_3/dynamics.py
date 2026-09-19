"""Unknown changes: fast-state transition, forgetting, and strong direct rivals."""
import json
import numpy as np
from . import world as W
from ..v16.learning import learn, encoding_cost
from ..v16.world import execute

CONDITIONS=('stationary','goal','belief','skill','tool','opportunity','gradual','return')
METHODS=('static','window4','discounted','selective-transition','all-transition','known-change')


def unit(index,cell=0,condition='goal',length=32,split='test'):
    r=W.rng('dynamics',split,index,condition)
    w=W.make_world(cell,index);initial=int(r.integers(len(W.STATES)))
    change=int(r.integers(10,19));second=change+7
    states=[];worlds=[];history=[];targets=[]
    for t in range(length):
        state=list(W.STATES[initial]);wt=dict(w)
        changed=t>=change and condition!='stationary'
        if condition=='return' and t>=second:changed=False
        if condition=='gradual':changed=r.random()<np.clip((t-change)/8,0,1)
        if changed:
            if condition in ('goal','return','gradual'):state[1]^=1
            elif condition=='belief':state[2]^=1
            elif condition=='skill':state[0]=(state[0]+1)%3
            elif condition=='opportunity':wt['endogenous']=not wt['endogenous']
        c=W.context(budget=1 if condition=='tool' and changed else 2)
        if condition=='tool' and changed:c['offered']=[x for x in range(4) if x!=w['groups'][1][0]]
        # Fixed sentinel cadence is public and independent of the unannounced switch.
        if t%4==1:c['signal']=0
        elif t%4==3:c['goal']=1
        s=W.STATES.index(tuple(state))
        observation=W.observe(wt,s,c,r,f'episode-{t}')
        history.append(dict(history[-1]) if w['shared'] and t%3 else observation)
        target_context=dict(W.FUTURES[t%4],offered=c['offered'])
        if condition=='tool' and changed:target_context['budget']=1
        targets.append(W.observe(wt,s,target_context,r,f'target-{t}'))
        states.append(s);worlds.append(wt)
    rows=[];prior=np.ones(len(W.STATES))/len(W.STATES)
    for method in METHODS:
        weights=prior.copy();trace=[];seen=[];sources=set()
        for t,obs in enumerate(history):
            reset_probability=0.
            if method in ('selective-transition','all-transition'):
                kind='fast' if method=='selective-transition' else 'all'
                stay=.88*weights
                reset=.12*(weights@W.transition(kind,1.))
                weights=stay+reset
                lik=W.likelihood(w,obs)
                reset_probability=float((reset@lik)/(weights@lik)) if obs['source'] not in sources else 0.
            elif method=='known-change' and (t==change or condition=='return' and t==second):
                axis={'goal':'goal','return':'goal','gradual':'goal','belief':'belief','skill':'skill'}.get(condition)
                if axis:weights=weights@W.transition(axis,1.)
            # Score before revealing the current episode. Tool access is public.
            prediction=weights@W.matrix(w,targets[t]['context'])
            actual=W.PROGRAMS.index(tuple(targets[t]['program']))
            true=W.matrix(worlds[t],targets[t]['context'])[states[t]]
            ll=W.cross_entropy(true,prediction)
            seen.append(obs)
            if obs['source'] in sources:pass
            elif method=='window4':weights=W.posterior(W.packet(w,seen[-4:]))
            elif method=='discounted':
                weights=weights**.8;weights/=weights.sum()
                weights*=W.likelihood(w,obs);weights/=weights.sum()
            else:
                weights*=W.likelihood(w,obs);weights/=weights.sum()
            sources.add(obs['source'])
            skill_mass=np.array([sum(weights[j] for j,s in enumerate(W.STATES) if s[0]==k) for k in range(3)])
            trace.append(dict(step=t,expected_loss=W.loss_record(ll),observed_loss=W.loss_record(-np.log(prediction[actual])),
                              change_probability=reset_probability,posterior=weights.tolist(),
                              true_skill_mass=float(skill_mass[W.STATES[states[t]][0]])))
        # Native learner has identical evidence under every inference method.
        acq=learn([h['program'] for h in history],[h['artifact'] for h in history],capacity=2)
        target=sum(1<<x for x in (w['groups'][index%2]+w['groups'][1-index%2][:1]))
        options=[p for p in W.PROGRAMS if encoding_cost(p,acq.library)<=2]
        chosen=min(options,key=lambda p:((execute(p).artifact^target).bit_count(),len(p),p))
        false_resets=sum(x['change_probability']>.5 for x in trace[:change])
        detections=[x['step']-change for x in trace[change:] if x['change_probability']>.5]
        rows.append(dict(method=method,trace=trace,false_resets=false_resets,
                         detection_delay=min(detections) if detections else None,
                         shared_enactment=dict(target=target,program=list(chosen),success=execute(chosen).artifact==target,
                           library=[list(p) for p in acq.library],practice_actions=acq.processing_cost,
                           definition_actions=acq.definition_cost,planning_evaluations=len(options),
                           execution_actions=len(chosen)),
                         known_change_privileged=method=='known-change'))
    return dict(family='B',index=index,cell=cell,condition=condition,public=json.loads(W.packet(w,history)),
                evaluator=dict(states=states,worlds=worlds,change=change,return_at=second,targets=targets),rows=rows)
