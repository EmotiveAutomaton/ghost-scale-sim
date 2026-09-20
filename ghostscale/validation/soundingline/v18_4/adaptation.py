"""Finite hazard mixtures: common-trace prequential prediction and native matching."""
import numpy as np
from ..v18_3 import world as W
from ..v16.world import execute

HAZARDS=(.01,.04,.12,.35,.8)
ROLES=('goal','belief','skill','fast','all')


def model_bank():
    kernels=[np.eye(len(W.STATES))];names=['static']
    for role in ROLES:
        for hazard in HAZARDS:
            kernels.append(W.transition(role,hazard));names.append(f'{role}:{hazard}')
    return names,np.array(kernels)


def filter_step(weights,kernels,likelihood=None):
    predicted=np.einsum('ms,mst->mt',weights,kernels)
    if likelihood is None:return predicted,np.ones(len(weights))
    evidence=predicted@likelihood
    if np.any(evidence<=0):raise ValueError('empty finite support')
    posterior=predicted*likelihood[None,:]/evidence[:,None]
    return posterior,evidence


def stream(index,cell,condition,length,copy_span):
    w=W.make_world(cell,18540000+index);r=W.rng('v18.4-adaptation',index,cell,condition,length)
    initial=list(W.STATES[int(r.integers(len(W.STATES)))]);change=length//3+int(r.integers(-2,3))
    history=[];states=[];worlds=[];queries=[];targets=[]
    for t in range(length):
        state=list(initial);wt=dict(w);changed=t>=change and condition!='stationary'
        if condition=='return' and t>=2*length//3:changed=False
        if condition=='gradual':changed=r.random()<np.clip((t-change)/12,0,1)
        if changed:
            if condition in ('goal','return','gradual','joint'):state[1]^=1
            if condition in ('belief','joint'):state[2]^=1
            if condition in ('skill','joint'):state[0]=(state[0]+1)%3
            if condition=='opportunity':wt['endogenous']=not wt['endogenous']
            if condition=='rule':wt['rule']='lexicographic'
        s=W.STATES.index(tuple(state));context=W.QUERIES[t%5]
        observation=W.observe(wt,s,context,r,f'root-{t}')
        copied=t>0 and copy_span>1 and t%copy_span!=0
        history.append(dict(history[-1]) if copied else observation)
        query=W.FUTURES[t%len(W.FUTURES)]
        queries.append(query);targets.append(W.observe(wt,s,query,r,f'target-{t}'))
        states.append(s);worlds.append(wt)
    return w,history,dict(states=states,worlds=worlds,change=change,queries=queries,targets=targets)


def unit(index,cell=0,condition='goal',length=32,copy_span=1,split='test'):
    w,history,truth=stream(index,cell,condition,length,copy_span)
    names,kernels=model_bank();n=len(W.STATES);weights=np.ones((len(names),n))/n
    mixture=np.ones(len(names))/(2*(len(names)-1));mixture[0]=.5
    # Fixed competing models have equal starting state priors and identical evidence.
    selected={name:names.index(name) for name in ('static','fast:0.12','all:0.12','goal:0.04','skill:0.04')}
    methods=list(selected)+['hazard-role-mixture','known-state-ceiling']
    traces={m:[] for m in methods};seen=set()
    artifacts=sorted(set(int(a) for a in W.ARTIFACTS));programs={a:min((p for p in W.PROGRAMS if execute(p).artifact==a),key=lambda p:(len(p),p)) for a in artifacts}
    for t,obs in enumerate(history):
        predicted,_=filter_step(weights,kernels)
        matrix=W.artifact_matrix(w,truth['queries'][t]);true=W.artifact_matrix(truth['worlds'][t],truth['queries'][t])[truth['states'][t]]
        forecasts={m:predicted[j]@matrix for m,j in selected.items()}
        forecasts['hazard-role-mixture']=(mixture@predicted)@matrix
        forecasts['known-state-ceiling']=true
        for method,p in forecasts.items():
            # A common deterministic numerical tie rule is fixed before outcomes.
            tied=[a for a in artifacts if p[a]>=max(p[artifacts])-1e-10]
            chosen=tied[(index+t)%len(tied)];execution=execute(programs[chosen])
            actual=truth['targets'][t]['artifact'];loss=W.cross_entropy(true,p)
            traces[method].append(dict(step=t,prediction=p.tolist(),expected_loss=float(loss),
                expected_match=float(true[chosen]),observed_match=execution.artifact==actual,
                constructed_artifact=execution.artifact,program=list(programs[chosen]),execution_actions=execution.primitive_cost,
                abstain_at_half=bool(p[chosen]<.5),abstention_loss=float(.5 if p[chosen]<.5 else 1-true[chosen])))
        if obs['source'] not in seen:
            weights,evidence=filter_step(weights,kernels,W.likelihood(w,obs))
            mixture*=evidence;mixture/=mixture.sum();seen.add(obs['source'])
        else:weights=predicted
    rows=[]
    for method,trace in traces.items():
        rows.append(dict(method=method,expected_loss=float(np.mean([x['expected_loss'] for x in trace])),
            pre_change_loss=float(np.mean([x['expected_loss'] for x in trace[:truth['change']]])),
            post_change_loss=float(np.mean([x['expected_loss'] for x in trace[truth['change']:]])),
            expected_match=float(np.mean([x['expected_match'] for x in trace])),
            abstention_loss=float(np.mean([x['abstention_loss'] for x in trace])),trace=trace))
    return dict(family='U',index=index,cell=cell,condition=condition,length=length,copy_span=copy_span,
        world=w,history=history,evaluator=truth,model_names=names,final_model_weights=mixture.tolist(),rows=rows,
        scope='common-trace finite forecasting and native artifact matching; no acquired-learning advantage; known-state comparator privileged')


def verify(unit):
    if abs(sum(unit['final_model_weights'])-1)>1e-12:raise ValueError('mixture not normalized')
    for row in unit['rows']:
        values=[]
        for item in row['trace']:
            p=np.array(item['prediction']);t=item['step'];truth=unit['evaluator']
            if np.any(p<0) or abs(p.sum()-1)>1e-10:raise ValueError('forecast invalid')
            true=W.artifact_matrix(truth['worlds'][t],truth['queries'][t])[truth['states'][t]]
            value=sum(-float(q)*__import__('math').log(float(v)) for q,v in zip(true,p) if q>0)
            if abs(value-item['expected_loss'])>1e-10:raise ValueError('scalar score differs')
            if execute(item['program']).artifact!=item['constructed_artifact']:raise ValueError('native match differs')
            values.append(value)
        if abs(sum(values)/len(values)-row['expected_loss'])>1e-10:raise ValueError('aggregate differs')
    return True
