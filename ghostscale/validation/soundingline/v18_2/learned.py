"""Small genuinely trained NumPy classifiers, split/flat with matched evidence.

No latent labels train the predictor. Targets are independently sampled future
artifacts. Held-out scores cannot change the model or select a compression axis.
"""
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
from ..v16.records import write,read,now
from . import model as m
from .verify import interval

CONTEXT=11
OBS=44
HISTORY=32*OBS
CURRENT=CONTEXT+12


def align_public(payload):
    """Invertible public role-coordinate transform; no evaluator state used."""
    p=m.parse(payload)
    if p['world']['family']!='board':raise ValueError('restricted-world alignment not admitted')
    physical=[a for group in p['world']['groups'] for a in group]
    inverse={a:i for i,a in enumerate(physical)}
    decode=[sum(1<<physical[i] for i in range(4) if artifact&(1<<i)) for artifact in range(16)]
    encode={a:i for i,a in enumerate(decode)}
    for obs in p['history']:
        obs['artifact']=encode[obs['artifact']]
        if obs['program'] is not None:obs['program']=[inverse[a] for a in obs['program']]
    p['world']['groups']=[[0,1],[2,3]]
    return m.canonical(p),decode


def represent(payload,aligned=False):
    if aligned:
        transformed,decode=align_public(payload)
        return features(transformed),decode
    return features(payload),list(range(16))


def predict(net,payload,aligned=False):
    x,decode=represent(payload,aligned);q=net.forward(x[None,:])[0][0]
    result=np.zeros(16);result[decode]=q
    return result


def context_features(c):
    x=np.zeros(CONTEXT)
    for offset,key in ((0,'goal'),(4,'signal'),(7,'reader_fact')):
        value=c[key];x[offset+((3 if key=='goal' else 2) if value is None else value)]=1
    x[10]=c['noticed'];return x


def features(payload):
    public=m.parse(payload);h=np.zeros(HISTORY);c=np.zeros(CURRENT)
    for index,obs in enumerate(public['history'][-32:]):
        x=np.zeros(OBS);x[:11]=context_features(obs['context']);x[11+obs['artifact']]=1
        if obs['program'] is not None:
            for j,a in enumerate(obs['program']):x[27+j*4+a]=1
        x[43]=1;h[index*OBS:(index+1)*OBS]=x
    c[:11]=context_features(public['current'])
    for i,group in enumerate(public['world']['groups']):
        for a in group:c[11+i*4+a]=1
    c[19]=public['world']['price'];c[20]=public['world']['max_code']/3
    c[21]=public['world']['family']=='restricted';c[22]=public['world']['temperature']
    return np.concatenate((h,c)).astype(np.float32)


class Network:
    def __init__(self,kind,seed=0,rank=24):
        self.kind=kind;self.rank=rank;rng=np.random.default_rng(seed)
        shapes=([(HISTORY,rank),(CURRENT,8),(rank+8,16)] if kind=='split' else [(HISTORY+CURRENT,rank),(rank,16)])
        self.parameters=[]
        for n,k in shapes:
            self.parameters.extend([rng.normal(0,1/math.sqrt(n),(n,k)).astype(np.float32),np.zeros(k,np.float32)])

    def forward(self,x):
        p=self.parameters
        if self.kind=='split':
            h=np.tanh(x[:,:HISTORY]@p[0]+p[1]);c=np.tanh(x[:,HISTORY:]@p[2]+p[3])
            z=np.concatenate((h,c),axis=1);logits=z@p[4]+p[5];cache=(x,h,c,z)
        else:
            h=np.tanh(x@p[0]+p[1]);logits=h@p[2]+p[3];cache=(x,h)
        exp=np.exp(logits-logits.max(axis=1,keepdims=True));return exp/exp.sum(axis=1,keepdims=True),cache

    def gradients(self,x,y):
        q,cache=self.forward(x);d=q.copy();d[np.arange(len(y)),y]-=1;d/=len(y)
        p=self.parameters
        if self.kind=='split':
            x,h,c,z=cache;dz=d@p[4].T
            dh=dz[:,:self.rank]*(1-h*h);dc=dz[:,self.rank:]*(1-c*c)
            gradients=[x[:,:HISTORY].T@dh,dh.sum(0),x[:,HISTORY:].T@dc,dc.sum(0),z.T@d,d.sum(0)]
        else:
            x,h=cache;dh=(d@p[2].T)*(1-h*h)
            gradients=[x.T@dh,dh.sum(0),h.T@d,d.sum(0)]
        return gradients

    def save(self,path):
        np.savez_compressed(path,**{f'p{i}':p for i,p in enumerate(self.parameters)})

    def swap_prediction(self,receiver,donor,part):
        if self.kind!='split':raise ValueError('flat coordinates have no assigned psychological role')
        _,a=self.forward(receiver);_,b=self.forward(donor)
        # These are architectural long/current blocks, not established skill/goal neurons.
        z=np.concatenate((b[1] if part=='history' else a[1],b[2] if part=='current' else a[2]),axis=1)
        logits=z@self.parameters[4]+self.parameters[5];q=np.exp(logits-logits.max(axis=1,keepdims=True))
        return q/q.sum(axis=1,keepdims=True)


def train_pair(x,y,dev_x,dev_y,seed,limited,heartbeat,epochs=80,cap=1800):
    models={k:Network(k,seed) for k in ('split','flat')}
    opt={k:([np.zeros_like(p) for p in n.parameters],[np.zeros_like(p) for p in n.parameters]) for k,n in models.items()}
    best={k:(float('inf'),None,0) for k in models};curve=[];step=0;start=time.monotonic()
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        if limited() or time.monotonic()-start>cap:break
        indices=rng.permutation(len(y))
        for first in range(0,len(y),128):
            batch=indices[first:first+128];step+=1
            for kind,net in models.items():
                gradients=net.gradients(x[batch],y[batch]);first_m,second=opt[kind]
                for p,g,a,b in zip(net.parameters,gradients,first_m,second):
                    a*=.9;a+=.1*g;b*=.999;b+=.001*g*g
                    p-=.002*(a/(1-.9**step))/(np.sqrt(b/(1-.999**step))+1e-8)
        record=dict(epoch=epoch+1,examples=len(y),seconds=time.monotonic()-start)
        for kind,net in models.items():
            q,_=net.forward(dev_x);loss=float(-np.log(np.maximum(q[np.arange(len(dev_y)),dev_y],1e-12)).mean())
            record[kind]=loss
            if loss<best[kind][0]:best[kind]=(loss,[p.copy() for p in net.parameters],epoch+1)
        curve.append(record);heartbeat(phase='fit',epoch=epoch+1,elapsed_seconds=time.monotonic()-start)
    for kind,net in models.items():
        if best[kind][1] is None:raise RuntimeError('no completed training epoch')
        net.parameters=best[kind][1]
    return models,dict(curve=curve,chosen={k:dict(dev_log_loss=v[0],epoch=v[2]) for k,v in best.items()},
                       epochs_completed=len(curve),matched_examples=True,matched_updates=step,
                       parameters={k:sum(p.size for p in n.parameters) for k,n in models.items()})


def build_data(namespace,split,count,limited,heartbeat,aligned=False,probe_mode='standard'):
    xs=[];ys=[];ids=[]
    for index in range(count):
        if limited():raise RuntimeError('deadline during training generation')
        case=m.make_case(namespace,index,split,probe_mode=probe_mode)
        for j in range(4):
            # Goal=1/signal=1 held out as a specific crossed state for G4.
            if split=='train' and j==3:continue
            for tier in m.TIERS:
                x,decode=represent(m.public_packet(case,tier,j),aligned)
                xs.append(x);ys.append(decode.index(case['probes'][j]['observed']['artifact']));ids.append(case['case_id'])
        if index%64==0:heartbeat(phase='training-data',split=split,makers=index)
    return np.array(xs,np.float32),np.array(ys,int),ids


def geometry(x,train_x,train_y,dev_x,dev_y,test_truth,seed):
    # All projections trained only on training data. Rank 8 is the decoder
    # constraint; same underlying episodes, no private state labels.
    rng=np.random.default_rng(seed);d=48
    # Fixed data-independent preprocessing bounds the SVD; shared by all arms.
    sketch=rng.normal(0,1/math.sqrt(train_x.shape[1]),(train_x.shape[1],d)).astype(np.float32)
    a=train_x@sketch;b=x@sketch;mean=a.mean(0);a-=mean;b-=mean
    rotation=np.linalg.qr(rng.normal(size=(d,d)))[0]
    invariant=float(np.max(np.abs(b-(b@rotation)@rotation.T)))
    y=np.eye(16)[train_y];cross=a.T@(y-y.mean(0))
    alignment=np.linalg.svd(cross,full_matrices=False)[0][:,:8]
    pca=np.linalg.svd(a,full_matrices=False)[2][:8].T
    random_axes=np.linalg.qr(rng.normal(size=(d,8)))[0]
    result={}
    for name,projection in [('learned-alignment',alignment),('variance-pca',pca),('random-axes',random_axes)]:
        for nonlinear in (False,True):
            # Nonlinear mixing is explicitly a different representation.
            train=(np.tanh(a) if nonlinear else a)@projection
            test=(np.tanh(b) if nonlinear else b)@projection
            design=np.column_stack((train,np.ones(len(train))))
            coef=np.linalg.solve(design.T@design+np.eye(9)*1.,design.T@y)
            prediction=np.maximum(np.column_stack((test,np.ones(len(test))))@coef,1e-5)
            prediction/=prediction.sum(axis=1,keepdims=True)
            result[name+('-nonlinear' if nonlinear else '-linear')]=prediction
    return result,dict(full_inverse_max_error=invariant,rank=8,preprocessing_dimension=d,
                       scope='shared fixed sketch then rank-limited linear probability decoder; learned predictors reported separately',
                       inverse_shared=True,transform_seed=seed)


def run_comparison(root,design,limited,heartbeat):
    from .runtime import keep,aggregate
    start=time.monotonic()
    aligned=design.get('aligned',False)
    x,y,train_ids=build_data(design['namespace']+'-train','train',1250,limited,heartbeat,aligned,design.get('probe_mode','standard'))
    dx,dy,dev_ids=build_data(design['namespace']+'-dev','dev',64,limited,heartbeat,aligned,design.get('probe_mode','standard'))
    if (root/'FIT.json').exists():
        from ..v16.records import file_digest
        fit=read(root/'FIT.json')
        if fit.get('model_sha256')!={k:file_digest(root/f'{k}.npz') for k in ('split','flat')}:
            raise ValueError('retained trained model checksum mismatch')
        models={k:Network(k,design.get('seed',0)) for k in ('split','flat')}
        for k,net in models.items():
            with np.load(root/f'{k}.npz') as saved:net.parameters=[saved[f'p{i}'] for i in range(len(net.parameters))]
    else:
        models,receipt=train_pair(x,y,dx,dy,design.get('seed',0),limited,heartbeat,epochs=design.get('epochs',80))
        for k,n in models.items():n.save(root/f'{k}.npz')
        from ..v16.records import file_digest
        write(root/'FIT.json',dict(**receipt,training_makers=1250,training_history_episodes=10000,
              supervised_examples=len(y),development_makers=64,training_seconds=time.monotonic()-start,
              labels='sampled future artifacts, never latent state',crossed_goal1_signal1_excluded=True,
              model_sha256={k:file_digest(root/f'{k}.npz') for k in models},
              representation='invertible known public-role alignment' if aligned else 'untransformed physical coordinates'))
    cases=[m.make_case(design['namespace']+'-test',i,'test',family=design.get('family','board'),length=design.get('length',8),probe_mode=design.get('probe_mode','standard')) for i in range(design.get('histories',128))]
    represented=[represent(m.public_packet(c,t,j),aligned) for c in cases for t in m.TIERS for j in range(4)]
    tx=np.array([item[0] for item in represented])
    predictions={k:n.forward(tx)[0] for k,n in models.items()}
    projected,geom=geometry(tx,x,y,dx,dy,None,design.get('seed',0));predictions.update(projected)
    write(root/'GEOMETRY.json',geom)
    offset=0;evaluation_start=time.monotonic()
    for first in range(0,len(cases),8):
        if (root/'blocks'/f'test-{first:05d}.json').exists():
            from .runtime import load
            offset+=len(load(root,f'test-{first:05d}'))*12
            continue
        if limited():return
        cpu=time.process_time();wall=time.monotonic();units=[]
        for case in cases[first:first+8]:
            rows=m.evaluate(case)
            for base_row in rows:
                base_row['condition']='crossed-heldout' if base_row['probe']==3 else 'base'
            for tier in m.TIERS:
                for j,probe in enumerate(case['probes']):
                    truth=m.artifacts(m.policy(case['world'],case['truth']['future_state'],probe['context']))
                    for method,values in predictions.items():
                        q=np.zeros(16);q[represented[offset][1]]=values[offset]
                        rows.append(dict(tier=tier,probe=j,method=method,condition='crossed-heldout' if j==3 else 'base',
                            instrument='valid',result=dict(probabilities=q.tolist()),
                            scores=m.score(q,truth,probe['observed']['artifact'])))
                    offset+=1
            # Causal architectural intervention with explicit evaluator selection.
            a=m.public_packet(case,'process-history',0);b=json.loads(a);b['current']['goal']=1 if b['current']['goal']==0 else 0
            ax=represent(a,aligned)[0][None,:];bx=represent(m.canonical(b),aligned)[0][None,:]
            swapped=models['split'].swap_prediction(ax,bx,'current')[0]
            actual=models['split'].forward(bx)[0][0]
            rows.append(dict(tier='process-history',probe=0,method='split-current-swap',condition='intervention-diagnostic',
                instrument='valid',result=dict(swapped=swapped.tolist(),full_counterfactual=actual.tolist(),
                scope='architectural current block; no claim of identified goal or skill neurons'),
                scores=dict(prediction_distance=float(np.max(np.abs(swapped-actual))))))
            if design.get('interventions'):
                # Evaluator chooses counterfactual siblings; only their public
                # observations enter readers. Keep all interventions clustered.
                for condition,change in [('false-belief',dict(signal=0,reader_fact=0)),
                    ('reader-only',dict(signal=0,reader_fact=1)),('maker-correction',dict(signal=1,reader_fact=1))]:
                    public=json.loads(m.public_packet(case,'process-history',0));public['current'].update(change)
                    payload=m.canonical(public)
                    truth=m.artifacts(m.policy(case['world'],case['truth']['future_state'],public['current']))
                    observed=random.Random(m.seed(case['case_id'],condition)).choices(range(16),weights=truth)[0]
                    for method,net in models.items():
                        q=predict(net,payload,aligned)
                        rows.append(dict(tier='process-history',probe=0,method=method,condition='g2-'+condition,instrument='valid',
                            result=dict(probabilities=q.tolist(),public=public,truth_evaluator_only=truth.tolist(),observed_evaluator_only=observed),
                            scores=m.score(q,truth,observed)))
                donor=copy.deepcopy(case);state=case['truth']['future_state'][:];state[0]=(state[0]+1)%3
                rng=random.Random(m.seed(case['case_id'],'skill-intervention'))
                for h,obs in enumerate(donor['history']):
                    donor['history'][h],_=m.draw(case['world'],state,obs['context'],rng)
                donor_payload=m.public_packet(donor,'process-history',0)
                own_x,_=represent(m.public_packet(case,'process-history',0),aligned)
                donor_x,decode=represent(donor_payload,aligned)
                swapped=models['split'].swap_prediction(own_x[None,:],donor_x[None,:],'history')[0]
                q=np.zeros(16);q[decode]=swapped
                truth=m.artifacts(m.policy(case['world'],state,case['probes'][0]['context']))
                observed=rng.choices(range(16),weights=truth)[0]
                rows.append(dict(tier='process-history',probe=0,method='split-history-swap',condition='g1-skill-intervention',instrument='valid',
                    result=dict(probabilities=q.tolist(),donor_public=json.loads(donor_payload),truth_evaluator_only=truth.tolist(),
                                scope='history-block intervention after changing acquired skill only; block is not uniquely a skill coordinate'),
                    scores=m.score(q,truth,observed)))
                for policy_change in (False,True):
                    for goal_change in (False,True):
                        sibling=copy.deepcopy(case);state=case['truth']['initial_state'][:]
                        rng=random.Random(m.seed(case['case_id'],'learned-persistence'))
                        for h,obs in enumerate(sibling['history']):
                            local=state[:]
                            if policy_change and h>=len(sibling['history'])//2:local[1]=(local[1]+1)%3
                            sibling['history'][h],_=m.draw(case['world'],local,obs['context'],rng)
                        if policy_change:state[1]=(state[1]+1)%3
                        ctx=sibling['probes'][0]['context'];ctx['goal']=int(goal_change)
                        payload=m.public_packet(sibling,'process-history',0)
                        truth=m.artifacts(m.policy(case['world'],state,ctx));observed=rng.choices(range(16),weights=truth)[0]
                        for method,net in models.items():
                            q=predict(net,payload,aligned)
                            rows.append(dict(tier='process-history',probe=0,method=method,
                                condition=f'g1-goal-{int(goal_change)}-policy-{int(policy_change)}',instrument='valid',
                                result=dict(probabilities=q.tolist(),public=json.loads(payload),truth_evaluator_only=truth.tolist()),
                                scores=m.score(q,truth,observed)))
            units.append(dict(case=case,rows=rows))
        keep(root,f'test-{first:05d}',units,time.process_time()-cpu,time.monotonic()-wall)
        heartbeat(phase='held-out',completed=first+len(units))
        if first==8:
            elapsed=time.monotonic()-evaluation_start
            write(root/'FORECAST-test.json',dict(completed_units=16,evaluation_seconds=elapsed,fit_and_setup_seconds=evaluation_start-start,
                observed_remaining_seconds=elapsed/16*(len(cases)-16),twice_as_fast_remaining_seconds=elapsed/32*(len(cases)-16)))
    aggregate(root)
