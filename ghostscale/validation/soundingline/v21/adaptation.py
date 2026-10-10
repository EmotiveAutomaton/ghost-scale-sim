"""B07 sparse execution adaptation, with all shifted labels charged to k.

The modular arm receives a registered two-kernel execution library (native or
evidence-bit flip). This supplied factorization is an explicit scientific privilege.
No shifted validation set secretly enlarges the adaptation budget.
"""
import copy
from pathlib import Path
import numpy as np
from ..v18_3.io import read,write,file_digest
from ..v20.readers import features
from . import world as W,oracle as O
from .readers import Direct
from .studies import data,retain,controls

def context_code(x):return ((np.maximum(x[:,3:6],0)*np.array([4,2,1])).sum(1)).astype(int)
def artifact_code(x):return (x[:,:3]@np.array([4,2,1])).astype(int)
def policy_view(x):
    out=x.copy();out[:,:3]=-1;return out

def execution_table(x,y):
    table=np.full((128,8,8),.5);np.add.at(table,(y,context_code(x),artifact_code(x)),1)
    return table/table.sum(2,keepdims=True)

def execution_weight(table,x,y):
    c=context_code(x);a=artifact_code(x)
    logs=np.array([np.log(table[y,c,a]).sum(),np.log(table[y,c,a^2]).sum()])
    return np.exp(logs-O.logsumexp(logs,0))

def modular(policy,table,x,weight):
    c=context_code(x);a=artifact_code(x)
    emission=weight[0]*table[:,c,a].T+weight[1]*table[:,c,a^2].T
    return O.normalize_log(policy.predict(policy_view(x))+np.log(emission))

def finetune(original,x,y,steps=60):
    model=copy.deepcopy(original)
    if not len(y):return model
    z=features(x).astype(np.float32);m=[np.zeros_like(p) for p in model.params];v=[np.zeros_like(p) for p in model.params]
    for step in range(1,steps+1):
        a,b,c,d=model.params;h=np.tanh(z@a+b);pp=np.exp(O.normalize_log((h@c+d).astype(float))).astype(np.float32)
        pp[np.arange(len(y)),y]-=1;pp/=len(y);dh=(pp@c.T)*(1-h*h)
        gradients=[z.T@dh+1e-4*a,dh.sum(0),h.T@pp+1e-4*c,pp.sum(0)]
        for j,(p,g) in enumerate(zip(model.params,gradients)):
            m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
            p-=.001*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
    return model

def run(root,d,pulse):
    src=Path(d['headroom_root']);s=read(src/'SUMMARY.json');receipt=read(src/'COMPLETE.json')
    if file_digest(src/'SUMMARY.json')!=receipt['files']['SUMMARY.json']:raise ValueError('changed gate')
    original=read(src/'PLAN.json')['design']
    if original.get('history_shift')!='native' or original.get('shift')!='changed-tool':raise ValueError('sparse-current-shift ceiling required')
    if not s['screen_admitted']:return dict(card='G21-B07',population=[],state=s['screen_disposition'],controls=controls())
    population=[];metadata=[]
    for fit in d.get('fit_seeds',[0,1]):
        x,y=data(d,'training',fit);vx,vy=data(d,'validation',fit)
        policy=Direct(policy_view(x),y,policy_view(vx),vy,fit_seed=fit,pulse=pulse)
        direct=Direct(x,y,vx,vy,fit_seed=fit,pulse=pulse)
        def fresh(n,step):return data(dict(d,training_lineages=[72000000+fit*100000+step],training_cases=n),'training',fit)
        fresh_model=Direct(x,y,vx,vy,fit_seed=fit,pulse=pulse,fresh=fresh)
        write(root/'fits'/f'{fit}.json',dict(policy=policy.metadata,direct=direct.metadata,fresh=fresh_model.metadata))
        table=execution_table(x,y);adapt_x=[];adapt_y=[]
        for j in range(256):
            lineage=d['training_lineages'][j%len(d['training_lineages'])];rows=W.enumerate_world(lineage,'changed-tool');chosen=W.sampled(rows,1,'v21-adapt',fit,j)
            xx,_=W.inputs(W.enumerate_world(lineage),chosen,d['tier'],d['history'],('adapt',fit,j),history_tier=d.get('history_tier'),history_observed=d.get('history_observed',False))
            adapt_x.append(xx[0]);adapt_y.append(int(chosen[0,10]))
        adapt_x=np.array(adapt_x);adapt_y=np.array(adapt_y)
        target=root/'training'/f'{fit}-adaptation_points.npz';target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,evidence=adapt_x,labels=adapt_y)
        for k in (0,16,64,256):
            weight=execution_weight(table,adapt_x[:k],adapt_y[:k]) if k else np.array([1.,0.])
            direct_k=finetune(direct,adapt_x[:k],adapt_y[:k]);fresh_k=finetune(fresh_model,adapt_x[:k],adapt_y[:k])
            for lineage in d['lineages']:
                pulse(phase='sparse-execution-adaptation',fit=fit,k=k,lineage=lineage)
                xx=np.load(src/'reader'/f'{lineage}_points.npz')['evidence'];e=np.load(src/'evaluator'/f'{lineage}_points.npz');yy=e['truth'];support=np.isfinite(e['supplied_logp'])
                reference=-e['marginal_logp'][np.arange(len(yy)),yy].mean()
                for name,lp in [('modular',modular(policy,table,xx,weight)),('direct-finetuned',direct_k.predict(xx)),('fresh-law-finetuned',fresh_k.predict(xx))]:
                    lp=O.normalize_log(np.where(support,lp,-np.inf));a=retain(root,f'{fit}-{k}-{lineage}-{name}',lp,yy)
                    population.append(dict(fit=fit,k=k,lineage=lineage,reader=name,metrics=a,excess_over_shift_aware_ceiling=a['log_loss']-float(reference)))
            metadata.append(dict(fit=fit,k=k,execution_kernel_weights=weight.tolist(),shifted_labels=k,shifted_validation_labels=0,finetune_steps=60,finetune_learning_rate=.001))
    write(root/'ADAPTATION_METADATA.json',metadata)
    return dict(card='G21-B07',population=population,controls=controls(),fixed_target_excess_nats=.1,
        promising_rule='modular reaches <= .1 mean excess using <= one quarter of the same-evidence direct adaptation labels; whole-world uncertainty and fresh-law comparison reported separately',
        scope='Constructed sparse-current-execution change with native past records; registered two-kernel privilege; no general causal-transfer claim')
