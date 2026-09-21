"""Native factor counterfactuals and saved-reader capability, without fitting."""
from collections import defaultdict
from itertools import product
import gzip
import time
import numpy as np
from scipy.special import erf
from ..v18_3.io import canonical, read, write, file_digest
from ..v18_3.world import rng
from . import local_world as L
from .readout_model import save_arrays


def endpoint_law(world, maker, context):
    initial=tuple(context['initial']); mass={(initial, initial):1.}
    for step in range(3):
        following=defaultdict(float)
        for (artifact, previous), weight in mass.items():
            for _, operation, probability in L.choices(world,maker,artifact,step,context):
                after=L.execute(artifact,previous,operation,maker)
                following[after,artifact]+=weight*probability
        mass=following
    result=np.zeros(8)
    for (artifact,_),weight in mass.items():result[artifact[0]+artifact[1]*2+artifact[2]*4]+=weight
    return result


def hybrid(recipient,donor,factor):
    if factor not in (0,2):raise ValueError('only purpose and belief admitted')
    value=list(recipient);value[factor]=donor[factor];return tuple(value)


def donor_for(maker,factor,change):
    value=list(maker);value[1]=1-value[1]
    if change:value[factor]=1-value[factor]
    return tuple(value)


def sample_episode(world,maker,context,uniforms):
    artifact=previous=tuple(context['initial']);events=[]
    for step,u in enumerate(uniforms):
        choices=list(L.choices(world,maker,artifact,step,context));cum=np.cumsum([p for _,_,p in choices])
        goal,operation,_=choices[min(int(np.searchsorted(cum,u,side='right')),len(choices)-1)]
        after=L.execute(artifact,previous,operation,maker)
        events.append(dict(step=step,goal=goal,operation=operation,before=list(artifact),after=list(after),undo_buffer=list(previous)))
        previous,artifact=artifact,after
    return artifact[0]+artifact[1]*2+artifact[2]*4,events


def softmax(x):
    exp=np.exp(x-x.max(-1,keepdims=True));return exp/exp.sum(-1,keepdims=True)


def decode(parameters,state):
    p=softmax((state@parameters['head.weight'].T+parameters['head.bias']).reshape(*state.shape[:-1],4,8))
    if not np.isfinite(p).all() or np.any(p<=0):raise ValueError('invalid probability')
    return p


def final_normalize(parameters,state):
    return ((state-state.mean(-1,keepdims=True))/np.sqrt(state.var(-1,keepdims=True)+1e-5)
        *parameters['encoder.norm2.weight']+parameters['encoder.norm2.bias'])


def reconstruct(parameters,codes,novel,kind,site='final'):
    # Double-precision reconstruction of the fixed final state; no Torch fitting.
    if site not in ('final','pre-final-normalization') or (site!='final' and kind!='transformer'):
        raise ValueError('unadmitted intervention site')
    a={k:v.astype(float) for k,v in parameters.items()};n,t=codes.shape
    if t!=32 or codes.min()<0 or codes.max()>31:raise ValueError('invalid observed stream')
    tokens=np.full((n,33),32);positions=np.cumsum(novel,axis=1)
    for i in range(n):tokens[i,1:1+sum(novel[i])]=codes[i,novel[i]]
    x=a['embedding.weight'][tokens]
    def linear(z,key):return z@a[key+'.weight'].T+a[key+'.bias']
    if kind=='recurrent':
        h=np.zeros((n,38));hidden=[]
        for j in range(33):
            ir,iz,iv=np.split(x[:,j]@a['encoder.weight_ih_l0'].T+a['encoder.bias_ih_l0'],3,-1)
            hr,hz,hv=np.split(h@a['encoder.weight_hh_l0'].T+a['encoder.bias_hh_l0'],3,-1)
            reset=1/(1+np.exp(-(ir+hr)));update=1/(1+np.exp(-(iz+hz)))
            h=(1-update)*np.tanh(iv+reset*hv)+update*h;hidden.append(h.copy())
        h=np.stack(hidden,1)
    elif kind=='transformer':
        x=x+a['positions'];q,k,v=np.split(x@a['encoder.self_attn.in_proj_weight'].T+a['encoder.self_attn.in_proj_bias'],3,-1)
        def heads(z):return z.reshape(n,33,4,8).transpose(0,2,1,3)
        q,k,v=map(heads,(q,k,v));scores=q@k.transpose(0,1,3,2)/np.sqrt(8)
        scores=np.where(np.triu(np.ones((33,33),bool),1),-np.inf,scores)
        attention=(softmax(scores)@v).transpose(0,2,1,3).reshape(n,33,32)
        def norm(z,key):return (z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+1e-5)*a[key+'.weight']+a[key+'.bias']
        h=norm(x+linear(attention,'encoder.self_attn.out_proj'),'encoder.norm1')
        f=linear(h,'encoder.linear1');f=.5*f*(1+erf(f/np.sqrt(2)))
        pre=h+linear(f,'encoder.linear2')
        h=pre if site=='pre-final-normalization' else norm(pre,'encoder.norm2')
    else:raise ValueError('unadmitted saved architecture')
    state=h[np.arange(n)[:,None],positions]
    return state,decode(a,final_normalize(a,state) if site=='pre-final-normalization' else state)


def loss(target,pred):
    logs=np.zeros_like(pred);np.log(pred,out=logs,where=pred>0)
    value=-(target*logs).sum(-1)
    return np.where(np.any((target>0)&(pred<=0),axis=-1),np.inf,value)


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def check_hash(path,expected):
    if file_digest(path)!=expected:raise ValueError('frozen input corruption')


def controls():
    r=(0,0,0,0);d=donor_for(r,0,True);stay=donor_for(r,0,False)
    return {'live:native_dependency':L.execute((1,0,0),(0,0,0),'repair-evidence',(0,1,0,0))==(1,1,0),
        'placebo:constant_output':L.execute((1,0,0),(0,0,0),'inspect',r)==(1,0,0),
        'positive:hybrid_only_target':hybrid(r,d,0)==(1,0,0,0),
        'positive:stay_ignores_nuisance':hybrid(r,stay,0)==r and stay!=r,
        'positive:uniform_normalization':bool(np.allclose(softmax(np.zeros((2,8))),1/8)),
        'positive:known_log_loss':bool(np.allclose(loss(np.array([[.5,.5]]),np.array([[.5,.5]])),np.log(2)))}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('fixture control failed before outcomes')
    for n,h in cfg['input_files'].items():check_hash(root/'inputs'/n,h)
    capability=read(root/'inputs/capability/SUMMARY.json')
    if not all(r['passed'] for r in capability['capability']):raise ValueError('saved reader capability not admitted')
    parameters={};max_p=max_h=0.;timing=[];began=time.process_time()
    for draw,seed,kind in product(cfg['training_draws'],cfg['fit_seeds'],cfg['kinds']):
        pulse(phase='saved-model-controls',draw=draw,seed=seed,kind=kind)
        key=f'{draw}-{seed}-{kind}';a=arrays(root/'inputs/models'/f'{key}.npz');parameters[draw,seed,kind]=a
        original=arrays(root/'inputs/previous'/f'{draw}-reader.npz');saved=arrays(root/'inputs/previous'/f'{key}.npz')
        hidden,pred=reconstruct(a,original['codes'],original['novel'],kind)
        max_p=max(max_p,float(abs(pred-saved['probabilities']).max()));max_h=max(max_h,float(abs(hidden-saved['hidden']).max()))
        if max_p>2e-6 or max_h>2e-5:raise ValueError('saved-model reconstruction mismatch')
        if not np.array_equal(decode({k:v.astype(float) for k,v in a.items()},hidden),pred):raise ValueError('no-op decode failed')
    checks['positive:saved_model_reconstruction']=True;checks['positive:no_op_decode']=True
    write(root/'MODEL_CONTROLS.json',dict(controls=checks,max_probability_error=max_p,max_hidden_error=max_h))
    timing.append(dict(phase='saved-model-controls',cpu_seconds=time.process_time()-began));rows=[];effects=[];native=[]
    means={d:arrays(root/'inputs/means'/f'{d}.npz')['mean'].reshape(4,8) for d in cfg['training_draws']}
    maker_index={m:i for i,m in enumerate(L.MAKERS)};maximum_law_error=0.
    for split,lineages in [('train',cfg['train_lineages']),('development',cfg['development_lineages'])]:
        for lineage in lineages:
            began=time.process_time();pulse(phase='native-counterfactual-laws',split=split,lineage=lineage)
            world=L.law(lineage);laws=np.array([[endpoint_law(world,maker,ctx) for ctx in L.CONTEXTS] for maker in L.MAKERS])
            enumerated=np.zeros_like(laws);records=L.enumerate_world(world)
            for r in records:
                a=r['final'];enumerated[r['maker_index'],r['context_index'],a[0]+2*a[1]+4*a[2]]+=r['probability']*64
            error=float(abs(laws-enumerated).max());maximum_law_error=max(maximum_law_error,error)
            if error>1e-12 or not np.allclose(laws.sum(-1),1,atol=1e-12):raise ValueError('native law audit failed')
            save_arrays(root/'evaluator'/f'{split}-{lineage}-laws.npz',laws=laws)
            for draw in cfg['training_draws']:
                random=rng('v19-C1-native-fixtures',split,lineage,draw);contexts=random.integers(4,size=32);uniforms=random.random((32,3))
                codes=[];conditionals=[];posteriors=[]
                for mi,maker in enumerate(L.MAKERS):
                    stream=[];post=np.ones(16)/16;cond=[];posts=[]
                    for step,(ci,uu) in enumerate(zip(contexts,uniforms)):
                        endpoint,events=sample_episode(world,maker,L.CONTEXTS[int(ci)],uu);code=8*int(ci)+endpoint
                        stream.append(code);post=post*laws[:,int(ci),endpoint];post=post/post.sum();cond.append(np.einsum('i,ijk->jk',post,laws));posts.append(post.copy())
                        native.append(dict(split=split,lineage=lineage,draw=draw,maker_index=mi,position=step,context=int(ci),uniforms=uu.tolist(),code=code,steps=events))
                    codes.append(stream);conditionals.append(cond);posteriors.append(posts)
                order=rng('v19-C1-anonymous-order',split,lineage,draw).permutation(16);inverse=np.argsort(order)
                codes=np.array(codes)[order];novel=np.ones_like(codes,dtype=bool);conditional=np.array(conditionals)[order]
                stem=f'{split}-{lineage}-{draw}'
                save_arrays(root/'reader'/f'{stem}.npz',codes=codes,novel=novel)
                save_arrays(root/'evaluator'/f'{stem}.npz',conditional=conditional,posteriors=np.array(posteriors)[order],maker=np.array(L.MAKERS)[order],maker_indices=order)
                predictions={};
                for seed,kind in product(cfg['fit_seeds'],cfg['kinds']):
                    pulse(phase='saved-reader-fixtures',split=split,lineage=lineage,draw=draw,seed=seed,kind=kind)
                    hidden,pred=reconstruct(parameters[draw,seed,kind],codes,novel,kind);predictions[seed,kind]=pred
                    save_arrays(root/'forecasts'/f'{stem}-{seed}-{kind}.npz',hidden=hidden,probabilities=pred)
                    for length in cfg['lengths']:
                        target=conditional[:,length-1];no=np.broadcast_to(means[draw],target.shape)
                        rows.append(dict(record='capability',split=split,lineage=lineage,draw=draw,seed=seed,kind=kind,length=length,
                            loss=float(loss(target,pred[:,length-1]).mean()),mean_loss=float(loss(target,no).mean()),exact_loss=float(loss(target,target).mean()),
                            prediction_std=float(pred[:,length-1].std(0).mean()),histories=16))
                for ri,recipient in enumerate(L.MAKERS):
                    for factor,change in product((0,2),(False,True)):
                        donor=donor_for(recipient,factor,change);di=maker_index[donor];target=hybrid(recipient,donor,factor);hi=maker_index[target]
                        tv=.5*abs(laws[hi]-laws[ri]).sum(-1);collateral=.5*abs(laws[di]-laws[hi]).sum(-1)
                        if not change and not np.array_equal(laws[hi],laws[ri]):raise ValueError('stay reference changed')
                        effects.append(dict(split=split,lineage=lineage,draw=draw,recipient=ri,donor=di,hybrid=hi,factor=factor,change=change,tv=tv.tolist(),full_donor_tv=collateral.tolist()))
                        for (seed,kind),pred in predictions.items():
                            for length in cfg['lengths']:
                                rows.append(dict(record='full-replacement',split=split,lineage=lineage,draw=draw,seed=seed,kind=kind,length=length,
                                    recipient=ri,donor=di,hybrid=hi,factor=factor,change=change,
                                    recipient_loss=float(loss(laws[hi],pred[inverse[ri],length-1]).mean()),donor_loss=float(loss(laws[hi],pred[inverse[di],length-1]).mean()),
                                    oracle_loss=float(loss(laws[hi],laws[hi]).mean()),
                                    prediction_tv=float((.5*abs(pred[inverse[di],length-1]-pred[inverse[ri],length-1]).sum(-1)).mean())))
            timing.append(dict(phase='lineage',split=split,lineage=lineage,cpu_seconds=time.process_time()-began))
    (root/'raw').mkdir(exist_ok=True)
    for n,data in [('native',native),('fixtures',effects),('scores',rows)]:
        (root/'raw'/f'{n}_points.json.gz').write_bytes(gzip.compress(canonical(data),mtime=0))
    write(root/'TIMING.jsonl',dict(measurements=timing,accounting='component times are included in native charge, not added twice'))
    write(root/'INPUT_SCHEMA.json',dict(reader=['codes','novel'],evaluator=['maker','factor','change','hybrid','law','posterior','conditional','hidden','forecasts','scores','uniforms','steps'],alignment_fits=0))
    return dict(controls={**checks,'positive:every_law_enumeration':maximum_law_error<1e-12,'positive:all_stay_identities':True},
        max_law_error=maximum_law_error,fixture_pairs=len(effects),score_rows=len(rows),native_episodes=len(native),
        training_lineages=cfg['train_lineages'],development_lineages=cfg['development_lineages'],
        new_model_settings=0,alignment_fits=0,scope='native purpose/belief fixture and frozen-reader capability audit; no selective-interchange acceptance',
        warrant='exploratory constructed method; miniature — architecture untested')
