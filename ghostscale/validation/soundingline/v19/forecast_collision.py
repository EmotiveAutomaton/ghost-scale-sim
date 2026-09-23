"""Exact frozen forecast partitions with law-resampled conditional uncertainty."""
import numpy as np
from itertools import product
import gzip
import json
import time
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import goal_decision as D

ARMS=('raw-history','frozen-latent','learned-bank','matched-frequency')
FORECASTS=('restricted','goal-product')
WEIGHTS=('native','equal-frame')

METRICS = ('squared_error', 'group_bias', 'within_law_variation', 'across_law_variation')


def partition(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 3 or p.shape[1:] != (3, 3) or not np.isfinite(p).all():
        raise ValueError('forecast shape or finite')
    if np.min(p) < 0 or np.max(p) > 1+1e-12 or not np.allclose(p.sum(2), 1, rtol=0, atol=1e-12):
        raise ValueError('forecast simplex')
    # Python binary64 equality, including equal signed zero; no tolerance merges.
    seen = {}; ids = []
    for row in p:
        key = tuple(row.ravel())
        if key not in seen: seen[key] = len(seen)
        ids.append(seen[key])
    return np.asarray(ids, dtype=np.int64)


def sufficient(p, target, weight, ids=None):
    p, target, weight = (np.asarray(x, dtype=float) for x in (p, target, weight))
    actual = partition(p)
    if ids is not None and not np.array_equal(ids, actual): raise ValueError('faulty merge')
    ids = actual
    laws, frames = weight.shape if weight.ndim == 2 else (0, 0)
    if target.shape != (laws, frames, 3, 3) or len(p) != frames: raise ValueError('target shape')
    if not np.isfinite(target).all() or not np.isfinite(weight).all(): raise ValueError('nonfinite')
    if np.min(target) < 0 or np.max(target) > 1+1e-12 or not np.allclose(target.sum(3),1,rtol=0,atol=1e-12): raise ValueError('target simplex')
    if np.min(weight) < 0 or not np.allclose(weight.sum(1),1,rtol=0,atol=1e-12): raise ValueError('weight normalization')
    count = int(ids.max())+1
    mass = np.zeros((laws,count)); sums = np.zeros((laws,count,9)); square = np.zeros((laws,9))
    for law in range(laws):
        np.add.at(mass[law],ids,weight[law])
        np.add.at(sums[law],ids,weight[law,:,None]*target[law].reshape(frames,9))
        square[law] = (weight[law,:,None]*target[law].reshape(frames,9)**2).sum(0)
    centered = square-np.divide(sums*sums,mass[:,:,None],out=np.zeros_like(sums),where=mass[:,:,None]>0).sum(1)
    loss = (weight[:,:,None,None]*(p[None]-target)**2).sum(1).reshape(laws,9)
    if np.min(centered)<-1e-12: raise ValueError('negative within-law variance')
    return dict(ids=ids,mass=mass,target_sum=sums,target_square=square,within=centered,loss=loss)


def resampled(s, counts, chunk=128):
    """Recompute each pooled target from law multiplicities, never pooled means."""
    counts=np.asarray(counts,dtype=float)
    laws,groups=s['mass'].shape
    if counts.ndim!=2 or counts.shape[1]!=laws or np.min(counts)<0 or not np.all(counts.sum(1)==laws):
        raise ValueError('law multiplicities')
    out=np.empty((len(counts),len(METRICS),9))
    common_mass=np.array_equal(s['mass'],np.broadcast_to(s['mass'][0],s['mass'].shape))
    gram=None
    if common_mass:
        # Equal-frame groups have the same mass under every law. Expand the
        # pooled squared numerator once; the law-resample target still changes.
        scaled=np.divide(s['target_sum'],np.sqrt(s['mass'])[:,:,None],
                         out=np.zeros_like(s['target_sum']),where=s['mass'][:,:,None]>0)
        gram=np.einsum('lgk,mgk->lmk',scaled,scaled,optimize=True)
    for start in range(0,len(counts),chunk):
        n=counts[start:start+chunk]
        second=n@s['target_square']/laws
        if common_mass:
            pooled=np.einsum('bi,bij->bj',n,(n@gram.reshape(laws,-1)).reshape(len(n),laws,9))/laws**2
        else:
            mass=n@s['mass']; sums=(n@s['target_sum'].reshape(laws,-1)).reshape(len(n),groups,9)
            pooled=np.divide(sums*sums,mass[:,:,None],out=np.zeros_like(sums),where=mass[:,:,None]>0).sum(1)/laws
        variation=second-pooled; within=n@s['within']/laws; loss=n@s['loss']/laws
        out[start:start+len(n)]=np.stack((loss,loss-variation,within,variation-within),axis=1)
    if not np.isfinite(out).all() or np.min(out)<-1e-12: raise ValueError('invalid decomposition')
    return out.reshape(len(counts),len(METRICS),3,3)


def controls():
    p=np.tile([.5,.5,0.],(2,3,1)); target=np.array([p.copy(),p.copy()])
    target[0,:,:,:2]=[1,0];target[1,:,:,:2]=[0,1]
    w=np.full((2,2),.5);s=sufficient(p,target,w);v=resampled(s,[[1,1]])[0]
    one=resampled(s,[[2,0]])[0]
    return {'live:opposing_targets_same_forecast':bool(np.allclose(v[3,:,:2],.25,atol=1e-14,rtol=0)),
            'placebo:identical_native_targets':bool(np.max(abs(resampled(sufficient(p,p[None],w[:1]),[[1]])))==0),
            'positive:law_resampling_rebuilds_targets':bool(np.max(abs(one[3]))==0 and np.allclose(one[1,:,:2],.25)),
            'positive:within_across_conservation':bool(np.allclose(v[0],v[1]+v[2]+v[3],atol=1e-14,rtol=0))}


def summarize(points, boot, cfg):
    ds,ss,bs=(cfg[k] for k in ('training_draws','fit_seeds','budgets'))
    fields=('draw','seed','budget','arm','forecast','weighting')
    idx={tuple(r[k] for k in fields):r for r in points}
    expected=set(product(ds,ss,bs,ARMS,FORECASTS,WEIGHTS))
    if len(idx)!=len(points) or set(idx)!=expected:raise ValueError('summary roster')
    means=[];contrasts=[];areas=[]
    def values(b,a,f,w):
        return np.array([idx[d,s,b,a,f,w]['values'] for d,s in product(ds,ss)]).reshape(len(ds),len(ss),4,3,3)
    def interval(v):return dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)))
    for f,w,mi,t,g in product(FORECASTS,WEIGHTS,range(4),range(3),range(3)):
        ident=dict(forecast=f,weighting=w,metric=METRICS[mi],step=t,goal=g)
        for a,b in product(ARMS,bs):
            v=values(b,a,f,w)[:,:,mi,t,g]
            means.append(dict(ident,arm=a,budget=b,mean=float(v.mean()),draw_means=v.mean(1).tolist(),fit_seed_means=v.mean(0).tolist(),**interval(boot[b,a,f,w][:,mi,t,g])))
        for baseline in (a for a in ARMS if a!='learned-bank'):
            vv=[];bb=[]
            for b in bs:
                v=values(b,'learned-bank',f,w)[:,:,mi,t,g]-values(b,baseline,f,w)[:,:,mi,t,g]
                bv=boot[b,'learned-bank',f,w][:,mi,t,g]-boot[b,baseline,f,w][:,mi,t,g]
                vv.append(v);bb.append(bv)
                contrasts.append(dict(ident,arm='learned-bank',baseline=baseline,budget=b,mean=float(v.mean()),draw_means=v.mean(1).tolist(),fit_seed_means=v.mean(0).tolist(),**interval(bv)))
            v=np.trapezoid(vv,x=np.log(bs),axis=0)/np.log(bs[-1]/bs[0]);bv=np.trapezoid(bb,x=np.log(bs),axis=0)/np.log(bs[-1]/bs[0])
            areas.append(dict(ident,arm='learned-bank',baseline=baseline,mean=float(v.mean()),draw_means=v.mean(1).tolist(),fit_seed_means=v.mean(0).tolist(),**interval(bv)))
    return dict(estimates=means,contrasts=contrasts,normalized_log_budget_area=areas)


def run(root,plan,pulse):
    cfg=plan['design'];base=root/'inputs';checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    parent=read(base/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if cfg[k]!=parent[k]:raise ValueError('population')
    if cfg['tiers']!=['E2-full'] or cfg['bootstrap_seed']!=191022:raise ValueError('design')
    if len(cfg['budgets'])<2 or any(a>=b for a,b in zip(cfg['budgets'],cfg['budgets'][1:])):raise ValueError('budgets')
    packets=read(base/'reader/PACKETS.json');keys=sorted(packets['packets']);ops=[]
    for k in keys:
        p=packets['packets'][k];D.S.validate(p)
        if digest(p)!=k or p['tier']!='E2-full':raise ValueError('packet')
        ops.append([D.S.L.OPERATIONS.index(e['operation']) for e in p['inputs']['observations']])
    codes=np.asarray(ops,dtype=int)@np.array([36,6,1]);ls=cfg['development_lineages'];refs=read(base/'evaluator/REFERENCES.json');pos={k:i for i,k in enumerate(keys)}
    if len(refs)!=len(ls) or {r['lineage'] for r in refs}!=set(ls):raise ValueError('lineage roster')
    target=np.zeros((len(ls),len(keys),27));mass=np.zeros((len(ls),len(keys)))
    for r in refs:
        li=ls.index(r['lineage'])
        if r['tier']!='E2-full' or len(r['frames'])!=len(keys) or {f['frame'] for f in r['frames']}!=set(keys):raise ValueError('frame roster')
        for f in r['frames']:
            i=pos[f['frame']];mass[li,i]=f['mass']
            if len(dict(f['target']))!=len(f['target']):raise ValueError('duplicate target')
            for k,w in f['target']:
                if not 0<=k<5832 or k%216!=codes[i] or w<=0:raise ValueError('support')
                target[li,i,k//216]=w
    if not np.isfinite(target).all() or not np.allclose(target.sum(2),1,rtol=0,atol=1e-12):raise ValueError('native normalization')
    native=np.stack([target@(D.GOALS[:,t,None]==np.arange(3)) for t in range(3)],axis=2)
    old=json.loads(gzip.decompress((base/'parent/goal_decision_points.json.gz').read_bytes()));idx={tuple(r[k] for k in D.FIELDS):r for r in old}
    expected=set(product(cfg['tiers'],cfg['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,cfg['training_draws'],cfg['fit_seeds']))
    if len(idx)!=len(old) or set(idx)!=expected:raise ValueError('parent roster')
    write(root/'reader/PACKETS.json',packets);write(root/'evaluator/REFERENCES.json',refs)
    np.savez_compressed(root/'evaluator/NATIVE_FORECASTS.npz',marginals=native,weights=mass)
    write(root/'ARRAY_SCHEMA.json',dict(laws=ls,frames=keys,metrics=list(METRICS),group_membership='ids maps every listed frame to its exact forecast group; every law/frame pair is retained',mass_axes=['law','group'],sum_axes=['law','group','flattened position/goal'],pooled_target_axes=['group','position','goal'],undefined='NaN group mean iff pooled mass zero',bootstrap='multinomial law counts; rebuild target sums and masses, both draws and all seeds paired',evidence_role='all group arrays and native means evaluator only'))
    (root/'groups').mkdir();points=[];boot={};timing=[];parent_error=0.;parents=0
    sample=np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls)))
    counts=np.array([(sample==i).sum(1) for i in range(len(ls))]).T
    write(root/'BOOTSTRAP.json',dict(seed=cfg['bootstrap_seed'],resamples=cfg['bootstrap_resamples'],lineages=ls,count_digest=digest(counts.tolist())))
    for draw,seed,budget,arm,forecast in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS,FORECASTS):
        pulse(phase='exact-forecast-collisions',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast);tick=time.process_time()
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(base/'decisions'/(stem+'.npz'),allow_pickle=False) as z:
            q=z['probabilities'];d=D.decisions(q)
            if not np.array_equal(z['operations'],codes) or not np.array_equal(z['coordinate'],d['coordinate']) or not np.array_equal(z['marginals'],d['marginals']):raise ValueError('decision identity')
            marg=z['marginals']
        if np.any((q[None]<=0)&(target>0)):raise ValueError('fine support')
        logq=np.zeros_like(q);np.log(q,out=logq,where=q>0);loss=-(target*mass[:,:,None]*logq).sum((1,2))
        chosen=D.GOALS[d['coordinate']];correct=np.take_along_axis(native,np.broadcast_to(chosen[None,:,:,None],(len(ls),len(keys),3,1)),3)[:,:,:,0];accuracy=(correct*mass[:,:,None]).sum((1,2))/3
        for li,l in enumerate(ls):
            prior=idx['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]
            error=max(abs(loss[li]-prior['loss']),abs(accuracy[li]-prior['goal_accuracy']));parent_error=max(parent_error,float(error))
            if error>1e-10:raise ValueError('parent score')
            parents+=1
        for wi,(weighting,w) in enumerate(zip(WEIGHTS,(mass,np.full_like(mass,1/len(keys))))):
            s=sufficient(marg,native,w);values=resampled(s,np.ones((1,len(ls)),dtype=int))[0];bv=resampled(s,counts)
            total=s['mass'].sum(0);pooled=np.divide(s['target_sum'].sum(0),total[:,None],out=np.full((len(total),9),np.nan),where=total[:,None]>0).reshape(-1,3,3)
            if not np.allclose(pooled[total>0].sum(2),1,atol=1e-12,rtol=0):raise ValueError('group simplex')
            sizes=np.bincount(s['ids']);first=np.array([np.flatnonzero(s['ids']==i)[0] for i in range(len(sizes))])
            np.savez_compressed(root/'groups'/(stem+'-'+weighting+'.npz'),**s,group_target=pooled,group_forecast=marg[first],frame_counts=sizes,law_frame_counts=sizes*len(ls),positive_mass_counts=np.array([(w[:,s['ids']==i]>0).sum() for i in range(len(sizes))]),values=values)
            points.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,weighting=weighting,groups=len(sizes),singleton_frame_groups=int((sizes==1).sum()),singleton_law_frame_groups=int((sizes*len(ls)==1).sum()),values=values.tolist()))
            key=budget,arm,forecast,weighting
            if key not in boot:boot[key]=np.zeros_like(bv)
            boot[key]+=bv/(len(cfg['training_draws'])*len(cfg['fit_seeds']))
        timing.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,cpu_seconds=time.process_time()-tick))
    (root/'forecast_collision_points.json.gz').write_bytes(gzip.compress(canonical(points),mtime=0));write(root/'TIMING.jsonl',dict(measurements=timing))
    write(root/'PARENT_REPRODUCTION.json',dict(passed=True,cells=parents,max_error=parent_error))
    return dict(controls=checks,settings=len(points)//2,group_bundles=len(points),packets=len(keys),parent_cells=parents,parent_max_error=parent_error,**summarize(points,boot,cfg),scope='exact binary64 full-vector groups;all law/frame pairs;native and equal-frame populations;conditional law-bootstrap intervals;no near-equality merging or historical correspondence',fits=0)
