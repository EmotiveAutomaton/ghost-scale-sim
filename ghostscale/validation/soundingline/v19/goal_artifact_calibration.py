"""Public-endpoint grouping of complete goal calibration sufficient sums."""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import goal_class_reliability as C
from . import goal_decision as D

ARMS=C.ARMS
FORECASTS=C.FORECASTS
WEIGHTS=C.WEIGHTS
FIELDS=C.FIELDS
ARTIFACTS=tuple(product((0,1),repeat=3))


def artifact_groups(packets):
    labels=[]
    for packet in packets:
        D.S.validate(packet)
        if packet['tier']!='E2-full':raise ValueError('tier')
        artifact=tuple(packet['inputs']['artifact'])
        labels.append(ARTIFACTS.index(artifact))
    return np.repeat(np.array(labels,dtype=int)[:,None],3,axis=1)


METRICS=('pooled_bin_error','grouped_bin_error','cancellation_gap')


def sufficient(p,c,w,artifacts):
    """Axes: law, position, goal, public artifact, probability bin."""
    p,c,w=(np.asarray(x,float) for x in (p,c,w));op=np.asarray(artifacts)
    if p.ndim!=3 or p.shape[1:]!=(3,3) or c.ndim!=4 or c.shape[1:]!=p.shape or w.shape!=c.shape[:2]:raise ValueError('shape')
    if op.shape!=p.shape[:2] or not np.issubdtype(op.dtype,np.integer) or np.any(op<0) or np.any(op>=8):raise ValueError('artifact')
    C.probabilities(p);C.probabilities(c)
    if not np.allclose(p.sum(2),1,rtol=0,atol=C.TOL) or not np.allclose(c.sum(3),1,rtol=0,atol=C.TOL):raise ValueError('conservation')
    if not np.isfinite(w).all() or np.any(w<0) or not np.allclose(w.sum(1),1,rtol=0,atol=1e-10):raise ValueError('weight')
    shape=(len(c),3,3,8,10)
    bw=np.zeros(shape);bp=np.zeros(shape);bc=np.zeros(shape)
    ids=C.bins(p)
    for t,g in product(range(3),range(3)):
        code=op[:,t]*10+ids[:,t,g]
        for law in range(len(c)):
            bw[law,t,g]=np.bincount(code,weights=w[law],minlength=80).reshape(8,10)
            bp[law,t,g]=np.bincount(code,weights=w[law]*p[:,t,g],minlength=80).reshape(8,10)
            bc[law,t,g]=np.bincount(code,weights=w[law]*c[law,:,t,g],minlength=80).reshape(8,10)
    delta=bp-bc
    pooled=np.abs(delta.sum(3)).sum(3);grouped=np.abs(delta).sum((3,4));gap=grouped-pooled
    if np.min(gap)<-1e-12:raise ValueError('triangle inequality')
    mass=bw.sum(4)
    conditional_signed=np.divide(delta.sum(4),mass,out=np.full_like(mass,np.nan),where=mass>0)
    means_p=np.divide(bp,bw,out=np.full_like(bp,np.nan),where=bw>0)
    means_c=np.divide(bc,bw,out=np.full_like(bc,np.nan),where=bw>0)
    return dict(bin_weight=bw,bin_forecast_sum=bp,bin_correct_sum=bc,
        artifact_mass=mass,conditional_signed_error=conditional_signed,
        bin_forecast_mean=means_p,bin_correct_mean=means_c,
        pooled_bin_error=pooled,grouped_bin_error=grouped,cancellation_gap=gap)


def controls():
    p=np.tile([.6,.3,.1],(2,3,1));c=p[None].copy()
    c[0,0,:,0]+=.1;c[0,0,:,1]-=.1;c[0,1,:,0]-=.1;c[0,1,:,1]+=.1
    op=np.array([[0]*3,[1]*3]);w=np.array([[.5,.5]])
    x=sufficient(p,c,w,op);own=sufficient(p,p[None],w,op)
    return {'live:artifact_cancellation':bool(np.max(x['pooled_bin_error'])<1e-14 and np.min(x['cancellation_gap'][0,:,:2])>.099),
        'placebo:native_self':bool(np.max(own['grouped_bin_error'])==0),
        'positive:mass':bool(np.max(abs(x['bin_weight'].sum((3,4))-1))<1e-14),
        'positive:empty_groups':bool(np.isnan(x['conditional_signed_error'][...,2:]).all())}


def aggregate(rows,cfg):
    ls,ds,ss,bs=(cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    expected=set(product(cfg['tiers'],bs,ARMS,FORECASTS,WEIGHTS,range(3),range(3),ls,ds,ss))
    idx={tuple(r[k] for k in FIELDS):r for r in rows}
    if len(idx)!=len(rows) or set(idx)!=expected:raise ValueError('stratum roster')
    sample=np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls)))
    def estimate(v):
        line=v.mean((1,2));boot=line[sample].mean(1)
        return dict(mean=float(line.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),lineage_values=line.tolist(),draw_means=v.mean((0,2)).tolist(),feature_seed_means=v.mean((0,1)).tolist())
    means=[];contrasts=[];areas=[];gap_estimates=[]
    for tier,forecast,weighting,t,g in product(cfg['tiers'],FORECASTS,WEIGHTS,range(3),range(3)):
        def values(a,b,m):return np.array([idx[tier,b,a,forecast,weighting,t,g,l,d,s][m] for l,d,s in product(ls,ds,ss)]).reshape(len(ls),len(ds),len(ss))
        identity=dict(tier=tier,forecast=forecast,weighting=weighting,step=t,goal=g)
        for arm,budget in product(ARMS,bs):
            means.append(dict(identity,arm=arm,budget=budget,**{m:float(values(arm,budget,m).mean()) for m in METRICS}))
            gap_estimates.append(dict(identity,arm=arm,budget=budget,**estimate(values(arm,budget,'cancellation_gap'))))
        for baseline,metric in product(tuple(a for a in ARMS if a!='learned-bank'),METRICS):
            curves=[];ident=dict(identity,arm='learned-bank',baseline=baseline,metric=metric)
            for b in bs:
                v=values('learned-bank',b,metric)-values(baseline,b,metric);curves.append(v)
                contrasts.append(dict(ident,budget=b,**estimate(v)))
            areas.append(dict(ident,**estimate(np.trapezoid(curves,x=np.log(bs),axis=0)/np.log(bs[-1]/bs[0]))))
    return dict(means=means,gap_estimates=gap_estimates,contrasts=contrasts,normalized_log_budget_area=areas)


def run(root,plan,pulse):
    cfg=plan['design'];base=root/'inputs';checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    parent=read(base/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if cfg[k]!=parent[k]:raise ValueError('population')
    if cfg['tiers']!=['E2-full'] or cfg['bin_edges']!=C.EDGES.tolist() or cfg['artifacts']!=[list(a) for a in ARTIFACTS]:raise ValueError('design')
    packets=read(base/'reader/PACKETS.json');keys=sorted(packets['packets']);ops=[]
    for k in keys:
        p=packets['packets'][k];D.S.validate(p)
        if digest(p)!=k or p['tier']!='E2-full':raise ValueError('packet')
        ops.append([D.S.L.OPERATIONS.index(e['operation']) for e in p['inputs']['observations']])
    ops=np.asarray(ops,dtype=int);codes=ops@np.array([36,6,1])
    artifacts=artifact_groups([packets['packets'][k] for k in keys])
    ls=cfg['development_lineages'];refs=read(base/'evaluator/REFERENCES.json');pos={k:i for i,k in enumerate(keys)}
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
    if not np.isfinite(target).all() or not np.allclose(target.sum(2),1,rtol=0,atol=C.TOL):raise ValueError('native normalization')
    native=np.stack([target@(D.GOALS[:,t,None]==np.arange(3)) for t in range(3)],axis=2)
    old=json.loads(gzip.decompress((base/'parent/goal_decision_points.json.gz').read_bytes()))
    idx={tuple(r[k] for k in D.FIELDS):r for r in old}
    expected=set(product(cfg['tiers'],cfg['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,cfg['training_draws'],cfg['fit_seeds']))
    if len(idx)!=len(old) or set(idx)!=expected:raise ValueError('parent roster')
    write(root/'reader/PACKETS.json',packets);write(root/'evaluator/REFERENCES.json',refs)
    write(root/'ARRAY_SCHEMA.json',dict(axes=['law','weighting','position','goal','artifact','bin'],law=ls,weighting=list(WEIGHTS),artifact=cfg['artifacts'],bin_edges=cfg['bin_edges'],empty_means='NaN iff corresponding weight is exactly zero',sums='joint population mass; group conditioning occurs only by division after aggregation'))
    (root/'groups').mkdir();rows=[];timing=[];parent_error=0.;pool_error=0.;frames=0;parents=0
    weights=(mass,np.full_like(mass,1/len(keys)))
    native_groups=[sufficient(native[li],native[li:li+1],w[li:li+1],artifacts) for li in range(len(ls)) for w in weights]
    if any(np.max(v['grouped_bin_error'])!=0 for v in native_groups):raise ValueError('native calibration')
    np.savez_compressed(root/'evaluator/NATIVE_GROUPS.npz',**{k:np.concatenate([x[k] for x in native_groups],axis=0).reshape(len(ls),2,*native_groups[0][k].shape[1:]) for k in native_groups[0]})
    for draw,seed,budget,arm,forecast in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS,FORECASTS):
        pulse(phase='artifact-conditioned-calibration',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast);tick=time.process_time()
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(base/'decisions'/(stem+'.npz'),allow_pickle=False) as z:
            q=z['probabilities'];d=D.decisions(q)
            if not np.array_equal(z['operations'],codes) or not np.array_equal(z['coordinate'],d['coordinate']) or not np.array_equal(z['marginals'],d['marginals']):raise ValueError('decision identity')
            marg=z['marginals']
        if np.any((q[None]<=0)&(target>0)):raise ValueError('fine support')
        logq=np.zeros_like(q);np.log(q,out=logq,where=q>0);loss=-(target*mass[:,:,None]*logq).sum((1,2))
        chosen=D.GOALS[d['coordinate']];correct=np.take_along_axis(native,np.broadcast_to(chosen[None,:,:,None],(len(ls),len(keys),3,1)),3)[:,:,:,0]
        accuracy=(correct*mass[:,:,None]).sum((1,2))/3
        values=[sufficient(marg,native,w,artifacts) for w in weights]
        np.savez_compressed(root/'groups'/(stem+'.npz'),**{k:np.stack([x[k] for x in values],axis=1) for k in values[0]})
        for li,l in enumerate(ls):
            prior=idx['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]
            error=max(abs(loss[li]-prior['loss']),abs(accuracy[li]-prior['goal_accuracy']));parent_error=max(parent_error,float(error))
            if error>1e-10:raise ValueError('parent score')
            parents+=1
            for wi,weighting in enumerate(WEIGHTS):
                val=values[wi]
                for t,g in product(range(3),range(3)):
                    # Group sums marginalized over artifact reproduce the
                    # ungrouped measure on identical raw probabilities.
                    pooled=C.measure(marg[:,t,g],native[li,:,t,g],weights[wi][li])
                    for k,oldkey in (('bin_weight','bin_weight'),('bin_forecast_sum','bin_confidence_sum'),('bin_correct_sum','bin_correct_sum')):
                        e=float(np.max(abs(val[k][li,t,g].sum(0)-pooled[oldkey])));pool_error=max(pool_error,e)
                        if e>1e-10:raise ValueError('pooled identity')
                    rows.append(dict(tier='E2-full',budget=budget,arm=arm,forecast=forecast,weighting=weighting,step=t,goal=g,lineage=l,draw=draw,seed=seed,**{m:float(val[m][li,t,g]) for m in METRICS}))
        frames+=len(keys);timing.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,cpu_seconds=time.process_time()-tick))
    (root/'goal_artifact_calibration_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'TIMING.jsonl',dict(measurements=timing));write(root/'PARENT_REPRODUCTION.json',dict(passed=True,cells=parents,max_error=parent_error))
    return dict(controls=checks,rows=len(rows),frame_forecasts=frames,packets=len(keys),parent_cells=parents,parent_max_error=parent_error,max_pooled_error=pool_error,**aggregate(rows,cfg),fits=0,scope='public artifact-conditioned calibration; joint group/bin sufficient sums; no historical correspondence')
