"""Independent scalar residual moments, centered variance and paired-law regrouping."""
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .goal_class_reliability_review import marginals, compare, GOALS, FIELDS
from .joint_factorization_review import close, references
from .witnessed_goal_review import witness, validate, OPERATIONS

ARMS=('raw-history','frozen-latent','learned-bank','matched-frequency')
FORECASTS=('restricted','goal-product')
WEIGHTS=('native','equal-frame')
EDGES=tuple(i/10 for i in range(11))
METRICS=('squared_bias','within_variance','squared_error')
COMPONENTS=('squared_bias','within_variance')


def bin_id(p):
    if not math.isfinite(p) or p<0 or p>1+4e-12:raise ValueError('probability')
    return sum(p>=edge for edge in EDGES[1:-1])


def memberships(p):
    p=np.asarray(p,float)
    return np.array([bin_id(float(x)) for x in p.ravel()],dtype=int).reshape(p.shape)


def sufficient(p,c,w):
    """Accurate scalar sums and centered variance, without moment subtraction."""
    p,c,w=(np.asarray(x,float) for x in (p,c,w))
    if p.ndim!=3 or p.shape[1:]!=(3,3) or c.ndim!=4 or c.shape[1:]!=p.shape or w.shape!=c.shape[:2]:raise ValueError('shape')
    for a in (p,c):
        if not np.isfinite(a).all() or np.any(a<0) or np.any(a>1+4e-12):raise ValueError('probability')
        for row in a.reshape(-1,3):close(math.fsum(row),1.,4e-12)
    if not np.isfinite(w).all() or np.any(w<0):raise ValueError('weight')
    for row in w:close(math.fsum(row),1.)
    ids=memberships(p);shape=(len(c),3,3,10)
    sums={k:np.zeros(shape) for k in ('bin_weight','bin_residual_sum','bin_residual_squared_sum','bin_bias_mass','bin_variance_mass')}
    moments={k:np.full(shape,np.nan) for k in ('bin_residual_mean','bin_residual_second_moment','bin_residual_variance')}
    metrics={k:np.zeros(shape[:-1]) for k in METRICS}
    for t,g in product(range(3),range(3)):
        groups=[[i for i,code in enumerate(ids[:,t,g]) if code==b] for b in range(10)]
        for li in range(len(c)):
            residual=[float(p[i,t,g])-float(c[li,i,t,g]) for i in range(len(p))];wv=w[li].tolist()
            for b,indices in enumerate(groups):
                key=li,t,g,b
                mass=math.fsum(wv[i] for i in indices)
                first=math.fsum(wv[i]*residual[i] for i in indices)
                second=math.fsum(wv[i]*residual[i]**2 for i in indices)
                sums['bin_weight'][key]=mass;sums['bin_residual_sum'][key]=first;sums['bin_residual_squared_sum'][key]=second
                if mass:
                    mean=first/mass
                    centered=math.fsum(wv[i]*(residual[i]-mean)**2 for i in indices)
                    moments['bin_residual_mean'][key]=mean
                    moments['bin_residual_second_moment'][key]=second/mass
                    moments['bin_residual_variance'][key]=centered/mass
                    sums['bin_bias_mass'][key]=mass*mean**2
                    sums['bin_variance_mass'][key]=centered
                    close(second,sums['bin_bias_mass'][key]+centered,1e-12)
            close(math.fsum(sums['bin_weight'][li,t,g]),1.)
            metrics['squared_bias'][li,t,g]=math.fsum(sums['bin_bias_mass'][li,t,g])
            metrics['within_variance'][li,t,g]=math.fsum(sums['bin_variance_mass'][li,t,g])
            metrics['squared_error'][li,t,g]=math.fsum(wv[i]*residual[i]**2 for i in range(len(p)))
            close(metrics['squared_bias'][li,t,g]+metrics['within_variance'][li,t,g],metrics['squared_error'][li,t,g],1e-12)
    return dict(sums,**moments,**metrics)


def compare_arrays(a,b):
    if set(a)!=set(b):raise ValueError('array fields')
    error=0.
    for k,x in a.items():
        x,y=np.asarray(x),np.asarray(b[k])
        if x.shape!=y.shape or not np.array_equal(np.isnan(x),np.isnan(y)):raise ValueError('shape or empty mean')
        mask=~np.isnan(x);error=max(error,close(x[mask],y[mask]))
    return error


def controls():
    p=np.tile([.15,.85,0.],(2,3,1));c=p[None].copy()
    c[0,0,:,:2]+=[.05,-.05];c[0,1,:,:2]+=[-.05,.05]
    x=sufficient(p,c,np.array([[.5,.5]]));y=sufficient(p,c,np.array([[.25,.75]]));own=sufficient(p,p[None],np.array([[.5,.5]]))
    return {'live:opposing_residual_variance':bool(np.max(x['squared_bias'])<1e-28 and np.allclose(x['within_variance'][...,:2],.0025,rtol=0,atol=1e-14)),
        'positive:unequal_weights':bool(np.allclose(y['squared_bias'][...,:2],.000625,rtol=0,atol=1e-14) and np.allclose(y['within_variance'][...,:2],.001875,rtol=0,atol=1e-14)),
        'placebo:native_self':bool(all(np.max(own[m])==0 for m in METRICS)),
        'positive:empty_means':bool(all(np.isnan(x[k][x['bin_weight']==0]).all() for k in ('bin_residual_mean','bin_residual_second_moment','bin_residual_variance')))}


def regroup(rows,cfg):
    ls,ds,ss,bs=(cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    expected=set(product(cfg['tiers'],bs,ARMS,FORECASTS,WEIGHTS,range(3),range(3),ls,ds,ss))
    idx={tuple(r[k] for k in FIELDS):r for r in rows}
    if len(idx)!=len(rows) or set(idx)!=expected:raise ValueError('stratum roster')
    if len(bs)<2 or any(x>=y for x,y in zip(bs,bs[1:])):raise ValueError('budget order')
    sample=np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls)))
    counts=np.array([np.bincount(s,minlength=len(ls)) for s in sample])
    mean=lambda v:math.fsum(v)/len(v)
    def estimate(v):
        if not all(math.isfinite(x) for x in v.values()):raise ValueError('nonfinite estimate')
        line=[mean([v[l,d,s] for d,s in product(ds,ss)]) for l in ls]
        boot=counts@np.array(line)/len(ls)
        return dict(mean=mean(line),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),lineage_values=line,
            draw_means=[mean([v[l,d,s] for l,s in product(ls,ss)]) for d in ds],feature_seed_means=[mean([v[l,d,s] for l,d in product(ls,ds)]) for s in ss])
    means=[];contrasts=[];areas=[];gaps=[]
    for tier,forecast,weighting,step,goal in product(cfg['tiers'],FORECASTS,WEIGHTS,range(3),range(3)):
        def values(arm,b,m):return {(l,d,s):idx[tier,b,arm,forecast,weighting,step,goal,l,d,s][m] for l,d,s in product(ls,ds,ss)}
        ident=dict(tier=tier,forecast=forecast,weighting=weighting,step=step,goal=goal)
        for arm,budget in product(ARMS,bs):
            means.append(dict(ident,arm=arm,budget=budget,**{m:mean(list(values(arm,budget,m).values())) for m in METRICS}))
            for metric in COMPONENTS:
                gaps.append(dict(ident,arm=arm,budget=budget,metric=metric,**estimate(values(arm,budget,metric))))
        for baseline,metric in product(tuple(a for a in ARMS if a!='learned-bank'),METRICS):
            identity=dict(ident,arm='learned-bank',baseline=baseline,metric=metric);curves={}
            for b in bs:
                a=values('learned-bank',b,metric);r=values(baseline,b,metric);v={k:a[k]-r[k] for k in a};curves[b]=v
                contrasts.append(dict(identity,budget=b,**estimate(v)))
            area={k:math.fsum((curves[x][k]+curves[y][k])*.5*math.log(y/x) for x,y in zip(bs,bs[1:]))/math.log(bs[-1]/bs[0]) for k in product(ls,ds,ss)}
            areas.append(dict(identity,**estimate(area)))
    return dict(means=means,component_estimates=gaps,contrasts=contrasts,normalized_log_budget_area=areas)


def run(root,plan,pulse):
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    cfg=plan['design'];base=root/'inputs';original=base/'original';parent=base/'parent'
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target plan')
    design=read(original/'PLAN.json')['design'];prior=read(parent/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if design[k]!=prior[k]:raise ValueError('population')
    if design['tiers']!=['E2-full'] or design['bin_edges']!=list(EDGES):raise ValueError('design')
    packets=read(original/'reader/PACKETS.json');refs=read(original/'evaluator/REFERENCES.json')
    if packets!=read(parent/'reader/PACKETS.json') or refs!=read(parent/'evaluator/REFERENCES.json'):raise ValueError('evidence changed')
    keys=sorted(packets['packets']);codes=[]
    for k in keys:
        validate(packets['packets'][k])
        if digest(packets['packets'][k])!=k:raise ValueError('packet identity')
        codes.append(witness(packets['packets'][k]))
    ls=design['development_lineages'];truth=np.zeros((len(ls),len(keys),27));mass=np.zeros((len(ls),len(keys)))
    for i,(labels,target,weights) in enumerate(references(refs,'E2-full',keys,ls)):
        if any(int(k)%216!=codes[i] for k in labels):raise ValueError('native support')
        truth[:,i,labels//216]=target;mass[:,i]=weights
    scalar_marg=np.array([marginals(q) for q in truth])
    # Producer did not save this projection: reconstruct its binary64 values,
    # then verify every entry independently before exact bin assignments.
    native=np.stack([truth@(np.array(GOALS)[:,t,None]==np.arange(3)) for t in range(3)],axis=2)
    errors=dict(marginal=close(scalar_marg,native,1e-12),array=0.,score=0.,parent=0.)
    raw=json.loads(gzip.decompress((original/'goal_squared_decomposition_points.json.gz').read_bytes()))
    original_groups=regroup(raw,design);idx={tuple(r[k] for k in FIELDS):r for r in raw}
    old=json.loads(gzip.decompress((parent/'parent/goal_decision_points.json.gz').read_bytes()))
    fields=('tier','budget','arm','lineage','draw','seed');oi={tuple(r[k] for k in fields):r for r in old}
    expected=set(product(design['tiers'],design['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,design['training_draws'],design['fit_seeds']))
    if len(oi)!=len(old) or set(oi)!=expected:raise ValueError('parent roster')
    weights=(mass,np.full_like(mass,1/len(keys)))
    native_groups=[sufficient(native[li],native[li:li+1],w[li:li+1]) for li in range(len(ls)) for w in weights]
    ng={k:np.concatenate([x[k] for x in native_groups],axis=0).reshape(len(ls),2,*native_groups[0][k].shape[1:]) for k in native_groups[0]}
    with np.load(original/'evaluator/NATIVE_GROUPS.npz',allow_pickle=False) as z:errors['array']=compare_arrays(ng,dict(z))
    if np.max(ng['squared_error'])!=0:raise ValueError('native calibration')
    with np.load(original/'evaluator/NATIVE_FORECASTS.npz',allow_pickle=False) as z:errors['array']=max(errors['array'],compare_arrays(dict(marginals=native,weights=mass),dict(z)))
    np.savez_compressed(root/'NATIVE_RECONSTRUCTION.npz',**ng)
    schema=read(original/'ARRAY_SCHEMA.json')
    if schema['axes']!=['law','weighting','position','goal','bin'] or schema['law']!=ls or schema['weighting']!=list(WEIGHTS) or schema['bin_edges']!=list(EDGES):raise ValueError('array schema')
    rows=[];frames=parents=empty_bins=0;(root/'reconstructed_groups').mkdir()
    for draw,seed,budget,arm,forecast in product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS,FORECASTS):
        pulse(phase='independent-squared-decomposition',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast)
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(parent/'decisions'/(stem+'.npz'),allow_pickle=False) as z:
            q=z['probabilities'];m=z['marginals'];coordinate=z['coordinate'];operations=z['operations']
        if q.shape!=(len(keys),27) or m.shape!=(len(keys),3,3):raise ValueError('forecast shape')
        errors['marginal']=max(errors['marginal'],close(marginals(q),m,1e-12))
        chosen=np.array([[max(range(3),key=lambda k:float(v[k])) for v in row] for row in m])
        if not np.array_equal(operations,codes) or not np.array_equal(coordinate,chosen@np.array([9,3,1])):raise ValueError('parent decision')
        with np.load(original/'forecasts'/(stem+'.npz'),allow_pickle=False) as z:
            if set(z.files)!={'marginals','bin_ids'} or not np.array_equal(z['marginals'],m) or not np.array_equal(z['bin_ids'],memberships(m)):raise ValueError('saved forecast or bin membership')
        values=[sufficient(m,native,w) for w in weights]
        rebuilt={k:np.stack([x[k] for x in values],axis=1) for k in values[0]}
        with np.load(original/'groups'/(stem+'.npz'),allow_pickle=False) as z:errors['array']=max(errors['array'],compare_arrays(rebuilt,dict(z)))
        np.savez_compressed(root/'reconstructed_groups'/(stem+'.npz'),**rebuilt)
        empty_bins+=int(np.sum(rebuilt['bin_weight']==0))
        for li,l in enumerate(ls):
            terms=[]
            for i,k in zip(*np.nonzero(truth[li])):
                if q[i,k]<=0:raise ValueError('fine support')
                terms.append(-float(mass[li,i])*float(truth[li,i,k])*math.log(float(q[i,k])))
            loss=math.fsum(terms)
            accuracy=math.fsum(float(mass[li,i])*float(native[li,i,t,chosen[i,t]])/3 for i,t in product(range(len(keys)),range(3)))
            prior=oi['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]
            errors['parent']=max(errors['parent'],close(loss,prior['loss']),close(accuracy,prior['goal_accuracy']));parents+=1
            for wi,weighting,t,g in product(range(2),WEIGHTS,range(3),range(3)):
                if weighting!=WEIGHTS[wi]:continue
                r=dict(tier='E2-full',budget=budget,arm=arm,forecast=forecast,weighting=weighting,step=t,goal=g,lineage=l,draw=draw,seed=seed,**{k:float(values[wi][k][li,t,g]) for k in METRICS})
                errors['score']=max(errors['score'],compare(r,idx[tuple(r[k] for k in FIELDS)]));rows.append(r)
        frames+=len(keys)
    summary=read(original/'SUMMARY.json')
    for k,v in dict(rows=len(rows),packets=len(keys),frame_forecasts=frames,parent_cells=parents).items():
        if summary[k]!=v:raise ValueError('coverage')
    groups=regroup(rows,design);expected={k:summary[k] for k in groups}
    errors['regroup']=compare(groups,expected);errors['original_regroup']=compare(original_groups,expected)
    write(root/'INDEPENDENT_REGROUP.json',groups);write(root/'ORIGINAL_ROW_REGROUP.json',original_groups)
    (root/'reconstructed_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    result=dict(passed=True,rows=len(rows),packets=len(keys),frame_forecasts=frames,parent_cells=parents,group_bundles=frames//len(keys),empty_bins=empty_bins,controls=checks,
        **{'max_'+k+'_error':v for k,v in errors.items()},scope='scalar ten-bin memberships, residual moments, centered variance, empty means, unbinned errors, native self, parent identities and paired-law estimates; no new fit or historical correspondence')
    write(root/'INDEPENDENT_REVIEW.json',result);return result
