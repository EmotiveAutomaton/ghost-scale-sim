"""Independent scalar absolute residuals, signed-bin sums and paired-law regrouping."""
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
RESOLUTIONS=(5,10,20)
OFFSETS=(0,5,15,35)
EDGES={n:tuple(i/n for i in range(n+1)) for n in RESOLUTIONS}
METRICS=('absolute_error','bin_error_5','bin_error_10','bin_error_20','gap_5','gap_10','gap_20')
INCREMENTS=('gap_5','gap_10','gap_20')
from .goal_bin_resolution_review import bin_id, memberships


def _scalar_close(a,b,tolerance=1e-10):
    """Same absolute finite comparison without allocating two scalar arrays."""
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>tolerance:
        raise ValueError('scalar mismatch')


def bin_sufficient(p,c,w):
    """Independent scalar sums; retain every bin and all conservation checks.

    The earlier checker allocated NumPy arrays for thousands of scalar checks
    per bundle. Local Python floats retain the same sums and absolute tolerances.
    No producer helper or previously accepted aggregate supplies these values.
    """
    p,c,w=(np.asarray(x,float) for x in (p,c,w))
    if p.ndim!=3 or p.shape[1:]!=(3,3) or c.ndim!=4 or c.shape[1:]!=p.shape or w.shape!=c.shape[:2]:raise ValueError('shape')
    for a in (p,c):
        if not np.isfinite(a).all() or np.any(a<0) or np.any(a>1+4e-12):raise ValueError('probability')
        for row in a.reshape(-1,3).tolist():_scalar_close(math.fsum(row),1.,4e-12)
    if not np.isfinite(w).all() or np.any(w<0):raise ValueError('weight')
    weights=w.tolist()
    for row in weights:_scalar_close(math.fsum(row),1.)
    ids=memberships(p)
    if not np.array_equal(ids[0],ids[1]//2) or not np.array_equal(ids[1],ids[2]//2):raise ValueError('nested membership')
    shape=(len(c),3,3,35)
    sums={k:np.zeros(shape) for k in ('bin_weight','bin_forecast_sum','bin_correct_sum')}
    squared=np.zeros(shape[:-1])
    pm=np.full(shape,np.nan);cm=np.full(shape,np.nan)
    errors=[np.zeros(shape[:-1]) for _ in RESOLUTIONS]
    for t,g in product(range(3),range(3)):
        pv=p[:,t,g].tolist()
        groups=[[i for i,code in enumerate(ids[ri,:,t,g]) if code==b] for ri,n in enumerate(RESOLUTIONS) for b in range(n)]
        for li,weights_row in enumerate(weights):
            cv=c[li,:,t,g].tolist();wv=weights_row
            squared[li,t,g]=math.fsum(wv[i]*(pv[i]-cv[i])**2 for i in range(len(p)))
            bw=[math.fsum(wv[i] for i in indices) for indices in groups]
            bp=[math.fsum(wv[i]*pv[i] for i in indices) for indices in groups]
            bc=[math.fsum(wv[i]*cv[i] for i in indices) for indices in groups]
            for name,v in zip(sums,(bw,bp,bc)):sums[name][li,t,g]=v
            for ri,n in enumerate(RESOLUTIONS):
                a,b=OFFSETS[ri:ri+2]
                _scalar_close(math.fsum(bw[a:b]),1.)
                errors[ri][li,t,g]=math.fsum(abs(bp[j]-bc[j]) for j in range(a,b))
                if ri:
                    start=OFFSETS[ri-1]
                    for j in range(n//2):
                        for v in (bw,bp,bc):_scalar_close(v[start+j],math.fsum(v[a+2*j:a+2*j+2]))
            for b in range(35):
                if bw[b]>0:pm[li,t,g,b]=bp[b]/bw[b];cm[li,t,g,b]=bc[b]/bw[b]
    increments=[errors[1]-errors[0],errors[2]-errors[1]]
    if min(np.min(v) for v in increments)<-1e-12:raise ValueError('refinement monotonicity')
    return dict(sums,bin_forecast_mean=pm,bin_correct_mean=cm,
        **dict(zip(('bin_error_5','bin_error_10','bin_error_20','increment_10_5','increment_20_10','squared_error'),[*errors,*increments,squared])))


def sufficient(p,c,w):
    """Scalar residuals and absolute sums, checked against signed-bin sums."""
    base=bin_sufficient(p,c,w)
    p,c,w=(np.asarray(x,float) for x in (p,c,w))
    ids=memberships(p)
    residual=np.empty_like(c);contribution=np.empty_like(c)
    absolute=np.zeros((len(c),3,3))
    weight_rows=w.tolist()
    for t,g in product(range(3),range(3)):
        pv=p[:,t,g].tolist()
        groups=[[i for i,code in enumerate(ids[ri,:,t,g]) if code==b] for ri,n in enumerate(RESOLUTIONS) for b in range(n)]
        for li in range(len(c)):
            cv=c[li,:,t,g].tolist();wv=weight_rows[li]
            e=[pv[i]-cv[i] for i in range(len(p))]
            weighted=[wv[i]*abs(e[i]) for i in range(len(p))]
            residual[li,:,t,g]=e;contribution[li,:,t,g]=weighted
            absolute[li,t,g]=math.fsum(weighted)
            for ri,n in enumerate(RESOLUTIONS):
                a,b=OFFSETS[ri:ri+2]
                signed=math.fsum(abs(math.fsum(wv[i]*e[i] for i in indices)) for indices in groups[a:b])
                _scalar_close(signed,float(base[f'bin_error_{n}'][li,t,g]),1e-12)
    gaps={f'gap_{n}':absolute-base[f'bin_error_{n}'] for n in RESOLUTIONS}
    if min(np.min(v) for v in gaps.values()) < -1e-12:raise ValueError('triangle inequality')
    if np.min(gaps['gap_5']-gaps['gap_10']) < -1e-12 or np.min(gaps['gap_10']-gaps['gap_20']) < -1e-12:raise ValueError('envelope nesting')
    return dict(base,absolute_error=absolute,**gaps,frame_residual=residual,frame_absolute_contribution=contribution)


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
    x=sufficient(p,c,np.array([[.5,.5]]));y=sufficient(p,c,np.array([[.25,.75]]))
    same=p[None].copy();same[0,:,:,:2]+=[.05,-.05]
    constant=sufficient(p,same,np.array([[.5,.5]]));own=sufficient(p,p[None],np.array([[.5,.5]]))
    return {'live:opposing_residual_envelope':bool(all(np.allclose(x[m][...,:2],.05,rtol=0,atol=1e-14) for m in INCREMENTS)),
        'positive:unequal_weights':bool(np.allclose(y['gap_20'][...,:2],.025,rtol=0,atol=1e-14)),
        'positive:constant_error_tight_bound':bool(all(np.max(abs(constant[m]))<1e-14 for m in INCREMENTS)),
        'placebo:native_self':bool(all(np.max(own[m])==0 for m in METRICS))}


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
            for metric in INCREMENTS:
                gaps.append(dict(ident,arm=arm,budget=budget,metric=metric,**estimate(values(arm,budget,metric))))
        for baseline,metric in product(tuple(a for a in ARMS if a!='learned-bank'),METRICS):
            identity=dict(ident,arm='learned-bank',baseline=baseline,metric=metric);curves={}
            for b in bs:
                a=values('learned-bank',b,metric);r=values(baseline,b,metric);v={k:a[k]-r[k] for k in a};curves[b]=v
                contrasts.append(dict(identity,budget=b,**estimate(v)))
            area={k:math.fsum((curves[x][k]+curves[y][k])*.5*math.log(y/x) for x,y in zip(bs,bs[1:]))/math.log(bs[-1]/bs[0]) for k in product(ls,ds,ss)}
            areas.append(dict(identity,**estimate(area)))
    return dict(means=means,envelope_estimates=gaps,contrasts=contrasts,normalized_log_budget_area=areas)


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
    if design['tiers']!=['E2-full'] or design['bin_edges']!={str(n):list(EDGES[n]) for n in RESOLUTIONS} or design['resolutions']!=list(RESOLUTIONS):raise ValueError('design')
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
    errors=dict(marginal=close(scalar_marg,native,1e-12),array=0.,residual=0.,score=0.,parent=0.,accepted_bin=0.)
    raw=json.loads(gzip.decompress((original/'goal_absolute_envelope_points.json.gz').read_bytes()))
    original_groups=regroup(raw,design);idx={tuple(r[k] for k in FIELDS):r for r in raw}
    old=json.loads(gzip.decompress((parent/'parent/goal_decision_points.json.gz').read_bytes()))
    fields=('tier','budget','arm','lineage','draw','seed');oi={tuple(r[k] for k in fields):r for r in old}
    expected=set(product(design['tiers'],design['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,design['training_draws'],design['fit_seeds']))
    if len(oi)!=len(old) or set(oi)!=expected:raise ValueError('parent roster')
    binrows=json.loads(gzip.decompress((parent/'parent_bin/goal_bin_resolution_points.json.gz').read_bytes()))
    bins_idx={tuple(r[k] for k in FIELDS):r for r in binrows}
    if len(bins_idx)!=len(binrows) or set(bins_idx)!=set(idx):raise ValueError('accepted bin roster')
    weights=(mass,np.full_like(mass,1/len(keys)))
    native_groups=[sufficient(native[li],native[li:li+1],w[li:li+1]) for li in range(len(ls)) for w in weights]
    ng={k:np.concatenate([x[k] for x in native_groups],axis=0).reshape(len(ls),2,*native_groups[0][k].shape[1:]) for k in native_groups[0] if not k.startswith('frame_')}
    with np.load(original/'evaluator/NATIVE_GROUPS.npz',allow_pickle=False) as z:errors['array']=compare_arrays(ng,dict(z))
    if np.max(ng['bin_error_20'])!=0:raise ValueError('native calibration')
    with np.load(original/'evaluator/NATIVE_FORECASTS.npz',allow_pickle=False) as z:errors['array']=max(errors['array'],compare_arrays(dict(marginals=native,weights=mass),dict(z)))
    np.savez_compressed(root/'NATIVE_RECONSTRUCTION.npz',**ng)
    schema=read(original/'ARRAY_SCHEMA.json')
    if schema['axes']!=['law','weighting','position','goal','packed_bin'] or schema['law']!=ls or schema['weighting']!=list(WEIGHTS) or schema['resolutions']!=list(RESOLUTIONS) or schema['offsets']!=list(OFFSETS) or schema['bin_edges']!={str(n):list(EDGES[n]) for n in RESOLUTIONS}:raise ValueError('array schema')
    if schema['residual_axes']!=['law','frame','position','goal'] or schema['contribution_axes']!=['law','weighting','frame','position','goal'] or schema['frame']!=keys:raise ValueError('residual schema')
    rows=[];frames=parents=empty_bins=0;(root/'reconstructed_groups').mkdir()
    for draw,seed,budget,arm,forecast in product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS,FORECASTS):
        pulse(phase='independent-absolute-envelope',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast)
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
        rebuilt={k:np.stack([x[k] for x in values],axis=1) for k in values[0] if not k.startswith('frame_')}
        with np.load(original/'groups'/(stem+'.npz'),allow_pickle=False) as z:errors['array']=max(errors['array'],compare_arrays(rebuilt,dict(z)))
        raw_rebuilt=dict(signed=values[0]['frame_residual'],absolute_contribution=np.stack([x['frame_absolute_contribution'] for x in values],axis=1))
        with np.load(original/'residuals'/(stem+'.npz'),allow_pickle=False) as z:errors['residual']=max(errors['residual'],compare_arrays(raw_rebuilt,dict(z)))
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
                br=bins_idx[tuple(r[k] for k in FIELDS)]
                for n in RESOLUTIONS:errors['accepted_bin']=max(errors['accepted_bin'],close(r[f'bin_error_{n}'],br[f'bin_error_{n}']))
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
        **{'max_'+k+'_error':v for k,v in errors.items()},scope='scalar absolute residuals, signed-bin sums, all raw contributions, accepted bin scores, empty means and envelope gaps, native self, parent identities and paired-law estimates; no new fit or historical correspondence')
    write(root/'INDEPENDENT_REVIEW.json',result);return result
