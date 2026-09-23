"""Independent scalar collision moments and paired law-multiplicity review."""
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .goal_class_reliability_review import marginals, compare, GOALS
from .joint_factorization_review import close, references
from .witnessed_goal_review import witness, validate

ARMS=('raw-history','frozen-latent','learned-bank','matched-frequency')
FORECASTS=('restricted','goal-product')
WEIGHTS=('native','equal-frame')
METRICS=('squared_error','group_bias','within_law_variation','across_law_variation')


def sufficient(p,target,weight):
    p,target,weight=(np.asarray(v,float) for v in (p,target,weight))
    if p.ndim!=3 or p.shape[1:]!=(3,3) or target.ndim!=4 or target.shape[1:]!=p.shape or weight.shape!=target.shape[:2]:raise ValueError('shape')
    for a in (p,target):
        if not np.isfinite(a).all() or np.any(a<0) or np.any(a>1+1e-12):raise ValueError('probability')
        if any(abs(math.fsum(v)-1)>1e-12 for v in a.reshape(-1,3).tolist()):raise ValueError('simplex')
    ww=weight.tolist()
    if not np.isfinite(weight).all() or np.any(weight<0) or any(abs(math.fsum(w)-1)>1e-12 for w in ww):raise ValueError('weight')
    # Sorted equality classes are independent of the producer's insertion map.
    keys=[tuple(v) for v in p.reshape(len(p),9).tolist()]
    groups={}
    for i in sorted(range(len(p)),key=keys.__getitem__):groups.setdefault(keys[i],[]).append(i)
    members=sorted(groups.values(),key=lambda v:v[0]);ids=np.empty(len(p),np.int64)
    for j,indices in enumerate(members):ids[indices]=j
    laws=len(target);ng=len(members)
    mass=np.zeros((laws,ng));sums=np.zeros((laws,ng,9))
    square=np.zeros((laws,9));within=np.zeros_like(square);loss=np.zeros_like(square)
    for l,w in enumerate(ww):
        mass[l]=[math.fsum(w[i] for i in ix) for ix in members]
        for k in range(9):
            t=target[l].reshape(-1,9)[:,k].tolist();q=p.reshape(-1,9)[:,k].tolist()
            sums[l,:,k]=[math.fsum(w[i]*t[i] for i in ix) for ix in members]
            square[l,k]=math.fsum(w[i]*t[i]*t[i] for i in range(len(p)))
            loss[l,k]=math.fsum(w[i]*(q[i]-t[i])**2 for i in range(len(p)))
            within[l,k]=math.fsum(w[i]*(t[i]-sums[l,j,k]/mass[l,j])**2
                for j,ix in enumerate(members) if mass[l,j]>0 for i in ix)
    total=mass.sum(0);pooled=np.full((ng,9),np.nan)
    for j in range(ng):
        if total[j]>0:
            for k in range(9):pooled[j,k]=math.fsum(sums[:,j,k])/math.fsum(mass[:,j])
    return dict(ids=ids,mass=mass,target_sum=sums,target_square=square,within=within,loss=loss,
        group_target=pooled.reshape(-1,3,3),group_forecast=p[[ix[0] for ix in members]],
        frame_counts=np.array(list(map(len,members))),law_frame_counts=np.array(list(map(len,members)))*laws,
        positive_mass_counts=np.array([sum(w[i]>0 for w in ww for i in ix) for ix in members]))


def resample(s,counts):
    """Normalized law weights; no producer routine or pooled group mean input."""
    counts=np.asarray(counts,float);laws,ng=s['mass'].shape
    if counts.ndim!=2 or counts.shape[1]!=laws or np.any(counts<0) or not np.all(counts.sum(1)==laws):raise ValueError('counts')
    out=np.empty((len(counts),4,9));same=np.array_equal(s['mass'],np.broadcast_to(s['mass'][0],s['mass'].shape))
    if same:
        gram=np.zeros((9,laws,laws))
        for k in range(9):
            a=s['target_sum'][:,:,k];scaled=np.divide(a,s['mass'],out=np.zeros_like(a),where=s['mass']>0)
            gram[k]=scaled@a.T
    for start in range(0,len(counts),96):
        w=counts[start:start+96]/laws
        if same:
            pooled=np.stack([np.sum((w@g)*w,axis=1) for g in gram],axis=1)
        else:
            masses=w@s['mass'];pooled=np.zeros((len(w),9))
            for k in range(9):
                sums=w@s['target_sum'][:,:,k]
                pooled[:,k]=np.divide(sums*sums,masses,out=np.zeros_like(sums),where=masses>0).sum(1)
        variation=w@s['target_square']-pooled;within=w@s['within'];loss=w@s['loss']
        out[start:start+len(w)]=np.stack([loss,loss-variation,within,variation-within],axis=1)
    if not np.isfinite(out).all() or np.min(out)<-1e-12:raise ValueError('decomposition')
    return out.reshape(-1,4,3,3)


def arrays(a,b):
    if set(a)!=set(b):raise ValueError('array fields')
    error=0.
    for k in a:
        x,y=np.asarray(a[k]),np.asarray(b[k])
        if x.shape!=y.shape or not np.array_equal(np.isnan(x),np.isnan(y)):raise ValueError('array shape/undefined')
        if k in ('ids','frame_counts','law_frame_counts','positive_mass_counts','group_forecast'):
            if not np.array_equal(x,y):raise ValueError('exact membership/forecast')
        else:error=max(error,close(x[~np.isnan(x)],y[~np.isnan(y)]))
    return error


def summarize(points,boot,cfg):
    ds,ss,bs=(cfg[k] for k in ('training_draws','fit_seeds','budgets'))
    fields=('draw','seed','budget','arm','forecast','weighting')
    rows={tuple(r[k] for k in fields):r for r in points}
    if len(rows)!=len(points) or set(rows)!=set(product(ds,ss,bs,ARMS,FORECASTS,WEIGHTS)):raise ValueError('roster')
    if len(bs)<2 or any(x>=y for x,y in zip(bs,bs[1:])):raise ValueError('budgets')
    mean=lambda v:math.fsum(v)/len(v)
    def report(v,b):
        return dict(mean=mean(list(v.values())),draw_means=[mean([v[d,s] for s in ss]) for d in ds],
            fit_seed_means=[mean([v[d,s] for d in ds]) for s in ss],low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)))
    estimates=[];contrasts=[];areas=[]
    for f,w,m,t,g in product(FORECASTS,WEIGHTS,range(4),range(3),range(3)):
        ident=dict(forecast=f,weighting=w,metric=METRICS[m],step=t,goal=g)
        def value(b,a):return {(d,s):rows[d,s,b,a,f,w]['values'][m][t][g] for d,s in product(ds,ss)}
        for a,b in product(ARMS,bs):estimates.append(dict(ident,arm=a,budget=b,**report(value(b,a),boot[b,a,f,w][:,m,t,g])))
        for a in (x for x in ARMS if x!='learned-bank'):
            curves={};boots={}
            for b in bs:
                bank,other=value(b,'learned-bank'),value(b,a);curves[b]={k:bank[k]-other[k] for k in bank}
                boots[b]=boot[b,'learned-bank',f,w][:,m,t,g]-boot[b,a,f,w][:,m,t,g]
                contrasts.append(dict(ident,arm='learned-bank',baseline=a,budget=b,**report(curves[b],boots[b])))
            area={k:math.fsum((curves[x][k]+curves[y][k])*.5*math.log(y/x) for x,y in zip(bs,bs[1:]))/math.log(bs[-1]/bs[0]) for k in product(ds,ss)}
            ba=sum((boots[x]+boots[y])*.5*math.log(y/x) for x,y in zip(bs,bs[1:]))/math.log(bs[-1]/bs[0])
            areas.append(dict(ident,arm='learned-bank',baseline=a,**report(area,ba)))
    return dict(estimates=estimates,contrasts=contrasts,normalized_log_budget_area=areas)


def controls():
    p=np.tile([.5,.5,0.],(2,3,1));t=np.array([p.copy(),p.copy()]);t[0,:,:,:2]=[1,0];t[1,:,:,:2]=[0,1]
    s=sufficient(p,t,np.full((2,2),.5));v=resample(s,[[1,1],[2,0]])
    return {'live:opposing_laws':bool(np.allclose(v[0,3,:,:2],.25,rtol=0,atol=1e-14)),
        'positive:resampled_law_targets':bool(np.max(abs(v[1,3]))<1e-14 and np.allclose(v[1,1,:,:2],.25)),
        'placebo:identical_targets':bool(np.max(abs(resample(sufficient(p,p[None],[[.5,.5]]),[[1]])))==0)}


def run(root,plan,pulse):
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    cfg=plan['design'];base=root/'inputs';original=base/'original';parent=base/'parent'
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('plan')
    design=read(original/'PLAN.json')['design'];prior=read(parent/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if design[k]!=prior[k]:raise ValueError('population')
    if design['tiers']!=['E2-full'] or design['bootstrap_seed']!=191022:raise ValueError('design')
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
    native_scalar=np.array([marginals(q) for q in truth])
    with np.load(original/'evaluator/NATIVE_FORECASTS.npz',allow_pickle=False) as z:
        native=z['marginals'];error_native=close(native_scalar,native,1e-12);close(mass,z['weights'])
    schema=read(original/'ARRAY_SCHEMA.json')
    if schema['laws']!=ls or schema['frames']!=keys or schema['metrics']!=list(METRICS):raise ValueError('schema')
    old=json.loads(gzip.decompress((parent/'parent/goal_decision_points.json.gz').read_bytes()))
    fields=('tier','budget','arm','lineage','draw','seed');oi={tuple(r[k] for k in fields):r for r in old}
    expected=set(product(design['tiers'],design['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,design['training_draws'],design['fit_seeds']))
    if len(oi)!=len(old) or set(oi)!=expected:raise ValueError('parent roster')
    raw=json.loads(gzip.decompress((original/'forecast_collision_points.json.gz').read_bytes()));rfields=('draw','seed','budget','arm','forecast','weighting')
    ri={tuple(r[k] for k in rfields):r for r in raw}
    if len(ri)!=len(raw) or set(ri)!=set(product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS,FORECASTS,WEIGHTS)):raise ValueError('point roster')
    sample=np.random.default_rng(design['bootstrap_seed']).integers(len(ls),size=(design['bootstrap_resamples'],len(ls)))
    counts=np.array([np.bincount(v,minlength=len(ls)) for v in sample]);record=read(original/'BOOTSTRAP.json')
    if record!=dict(seed=design['bootstrap_seed'],resamples=design['bootstrap_resamples'],lineages=ls,count_digest=digest(counts.tolist())):raise ValueError('bootstrap')
    boot={};original_boot={};rows=[];errors=dict(native=error_native,marginal=0.,array=0.,parent=0.,point=0.);parents=0
    (root/'reconstructed_groups').mkdir()
    for draw,seed,budget,arm,forecast in product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS,FORECASTS):
        pulse(phase='independent-forecast-collision',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast)
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(parent/'decisions'/(stem+'.npz'),allow_pickle=False) as z:
            q=z['probabilities'];m=z['marginals'];chosen=np.array([[max(range(3),key=lambda k:float(v[k])) for v in row] for row in m])
            errors['marginal']=max(errors['marginal'],close(marginals(q),m,1e-12))
            if not np.array_equal(z['operations'],codes) or not np.array_equal(z['coordinate'],chosen@np.array([9,3,1])):raise ValueError('parent decision')
        for li,l in enumerate(ls):
            terms=[]
            for i,k in zip(*np.nonzero(truth[li])):
                if q[i,k]<=0:raise ValueError('fine support')
                terms.append(-float(mass[li,i])*float(truth[li,i,k])*math.log(float(q[i,k])))
            loss=math.fsum(terms);accuracy=math.fsum(float(mass[li,i])*float(native[li,i,t,chosen[i,t]])/3 for i,t in product(range(len(keys)),range(3)))
            previous=oi['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]
            errors['parent']=max(errors['parent'],close(loss,previous['loss']),close(accuracy,previous['goal_accuracy']));parents+=1
        for weighting,w in zip(WEIGHTS,(mass,np.full_like(mass,1/len(keys)))):
            s=sufficient(m,native,w);v=resample(s,np.ones((1,len(ls)),int))[0];s['values']=v
            with np.load(original/'groups'/(stem+'-'+weighting+'.npz'),allow_pickle=False) as z:saved=dict(z)
            errors['array']=max(errors['array'],arrays(s,saved))
            np.savez_compressed(root/'reconstructed_groups'/(stem+'-'+weighting+'.npz'),**s)
            row=dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,weighting=weighting,groups=len(s['mass'][0]),singleton_frame_groups=int(sum(s['frame_counts']==1)),singleton_law_frame_groups=int(sum(s['law_frame_counts']==1)),values=v.tolist())
            errors['point']=max(errors['point'],compare(row,ri[tuple(row[k] for k in rfields)]));rows.append(row)
            key=budget,arm,forecast,weighting;n=len(design['training_draws'])*len(design['fit_seeds'])
            for dest,data in ((boot,s),(original_boot,saved)):
                b=resample(data,counts)/n
                if key not in dest:dest[key]=np.zeros_like(b)
                dest[key]+=b
    summary=read(original/'SUMMARY.json')
    for k,v in dict(settings=len(rows)//2,group_bundles=len(rows),packets=len(keys),parent_cells=parents).items():
        if summary[k]!=v:raise ValueError('coverage')
    regroup=summarize(rows,boot,design);old_regroup=summarize(raw,original_boot,design)
    expected={k:summary[k] for k in regroup};errors['regroup']=compare(regroup,expected);errors['original_regroup']=compare(old_regroup,expected)
    write(root/'INDEPENDENT_REGROUP.json',regroup);write(root/'ORIGINAL_ROW_REGROUP.json',old_regroup)
    (root/'reconstructed_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    result=dict(passed=True,group_bundles=len(rows),settings=len(rows)//2,packets=len(keys),parent_cells=parents,controls=checks,**{'max_'+k+'_error':v for k,v in errors.items()},scope='independent scalar target moments and centered within-law errors;exact complete membership;parent identities;both original and reconstructed law-multiplicity regrouping;third regroup remains separate;no fit or historical correspondence')
    write(root/'INDEPENDENT_REVIEW.json',result);return result
