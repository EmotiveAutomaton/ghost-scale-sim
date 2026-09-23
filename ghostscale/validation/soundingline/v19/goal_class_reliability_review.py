"""Independent complete-class reliability and one-hot Brier reconstruction."""
from bisect import bisect_right
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .joint_factorization_review import close, references
from .witnessed_goal_review import witness, validate

ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
FORECASTS = ('restricted', 'goal-product')
WEIGHTS = ('native', 'equal-frame')
EDGES = tuple(i/10 for i in range(11))
GOALS = tuple((k//9, k//3 % 3, k % 3) for k in range(27))
FIELDS = ('tier', 'budget', 'arm', 'forecast', 'weighting', 'step', 'goal', 'lineage', 'draw', 'seed')
METRICS = ('signed_error', 'bin_error', 'squared_error', 'brier', 'uncertainty')


def bounded(x):
    x=float(x)
    if not math.isfinite(x) or x < 0 or x > 1+4e-12: raise ValueError('probability bounds')
    return min(x,1.)


def bin_id(x):
    bounded(x)
    return min(bisect_right(EDGES,float(x))-1,9)


def measure(p,c,w):
    p,c,w=(np.asarray(a,dtype=float) for a in (p,c,w))
    if p.ndim!=1 or p.shape!=c.shape or p.shape!=w.shape: raise ValueError('metric shape')
    if not np.isfinite(w).all() or np.any(w<0) or abs(math.fsum(w)-1)>1e-10: raise ValueError('weights')
    pv,cv,wv=p.tolist(),c.tolist(),w.tolist()
    pp=[bounded(x) for x in pv];cc=[bounded(x) for x in cv]
    ids=[bin_id(x) for x in pv]
    by_bin=[[i for i,k in enumerate(ids) if k==b] for b in range(10)]
    bw=[math.fsum(wv[i] for i in group) for group in by_bin]
    bp=[math.fsum(wv[i]*pv[i] for i in group) for group in by_bin]
    bc=[math.fsum(wv[i]*cv[i] for i in group) for group in by_bin]
    clipped=[x!=xx or y!=yy for x,xx,y,yy in zip(pv,pp,cv,cc)]
    return dict(confidence=math.fsum(z*x for x,z in zip(pv,wv)),
        correctness=math.fsum(z*y for y,z in zip(cv,wv)),
        signed_error=math.fsum(z*(x-y) for x,y,z in zip(pv,cv,wv)),
        bin_error=math.fsum(abs(x-y) for x,y in zip(bp,bc)),
        squared_error=math.fsum(z*(x-y)**2 for x,y,z in zip(pp,cc,wv)),
        brier=math.fsum(z*(y*(1-x)**2+(1-y)*x*x) for x,y,z in zip(pp,cc,wv)),
        uncertainty=math.fsum(z*y*(1-y) for y,z in zip(cc,wv)),
        clipped_count=sum(clipped),clipped_mass=math.fsum(z for z,k in zip(wv,clipped) if k),
        bin_weight=bw,bin_confidence_sum=bp,bin_correct_sum=bc,
        bin_confidence=[x/z if z else None for x,z in zip(bp,bw)],
        bin_correctness=[y/z if z else None for y,z in zip(bc,bw)])


def marginals(q):
    q=np.asarray(q)
    if q.ndim!=2 or q.shape[1]!=27 or not np.isfinite(q).all() or np.any(q<0): raise ValueError('fine probabilities')
    for r in q:close(math.fsum(r),1.,4e-12)
    groups=[[k for k,g in enumerate(GOALS) if g[t]==v] for t,v in product(range(3),range(3))]
    return np.array([[math.fsum(float(row[k]) for k in group) for group in groups] for row in q]).reshape(len(q),3,3)


def score_classes(p,c,w):
    p,c,w=(np.asarray(x,dtype=float) for x in (p,c,w))
    if p.ndim!=2 or p.shape!=c.shape or p.shape[1]!=3:raise ValueError('class shape')
    for r in list(p)+list(c):close(math.fsum(r),1.,4e-12)
    rows=[measure(p[:,k],c[:,k],w) for k in range(3)]
    pp=[[bounded(x) for x in r] for r in p]
    cc=[[bounded(x) for x in r] for r in c]
    # Enumerate each one-hot outcome, independent of the producer expansion.
    direct=math.fsum(weight*target*math.fsum((x-(j==k))**2 for j,x in enumerate(pred))
        for pred,native,weight in zip(pp,cc,w.tolist()) for k,target in enumerate(native))
    total=math.fsum(r['brier'] for r in rows)
    uncertainty=math.fsum(r['uncertainty'] for r in rows)
    squared=math.fsum(r['squared_error'] for r in rows)
    error=max(abs(total-direct),abs(total-uncertainty-squared))
    if error>1e-10:raise ValueError('Brier identity')
    return rows,dict(multiclass_brier=total,native_uncertainty=uncertainty,
        squared_forecast_error=squared,direct_brier=direct,identity_error=error,
        normalization_error=max(abs(math.fsum(r)-1) for r in pp+cc))


def controls():
    rows,total=score_classes([[.6,.35,.05]],[[.6,.2,.2]],[1.])
    own,reference=score_classes([[.6,.2,.2]],[[.6,.2,.2]],[1.])
    return {'live:hidden_unselected_error':rows[0]['squared_error']==0 and abs(total['squared_forecast_error']-.045)<1e-14,
        'placebo:native_self_calibrated':all(r['bin_error']==r['squared_error']==0 for r in own),
        'positive:one_hot_expectation':abs(total['direct_brier']-.605)<1e-14,
        'positive:uncertainty':abs(reference['multiclass_brier']-.56)<1e-14,
        'positive:boundary':bin_id(.1)==1 and bin_id(1.)==9}


def compare(a,b,tol=1e-10):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('record fields')
        return max((compare(v,b[k],tol) for k,v in a.items()),default=0.)
    if isinstance(a,(list,tuple)):
        if not isinstance(b,(list,tuple)) or len(a)!=len(b):raise ValueError('list shape')
        return max((compare(x,y,tol) for x,y in zip(a,b)),default=0.)
    if a is None or b is None or isinstance(a,(str,bool)) or isinstance(b,(str,bool)):
        if a!=b:raise ValueError('identity or undefined value')
        return 0.
    return close(a,b,tol)


def regroup(rows,cfg):
    ls,ds,ss,bs=(cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    expected=set(product(cfg['tiers'],bs,ARMS,FORECASTS,WEIGHTS,range(3),range(3),ls,ds,ss))
    idx={tuple(r[k] for k in FIELDS):r for r in rows}
    if len(idx)!=len(rows) or set(idx)!=expected:raise ValueError('stratum roster')
    if len(bs)<2 or any(x>=y for x,y in zip(bs,bs[1:])):raise ValueError('budget order')
    sample=np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls)))
    counts=np.array([np.bincount(s,minlength=len(ls)) for s in sample])
    mean=lambda x:math.fsum(x)/len(x)
    def estimate(v):
        bad=sum(not math.isfinite(x) for x in v.values())
        if bad:return dict(defined=False,nonfinite_pairs=bad)
        line=[mean([v[l,d,s] for d,s in product(ds,ss)]) for l in ls]
        boot=counts@np.array(line)/len(ls)
        return dict(defined=True,nonfinite_pairs=0,mean=mean(line),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),lineage_values=line,
            draw_means=[mean([v[l,d,s] for l,s in product(ls,ss)]) for d in ds],feature_seed_means=[mean([v[l,d,s] for l,d in product(ls,ds)]) for s in ss])
    means=[];contrasts=[];areas=[]
    for tier,forecast,weighting,step,goal in product(cfg['tiers'],FORECASTS,WEIGHTS,range(3),range(3)):
        def row(arm,b,l,d,s):return idx[tier,b,arm,forecast,weighting,step,goal,l,d,s]
        for arm,budget in product(ARMS,bs):
            rr=[row(arm,budget,l,d,s) for l,d,s in product(ls,ds,ss)]
            m={k:mean([r[k] for r in rr]) if all(r[k] is not None for r in rr) else None for k in METRICS+('confidence','correctness','clipped_mass')}
            sums={k:[mean([r[k][i] for r in rr]) for i in range(10)] for k in ('bin_weight','bin_confidence_sum','bin_correct_sum')}
            bw=sums['bin_weight'];bp=sums['bin_confidence_sum'];bc=sums['bin_correct_sum']
            means.append(dict(tier=tier,forecast=forecast,weighting=weighting,step=step,goal=goal,arm=arm,budget=budget,
                pooled_bin_error=math.fsum(abs(x-y) for x,y in zip(bp,bc)),**m,**sums,
                bin_confidence=[x/z if z else None for x,z in zip(bp,bw)],bin_correctness=[y/z if z else None for y,z in zip(bc,bw)]))
        for baseline,metric in product(tuple(a for a in ARMS if a!='learned-bank'),METRICS):
            identity=dict(tier=tier,forecast=forecast,weighting=weighting,step=step,goal=goal,arm='learned-bank',baseline=baseline,metric=metric)
            curves={}
            for budget in bs:
                v={}
                for l,d,s in product(ls,ds,ss):
                    a=row('learned-bank',budget,l,d,s)[metric];b=row(baseline,budget,l,d,s)[metric]
                    v[l,d,s]=a-b if a is not None and b is not None else float('nan')
                curves[budget]=v;contrasts.append(dict(identity,budget=budget,**estimate(v)))
            v={k:math.fsum((curves[x][k]+curves[y][k])*.5*math.log(y/x) for x,y in zip(bs,bs[1:]))/math.log(bs[-1]/bs[0]) for k in product(ls,ds,ss)}
            areas.append(dict(identity,**estimate(v)))
    equal=[]
    mi={(r['tier'],r['forecast'],r['weighting'],r['step'],r['arm'],r['budget'],r['goal']):r for r in means}
    for tier,forecast,weighting,step,arm,budget in product(cfg['tiers'],FORECASTS,WEIGHTS,range(3),ARMS,bs):
        rr=[mi[tier,forecast,weighting,step,arm,budget,g] for g in range(3)]
        equal.append(dict(tier=tier,forecast=forecast,weighting=weighting,step=step,arm=arm,budget=budget,
            **{k:mean([r[k] for r in rr]) for k in METRICS},
            multiclass_brier=math.fsum(r['brier'] for r in rr),native_uncertainty=math.fsum(r['uncertainty'] for r in rr),
            squared_forecast_error=math.fsum(r['squared_error'] for r in rr)))
    return dict(means=means,equal_class_means=equal,contrasts=contrasts,normalized_log_budget_area=areas)


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
    keys=sorted(packets['packets']);ops=[]
    for k in keys:
        validate(packets['packets'][k])
        if digest(packets['packets'][k])!=k:raise ValueError('packet identity')
        ops.append(witness(packets['packets'][k]))
    ls=design['development_lineages'];truth=np.zeros((len(ls),len(keys),27));mass=np.zeros((len(ls),len(keys)))
    for i,(labels,target,weights) in enumerate(references(refs,'E2-full',keys,ls)):
        if any(int(k)%216!=ops[i] for k in labels):raise ValueError('native support')
        truth[:,i,labels//216]=target;mass[:,i]=weights
    scalar_marg=np.array([marginals(q) for q in truth])
    # Recreate this unsaved projection only; every probability is scalar-checked.
    native_marg=np.stack([truth @ (np.array(GOALS)[:,t,None]==np.arange(3)) for t in range(3)],axis=2)
    errors=dict(marginal=close(scalar_marg,native_marg,1e-12),score=0.,parent=0.,saved=0.)
    raw=json.loads(gzip.decompress((original/'goal_class_reliability_points.json.gz').read_bytes()))
    original_groups=regroup(raw,design);idx={tuple(r[k] for k in FIELDS):r for r in raw}
    old=json.loads(gzip.decompress((parent/'parent/goal_decision_points.json.gz').read_bytes()))
    oldfields=('tier','budget','arm','lineage','draw','seed');oi={tuple(r[k] for k in oldfields):r for r in old}
    expected=set(product(design['tiers'],design['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,design['training_draws'],design['fit_seeds']))
    if len(oi)!=len(old) or set(oi)!=expected:raise ValueError('parent roster')
    native=read(original/'evaluator/NATIVE_CLASS_RELIABILITY.json');ni={(r['lineage'],r['weighting'],r['step'],r['goal']):r for r in native}
    if len(ni)!=len(native) or set(ni)!=set(product(ls,WEIGHTS,range(3),range(3))):raise ValueError('native roster')
    uniform=np.full(len(keys),1/len(keys));rows=[];native_rows=[];frames=reproduced=0
    decompositions=read(original/'BRIER_DECOMPOSITION.json')
    df=tuple(k for k in FIELDS if k!='goal')
    di={tuple(r[k] for k in df):r for r in decompositions}
    expected_decompositions={tuple(r[k] for k in df) for r in raw}
    if len(di)!=len(decompositions) or set(di)!=expected_decompositions:raise ValueError('decomposition roster')
    rebuilt_decompositions=[]
    for li,l in enumerate(ls):
        for weighting,step in product(WEIGHTS,range(3)):
            values,_=score_classes(native_marg[li,:,step],native_marg[li,:,step],mass[li] if weighting=='native' else uniform)
            for goal,value in enumerate(values):
                r=dict(lineage=l,weighting=weighting,step=step,goal=goal,**value)
                errors['score']=max(errors['score'],compare(r,ni[l,weighting,step,goal]));native_rows.append(r)
    for draw,seed,budget,arm,forecast in product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS,FORECASTS):
        pulse(phase='independent-goal-class-reliability',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast)
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(parent/'decisions'/(stem+'.npz'),allow_pickle=False) as z:
            q=z['probabilities'];m=z['marginals'];coordinate=z['coordinate'];operations=z['operations']
        if q.shape!=(len(keys),27) or m.shape!=(len(keys),3,3):raise ValueError('forecast shape')
        errors['marginal']=max(errors['marginal'],close(marginals(q),m,1e-12))
        chosen=np.array([[max(range(3),key=lambda k:float(v[k])) for v in row] for row in m])
        if not np.array_equal(operations,ops) or not np.array_equal(coordinate,chosen@np.array([9,3,1])):raise ValueError('parent decision')
        p=np.array([[m[i,t,chosen[i,t]] for t in range(3)] for i in range(len(keys))])
        correct=np.array([[[native_marg[li,i,t,chosen[i,t]] for t in range(3)] for i in range(len(keys))] for li in range(len(ls))])
        with np.load(original/'reports'/(stem+'.npz'),allow_pickle=False) as z:saved={k:z[k] for k in z.files}
        if set(saved)!={'marginals','native_marginals','bin_ids'}:raise ValueError('saved fields')
        errors['saved']=max(errors['saved'],close(saved['marginals'],m,1e-12),close(saved['native_marginals'],native_marg,1e-12))
        # Exact bins and clipping refer to validated saved binary64 values.
        m=saved['marginals'];c=saved['native_marginals']
        expected_bins=np.array([bin_id(x) for x in m.ravel()]).reshape(m.shape)
        if not np.array_equal(saved['bin_ids'],expected_bins):raise ValueError('bin identity')
        # Independent sparse fine-label loss, with complete native support.
        losses=[]
        for li in range(len(ls)):
            terms=[]
            for i,k in zip(*np.nonzero(truth[li])):
                if q[i,k]<=0:raise ValueError('fine support')
                terms.append(-float(mass[li,i])*float(truth[li,i,k])*math.log(float(q[i,k])))
            losses.append(math.fsum(terms))
        for li,l in enumerate(ls):
            accuracy=math.fsum(float(mass[li,i])*float(correct[li,i,t])/3 for i,t in product(range(len(keys)),range(3)))
            before=oi['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]
            errors['parent']=max(errors['parent'],close(losses[li],before['loss']),close(accuracy,before['goal_accuracy']));reproduced+=1
            for weighting,step in product(WEIGHTS,range(3)):
                identity=dict(tier='E2-full',budget=budget,arm=arm,forecast=forecast,weighting=weighting,step=step,lineage=l,draw=draw,seed=seed)
                values,decomposition=score_classes(m[:,step],c[li,:,step],mass[li] if weighting=='native' else uniform)
                dec=dict(identity,**decomposition)
                errors['score']=max(errors['score'],compare(dec,di[tuple(identity[k] for k in df)]))
                rebuilt_decompositions.append(dec)
                for goal,value in enumerate(values):
                    r=dict(identity,goal=goal,fine_loss=losses[li],**value)
                    errors['score']=max(errors['score'],compare(r,idx[tuple(r[k] for k in FIELDS)]));rows.append(r)
        frames+=len(keys)
    summary=read(original/'SUMMARY.json')
    for k,v in dict(rows=len(rows),native_rows=len(native_rows),packets=len(keys),frame_forecasts=frames,parent_cells=reproduced,decomposition_cells=len(rebuilt_decompositions)).items():
        if summary[k]!=v:raise ValueError('coverage')
    close(summary['max_brier_identity_error'],max(r['identity_error'] for r in rebuilt_decompositions))
    write(root/'BRIER_RECONSTRUCTION.json',rebuilt_decompositions)
    groups=regroup(rows,design);expected={k:summary[k] for k in groups}
    errors['regroup']=compare(groups,expected);errors['original_regroup']=compare(original_groups,expected)
    write(root/'INDEPENDENT_REGROUP.json',groups);write(root/'ORIGINAL_ROW_REGROUP.json',original_groups);write(root/'NATIVE_RECONSTRUCTION.json',native_rows)
    (root/'reconstructed_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    result=dict(passed=True,rows=len(rows),native_rows=len(native_rows),packets=len(keys),frame_forecasts=frames,parent_cells=reproduced,decomposition_cells=len(rebuilt_decompositions),controls=checks,
        **{'max_'+k+'_error':v for k,v in errors.items()},scope='scalar goal-class marginals, complete bins, one-hot expectation, Brier decomposition and paired-lineage reconstruction; validated binary64 values define bins and clipping; no historical correspondence')
    write(root/'INDEPENDENT_REVIEW.json',result);return result
