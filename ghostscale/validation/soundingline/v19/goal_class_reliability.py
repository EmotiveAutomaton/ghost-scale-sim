"""Complete class-wise reliability of frozen goal probabilities."""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import goal_decision as D

ARMS = D.ARMS
FORECASTS = D.FORECASTS
WEIGHTS = ('native', 'equal-frame')
EDGES = np.arange(11, dtype=float)/10
TOL = 4e-12
FIELDS = ('tier', 'budget', 'arm', 'forecast', 'weighting', 'step', 'goal', 'lineage', 'draw', 'seed')
METRICS = ('signed_error', 'bin_error', 'squared_error', 'brier', 'uncertainty')


def probabilities(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all() or np.any(x<0) or np.any(x>1+TOL):raise ValueError('probability bounds')
    return np.minimum(x,1.)

def bins(x):
    probabilities(x)
    return np.minimum(np.searchsorted(EDGES,x,side='right')-1,9)

def measure(p,c,w):
    p,c,w=(np.asarray(x,float) for x in (p,c,w))
    if p.ndim!=1 or p.shape!=c.shape or p.shape!=w.shape:raise ValueError('metric shape')
    pp,cc=probabilities(p),probabilities(c)
    if not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-1)>1e-10:raise ValueError('weights')
    codes=bins(p);bw=np.bincount(codes,weights=w,minlength=10)
    bp=np.bincount(codes,weights=w*p,minlength=10);bc=np.bincount(codes,weights=w*c,minlength=10)
    return dict(confidence=float(w@p),correctness=float(w@c),signed_error=float(w@(p-c)),
        bin_error=float(np.abs(bp-bc).sum()),squared_error=float(w@((pp-cc)**2)),
        brier=float(w@(cc*(1-pp)**2+(1-cc)*pp**2)),uncertainty=float(w@(cc*(1-cc))),
        clipped_count=int(np.count_nonzero((p!=pp)|(c!=cc))),clipped_mass=float(w@((p!=pp)|(c!=cc))),
        bin_weight=bw.tolist(),bin_confidence_sum=bp.tolist(),bin_correct_sum=bc.tolist(),
        bin_confidence=[float(bp[i]/bw[i]) if bw[i]>0 else None for i in range(10)],
        bin_correctness=[float(bc[i]/bw[i]) if bw[i]>0 else None for i in range(10)])

def score_classes(p,c,w):
    p,c=(np.asarray(x,float) for x in (p,c));w=np.asarray(w,float)
    if p.ndim!=2 or p.shape!=c.shape or p.shape[1]!=3:raise ValueError('class shape')
    pp,cc=probabilities(p),probabilities(c)
    if not np.allclose(p.sum(1),1,atol=TOL,rtol=0) or not np.allclose(c.sum(1),1,atol=TOL,rtol=0):raise ValueError('class conservation')
    rows=[measure(p[:,g],c[:,g],w) for g in range(3)]
    # Exact expectation of squared distance to each one-hot outcome; scalar-class
    # sum and uncertainty decomposition are independent algebraic forms.
    direct=float(w@(cc*(1-2*pp+(pp**2).sum(1)[:,None])).sum(1))
    summed=sum(r['brier'] for r in rows)
    decomposition=sum(r['uncertainty']+r['squared_error'] for r in rows)
    error=max(abs(summed-direct),abs(summed-decomposition))
    if error>1e-10:raise ValueError('Brier identity')
    return rows,dict(multiclass_brier=summed,native_uncertainty=sum(r['uncertainty'] for r in rows),
        squared_forecast_error=sum(r['squared_error'] for r in rows),direct_brier=direct,
        identity_error=error,normalization_error=float(max(np.max(abs(pp.sum(1)-1)),np.max(abs(cc.sum(1)-1)))))

def controls():
    p=np.array([[.6,.35,.05]]);c=np.array([[.6,.2,.2]])
    rows,total=score_classes(p,c,[1]);own,_=score_classes(c,c,[1])
    return {'live:hidden_unselected_error':rows[0]['squared_error']==0 and total['squared_forecast_error']>.04,
        'placebo:native_self_calibrated':all(r['bin_error']==r['squared_error']==0 for r in own),
        'positive:uncertainty_decomposition':total['identity_error']<1e-14,
        'positive:edges':bins(np.array([0.,.1,1.])).tolist()==[0,1,9]}

def aggregate(rows, cfg):
    ls, ds, ss, bs=(cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    expected=set(product(cfg['tiers'],bs,ARMS,FORECASTS,WEIGHTS,range(3),range(3),ls,ds,ss))
    idx={tuple(r[k] for k in FIELDS):r for r in rows}
    if len(idx)!=len(rows) or set(idx)!=expected: raise ValueError('stratum roster')
    sample=np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls)))
    def estimate(v):
        n=int((~np.isfinite(v)).sum())
        if n: return dict(defined=False,nonfinite_pairs=n)
        line=v.mean((1,2)); boot=line[sample].mean(1)
        return dict(defined=True,nonfinite_pairs=0,mean=float(line.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),lineage_values=line.tolist(),draw_means=v.mean((0,2)).tolist(),feature_seed_means=v.mean((0,1)).tolist())
    means=[];contrasts=[];areas=[]
    for tier,forecast,weighting,step,goal in product(cfg['tiers'],FORECASTS,WEIGHTS,range(3),range(3)):
        def records(arm,budget):return [idx[tier,budget,arm,forecast,weighting,step,goal,l,d,s] for l,d,s in product(ls,ds,ss)]
        def values(arm,budget,metric):return np.array([r[metric] if r[metric] is not None else np.nan for r in records(arm,budget)]).reshape(len(ls),len(ds),len(ss))
        for arm,budget in product(ARMS,bs):
            rr=records(arm,budget)
            m={k:float(np.mean([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in METRICS+('confidence','correctness','clipped_mass')}
            sums={k:np.asarray([r[k] for r in rr]).mean(0) for k in ('bin_weight','bin_confidence_sum','bin_correct_sum')}
            bw=sums['bin_weight'];bp=sums['bin_confidence_sum'];bc=sums['bin_correct_sum']
            means.append(dict(tier=tier,forecast=forecast,weighting=weighting,step=step,goal=goal,arm=arm,budget=budget,
                pooled_bin_error=float(np.abs(bp-bc).sum()),
                **m,**{k:v.tolist() for k,v in sums.items()},
                bin_confidence=[float(bp[i]/bw[i]) if bw[i] else None for i in range(10)],
                bin_correctness=[float(bc[i]/bw[i]) if bw[i] else None for i in range(10)]))
        for baseline,metric in product(tuple(a for a in ARMS if a!='learned-bank'),METRICS):
            identity=dict(tier=tier,forecast=forecast,weighting=weighting,step=step,goal=goal,arm='learned-bank',baseline=baseline,metric=metric)
            curves=[]
            for budget in bs:
                v=values('learned-bank',budget,metric)-values(baseline,budget,metric)
                curves.append(v);contrasts.append(dict(identity,budget=budget,**estimate(v)))
            area=np.trapezoid(curves,x=np.log(bs),axis=0)/np.log(bs[-1]/bs[0])
            areas.append(dict(identity,**estimate(area)))
    equal=[]
    for tier,forecast,weighting,step,arm,budget in product(cfg['tiers'],FORECASTS,WEIGHTS,range(3),ARMS,bs):
        rr=[r for r in means if all(r[k]==v for k,v in dict(tier=tier,forecast=forecast,weighting=weighting,step=step,arm=arm,budget=budget).items())]
        if len(rr)!=3:raise ValueError('equal class roster')
        equal.append(dict(tier=tier,forecast=forecast,weighting=weighting,step=step,arm=arm,budget=budget,
            **{k:float(np.mean([r[k] for r in rr])) for k in METRICS},
            multiclass_brier=sum(r['brier'] for r in rr),native_uncertainty=sum(r['uncertainty'] for r in rr),squared_forecast_error=sum(r['squared_error'] for r in rr)))
    return dict(means=means,equal_class_means=equal,contrasts=contrasts,normalized_log_budget_area=areas)


def run(root,plan,pulse):
    cfg=plan['design'];base=root/'inputs';checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    parent=read(base/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if cfg[k]!=parent[k]:raise ValueError('population')
    if cfg['tiers']!=['E2-full'] or cfg['bin_edges']!=EDGES.tolist():raise ValueError('design')
    packets=read(base/'reader/PACKETS.json');keys=sorted(packets['packets']);ops=[]
    for k in keys:
        p=packets['packets'][k];D.S.validate(p)
        if digest(p)!=k or p['tier']!='E2-full':raise ValueError('packet')
        ops.append(sum(D.S.L.OPERATIONS.index(e['operation'])*6**(2-t) for t,e in enumerate(p['inputs']['observations'])))
    ls=cfg['development_lineages'];refs=read(base/'evaluator/REFERENCES.json');pos={k:i for i,k in enumerate(keys)}
    if len(refs)!=len(ls) or {r['lineage'] for r in refs}!=set(ls):raise ValueError('lineage roster')
    targets=np.zeros((len(ls),len(keys),27));mass=np.zeros((len(ls),len(keys)))
    for r in refs:
        li=ls.index(r['lineage'])
        if r['tier']!='E2-full' or len(r['frames'])!=len(keys) or {f['frame'] for f in r['frames']}!=set(keys):raise ValueError('frame roster')
        for f in r['frames']:
            i=pos[f['frame']];mass[li,i]=f['mass']
            if len(dict(f['target']))!=len(f['target']):raise ValueError('duplicate target')
            for k,w in f['target']:
                if not 0<=k<5832 or k%216!=ops[i] or w<=0:raise ValueError('support')
                targets[li,i,k//216]=w
    if (not np.isfinite(targets).all() or not np.isfinite(mass).all() or np.any(mass<=0)
        or not np.allclose(targets.sum(2),1,rtol=0,atol=TOL)
        or not np.allclose(mass.sum(1),1,rtol=0,atol=1e-10)):raise ValueError('native normalization')
    native_marg=np.stack([targets @ (D.GOALS[:,t,None]==np.arange(3)) for t in range(3)],axis=2)
    write(root/'reader/PACKETS.json',packets);write(root/'evaluator/REFERENCES.json',refs)
    raw=json.loads(gzip.decompress((base/'parent/goal_decision_points.json.gz').read_bytes()))
    idx={tuple(r[k] for k in D.FIELDS):r for r in raw}
    expected=set(product(cfg['tiers'],cfg['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,cfg['training_draws'],cfg['fit_seeds']))
    if len(idx)!=len(raw) or set(idx)!=expected:raise ValueError('parent roster')
    rows=[];native=[];decompositions=[];frames=0;error=0.;reproduced=0;timing=[];(root/'reports').mkdir()
    uniform=np.full(len(keys),1/len(keys))
    for li,l in enumerate(ls):
        for weighting,step in product(WEIGHTS,range(3)):
            vv,identity=score_classes(native_marg[li,:,step],native_marg[li,:,step],mass[li] if weighting=='native' else uniform)
            for goal,v in enumerate(vv):native.append(dict(lineage=l,weighting=weighting,step=step,goal=goal,**v))
    write(root/'evaluator/NATIVE_CLASS_RELIABILITY.json',native)
    for draw,seed,budget,arm,forecast in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS,FORECASTS):
        pulse(phase='goal-class-reliability',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast);tick=time.process_time()
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(base/'decisions'/(stem+'.npz'),allow_pickle=False) as z:
            q=z['probabilities'];d=D.decisions(q)
            if not np.array_equal(z['operations'],ops) or not np.array_equal(z['coordinate'],d['coordinate']):raise ValueError('decision identity')
            if not np.array_equal(z['marginals'],d['marginals']):raise ValueError('marginal identity')
            marg=z['marginals']
        chosen=D.GOALS[d['coordinate']];p=np.take_along_axis(marg,chosen[:,:,None],2)[:,:,0]
        correct=np.take_along_axis(native_marg,np.broadcast_to(chosen[None,:,:,None],(len(ls),len(keys),3,1)),3)[:,:,:,0]
        np.savez_compressed(root/'reports'/(stem+'.npz'),marginals=marg,native_marginals=native_marg,bin_ids=bins(marg))
        if np.any((q[None]<=0)&(targets>0)):raise ValueError('fine support')
        logq=np.zeros_like(q);np.log(q,out=logq,where=q>0)
        loss=-(targets*mass[:,:,None]*logq).sum((1,2));accuracy=(correct*mass[:,:,None]).sum((1,2))/3
        for li,l in enumerate(ls):
            old=idx['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]
            for a,b in ((loss[li],old['loss']),(accuracy[li],old['goal_accuracy'])):
                e=abs(float(a)-b);error=max(error,e)
                if e>1e-10:raise ValueError('parent score')
            reproduced+=1
            for weighting,step in product(WEIGHTS,range(3)):
                vv,identity=score_classes(marg[:,step],native_marg[li,:,step],mass[li] if weighting=='native' else uniform)
                ident=dict(tier='E2-full',budget=budget,arm=arm,forecast=forecast,weighting=weighting,step=step,lineage=l,draw=draw,seed=seed)
                decompositions.append(dict(ident,**identity))
                for goal,v in enumerate(vv):rows.append(dict(ident,goal=goal,fine_loss=float(loss[li]),**v))
        frames+=len(keys);timing.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,cpu_seconds=time.process_time()-tick))
    (root/'goal_class_reliability_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'BRIER_DECOMPOSITION.json',decompositions)
    write(root/'TIMING.jsonl',dict(measurements=timing));write(root/'PARENT_REPRODUCTION.json',dict(passed=True,cells=reproduced,max_error=error))
    return dict(controls=checks,rows=len(rows),native_rows=len(native),frame_forecasts=frames,packets=len(keys),parent_cells=reproduced,parent_max_error=error,**aggregate(rows,cfg),fits=0,decomposition_cells=len(decompositions),max_brier_identity_error=max(r['identity_error'] for r in decompositions),scope='all three goal marginals, fixed bins and multiclass Brier decomposition; native targets evaluator-only; no historical correspondence')
