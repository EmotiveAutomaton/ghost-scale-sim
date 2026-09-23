"""Frozen temporal products separated from training-alphabet mass allocation."""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import joint_support as S
from .joint_uncertainty import distribution
from .temporal_factorization import factorize, N, ARMS, FIELDS


def mass_match(q, temporal, alphabet):
    """Keep original within-group distributions; take product group masses.

    Groups are training labels and their complement, never native support.
    A positive requested mass without an original conditional is impossible.
    """
    q=np.asarray(q,float);temporal=np.asarray(temporal,float);alphabet=np.asarray(alphabet)
    if q.ndim!=2 or q.shape[1]!=N or temporal.shape!=q.shape:raise ValueError('distribution shape')
    for v in (q,temporal):
        if not np.isfinite(v).all() or np.any(v<0) or not np.allclose(v.sum(1),1,rtol=0,atol=3e-12):raise ValueError('distribution normalization')
    if alphabet.ndim!=1 or not np.issubdtype(alphabet.dtype,np.integer) or len(set(alphabet.tolist()))!=len(alphabet) or np.any(alphabet<0) or np.any(alphabet>=N):raise ValueError('alphabet')
    known=np.zeros(N,bool);known[alphabet]=True
    result=np.zeros_like(q);before=[];after=[]
    for mask in (known,~known):
        a=q[:,mask].sum(1);b=temporal[:,mask].sum(1)
        if np.any((a==0)&(b>0)):raise ValueError('positive target mass with undefined original conditional')
        scale=np.divide(b,a,out=np.zeros_like(a),where=a>0)
        result[:,mask]=q[:,mask]*scale[:,None];before.append(a);after.append(b)
    if not np.allclose(result.sum(1),1,rtol=0,atol=4e-12):raise ValueError('matched normalization')
    return result,np.stack(before,axis=1),np.stack(after,axis=1)


def controls():
    q=np.zeros((1,N));q[0,:4]=[.1,.2,.3,.4]
    p=q.copy();p[0,:4]=[.3,.3,.2,.2]
    r,_,_=mass_match(q,p,np.array([0,1]))
    expected=np.zeros_like(q);expected[0,:4]=[.2,.4,6/35,8/35]
    return {'live:group_mass_changed':bool(np.allclose(r,expected,rtol=0,atol=1e-15)),
            'placebo:identical_masses_identity':bool(np.array_equal(mass_match(q,q,np.array([0,1]))[0],q)),
            'positive:conditional_odds_preserved':bool(abs(r[0,0]/r[0,1]-.5)<1e-15 and abs(r[0,2]/r[0,3]-.75)<1e-15)}


def aggregate(rows,cfg):
    idx={tuple(r[k] for k in FIELDS):r for r in rows}
    ls,ds,ss,bs=(cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    arms=tuple(a+s for a in ARMS for s in ('','-mass-matched','-product'))
    expected=set(product(cfg['tiers'],bs,arms,ls,ds,ss))
    if len(idx)!=len(rows) or set(idx)!=expected:raise ValueError('complete stratum roster')
    if len(bs)<2 or any(b>=c for b,c in zip(bs,bs[1:])):raise ValueError('ordered budgets')
    samples=np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls)))
    def estimate(v):
        line=v.mean((1,2));boot=line[samples].mean(1)
        return dict(mean=float(line.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),
            lineage_values=line.tolist(),draw_means=v.mean((0,2)).tolist(),feature_seed_means=v.mean((0,1)).tolist())
    contrasts=[];areas=[];means=[];metrics=sorted(set(rows[0])-set(FIELDS)-{'frames'})
    for tier in cfg['tiers']:
        for arm,budget in product(arms,bs):
            rr=[idx[tier,budget,arm,l,d,s] for l,d,s in product(ls,ds,ss)]
            means.append(dict(tier=tier,budget=budget,arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in metrics}))
        for arm in ARMS:
            for suffix,baseline in (('-mass-matched',''),('-product','-mass-matched')):
                curve=[]
                for budget in bs:
                    v=np.array([[[idx[tier,budget,arm+suffix,l,d,s]['loss']-idx[tier,budget,arm+baseline,l,d,s]['loss'] for s in ss] for d in ds] for l in ls])
                    contrasts.append(dict(tier=tier,budget=budget,arm=arm+suffix,baseline=arm+baseline,**estimate(v)));curve.append(v)
                area=np.trapezoid(curve,x=np.log(bs),axis=0)/np.log(bs[-1]/bs[0])
                areas.append(dict(tier=tier,arm=arm+suffix,baseline=arm+baseline,**estimate(area)))
    return dict(contrasts=contrasts,normalized_log_budget_area=areas,means=means)


def run(root,plan,pulse):
    cfg=plan['design'];base=root/'inputs';checks=controls()
    write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('known-answer controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    parent=read(base/'PLAN.json')['design'];temporal_plan=read(base/'temporal/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if parent[k]!=cfg[k] or temporal_plan[k]!=cfg[k]:raise ValueError('parent population changed')
    packets=read(base/'reader/PACKETS.json');refs=read(base/'evaluator/REFERENCES.json')
    for key,p in packets['packets'].items():
        S.validate(p)
        if key!=digest(p):raise ValueError('packet identity')
    write(root/'reader/PACKETS.json',packets);write(root/'evaluator/REFERENCES.json',refs)
    old=read(base/'SUMMARY.json')['cells']
    temporal=json.loads(gzip.decompress((base/'temporal/temporal_points.json.gz').read_bytes()))
    idx={tuple(r[k] for k in FIELDS):r for r in old};tidx={tuple(r[k] for k in FIELDS):r for r in temporal}
    expected=set(product(cfg['tiers'],cfg['budgets'],ARMS,cfg['development_lineages'],cfg['training_draws'],cfg['fit_seeds']))
    texpected=set(product(cfg['tiers'],cfg['budgets'],tuple(a+s for a in ARMS for s in ('','-product')),cfg['development_lineages'],cfg['training_draws'],cfg['fit_seeds']))
    if len(idx)!=len(old) or set(idx)!=expected or len(tidx)!=len(temporal) or set(tidx)!=texpected:raise ValueError('parent complete roster')
    native=read(base/'temporal/NATIVE_SCORES.json')
    native_idx={(r['tier'],r['lineage'],r['arm']):r for r in native}
    if len(native_idx)!=len(native) or set(native_idx)!=set(product(cfg['tiers'],cfg['development_lineages'],('native-joint','native-product'))):raise ValueError('native roster')
    write(root/'evaluator/INHERITED_NATIVE_SCORES.json',native)
    rows=[];timing=[];reproduced=0;max_error=0.;temporal_error=0.;frames_scored=0
    (root/'masses').mkdir()
    for tier in cfg['tiers']:
        keys=sorted(k for k,p in packets['packets'].items() if p['tier']==tier)
        ref=S.reference_arrays(refs,tier,keys,cfg['development_lineages'],np.ones((len(keys),N),bool))
        if set(r['lineage'] for r in refs if r['tier']==tier)!=set(cfg['development_lineages']):raise ValueError('native lineage roster')
        for draw,seed,budget,arm in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS):
            pulse(phase='temporal-alphabet-mass',tier=tier,draw=draw,seed=seed,budget=budget,arm=arm)
            tick=time.process_time();stem=f'{draw}-{tier}-{seed}-{budget}-{arm}'
            if read(base/'forecasts'/(stem+'-frames.json'))!=keys:raise ValueError('forecast frame roster')
            with np.load(base/'forecasts'/(stem+'.npz'),allow_pickle=False) as data:
                if set(data.files)!={'probabilities','alphabet'}:raise ValueError('forecast fields')
                alphabet=data['alphabet'];q=distribution(data['probabilities'],alphabet,budget)
            if len(q)!=len(keys):raise ValueError('forecast length')
            fact,steps=factorize(q);matched,before,after=mass_match(q,fact,alphabet)
            np.savez_compressed(root/'masses'/(stem+'.npz'),steps=steps,original=before,product=after)
            scores={'':S.scores(q,alphabet,ref),'-mass-matched':S.scores(matched,alphabet,ref),'-product':S.scores(fact,alphabet,ref)}
            for li,lineage in enumerate(cfg['development_lineages']):
                prior=idx[tier,budget,arm,lineage,draw,seed]
                native_loss=native_idx[tier,lineage,'native-joint']['loss']
                for k,v in scores[''].items():
                    error=abs(float(v[li])-prior[k]);max_error=max(max_error,error)
                    if error>1e-10:raise ValueError('original score reproduction')
                for suffix in ('','-product'):
                    prior=tidx[tier,budget,arm+suffix,lineage,draw,seed]
                    for k,v in scores[suffix].items():
                        error=abs(float(v[li])-prior[k]);temporal_error=max(temporal_error,error)
                        if error>1e-10:raise ValueError('temporal score reproduction')
                    if abs(prior['excess_loss']-(float(scores[suffix]['loss'][li])-native_loss))>1e-10:raise ValueError('native score binding')
                reproduced+=1
                for suffix,values in scores.items():
                    excess=float(values['loss'][li])-native_loss
                    if excess<-1e-10:raise ValueError('negative excess')
                    rows.append(dict(tier=tier,budget=budget,arm=arm+suffix,lineage=lineage,draw=draw,seed=seed,frames=len(keys),**{k:float(v[li]) for k,v in values.items()},excess_loss=excess))
            frames_scored+=len(keys)
            timing.append(dict(tier=tier,draw=draw,seed=seed,budget=budget,arm=arm,cpu_seconds=time.process_time()-tick))
    if reproduced!=len(old):raise ValueError('original coverage')
    (root/'mass_control_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'PARENT_REPRODUCTION.json',dict(passed=True,original_cells=reproduced,temporal_cells=2*reproduced,original_max_error=max_error,temporal_max_error=temporal_error,metrics='all original and temporal scores;native scores inherited'))
    write(root/'TIMING.jsonl',dict(measurements=timing));checks['positive:complete_parent_reproduction']=True
    return dict(controls=checks,rows=len(rows),original_cells_reproduced=reproduced,temporal_cells_reproduced=2*reproduced,
        original_max_error=max_error,temporal_max_error=temporal_error,frame_forecasts=frames_scored,packets=len(packets['packets']),
        inherited_native_rows=len(native),**aggregate(rows,cfg),fits=0,
        scope='constructed-method alphabet-mass diagnostic; original conditional distributions and temporal group masses; native scores inherited evaluator rulers; no support mask,fit,new sample or correspondence claim')
