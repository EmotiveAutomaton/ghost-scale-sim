"""Independent saved-output verification, executed by the existing scientific owner.

This handler never fits a model or imports the producer's scoring/gradient code.
All inputs are frozen scientific/evaluator records, never blind reader inputs.
"""
from collections import defaultdict
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def pad(a):
    return np.pad(a, ((0, 0), (0, 512-a.shape[1])))


def normalize(a, sizes):
    parts=[]; start=0
    for size in sizes:
        p=np.maximum(a[:, start:start+size], 0)+1e-6
        parts.append(p/p.sum(1, keepdims=True)); start+=size
    return np.concatenate(parts, axis=1)


def objective(x, y, w, sizes):
    logits=x@w; lp=np.empty_like(logits); start=0
    for size in sizes:
        a=logits[:, start:start+size]
        lp[:, start:start+size]=a-np.logaddexp.reduce(a, axis=1)[:, None]
        start+=size
    reg=w.copy(); reg[0]=0
    return (float(-np.sum(y*lp)/len(x)+.005*np.sum(reg**2)),
            x.T@(np.exp(lp)-y)/len(x)+.01*reg, np.exp(lp))


def controls():
    x=np.array([[1., -1.], [1., 1.]])
    y=np.array([[1., 0.], [0., 1.]])
    value, grad, p=objective(x, y, np.zeros((2, 2)), [2])
    _, null, _=objective(x, np.full((2, 2), .5), np.zeros((2, 2)), [2])
    w=np.array([[.1, -.2], [.3, .4]]); _, g, _=objective(x, y, w, [2])
    errors=[]
    for index in np.ndindex(w.shape):
        a=w.copy(); b=w.copy(); a[index]+=1e-6; b[index]-=1e-6
        errors.append(abs((objective(x,y,a,[2])[0]-objective(x,y,b,[2])[0])/2e-6-g[index]))
    return dict(live_known_loss=abs(value-math.log(2))<1e-12,
        live_known_gradient=np.max(abs(grad-np.array([[0.,0.],[.5,-.5]])))<1e-12,
        placebo_constant_target=np.max(abs(null))<1e-12,
        positive_independent_finite_difference=max(errors)<1e-8,
        positive_normalization=np.max(abs(p.sum(1)-1))<1e-12)


def run(out, plan, pulse):
    checks={k:bool(v) for k,v in controls().items()}
    if not all(checks.values()):
        raise ValueError('independent verifier controls failed before result inspection')
    for name, h in plan['design']['input_files'].items():
        if file_digest(out/'inputs'/name)!=h:
            raise ValueError('verification input changed: '+name)
    root=out/'inputs/original'; target=read(root/'PLAN.json'); cfg=target['design']
    done=read(root/'COMPLETE.json'); summary=read(root/'SUMMARY.json')
    assert done['plan_sha256']==file_digest(root/'PLAN.json')
    for n,h in {**done['files'], **done.get('execution_measurements',{})}.items():
        assert file_digest(root/n)==h, n
    previous=read(out/'inputs/predecessor/OPTIMIZATION.json')['fits']
    fitkeys=('domain','draw','tier','seed','arm','budget')
    oldfits={tuple(o[k] for k in fitkeys):o for o in previous}
    opt=read(root/'OPTIMIZATION.json')['fits']
    fits={tuple(o[k] for k in fitkeys):o for o in opt}
    assert set(fits)==set(oldfits) and len(fits)==len(opt)
    diagnostics=[]; maxforecast=0.; maxerror=0.; minprob=1.; normerror=0.
    truth={}; forecasts={}
    for packet in cfg['packets']:
        domain=packet['domain']; sizes=[3,3,3,6,6,6] if domain=='local' else [16,16]
        old_sizes=[8]*4 if domain=='local' else [16]*5
        parent=root/'inputs'/packet['name']
        for draw in cfg['training_draws']:
            for tier in packet['tiers']:
                train=arrays(parent/'evaluator'/f'train-{draw}-{tier}.npz')
                test=arrays(parent/'evaluator'/f'development-{draw}-{tier}.npz'); truth[draw,tier]=test
                for seed in cfg['fit_seeds']:
                    stem=f'{draw}-{tier}-{seed}'; old=arrays(parent/'models'/f'{stem}-old.npz')
                    def reps(data):
                        h=np.column_stack((np.ones(len(data['x'])),np.tanh(pad(data['x'])@old['history_weights']+old['history_bias'])))
                        return {'raw-history':data['x'],'learned-bank':normalize(h@old['old_head'],old_sizes),
                                'frozen-latent':h@old['latent_basis'],'exact-bank-oracle':data['bank']}
                    tr=reps(train); te=reps(test)
                    for arm in tr:
                        f=arrays(parent/'models'/f'{stem}-{arm}-{min(packet["budgets"])}.npz')
                        def encode(a):
                            return np.column_stack((np.ones(len(a)),np.tanh(pad(a)@f['readout_weights']+f['readout_bias'])))
                        x=encode(tr[arm]); xt=encode(te[arm])
                        for budget in packet['budgets']:
                            key=(domain,draw,tier,seed,arm,budget); name=f'{domain}-{stem}-{arm}-{budget}'
                            w=arrays(root/'models'/f'{name}.npz')['head']
                            p=arrays(root/'forecasts'/f'{name}.npz'); forecasts[name+'.npz']=p
                            value,g,_=objective(x[:budget],train['target'][:budget],w,sizes)
                            _,_,pred=objective(xt,test['target'],w,sizes)
                            trace=fits[key]['optimization']; last=trace[-1]; oldlast=oldfits[key]['optimization'][-1]
                            assert trace[0]['step']==0 and last['step']<=1000 and last['function_evaluations']<=2000
                            assert abs(trace[0]['objective']-sum(math.log(s) for s in sizes))<1e-10
                            assert abs(value-last['objective'])<1e-10
                            assert abs(np.linalg.norm(g)-last['gradient_norm'])<1e-10
                            assert abs(abs(g).max()-last['gradient_max'])<1e-10
                            assert value<=oldlast['objective']+1e-10
                            maxforecast=max(maxforecast,float(abs(pred-p['probabilities']).max()))
                            assert maxforecast<1e-12
                            diagnostics.append(dict(zip(fitkeys,key),objective=value,predecessor_objective=oldlast['objective'],
                                gradient_norm=float(np.linalg.norm(g)),gradient_max=float(abs(g).max()),
                                predecessor_gradient_norm=oldlast['gradient_norm'],gtol_met=bool(abs(g).max()<=1e-7),
                                solver_success=last['solver_success'],solver_message=last['solver_message'],iterations=last['step']))
                            pulse(phase='independent-saved-head',verified_heads=len(diagnostics))
                    for budget in packet['budgets']:
                        for arm in ('nested-label-mean','exact-reference'):
                            name=f'{domain}-{stem}-{arm}-{budget}.npz'; f=arrays(root/'forecasts'/name); forecasts[name]=f
                            raw=np.tile(train['target'][:budget].mean(0),(len(test['x']),1)) if arm=='nested-label-mean' else test['exact']
                            assert np.max(abs(normalize(raw,sizes)-f['probabilities']))<1e-12
    rows=json.loads(gzip.decompress((root/'raw/probability-head_points.json.gz').read_bytes()))
    key=('draw','tier','seed','arm','budget','lineage','world_class','query')
    assert len(rows)==summary['rows'] and len({tuple(r[k] for k in key) for r in rows})==len(rows)
    groups=defaultdict(list)
    for r in rows:
        draw,tier,seed,arm,budget,lineage,cl,q=(r[k] for k in key)
        domain='old' if tier=='artifact-history' else 'local'; sizes=[16,16] if domain=='old' else [3,3,3,6,6,6]
        f=forecasts[f'{domain}-{draw}-{tier}-{seed}-{arm}-{budget}.npz']; t=truth[draw,tier]
        assert np.array_equal(f['ids'],t['ids'])
        mask=(t['ids'][:,0]==lineage)&((t['ids'][:,2]==cl) if cl!=-1 else True)
        assert int(mask.sum())==r['histories']; sl=slice(sum(sizes[:q]),sum(sizes[:q+1]))
        y=t['target'][mask,sl]; p=f['probabilities'][mask,sl]
        loss=float(np.mean(np.sum(-y*np.log(p),axis=1))); brier=float(np.mean(np.sum((p-y)**2,axis=1)))
        maxerror=max(maxerror,abs(loss-r['loss']),abs(brier-r['brier'])); minprob=min(minprob,float(p.min()))
        normerror=max(normerror,float(abs(p.sum(1)-1).max()))
        for kind in (('all',) if domain=='old' else ('all','goals' if q<3 else 'operations')):
            groups[kind,tier,cl,arm,budget,lineage].append(loss)
    assert maxerror<1e-12 and minprob>0 and normerror<1e-12
    for cell in summary['cells']:
        rr=[r['loss'] for r in rows if all(r[k]==cell[k] for k in ('tier','world_class','arm','budget','query'))]
        assert abs(math.fsum(rr)/len(rr)-cell['loss'])<1e-12
    means={k:math.fsum(v)/len(v) for k,v in groups.items()}
    for (kind,tier,cl,arm,b,l),v in means.items():
        if cl==-1:
            assert abs(v-sum(means[kind,tier,z,arm,b,l] for z in range(4))/4)<1e-12
    def interval(v):
        v=np.asarray(v); rng=np.random.default_rng(190501)
        boot=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
        return dict(mean=float(v.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),lineages=len(v))
    comparisons=[]; absolute=[]; variation=[]
    for packet in cfg['packets']:
        budgets=packet['budgets']; weights=np.array([.5,.5]) if len(budgets)==2 else np.array([1,2,2,1])/6
        for tier in packet['tiers']:
            lineages=sorted({r['lineage'] for r in rows if r['tier']==tier})
            for kind in (('all',) if packet['domain']=='old' else ('all','goals','operations')):
                for cl in ([-1,0,1,2,3] if packet['domain']=='old' else [0]):
                    for baseline in ('raw-history','frozen-latent','nested-label-mean','exact-reference'):
                        v=np.array([[means[kind,tier,cl,'learned-bank',b,l]-means[kind,tier,cl,baseline,b,l] for l in lineages] for b in budgets])
                        comparisons.append(dict(target=kind,tier=tier,world_class=cl,baseline=baseline,**interval(weights@v),
                            budgets=[dict(budget=b,**interval(a)) for b,a in zip(budgets,v)]))
                    for arm in ('raw-history','learned-bank','frozen-latent','exact-bank-oracle','nested-label-mean','exact-reference'):
                        for b in budgets:
                            absolute.append(dict(target=kind,tier=tier,world_class=cl,arm=arm,budget=b,**interval([means[kind,tier,cl,arm,b,l] for l in lineages])))
            for draw in cfg['training_draws']:
                for seed in cfg['fit_seeds']:
                    for baseline in ('raw-history','frozen-latent','nested-label-mean'):
                        cl=-1 if packet['domain']=='old' else 0; v=[]
                        for b in budgets:
                            def avg(arm):
                                values=[r['loss'] for r in rows if (r['tier'],r['world_class'],r['draw'],r['seed'],r['arm'],r['budget'])==(tier,cl,draw,seed,arm,b)]
                                return math.fsum(values)/len(values)
                            v.append(avg('learned-bank')-avg(baseline))
                        variation.append(dict(tier=tier,draw=draw,feature_seed=seed,baseline=baseline,area_difference=float(weights@v)))
    write(out/'INDEPENDENT_REVIEW.json',dict(passed=True,target_plan_sha256=done['plan_sha256'],score_rows_recomputed=len(rows),
        forecast_files=len(forecasts),heads_verified=len(diagnostics),max_score_error=maxerror,max_forecast_error=maxforecast,
        min_probability=minprob,max_normalization_error=normerror,optimization=diagnostics,comparisons=comparisons,
        absolute=absolute,fit_variation=variation,controls=checks,
        uncertainty='paired development lineages conditional on two feature seeds and two training draws; equal old classes; no confirmation',
        scope='independent saved-output reconstruction only; original adjacent and portable replays require separate hash comparison'))
    return dict(controls=checks,score_rows=len(rows),heads=len(diagnostics),target_plan_sha256=done['plan_sha256'],
        warrant='verification only; no new fitted setting or scientific population')
