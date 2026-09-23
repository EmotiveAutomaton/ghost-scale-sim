"""Unbinned absolute residual bounds for frozen local-goal probability partitions."""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import goal_class_reliability as C
from . import goal_decision as D

ARMS, FORECASTS, WEIGHTS, FIELDS = C.ARMS, C.FORECASTS, C.WEIGHTS, C.FIELDS
RESOLUTIONS = (5, 10, 20)
EDGES = {n: np.array([i/n for i in range(n+1)]) for n in RESOLUTIONS}
OFFSETS = (0, 5, 15, 35)
METRICS = ('absolute_error', 'bin_error_5', 'bin_error_10', 'bin_error_20',
           'gap_5', 'gap_10', 'gap_20')
INCREMENTS = ('gap_5', 'gap_10', 'gap_20')
from . import goal_bin_resolution as B
bins = B.bins


def sufficient(p, c, w):
    """Triangle bound with unchanged parent bins and retained frame evidence."""
    base = B.sufficient(p, c, w)
    p, c, w = (np.asarray(x, float) for x in (p, c, w))
    residual = p[None] - c
    contribution = w[:, :, None, None] * np.abs(residual)
    absolute = contribution.sum(axis=1)
    gaps = {f'gap_{n}': absolute-base[f'bin_error_{n}'] for n in RESOLUTIONS}
    if min(float(np.min(v)) for v in gaps.values()) < -1e-12:
        raise ValueError('triangle inequality')
    if np.min(gaps['gap_5']-gaps['gap_10']) < -1e-12 or np.min(gaps['gap_10']-gaps['gap_20']) < -1e-12:
        raise ValueError('envelope nesting')
    return dict(base, absolute_error=absolute, **gaps,
                frame_residual=residual, frame_absolute_contribution=contribution)


def controls():
    p = np.tile([.15,.85,0.],(2,3,1)); c=p[None].copy()
    c[0,0,:,:2]+=[.05,-.05]; c[0,1,:,:2]+=[-.05,.05]
    w=np.array([[.5,.5]]); x=sufficient(p,c,w)
    same=p[None].copy(); same[0,:,:,:2]+=[.05,-.05]
    y=sufficient(p,same,w); own=sufficient(p,p[None],w)
    return {'live:opposing_errors_hidden_by_all_partitions': bool(np.allclose(x['gap_20'][...,:2],.05,rtol=0,atol=1e-14)),
            'positive:constant_signed_error_tight_bound': bool(all(np.max(abs(y[m]))<1e-14 for m in INCREMENTS)),
            'placebo:native_self': bool(all(np.max(abs(own[m]))==0 for m in METRICS)),
            'positive:raw_absolute_sum': bool(np.allclose(x['frame_absolute_contribution'].sum(1),x['absolute_error'],rtol=0,atol=1e-14))}


def aggregate(rows, cfg):
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    expected = set(product(cfg['tiers'], bs, ARMS, FORECASTS, WEIGHTS, range(3), range(3), ls, ds, ss))
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    if len(idx) != len(rows) or set(idx) != expected:
        raise ValueError('stratum roster')
    if len(bs) < 2 or any(a >= b for a, b in zip(bs, bs[1:])):
        raise ValueError('budget order')
    sample = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls), size=(cfg['bootstrap_resamples'], len(ls)))
    def estimate(v):
        if not np.isfinite(v).all(): raise ValueError('nonfinite estimate')
        line = v.mean((1, 2)); boot = line[sample].mean(1)
        return dict(mean=float(line.mean()), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)), lineage_values=line.tolist(), draw_means=v.mean((0, 2)).tolist(), feature_seed_means=v.mean((0, 1)).tolist())
    means, contrasts, areas, increments = [], [], [], []
    for tier, forecast, weighting, t, g in product(cfg['tiers'], FORECASTS, WEIGHTS, range(3), range(3)):
        def values(a, b, m):
            return np.array([idx[tier, b, a, forecast, weighting, t, g, l, d, s][m] for l, d, s in product(ls, ds, ss)]).reshape(len(ls), len(ds), len(ss))
        ident = dict(tier=tier, forecast=forecast, weighting=weighting, step=t, goal=g)
        for arm, budget in product(ARMS, bs):
            means.append(dict(ident, arm=arm, budget=budget, **{m:float(values(arm, budget, m).mean()) for m in METRICS}))
            for m in INCREMENTS:
                increments.append(dict(ident, arm=arm, budget=budget, metric=m, **estimate(values(arm, budget, m))))
        for baseline, metric in product(tuple(a for a in ARMS if a != 'learned-bank'), METRICS):
            curves=[]; identity=dict(ident, arm='learned-bank', baseline=baseline, metric=metric)
            for b in bs:
                v=values('learned-bank', b, metric)-values(baseline, b, metric); curves.append(v)
                contrasts.append(dict(identity, budget=b, **estimate(v)))
            areas.append(dict(identity, **estimate(np.trapezoid(curves, x=np.log(bs), axis=0)/np.log(bs[-1]/bs[0]))))
    return dict(means=means, envelope_estimates=increments, contrasts=contrasts, normalized_log_budget_area=areas)


def run(root,plan,pulse):
    cfg=plan['design'];base=root/'inputs';checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    parent=read(base/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if cfg[k]!=parent[k]:raise ValueError('population')
    if cfg['tiers']!=['E2-full'] or cfg['resolutions']!=list(RESOLUTIONS) or cfg['bin_edges']!={str(n):EDGES[n].tolist() for n in RESOLUTIONS}:raise ValueError('design')
    packets=read(base/'reader/PACKETS.json');keys=sorted(packets['packets']);ops=[]
    for k in keys:
        p=packets['packets'][k];D.S.validate(p)
        if digest(p)!=k or p['tier']!='E2-full':raise ValueError('packet')
        ops.append([D.S.L.OPERATIONS.index(e['operation']) for e in p['inputs']['observations']])
    ops=np.asarray(ops,dtype=int);codes=ops@np.array([36,6,1])
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
    (root/'evaluator').mkdir(exist_ok=True)
    np.savez_compressed(root/'evaluator/NATIVE_FORECASTS.npz',marginals=native,weights=mass)
    old=json.loads(gzip.decompress((base/'parent/goal_decision_points.json.gz').read_bytes()))
    idx={tuple(r[k] for k in D.FIELDS):r for r in old}
    expected=set(product(cfg['tiers'],cfg['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,cfg['training_draws'],cfg['fit_seeds']))
    if len(idx)!=len(old) or set(idx)!=expected:raise ValueError('parent roster')
    bin_rows=json.loads(gzip.decompress((base/'parent_bin/goal_bin_resolution_points.json.gz').read_bytes()))
    bin_idx={tuple(r[k] for k in FIELDS):r for r in bin_rows}
    expected_bins=set(product(cfg['tiers'],cfg['budgets'],ARMS,FORECASTS,WEIGHTS,range(3),range(3),ls,cfg['training_draws'],cfg['fit_seeds']))
    if len(bin_idx)!=len(bin_rows) or set(bin_idx)!=expected_bins:raise ValueError('bin parent roster')
    write(root/'reader/PACKETS.json',packets);write(root/'evaluator/REFERENCES.json',refs)
    write(root/'ARRAY_SCHEMA.json',dict(axes=['law','weighting','position','goal','packed_bin'],law=ls,weighting=list(WEIGHTS),resolutions=list(RESOLUTIONS),offsets=list(OFFSETS),bin_edges=cfg['bin_edges'],empty_means='NaN iff bin weight equals zero',sums='joint population mass; each resolution preserves mass one',residual_axes=['law','frame','position','goal'],contribution_axes=['law','weighting','frame','position','goal'],frame=keys,raw_residual_location='residuals; evaluator only'))
    (root/'groups').mkdir();(root/'forecasts').mkdir();(root/'residuals').mkdir();rows=[];timing=[];parent_error=0.;pool_error=0.;frames=0;parents=0
    weights=(mass,np.full_like(mass,1/len(keys)))
    native_groups=[sufficient(native[li],native[li:li+1],w[li:li+1]) for li in range(len(ls)) for w in weights]
    if any(np.max(v['bin_error_20'])!=0 for v in native_groups):raise ValueError('native calibration')
    np.savez_compressed(root/'evaluator/NATIVE_GROUPS.npz',**{k:np.concatenate([x[k] for x in native_groups],axis=0).reshape(len(ls),2,*native_groups[0][k].shape[1:]) for k in native_groups[0] if not k.startswith('frame_')})
    for draw,seed,budget,arm,forecast in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS,FORECASTS):
        pulse(phase='absolute-error-envelope',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast);tick=time.process_time()
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(base/'decisions'/(stem+'.npz'),allow_pickle=False) as z:
            q=z['probabilities'];d=D.decisions(q)
            if not np.array_equal(z['operations'],codes) or not np.array_equal(z['coordinate'],d['coordinate']) or not np.array_equal(z['marginals'],d['marginals']):raise ValueError('decision identity')
            marg=z['marginals']
        if np.any((q[None]<=0)&(target>0)):raise ValueError('fine support')
        logq=np.zeros_like(q);np.log(q,out=logq,where=q>0);loss=-(target*mass[:,:,None]*logq).sum((1,2))
        chosen=D.GOALS[d['coordinate']];correct=np.take_along_axis(native,np.broadcast_to(chosen[None,:,:,None],(len(ls),len(keys),3,1)),3)[:,:,:,0]
        accuracy=(correct*mass[:,:,None]).sum((1,2))/3
        values=[sufficient(marg,native,w) for w in weights]
        np.savez_compressed(root/'forecasts'/(stem+'.npz'),marginals=marg,bin_ids=np.stack([bins(marg,n) for n in RESOLUTIONS]))
        np.savez_compressed(root/'groups'/(stem+'.npz'),**{k:np.stack([x[k] for x in values],axis=1) for k in values[0] if not k.startswith('frame_')})
        np.savez_compressed(root/'residuals'/(stem+'.npz'),signed=values[0]['frame_residual'],absolute_contribution=np.stack([x['frame_absolute_contribution'] for x in values],axis=1))
        for li,l in enumerate(ls):
            prior=idx['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]
            error=max(abs(loss[li]-prior['loss']),abs(accuracy[li]-prior['goal_accuracy']));parent_error=max(parent_error,float(error))
            if error>1e-10:raise ValueError('parent score')
            parents+=1
            for wi,weighting in enumerate(WEIGHTS):
                val=values[wi]
                for t,g in product(range(3),range(3)):
                    # Preserve the accepted ten-bin sums and unbinned score.
                    pooled=C.measure(marg[:,t,g],native[li,:,t,g],weights[wi][li])
                    for k,oldkey in (('bin_weight','bin_weight'),('bin_forecast_sum','bin_confidence_sum'),('bin_correct_sum','bin_correct_sum')):
                        e=float(np.max(abs(val[k][li,t,g,5:15]-pooled[oldkey])));pool_error=max(pool_error,e)
                        if e>1e-10:raise ValueError('ten-bin identity')
                    e=abs(val['squared_error'][li,t,g]-pooled['squared_error']);pool_error=max(pool_error,float(e))
                    if e>1e-10:raise ValueError('unbinned score identity')
                    br=bin_idx['E2-full',budget,arm,forecast,weighting,t,g,l,draw,seed]
                    for n in RESOLUTIONS:
                        e=abs(val[f'bin_error_{n}'][li,t,g]-br[f'bin_error_{n}']);pool_error=max(pool_error,float(e))
                        if e>1e-10:raise ValueError('accepted bin score identity')
                    rows.append(dict(tier='E2-full',budget=budget,arm=arm,forecast=forecast,weighting=weighting,step=t,goal=g,lineage=l,draw=draw,seed=seed,**{m:float(val[m][li,t,g]) for m in METRICS}))
        frames+=len(keys);timing.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,cpu_seconds=time.process_time()-tick))
    (root/'goal_absolute_envelope_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'TIMING.jsonl',dict(measurements=timing));write(root/'PARENT_REPRODUCTION.json',dict(passed=True,cells=parents,max_error=parent_error))
    return dict(controls=checks,rows=len(rows),frame_forecasts=frames,packets=len(keys),parent_cells=parents,parent_max_error=parent_error,max_pooled_error=pool_error,**aggregate(rows,cfg),fits=0,scope='unbinned absolute residual minus frozen 5/10/20-bin discrepancies; raw signed residuals and weighted absolute contributions; no forecast improvement or historical correspondence')
