"""Operation/conditional-goal chain rule on frozen joint process forecasts.

This handler does not fit, restrict support or change the native population.
Parent localization/abstention metrics remain explicitly inherited evidence.
"""
from collections import defaultdict
from itertools import product
import gzip
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .joint_support_review import validate as validate_packet

N = 5832
ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
COMPONENTS = ('loss', 'operation_loss', 'conditional_goal_loss',
              'joint_excess', 'operation_excess', 'conditional_goal_excess')


def distribution(probabilities, alphabet, budget):
    p = np.asarray(probabilities, float); a = np.asarray(alphabet)
    if (a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or len(set(a)) != len(a)
            or len(a) >= N or np.any(a < 0) or np.any(a >= N)
            or p.ndim != 2 or p.shape[1] != len(a) or np.any(p < 0)
            or not np.isfinite(p).all()):
        raise ValueError('forecast/alphabet')
    if not np.allclose(p.sum(1), budget/(budget+1), rtol=0, atol=1e-12):
        raise ValueError('unknown allocation')
    q = np.full((len(p), N), 1/((budget+1)*(N-len(a))))
    q[:, a] = p
    return q


def components(q, labels, target):
    """One frame, one or more native laws; code = goal_sequence*216+ops."""
    q = np.asarray(q, float); labels = np.asarray(labels); t = np.asarray(target, float)
    if (q.shape != (N,) or t.ndim != 2 or t.shape[1] != len(labels)
            or labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer)
            or len(set(labels)) != len(labels) or np.any(labels < 0) or np.any(labels >= N)
            or not np.isfinite(q).all() or not np.isfinite(t).all()
            or np.any(q < 0) or np.any(t <= 0)):
        raise ValueError('distribution or target')
    if not np.isclose(q.sum(), 1, rtol=0, atol=1e-12) or not np.allclose(t.sum(1), 1, rtol=0, atol=1e-10):
        raise ValueError('normalization')
    op = labels % 216
    marginal = q.reshape(27, 216).sum(0)
    if np.any(q[labels] <= 0): raise ValueError('zero forecast on true label')
    native_op = np.zeros((len(t), 216))
    for j, code in enumerate(op): native_op[:, code] += t[:, j]
    joint = -(t @ np.log(q[labels]))
    operation = -(t @ np.log(marginal[op]))
    conditional = -(t @ np.log(q[labels]/marginal[op]))
    entropy = -(t*np.log(t)).sum(1)
    op_entropy = -(native_op*np.log(np.where(native_op > 0, native_op, 1))).sum(1)
    conditional_entropy = -(t*np.log(t/native_op[:, op])).sum(1)
    chain_error = max(float(np.max(np.abs(joint-operation-conditional))),
                      float(np.max(np.abs(entropy-op_entropy-conditional_entropy))))
    if chain_error > 1e-10: raise ValueError('chain rule')
    result = dict(loss=joint, operation_loss=operation, conditional_goal_loss=conditional,
                  joint_entropy=entropy, operation_entropy=op_entropy,
                  conditional_goal_entropy=conditional_entropy,
                  joint_excess=joint-entropy, operation_excess=operation-op_entropy,
                  conditional_goal_excess=conditional-conditional_entropy)
    if min(float(np.min(result[k])) for k in COMPONENTS[3:]) < -1e-10:
        raise ValueError('negative relative entropy')
    return result, chain_error


def regroup(rows, cfg):
    keys = ('tier','budget','arm','lineage','draw','seed')
    idx = {tuple(r[k] for k in keys):r for r in rows}
    if len(idx) != len(rows): raise ValueError('duplicate strata')
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls), size=(cfg['bootstrap_resamples'],len(ls)))
    def estimate(x):
        v = x.mean((1,2)); boot = v[samples].mean(1)
        return dict(mean=float(v.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),
                    lineage_values=v.tolist(),draw_means=x.mean((0,2)).tolist(),feature_seed_means=x.mean((0,1)).tolist())
    contrasts=[]; areas=[]; means=[]
    for tier in cfg['tiers']:
        for arm in ARMS:
            for budget in bs:
                rr=[idx[tier,budget,arm,l,d,s] for l,d,s in product(ls,ds,ss)]
                means.append(dict(tier=tier,budget=budget,arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in COMPONENTS+('joint_entropy','operation_entropy','conditional_goal_entropy')}))
        for base in ARMS:
            if base == 'learned-bank': continue
            for metric in COMPONENTS:
                curve=[]
                for budget in bs:
                    x=np.array([[[idx[tier,budget,'learned-bank',l,d,s][metric]-idx[tier,budget,base,l,d,s][metric] for s in ss] for d in ds] for l in ls])
                    contrasts.append(dict(tier=tier,budget=budget,arm='learned-bank',baseline=base,metric=metric,**estimate(x)));curve.append(x)
                area=np.trapezoid(curve,x=np.log(bs),axis=0)/np.log(bs[-1]/bs[0])
                areas.append(dict(tier=tier,arm='learned-bank',baseline=base,metric=metric,**estimate(area)))
    return dict(contrasts=contrasts,normalized_log_budget_area=areas,means=means)


def controls():
    uniform=np.ones(N)/N
    same_op,_=components(uniform,np.array([0,216]),np.array([[.5,.5]]))
    q=np.zeros(N);q[[0,1]]=[.25,.75]
    exact,_=components(q,np.array([0,1]),np.array([[.25,.75]]))
    return {'live:uniform_operation_loss':bool(np.isclose(same_op['operation_loss'][0],np.log(216))),
            'positive:same_operation_goal_entropy':bool(np.isclose(same_op['conditional_goal_entropy'][0],np.log(2))),
            'placebo:perfect_joint_zero_excess':bool(all(abs(exact[k][0])<1e-12 for k in COMPONENTS[3:])),
            'positive:distinct_operations_same_goal':bool(exact['conditional_goal_entropy'][0]==0)}


def run(root, plan, pulse):
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('known-answer controls')
    cfg=plan['design'];base=root/'inputs'
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input hash')
    parent=read(base/'PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if parent[k]!=cfg[k]:raise ValueError('parent roster')
    packets=read(base/'reader/PACKETS.json')
    for key,p in packets['packets'].items():
        validate_packet(p)
        if key!=digest(p):raise ValueError('packet key')
    write(root/'reader/PACKETS.json',packets)
    refs=read(base/'evaluator/REFERENCES.json');write(root/'evaluator/REFERENCES.json',refs)
    fields=('tier','budget','arm','lineage','draw','seed')
    old=read(base/'SUMMARY.json')['cells'];idx={tuple(r[k] for k in fields):r for r in old}
    if len(idx)!=len(old):raise ValueError('parent duplicate')
    rows=[];max_chain=max_original=0.;frames_scored=0
    for tier in cfg['tiers']:
        keys=sorted(k for k,p in packets['packets'].items() if p['tier']==tier)
        records=[r for r in refs if r['tier']==tier]
        bylin={r['lineage']:{f['frame']:f for f in r['frames']} for r in records}
        if len(records)!=len(cfg['development_lineages']) or set(bylin)!=set(cfg['development_lineages']):raise ValueError('reference lineage roster')
        if any(set(f)!=set(keys) for f in bylin.values()):raise ValueError('reference frame roster')
        frame_refs=[]
        for key in keys:
            labels=sorted(k for k,v in bylin[cfg['development_lineages'][0]][key]['target'])
            t=[];mass=[]
            for l in cfg['development_lineages']:
                f=bylin[l][key];target=dict(f['target'])
                if sorted(target)!=labels or len(target)!=len(f['target']) or f['mass']<=0:raise ValueError('target coverage')
                t.append([target[k] for k in labels]);mass.append(f['mass'])
            frame_refs.append((np.array(labels),np.array(t),np.array(mass)))
        if not np.allclose(np.sum([x[2] for x in frame_refs],axis=0),1,rtol=0,atol=1e-10):raise ValueError('native mass')
        for draw,seed,budget,arm in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS):
            pulse(phase='joint-uncertainty',tier=tier,draw=draw,seed=seed,budget=budget,arm=arm)
            stem=f'{draw}-{tier}-{seed}-{budget}-{arm}'
            if read(base/'forecasts'/(stem+'-frames.json'))!=keys:raise ValueError('forecast frame roster')
            with np.load(base/'forecasts'/(stem+'.npz'),allow_pickle=False) as data:
                if set(data.files)!= {'probabilities','alphabet'}:raise ValueError('forecast fields')
                q=distribution(data['probabilities'],data['alphabet'],budget)
            if len(q)!=len(keys):raise ValueError('forecast length')
            totals=defaultdict(lambda:np.zeros(len(cfg['development_lineages'])))
            for i,(labels,target,mass) in enumerate(frame_refs):
                values,error=components(q[i],labels,target);max_chain=max(max_chain,error)
                for k,v in values.items():totals[k]+=mass*v
                frames_scored+=1
            for j,lineage in enumerate(cfg['development_lineages']):
                prior=idx[tier,budget,arm,lineage,draw,seed]
                error=abs(prior['loss']-totals['loss'][j]);max_original=max(max_original,error)
                if error>1e-10:raise ValueError('original joint loss')
                rows.append(dict(prior,**{k:float(v[j]) for k,v in totals.items()}))
    if len(rows)!=len(old):raise ValueError('complete roster')
    (root/'uncertainty_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'ORIGINAL_REPRODUCTION.json',dict(passed=True,rows=len(rows),maximum_joint_loss_error=max_original,
        inherited_fields='all parent localization,abstention,candidate and compatibility metrics; only joint loss independently recomputed here'))
    checks['positive:complete_original_loss_reproduction']=True
    return dict(controls=checks,rows=len(rows),frame_forecasts_scored=frames_scored,packets=len(packets['packets']),
                max_chain_error=max_chain,max_original_loss_error=max_original,**regroup(rows,cfg),
                scope='constructed-method operation/conditional-goal loss and entropy decomposition; frozen fits and native targets inherited; no support restriction,training or confirmation')
