"""Exact joint versus marginal forecast partitions; native joint uncertainty."""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import goal_decision as D

ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
FORECASTS = ('restricted', 'goal-product')
WEIGHTS = ('native', 'equal-frame')
PARTITIONS = ('joint', 'marginal')
METRICS = ('total_variation', 'within_law_variation', 'across_law_variation')


def partition(p, kind):
    p = np.asarray(p, float)
    shape = (27,) if kind == 'joint' else (3, 3) if kind == 'marginal' else None
    if shape is None or p.ndim != len(shape)+1 or p.shape[1:] != shape:
        raise ValueError('forecast shape')
    if not np.isfinite(p).all() or np.min(p) < 0 or np.max(p) > 1+1e-12:
        raise ValueError('forecast probability')
    if not np.allclose(p.sum(-1), 1, atol=1e-12, rtol=0):
        raise ValueError('forecast simplex')
    seen = {}; ids = []
    for row in p.reshape(len(p), -1):
        key = tuple(row)
        if key not in seen: seen[key] = len(seen)
        ids.append(seen[key])
    return np.asarray(ids, dtype=np.int64)


def refinement(joint, marginal):
    joint, marginal = np.asarray(joint), np.asarray(marginal)
    if joint.ndim != 1 or joint.shape != marginal.shape: raise ValueError('membership shape')
    # Report numerical exceptions; never silently merge near-equal forecasts.
    exceptions = [int(j) for j in np.unique(joint) if len(set(marginal[joint == j])) != 1]
    splits = [int(m) for m in np.unique(marginal) if len(set(joint[marginal == m])) > 1]
    return dict(joint_refines_marginal=not exceptions, exception_joint_groups=exceptions,
                split_marginal_groups=splits)


def sufficient(p, kind, target, weight, ids=None):
    actual = partition(p, kind)
    if ids is not None and not np.array_equal(ids, actual): raise ValueError('faulty membership')
    target, weight = np.asarray(target, float), np.asarray(weight, float)
    if weight.ndim != 2 or target.shape != (*weight.shape, 27) or len(actual) != weight.shape[1]:
        raise ValueError('target shape')
    if not np.isfinite(target).all() or np.min(target) < 0 or np.max(target) > 1+1e-12 or not np.allclose(target.sum(2), 1, atol=1e-12, rtol=0):
        raise ValueError('target simplex')
    if not np.isfinite(weight).all() or np.min(weight) < 0 or not np.allclose(weight.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError('weight normalization')
    laws, frames = weight.shape; groups = int(actual.max())+1
    mass = np.zeros((laws, groups)); sums = np.zeros((laws, groups, 27))
    square = (weight[:, :, None]*target**2).sum((1, 2))
    for law in range(laws):
        np.add.at(mass[law], actual, weight[law])
        np.add.at(sums[law], actual, weight[law, :, None]*target[law])
    means = np.divide(sums, mass[:, :, None], out=np.zeros_like(sums), where=mass[:, :, None] > 0)
    # Directly centered residual avoids subtraction of nearly equal moments.
    within = (weight[:, :, None]*(target-means[:, actual])**2).sum((1, 2))
    left, right = np.triu_indices(laws, 1)
    pair = mass[left]*mass[right]*((means[left]-means[right])**2).sum(2)
    total = mass.sum(0)
    pooled = np.divide(sums.sum(0), total[:, None], out=np.full((groups, 27), np.nan), where=total[:, None] > 0)
    if not np.allclose(pooled[total > 0].sum(1), 1, atol=1e-12, rtol=0): raise ValueError('pooled simplex')
    return dict(ids=actual, mass=mass, target_sum=sums, target_square=square,
                within=within, pair_distance=pair, pair_left=left, pair_right=right,
                group_target=pooled, frame_counts=np.bincount(actual),
                positive_mass_counts=np.array([(weight[:, actual == g] > 0).sum() for g in range(groups)]))


def resample(s, counts, chunk=128):
    """Exact pair-distance identity, with group masses rebuilt in each resample."""
    n = np.asarray(counts, float); laws, groups = s['mass'].shape
    if n.ndim != 2 or n.shape[1] != laws or np.any(n < 0) or not np.all(n.sum(1) == laws):
        raise ValueError('law multiplicities')
    same = np.array_equal(s['mass'], np.broadcast_to(s['mass'][0], s['mass'].shape))
    if same:
        reduced = np.divide(s['pair_distance'], s['mass'][0], out=np.zeros_like(s['pair_distance']), where=s['mass'][0] > 0).sum(1)
    out = np.empty((len(n), 3))
    for start in range(0, len(n), chunk):
        w = n[start:start+chunk]/laws; products = w[:, s['pair_left']]*w[:, s['pair_right']]
        if same: across = products@reduced
        else:
            masses = w@s['mass']; numerators = products@s['pair_distance']
            across = np.divide(numerators, masses, out=np.zeros_like(numerators), where=masses > 0).sum(1)
        within = w@s['within']; out[start:start+len(w)] = np.stack([within+across, within, across], axis=1)
    if not np.isfinite(out).all() or np.min(out) < -1e-12: raise ValueError('variation')
    return out


def controls():
    q = np.zeros((2, 27)); q[0, [0, 13]] = .5; q[1, [1, 12]] = .5
    # Swap the last goal while preserving each of the three marginals.
    marg = np.stack([q@(D.GOALS[:, t, None] == np.arange(3)) for t in range(3)], axis=1)
    target = q[None]; weight = np.full((1, 2), .5)
    joint = resample(sufficient(q, 'joint', target, weight), [[1]])[0]
    marginal = resample(sufficient(marg, 'marginal', target, weight), [[1]])[0]
    identical = resample(sufficient(q[:1], 'joint', q[:1][None], [[1.]]), [[1]])[0]
    return {'live:equal_marginals_different_dependence': bool(marginal[0] > 0 and joint[0] == 0),
            'positive:exact_refinement': refinement(partition(q, 'joint'), partition(marg, 'marginal'))['joint_refines_marginal'],
            'placebo:identical_target': bool(np.max(abs(identical)) == 0)}


def summarize(points, boot, cfg):
    ds, ss, bs = (cfg[k] for k in ('training_draws', 'fit_seeds', 'budgets'))
    fields = ('draw', 'seed', 'budget', 'arm', 'forecast', 'weighting', 'partition')
    index = {tuple(r[k] for k in fields): r for r in points}
    if len(index) != len(points) or set(index) != set(product(ds, ss, bs, ARMS, FORECASTS, WEIGHTS, PARTITIONS)): raise ValueError('summary roster')
    if len(bs) < 2 or any(a >= b for a, b in zip(bs, bs[1:])): raise ValueError('budgets')
    def grid(b, a, f, w, p): return np.array([[index[d, s, b, a, f, w, p]['values'] for s in ss] for d in ds])
    def report(v, b): return dict(mean=float(v.mean()), draw_means=v.mean(1).tolist(), fit_seed_means=v.mean(0).tolist(), low=float(np.quantile(b, .025)), high=float(np.quantile(b, .975)))
    estimates = []; contrasts = []; areas = []
    for a, f, w, mi in product(ARMS, FORECASTS, WEIGHTS, range(3)):
        ident = dict(arm=a, forecast=f, weighting=w, metric=METRICS[mi]); vv = []; bb = []
        for b in bs:
            for p in PARTITIONS:
                estimates.append(dict(ident, partition=p, budget=b, **report(grid(b,a,f,w,p)[:,:,mi], boot[b,a,f,w,p][:,mi])))
            v = grid(b,a,f,w,'joint')[:,:,mi]-grid(b,a,f,w,'marginal')[:,:,mi]
            bv = boot[b,a,f,w,'joint'][:,mi]-boot[b,a,f,w,'marginal'][:,mi]; vv.append(v); bb.append(bv)
            contrasts.append(dict(ident, contrast='joint-minus-marginal', budget=b, **report(v,bv)))
        area = np.trapezoid(vv, x=np.log(bs), axis=0)/np.log(bs[-1]/bs[0])
        ba = np.trapezoid(bb, x=np.log(bs), axis=0)/np.log(bs[-1]/bs[0])
        areas.append(dict(ident, contrast='joint-minus-marginal', **report(area,ba)))
    return dict(estimates=estimates, contrasts=contrasts, normalized_log_budget_area=areas)


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    parent = read(base/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if cfg[k] != parent[k]: raise ValueError('population')
    if cfg['tiers'] != ['E2-full'] or cfg['bootstrap_seed'] != 191023: raise ValueError('design')
    packets = read(base/'reader/PACKETS.json'); keys = sorted(packets['packets']); ops = []
    for k in keys:
        p = packets['packets'][k]; D.S.validate(p)
        if digest(p) != k or p['tier'] != 'E2-full': raise ValueError('packet')
        ops.append([D.S.L.OPERATIONS.index(e['operation']) for e in p['inputs']['observations']])
    codes = np.asarray(ops, int)@np.array([36,6,1]); ls = cfg['development_lineages']; refs = read(base/'evaluator/REFERENCES.json'); pos = {k:i for i,k in enumerate(keys)}
    if len(refs) != len(ls) or {r['lineage'] for r in refs} != set(ls): raise ValueError('lineage roster')
    target = np.zeros((len(ls),len(keys),27)); mass = np.zeros((len(ls),len(keys)))
    for r in refs:
        li = ls.index(r['lineage'])
        if r['tier'] != 'E2-full' or len(r['frames']) != len(keys) or {f['frame'] for f in r['frames']} != set(keys): raise ValueError('frame roster')
        for frame in r['frames']:
            i = pos[frame['frame']]; mass[li,i] = frame['mass']
            if len(dict(frame['target'])) != len(frame['target']): raise ValueError('duplicate target')
            for k,w in frame['target']:
                if not 0 <= k < 5832 or k%216 != codes[i] or w <= 0: raise ValueError('support')
                target[li,i,k//216] = w
    old = json.loads(gzip.decompress((base/'parent/goal_decision_points.json.gz').read_bytes())); index = {tuple(r[k] for k in D.FIELDS):r for r in old}
    expected = set(product(cfg['tiers'],cfg['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,cfg['training_draws'],cfg['fit_seeds']))
    if len(index) != len(old) or set(index) != expected: raise ValueError('parent roster')
    write(root/'reader/PACKETS.json',packets); write(root/'evaluator/REFERENCES.json',refs)
    np.savez_compressed(root/'evaluator/NATIVE_JOINT.npz',targets=target,weights=mass)
    write(root/'ARRAY_SCHEMA.json',dict(laws=ls,frames=keys,metrics=METRICS,partitions=PARTITIONS,membership='every frame occurs once per partition and under every law;zero masses retained',evidence_role='targets,group means,moments and scores are evaluator only',undefined='NaN pooled target iff all laws have zero group mass',bootstrap='rebuild each group mass from paired law multiplicities;pair-distance identity recomputes pooled uncertainty'))
    sample = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls))); counts = np.array([(sample==i).sum(1) for i in range(len(ls))]).T
    write(root/'BOOTSTRAP.json',dict(seed=cfg['bootstrap_seed'],resamples=cfg['bootstrap_resamples'],lineages=ls,count_digest=digest(counts.tolist())))
    (root/'groups').mkdir(); points=[]; partitions=[]; boot={}; timing=[]; parents=0; error=0.; cache={}; bindings={}
    for draw,seed,budget,arm,forecast in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS,FORECASTS):
        pulse(phase='joint-forecast-partitions',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast); tick=time.process_time()
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(base/'decisions'/(stem+'.npz'),allow_pickle=False) as z:
            q=z['probabilities']; d=D.decisions(q); marg=z['marginals']
            if not np.array_equal(z['operations'],codes) or not np.array_equal(z['coordinate'],d['coordinate']) or not np.array_equal(marg,d['marginals']): raise ValueError('decision identity')
        if np.any((q[None]<=0)&(target>0)): raise ValueError('fine support')
        logq=np.zeros_like(q); np.log(q,out=logq,where=q>0); loss=-(target*mass[:,:,None]*logq).sum((1,2))
        for li,l in enumerate(ls):
            delta=abs(loss[li]-index['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]['loss']); error=max(error,float(delta)); parents+=1
            if delta>1e-10: raise ValueError('parent loss')
        ids={p:partition(v,p) for p,v in zip(PARTITIONS,(q,marg))}; relation=refinement(ids['joint'],ids['marginal'])
        partitions.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,joint_groups=int(ids['joint'].max())+1,marginal_groups=int(ids['marginal'].max())+1,**relation))
        for weighting,w in zip(WEIGHTS,(mass,np.full_like(mass,1/len(keys)))):
            for kind,v in zip(PARTITIONS,(q,marg)):
                identity=weighting,ids[kind].tobytes()
                if identity not in cache:
                    s=sufficient(v,kind,target,w,ids[kind]); values=resample(s,np.ones((1,len(ls)),int))[0]; bv=resample(s,counts)
                    filename='groups/'+stem+'-'+weighting+'-'+kind+'.npz'
                    np.savez_compressed(root/filename,**s,values=values)
                    cache[identity]=(values,bv,filename,len(s['mass'][0]))
                values,bv,filename,ng=cache[identity];binding=stem+'-'+weighting+'-'+kind
                bindings[binding]=dict(file=filename,sha256=file_digest(root/filename),partition_membership_sha256=digest(ids[kind].tolist()))
                points.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,weighting=weighting,partition=kind,groups=ng,group_file=filename,values=values.tolist()))
                key=budget,arm,forecast,weighting,kind
                if key not in boot:boot[key]=np.zeros_like(bv)
                boot[key]+=bv/(len(cfg['training_draws'])*len(cfg['fit_seeds']))
        timing.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,cpu_seconds=time.process_time()-tick))
    (root/'joint_partition_points.json.gz').write_bytes(gzip.compress(canonical(points),mtime=0));write(root/'PARTITION_RELATIONS.json',partitions);write(root/'GROUP_BINDINGS.json',bindings);write(root/'TIMING.jsonl',dict(measurements=timing));write(root/'PARENT_REPRODUCTION.json',dict(passed=True,cells=parents,max_error=error))
    return dict(controls=checks,settings=len(partitions),group_bundles=len(points),unique_group_arrays=len(cache),packets=len(keys),parent_cells=parents,parent_max_error=error,refinement_exceptions=sum(not r['joint_refines_marginal'] for r in partitions),**summarize(points,boot,cfg),scope='exact complete joint/marginal forecast partitions;native joint-target variation within and across laws;conditional paired-law uncertainty;no fit or historical correspondence',fits=0)
