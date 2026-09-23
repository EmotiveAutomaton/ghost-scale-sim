"""Independent scalar report policies, sparse native scores and paired estimates.

Binary64 marginal arithmetic is checked against accurate scalar sums before
strict threshold choices. No producer policy, score or aggregation is imported.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .goal_decision_review import GOALS, ARMS, FORECASTS, DECISION_FIELDS, decide
from .joint_factorization_review import close, references
from .witnessed_goal_review import witness, validate

POLICIES = ('always-coordinate', 'joint-all-or-none', 'step-abstention')
COSTS = (.1, .25, .5)
FIELDS = ('tier', 'budget', 'arm', 'forecast', 'policy', 'cost', 'lineage', 'draw', 'seed')
METRICS = ('coverage', 'correct', 'incorrect', 'cost_value', 'forecast_cost',
           'incompatible', 'full_incompatible', 'loss')


def choose(q, marg, cost):
    """One frame: exact executed modes and thresholds after scalar validation."""
    if cost not in COSTS: raise ValueError('undeclared cost')
    d, _ = decide(q)
    close(marg, d['marginals'], 1e-12)
    coordinate = tuple(max(range(3), key=lambda k: float(marg[t, k])) for t in range(3))
    joint = GOALS[max(range(27), key=lambda k: float(q[k]))]
    risks = [1.-float(marg[t, joint[t]]) for t in range(3)]
    # Explicit binary64 left-to-right addition matches a length-three mean.
    joint_risk = ((risks[0]+risks[1])+risks[2])/3
    return {POLICIES[0]: coordinate,
            POLICIES[1]: joint if joint_risk < cost else (-1, -1, -1),
            POLICIES[2]: tuple(g if 1.-float(marg[t, g]) < cost else -1 for t, g in enumerate(coordinate))}


def scores(q, report, reference, cost):
    """Enumerate goal labels for a partial report; vectorize only over lineages."""
    if len(report) != 3 or any(v not in (-1, 0, 1, 2) for v in report): raise ValueError('report')
    labels, target, mass = reference
    active = sum(v >= 0 for v in report); coverage = active/3
    hits = [sum(v >= 0 and v == g[t] for t, v in enumerate(report)) for g in GOALS]
    compatible = [all(v < 0 or v == g[t] for t, v in enumerate(report)) for g in GOALS]
    if any(q[int(k)] <= 0 for k in labels): raise ValueError('zero true probability')
    correct = target @ np.array([hits[int(k)]/3 for k in labels])
    incorrect = coverage-correct
    extension = (target[:, [i for i, k in enumerate(labels) if compatible[int(k)]]] > 0).any(1)
    forecast_cost = math.fsum(float(q[k])*((active-hits[k])+cost*(3-active))/3 for k in range(27))
    value = dict(coverage=np.full(len(mass), coverage), correct=correct, incorrect=incorrect,
        cost_value=incorrect+cost*(1-coverage), forecast_cost=np.full(len(mass), forecast_cost),
        incompatible=(~extension) & (active > 0), full_incompatible=(~extension) & (active == 3),
        loss=target @ np.array([-math.log(float(q[int(k)])) for k in labels]))
    return {k:mass*v for k,v in value.items()}


def controls():
    q = np.zeros(27); q[[0, 1, 2]] = 1/3; d, _ = decide(q)
    p = choose(q, d['marginals'], .25)
    u = np.full(27, 1/27); ud, _ = decide(u); r = choose(u, ud['marginals'], .1)
    v = scores(u, r[POLICIES[2]], (np.arange(27), u[None], np.ones(1)), .1)
    edge = np.zeros(27); edge[[0, 13]] = [.75, .25]; ed, _ = decide(edge)
    return {'live:partial_report': p[POLICIES[2]] == (0, 0, -1),
            'placebo:empty_report': bool(r[POLICIES[2]] == (-1, -1, -1) and v['coverage'][0] == 0),
            'positive:empty_cost': bool(abs(v['cost_value'][0]-.1) < 1e-14),
            'positive:threshold_equality': choose(edge, ed['marginals'], .25)[POLICIES[2]] == (-1, -1, -1)}


def regroup(rows, cfg):
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    idx = {tuple(r[k] for k in FIELDS):r for r in rows}
    if len(idx) != len(rows) or set(idx) != set(product(cfg['tiers'],bs,ARMS,FORECASTS,POLICIES,COSTS,ls,ds,ss)):
        raise ValueError('stratum roster')
    if len(bs) < 2 or any(a >= b for a,b in zip(bs,bs[1:])): raise ValueError('budget order')
    sample = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls)))
    counts = np.array([np.bincount(s,minlength=len(ls)) for s in sample])
    mean = lambda v: math.fsum(v)/len(v)
    def estimate(values):
        line = [mean([values[l,d,s] for d,s in product(ds,ss)]) for l in ls]
        boot = counts @ np.array(line)/len(ls)
        return dict(mean=mean(line),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),
            lineage_values=line,draw_means=[mean([values[l,d,s] for l,s in product(ls,ss)]) for d in ds],
            feature_seed_means=[mean([values[l,d,s] for l,d in product(ls,ds)]) for s in ss])
    means=[];contrasts=[];areas=[]
    for tier,arm,forecast,cost in product(cfg['tiers'],ARMS,FORECASTS,COSTS):
        for policy,budget in product(POLICIES,bs):
            rr=[idx[tier,budget,arm,forecast,policy,cost,l,d,s] for l,d,s in product(ls,ds,ss)]
            m={k:mean([r[k] for r in rr]) for k in METRICS}
            means.append(dict(tier=tier,arm=arm,forecast=forecast,policy=policy,cost=cost,budget=budget,
                conditional_error=m['incorrect']/m['coverage'] if m['coverage']>0 else None,**m))
        for policy,metric in product(POLICIES[1:],METRICS):
            curves={};identity=dict(tier=tier,arm=arm,forecast=forecast,policy=policy,cost=cost,metric=metric,baseline=POLICIES[0])
            for budget in bs:
                values={(l,d,s):idx[tier,budget,arm,forecast,policy,cost,l,d,s][metric]-idx[tier,budget,arm,forecast,POLICIES[0],cost,l,d,s][metric] for l,d,s in product(ls,ds,ss)}
                curves[budget]=values;contrasts.append(dict(identity,budget=budget,**estimate(values)))
            area={k:math.fsum((curves[x][k]+curves[y][k])*.5*math.log(y/x) for x,y in zip(bs,bs[1:]))/math.log(bs[-1]/bs[0]) for k in product(ls,ds,ss)}
            areas.append(dict(identity,**estimate(area)))
    return dict(means=means,contrasts=contrasts,normalized_log_budget_area=areas)


def compare_groups(actual, expected):
    error=0.
    if set(actual)!=set(expected): raise ValueError('summary groups')
    for k in actual:
        if len(actual[k])!=len(expected[k]): raise ValueError('summary roster')
        for a,b in zip(actual[k],expected[k]):
            if set(a)!=set(b): raise ValueError('summary fields')
            for n in a:
                if isinstance(a[n],str) or a[n] is None or b[n] is None:
                    if a[n]!=b[n]: raise ValueError('summary identity or undefined ratio')
                else:error=max(error,close(a[n],b[n]))
    return error


def run(root, plan, pulse):
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    cfg=plan['design'];base=root/'inputs';original=base/'original';parent=base/'parent'
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target plan')
    design=read(original/'PLAN.json')['design'];prior=read(parent/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if design[k]!=prior[k]:raise ValueError('population')
    if design['tiers']!=['E2-full'] or design['costs']!=list(COSTS):raise ValueError('evidence or costs')
    packets=read(original/'reader/PACKETS.json')
    if packets!=read(parent/'reader/PACKETS.json'):raise ValueError('reader changed')
    keys=sorted(packets['packets']);ops=[]
    for k in keys:
        p=packets['packets'][k];validate(p)
        if digest(p)!=k:raise ValueError('packet identity')
        ops.append(witness(p))
    refs=read(original/'evaluator/REFERENCES.json')
    if refs!=read(parent/'evaluator/REFERENCES.json'):raise ValueError('references changed')
    ls=design['development_lineages'];reference=[]
    for op,(labels,target,mass) in zip(ops,references(refs,'E2-full',keys,ls)):
        if any(int(k)%216!=op for k in labels):raise ValueError('native support')
        reference.append((labels//216,target,mass))
    raw=json.loads(gzip.decompress((original/'goal_abstention_points.json.gz').read_bytes()))
    original_groups=regroup(raw,design);idx={tuple(r[k] for k in FIELDS):r for r in raw}
    old=json.loads(gzip.decompress((parent/'parent/goal_decision_points.json.gz').read_bytes()))
    old_fields=('tier','budget','arm','lineage','draw','seed')
    old_idx={tuple(r[k] for k in old_fields):r for r in old}
    expected=set(product(design['tiers'],design['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,design['training_draws'],design['fit_seeds']))
    if len(old_idx)!=len(old) or set(old_idx)!=expected:raise ValueError('parent roster')
    native=read(original/'evaluator/NATIVE_SCORES.json');ni={(r['lineage'],r['cost'],r['policy']):r for r in native}
    if len(ni)!=len(native) or set(ni)!=set(product(ls,COSTS,POLICIES)):raise ValueError('native roster')
    error=dict(probability=0.,marginal=0.,score=0.,parent=0.)
    def compare_row(values,saved,identity):
        if set(saved)!=set(values)|set(identity) or any(saved[k]!=v for k,v in identity.items()):raise ValueError('row identity')
        for k,v in values.items():
            if v is None or saved[k] is None:
                if v!=saved[k]:raise ValueError('undefined conditional error')
            else:error['score']=max(error['score'],close(v,saved[k]))
    def load(path):
        with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        expected={'probabilities'}|{f'{cost}-{p}' for cost,p in product(COSTS,POLICIES)}
        if set(v)!=expected or v['probabilities'].shape!=(len(keys),27):raise ValueError('report fields or shape')
        if any(v[k].shape!=(len(keys),3) for k in v if k!='probabilities'):raise ValueError('report shape')
        return v
    def check_frames(q,marg,saved,refs):
        if not np.array_equal(q,saved['probabilities']):raise ValueError('changed probabilities')
        totals={(cost,p):defaultdict(list) for cost,p in product(COSTS,POLICIES)}
        for i,ref in enumerate(refs):
            d,_=decide(q[i]);error['marginal']=max(error['marginal'],close(d['marginals'],marg[i],1e-12))
            for cost in COSTS:
                reports=choose(q[i],marg[i],cost)
                for p in POLICIES:
                    if not np.array_equal(reports[p],saved[f'{cost}-{p}'][i]):raise ValueError('report choice')
                    for k,v in scores(q[i],reports[p],ref,cost).items():totals[cost,p][k].append(v)
        return {key:{k:np.sum(v,axis=0) for k,v in values.items()} for key,values in totals.items()}
    def row_values(values,li):
        v={k:float(a[li]) for k,a in values.items()}
        return dict(v,conditional_error=v['incorrect']/v['coverage'] if v['coverage']>0 else None)
    native_rows=[]
    for li,l in enumerate(ls):
        pulse(phase='independent-native-goal-abstention',lineage=l)
        q=np.zeros((len(keys),27));rr=[]
        for i,(labels,target,mass) in enumerate(reference):q[i,labels]=target[li];rr.append((labels,target[li:li+1],mass[li:li+1]))
        # Native records save probabilities but no marginals. Reproduce only this
        # arithmetic primitive, then independently validate every sum and choice.
        marg=np.stack([q @ (np.array(GOALS)[:,t,None]==np.arange(3)) for t in range(3)],axis=1)
        for (cost,p),values in check_frames(q,marg,load(original/'evaluator'/f'native-{l}.npz'),rr).items():
            v=row_values(values,0);identity=dict(tier='E2-full',lineage=l,cost=cost,policy=p)
            compare_row(v,ni[l,cost,p],identity);native_rows.append(dict(identity,**v))
    rows=[];frames=reproduced=0
    for draw,seed,budget,arm,forecast in product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS,FORECASTS):
        pulse(phase='independent-learned-goal-abstention',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast)
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(parent/'decisions'/(stem+'.npz'),allow_pickle=False) as z:inherited={k:z[k] for k in z.files}
        if set(inherited)!=DECISION_FIELDS|{'probabilities','operations'} or not np.array_equal(inherited['operations'],ops):raise ValueError('parent decisions')
        q=inherited['probabilities'];marg=inherited['marginals']
        if q.shape!=(len(keys),27) or marg.shape!=(len(keys),3,3):raise ValueError('parent shapes')
        for i in range(len(keys)):decide(q[i],{k:inherited[k][i] for k in DECISION_FIELDS})
        values=check_frames(q,marg,load(original/'reports'/(stem+'.npz')),reference);frames+=len(keys)
        for (cost,p),totals in values.items():
            for li,l in enumerate(ls):
                v=row_values(totals,li);identity=dict(tier='E2-full',budget=budget,arm=arm,forecast=forecast,policy=p,cost=cost,lineage=l,draw=draw,seed=seed,frames=len(keys))
                compare_row(v,idx[tuple(identity[k] for k in FIELDS)],identity);rows.append(dict(identity,**v))
                if cost==COSTS[0] and p==POLICIES[0]:
                    previous=old_idx['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]
                    joint_previous=old_idx['E2-full',budget,arm+'-'+forecast+'-joint',l,draw,seed]
                    error['parent']=max(error['parent'],close([v['loss'],v['correct'],v['loss']],
                        [previous['loss'],previous['goal_accuracy'],joint_previous['loss']]))
                    reproduced+=1
    summary=read(original/'SUMMARY.json')
    for k,v in dict(rows=len(rows),native_rows=len(native_rows),packets=len(keys),frame_forecasts=frames,parent_cells=reproduced).items():
        if summary[k]!=v:raise ValueError('coverage')
    groups=regroup(rows,design);expected={k:summary[k] for k in groups}
    error['regroup']=compare_groups(groups,expected);error['original_regroup']=compare_groups(original_groups,expected)
    write(root/'INDEPENDENT_REGROUP.json',groups);write(root/'ORIGINAL_ROW_REGROUP.json',original_groups)
    write(root/'NATIVE_RECONSTRUCTION.json',native_rows)
    (root/'reconstructed_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    result=dict(passed=True,rows=len(rows),native_rows=len(native_rows),packets=len(keys),frame_forecasts=frames,parent_cells=reproduced,
        controls=checks,**{'max_'+k+'_error':v for k,v in error.items()},
        scope='independent scalar report policies and native scores;verified binary64 marginal arithmetic before exact thresholds;paired conditional-fit estimates;no historical or human-intent claim')
    write(root/'INDEPENDENT_REVIEW.json',result);return result
