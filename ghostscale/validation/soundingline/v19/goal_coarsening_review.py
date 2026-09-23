"""Independent integer partitions, partial reports, fine-support scores and regroup.

Saved binary64 projections are checked against scalar sums before exact decisions.
No producer geometry, policy, score or aggregation implementation is imported.
"""
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .joint_factorization_review import close, references
from .witnessed_goal_review import witness, validate
from .goal_abstention_review import compare_groups

ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
FORECASTS = ('restricted', 'goal-product')
COSTS = (.1, .25, .5)
PARTITIONS = {'fine': (0, 1, 2), 'meaning': (0, 1, 1),
              'dependency': (1, 0, 1), 'presentation': (1, 1, 0), 'one-class': (0, 0, 0)}
CHOICES = ('mapped-fine', 'coarse-mode')
POLICIES = ('always', 'abstain')
FIELDS = ('tier', 'budget', 'arm', 'forecast', 'partition', 'choice', 'policy', 'cost', 'lineage', 'draw', 'seed')
METRICS = ('coverage', 'correct', 'incorrect', 'cost_value', 'forecast_cost', 'incompatible',
           'full_incompatible', 'coarse_loss', 'fine_loss', 'alternatives')
PAIRED = ('coverage', 'incorrect', 'cost_value', 'coarse_loss', 'alternatives')
GOALS = tuple((k//9, k//3 % 3, k % 3) for k in range(27))


def geometry(partition):
    mapping = PARTITIONS[partition]; n = max(mapping)+1
    paths = tuple(product(range(n), repeat=3))
    codes = tuple(n*n*mapping[a]+n*mapping[b]+mapping[c] for a,b,c in GOALS)
    reports = tuple(product(range(-1, n), repeat=3))
    hits = np.array([[sum(v >= 0 and v == g[t] for t,v in enumerate(r))/3 for g in paths] for r in reports])
    fine_hits = np.array([[sum(v >= 0 and v == mapping[g[t]] for t,v in enumerate(r))/3 for g in GOALS] for r in reports])
    compatible = np.array([[all(v < 0 or v == mapping[g[t]] for t,v in enumerate(r)) for g in GOALS] for r in reports])
    coverage = np.array([sum(v >= 0 for v in r)/3 for r in reports])
    alternatives = np.array([sum(mapping.count(v) for v in r if v >= 0)/3 for r in reports])
    return dict(n=n, mapping=mapping, paths=paths, codes=codes, reports=reports,
        report_index={r:i for i,r in enumerate(reports)}, hits=hits,
        fine_hits=fine_hits, compatible=compatible, coverage=coverage, alternatives=alternatives)


def project(q, geo):
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] != 27 or not np.isfinite(q).all() or np.any(q < 0): raise ValueError('fine forecast')
    for row in q: close(math.fsum(row.tolist()), 1., 4e-12)
    groups = [[k for k,c in enumerate(geo['codes']) if c == code] for code in range(geo['n']**3)]
    p = np.array([[math.fsum(float(row[k]) for k in group) for group in groups] for row in q])
    return p


def marginals(p, paths, n):
    return np.array([[[math.fsum(float(row[k]) for k,g in enumerate(paths) if g[t] == v)
                       for v in range(n)] for t in range(3)] for row in p])


def choose(fine_marg, coarse_marg, geo, choice, cost, policy):
    if choice not in CHOICES or cost not in COSTS or policy not in POLICIES: raise ValueError('policy')
    result=[]
    for fm,cm in zip(fine_marg,coarse_marg):
        r=[]
        for t in range(3):
            v = geo['mapping'][max(range(3),key=lambda k:float(fm[t,k]))] if choice=='mapped-fine' else max(range(geo['n']),key=lambda k:float(cm[t,k]))
            r.append(v if policy=='always' or 1.-float(cm[t,v]) < cost else -1)
        result.append(r)
    return np.array(result,dtype=int)


def tables(truth, mass, geo):
    """Accumulate directly over fine labels; never project evaluator targets."""
    truth=np.asarray(truth);mass=np.asarray(mass)
    if (truth.ndim!=3 or truth.shape[-1]!=27 or truth.shape[:2]!=mass.shape
        or not np.isfinite(truth).all() or not np.isfinite(mass).all()
        or np.any(truth<0) or np.any(mass<0)
        or not np.allclose(truth.sum(2),1,rtol=0,atol=1e-10)
        or not np.allclose(mass.sum(1),1,rtol=0,atol=1e-10)):raise ValueError('native law')
    correct=np.zeros((*mass.shape,len(geo['reports'])))
    possible=np.zeros(correct.shape,dtype=bool)
    for k in range(27):
        correct += truth[:,:,k,None]*geo['fine_hits'][:,k]
        possible |= (truth[:,:,k,None]>0) & geo['compatible'][:,k]
    return dict(truth=truth,mass=mass,correct=correct,possible=possible,geo=geo)


def score(q,p,report,table,cost):
    geo=table['geo'];mass=table['mass'];truth=table['truth']
    if report.shape!=(len(q),3) or cost not in COSTS:raise ValueError('report shape or cost')
    if p.shape!=(len(q),geo['n']**3) or np.any(p<0) or not np.isfinite(p).all():raise ValueError('coarse forecast')
    try:codes=np.array([geo['report_index'][tuple(r)] for r in report])
    except KeyError as e:raise ValueError('report vocabulary') from e
    frame=np.arange(len(q));cov=geo['coverage'][codes]
    coverage=np.sum(mass*cov,axis=1)
    correct=np.sum(mass*table['correct'][:,frame,codes],axis=1)
    incorrect=coverage-correct
    unsupported=~table['possible'][:,frame,codes]
    forecast_correct=np.sum(p*geo['hits'][codes],axis=1)
    coarse_loss=np.zeros(len(mass));fine_loss=np.zeros(len(mass))
    for k,c in enumerate(geo['codes']):
        if np.any((truth[:,:,k]>0)&((q[:,k]<=0)|(p[:,c]<=0))):raise ValueError('zero true probability')
        fine_log=np.array([-math.log(float(v)) if v>0 else 0. for v in q[:,k]])
        coarse_log=np.array([-math.log(float(v)) if v>0 else 0. for v in p[:,c]])
        fine_loss+=np.sum(mass*truth[:,:,k]*fine_log,axis=1)
        coarse_loss+=np.sum(mass*truth[:,:,k]*coarse_log,axis=1)
    return dict(coverage=coverage,correct=correct,incorrect=incorrect,cost_value=incorrect+cost*(1-coverage),
        forecast_cost=np.sum(mass*(cov-forecast_correct+cost*(1-cov)),axis=1),
        incompatible=np.sum(mass*(unsupported & (cov>0)),axis=1),
        full_incompatible=np.sum(mass*(unsupported & (cov==1)),axis=1),
        coarse_loss=coarse_loss,fine_loss=fine_loss,alternatives=np.sum(mass*geo['alternatives'][codes],axis=1))


def controls():
    q=np.zeros((1,27));q[0,[0,13,26]]=[.4,.3,.3]
    g=geometry('meaning');p=project(q,g);fm=marginals(q,GOALS,3);cm=marginals(p,g['paths'],g['n'])
    a=choose(fm,cm,g,'mapped-fine',.1,'always');b=choose(fm,cm,g,'coarse-mode',.1,'always')
    truth=np.zeros((1,1,27));truth[0,0,13]=1
    v=score(q,p,b,tables(truth,np.ones((1,1)),g),.1)
    u=np.ones((1,27))/27;fg=geometry('fine');empty=choose(marginals(u,GOALS,3),marginals(u,GOALS,3),fg,'coarse-mode',.1,'abstain')
    return {'live:noncommuting_modes':bool(np.all(a==0) and np.all(b==1)),
        'positive:broad_correct_fine_unknown':bool(v['correct'][0]==1 and v['alternatives'][0]==2),
        'placebo:uniform_fine_abstains':bool(np.all(empty==-1)),
        'positive:one_class_mass':bool(abs(project(q,geometry('one-class'))[0,0]-1)<1e-14)}


def regroup(rows,cfg):
    ls,ds,ss,bs=(cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    idx={tuple(r[k] for k in FIELDS):r for r in rows}
    if len(idx)!=len(rows) or set(idx)!=set(product(cfg['tiers'],bs,ARMS,FORECASTS,PARTITIONS,CHOICES,POLICIES,COSTS,ls,ds,ss)):raise ValueError('stratum roster')
    if len(bs)<2 or any(a>=b for a,b in zip(bs,bs[1:])):raise ValueError('budget order')
    sample=np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls)))
    counts=np.array([np.bincount(s,minlength=len(ls)) for s in sample])
    mean=lambda x:math.fsum(x)/len(x)
    def estimate(v):
        line=[mean([v[l,d,s] for d,s in product(ds,ss)]) for l in ls];boot=counts@np.array(line)/len(ls)
        return dict(mean=mean(line),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),lineage_values=line,
            draw_means=[mean([v[l,d,s] for l,s in product(ls,ss)]) for d in ds],feature_seed_means=[mean([v[l,d,s] for l,d in product(ls,ds)]) for s in ss])
    means=[];contrasts=[];areas=[]
    for tier,arm,forecast,partition,cost in product(cfg['tiers'],ARMS,FORECASTS,PARTITIONS,COSTS):
        for choice,policy,budget in product(CHOICES,POLICIES,bs):
            rr=[idx[tier,budget,arm,forecast,partition,choice,policy,cost,l,d,s] for l,d,s in product(ls,ds,ss)]
            m={k:mean([r[k] for r in rr]) for k in METRICS}
            means.append(dict(tier=tier,arm=arm,forecast=forecast,partition=partition,choice=choice,policy=policy,cost=cost,budget=budget,
                conditional_error=m['incorrect']/m['coverage'] if m['coverage']>0 else None,
                alternatives_per_claim=m['alternatives']/m['coverage'] if m['coverage']>0 else None,**m))
        comparisons=[(c,'abstain',c,'always') for c in CHOICES]
        if partition not in ('fine','one-class'):comparisons.extend([('coarse-mode',p,'mapped-fine',p) for p in POLICIES])
        for (choice,policy,base_choice,base_policy),metric in product(comparisons,PAIRED):
            identity=dict(tier=tier,arm=arm,forecast=forecast,partition=partition,cost=cost,choice=choice,policy=policy,base_choice=base_choice,base_policy=base_policy,metric=metric)
            curves={}
            for budget in bs:
                v={(l,d,s):idx[tier,budget,arm,forecast,partition,choice,policy,cost,l,d,s][metric]-idx[tier,budget,arm,forecast,partition,base_choice,base_policy,cost,l,d,s][metric] for l,d,s in product(ls,ds,ss)}
                curves[budget]=v;contrasts.append(dict(identity,budget=budget,**estimate(v)))
            v={k:math.fsum((curves[x][k]+curves[y][k])*.5*math.log(y/x) for x,y in zip(bs,bs[1:]))/math.log(bs[-1]/bs[0]) for k in product(ls,ds,ss)}
            areas.append(dict(identity,**estimate(v)))
    return dict(means=means,contrasts=contrasts,normalized_log_budget_area=areas)


def load(path,frames):
    with np.load(path,allow_pickle=False) as z:saved={k:z[k] for k in z.files}
    expected={p+'-'+k for p in PARTITIONS for k in ('probabilities','marginals')}|{f'{p}-{ch}-{c}-{pol}' for p,ch,c,pol in product(PARTITIONS,CHOICES,COSTS,POLICIES)}
    if set(saved)!=expected:raise ValueError('saved fields')
    for p,m in PARTITIONS.items():
        n=max(m)+1
        for k,shape in [('probabilities',(frames,n**3)),('marginals',(frames,3,n))]:
            if saved[p+'-'+k].shape!=shape:raise ValueError('saved shape')
        for ch,c,pol in product(CHOICES,COSTS,POLICIES):
            if saved[f'{p}-{ch}-{c}-{pol}'].shape!=(frames,3):raise ValueError('report shape')
    return saved


def reconstruct(q,saved,prepared):
    fm=marginals(q,GOALS,3);error=close(fm,saved['fine-marginals'],1e-12)
    # Saved fine marginals use the same exact binary64 projection as the fine
    # coordinate parent. Scalar sums validate values, without changing ties.
    fm=saved['fine-marginals'];values={}
    for part in PARTITIONS:
        table=prepared[part];g=table['geo'];p=project(q,g)
        error=max(error,close(p,saved[part+'-probabilities'],1e-12))
        p=saved[part+'-probabilities'];cm=marginals(p,g['paths'],g['n'])
        error=max(error,close(cm,saved[part+'-marginals'],1e-12));cm=saved[part+'-marginals']
        for ch,c,pol in product(CHOICES,COSTS,POLICIES):
            r=choose(fm,cm,g,ch,c,pol)
            if not np.array_equal(r,saved[f'{part}-{ch}-{c}-{pol}']):raise ValueError('report choice')
            values[part,ch,c,pol]=score(q,p,r,table,c)
    return values,error


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
    if design['tiers']!=['E2-full'] or design['costs']!=list(COSTS) or design['partitions']!={k:list(v) for k,v in PARTITIONS.items()}:raise ValueError('design')
    packets=read(original/'reader/PACKETS.json');refs=read(original/'evaluator/REFERENCES.json')
    if packets!=read(parent/'reader/PACKETS.json') or refs!=read(parent/'evaluator/REFERENCES.json'):raise ValueError('evidence changed')
    keys=sorted(packets['packets']);ops=[]
    for key in keys:
        packet=packets['packets'][key];validate(packet)
        if digest(packet)!=key:raise ValueError('packet identity')
        ops.append(witness(packet))
    ls=design['development_lineages'];truth=np.zeros((len(ls),len(keys),27));mass=np.zeros((len(ls),len(keys)))
    for i,(labels,target,weights) in enumerate(references(refs,'E2-full',keys,ls)):
        if any(int(k)%216!=ops[i] for k in labels):raise ValueError('native support')
        truth[:,i,labels//216]=target;mass[:,i]=weights
    prepared={p:tables(truth,mass,geometry(p)) for p in PARTITIONS}
    raw=json.loads(gzip.decompress((original/'goal_coarsening_points.json.gz').read_bytes()))
    original_groups=regroup(raw,design);idx={tuple(r[k] for k in FIELDS):r for r in raw}
    old=json.loads(gzip.decompress((parent/'parent/goal_abstention_points.json.gz').read_bytes()))
    oldfields=('tier','budget','arm','forecast','policy','cost','lineage','draw','seed')
    oi={tuple(r[k] for k in oldfields):r for r in old}
    expected=set(product(design['tiers'],design['budgets'],ARMS,FORECASTS,('always-coordinate','joint-all-or-none','step-abstention'),COSTS,ls,design['training_draws'],design['fit_seeds']))
    if len(oi)!=len(old) or set(oi)!=expected:raise ValueError('parent roster')
    native=read(original/'evaluator/NATIVE_SCORES.json');nf=('lineage','partition','choice','cost','policy');ni={tuple(r[k] for k in nf):r for r in native}
    if len(ni)!=len(native) or set(ni)!=set(product(ls,PARTITIONS,CHOICES,COSTS,POLICIES)):raise ValueError('native roster')
    error=dict(projection=0.,score=0.,parent=0.);rows=[];native_rows=[];reproduced=frames=0
    def row(values,li):
        v={k:float(x[li]) for k,x in values.items()}
        return dict(v,conditional_error=v['incorrect']/v['coverage'] if v['coverage']>0 else None,
            alternatives_per_claim=v['alternatives']/v['coverage'] if v['coverage']>0 else None)
    def check(v,saved,identity):
        if set(saved)!=set(v)|set(identity) or any(saved[k]!=x for k,x in identity.items()):raise ValueError('row identity')
        for k,x in v.items():
            if x is None or saved[k] is None:
                if x!=saved[k]:raise ValueError('undefined ratio')
            else:error['score']=max(error['score'],close(x,saved[k]))
    for li,l in enumerate(ls):
        pulse(phase='independent-native-coarsening',lineage=l)
        pp={p:tables(truth[li:li+1],mass[li:li+1],geometry(p)) for p in PARTITIONS}
        values,e=reconstruct(truth[li],load(original/'evaluator'/f'native-{l}.npz',len(keys)),pp);error['projection']=max(error['projection'],e)
        for (p,ch,c,pol),totals in values.items():
            v=row(totals,0);identity=dict(tier='E2-full',lineage=l,partition=p,choice=ch,cost=c,policy=pol)
            check(v,ni[l,p,ch,c,pol],identity);native_rows.append(dict(identity,**v))
    for draw,seed,budget,arm,forecast in product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS,FORECASTS):
        pulse(phase='independent-learned-coarsening',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast)
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(parent/'reports'/(stem+'.npz'),allow_pickle=False) as z:q=z['probabilities']
        if q.shape!=(len(keys),27):raise ValueError('forecast roster')
        values,e=reconstruct(q,load(original/'reports'/(stem+'.npz'),len(keys)),prepared);error['projection']=max(error['projection'],e);frames+=len(keys)
        for (p,ch,c,pol),totals in values.items():
            for li,l in enumerate(ls):
                v=row(totals,li);identity=dict(tier='E2-full',budget=budget,arm=arm,forecast=forecast,partition=p,choice=ch,cost=c,policy=pol,lineage=l,draw=draw,seed=seed,frames=len(keys))
                check(v,idx[tuple(identity[k] for k in FIELDS)],identity);rows.append(dict(identity,**v))
                if p=='fine' and ch=='coarse-mode':
                    before=oi['E2-full',budget,arm,forecast,'always-coordinate' if pol=='always' else 'step-abstention',c,l,draw,seed]
                    for k in METRICS:
                        if k in before:error['parent']=max(error['parent'],close(v[k],before[k]))
                    error['parent']=max(error['parent'],close([v['fine_loss'],v['coarse_loss']],[before['loss']]*2));reproduced+=1
    summary=read(original/'SUMMARY.json')
    for k,v in dict(rows=len(rows),native_rows=len(native_rows),packets=len(keys),frame_forecasts=frames,parent_cells=reproduced).items():
        if summary[k]!=v:raise ValueError('coverage')
    groups=regroup(rows,design);expected={k:summary[k] for k in groups}
    error['regroup']=compare_groups(groups,expected);error['original_regroup']=compare_groups(original_groups,expected)
    write(root/'INDEPENDENT_REGROUP.json',groups);write(root/'ORIGINAL_ROW_REGROUP.json',original_groups)
    write(root/'NATIVE_RECONSTRUCTION.json',native_rows)
    (root/'reconstructed_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    result=dict(passed=True,rows=len(rows),native_rows=len(native_rows),packets=len(keys),frame_forecasts=frames,parent_cells=reproduced,
        controls=checks,**{'max_'+k+'_error':v for k,v in error.items()},scope='integer path projection, scalar sums, fine-support score reconstruction and paired lineage multiplicities; saved binary64 probabilities checked before exact choices; broad correctness is not fine-goal recovery')
    write(root/'INDEPENDENT_REVIEW.json',result);return result
