"""Independent exact classes, scalar target moments and joint partition review."""
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .goal_class_reliability_review import marginals, compare
from .joint_factorization_review import close, references
from .witnessed_goal_review import witness, validate
from .joint_partition_regroup import resample

ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
FORECASTS = ('restricted', 'goal-product')
WEIGHTS = ('native', 'equal-frame')
PARTITIONS = ('joint', 'marginal')
METRICS = ('total_variation', 'within_law_variation', 'across_law_variation')


def membership(p, kind):
    p = np.asarray(p, float)
    shape = (27,) if kind == 'joint' else (3, 3) if kind == 'marginal' else None
    if shape is None or p.ndim != len(shape)+1 or p.shape[1:] != shape:
        raise ValueError('forecast shape')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1+1e-12):
        raise ValueError('forecast probability')
    if any(abs(math.fsum(v)-1) > 1e-12 for v in p.reshape(-1, shape[-1]).tolist()):
        raise ValueError('forecast simplex')
    keys = [tuple(v) for v in p.reshape(len(p), -1).tolist()]
    ordered = sorted(range(len(p)), key=keys.__getitem__)
    groups = []
    for i in ordered:
        if not groups or keys[i] != keys[groups[-1][0]]: groups.append([])
        groups[-1].append(i)
    groups.sort(key=lambda v: min(v))
    ids = np.empty(len(p), np.int64)
    for g, members in enumerate(groups): ids[members] = g
    return ids, groups


def sufficient(p, kind, target, weight):
    ids, groups = membership(p, kind)
    target, weight = np.asarray(target, float), np.asarray(weight, float)
    if weight.ndim != 2 or target.shape != (*weight.shape, 27) or len(ids) != weight.shape[1]:
        raise ValueError('target shape')
    if not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1+1e-12):
        raise ValueError('target probability')
    if any(abs(math.fsum(v)-1) > 1e-12 for v in target.reshape(-1, 27).tolist()):
        raise ValueError('target simplex')
    if not np.isfinite(weight).all() or np.any(weight < 0) or any(abs(math.fsum(w)-1)>1e-12 for w in weight.tolist()):
        raise ValueError('weight')
    laws, frames = weight.shape; ng = len(groups)
    mass = np.zeros((laws, ng)); sums = np.zeros((laws, ng, 27))
    square = np.zeros(laws); within = np.zeros(laws)
    for l in range(laws):
        w = weight[l].tolist(); t = target[l].tolist()
        mass[l] = [math.fsum(w[i] for i in members) for members in groups]
        # Singleton groups are exact scalar products, including zero masses.
        for g, members in enumerate(groups):
            if len(members) == 1:
                i = members[0]; sums[l, g] = [w[i]*v for v in t[i]]
            else:
                sums[l, g] = [math.fsum(w[i]*t[i][k] for i in members) for k in range(27)]
        square[l] = math.fsum(w[i]*math.fsum(v*v for v in t[i]) for i in range(frames))
        within[l] = math.fsum(w[i]*math.fsum((t[i][k]-sums[l,g,k]/mass[l,g])**2 for k in range(27))
            for g,members in enumerate(groups) if mass[l,g]>0 and len(members)>1 for i in members)
    mean = np.divide(sums, mass[:,:,None], out=np.zeros_like(sums), where=mass[:,:,None]>0)
    pairs = list(product(range(laws), repeat=2)); pairs = [(l,r) for l,r in pairs if l<r]
    left = np.array([l for l,r in pairs], int); right = np.array([r for l,r in pairs], int)
    # Expanded squares independently check the producer's centered differences.
    second = (mean*mean).sum(2)
    pair = mass[left]*mass[right]*(second[left]+second[right]-2*(mean[left]*mean[right]).sum(2))
    pooled = np.full((ng,27), np.nan)
    for g in range(ng):
        den = math.fsum(mass[:,g])
        if den > 0: pooled[g] = [math.fsum(sums[:,g,k])/den for k in range(27)]
    return dict(ids=ids,mass=mass,target_sum=sums,target_square=square,within=within,
        pair_distance=pair,pair_left=left,pair_right=right,group_target=pooled,
        frame_counts=np.array(list(map(len,groups))),
        positive_mass_counts=np.array([sum(weight[l,i]>0 for l in range(laws) for i in members) for members in groups]))


def arrays(actual, saved):
    if set(actual) != set(saved): raise ValueError('array fields')
    error = 0.
    for k, x in actual.items():
        x,y = np.asarray(x),np.asarray(saved[k])
        if x.shape != y.shape or not np.array_equal(np.isnan(x),np.isnan(y)): raise ValueError('array shape/undefined')
        if k in ('ids','frame_counts','positive_mass_counts','pair_left','pair_right'):
            if not np.array_equal(x,y): raise ValueError('exact membership/counts')
        else: error = max(error, close(x[~np.isnan(x)],y[~np.isnan(y)],1e-12))
    return error


def relation(joint, marginal):
    jm = {}; mj = {}
    for j,m in zip(joint,marginal):
        jm.setdefault(int(j),set()).add(int(m)); mj.setdefault(int(m),set()).add(int(j))
    bad = sorted(j for j,v in jm.items() if len(v)>1)
    return dict(joint_refines_marginal=not bad,exception_joint_groups=bad,
        split_marginal_groups=sorted(m for m,v in mj.items() if len(v)>1))


def summarize(points, boot, cfg):
    ds,ss,bs = (cfg[k] for k in ('training_draws','fit_seeds','budgets'))
    fields = ('draw','seed','budget','arm','forecast','weighting','partition')
    rows = {tuple(r[k] for k in fields):r for r in points}
    if len(rows)!=len(points) or set(rows)!=set(product(ds,ss,bs,ARMS,FORECASTS,WEIGHTS,PARTITIONS)): raise ValueError('roster')
    if len(bs)<2 or any(x>=y for x,y in zip(bs,bs[1:])): raise ValueError('budgets')
    mean = lambda v: math.fsum(v)/len(v)
    def report(v,b):
        return dict(mean=mean(list(v.values())),draw_means=[mean([v[d,s] for s in ss]) for d in ds],
            fit_seed_means=[mean([v[d,s] for d in ds]) for s in ss],low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)))
    estimates=[];contrasts=[];areas=[]
    for a,f,w,m in product(ARMS,FORECASTS,WEIGHTS,range(3)):
        ident=dict(arm=a,forecast=f,weighting=w,metric=METRICS[m]);curves={};boots={}
        for b in bs:
            values={p:{(d,s):rows[d,s,b,a,f,w,p]['values'][m] for d,s in product(ds,ss)} for p in PARTITIONS}
            for p in PARTITIONS: estimates.append(dict(ident,partition=p,budget=b,**report(values[p],boot[b,a,f,w,p][:,m])))
            curves[b]={k:values['joint'][k]-values['marginal'][k] for k in values['joint']}
            boots[b]=boot[b,a,f,w,'joint'][:,m]-boot[b,a,f,w,'marginal'][:,m]
            contrasts.append(dict(ident,contrast='joint-minus-marginal',budget=b,**report(curves[b],boots[b])))
        area={k:math.fsum((curves[x][k]+curves[y][k])*.5*math.log(y/x) for x,y in zip(bs,bs[1:]))/math.log(bs[-1]/bs[0]) for k in product(ds,ss)}
        ba=sum((boots[x]+boots[y])*.5*math.log(y/x) for x,y in zip(bs,bs[1:]))/math.log(bs[-1]/bs[0])
        areas.append(dict(ident,contrast='joint-minus-marginal',**report(area,ba)))
    return dict(estimates=estimates,contrasts=contrasts,normalized_log_budget_area=areas)


def controls():
    q=np.zeros((2,27));q[0,[0,13]]=.5;q[1,[1,12]]=.5
    j=resample(sufficient(q,'joint',q[None],[[.5,.5]]),[[1]])[0]
    m=resample(sufficient(marginals(q),'marginal',q[None],[[.5,.5]]),[[1]])[0]
    z=resample(sufficient(q[:1],'joint',q[:1][None],[[1.]]),[[1]])[0]
    return {'live:equal_marginal_distinct_joint':bool(m[0]>0 and abs(j[0])<1e-14),
        'positive:joint_refines_marginal':relation(membership(q,'joint')[0],membership(marginals(q),'marginal')[0])['joint_refines_marginal'],
        'placebo:identical_targets':bool(np.max(abs(z))<1e-14)}


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
    if design['tiers']!=['E2-full'] or design['bootstrap_seed']!=191023:raise ValueError('design')
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
    with np.load(original/'evaluator/NATIVE_JOINT.npz',allow_pickle=False) as z:
        native_error=max(close(truth,z['targets'],1e-12),close(mass,z['weights'],1e-12))
    schema=read(original/'ARRAY_SCHEMA.json')
    if schema['laws']!=ls or schema['frames']!=keys or schema['metrics']!=list(METRICS) or schema['partitions']!=list(PARTITIONS):raise ValueError('schema')
    old=json.loads(gzip.decompress((parent/'parent/goal_decision_points.json.gz').read_bytes()));fields=('tier','budget','arm','lineage','draw','seed')
    oi={tuple(r[k] for k in fields):r for r in old}
    expected=set(product(design['tiers'],design['budgets'],tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))),ls,design['training_draws'],design['fit_seeds']))
    if len(oi)!=len(old) or set(oi)!=expected:raise ValueError('parent roster')
    raw=json.loads(gzip.decompress((original/'joint_partition_points.json.gz').read_bytes()));rf=('draw','seed','budget','arm','forecast','weighting','partition')
    ri={tuple(r[k] for k in rf):r for r in raw}
    if len(ri)!=len(raw) or set(ri)!=set(product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS,FORECASTS,WEIGHTS,PARTITIONS)):raise ValueError('point roster')
    bindings=read(original/'GROUP_BINDINGS.json');remaining=set(bindings);relations=[]
    sample=np.random.default_rng(design['bootstrap_seed']).integers(len(ls),size=(design['bootstrap_resamples'],len(ls)))
    counts=np.array([np.bincount(v,minlength=len(ls)) for v in sample])
    if read(original/'BOOTSTRAP.json')!=dict(seed=design['bootstrap_seed'],resamples=design['bootstrap_resamples'],lineages=ls,count_digest=digest(counts.tolist())):raise ValueError('bootstrap')
    boot={};original_boot={};rows=[];cache={};file_owners={};parents=0
    errors=dict(native=native_error,marginal=0.,array=0.,parent=0.,point=0.)
    (root/'reconstructed_groups').mkdir();output_bindings={}
    for draw,seed,budget,arm,forecast in product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS,FORECASTS):
        pulse(phase='independent-joint-partition',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast)
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
            errors['parent']=max(errors['parent'],close(math.fsum(terms),oi['E2-full',budget,arm+'-'+forecast+'-coordinate',l,draw,seed]['loss']));parents+=1
        ids={p:membership(v,p)[0] for p,v in zip(PARTITIONS,(q,m))}
        relations.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,joint_groups=int(ids['joint'].max())+1,marginal_groups=int(ids['marginal'].max())+1,**relation(ids['joint'],ids['marginal'])))
        for weighting,w in zip(WEIGHTS,(mass,np.full_like(mass,1/len(keys)))):
            for kind,p in zip(PARTITIONS,(q,m)):
                binding=stem+'-'+weighting+'-'+kind;bound=bindings[binding];remaining.remove(binding)
                identity=(weighting,tuple(ids[kind].tolist()));filename=bound['file']
                if bound['partition_membership_sha256']!=digest(ids[kind].tolist()) or file_digest(original/filename)!=bound['sha256']:raise ValueError('group binding')
                if filename in file_owners and file_owners[filename]!=identity:raise ValueError('cache alias')
                file_owners[filename]=identity
                with np.load(original/filename,allow_pickle=False) as z:saved=dict(z)
                if not np.array_equal(saved['ids'],ids[kind]):raise ValueError('setting membership')
                if identity not in cache:
                    s=sufficient(p,kind,truth,w);v=resample(s,np.ones((1,len(ls)),int))[0];s['values']=v
                    out='reconstructed_groups/'+binding+'.npz';np.savez_compressed(root/out,**s)
                    cache[identity]=(s,resample(s,counts),resample(saved,counts),out)
                s,bv,obv,out=cache[identity];errors['array']=max(errors['array'],arrays(s,saved))
                row=dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,weighting=weighting,partition=kind,groups=len(s['mass'][0]),group_file=filename,values=s['values'].tolist())
                errors['point']=max(errors['point'],compare(row,ri[tuple(row[k] for k in rf)]));rows.append(row)
                output_bindings[binding]=dict(file=out,sha256=file_digest(root/out),original_file=filename)
                key=budget,arm,forecast,weighting,kind;n=len(design['training_draws'])*len(design['fit_seeds'])
                for dest,value in ((boot,bv),(original_boot,obv)):
                    if key not in dest:dest[key]=np.zeros_like(value)
                    dest[key]+=value/n
    if remaining:raise ValueError('extra bindings')
    compare(relations,read(original/'PARTITION_RELATIONS.json'))
    summary=read(original/'SUMMARY.json')
    for k,v in dict(settings=len(relations),group_bundles=len(rows),unique_group_arrays=len(cache),packets=len(keys),parent_cells=parents,refinement_exceptions=sum(not r['joint_refines_marginal'] for r in relations)).items():
        if summary[k]!=v:raise ValueError('coverage')
    regroup=summarize(rows,boot,design);old_regroup=summarize(raw,original_boot,design)
    expected={k:summary[k] for k in regroup};errors['regroup']=compare(regroup,expected);errors['original_regroup']=compare(old_regroup,expected)
    write(root/'INDEPENDENT_REGROUP.json',regroup);write(root/'ORIGINAL_ROW_REGROUP.json',old_regroup);write(root/'GROUP_BINDINGS.json',output_bindings)
    (root/'reconstructed_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    result=dict(passed=True,group_bundles=len(rows),unique_group_arrays=len(cache),settings=len(relations),packets=len(keys),parent_cells=parents,controls=checks,**{'max_'+k+'_error':v for k,v in errors.items()},scope='independent scalar moments;all exact memberships and cache bindings;full joint targets;expanded-square pair checks;original and reconstructed cross-moment law resampling;separate summary assembly;third regroup remains pending')
    write(root/'INDEPENDENT_REVIEW.json',result);return result
