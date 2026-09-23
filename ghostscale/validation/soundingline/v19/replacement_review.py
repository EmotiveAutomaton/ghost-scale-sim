"""Independent prior-preserving group replacement audit; no pooling or feedback producer imported.

Reuses the separately validated explicit path sum and native mechanics from the
support/transfer reviewers. Parent training and probability validity are inherited.
"""
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, digest
from . import support_review as V, transfer_review as T

MAPS = ('unpooled', 'pool-undo', 'pool-current-wrong')
ARMS = tuple(f'{m}__{u}' for m in MAPS for u in ('add','replace'))
AXES = ('mode', 'rule', 'order', 'feedback', 'budget', 'arm', 'change', 'subset', 'weighting')
METRICS = ('loss', 'squared_error', 'true_probability')


# These helpers are an already independently validated reviewer, not producer code.
from .feedback_review import inputs, reports, near, strata


def group_key(key, arm):
    if arm not in ARMS: raise ValueError('arm')
    arm=arm.split('__')[0]
    s,b,a,u,op=key
    if s==1 and op==3:
        if arm=='pool-undo': return s,b,a,-1,op
        if arm=='pool-current-wrong': return s,b,-1,u,op
    return tuple(key)


def revised(counts, observed, arm):
    if counts.shape!=(2,2,8,8,6,8) or not np.isfinite(counts).all() or np.any(counts<1):
        raise ValueError('unit prior counts')
    if arm not in ARMS: raise ValueError('arm')
    unique={}
    for key,target in observed:
        key=tuple(key)
        if key not in inputs('forward') or target not in range(8): raise ValueError('feedback')
        if key in unique and unique[key]!=target: raise ValueError('conflicting source')
        unique[key]=target
    result=counts.copy()
    # Reconstruct each destination/outcome by scalar summation over its full
    # equivalence class. No producer group accumulator or update code is called.
    observed_groups={group_key(k,arm) for k in unique}
    for key in inputs('forward'):
        replace=arm.endswith('__replace') and group_key(key,arm) in observed_groups
        members=[src for src in inputs('forward') if group_key(src,arm)==group_key(key,arm)]
        for target in range(8):
            result[key][target]=1+(0 if replace else math.fsum(float(counts[src][target]-1) for src in members))+sum(unique.get(src)==target for src in members)
    info=dict(unique_inputs=len(unique),observed_groups=len({group_key(k,arm) for k in unique}),
              total_tool_groups=len({group_key(k,arm) for k in inputs('forward')}),prior_per_outcome_per_group=1)
    return result,result/result.sum(-1,keepdims=True),info


def regroup(cells, lineages, draws, cfg):
    lookup={(r['lineage'],r['draw'],*(r[k] for k in AXES)):r for r in cells}
    if len(lookup)!=len(cells): raise ValueError('duplicate cell')
    keys=sorted({tuple(r[k] for k in AXES) for r in cells})
    estimates=[]
    for key in keys:
        comparisons=[('mean',None)]
        mapping,update=key[5].split('__')
        if mapping!='unpooled': comparisons.append(('pool-minus-unpooled',key[:5]+('unpooled__'+update,)+key[6:]))
        if update=='replace': comparisons.append(('replace-minus-add',key[:5]+(mapping+'__add',)+key[6:]))
        if key[4]!=0: comparisons.append(('feedback-minus-zero',key[:4]+(0,)+key[5:]))
        if key[3]=='wrong': comparisons.append(('wrong-minus-true',key[:3]+('true',)+key[4:]))
        if key[2]=='reverse': comparisons.append(('reverse-minus-forward',key[:2]+('forward',)+key[3:]))
        for label,base in comparisons:
            per_draw=[]
            for draw in draws:
                values=[]
                for lineage in lineages:
                    a=lookup[(lineage,draw,*key)]['loss'];b=lookup[(lineage,draw,*base)]['loss'] if base else 0.
                    values.append(None if a is None or b is None else a-b)
                per_draw.append(values)
            complete=[i for i in range(len(lineages)) if all(v[i] is not None for v in per_draw)]
            vals=[math.fsum(v[i] for v in per_draw)/len(draws) for i in complete]
            estimates.append(dict(zip(AXES,key),contrast=label,complete_lineages=[lineages[i] for i in complete],
                draw_means=[float(np.mean([v[i] for i in complete])) if complete else None for v in per_draw],
                **(V.interval(vals,cfg) if vals else dict(mean=None,low=None,high=None,lineage_values=[]))))
    return dict(estimates=estimates,population='two saved draws averaged within paired coefficient lineages; eight lineages weighted equally; native query mass and equal-query estimands separate; intervals condition on retained fits and deterministic feedback roster',
        bootstrap_resamples=cfg['bootstrap_resamples'],bootstrap_seed=cfg['bootstrap_seed'],practical_margin_nats=.02,
        limitation='equal-query estimates replicate the same fixed predictions across lineages; a degenerate lineage interval does not quantify training or feedback-population uncertainty')


def review(original,output,cfg,pulse=lambda **kw:None):
    design=read(original/'PLAN.json')['design'];summary=read(original/'SUMMARY.json');base=original/'inputs'
    if design['arms']!=list(ARMS) or design['epsilon']!=1/32 or design['saved_prior']!=1: raise ValueError('design')
    if design['budgets']!=[0,16] or design['orders']!=['forward','reverse'] or design['feedback']!=['true','wrong']: raise ValueError('roster design')
    truth=read(base/'QUERY_TRUTH.json');qs=[tuple(r['query']) for r in truth]
    if len(qs)!=design['queries'] or len(set(qs))!=len(qs): raise ValueError('queries')
    targets={rule:np.array([T.endpoint(q,rule) for q in qs]) for rule in T.RULES}
    if truth!=[dict(query=list(q),targets={rule:int(targets[rule][i]) for rule in T.RULES}) for i,q in enumerate(qs)]: raise ValueError('truth')
    public=[dict(skill=q[0],belief_error=q[1],initial=list(V.ARTS[q[2]]),operations=[V.OPS[o] for o in q[3:]]) for q in qs]
    if public!=read(original/'reader/QUERIES.json'): raise ValueError('query projection')
    expected_roster={o:[list(k) for k in inputs(o)] for o in design['orders']}
    if read(original/'evaluator/FEEDBACK_ROSTER.json')!=expected_roster: raise ValueError('feedback roster')
    if 'feedback_rosters' in design and design['feedback_rosters']!=expected_roster: raise ValueError('frozen feedback roster')
    populations={};population_rows=[];index={q:i for i,q in enumerate(qs)};error=0.;native_paths=0
    for lineage,rule in product(design['lineages'],T.RULES):
        mass=np.zeros(len(qs));records=V.zipped(base/'raw'/f'{lineage}-{rule}_points.json.gz');native_paths+=len(records)
        for r in records:
            i=index[V.query(r)]
            if V.code(r['final'])!=targets[rule][i] or not math.isfinite(r['probability']) or r['probability']<0: raise ValueError('native target/mass')
            mass[i]+=r['probability']
        error=max(error,near([mass.sum()],[1.]));populations[lineage,rule]=mass
        population_rows.append(dict(lineage=lineage,rule=rule,masses=mass.tolist()))
    saved_pop=read(original/'evaluator/POPULATIONS.json')
    if len(saved_pop)!=len(population_rows): raise ValueError('population count')
    for a,b in zip(saved_pop,population_rows,strict=True):
        if (a['lineage'],a['rule'])!=(b['lineage'],b['rule']): raise ValueError('population identity')
        error=max(error,near(a['masses'],b['masses']))
    cells=[];mapping=[];group_rows=[];reader_ids={'QUERIES'};forecast_names=set();model_names=set();vectors=0;baseline_vectors=0
    changed=targets[T.RULES[0]]!=targets[T.RULES[1]]
    for draw,mode in product(design['training_draws'],('original','composition-holdout')):
        pulse(phase='independent-pooled-replacement',draw=draw,mode=mode)
        with np.load(base/'models'/f'{draw}-{mode}.npz',allow_pickle=False) as z:
            counts=z['transition_counts'];direct_keys={tuple(k) for k in z['direct_keys']} if 'direct_keys' in z else None
        diagnostics=read(base/'forecasts'/f'{draw}-{mode}-support.json')
        if len(diagnostics)!=len(qs): raise ValueError('support size')
        if direct_keys is not None:
            for q,d in zip(qs,diagnostics,strict=True):
                a=u=q[2];visits=[]
                for op in q[3:]:
                    visits.append(bool(counts[q[0],q[1],a,u,op].sum()>8))
                    after=V.code(V.execute(V.ARTS[a],V.ARTS[u],V.OPS[op],q[0],q[1],'original'));u,a=a,after
                if d!=dict(query_seen=q in direct_keys,all_primitives_seen=all(visits),visited_primitives=sum(visits),withheld_composition=V.held(q)): raise ValueError('support diagnostic')
        baseline=np.array([31/32*V.propagate(V.normalize(counts),q)+1/256 for q in qs])
        with np.load(base/'forecasts'/f'{draw}-{mode}.npz',allow_pickle=False) as z:
            error=max(error,near(z['queries'],qs,0),near(z['learned-exact'],baseline));baseline_vectors+=len(qs)
        for rule,order,feedback,budget in product(T.RULES,design['orders'],design['feedback'],design['budgets']):
            observed=reports(order,budget,rule,feedback=='wrong')
            visible=[dict(skill=k[0],belief_error=k[1],current=list(V.ARTS[k[2]]),undo_buffer=list(V.ARTS[k[3]]),operation='accept-tool',next_artifact=list(V.ARTS[t])) for k,t in observed]
            content_id=digest(visible);reader_ids.add(content_id)
            if read(original/'reader'/f'{content_id}.json')!=visible: raise ValueError('feedback projection')
            mapping.append(dict(draw=draw,mode=mode,rule=rule,order=order,feedback=feedback,budget=budget,reader_id=content_id))
            for arm in ARMS:
                revised_counts,table,info=revised(counts,observed,arm)
                key=f'{draw}-{mode}-{rule}-{order}-{feedback}-{budget}-{arm}';model_names.add(key+'.npz');forecast_names.add(key+'_points.npz')
                with np.load(original/'models'/f'{key}.npz',allow_pickle=False) as z:
                    if set(z.files)!={'counts','table'}: raise ValueError('table schema')
                    error=max(error,near(z['counts'],revised_counts),near(z['table'],table))
                predicted=np.array([31/32*V.propagate(table,q)+1/256 for q in qs]);vectors+=len(qs)
                prob=predicted[np.arange(len(qs)),targets[rule]]
                values=dict(loss=-np.log(prob),squared_error=np.sum((predicted-np.eye(8)[targets[rule]])**2,axis=1),true_probability=prob)
                with np.load(original/'forecasts'/f'{key}_points.npz',allow_pickle=False) as z:
                    if set(z.files)!={'queries','predictions','path_sum_max_abs',*METRICS}: raise ValueError('forecast schema')
                    error=max(error,near(z['queries'],qs,0),near(z['predictions'],predicted))
                    if z['path_sum_max_abs'].shape!=() or not 0<=float(z['path_sum_max_abs'])<=1e-12: raise ValueError('path error')
                    for metric in METRICS:error=max(error,near(z[metric],values[metric]))
                if budget==0 and arm=='unpooled__add': error=max(error,near(predicted,baseline))
                identity=dict(draw=draw,mode=mode,rule=rule,order=order,feedback=feedback,budget=budget,arm=arm)
                group_rows.append(dict(identity,**info))
                for lineage in design['lineages']: cells.extend(strata(values,populations[lineage,rule],diagnostics,changed,dict(identity,lineage=lineage)))
    expected_maps={arm:[dict(input=list(k),group=list(group_key(k,arm))) for k in inputs('forward')] for arm in ARMS}
    if read(original/'evaluator/POOL_MAPS.json')!=expected_maps: raise ValueError('pool map')
    if read(original/'evaluator/GROUP_COUNTS.json')!=group_rows: raise ValueError('group denominator')
    if mapping!=read(original/'evaluator/FEEDBACK_MAP.json'): raise ValueError('feedback mapping')
    if {p.stem for p in (original/'reader').glob('*.json')}!=reader_ids: raise ValueError('extra reader evidence')
    if {p.name for p in (original/'models').glob('*.npz')}!=model_names or {p.name for p in (original/'forecasts').glob('*.npz')}!=forecast_names: raise ValueError('output coverage')
    if len(cells)!=len(summary['cells']): raise ValueError('cell count')
    for a,b in zip(summary['cells'],cells,strict=True):
        for k in ('draw','lineage',*AXES,'queries'):
            if a[k]!=b[k]: raise ValueError('cell identity/denominator')
        for k in ('population_mass',*METRICS):
            if b[k] is None:
                if a[k] is not None: raise ValueError('empty stratum')
            else:error=max(error,near([a[k]],[b[k]]))
    if summary['score_rows']!=vectors*len(design['lineages']) or summary['fits']!=0 or summary['queries']!=len(qs): raise ValueError('totals')
    if read(original/'PARENT_REPRODUCTION.json')!=dict(passed=True,forecast_vectors=baseline_vectors,saved_unit_prior=True): raise ValueError('parent receipt')
    timing=read(original/'TIMING.jsonl')['measurements']
    if {(r['draw'],r['mode']) for r in timing}!=set(product(design['training_draws'],('original','composition-holdout'))) or len(timing)!=2*len(design['training_draws']) or any(r['cpu_seconds']<0 or r['new_fits']!=0 for r in timing): raise ValueError('timing')
    write(output/'INDEPENDENT_REGROUP.json',regroup(cells,design['lineages'],design['training_draws'],cfg))
    result=dict(passed=True,cells=len(cells),forecast_vectors=vectors,score_rows=summary['score_rows'],native_paths=native_paths,baseline_vectors=baseline_vectors,max_error=error,reader_packets=len(reader_ids),
        scope='independent supplied group maps,scalar replacement/addition with one retained prior,feedback,updated tables,path-sum forecasts,proper scores,population/support strata and paired conditional intervals; parent training/path probabilities inherit bound independent validation')
    write(output/'NUMERICAL_REVIEW.json',result)
    return result


def run(root,plan,pulse):
    cfg=plan['design']
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h: raise ValueError('input binding')
    original=root/'inputs/original'
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']: raise ValueError('parent binding')
    result=review(original,root,cfg,pulse)
    return dict(result,controls={'live:all_forecasts_and_scores_rebuilt':True,'positive:parent_identity':True,'placebo:empty_and_zero_feedback_strata':True,'positive:group_denominators_and_prior':True})
