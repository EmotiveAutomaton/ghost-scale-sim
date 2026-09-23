"""Prior-preserving replacement of observed equivalence groups; no fitting."""
from itertools import product
import time
import numpy as np
from ..v18_3.io import digest, read, write, file_digest
from . import primitive_feedback as F, primitive_pooling as P

UPDATES = ('add', 'replace')
ARMS = tuple(f'{mapping}__{update}' for mapping in P.ARMS for update in UPDATES)
BUDGETS = (0, 16)


def group(key, arm):
    if arm not in ARMS: raise ValueError('arm')
    return P.group(key, arm.split('__')[0])


def updated_counts(counts, observed, arm):
    if arm not in ARMS: raise ValueError('arm')
    mapping, update = arm.split('__')
    # The established additive routine validates legal inputs, priors and duplicates.
    result, info = P.pooled_counts(counts, observed, mapping)
    if update == 'replace':
        unique = {tuple(k): t for k, t in observed}
        fresh = {}
        for key, target in unique.items():
            g = P.group(key, mapping)
            if g not in fresh: fresh[g] = np.ones(8)
            fresh[g][target] += 1
        for key in F.roster('forward'):
            if P.group(key, mapping) in fresh:
                result[key] = fresh[P.group(key, mapping)]
    return result, info


def controls():
    counts = np.ones((2,2,8,8,6,8)); key = (1,0,2,0,3); target = F.observation(key,'original')
    counts[key][(target+1)%8] += 20
    obs = [(key,target),((1,0,2,1,3),target)]
    baseline,_ = updated_counts(counts,[],'pool-undo__replace')
    replace,info = updated_counts(counts,obs,'pool-undo__replace')
    add,_ = updated_counts(counts,obs,'pool-undo__add')
    repeat,_ = updated_counts(counts,obs*2,'pool-undo__replace')
    uniform,_ = updated_counts(np.ones_like(counts),[],'pool-undo__replace')
    untouched = np.ones(counts.shape[:-1],bool); untouched[1,0,2,:,3] = False
    q = (1,0,2,3,5,4); table = F.R.normalized(replace)
    return dict(P.controls(), **{
        'live:replacement_discards_stale_group_counts': bool(replace[key].sum()==10 and add[key].sum()==30),
        'positive:single_prior_and_unique_inputs': bool(replace[key][target]==3 and info['unique_inputs']==2),
        'positive:unobserved_group_identity': np.array_equal(replace[untouched],baseline[untouched]),
        'positive:replacement_deduplicates': np.array_equal(replace,repeat),
        'placebo:uniform_zero_feedback': np.array_equal(uniform,np.ones_like(counts)),
        'positive:replacement_path_sum': np.allclose(F.path_sum(table,q),F.R.propagate(table,q),atol=1e-12,rtol=0)})


def run(root, plan, pulse):
    cfg=plan['design']; inputs=root/'inputs'; checks=controls()
    if not all(checks.values()) or cfg['arms'] != list(ARMS) or cfg['budgets'] != list(BUDGETS):
        raise ValueError('pooling admission')
    if cfg['orders'] != list(F.ORDERS) or cfg['feedback'] != list(F.FEEDBACK) or cfg['epsilon'] != 1/32 or cfg['saved_prior'] != 1:
        raise ValueError('frozen comparison')
    for n,h in cfg['input_files'].items():
        if file_digest(inputs/n) != h: raise ValueError('input binding')
    for folder in ('models','forecasts','reader','evaluator'): (root/folder).mkdir()
    write(root/'CONTROLS.json', checks)
    truth=read(inputs/'QUERY_TRUTH.json'); qs=[tuple(r['query']) for r in truth]
    if len(qs) != cfg['queries'] or len(set(qs)) != len(qs): raise ValueError('query roster')
    write(root/'reader/QUERIES.json',[F.R.visible(q) for q in qs])
    targets={r:np.array([t['targets'][r] for t in truth]) for r in F.M.RULES}
    changed=targets['original'] != targets['presentation-tool']; populations={}; index={q:i for i,q in enumerate(qs)}
    for lineage,rule in product(cfg['lineages'],F.M.RULES):
        mass=np.zeros(len(qs))
        for row in F.read_gzip(inputs/'raw'/f'{lineage}-{rule}_points.json.gz'):
            i=index[F.R.query(row)]
            if F.R.code(row['final']) != targets[rule][i] or row['probability'] < 0: raise ValueError('native target/mass')
            mass[i] += row['probability']
        if not np.isclose(mass.sum(),1,atol=1e-12,rtol=0): raise ValueError('population mass')
        populations[lineage,rule]=mass
    write(root/'evaluator/POPULATIONS.json',[dict(lineage=l,rule=r,masses=m.tolist()) for (l,r),m in populations.items()])
    write(root/'evaluator/POOL_MAPS.json',{arm:[dict(input=list(k),group=list(group(k,arm))) for k in F.roster('forward')] for arm in ARMS})
    write(root/'evaluator/FEEDBACK_ROSTER.json',{o:[list(k) for k in F.roster(o)] for o in F.ORDERS})
    cells=[]; mappings=[]; timings=[]; rows=0; baseline_vectors=0; counts_log=[]
    for draw,mode in product(cfg['training_draws'],('original','composition-holdout')):
        start=time.process_time(); pulse(phase='pooled-replacement-parent',draw=draw,mode=mode)
        with np.load(inputs/'models'/f'{draw}-{mode}.npz',allow_pickle=False) as z: counts=z['transition_counts']
        with np.load(inputs/'forecasts'/f'{draw}-{mode}.npz',allow_pickle=False) as z:
            baseline=z['learned-exact']; saved_qs=z['queries']
        check=np.array([F.S.smooth(F.R.propagate(F.R.normalized(counts),q)) for q in qs])
        if not np.array_equal(saved_qs,qs) or not np.allclose(check,baseline,atol=1e-12,rtol=0): raise ValueError('parent reproduction')
        baseline_vectors += len(qs); diagnostics=read(inputs/'forecasts'/f'{draw}-{mode}-support.json')
        for rule,order,feedback,budget in product(F.M.RULES,F.ORDERS,F.FEEDBACK,BUDGETS):
            observed=[(k,F.observation(k,rule,feedback=='wrong')) for k in F.roster(order)[:budget]]
            public=[dict(skill=k[0],belief_error=k[1],current=list(F.R.ARTIFACTS[k[2]]),undo_buffer=list(F.R.ARTIFACTS[k[3]]),operation='accept-tool',next_artifact=list(F.R.ARTIFACTS[t])) for k,t in observed]
            public_id=digest(public); p=root/'reader'/f'{public_id}.json'
            if not p.exists(): write(p,public)
            mappings.append(dict(draw=draw,mode=mode,rule=rule,order=order,feedback=feedback,budget=budget,reader_id=public_id))
            for arm in ARMS:
                identity=dict(draw=draw,mode=mode,rule=rule,order=order,feedback=feedback,budget=budget,arm=arm)
                pulse(phase='pooled-replacement-query',**identity)
                revised,info=updated_counts(counts,observed,arm); table=F.R.normalized(revised)
                pred=np.array([F.S.smooth(F.R.propagate(table,q)) for q in qs])
                independent=np.array([F.S.smooth(F.path_sum(table,q)) for q in qs]); error=float(abs(pred-independent).max())
                if error > 1e-12: raise ValueError('path sum')
                if arm=='unpooled__add':
                    old=np.array([F.S.smooth(F.R.propagate(F.update(counts,observed,'add-one'),q)) for q in qs])
                    if not np.array_equal(pred,old): raise ValueError('unpooled update identity')
                prob=pred[np.arange(len(qs)),targets[rule]]; loss=-np.log(prob); squared=((pred-np.eye(8)[targets[rule]])**2).sum(1)
                key=f'{draw}-{mode}-{rule}-{order}-{feedback}-{budget}-{arm}'
                np.savez_compressed(root/'models'/f'{key}.npz',counts=revised,table=table)
                np.savez_compressed(root/'forecasts'/f'{key}_points.npz',queries=saved_qs,predictions=pred,loss=loss,squared_error=squared,true_probability=prob,path_sum_max_abs=error)
                counts_log.append(dict(identity,**info))
                for lineage in cfg['lineages']: cells.extend(F.cells_for(loss,squared,prob,populations[lineage,rule],diagnostics,changed,dict(identity,lineage=lineage)))
                rows += len(qs)*len(cfg['lineages'])
        timings.append(dict(draw=draw,mode=mode,cpu_seconds=time.process_time()-start,new_fits=0))
    write(root/'evaluator/FEEDBACK_MAP.json',mappings); write(root/'evaluator/GROUP_COUNTS.json',counts_log)
    write(root/'PARENT_REPRODUCTION.json',dict(passed=True,forecast_vectors=baseline_vectors,saved_unit_prior=True))
    write(root/'TIMING.jsonl',dict(measurements=timings,accounting='group addition/replacement,propagation,path sums and scores; no refit'))
    return dict(controls=checks,cells=cells,queries=len(qs),score_rows=rows,fits=0,scope='supplied semantic input equivalence; replacement versus addition of fixed feedback; no learned law discovery,process correspondence or human intent')
