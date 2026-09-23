"""Independent public support, sparse native scoring and paired regrouping.

The producer, its executors and scoring functions are not imported. Previously
verified fitted forecasts and native posterior masses are inherited inputs.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest

OPS = ('edit-claim', 'repair-evidence', 'replace-presentation', 'accept-tool', 'inspect', 'undo')
TIERS = ('E0', 'E1', 'E2-sparse', 'E2-full')
ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
PAIRS = [(a+'-restricted', a) for a in ARMS] + [
    ('learned-bank-restricted', a) for a in
    ('raw-history-restricted', 'frozen-latent-restricted', 'matched-frequency-restricted', 'uniform-support')]
N = 5832
DIGITS = np.array([[k//216//9, k//216//3%3, k//216%3,
                    k%216//36, k%216//6%6, k%6] for k in range(N)])


def close(a, b, tolerance=1e-8):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('shape or nonfinite value')
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    if error > tolerance: raise ValueError(f'independent support mismatch: {error}')
    return error


def arrays(path):
    with np.load(path, allow_pickle=False) as data: return {k:data[k] for k in data.files}


def transition(state, undo, operation, skill, belief):
    x, y, z = state
    if operation == 0: return (1-x, y, z)
    if operation == 1: return (x, (x+belief)%2 if skill else 1-y, z)
    if operation == 2: return (x, y, 1-z)
    if operation == 3:
        if not skill: raise ValueError('illegal tool')
        return (1-x, 1-x, z)
    if operation == 4: return state
    if operation == 5: return undo
    raise ValueError('operation')


def validate(p):
    if set(p) != {'schema', 'tier', 'inputs'} or p['schema'] != 'v19.local.public.1' or p['tier'] not in TIERS:
        raise ValueError('public packet schema')
    a = p['inputs']; expected = {'artifact'}
    if p['tier'] != 'E0': expected |= {'initial', 'requested_purpose'}
    if p['tier'].startswith('E2'): expected.add('observations')
    if set(a) != expected: raise ValueError('private or missing input')
    for k in ('artifact', 'initial'):
        if k in a and (len(a[k]) != 3 or any(x not in (0, 1) for x in a[k])): raise ValueError('artifact')
    if 'requested_purpose' in a and a['requested_purpose'] not in (0, 1): raise ValueError('request')
    events = a.get('observations', [])
    if len(events) != {'E0':0, 'E1':0, 'E2-sparse':1, 'E2-full':3}[p['tier']]: raise ValueError('witness length')
    for i, e in enumerate(events):
        if set(e) != {'step','operation','before','after','tool_proposal'} or e['step'] != i or e['operation'] not in OPS:
            raise ValueError('witness schema')
        for k in ('before', 'after'):
            if len(e[k]) != 3 or any(x not in (0,1) for x in e[k]): raise ValueError('witness artifact')
        if e['tool_proposal'] != (e['after'] if e['operation'] == 'accept-tool' else None): raise ValueError('proposal')


def enumerate_support(pulse=lambda **kw: None):
    index = defaultdict(set); paths = steps = 0
    for initial in product((0,1), repeat=3):
        pulse(phase='independent-public-mechanics', initial=initial)
        for skill, belief in product((0,1), repeat=2):
            for operations in product(range(6), repeat=3):
                if not skill and 3 in operations: continue
                artifact = undo = initial; events = []
                for i, op in enumerate(operations):
                    following = transition(artifact, undo, op, skill, belief)
                    events.append(dict(step=i, operation=OPS[op], before=list(artifact), after=list(following),
                                       tool_proposal=list(following) if op==3 else None))
                    undo, artifact = artifact, following; steps += 1
                op_code = operations[0]*36+operations[1]*6+operations[2]
                for tier in TIERS:
                    for purpose in ((0,) if tier == 'E0' else (0,1)):
                        a = dict(artifact=list(artifact))
                        if tier != 'E0': a.update(initial=list(initial), requested_purpose=purpose)
                        if tier.startswith('E2'): a['observations'] = events[:1] if tier == 'E2-sparse' else events
                        index[digest(dict(schema='v19.local.public.1', tier=tier, inputs=a))].add(op_code)
                paths += 1
    return index, dict(executions=paths, primitive_transitions=steps, initial_artifacts=8,
                      mechanics_assignments=4, goal_sequences_per_op_path=27)


def mask_for(packet, index):
    validate(packet)
    operations = index.get(digest(packet), set())
    return np.array([k % 216 in operations for k in range(N)], dtype=bool)


def expand(p, alphabet, budget):
    a = np.asarray(alphabet); p = np.asarray(p)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or len(set(a)) != len(a) or len(a)>=N or np.any(a<0) or np.any(a>=N):
        raise ValueError('alphabet')
    if p.ndim != 2 or p.shape[1] != len(a) or not np.isfinite(p).all() or np.any(p<0): raise ValueError('forecast')
    close(p.sum(1), np.full(len(p), budget/(budget+1)), 1e-12)
    q = np.ones((len(p),N)) / ((budget+1)*(N-len(a)))
    q[:,a] = p
    return q


def condition(p, mask):
    if p.shape != mask.shape or not np.isfinite(p).all() or np.any(p<0): raise ValueError('restriction shape')
    close(p.sum(1), np.ones(len(p)), 1e-12)
    q = np.array(p, copy=True); z = np.zeros(len(p)); valid = np.zeros(len(p), bool)
    for i in range(len(p)):
        z[i] = np.sum(p[i,mask[i]])
        if mask[i].any() and z[i]>0:
            q[i,:] = 0; q[i,mask[i]] = p[i,mask[i]] / z[i]; valid[i] = True
    return q, z, valid


def references(data, tier, keys, lineages, masks):
    selected = [r for r in data if r['tier']==tier]
    if len(selected)!=len(lineages) or {r['lineage'] for r in selected}!=set(lineages): raise ValueError('lineage roster')
    bylin={r['lineage']:{f['frame']:f for f in r['frames']} for r in selected}
    if any(set(d)!=set(keys) for d in bylin.values()): raise ValueError('frame roster')
    result=[]
    for i,key in enumerate(keys):
        labels=sorted(k for k,v in bylin[lineages[0]][key]['target'])
        if len(set(labels))!=len(labels) or not masks[i,labels].all(): raise ValueError('true process excluded')
        target=[]; masses=[]
        for lineage in lineages:
            f=bylin[lineage][key]; t=dict(f['target'])
            if sorted(t)!=labels or len(t)!=len(f['target']) or min(t.values())<=0 or f['mass']<=0: raise ValueError('target roster')
            close(sum(t.values()),1.,1e-10)
            target.append([t[k] for k in labels]);masses.append(f['mass'])
        result.append((np.array(labels),np.array(target),np.array(masses)))
    close(np.sum([r[2] for r in result],axis=0),np.ones(len(lineages)),1e-10)
    return result


def score(p, alphabet, refs):
    """Frame-local sparse truth sums, then weighted across each native lineage."""
    a=list(map(int,alphabet));known=set(a);unseen=[k for k in range(N) if k not in known]
    priority=np.array(a+unseen); result=defaultdict(lambda:np.zeros(len(refs[0][2])))
    for i,(truth,target,mass) in enumerate(refs):
        q=p[i];order=priority[np.argsort(-q[priority],kind='stable')]
        cumulative=np.cumsum(q[order]);size=int(np.searchsorted(cumulative,.9,side='left'))+1
        if size>N: raise ValueError('candidate normalization')
        mode=int(order[0]);tq=q[truth]
        if np.any(tq<=0): raise ValueError('zero probability true process')
        outside=np.array([k not in known for k in truth]);equal=DIGITS[truth]==DIGITS[mode]
        accuracy=target@equal
        values=dict(loss=-(target@np.log(tq)), squared_error=np.sum(q*q)-2*(target@tq)+np.sum(target*target,axis=1),
            compatible_mass=np.sum(q[truth]),candidate_coverage=target@np.isin(truth,order[:size]),candidate_size=size,
            top_incompatible=mode not in truth,abstain=q[mode]<.5,top_probability=q[mode],unknown_mass=np.sum(q[unseen]),
            truth_outside_alphabet=target@outside,goal_accuracy=accuracy[:,:3].mean(1),operation_accuracy=accuracy[:,3:].mean(1))
        values.update({f'goal_{j}':accuracy[:,j] for j in range(3)})
        values.update({f'operation_{j}':accuracy[:,j+3] for j in range(3)})
        for k,v in values.items():result[k]+=mass*v
    return dict(result)


def regroup(rows,cfg):
    fields=('tier','budget','arm','lineage','draw','seed')
    idx={tuple(r[k] for k in fields):r['loss'] for r in rows}
    if len(idx)!=len(rows):raise ValueError('duplicate stratum')
    ls,ds,ss,bs=(cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    draws=np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls),size=(cfg['bootstrap_resamples'],len(ls)))
    def receipt(x):
        v=x.mean((1,2));boot=v[draws].mean(1)
        return dict(mean=float(v.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),lineage_values=v.tolist(),
                    draw_means=x.mean((0,2)).tolist(),feature_seed_means=x.mean((0,1)).tolist())
    contrasts=[];areas=[];logb=np.log(bs);span=logb[-1]-logb[0]
    for tier in cfg['tiers']:
        for arm,base in PAIRS:
            curve=[]
            for budget in bs:
                x=np.array([[[idx[tier,budget,arm,l,d,s]-idx[tier,budget,base,l,d,s] for s in ss] for d in ds] for l in ls])
                contrasts.append(dict(tier=tier,budget=budget,arm=arm,baseline=base,**receipt(x)));curve.append(x)
            area=sum((curve[j]+curve[j+1])*(logb[j+1]-logb[j])/2 for j in range(len(bs)-1))/span
            areas.append(dict(tier=tier,arm=arm,baseline=base,**receipt(area)))
    return contrasts,areas


def controls():
    p=np.array([[.2,.3,.5],[1.,0.,0.]]);mask=np.array([[1,0,1],[0,1,0]],bool)
    q,z,ok=condition(p,mask)
    return {'live:conditional_probability':bool(np.allclose(q[0],[2/7,0,5/7])),
            'placebo:full_support_identity':bool(np.allclose(condition(p,np.ones_like(mask))[0],p)),
            'positive:zero_mass_fallback':bool(not ok[1] and np.array_equal(q[1],p[1])),
            'positive:undo':transition((0,0,0),(1,1,1),5,1,0)==(1,1,1)}


def run(root,plan,pulse):
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('known-answer controls')
    cfg=plan['design'];base=root/'inputs'
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input hash')
    original=base/'original';parent=base/'parent';target=read(original/'PLAN.json');design=target['design']
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target plan')
    packets=read(original/'reader/PACKETS.json')['packets']
    if packets!=read(parent/'reader/PACKETS.json')['packets']:raise ValueError('reader changed')
    for key,p in packets.items():
        validate(p)
        if key!=digest(p):raise ValueError('packet key')
    reference=read(parent/'evaluator/REFERENCES.json');summary=read(original/'SUMMARY.json')
    raw=json.loads(gzip.decompress((original/'raw/support_points.json.gz').read_bytes()))
    fields=('draw','tier','seed','budget','arm','lineage')
    lookup={tuple(r[k] for k in fields):r for r in raw}
    old={tuple(r[k] for k in fields):r for r in read(parent/'SUMMARY.json')['cells']}
    if len(lookup)!=len(raw):raise ValueError('duplicate raw rows')
    index,enum=enumerate_support(pulse)
    if enum!=summary['enumeration']:raise ValueError('enumeration counts')
    rows=[];stats=[];score_error=normalizer_error=original_error=0.;reproduced=0
    for tier in design['tiers']:
        keys=sorted(k for k,p in packets.items() if p['tier']==tier)
        if read(original/'evaluator'/f'{tier}-frames.json')!=keys:raise ValueError('support frame roster')
        masks=np.array([mask_for(packets[k],index) for k in keys])
        saved_masks=arrays(original/'evaluator'/f'{tier}-support.npz')
        if set(saved_masks)!= {'masks'} or not np.array_equal(masks,saved_masks['masks']):raise ValueError('public mask')
        if not masks.any(1).all():raise ValueError('empty support')
        stats.append(dict(tier=tier,packets=len(keys),minimum_labels=int(masks.sum(1).min()),maximum_labels=int(masks.sum(1).max())))
        refs=references(reference,tier,keys,design['development_lineages'],masks)
        for draw,seed,budget in product(design['training_draws'],design['fit_seeds'],design['budgets']):
            saved_z=arrays(original/'raw'/f'{draw}-{tier}-{seed}-{budget}-normalizers.npz')
            if set(saved_z)!=set(ARMS):raise ValueError('normalizer arms')
            metrics={}
            for arm in ARMS:
                pulse(phase='independent-support-scores',tier=tier,draw=draw,seed=seed,budget=budget,arm=arm)
                stem=f'{draw}-{tier}-{seed}-{budget}-{arm}'
                if read(parent/'forecasts'/(stem+'-frames.json'))!=keys:raise ValueError('forecast frames')
                saved=arrays(parent/'forecasts'/(stem+'.npz'));alphabet=saved['alphabet']
                full=expand(saved['probabilities'],alphabet,budget)
                restricted,z,valid=condition(full,masks)
                normalizer_error=max(normalizer_error,close(z,saved_z[arm],1e-12))
                if not valid.all():raise ValueError('restriction failure')
                # Saved normalizers define the actual scored floating-point forecast.
                # Independently summed normalizers above must agree first.
                restricted=full*masks/saved_z[arm][:,None]
                metrics[arm]=score(full,alphabet,refs)
                metrics[arm+'-restricted']=score(restricted,alphabet,refs)
                if arm=='matched-frequency':metrics['uniform-support']=score(masks/masks.sum(1)[:,None],alphabet,refs)
            for arm,metrics_by_name in metrics.items():
                for j,lineage in enumerate(design['development_lineages']):
                    row=dict(draw=draw,tier=tier,seed=seed,budget=budget,arm=arm,lineage=lineage,frames=len(keys),
                             **{k:float(v[j]) for k,v in metrics_by_name.items()})
                    key=tuple(row[k] for k in fields);expected=lookup[key]
                    if set(row)!=set(expected):raise ValueError('metric coverage')
                    score_error=max(score_error,close([row[k] for k in metrics_by_name],[expected[k] for k in metrics_by_name]))
                    if arm in ARMS:
                        original_error=max(original_error,close([row[k] for k in metrics_by_name],[old[key][k] for k in metrics_by_name]));reproduced+=1
                    rows.append(row)
    if len(rows)!=len(raw) or len(rows)!=summary['rows'] or stats!=summary['support'] or reproduced!=len(old):raise ValueError('whole roster')
    contrasts,areas=regroup(rows,design);regroup_error=0.
    for rebuilt,saved in ((contrasts,summary['contrasts']),(areas,summary['normalized_log_budget_area'])):
        if len(rebuilt)!=len(saved):raise ValueError('contrast roster')
        for a,b in zip(rebuilt,saved):
            if set(a)!=set(b):raise ValueError('contrast fields')
            for k in a:
                if k in ('tier','budget','arm','baseline'):
                    if a[k]!=b[k]:raise ValueError('contrast identity')
                else:regroup_error=max(regroup_error,close(a[k],b[k]))
    write(root/'INDEPENDENT_REGROUP.json',dict(contrasts=contrasts,normalized_log_budget_area=areas))
    (root/'review_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    checks['positive:complete_reconstruction']=True
    return dict(controls=checks,rows=len(rows),original_cells_reproduced=reproduced,packets=len(packets),enumeration=enum,
        contrasts=len(contrasts),learning_areas=len(areas),max_score_error=score_error,max_normalizer_error=normalizer_error,
        max_original_error=original_error,max_regroup_error=regroup_error,target_plan_sha256=cfg['target_plan_sha256'],
        scope='independent legal mechanics and native weighted scores; accepted parent fits and native posteriors inherited; complete scientific adjudication pending')
