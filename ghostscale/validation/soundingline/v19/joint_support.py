"""Public-mechanics support restriction of frozen joint-process readouts.

No policy, fitted parameter, evaluator label or native population mass enters a
support mask. Every syntactic goal sequence is retained for each legal op path.
"""
from collections import defaultdict
from itertools import product
import gzip
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import local_world as L, joint_review as V

UNIVERSE = 5832
TIERS = ('E0', 'E1', 'E2-sparse', 'E2-full')
ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
PAIRS = tuple((a+'-restricted', a) for a in ARMS) + (
    ('learned-bank-restricted', 'raw-history-restricted'),
    ('learned-bank-restricted', 'frozen-latent-restricted'),
    ('learned-bank-restricted', 'matched-frequency-restricted'),
    ('learned-bank-restricted', 'uniform-support'))
LABELS = np.arange(UNIVERSE)
DIGITS = np.array([V.digits(k)[0]+V.digits(k)[1] for k in LABELS])


def validate(packet):
    L.validate_public(packet)
    if packet['tier'] not in TIERS: raise ValueError('tier')
    a = packet['inputs']
    for k in ('artifact', 'initial'):
        if k in a and (len(a[k]) != 3 or any(v not in (0, 1) for v in a[k])):
            raise ValueError('artifact bits')
    if 'requested_purpose' in a and a['requested_purpose'] not in (0, 1): raise ValueError('request')
    events = a.get('observations', [])
    if [e['step'] for e in events] != list(range({'E0':0,'E1':0,'E2-sparse':1,'E2-full':3}[packet['tier']])):
        raise ValueError('witness schedule')
    for e in events:
        if e['operation'] not in L.OPERATIONS: raise ValueError('operation')
        for k in ('before','after'):
            if len(e[k]) != 3 or any(v not in (0,1) for v in e[k]): raise ValueError('witness bits')
        if e['tool_proposal'] != (e['after'] if e['operation']=='accept-tool' else None):
            raise ValueError('tool proposal')


def legal_index(pulse=lambda **kw: None):
    """Exhaust all 8 starts, 4 mechanics assignments and legal 3-op paths.

    Hidden purpose/routine and policy choices are never consulted. A second
    scalar executor checks every primitive transition, including undo buffers.
    """
    index = defaultdict(set); executions = transitions = 0
    for initial in product((0,1), repeat=3):
        pulse(phase='public-mechanics-enumeration', initial=initial)
        for skill, belief in product((0,1), repeat=2):
            ops = [i for i in range(6) if skill or i != 3]
            for path in product(ops, repeat=3):
                a = previous = initial; events = []
                for step, op in enumerate(path):
                    after = L.execute(a, previous, L.OPERATIONS[op], (0,skill,belief,0))
                    if after != V.execute(a, previous, op, skill, belief): raise ValueError('independent mechanics')
                    events.append(dict(step=step,operation=L.OPERATIONS[op],before=list(a),after=list(after),
                                       tool_proposal=list(after) if op==3 else None))
                    previous,a=a,after; transitions += 1
                code = 36*path[0]+6*path[1]+path[2]
                for tier in TIERS:
                    for request in ((0,) if tier=='E0' else (0,1)):
                        inputs=dict(artifact=list(a))
                        if tier!='E0': inputs.update(initial=list(initial),requested_purpose=request)
                        if tier.startswith('E2'): inputs['observations']=events[:1] if tier=='E2-sparse' else events
                        key=digest(dict(schema='v19.local.public.1',tier=tier,inputs=inputs))
                        index[key].add(code)
                executions += 1
    return index, dict(executions=executions,primitive_transitions=transitions,
                      initial_artifacts=8,mechanics_assignments=4,goal_sequences_per_op_path=27)


def support(packet, index):
    validate(packet)
    mask = np.zeros(UNIVERSE, bool)
    for op in index.get(digest(packet), ()):
        mask[216*np.arange(27)+op] = True
    return mask


def expand(pred, alphabet, budget):
    p=np.asarray(pred,float); a=np.asarray(alphabet,int)
    if (p.ndim!=2 or p.shape[1]!=len(a) or len(set(a))!=len(a) or len(a)>=UNIVERSE
        or np.any(a<0) or np.any(a>=UNIVERSE) or not np.isfinite(p).all() or np.any(p<0)):
        raise ValueError('saved forecast')
    if not np.allclose(p.sum(1),budget/(budget+1),rtol=0,atol=1e-12): raise ValueError('unknown allocation')
    full=np.full((len(p),UNIVERSE),1/(budget+1)/(UNIVERSE-len(a)))
    full[:,a]=p
    return full


def restrict(p, mask):
    p=np.asarray(p,float); mask=np.asarray(mask,bool)
    if p.shape!=mask.shape or p.ndim!=2 or not np.isfinite(p).all() or np.any(p<0): raise ValueError('restriction shape')
    if not np.allclose(p.sum(1),1,rtol=0,atol=1e-12): raise ValueError('normalization')
    retained=np.sum(p*mask,axis=1)
    valid=mask.any(1)&(retained>0)
    # An invalid row is retained as an explicit fallback, not a successful restriction.
    result=p.copy(); result[valid]=p[valid]*mask[valid]/retained[valid,None]
    return result,retained,valid


def reference_arrays(references, tier, keys, lineages, masks):
    entries=[r for r in references if r['tier']==tier]
    if len(entries)!=len(lineages): raise ValueError('lineage coverage')
    bylin={r['lineage']:r for r in entries}; position={k:i for i,k in enumerate(keys)}
    template=bylin[lineages[0]]['frames']; coordinates=[]
    for frame in template:
        for k,w in frame['target']:
            if w<=0: raise ValueError('target support')
            coordinates.append((position[frame['frame']],k))
    coordinates=sorted(coordinates); coord={x:i for i,x in enumerate(coordinates)}
    rr=np.array([x[0] for x in coordinates]); kk=np.array([x[1] for x in coordinates])
    if not masks[rr,kk].all(): raise ValueError('true path excluded by public support')
    mass=np.zeros((len(lineages),len(keys))); weighted=np.zeros((len(lineages),len(rr))); squared=[]
    for li,lin in enumerate(lineages):
        seen=set(); sq=0.
        for frame in bylin[lin]['frames']:
            row=position[frame['frame']]; m=frame['mass']; mass[li,row]=m
            if abs(sum(w for _,w in frame['target'])-1)>1e-10: raise ValueError('target normalization')
            sq+=m*sum(w*w for _,w in frame['target'])
            for k,w in frame['target']:
                key=(row,k);seen.add(key);weighted[li,coord[key]]=m*w
        if seen!=set(coord) or abs(mass[li].sum()-1)>1e-10: raise ValueError('reference roster')
        squared.append(sq)
    native_support=np.zeros_like(masks);native_support[rr,kk]=True
    return dict(rows=rr,labels=kk,weighted=weighted,mass=mass,squared=np.array(squared),native_support=native_support)


def scores(p, alphabet, ref):
    """Original known-label priority, extended to unseen labels only as needed.

    At equal probabilities known alphabet labels precede unseen labels, each in
    original integer order. All coordinates remain in the fixed scored universe.
    Original metrics are reproduced before an extension is accepted.
    """
    alphabet=np.asarray(alphabet,int);unseen=np.setdiff1d(LABELS,alphabet)
    priority=np.r_[alphabet,unseen]
    order=priority[np.argsort(-p[:,priority],axis=1,kind='stable')]
    cumulative=np.cumsum(np.take_along_axis(p,order,axis=1),axis=1)
    size=(cumulative<.9).sum(1)+1
    if np.any(size>UNIVERSE):raise ValueError('candidate normalization')
    chosen=order[:,0]; covered=np.zeros(p.shape,bool)
    np.put_along_axis(covered,order,np.arange(UNIVERSE)[None,:]<size[:,None],axis=1)
    rr,kk=ref['rows'],ref['labels'];w=ref['weighted'];mass=ref['mass'];truthprob=p[rr,kk]
    if np.any(truthprob<=0):raise ValueError('true label has zero probability')
    acc=w@(DIGITS[kk]==DIGITS[chosen[rr]])
    result=dict(loss=-(w@np.log(truthprob)),squared_error=mass@np.square(p).sum(1)-2*(w@truthprob)+ref['squared'],
        compatible_mass=mass@(p*ref['native_support']).sum(1),candidate_coverage=w@covered[rr,kk],candidate_size=mass@size,
        top_incompatible=mass@(~ref['native_support'][np.arange(len(p)),chosen]),abstain=mass@(p[np.arange(len(p)),chosen]<.5),
        top_probability=mass@p[np.arange(len(p)),chosen],unknown_mass=mass@p[:,unseen].sum(1),
        truth_outside_alphabet=w@np.isin(kk,unseen),goal_accuracy=acc[:,:3].mean(1),operation_accuracy=acc[:,3:].mean(1))
    result.update({f'goal_{i}':acc[:,i] for i in range(3)});result.update({f'operation_{i}':acc[:,i+3] for i in range(3)})
    return result


def aggregate(rows,cfg):
    idx={(r['tier'],r['budget'],r['arm'],r['lineage'],r['draw'],r['seed']):r for r in rows}
    lins=cfg['development_lineages'];draws=cfg['training_draws'];seeds=cfg['fit_seeds'];budgets=cfg['budgets']
    random=np.random.default_rng(cfg['bootstrap_seed']);indices=random.integers(len(lins),size=(cfg['bootstrap_resamples'],len(lins)))
    def estimate(v):
        samples=v[indices].mean(1)
        return dict(mean=float(v.mean()),low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),lineage_values=v.tolist())
    contrasts=[];areas=[]
    for tier in cfg['tiers']:
        for arm,base in PAIRS:
            values=[]
            for budget in budgets:
                v=np.array([[[idx[tier,budget,arm,l,d,s]['loss']-idx[tier,budget,base,l,d,s]['loss'] for s in seeds] for d in draws] for l in lins])
                values.append(v)
                contrasts.append(dict(tier=tier,budget=budget,arm=arm,baseline=base,**estimate(v.mean((1,2))),draw_means=v.mean((0,2)).tolist(),feature_seed_means=v.mean((0,1)).tolist()))
            a=np.trapezoid(values,x=np.log(budgets),axis=0)/np.log(budgets[-1]/budgets[0])
            areas.append(dict(tier=tier,arm=arm,baseline=base,**estimate(a.mean((1,2))),draw_means=a.mean((0,2)).tolist(),feature_seed_means=a.mean((0,1)).tolist()))
    return contrasts,areas


def controls():
    p=np.array([[.2,.3,.5],[.1,.2,.7]]);mask=np.array([[True,False,True],[True,True,True]])
    q,z,valid=restrict(p,mask);empty=restrict(p,np.zeros_like(mask))[2]
    zero=restrict(np.array([[1.,0.]]),np.array([[False,True]]))[2]
    return {'live:support_restriction':bool(np.allclose(q[0],[2/7,0,5/7])),
        'placebo:full_support_identity':bool(np.array_equal(q[1],p[1])),
        'positive:empty_support_reported':bool(not empty.any()),'positive:zero_mass_reported':bool(not zero.any())}


def run(root,plan,pulse):
    cfg=plan['design'];base=root/'inputs';start=time.process_time();checks=controls()
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    parent=read(base/'PLAN.json')['design'];summary=read(base/'SUMMARY.json')
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if cfg[k]!=parent[k]:raise ValueError('parent population changed')
    packets=read(base/'reader/PACKETS.json')['packets'];references=read(base/'evaluator/REFERENCES.json')
    for folder in ('reader','evaluator','raw'):(root/folder).mkdir()
    index,enumeration=legal_index(pulse);support_time=time.process_time()-start
    write(root/'reader/PACKETS.json',dict(schema='v19.joint-readout.reader.1',packets=packets))
    oldidx={tuple(r[k] for k in ('draw','tier','seed','budget','arm','lineage')):r for r in summary['cells']}
    rows=[];max_error=0.;reproduced=0;timing=[];stats=[]
    for tier in cfg['tiers']:
        keys=sorted(k for k,p in packets.items() if p['tier']==tier)
        masks=np.array([support(packets[k],index) for k in keys])
        if not masks.any(1).all():raise ValueError('empty legal support')
        ref=reference_arrays(references,tier,keys,cfg['development_lineages'],masks)
        np.savez_compressed(root/'evaluator'/f'{tier}-support.npz',masks=masks)
        write(root/'evaluator'/f'{tier}-frames.json',keys)
        stats.append(dict(tier=tier,packets=len(keys),minimum_labels=int(masks.sum(1).min()),maximum_labels=int(masks.sum(1).max())))
        uniform=masks/masks.sum(1,keepdims=True)
        for draw,seed,budget in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets']):
            metrics={};normalizers={}
            for arm in ARMS:
                pulse(phase='joint-support-scoring',tier=tier,draw=draw,seed=seed,budget=budget,arm=arm)
                began=time.process_time();stem=f'{draw}-{tier}-{seed}-{budget}-{arm}'
                if read(base/'forecasts'/(stem+'-frames.json'))!=keys:raise ValueError('frame identity')
                with np.load(base/'forecasts'/(stem+'.npz'),allow_pickle=False) as z:
                    alphabet=z['alphabet'];full=expand(z['probabilities'],alphabet,budget)
                original=scores(full,alphabet,ref)
                for li,lin in enumerate(cfg['development_lineages']):
                    old=oldidx[draw,tier,seed,budget,arm,lin]
                    for k,v in original.items():
                        error=abs(v[li]-old[k]);max_error=max(max_error,float(error))
                        if error>1e-8:raise ValueError(f'original number reproduction: {k} {error}')
                    reproduced+=1
                restricted,retained,valid=restrict(full,masks)
                if not valid.all():raise ValueError('support restriction failed; original fallback retained in inputs')
                metrics[arm]=original;metrics[arm+'-restricted']=scores(restricted,alphabet,ref)
                normalizers[arm]=retained
                if arm=='matched-frequency':metrics['uniform-support']=scores(uniform,alphabet,ref)
                timing.append(dict(tier=tier,draw=draw,seed=seed,budget=budget,arm=arm,cpu_seconds=time.process_time()-began))
            np.savez_compressed(root/'raw'/f'{draw}-{tier}-{seed}-{budget}-normalizers.npz',**normalizers)
            for arm,m in metrics.items():
                for li,lin in enumerate(cfg['development_lineages']):
                    rows.append(dict(draw=draw,tier=tier,seed=seed,budget=budget,arm=arm,lineage=lin,frames=len(keys),**{k:float(v[li]) for k,v in m.items()}))
    if reproduced!=len(summary['cells']):raise ValueError('original cell coverage')
    (root/'raw/support_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    contrasts,areas=aggregate(rows,cfg)
    write(root/'ORIGINAL_REPRODUCTION.json',dict(passed=True,cells=reproduced,max_error=max_error,metrics='all original fields',source='frozen probability arrays; complete original native weighted target population'))
    write(root/'CONTROLS.json',checks)
    write(root/'TIMING.jsonl',dict(support_enumeration_cpu_seconds=support_time,measurements=timing))
    write(root/'EVIDENCE_ROLES.json',dict(reader='unchanged anonymous public packets',public_method='legal execution support including all goal sequences; no controller or native policy',evaluator='posterior targets, native weights, scores and fitted forecast inputs; not reader input'))
    return dict(controls=checks,rows=len(rows),support=stats,enumeration=enumeration,original_cells_reproduced=reproduced,original_max_error=max_error,
        contrasts=contrasts,normalized_log_budget_area=areas,fits=0,scope='supplied public-mechanics restriction of frozen learned forecasts; native process correspondence scored separately; miniature — architecture untested')
