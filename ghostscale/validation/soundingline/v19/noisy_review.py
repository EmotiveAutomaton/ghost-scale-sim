"""Independent scalar reconstruction of noisy mechanical disclosure."""
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import disclosure_review as D

POLICIES = ('none', 'skill', 'belief', 'entropy-choice')
MODELS = ('uniform-legal', 'native-law')
READERS = ('channel-aware', 'trusting')
RELIABILITIES = (.5, .75, 1.)
COSTS = (0., .02, .1)
AXES = ('rule', 'model', 'reliability', 'reader', 'policy', 'weighting', 'cost')
METRICS = (*D.METRICS, 'request_rate')


def conditional(legal, ends, weights, accuracy):
    """Endpoint sums from scalar Bayes weights, including explicit zero support."""
    if not .5 <= accuracy <= 1 or not math.isfinite(accuracy):
        raise ValueError('channel accuracy')
    if not legal or len(legal) != len(ends) or len(legal) != len(weights):
        raise ValueError('law shape')
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('law weight')
    D.V.V.near([math.fsum(weights)], [1.])
    prior = np.array([math.fsum(w for w,e in zip(weights,ends) if e==t) for t in range(8)])
    tables = {}; masses = {}; entropy = {}; fallback = {}
    for field,axis in (('skill',0),('belief',1)):
        tables[field] = {}; masses[field] = {}; fallback[field] = {}; terms = []
        for bit in (0,1):
            likelihood = [accuracy if q[axis]==bit else 1-accuracy for q in legal]
            joint = [w*l for w,l in zip(weights,likelihood,strict=True)]
            mass = math.fsum(joint); masses[field][bit] = mass
            if mass:
                p = [math.fsum(w for w,e in zip(joint,ends) if e==t)/mass for t in range(8)]
                fallback[field][bit] = None
            else:
                ids = [i for i,l in enumerate(likelihood) if l > 0]
                fallback[field][bit] = 'uniform-likelihood-supported-legal' if ids else 'uniform-whole-legal-group'
                if not ids: ids = list(range(len(legal)))
                p = [sum(ends[i]==t for i in ids)/len(ids) for t in range(8)]
            tables[field][bit] = np.array(p)
            terms.append(mass*D.V.entropy(p))
        entropy[field] = math.fsum(terms)
    return dict(prior=prior, tables=tables, reply_mass=masses, expected_entropy=entropy,
                choice='skill' if entropy['skill'] <= entropy['belief'] else 'belief', fallbacks=fallback)


def scores(predictions, probabilities, targets, weights):
    predictions=np.asarray(predictions); probabilities=np.asarray(probabilities)
    if predictions.shape != (len(targets),2,8) or probabilities.shape != (len(targets),2):
        raise ValueError('reply shape')
    if not np.isfinite(probabilities).all() or np.any(probabilities<0): raise ValueError('reply probability')
    for p in probabilities: D.V.V.near([math.fsum(p)],[1.])
    parts=[D.V.scores(predictions[:,bit],targets,[float(w)*float(p[bit]) for w,p in zip(weights,probabilities,strict=True)]) for bit in (0,1)]
    result={k:math.fsum(part[k] for part in parts) for k in parts[0]}
    result['residual_ambiguous_mass']=math.fsum(float(w)*float(pr[b]) for p,pr,w in zip(predictions,probabilities,weights,strict=True) for b in (0,1) if np.count_nonzero(p[b])>1)
    return result


def regroup(cells,lineages,cfg):
    lookup={(r['lineage'],*(r[k] for k in AXES)):r for r in cells}
    if len(lookup)!=len(cells):raise ValueError('duplicate cells')
    estimates=[]
    for key in sorted({tuple(r[k] for k in AXES) for r in cells}):
        bases=[('mean',None)]
        if key[4]!='none':bases.append(('minus-none',(*key[:4],'none',*key[5:])))
        if key[3]=='trusting':bases.append(('minus-channel-aware',(*key[:3],'channel-aware',*key[4:])))
        if key[4]=='entropy-choice':
            bases += [(f'minus-{p}',(*key[:4],p,*key[5:])) for p in ('skill','belief')]
        for contrast,base in bases:
            for metric in METRICS:
                values=[lookup[(lin,*key)][metric]-(lookup[(lin,*base)][metric] if base else 0.) for lin in lineages]
                estimates.append(dict(zip(AXES,key),contrast=contrast,metric=metric,**D.V.V.interval(values,cfg)))
    return dict(estimates=estimates,lineages=lineages,bootstrap_seed=cfg['bootstrap_seed'],bootstrap_resamples=cfg['bootstrap_resamples'],
                population='retained development laws weighted equally; native and equal-query populations separate',
                limitation='conditional on supplied laws and fixed queries; finite loss is not total loss when infinite mass is positive; no fits or confirmation')


def review(original,output,cfg,pulse=lambda **kw:None):
    design=read(original/'PLAN.json')['design']; summary=read(original/'SUMMARY.json');base=original/'inputs'
    for key,expected in [('policies',POLICIES),('models',MODELS),('readers',READERS),('reliabilities',RELIABILITIES),('costs',COSTS),('rules',D.V.T.RULES)]:
        if design[key]!=list(expected):raise ValueError('design')
    laws=read(base/'DISCLOSURE_LAWS.json');lookup={(r['lineage'],r['rule'],r['model'],r['reader_id']):r for r in laws}
    recorded=read(original/'evaluator/CHANNEL_LAWS.json')
    channel_lookup={(r['lineage'],r['rule'],r['model'],r['reliability'],r['reader_id']):r for r in recorded}
    if len(lookup)!=len(laws) or len(channel_lookup)!=len(recorded):raise ValueError('duplicate laws')
    cells=[];rows=[];sensitivity=[];scalar_sensitivity=[];error=0.;seen=set();input_seen=set();used=set();queries=None;field_ties=0;parent_ties=0;fallbacks=0
    for lin,rule,model in product(design['lineages'],design['rules'],MODELS):
        pulse(phase='independent-noisy-disclosure',lineage=lin,rule=rule,model=model)
        name=f'{lin}-{rule}-{model}_points.npz';input_seen.add(name)
        with np.load(base/'forecasts'/name,allow_pickle=False) as parent:
            qs=[tuple(map(int,q)) for q in parent['queries']];n=len(qs)
            if n!=design['queries'] or len(set(qs))!=n:raise ValueError('query roster')
            if queries is None:
                queries=qs;groups=D.V.groups_for(qs,'omit-both')
                if groups!=read(base/'MEMBERSHIP.json')['omit-both']:raise ValueError('membership')
            elif qs!=queries:raise ValueError('query identity')
            index={q:i for i,q in enumerate(qs)};targets=np.array([D.V.T.endpoint(q,rule) for q in qs]);mass=parent['mass'].copy()
            if mass.shape!=(n,) or not np.isfinite(mass).all() or np.any(mass<0):raise ValueError('population')
            error=max(error,D.V.V.near(parent['targets'],targets,0),D.V.V.near([math.fsum(mass)],[1.]))
            for accuracy in RELIABILITIES:
                before=np.zeros((n,8));choices=np.zeros(n,dtype=int);alt=choices.copy();scalar=choices.copy()
                tables={r:{f:np.zeros((n,2,8)) for f in ('skill','belief')} for r in READERS}
                for key,g in groups.items():
                    law=lookup[(lin,rule,model,key)];used.add((lin,rule,model,key));legal=g['legal_completions'];ends=[D.V.T.endpoint(q,rule) for q in legal]
                    raw=[float(mass[index[tuple(q)]]) if tuple(q) in index else 0. for q in legal];total=math.fsum(raw)
                    weights=[w/total for w in raw] if model=='native-law' and total else [1/len(legal)]*len(legal)
                    if law['legal_completions']!=legal or law['endpoints']!=ends:raise ValueError('mechanical law')
                    error=max(error,D.V.V.near(law['conditional_weights'],weights))
                    aware=conditional(legal,ends,weights,accuracy);trust=conditional(legal,ends,weights,1.)
                    row=channel_lookup[(lin,rule,model,accuracy,key)];choice=row['choice'];alternate=row['alternate_choice']
                    if choice not in ('skill','belief') or alternate not in ('skill','belief'):raise ValueError('choice')
                    entropy=aware['expected_entropy'];gap=abs(entropy['skill']-entropy['belief'])
                    if choice!=aware['choice']:
                        if gap>1e-14:raise ValueError('selection')
                        field_ties+=1
                    # Validate the vector diagnostic independently of score construction.
                    alt_entropy={f:float(np.sum([aware['reply_mass'][f][b]*float(-np.sum(p[p>0]*np.log(p[p>0]))) for b,p in aware['tables'][f].items()])) for f in ('skill','belief')}
                    alt_choice='skill' if alt_entropy['skill']<=alt_entropy['belief'] else 'belief'
                    if alternate!=alt_choice and abs(alt_entropy['skill']-alt_entropy['belief'])>1e-14:raise ValueError('alternate selection')
                    expected=dict(lineage=lin,rule=rule,model=model,reliability=accuracy,reader_id=key,expected_entropy=entropy,choice=choice,alternate_choice=alternate,alternate_entropy=alt_entropy,reply_mass=aware['reply_mass'],fallbacks={r:d['fallbacks'] for r,d in [('channel-aware',aware),('trusting',trust)]})
                    # JSON serializes binary reply keys as strings.
                    import json
                    error=max(error,D.V.compare(row,json.loads(json.dumps(expected))))
                    rows.append(expected);ids=g['indices'];before[ids]=aware['prior'];choices[ids]=int(choice=='belief');alt[ids]=int(alternate=='belief');scalar[ids]=int(aware['choice']=='belief')
                    for reader,dist in [('channel-aware',aware),('trusting',trust)]:
                        for field in ('skill','belief'):
                            tables[reader][field][ids]=np.stack([dist['tables'][field][b] for b in (0,1)])
                            fallbacks+=sum(v is not None for v in dist['fallbacks'][field].values())
                    if accuracy==.5:
                        for t in aware['tables'].values():
                            for p in t.values():error=max(error,D.V.V.near(p,aware['prior']))
                    if accuracy==1:
                        for i in ids:
                            error=max(error,D.V.V.near(parent['none'][i],aware['prior']))
                            for field,axis in (('skill',0),('belief',1)):
                                error=max(error,D.V.V.near(parent[field][i],aware['tables'][field][qs[i][axis]]))
                            old_choice='belief' if parent['choices'][i] else 'skill'
                            if old_choice!=choice:
                                if gap>1e-14:raise ValueError('truthful parent selection')
                                parent_ties+=1
                            error=max(error,D.V.V.near(parent['entropy-choice'][i],aware['tables'][old_choice][qs[i][int(old_choice=='belief')]]))
                for reader in READERS:
                    fname=f'{lin}-{rule}-{model}-{accuracy:g}-{reader}_points.npz';seen.add(fname);policy_scores={}
                    with np.load(original/'forecasts'/fname,allow_pickle=False) as saved:
                        keys={'queries','mass','targets','choices','alternate_choices',*POLICIES,'alternate-choice',*(p+'-reply-probabilities' for p in (*POLICIES,'alternate-choice'))}
                        if set(saved.files)!=keys:raise ValueError('forecast schema')
                        for k,v in dict(queries=qs,mass=mass,targets=targets,choices=choices,alternate_choices=alt).items():error=max(error,D.V.V.near(saved[k],v,0 if k!='mass' else 1e-12))
                        for policy in (*POLICIES,'alternate-choice','scalar-choice'):
                            axis=np.zeros(n,dtype=int) if policy in ('none','skill') else np.ones(n,dtype=int) if policy=='belief' else alt if policy=='alternate-choice' else scalar if policy=='scalar-choice' else choices
                            pred=np.repeat(before[:,None,:],2,axis=1) if policy=='none' else np.stack([tables[reader]['belief' if a else 'skill'][i] for i,a in enumerate(axis)])
                            probs=np.tile([1.,0.],(n,1)) if policy=='none' else np.array([[accuracy if q[int(a)]==b else 1-accuracy for b in (0,1)] for q,a in zip(qs,axis)])
                            if policy!='scalar-choice':error=max(error,D.V.V.near(saved[policy],pred),D.V.V.near(saved[policy+'-reply-probabilities'],probs))
                            for weighting,w in (('native',mass),('equal-query',np.full(n,1/n))):
                                result=scores(pred,probs,targets,w);policy_scores[(policy,weighting)]=result
                                if policy not in POLICIES:continue
                                fields=int(policy!='none')
                                for cost in COSTS:
                                    cells.append(dict(lineage=lin,rule=rule,model=model,reliability=accuracy,reader=reader,policy=policy,weighting=weighting,cost=cost,queries=n,groups=len(groups),request_rate=fields,skill_request_rate=math.fsum(float(x) for x,a in zip(w,axis) if a==0)*fields,belief_request_rate=math.fsum(float(x) for x,a in zip(w,axis) if a==1)*fields,net_finite_loss=result['finite_loss_contribution']+cost*fields,**result))
                    for weighting in ('native','equal-query'):
                        common=dict(lineage=lin,rule=rule,model=model,reliability=accuracy,reader=reader,weighting=weighting)
                        for label,choices_other,dest in [('alternate-choice',alt,sensitivity),('scalar-choice',scalar,scalar_sensitivity)]:
                            dest.append(dict(**common,changed_queries=int(np.count_nonzero(choices!=choices_other)),score_changes={k:policy_scores[(label,weighting)][k]-policy_scores[('entropy-choice',weighting)][k] for k in policy_scores[(label,weighting)]}))
    if len(rows)!=len(recorded) or len(used)!=len(laws):raise ValueError('law coverage')
    if {p.name for p in (original/'forecasts').glob('*.npz')}!=seen or {p.name for p in (base/'forecasts').glob('*.npz')}!=input_seen:raise ValueError('forecast coverage')
    error=max(error,D.V.compare(summary['cells'],cells),D.V.compare(read(original/'evaluator/TIE_SENSITIVITY.json'),sensitivity))
    packets={k:g['reader'] for k,g in groups.items()}
    for g in groups.values():
        for field,accuracy,bit in product(('skill','belief_error'),RELIABILITIES,(0,1)):
            packet=dict(g['reader'],requested_field=field,reply_value=bit,reply_reliability=accuracy);packets[digest(packet)]=packet
    if {p.stem for p in (original/'reader').glob('*.json')}!=set(packets):raise ValueError('reader coverage')
    for key,p in packets.items():
        if read(original/'reader'/f'{key}.json')!=p:raise ValueError('reader projection')
    checks=read(original/'CONTROLS.json')
    if not checks or not all(checks.values()) or checks!=summary['controls']:raise ValueError('controls')
    if summary['queries']!=len(queries) or summary['law_rows']!=len(rows) or summary['fits']!=0:raise ValueError('totals')
    timing=read(original/'TIMING.jsonl')
    if timing['fits']!=0 or timing['cpu_seconds']<0:raise ValueError('timing')
    write(output/'RECONSTRUCTED_CELLS.json',cells);write(output/'INDEPENDENT_REGROUP.json',regroup(cells,design['lineages'],cfg))
    write(output/'SELECTION_SENSITIVITY.json',dict(scalar_choices=scalar_sensitivity,vector_choices=sensitivity,field_roundoff_ties=field_ties,truthful_parent_tied_queries=parent_ties,scope='original literal choices preserved only within independently checked 1e-14 entropy ties; score effects separate'))
    result=dict(passed=True,cells=len(cells),law_rows=len(rows),queries=len(queries),reader_packets=len(packets),forecast_vectors=len(seen)*len(queries)*10,max_error=error,field_roundoff_ties=field_ties,truthful_parent_tied_queries=parent_ties,zero_model_mass_fallbacks=fallbacks,scope='independent legal mechanics,scalar noisy Bayes,all reply probabilities and branch proper scores,half-channel identity,truthful-parent forecasts,fallbacks,infinite loss,costs,all strata and reader allowlists; native masses inherit verified parent')
    write(output/'NUMERICAL_REVIEW.json',result);return result


def run(root,plan,pulse):
    cfg=plan['design'];original=root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target binding')
    result=review(original,root,cfg,pulse)
    return dict(result,controls={'live:all_reply_scores_reconstructed':True,'positive:half_and_truthful_identities':True,'placebo:zero_model_mass_and_explicit_infinite_loss':True})
