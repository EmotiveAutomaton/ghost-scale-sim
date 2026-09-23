"""Scalar reconstruction of fixed decisions under unknown reply reliability."""
from itertools import product
import math
import numpy as np
from ..v18_3.io import read,write,digest,file_digest
from . import noisy_review as N

ARMS=('none','equal-mixture','normalized-envelope')
AXES=('rule','model','field','arm','actual_accuracy','weighting','cost')
METRICS=('finite_loss_contribution','infinite_loss_mass','squared_error','true_probability',
         'residual_ambiguous_mass','net_finite_loss','expected_log_normalizer','actual_candidate_excluded_mass')


def review(original,out,cfg,pulse=lambda **kw:None):
    design=read(original/'PLAN.json')['design'];summary=read(original/'SUMMARY.json');base=original/'inputs';near=N.D.V.V.near
    for k,v in [('arms',ARMS),('reliabilities',N.RELIABILITIES),('models',N.MODELS),('costs',N.COSTS),('fields',('skill','belief')),('rules',N.D.V.T.RULES)]:
        if design[k]!=list(v):raise ValueError('design')
    laws=read(base/'DISCLOSURE_LAWS.json');lookup={(r['lineage'],r['rule'],r['model'],r['reader_id']):r for r in laws}
    if len(lookup)!=len(laws):raise ValueError('duplicate laws')
    error=0.;cells=[];used=set();seen=set();roster=None;fallbacks=[];inequalities=0
    for lin,rule,model in product(design['lineages'],design['rules'],N.MODELS):
        pulse(phase='independent-robust-reply',lineage=lin,rule=rule,model=model)
        name=f'{lin}-{rule}-{model}_points.npz';seen.add(name)
        with np.load(base/'forecasts'/name,allow_pickle=False) as parent,np.load(original/'forecasts'/name,allow_pickle=False) as saved:
            qs=[tuple(map(int,q)) for q in parent['queries']];n=len(qs);mass=parent['mass'];targets=np.array([N.D.V.T.endpoint(q,rule) for q in qs])
            if n!=design['queries'] or len(set(qs))!=n:raise ValueError('query roster')
            if roster is None:roster=qs
            elif roster!=qs:raise ValueError('query identity')
            if mass.shape!=(n,) or np.any(mass<0) or not np.isfinite(mass).all():raise ValueError('population')
            error=max(error,near(parent['targets'],targets,0),near([math.fsum(mass)],[1.]))
            groups=N.D.V.groups_for(qs,'omit-both');ids=sorted(groups);index={q:i for i,q in enumerate(qs)}
            if groups!=read(base/'MEMBERSHIP.json')['omit-both']:raise ValueError('membership')
            shape=(len(ids),2,2);expected=dict(queries=np.array(qs),mass=mass,targets=targets,candidates=np.zeros((*shape,3,8)),
                modeled_reply_mass=np.zeros((*shape,3)),compatible=np.zeros((*shape,3),bool),normalizer=np.zeros(shape),
                prior=np.zeros((len(ids),8)),mixture=np.zeros((*shape,8)),envelope=np.zeros((*shape,8)))
            qgroup=np.full(n,-1,int)
            for gi,key in enumerate(ids):
                g=groups[key];ident=(lin,rule,model,key);used.add(ident);law=lookup[ident];legal=g['legal_completions'];ends=[N.D.V.T.endpoint(q,rule) for q in legal]
                if legal!=law['legal_completions'] or ends!=law['endpoints']:raise ValueError('mechanics')
                raw=[float(mass[index[tuple(q)]]) if tuple(q) in index else 0. for q in legal];total=math.fsum(raw)
                weights=[w/total for w in raw] if model=='native-law' and total else [1/len(legal)]*len(legal)
                error=max(error,near(law['conditional_weights'],weights));ch=[N.conditional(legal,ends,weights,a) for a in N.RELIABILITIES]
                expected['prior'][gi]=ch[0]['prior'];qgroup[g['indices']]=gi
                for fi,field in enumerate(('skill','belief')):
                    for bit in (0,1):
                        ps=[v['tables'][field][bit] for v in ch];ms=[v['reply_mass'][field][bit] for v in ch];valid=[i for i,m in enumerate(ms) if m>0]
                        upper=[max(ps[i][y] for i in valid) for y in range(8)];z=math.fsum(upper)
                        mix=[math.fsum(ps[i][y] for i in valid)/len(valid) for y in range(8)];env=[v/z for v in upper]
                        for i in valid:
                            for y in range(8):
                                if env[y]*z+1e-14<ps[i][y]:raise ValueError('coding bound')
                        inequalities+=len(valid)*8
                        for k,v in [('candidates',ps),('modeled_reply_mass',ms),('compatible',[m>0 for m in ms]),('normalizer',z),('mixture',mix),('envelope',env)]:expected[k][gi,fi,bit]=v
                        for ai,d in enumerate(ch):
                            if not ms[ai]:fallbacks.append(dict(lineage=lin,rule=rule,model=model,reader_id=key,field=field,reply=bit,reliability=N.RELIABILITIES[ai],convention=d['fallbacks'][field][bit]))
            if np.any(qgroup<0):raise ValueError('coverage')
            expected['query_group']=qgroup
            for fi,field in enumerate(('skill','belief')):
                for arm in ARMS:
                    pred=np.repeat(expected['prior'][qgroup,None,:],2,axis=1) if arm=='none' else expected['mixture' if arm=='equal-mixture' else 'envelope'][qgroup,fi]
                    true=np.array([[p[int(y)] for p in rows] for rows,y in zip(pred,targets,strict=True)])
                    finite=np.array([[-math.log(p) if p>0 else 0. for p in row] for row in true])
                    squared=np.array([[math.fsum((float(p[y])-int(y==t))**2 for y in range(8)) for p in rows] for rows,t in zip(pred,targets,strict=True)])
                    prefix=f'{field}-{arm}'
                    for k,v in [('true-probability',true),('finite-log-loss',finite),('infinite-loss',true==0),('squared-error',squared)]:expected[prefix+'-'+k]=v
                    for ai,actual in enumerate(N.RELIABILITIES):
                        probs=np.array([[actual if q[fi]==b else 1-actual for b in (0,1)] for q in qs]);expected[f'{field}-actual-{actual:g}-reply-probabilities']=probs
                        for weighting,w in [('native',mass),('equal-query',np.full(n,1/n))]:
                            scores=N.scores(pred,probs,targets,w)
                            excluded=math.fsum(float(w[i])*float(probs[i,b])*int(not expected['compatible'][qgroup[i],fi,b,ai]) for i in range(n) for b in (0,1))
                            bound=math.fsum(float(w[i])*float(probs[i,b])*math.log(expected['normalizer'][qgroup[i],fi,b]) for i in range(n) for b in (0,1))
                            for cost in N.COSTS:
                                count=int(arm!='none');cells.append(dict(lineage=lin,rule=rule,model=model,field=field,arm=arm,actual_accuracy=actual,weighting=weighting,cost=cost,request_rate=count,queries=n,groups=len(ids),expected_log_normalizer=bound,actual_candidate_excluded_mass=excluded,net_finite_loss=scores['finite_loss_contribution']+cost*count,**scores))
            if set(saved.files)!=set(expected):raise ValueError('schema')
            for k,v in expected.items():error=max(error,near(saved[k],v,0 if k in ('queries','targets','query_group','compatible') or k.endswith('infinite-loss') else 1e-12))
    if used!=set(lookup) or {p.name for p in (original/'forecasts').glob('*.npz')}!=seen or {p.name for p in (base/'forecasts').glob('*.npz')}!=seen:raise ValueError('coverage')
    error=max(error,N.D.V.compare(summary['cells'],cells))
    if read(original/'evaluator/FALLBACKS.json')!=fallbacks:raise ValueError('fallback')
    packets={digest(g['reader']):g['reader'] for g in groups.values()}
    for g in groups.values():
        for field,bit in product(('skill','belief_error'),(0,1)):
            p=dict(g['reader'],requested_field=field,reply_value=bit,reliability_candidates=list(N.RELIABILITIES));packets[digest(p)]=p
    if read(original/'reader/REPLIES.json')!=packets or {p.name for p in (original/'reader').iterdir()}!={'REPLIES.json'}:raise ValueError('reader')
    if summary['reader_packets']!=len(packets) or summary['coding_inequalities']!=inequalities or summary['queries']!=len(roster) or summary['fits']!=0:raise ValueError('totals')
    checks=read(original/'CONTROLS.json')
    if not checks or not all(checks.values()) or checks!=summary['controls']:raise ValueError('controls')
    keyed={(r['lineage'],*(r[k] for k in AXES)):r for r in cells};estimates=[]
    if len(keyed)!=len(cells):raise ValueError('duplicate cells')
    for key in sorted({k[1:] for k in keyed}):
        contrasts=['mean'] if key[3]=='none' else ['mean','minus-none']
        for contrast,metric in product(contrasts,METRICS):
            basekey=(*key[:3],'none',*key[4:])
            values=[keyed[(lin,*key)][metric]-(keyed[(lin,*basekey)][metric] if contrast=='minus-none' else 0.) for lin in design['lineages']]
            estimates.append(dict(zip(AXES,key),metric=metric,contrast=contrast,**N.D.V.V.interval(values,cfg)))
    write(out/'RECONSTRUCTED_CELLS.json',cells);write(out/'INDEPENDENT_REGROUP.json',dict(estimates=estimates,lineages=design['lineages'],bootstrap_seed=cfg['bootstrap_seed'],bootstrap_resamples=cfg['bootstrap_resamples'],scope='fixed supplied-law development population; infinite loss and finite contribution separate'))
    result=dict(passed=True,cells=len(cells),coding_inequalities=inequalities,reader_packets=len(packets),max_error=error,scope='independent scalar mechanics,conditional decisions,branch scores,coding bound,roles and population summaries; native masses inherit verified parent')
    write(out/'NUMERICAL_REVIEW.json',result);return result


def run(root,plan,pulse):
    cfg=plan['design'];original=root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target binding')
    return dict(review(original,root,cfg,pulse),controls={'live:complete_scalar_reconstruction':True,'positive:coding_bound_and_support':True,'placebo:reader_allowlist_and_no_request':True})
