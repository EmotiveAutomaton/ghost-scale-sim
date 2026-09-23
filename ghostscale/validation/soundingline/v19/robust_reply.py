"""Fixed forecast decisions for a finite set of unspecified reply channels."""
from itertools import product
import math
import time
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import metadata_disclosure as D
from .noisy_disclosure import channel, expected_scores

ACCURACIES = (.5, .75, 1.)
FIELDS = ('skill', 'belief')
ARMS = ('none', 'equal-mixture', 'normalized-envelope')
COSTS = (0., .02, .1)


def decisions(candidates, masses):
    p=np.asarray(candidates,float);m=np.asarray(masses,float)
    if (p.ndim!=2 or p.shape[1]!=8 or m.shape!=(len(p),) or not len(p)
        or not np.isfinite(p).all() or not np.isfinite(m).all() or np.any(p<0) or np.any(m<0)
        or np.any(m>1+1e-12) or not np.allclose(p.sum(1),1,atol=1e-12,rtol=0)):raise ValueError('candidate set')
    valid=m>0
    if not valid.any():raise ValueError('empty compatible set')
    upper=p[valid].max(0);z=float(upper.sum());envelope=upper/z;mixture=p[valid].mean(0)
    if np.any(envelope[None,:]*z+1e-14<p[valid]):raise ValueError('coding bound')
    return dict(mixture=mixture,envelope=envelope,normalizer=z,compatible=valid)


def controls():
    p=np.zeros((3,8));p[0,0]=1;p[1,1]=1;p[2,:2]=.5
    live=decisions(p,[.5]*3);dup=decisions(np.vstack([p,p[0]]),[.5]*4)
    singleton=decisions(p,[1,0,0]);null=decisions(np.tile(p[0],(3,1)),[.5]*3)
    return {'live:disjoint_support_coding_bound':live['normalizer']==2 and np.array_equal(live['envelope'][:2],[.5,.5]),
            'placebo:constant_candidates_collapse':np.array_equal(null['envelope'],p[0]),
            'positive:impossible_candidates_excluded':np.array_equal(singleton['mixture'],p[0]),
            'positive:duplicate_envelope_invariant':np.array_equal(dup['envelope'],live['envelope']),
            'live:duplicate_changes_equal_mixture':not np.array_equal(dup['mixture'],live['mixture'])}


def run(root,plan,pulse):
    cfg=plan['design'];base=root/'inputs';start=time.process_time();checks=controls()
    if (cfg['reliabilities']!=list(ACCURACIES) or cfg['fields']!=list(FIELDS)
        or cfg['arms']!=list(ARMS) or cfg['costs']!=list(COSTS) or cfg['models']!=list(D.MODELS)
        or not all(checks.values())):raise ValueError('design')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    groups=read(base/'MEMBERSHIP.json')['omit-both'];ids=sorted(groups);laws=read(base/'DISCLOSURE_LAWS.json')
    lookup={(r['lineage'],r['rule'],r['model'],r['reader_id']):r for r in laws}
    if len(lookup)!=len(laws):raise ValueError('duplicate law')
    for d in ('reader','forecasts','evaluator'):(root/d).mkdir()
    packets={digest(g['reader']):g['reader'] for g in groups.values()}
    for g in groups.values():
        for field,bit in product(('skill','belief_error'),(0,1)):
            p=dict(g['reader'],requested_field=field,reply_value=bit,reliability_candidates=list(ACCURACIES));packets[digest(p)]=p
    write(root/'reader/REPLIES.json',packets);write(root/'CONTROLS.json',checks)
    write(root/'evaluator/INDEX.json',dict(group_ids=ids,fields=FIELDS,replies=[0,1],reliabilities=ACCURACIES,arms=ARMS,
        bound='normalized envelope >= each compatible forecast / normalizer; no actual-channel guarantee outside supplied support'))
    cells=[];seen=set();roster=None;bounds=0;fallbacks=[]
    for lin,rule,model in product(cfg['lineages'],cfg['rules'],D.MODELS):
        pulse(phase='robust-finite-reply',lineage=lin,rule=rule,model=model)
        with np.load(base/'forecasts'/f'{lin}-{rule}-{model}_points.npz',allow_pickle=False) as parent:
            qs=[tuple(map(int,q)) for q in parent['queries']];n=len(qs);mass=parent['mass'].copy();targets=parent['targets'].copy()
            if n!=cfg['queries'] or len(set(qs))!=n:raise ValueError('query roster')
            if roster is None:roster=qs
            elif qs!=roster:raise ValueError('query identity')
            if D.P.membership(qs)['omit-both']!=groups or not np.array_equal(targets,[D.P.T.oracle(q,rule) for q in qs]):raise ValueError('reference')
            if mass.shape!=(n,) or not np.isfinite(mass).all() or np.any(mass<0) or not math.isclose(float(mass.sum()),1,abs_tol=1e-12,rel_tol=0):raise ValueError('mass')
            shape=(len(ids),2,2);saved=dict(queries=np.array(qs),mass=mass,targets=targets,
                candidates=np.zeros((*shape,3,8)),modeled_reply_mass=np.zeros((*shape,3)),compatible=np.zeros((*shape,3),bool),
                normalizer=np.zeros(shape),prior=np.zeros((len(ids),8)),mixture=np.zeros((*shape,8)),envelope=np.zeros((*shape,8)))
            qgroup=np.full(n,-1,np.int32)
            for gi,key in enumerate(ids):
                g=groups[key];ident=(lin,rule,model,key);seen.add(ident);law=lookup[ident];legal=law['legal_completions'];ends=law['endpoints']
                if legal!=g['legal_completions'] or ends!=[D.P.T.oracle(q,rule) for q in legal]:raise ValueError('law mechanics')
                ch=[channel([tuple(q) for q in legal],ends,law['conditional_weights'],a) for a in ACCURACIES]
                saved['prior'][gi]=ch[0]['prior'];qgroup[g['indices']]=gi
                for fi,field in enumerate(FIELDS):
                    for bit in (0,1):
                        p=np.array([d['tables'][field][bit] for d in ch]);m=np.array([d['reply_mass'][field][bit] for d in ch]);r=decisions(p,m)
                        saved['candidates'][gi,fi,bit]=p;saved['modeled_reply_mass'][gi,fi,bit]=m
                        for k in ('compatible','normalizer','mixture','envelope'):saved[k][gi,fi,bit]=r[k]
                        bounds+=int(r['compatible'].sum())*8
                        for ai,d in enumerate(ch):
                            if not m[ai]:fallbacks.append(dict(lineage=lin,rule=rule,model=model,reader_id=key,field=field,reply=bit,reliability=ACCURACIES[ai],convention=d['fallbacks'][field][bit]))
            if np.any(qgroup<0):raise ValueError('query coverage')
            saved['query_group']=qgroup
            for fi,field in enumerate(FIELDS):
                for arm in ARMS:
                    pred=np.repeat(saved['prior'][qgroup,None,:],2,axis=1) if arm=='none' else saved['mixture' if arm=='equal-mixture' else 'envelope'][qgroup,fi]
                    true=pred[np.arange(n)[:,None],np.arange(2),targets[:,None]]
                    finite=np.zeros_like(true);np.log(true,out=finite,where=true>0);finite=-finite
                    square=np.sum((pred-np.eye(8)[targets,None,:])**2,axis=2)
                    prefix=f'{field}-{arm}';saved[prefix+'-true-probability']=true;saved[prefix+'-finite-log-loss']=finite
                    saved[prefix+'-infinite-loss']=true==0;saved[prefix+'-squared-error']=square
                    for ai,actual in enumerate(ACCURACIES):
                        probs=np.where(np.array([q[fi] for q in qs])[:,None]==np.arange(2),actual,1-actual)
                        saved[f'{field}-actual-{actual:g}-reply-probabilities']=probs
                        for weighting,w in [('native',mass),('equal-query',np.full(n,1/n))]:
                            score=expected_scores(pred,probs,targets,w)
                            excluded=float(np.sum(w[:,None]*probs*(~saved['compatible'][qgroup,fi,:,ai])))
                            bound=float(np.sum(w[:,None]*probs*np.log(saved['normalizer'][qgroup,fi])))
                            for cost in COSTS:
                                count=int(arm!='none');cells.append(dict(lineage=lin,rule=rule,model=model,field=field,arm=arm,actual_accuracy=actual,
                                    weighting=weighting,cost=cost,request_rate=count,queries=n,groups=len(ids),
                                    expected_log_normalizer=bound,actual_candidate_excluded_mass=excluded,net_finite_loss=score['finite_loss_contribution']+cost*count,**score))
            np.savez_compressed(root/'forecasts'/f'{lin}-{rule}-{model}_points.npz',**saved)
    if seen!=set(lookup):raise ValueError('law coverage')
    write(root/'evaluator/FALLBACKS.json',fallbacks)
    write(root/'TIMING.jsonl',dict(cpu_seconds=time.process_time()-start,fits=0,scope='fixed forecast rules; no posterior over channels or fitted decision'))
    return dict(controls=checks,cells=cells,queries=cfg['queries'],reader_packets=len(packets),coding_inequalities=bounds,fits=0,
                scope='finite supplied-law decision comparison; coding bound relative to compatible forecasts only; no minimax,calibration,learned-access or historical-process claim')
