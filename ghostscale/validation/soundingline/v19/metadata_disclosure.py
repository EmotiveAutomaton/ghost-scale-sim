"""Fixed one-field disclosures under explicit supplied conditional laws."""
from itertools import product
import math
import time
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import input_privilege as P

POLICIES=('none','skill','belief','entropy-choice','equal-request','all-fields')
MODELS=('uniform-legal','native-law')
COSTS=(0.,.02,.1)


def distributions(legal, endpoints, weights):
    """Whole visible-group law, never an individual hidden query or endpoint."""
    weights=np.asarray(weights,float);endpoints=np.asarray(endpoints,int)
    if (not legal or len(set(legal))!=len(legal) or weights.shape!=(len(legal),)
        or endpoints.shape!=weights.shape or np.any(endpoints<0) or np.any(endpoints>=8)
        or not np.isfinite(weights).all() or np.any(weights<0)
        or not np.isclose(weights.sum(),1,atol=1e-12,rtol=0)):
        raise ValueError('conditional law')
    before=np.bincount(endpoints,weights=weights,minlength=8)
    tables={};expected={}
    for field,axis in (('skill',0),('belief',1)):
        tables[field]={};ent=0.
        for value in sorted({q[axis] for q in legal}):
            ix=np.array([i for i,q in enumerate(legal) if q[axis]==value]);mass=float(weights[ix].sum())
            # A zero-probability disclosure has an explicit uniform-legal fallback.
            p=np.bincount(endpoints[ix],weights=weights[ix] if mass else None,minlength=8).astype(float)
            p/=mass if mass else len(ix)
            tables[field][value]=p;ent+=mass*P.entropy(p)
        expected[field]=ent
    chosen='skill' if expected['skill']<=expected['belief'] else 'belief'
    return before,tables,expected,chosen


def group_laws(queries,groups,targets,mass,rule):
    index={q:i for i,q in enumerate(queries)};outputs={};rows=[]
    for model in MODELS:
        before=np.zeros((len(queries),8));skill=before.copy();belief=before.copy();choice=before.copy();chosen=np.zeros(len(queries),dtype=np.int8)
        for key,row in groups.items():
            legal=[tuple(q) for q in row['legal_completions']]
            ends=[P.T.oracle(q,rule) for q in legal]
            weights=np.ones(len(legal))/len(legal)
            raw=np.array([mass[index[q]] if q in index else 0. for q in legal]);zero=not bool(raw.sum())
            if model=='native-law' and not zero:weights=raw/raw.sum()
            base,tables,ent,field=distributions(legal,ends,weights)
            for i in row['indices']:
                q=queries[i];before[i]=base;skill[i]=tables['skill'][q[0]];belief[i]=tables['belief'][q[1]]
                chosen[i]=0 if field=='skill' else 1;choice[i]=skill[i] if field=='skill' else belief[i]
            rows.append(dict(reader_id=key,model=model,legal_completions=[list(q) for q in legal],endpoints=ends,conditional_weights=weights.tolist(),
                expected_entropy=ent,choice=field,prior_entropy=P.entropy(base),zero_native_mass=zero,
                zero_mass_fallback='uniform-legal' if model=='native-law' and zero else None))
        outputs[model]=dict(none=before,skill=skill,belief=belief,**{'entropy-choice':choice,'choices':chosen,'all-fields':np.eye(8)[targets]})
    return outputs,rows


def controls():
    legal=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    _,_,ent,choice=distributions(legal,[0,1,0,1],[.25]*4)
    _,_,null,tie=distributions(legal,[2]*4,[.25]*4)
    return {'live:belief_informative':choice=='belief' and ent['belief']==0 and ent['skill']>0,
            'placebo:constant_endpoint_tie':tie=='skill' and null=={'skill':0.,'belief':0.},
            'positive:skill_informative':distributions(legal,[0,0,1,1],[.25]*4)[3]=='skill'}


def run(root,plan,pulse):
    cfg=plan['design'];base=root/'inputs';start=time.process_time();checks=controls()
    if not all(checks.values()) or cfg['policies']!=list(POLICIES) or cfg['models']!=list(MODELS) or cfg['costs']!=list(COSTS):raise ValueError('disclosure admission')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    truth=read(base/'QUERY_TRUTH.json');qs=[tuple(r['query']) for r in truth];index={q:i for i,q in enumerate(qs)}
    if len(qs)!=cfg['queries'] or len(index)!=len(qs):raise ValueError('queries')
    groups=P.membership(qs)['omit-both']
    if groups!=read(base/'MEMBERSHIP.json')['omit-both']:raise ValueError('frozen membership')
    for folder in ('reader','evaluator','forecasts'):(root/folder).mkdir()
    for key,row in groups.items():write(root/'reader'/f'{key}.json',row['reader'])
    # Anonymous legal reply packets expose only the requested field and its value.
    for row in groups.values():
        for q in row['legal_completions']:
            for field,axis in (('skill',0),('belief_error',1)):
                packet=dict(row['reader'],requested_field=field,disclosed_value=q[axis]);path=root/'reader'/f'{digest(packet)}.json'
                if not path.exists():write(path,packet)
    write(root/'CONTROLS.json',checks);cells=[];law_rows=[];paths=0
    for lineage,rule in product(cfg['lineages'],cfg['rules']):
        pulse(phase='metadata-disclosure',lineage=lineage,rule=rule)
        targets=np.array([P.T.oracle(q,rule) for q in qs]);mass=np.zeros(len(qs))
        if any(t['targets'][rule]!=int(v) for t,v in zip(truth,targets,strict=True)):raise ValueError('truth')
        records=P.read_gzip(base/'raw'/f'{lineage}-{rule}_points.json.gz');paths+=len(records)
        for r in records:
            i=index[P.R.query(r)];w=r['probability']
            if not math.isfinite(w) or w<0 or P.R.code(r['final'])!=targets[i]:raise ValueError('native path')
            mass[i]+=w
        if not np.isclose(mass.sum(),1,atol=1e-12,rtol=0):raise ValueError('population')
        predictions,rows=group_laws(qs,groups,targets,mass,rule)
        law_rows.extend(dict(lineage=lineage,rule=rule,**row) for row in rows)
        for model in MODELS:
            pred=predictions[model]
            np.savez_compressed(root/'forecasts'/f'{lineage}-{rule}-{model}_points.npz',queries=qs,targets=targets,mass=mass,**pred)
            for weighting,w in (('native',mass),('equal-query',np.ones(len(qs))/len(qs))):
                for policy in POLICIES:
                    ps=[pred['skill'],pred['belief']] if policy=='equal-request' else [pred[policy]]
                    scores=[P.proper(p,targets,w) for p in ps];metrics={k:float(np.mean([s[k] for s in scores])) for k in scores[0]}
                    ambiguous=float(np.mean([np.dot(w,(p>0).sum(1)>1) for p in ps]))
                    fields=0 if policy=='none' else 2 if policy=='all-fields' else 1
                    skill_rate=(float(np.dot(w,pred['choices']==0)) if policy=='entropy-choice' else .5 if policy=='equal-request' else float(policy in ('skill','all-fields')))
                    belief_rate=(float(np.dot(w,pred['choices']==1)) if policy=='entropy-choice' else .5 if policy=='equal-request' else float(policy in ('belief','all-fields')))
                    for cost in COSTS:
                        cells.append(dict(lineage=lineage,rule=rule,model=model,policy=policy,weighting=weighting,cost=cost,queries=len(qs),groups=len(groups),fields_requested=fields,
                            skill_request_rate=skill_rate,belief_request_rate=belief_rate,residual_ambiguous_mass=ambiguous,net_finite_loss=metrics['finite_loss_contribution']+cost*fields,**metrics))
    write(root/'evaluator/DISCLOSURE_LAWS.json',law_rows)
    write(root/'TIMING.jsonl',dict(cpu_seconds=time.process_time()-start,fits=0,scope='exact conditional entropy policy and disclosure scoring'))
    return dict(controls=checks,cells=cells,queries=len(qs),native_paths=paths,law_rows=len(law_rows),fits=0,
        scope='oracle value of one mechanically available field; expected scores for equal request randomization, not scores of averaged forecasts; supplied law; no learned access or historical process claim')
