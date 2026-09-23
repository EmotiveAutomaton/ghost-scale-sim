"""Cost-sensitive optional requests on frozen mechanically available fields."""
from itertools import product
import shutil
import time
import numpy as np
from ..v18_3.io import read, write, file_digest
from . import input_privilege as P

POLICIES = ('none','forced','optional','matched-rate')
COSTS = (0., .02, .1)
MODELS = ('uniform-legal','native-law')


def selection(gain, cost):
    gain=np.asarray(gain,float)
    if not np.isfinite(gain).all() or cost<0: raise ValueError('request benefit')
    return (gain>cost).astype(float)


def scores(before, requested, targets, weights, probability, cost):
    """Expected policy score, never a score of the mixed forecast."""
    probability=np.asarray(probability,float);weights=np.asarray(weights,float)
    if probability.shape!=weights.shape or np.any(probability<0) or np.any(probability>1):raise ValueError('request probability')
    parts=[P.proper(before,targets,weights*(1-probability)),P.proper(requested,targets,weights*probability)]
    metrics={k:float(sum(r[k] for r in parts)) for k in parts[0]}
    rate=float(np.dot(weights,probability))
    ambiguous=float(np.dot(weights,(1-probability)*((before>0).sum(1)>1)+probability*((requested>0).sum(1)>1)))
    return dict(request_rate=rate,residual_ambiguous_mass=ambiguous,net_finite_loss=metrics['finite_loss_contribution']+cost*rate,**metrics)


def controls():
    before=np.array([[.5,.5,0,0,0,0,0,0]])
    after=np.array([[1.,0,0,0,0,0,0,0]])
    value=scores(before,after,np.array([0]),np.array([1.]),np.array([.5]),.1)
    return {'live:informative_request':bool(selection([np.log(2)],.1)[0]),
            'placebo:constant_endpoint_declines':not bool(selection([0.],0)[0]),
            'positive:price_tie_declines':not bool(selection([.1],.1)[0]),
            'positive:expected_policy_score':bool(abs(value['finite_loss_contribution']-.5*np.log(2))<1e-12 and abs(value['net_finite_loss']-.5*np.log(2)-.05)<1e-12)}


def run(root,plan,pulse):
    cfg=plan['design'];start=time.process_time();base=root/'inputs';checks=controls()
    if not all(checks.values()) or cfg['policies']!=list(POLICIES) or cfg['costs']!=list(COSTS) or cfg['models']!=list(MODELS):raise ValueError('admission')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    laws=read(base/'DISCLOSURE_LAWS.json');lookup={}
    for row in laws:
        key=(row['lineage'],row['rule'],row['model'],row['reader_id'])
        if key in lookup:raise ValueError('duplicate law')
        lookup[key]=row
    groups=read(base/'MEMBERSHIP.json')['omit-both'];cells=[];decisions=[]
    (root/'reader').mkdir();(root/'evaluator').mkdir()
    for path in sorted((base/'reader').glob('*.json')):
        packet=read(path)
        if not set(packet)<={'initial','operations','requested_field','disclosed_value'}:raise ValueError('reader input')
        shutil.copyfile(path,root/'reader'/path.name)
    write(root/'CONTROLS.json',checks)
    for lineage,rule,model in product(cfg['lineages'],cfg['rules'],MODELS):
        pulse(phase='optional-disclosure',lineage=lineage,rule=rule,model=model)
        with np.load(base/'forecasts'/f'{lineage}-{rule}-{model}_points.npz',allow_pickle=False) as saved:
            before=saved['none'];after=saved['entropy-choice'];targets=saved['targets'];mass=saved['mass'];choices=saved['choices'];qs=saved['queries']
            if len(qs)!=cfg['queries']:raise ValueError('queries')
            gains=np.zeros(len(qs))
            for key,group in groups.items():
                law=lookup[(lineage,rule,model,key)];field=law['choice']
                gain=law['prior_entropy']-law['expected_entropy'][field]
                if any(choices[i]!=int(field=='belief') for i in group['indices']):raise ValueError('field binding')
                gains[group['indices']]=gain
            for weighting,w in (('native',mass),('equal-query',np.full(len(qs),1/len(qs)))):
                for cost in COSTS:
                    optional=selection(gains,cost);rate=float(np.dot(w,optional))
                    probabilities={'none':np.zeros(len(qs)),'forced':np.ones(len(qs)),'optional':optional,'matched-rate':np.full(len(qs),rate)}
                    decisions.append(dict(lineage=lineage,rule=rule,model=model,weighting=weighting,cost=cost,
                        gains=gains.tolist(),fields=choices.tolist(),optional=optional.tolist(),matched_request_probability=rate))
                    for policy,probability in probabilities.items():
                        cells.append(dict(lineage=lineage,rule=rule,model=model,weighting=weighting,cost=cost,policy=policy,
                            queries=len(qs),groups=len(groups),skill_request_rate=float(np.dot(w,probability*(choices==0))),
                            belief_request_rate=float(np.dot(w,probability*(choices==1))),**scores(before,after,targets,w,probability,cost)))
    write(root/'evaluator/DECISIONS.json',decisions)
    write(root/'TIMING.jsonl',dict(cpu_seconds=time.process_time()-start,fits=0,scope='optional request selection and expected proper scoring on frozen conditional laws'))
    return dict(controls=checks,cells=cells,queries=cfg['queries'],decisions=len(decisions),fits=0,
        scope='supplied-law optional metadata request; matched-rate random request uses the same visible-group field choice; no learned law, new draw or historical-process claim')
