"""Alphabet-mass control for goals conditional on complete witnessed operations."""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read,write,canonical,digest,file_digest
from . import joint_support as S
from .joint_uncertainty import distribution
from .witnessed_goal_factorization import factorize,N,ARMS,FIELDS
from .temporal_mass_control import mass_match,controls


def aggregate(rows,cfg):
    # The common paired estimator is applied to a declared bijection of arm names.
    from .temporal_mass_control import aggregate as common
    mapped=[]
    for r in rows:
        arm=r['arm']
        if arm.endswith('-restricted'):arm=arm[:-len('-restricted')]
        elif arm.endswith('-goal-product'):arm=arm[:-len('-goal-product')]+'-product'
        mapped.append(dict(r,arm=arm))
    result=common(mapped,cfg)
    for rows in result.values():
        for r in rows:
            for key in ('arm','baseline'):
                if key not in r:continue
                if r[key].endswith('-product'):r[key]=r[key][:-len('-product')]+'-goal-product'
                elif r[key] in ARMS:r[key]+='-restricted'
    return result


def run(root,plan,pulse):
    cfg=plan['design'];base=root/'inputs';checks=controls()
    write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('known-answer controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    parent=read(base/'witnessed/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if parent[k]!=cfg[k]:raise ValueError('parent population changed')
    if cfg['tiers']!=['E2-full']:raise ValueError('complete witnesses required')
    packets=read(base/'reader/PACKETS.json');refs=read(base/'evaluator/REFERENCES.json')
    packets=dict(packets,packets={k:p for k,p in packets['packets'].items() if p['tier']=='E2-full'})
    refs=[r for r in refs if r['tier']=='E2-full']
    for key,p in packets['packets'].items():
        S.validate(p)
        if key!=digest(p):raise ValueError('packet identity')
    write(root/'reader/PACKETS.json',packets);write(root/'evaluator/REFERENCES.json',refs)
    old=json.loads(gzip.decompress((base/'witnessed/witnessed_goal_points.json.gz').read_bytes()))
    idx={tuple(r[k] for k in FIELDS):r for r in old}
    expected=set(product(cfg['tiers'],cfg['budgets'],tuple(a+s for a in ARMS for s in ('','-restricted','-goal-product')),cfg['development_lineages'],cfg['training_draws'],cfg['fit_seeds']))
    if len(idx)!=len(old) or set(idx)!=expected:raise ValueError('parent complete roster')
    native=read(base/'witnessed/NATIVE_SCORES.json');native_idx={(r['tier'],r['lineage'],r['arm']):r for r in native}
    if len(native_idx)!=len(native) or set(native_idx)!=set(product(cfg['tiers'],cfg['development_lineages'],('native-joint','native-restricted','native-goal-product'))):raise ValueError('native roster')
    write(root/'evaluator/INHERITED_NATIVE_SCORES.json',native)
    rows=[];timing=[];reproduced=0;max_error=0.;frames_scored=0;(root/'masses').mkdir()
    tier='E2-full';keys=sorted(packets['packets']);opnames=('edit-claim','repair-evidence','replace-presentation','accept-tool','inspect','undo')
    operations=[]
    for key in keys:
        events=packets['packets'][key]['inputs']['observations']
        if len(events)!=3 or [e['step'] for e in events]!=[0,1,2]:raise ValueError('witness order')
        operations.append(sum(opnames.index(e['operation'])*6**(2-i) for i,e in enumerate(events)))
    operations=np.array(operations)
    ref=S.reference_arrays(refs,tier,keys,cfg['development_lineages'],np.ones((len(keys),N),bool))
    if set(r['lineage'] for r in refs)!=set(cfg['development_lineages']):raise ValueError('native lineage roster')
    for draw,seed,budget,arm in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS):
        pulse(phase='conditional-goal-alphabet-mass',tier=tier,draw=draw,seed=seed,budget=budget,arm=arm)
        tick=time.process_time();stem=f'{draw}-{tier}-{seed}-{budget}-{arm}'
        if read(base/'forecasts'/(stem+'-frames.json'))!=keys:raise ValueError('forecast frame roster')
        with np.load(base/'forecasts'/(stem+'.npz'),allow_pickle=False) as data:
            if set(data.files)!={'probabilities','alphabet'}:raise ValueError('forecast fields')
            alphabet=data['alphabet'];q=distribution(data['probabilities'],alphabet,budget)
        if len(q)!=len(keys):raise ValueError('forecast length')
        restricted,fact,marg,z=factorize(q,operations);matched,before,after=mass_match(restricted,fact,alphabet)
        np.savez_compressed(root/'masses'/(stem+'.npz'),goals=marg,normalizer=z,original=before,product=after,operations=operations)
        scores={'-restricted':S.scores(restricted,alphabet,ref),'-mass-matched':S.scores(matched,alphabet,ref),'-goal-product':S.scores(fact,alphabet,ref)}
        for li,lineage in enumerate(cfg['development_lineages']):
            native_loss=native_idx[tier,lineage,'native-joint']['loss']
            for suffix in ('-restricted','-goal-product'):
                prior=idx[tier,budget,arm+suffix,lineage,draw,seed]
                for k,v in scores[suffix].items():
                    error=abs(float(v[li])-prior[k]);max_error=max(max_error,error)
                    if error>1e-10:raise ValueError('witnessed score reproduction')
                if abs(prior['excess_loss']-(float(scores[suffix]['loss'][li])-native_loss))>1e-10:raise ValueError('native score binding')
                reproduced+=1
            for suffix,values in scores.items():
                excess=float(values['loss'][li])-native_loss
                if excess<-1e-10:raise ValueError('negative excess')
                rows.append(dict(tier=tier,budget=budget,arm=arm+suffix,lineage=lineage,draw=draw,seed=seed,frames=len(keys),**{k:float(v[li]) for k,v in values.items()},excess_loss=excess))
        frames_scored+=len(keys);timing.append(dict(tier=tier,draw=draw,seed=seed,budget=budget,arm=arm,cpu_seconds=time.process_time()-tick))
    if reproduced!=2*len(old)//3:raise ValueError('parent coverage')
    (root/'goal_mass_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'PARENT_REPRODUCTION.json',dict(passed=True,witnessed_cells=reproduced,max_error=max_error,metrics='all accepted restricted and goal-product scores; native scores inherited'))
    write(root/'TIMING.jsonl',dict(measurements=timing));checks['positive:complete_parent_reproduction']=True
    return dict(controls=checks,rows=len(rows),witnessed_cells_reproduced=reproduced,parent_max_error=max_error,
        frame_forecasts=frames_scored,packets=len(packets['packets']),inherited_native_rows=len(native),**aggregate(rows,cfg),fits=0,
        scope='conditional-goal training-alphabet mass diagnostic; public witnessed operations fixed; native scores inherited evaluator rulers; no fit,new sample or correspondence claim')
