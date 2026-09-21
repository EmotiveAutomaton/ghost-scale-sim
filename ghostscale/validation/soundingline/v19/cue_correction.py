"""Finite report likelihoods with explicit source replacement and retraction."""
from collections import defaultdict
from itertools import product
import gzip
import time
import numpy as np
from ..v18_3.io import canonical, digest, read, write, file_digest
from . import local_world as L
from .local_alternatives import identity, metrics

ARMS=('base','literal-rearrangement','equal-length-reread','irrelevant','unknown',
      'truthful','wrong','wrong-retracted','wrong-replaced','independent-conflict',
      'duplicate-aware','duplicate-naive')

def slots(arm,truth):
    def report(value,source='source-a'):return dict(operation='report',source=source,assertion=bool(value))
    empty=dict(operation='noop')
    if arm in ('base','literal-rearrangement','equal-length-reread'):return [empty,empty]
    if arm in ('irrelevant','unknown'):return [dict(operation=arm,source='source-a'),empty]
    if arm=='truthful':return [report(truth),empty]
    if arm=='wrong':return [report(not truth),empty]
    if arm=='wrong-retracted':return [report(not truth),dict(operation='retract',source='source-a')]
    if arm=='wrong-replaced':return [report(not truth),report(truth)]
    if arm=='independent-conflict':return [report(not truth),report(truth,'source-b')]
    if arm in ('duplicate-aware','duplicate-naive'):return [report(not truth),report(not truth)]
    raise ValueError('unadmitted arm')

def accepted_reports(reports,independent_duplicates=False):
    active={}
    for i,r in enumerate(reports):
        op=r.get('operation')
        if op=='noop':
            if set(r)!={'operation'}:raise ValueError('unexpected noop fields')
            continue
        if op not in ('report','retract','irrelevant','unknown'):raise ValueError('unknown operation')
        if not isinstance(r.get('source'),str) or not r['source']:raise ValueError('source required')
        if op=='report':
            if set(r)!={'operation','source','assertion'} or type(r['assertion']) is not bool:raise ValueError('binary assertion required')
            key=(r['source'],i) if independent_duplicates else r['source'];active[key]=r['assertion']
        else:
            if set(r)!={'operation','source'}:raise ValueError('unexpected report fields')
            if op=='retract':
                for key in list(active):
                    if key==r['source'] or isinstance(key,tuple) and key[0]==r['source']:del active[key]
    return list(active.values())

def posterior(base,relations,reports,reliability,independent_duplicates=False):
    if reliability not in (.5,.9):raise ValueError('unadmitted reliability')
    accepted=accepted_reports(reports,independent_duplicates)
    base=np.asarray(base,dtype=float);relations=np.asarray(relations,dtype=bool)
    if base.ndim!=1 or relations.shape!=base.shape or np.any(base<0) or not np.isfinite(base).all():raise ValueError('invalid prior')
    if not len(base):return base
    if not np.isclose(base.sum(),1):raise ValueError('unnormalized prior')
    if not accepted or reliability==.5:return base.copy()
    weights=base.copy()
    for assertion in accepted:weights*=np.where(relations==assertion,reliability,1-reliability)
    return weights/weights.sum()

def controls():
    base=np.array([.5,.5]);relation=np.array([False,True]);truth=posterior(base,relation,slots('truthful',True),.9)
    wrong=posterior(base,relation,slots('wrong',True),.9)
    return {'live:truthful_relation':bool(np.allclose(truth,[.1,.9])),
        'positive:wrong_relation':bool(np.allclose(wrong,[.9,.1])),
        'placebo:uninformative_report':bool(np.array_equal(posterior(base,relation,slots('truthful',True),.5),base)),
        'positive:retraction':bool(np.array_equal(posterior(base,relation,slots('wrong-retracted',True),.9),base)),
        'positive:duplicate_provenance':bool(np.array_equal(posterior(base,relation,slots('duplicate-aware',True),.9),wrong)),
        'positive:independent_conflict':bool(np.allclose(posterior(base,relation,slots('independent-conflict',True),.9),base)),
        'positive:replacement':bool(np.array_equal(posterior(base,relation,slots('wrong-replaced',True),.9),truth)),
        'positive:outside_support':len(posterior([],[],slots('truthful',True),.9))==0}

def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('cue controls failed')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('frozen cue input changed')
    cases=read(root/'inputs/PUBLIC_PACKET.json')['cases'];truths={r['case_id']:r for r in read(root/'inputs/EVALUATOR_ONLY.json')['cases']}
    rows=[];posteriors=[];public={};evaluation=[];timing=[];outside=[]
    for lineage in cfg['lineages']:
        before=time.process_time();pulse(phase='retained-hypotheses',lineage=lineage)
        records=__import__('json').loads(gzip.decompress((root/'inputs/evaluator'/f'lineage-{lineage}_points.json.gz').read_bytes()))
        groups=defaultdict(list)
        for r in records:groups[digest(L.project(r,'E1'))].append(r)
        for case in cases:
            if int(case['case_id'].split('-')[0])!=lineage:continue
            seed,index=map(int,case['case_id'].split('-')[1:3]);packet=case['packet'];L.validate_public(packet)
            assert digest(packet)==case['input_sha256'];group=groups[digest(packet)];tr=truths[case['case_id']]
            target=(tuple(tr['true_goals']),tuple(tr['true_operations']));actual=records[tr['trajectory_index']]
            if identity(actual)!=target or L.project(actual,'E1')!=packet:raise ValueError('retained truth mapping failed')
            relation=target[0][0]==target[0][2];legal=set(identity(r) for r in group)
            for prior in cfg['priors']:
                mass=defaultdict(float)
                for r in group:mass[identity(r)]+=r['probability']*(3 if prior=='self-like' and r['maker'][0]==0 else 1)
                support=sorted(mass);base=np.array([mass[k] for k in support]);base/=base.sum();relations=np.array([k[0][0]==k[0][2] for k in support])
                for reliability,arm in product(cfg['reliabilities'],ARMS):
                    reports=slots(arm,relation);p=posterior(base,relations,reports,reliability,arm=='duplicate-naive')
                    distribution={k:float(v) for k,v in zip(support,p)};scored=metrics(distribution,target,legal)
                    joint=float(-np.log(p[support.index(target)]));prob=float(p@relations)
                    reader=dict(schema='v19.cue.public.1',base=packet,reports=reports,reliability=reliability,
                        prior=prior,duplicate_interpretation='independent' if arm=='duplicate-naive' else 'source-aware')
                    frame=digest(reader);public[frame]=reader
                    row=dict(lineage=lineage,seed=seed,index=index,prior=prior,reliability=reliability,arm=arm,frame=frame,
                        **scored,relation_probability=prob,relation_squared_error=(prob-float(relation))**2,
                        posterior_tv=float(abs(p-base).sum()/2),entropy=float(-np.sum(p*np.log(p))),support_size=len(p))
                    if not np.isfinite(joint) or abs(row['loss']-joint)>1e-12:raise ValueError('nonfinite or inconsistent score')
                    rows.append(row);posteriors.append(dict(frame=frame,support=[[list(g),list(o)] for g,o in support],prior=base.tolist(),posterior=p.tolist()))
                    evaluation.append(dict(frame=frame,lineage=lineage,seed=seed,index=index,arm=arm,true_relation=relation,true_goals=list(target[0]),true_operations=list(target[1])))
            pulse(phase='cue-case',lineage=lineage,seed=seed,index=index)
        unknown=dict(schema='v19.local.public.1',tier='E0',inputs=dict(artifact=[2,0,0]))
        if not L.infer(unknown,records)['unknown']:raise ValueError('outside-alphabet reference failed')
        for reliability,arm in product(cfg['reliabilities'],ARMS):
            p=posterior([],[],slots(arm,True),reliability,arm=='duplicate-naive');outside.append(dict(lineage=lineage,reliability=reliability,arm=arm,unknown=len(p)==0,scope='outside-alphabet fixture only'))
        timing.append(dict(lineage=lineage,cpu_seconds=time.process_time()-before))
    (root/'raw').mkdir(exist_ok=True)
    for label,data in (('cue',rows),('posterior',posteriors),('evaluation',evaluation)):(root/'raw'/f'{label}_points.json.gz').write_bytes(gzip.compress(canonical(data),mtime=0))
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.cue.export.1',frames=public,role='anonymous supplied reports and base evidence; no truth labels or outcome-selected arm names'))
    write(root/'OUTSIDE_SUPPORT.json',outside);write(root/'TIMING.jsonl',dict(measurements=timing,accounting='included in native charge'))
    checks['positive:all_outside_unknown']=all(r['unknown'] for r in outside)
    by=defaultdict(list)
    for r in rows:by[r['prior'],r['reliability'],r['arm']].append(r)
    cells=[dict(prior=k[0],reliability=k[1],arm=k[2],cases=len(rr),loss=float(np.mean([r['loss'] for r in rr])),relation_squared_error=float(np.mean([r['relation_squared_error'] for r in rr])),coverage=float(np.mean([r['coverage'] for r in rr]))) for k,rr in sorted(by.items())]
    return dict(controls=checks,cells=cells,rows=len(rows),cases=len(cases),anonymous_frames=len(public),fits=0,new_model_settings=0,
        access='supplied exact process family; evaluator-selected relation cues add teacher information',
        scope='exploratory constructed method; miniature — architecture untested; outside-alphabet control does not establish unknown-cause discovery')
