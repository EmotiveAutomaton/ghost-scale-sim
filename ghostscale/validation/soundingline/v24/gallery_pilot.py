"""Constructed fixed-gallery reference calibration; no real-person inputs."""
import math
from pathlib import Path
import numpy as np
from ..v23.common import canonical, digest, read, write, rng
from ..v23.family import rows
from ..v23.pipeline import Stream
from ..v23 import ledger
from . import pilot
from .theory_pilot import close


def settings():
    return [dict(writers=n,features=d,condition=condition) for n in (9,45) for d in (1,17)
            for condition in ('independent_null','personal_signal','group_context_only','same_sample_overlap')]


def configuration():
    return dict(worlds=32,settings=settings(),groups=3,personal_sd=2.,context_sd=4.,noise_sd=1.,
                null_reference='all cyclic gallery-label shifts; fixed arrays and score',
                inference='descriptive calibration; not Sounding Line raw-data reproduction')


def validate_design(plan):
    if (plan['kind']!='pilot' or plan['family']!='PROMOTIONS' or plan['task']!='v24-gallery-pilot'
            or plan['identity']!='v24-gallery-controls-pilot-1' or plan['configuration']!=configuration()
            or plan['cpu_cap_seconds']!=1800 or plan['confirmation'] is not False):
        raise ValueError('unknown gallery-control pilot')


def generate(setting,identity):
    n=setting['writers'];d=setting['features'];condition=setting['condition']
    random=rng('v24-gallery-controls',identity,setting)
    # All arrays and labels are constructed. Stable nuisance groups are not people.
    group=np.arange(n)%3;personal=random.normal(size=(n,d))*2
    context=random.normal(size=(3,d))*4
    a=random.normal(size=(n,d));b=random.normal(size=(n,d))
    if condition=='personal_signal':a+=personal;b+=personal
    elif condition=='group_context_only':a+=context[group];b+=context[group]
    elif condition=='same_sample_overlap':b=a.copy()
    return dict(setting=setting,a=a.tolist(),b=b.tolist(),groups=group.tolist())


def weights(public,independent=False):
    a=np.asarray(public['a']);b=np.asarray(public['b']);n,d=b.shape
    if a.shape!=b.shape or n!=public['setting']['writers'] or d!=public['setting']['features']:
        raise ValueError('gallery shape changed')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite gallery')
    if independent:
        means=[math.fsum(float(b[i,j]) for i in range(n))/n for j in range(d)]
        variance=[math.fsum((float(b[i,j])-means[j])**2 for i in range(n))/n for j in range(d)]
        distances=[[math.fsum((float(a[i,j])-float(b[k,j]))**2/variance[j]
                       for j in range(d) if variance[j]>1e-24) for k in range(n)] for i in range(n)]
        result=[]
        for row in distances:
            best=min(row);chosen=[i for i,v in enumerate(row) if abs(v-best)<=1e-12]
            result.append([1/len(chosen) if i in chosen else 0. for i in range(n)])
        return np.asarray(result)
    variance=np.var(b,axis=0);used=variance>1e-24
    dist=np.sum((a[:,None,used]-b[None,:,used])**2/variance[used],axis=-1)
    tied=np.abs(dist-dist.min(axis=1,keepdims=True))<=1e-12
    return tied/tied.sum(axis=1,keepdims=True)


def metrics(public,independent=False):
    w=weights(public,independent);n=len(w);group=public['groups'];chance=1/n
    if independent:
        accuracy=math.fsum(float(w[i,i]) for i in range(n))/n
        all_reference=math.fsum(float(w[i,j]) for i in range(n) for j in range(n))/(n*n)
        excluded_reference=math.fsum(float(w[i,j]) for i in range(n) for j in range(n) if i!=j)/(n*(n-1))
        grouped=math.fsum(math.fsum(float(w[i,j]) for j in range(n) if group[j]==group[i])/
                          sum(g==group[i] for g in group) for i in range(n))/n
    else:
        accuracy=float(np.trace(w)/n)
        shifts=[float(w[np.arange(n),(np.arange(n)+shift)%n].mean()) for shift in range(n)]
        all_reference=float(np.mean(shifts));excluded_reference=float(np.mean(shifts[1:]))
        grouped=float(np.mean([w[i,np.asarray(group)==group[i]].mean() for i in range(n)]))
    if abs(all_reference-chance)>1e-12:raise ValueError('uniform gallery relabeling not centered')
    if abs(excluded_reference-(1-accuracy)/(n-1))>1e-12:raise ValueError('derangement identity failed')
    return dict(accuracy=accuracy,chance=chance,uniform_label_reference=all_reference,
        identity_excluding_reference=excluded_reference,within_group_reference=grouped,
        observed_minus_uniform=accuracy-all_reference,within_group_minus_global_chance=grouped-chance,
        identity_excluding_minus_chance=excluded_reference-chance)


def preflight(plan):
    validate_design(plan)
    # Positive, exact null, tie and overlap controls can each fail the instrument.
    base=dict(setting=dict(writers=9,features=1,condition='fixture'),a=[[float(i)] for i in range(9)],
              b=[[float(i)] for i in range(9)],groups=[i%3 for i in range(9)])
    pos=metrics(base);close(pos,metrics(base,True))
    tied=dict(base,a=[[0.] for _ in range(9)],b=[[0.] for _ in range(9)])
    null=metrics(tied);close(null,metrics(tied,True))
    if pos['accuracy']!=1 or pos['identity_excluding_reference']!=0 or abs(null['accuracy']-1/9)>1e-12:
        raise ValueError('gallery known-answer failure')
    return dict(passed=True,exact_null=True,exact_overlap=True,independent_distances=True,
                tie_fractional_credit=True,new_scientific_worlds=0)


def summarize(records,plan):
    cfg=plan['configuration'];keys=[tuple(r['identity']) for r in records]
    expected={(w,s) for w in range(cfg['worlds']) for s in range(len(cfg['settings']))}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('incomplete gallery grid')
    cells=[]
    for s,setting in enumerate(cfg['settings']):
        values=[r['metrics'] for r in records if r['identity'][1]==s]
        cells.append(dict(setting=setting,independent_worlds=len(values),
            means={key:math.fsum(v[key] for v in values)/len(values) for key in values[0]},
            world_accuracy=[v['accuracy'] for v in values]))
    return dict(state='DESCRIPTIVE_GALLERY_CONTROL_PILOT',cases=len(records),worlds=cfg['worlds'],cells=cells,
        exact_reference_centering=True,confirmation=False,
        limitations=['Constructed geometry only, not diagnosis or repair of real Sounding Line reference bias.',
                    'Overlap and nuisance conditions deliberately contain no person-specific causal mechanism.',
                    'No significance claims; all settings and worlds retained.'])


def execute(root,plan,pulse):
    validate_design(plan);root=Path(root);root.mkdir(exist_ok=False);scores=[]
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    streams={name:Stream(root/(name+'_points.jsonl.gz')) for name in ('public','evidence','ledger')}
    try:
        for w in range(plan['configuration']['worlds']):
            for s,setting in enumerate(settings()):
                ident=[w,s];public=generate(setting,[plan['identity'],*ident]);score=metrics(public)
                scores.append(dict(identity=ident,metrics=score))
                for key,value in [('public',public),('evidence',score),('ledger',pilot.emit(schema,[plan['identity'],*ident],public))]:
                    streams[key].write(dict(identity=ident,**{key:value}))
            pulse(world=w,cases=len(scores))
    finally:
        for stream in streams.values():stream.close()
    write(root/'SUMMARY.json',summarize(scores,plan))
    write(root/'RAW_INDEX.json',{p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in root.glob('*_points.jsonl.gz')})


def reconstruct(root,plan,pulse):
    validate_design(plan);root=Path(root);index=read(root/'RAW_INDEX.json');scores=[]
    if set(index)!={k+'_points.jsonl.gz' for k in ('public','evidence','ledger')}:raise ValueError('incomplete gallery files')
    for name,item in index.items():
        if digest(root/name)!=item['sha256']:raise ValueError('raw corruption')
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    for pub,ev,trans in zip(*(rows(root/(k+'_points.jsonl.gz')) for k in ('public','evidence','ledger')),strict=True):
        ident=pub['identity']
        if ev['identity']!=ident or trans['identity']!=ident:raise ValueError('stream alignment')
        expected=generate(settings()[ident[1]],[plan['identity'],*ident])
        if canonical(expected)!=canonical(pub['public']):raise ValueError('generated gallery changed')
        ledger.validate(trans['ledger'],schema)
        if canonical(pilot.emit(schema,[plan['identity'],*ident],pub['public']))!=canonical(trans['ledger']):raise ValueError('ledger mismatch')
        score=metrics(pub['public'],True);close(score,ev['evidence']);scores.append(dict(identity=ident,metrics=score))
        if len(scores)%16==0:pulse(cases=len(scores))
    close(summarize(scores,plan),read(root/'SUMMARY.json'))
    return dict(state='GALLERY_INDEPENDENT_REVIEW_PASSED',cases=len(scores),source_regeneration=True,
                independent_scalar_distances=True,independent_assignment_expectations=True,confirmation=False)
