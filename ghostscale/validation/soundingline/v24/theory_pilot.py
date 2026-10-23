"""Finite descriptive pilots of the six previously implemented V24 miniatures."""
import math
from pathlib import Path
import numpy as np
from ..v23.common import canonical, digest, read, write
from ..v23.family import rows
from ..v23.pipeline import Stream
from ..v23 import ledger
from . import pilot, needs_drives, setpoints, play_uncertainty, higher_goal_editing, context_convergence, disturbance

MODULES = dict(NEEDS_DRIVES=needs_drives, SETPOINTS=setpoints,
    PLAY_UNCERTAINTY=play_uncertainty, HIGHER_GOAL_EDITING=higher_goal_editing,
    CONTEXT_CONVERGENCE=context_convergence, DISTURBANCE=disturbance)
PREFIXES = (0, 8, 32, 64)
PRIORS = ((.8,.1,.1), (.1,.1,.8))


def configuration(family):
    module=MODULES[family]
    return dict(worlds=16, prefixes=list(PREFIXES), length=64,
        settings=[{}] if family=='CONTEXT_CONVERGENCE' else module.settings(),
        mechanisms=list(module.MECHANISMS), priors=[list(p) for p in PRIORS] if family=='CONTEXT_CONVERGENCE' else None,
        inference='descriptive-whole-world-only', miniature_architecture_untested=True)


def validate_design(plan):
    family=plan['family']
    if (plan['kind']!='pilot' or family not in MODULES or plan['task']!='v24-theory-pilot'
        or plan['configuration']!=configuration(family) or plan['confirmation'] is not False
        or plan['identity']!='v24-'+family.lower().replace('_','-')+'-pilot-1'
        or plan['cpu_cap_seconds']!=3300):
        raise ValueError('unknown finite theory pilot design')


def preflight(plan):
    validate_design(plan)
    controls=MODULES[plan['family']].controls()
    flags=[value for value in controls.values() if isinstance(value,(bool,np.bool_))]
    if not flags or not all(flags):raise ValueError('theory known-answer control failed')
    # Verify a whole maximum-length excluded case, using independently coded likelihoods.
    f=plan['family'];cfg=configuration(f)
    public,truth=generate(f,cfg['settings'][-1],cfg['mechanisms'][0],'v24-engineering-excluded-preflight')
    a=measure(f,public,truth);b=measure(f,public,truth,independent=True)
    close(a,b)
    return dict(passed=True,known_answers=controls,new_scientific_worlds=0,
                independent_likelihood_fixture=True)


def generate(family,setting,mechanism,identity):
    module=MODULES[family]
    if family=='CONTEXT_CONVERGENCE':return module.generate(mechanism,identity,length=64)
    size_key={'NEEDS_DRIVES':'trials','SETPOINTS':'steps','PLAY_UNCERTAINTY':'trials',
              'HIGHER_GOAL_EDITING':'drafts','DISTURBANCE':'steps'}[family]
    return module.generate(setting,mechanism,identity,**{size_key:64})


def softmax(values):
    maximum=max(values);weights=[math.exp(v-maximum) for v in values]
    total=math.fsum(weights)
    return [v/total for v in weights]


def independent_probability(family,mechanism,row):
    """Scalar reconstruction: no generator/reader probability helpers."""
    if family=='NEEDS_DRIVES':
        if not row['informative'] or (mechanism.startswith('need_') and row['satiated']==mechanism[-1]):return [.5,.5]
        if mechanism.startswith('need_'):return softmax([4*row['consequences'][str(i)][mechanism[-1]] for i in range(2)])
        if mechanism=='intrinsic_action':return [.9,.1]
        return [.85,.15] if row['cue']==0 else [.15,.85]
    if family=='SETPOINTS':
        if mechanism=='state_maintenance':
            d=max(0.,row['setpoint']-row['state']);u=[0.,d*row['effects'][1],d*row['effects'][2]]
        elif mechanism=='intrinsic_act':u=[0.,1.,1. if row['alternative'] else -1.]
        else:u=[0.,1. if row['cue']==1 else -.5,-.5]
        return softmax([4*x for x in u])
    if family=='PLAY_UNCERTAINTY':
        urgent=4. if row['urgent_need'] else 0.;info=float(row['informative_feedback'])
        if mechanism=='own_information':u=[0.,3*row['own_uncertainty']*info-urgent,-urgent]
        elif mechanism=='partner_information':u=[0.,-urgent,3*row['partner_uncertainty']*info-urgent]
        elif mechanism=='pragmatic':u=[2. if row['urgent_need'] else .5,-1.,-1.]
        else:u=[0.,.6-urgent,.6-urgent]
        return softmax([2*x for x in u])
    if family=='HIGHER_GOAL_EDITING':
        if mechanism=='higher_goal':p=.9 if row['conflict'] and row['higher_goal_observable'] and row['goal_violation'] else .1
        elif mechanism=='style_only':p=.85 if row['style_violation'] else .1
        elif mechanism=='generator_only':p=.1
        else:p=.5
        return [1-p,p]
    if family=='DISTURBANCE':
        actions=(-1.,0.,1.)
        if mechanism=='closed_loop':loss=[abs(row['state']+row['disturbance']+float(row['controllable'])*a-row['target']) for a in actions]
        elif mechanism=='open_loop':loss=[abs(a-row['planned_action']) for a in actions]
        else:loss=[abs(a-row['cue_action']) for a in actions]
        return softmax([-4*x for x in loss])
    raise ValueError('unknown family')


def scalar_posterior(family,public):
    module=MODULES[family];mass=[]
    for mechanism in module.MECHANISMS:
        loglike=[]
        for row in public['history']:
            choice=int(row['edited']) if family=='HIGHER_GOAL_EDITING' else int(row['action'])+(1 if family=='DISTURBANCE' else 0)
            loglike.append(math.log(independent_probability(family,mechanism,row)[choice]))
        mass.append(math.fsum(loglike))
    return softmax(mass)


def context_scalar(public,prior,support):
    mass=[];probabilities=dict(a=.8,b=.2,c=.65)
    for weight,mechanism in zip(prior,support,strict=True):
        terms=[math.log(weight)]
        for index,value in enumerate(public['observations']):
            p=.5 if index<2 else probabilities[mechanism]
            terms.append(math.log(p if value else 1-p))
        mass.append(math.fsum(terms))
    posterior=softmax(mass);index=len(public['observations'])
    prediction=math.fsum(w*(.5 if index<2 else probabilities[m]) for w,m in zip(posterior,support,strict=True))
    return dict(posterior=posterior,predictive_one=prediction)


def measure(family,public,truth,independent=False):
    module=MODULES[family];records=[]
    for length in PREFIXES:
        if family=='CONTEXT_CONVERGENCE':
            cut=dict(public,observations=public['observations'][:length])
            method=context_scalar if independent else context_convergence.read
            first=method(cut,PRIORS[0],module.MECHANISMS);second=method(cut,PRIORS[1],module.MECHANISMS)
            omitted=method(cut,(.5,.5),('a','b'))
            truth_p=.5 if length<2 else dict(a=.8,b=.2,c=.65)[truth['mechanism']]
            records.append(dict(prefix=length,posterior=first['posterior'],
                predictive_disagreement=abs(first['predictive_one']-second['predictive_one']),
                full_predictive_absolute_error=abs(first['predictive_one']-truth_p),
                omitted_predictive_absolute_error=abs(omitted['predictive_one']-truth_p)))
        else:
            cut=dict(public,history=public['history'][:length])
            posterior=scalar_posterior(family,cut) if independent else module.posterior(cut).tolist()
            record=dict(prefix=length,posterior=posterior)
            if family=='PLAY_UNCERTAINTY':
                if independent:
                    evidence=[r['feedback'] for r in cut['history'] if r['action']==1 and r['feedback'] is not None]
                    solution=softmax([math.fsum(math.log(.9 if value==answer else .1) for value in evidence) for answer in (0,1)])
                else:solution=play_uncertainty.solution_posterior(cut).tolist()
                record['solution_truth_probability']=solution[truth['solution']]
            records.append(record)
        p=records[-1]['posterior'];true_index=module.MECHANISMS.index(truth['mechanism'])
        if any(not math.isfinite(v) or v<0 for v in p) or abs(math.fsum(p)-1)>1e-10:raise ValueError('invalid posterior')
        if p[true_index]<=0:raise ValueError('true support lost; do not floor')
        tied=[i for i,v in enumerate(p) if abs(v-max(p))<=1e-12]
        records[-1].update(log_gain_over_uniform=math.log(p[true_index]*len(p)),
            tie_shared_accuracy=(1/len(tied) if true_index in tied else 0.),
            half_brier=.5*math.fsum((v-float(i==true_index))**2 for i,v in enumerate(p)))
    return records


def close(a,b):
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('record keys differ')
        for key in a:close(a[key],b[key])
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('record lengths differ')
        for x,y in zip(a,b,strict=True):close(x,y)
    elif isinstance(a,(float,int)):
        if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('independent numerical reconstruction differs')
    elif a!=b:raise ValueError('record identity differs')


def summarize(records,plan):
    cfg=plan['configuration'];expected={(w,s,m) for w in range(cfg['worlds']) for s in range(len(cfg['settings'])) for m in cfg['mechanisms']}
    keys=[tuple(r['identity']) for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('incomplete theory grid')
    cells=[]
    for s in range(len(cfg['settings'])):
        for m in cfg['mechanisms']:
            selected=[r for r in records if r['identity'][1:]==[s,m]]
            for index,length in enumerate(PREFIXES):
                metrics=[r['metrics'][index] for r in selected]
                cells.append(dict(setting=s,mechanism=m,prefix=length,worlds=len(metrics),
                    **{key:math.fsum(r[key] for r in metrics)/len(metrics) for key in metrics[0] if key not in ('prefix','posterior')}))
    world_means=[]
    for world in range(cfg['worlds']):
        selected=[r['metrics'][-1]['log_gain_over_uniform'] for r in records if r['identity'][0]==world]
        world_means.append(math.fsum(selected)/len(selected))
    return dict(state='DESCRIPTIVE_THEORY_PILOT',family=plan['family'],cases=len(records),
        independent_worlds=cfg['worlds'],whole_world_log_gains=world_means,cells=cells,
        confirmation=False,model_limit='Miniature: architecture untested; stipulated mechanisms and supplied likelihoods. No human or engine-admission inference.')


def execute(root,plan,pulse):
    validate_design(plan);root=Path(root);root.mkdir(exist_ok=False)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    f=plan['family'];cfg=plan['configuration'];scores=[]
    streams={name:Stream(root/(name+'_points.jsonl.gz')) for name in ('public','truth','evidence','ledger')}
    try:
        for w in range(cfg['worlds']):
            for s,setting in enumerate(cfg['settings']):
                for mechanism in cfg['mechanisms']:
                    identity=[w,s,mechanism]
                    public,truth=generate(f,setting,mechanism,[plan['identity'],*identity])
                    metrics=measure(f,public,truth);record=dict(identity=identity,metrics=metrics);scores.append(record)
                    bundle=pilot.emit(schema,[plan['identity'],*identity],public)
                    for key,value in [('public',public),('truth',truth),('evidence',metrics),('ledger',bundle)]:
                        streams[key].write(dict(identity=identity,**{key:value}))
            pulse(world=w,cases=len(scores))
    finally:
        for stream in streams.values():stream.close()
    write(root/'SUMMARY.json',summarize(scores,plan))
    write(root/'RAW_INDEX.json',{p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in root.glob('*_points.jsonl.gz')})


def reconstruct(root,plan,pulse):
    validate_design(plan);root=Path(root);index=read(root/'RAW_INDEX.json');scores=[]
    if set(index)!={k+'_points.jsonl.gz' for k in ('public','truth','evidence','ledger')}:raise ValueError('missing raw stream')
    for name,item in index.items():
        if digest(root/name)!=item['sha256'] or (root/name).stat().st_size!=item['bytes']:raise ValueError('raw corruption')
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin']);cfg=plan['configuration']
    for pub,truth,ev,transport in zip(*(rows(root/(k+'_points.jsonl.gz')) for k in ('public','truth','evidence','ledger')),strict=True):
        ident=pub['identity']
        if any(r['identity']!=ident for r in (truth,ev,transport)):raise ValueError('stream alignment')
        w,s,m=ident
        expected_public,expected_truth=generate(plan['family'],cfg['settings'][s],m,[plan['identity'],*ident])
        if canonical(expected_public)!=canonical(pub['public']) or expected_truth!=truth['truth']:raise ValueError('planted inputs differ')
        ledger.validate(transport['ledger'],schema)
        if canonical(pilot.emit(schema,[plan['identity'],*ident],pub['public']))!=canonical(transport['ledger']):raise ValueError('ledger mismatch')
        metrics=measure(plan['family'],pub['public'],truth['truth'],independent=True)
        close(metrics,ev['evidence']);scores.append(dict(identity=ident,metrics=metrics))
        if len(scores)%32==0:pulse(cases=len(scores))
    close(summarize(scores,plan),read(root/'SUMMARY.json'))
    return dict(state='THEORY_PILOT_INDEPENDENT_REVIEW_PASSED',cases=len(scores),worlds=cfg['worlds'],full_grid=True,
                canonical_transport=True,independent_scalar_likelihoods=True,inputs_regenerated=True,confirmation=False)
