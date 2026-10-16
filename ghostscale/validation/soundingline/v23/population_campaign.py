"""N1 complete population batches, restricted selection and precision follow-up.

Every unit is a whole population. Makers never become independent worlds. The
240-cell map receives two complete batches; only the 36 declared long-history
alias/drift cells can enter the calibrated family selection policy.
"""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
from . import adaptive,checker,checker_population,ledger,makers,population_pipeline as pipe,registry
from .common import Direct,digest,interval,read,write
import time


def base_worlds(v):
    return 8 if v['settings']['makers']*v['settings']['records']>=500000 else 16


def design(batch, active=None, *, precision=False):
    if not isinstance(batch,int) or batch<1 or (not precision and batch>7):raise ValueError('finite population rung required')
    population=registry.variants('N1')
    if active is not None:
        if len(set(active))!=len(active) or not set(active)<=set(v['id'] for v in population):
            raise ValueError('population membership invalid')
        population=[v for v in population if v['id'] in active]
    if not precision and batch<=2 and len(population)!=240:
        raise ValueError('both population burn-ins require the complete map')
    if not precision and batch>2 and any(not registry.selection_eligible(v) for v in population):
        raise ValueError('map cells do not enter family selection')
    if precision and any(registry.selection_eligible(v) for v in population):
        raise ValueError('precision extension is separate from the selected family')
    return [dict(variant=v,worlds=base_worlds(v)*(1 if precision else adaptive.RUNGS[batch-1]//16)) for v in population]


def validate(plan):
    precision=plan['task']=='population-precision'
    expected=design(plan['batch'],plan['variants'],precision=precision)
    if plan['population_design']!=expected:raise ValueError('population allocation changed')
    if not plan.get('fit_directory') or not plan.get('fit_complete_sha256'):
        raise ValueError('verified population fits required')
    if len(plan.get('canary_key',''))!=64:raise ValueError('evaluator-only canary key required')
    return expected


def run(root,plan,pulse):
    allocation=validate(plan);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    fitroot=Path(plan['fit_directory'])
    for name,expected in plan['fit_hashes'].items():
        if digest(fitroot/name)!=expected:raise ValueError('population fit changed')
    fitmeta=read(fitroot/'FITS.json');fitted={name:Direct.load(fitroot/(name+'.npz')) for name in fitmeta}
    phase=['precision' if plan['task']=='population-precision' else 'discovery',plan['batch']]
    summaries=[]
    for item in allocation:
        v=item['variant'];folder=root/v['id'];folder.mkdir();worlds=[]
        for world in range(item['worlds']):
            record=pipe.retained_world(folder/f'world-{world:04d}',v,phase,world,schema,plan['canary_key'],fitted,pulse)
            worlds.append(record);pulse(phase='population-batch',variant=v['id'],world=world+1,worlds=item['worlds'])
        estimates={key:interval([row[key] for row in worlds],[v['id'],phase,key])
                   for key in ('primary_mean','total_risk_ratio','harm_share')}
        result=dict(variant=v,worlds=len(worlds),cases_per_world=v['settings']['makers'],
            primary=estimates['primary_mean'],world_gains=[row['primary_mean'] for row in worlds],
            metrics=estimates,selection_eligible=registry.selection_eligible(v),
            independent_unit='whole population',precision_extension=plan['task']=='population-precision',
            raw_worlds={f'world-{w:04d}':digest(folder/f'world-{w:04d}'/'RAW_INDEX.json') for w in range(len(worlds))})
        write(folder/'SUMMARY.json',result);summaries.append(result)
    write(root/'SUMMARY.json',dict(state='POPULATION_BATCH_CHECKED',family='N1',batch=plan['batch'],
        precision_extension=plan['task']=='population-precision',variants=summaries,fits=fitmeta,
        original_and_replay_required=True,scientific_confirmation=False))


def reconstruct(root,plan):
    """Recompute every scored population from independent scalar score arithmetic."""
    root=Path(root);summary=read(root/'SUMMARY.json');checked={}
    for item in summary['variants']:
        folder=root/item['variant']['id'];gains=[]
        for world in range(item['worlds']):
            unit=folder/f'world-{world:04d}'
            for name,expected in read(unit/'RAW_INDEX.json').items():
                if digest(unit/name)!=expected:raise ValueError('population raw record changed')
            with np.load(unit/'prediction_points.npz') as pred,np.load(unit/'truth_points.npz') as truth,np.load(unit/'score_points.npz') as saved:
                rebuilt={name:checker.scores(pred[name],truth['y']) for name in pred.files}
                for name,row in rebuilt.items():
                    for metric,values in row.items():
                        if not np.allclose(values,saved[name+'__outside__'+metric],atol=1e-11,rtol=0):
                            raise ValueError('population saved scores changed')
                gain=rebuilt['individual-only']['log_loss']-rebuilt['law-marginal']['log_loss']
                if len(gain)!=item['cases_per_world']:raise ValueError('population coverage changed')
                gains.append(math.fsum(map(float,gain))/len(gain))
        if not np.allclose(gains,item['world_gains'],atol=1e-11,rtol=0):raise ValueError('population world aggregation changed')
        checked[item['variant']['id']]=dict(worlds=len(gains),makers_per_world=item['cases_per_world'])
    return dict(state='passed',independent_unit='whole population',variants=checked)


def decide(summaries,policy):
    estimates={v['id']:[] for v in registry.variants('N1') if registry.selection_eligible(v)}
    active=list(estimates);decision=None
    for batch,summary in enumerate(summaries,1):
        if summary['batch']!=batch or summary.get('precision_extension'):raise ValueError('population discovery sequence changed')
        expected=design(batch,None if batch<=2 else active)
        if [row['variant'] for row in summary['variants']]!=[row['variant'] for row in expected]:
            raise ValueError('population batch membership changed')
        for row in summary['variants']:
            if row['variant']['id'] in estimates:estimates[row['variant']['id']].append(row['primary'])
        identities=list(estimates)
        decision=adaptive.decide(batch,list(estimates.values()),[identities.index(i) for i in active],
            list(range(len(identities))),kill=policy['kill_threshold'],mu_scale=policy['prior'][0],tau_scale=policy['prior'][1])
        active=[identities[i] for i in decision['active']]
        if decision['state'] not in ('BURN_IN','CONTINUE') and batch<len(summaries):raise ValueError('population sampled after terminal decision')
    curve=[]
    for v in registry.variants('N1'):
        identity=v['id'];selected=identity in estimates
        gains=[x for summary in summaries for row in summary['variants'] if row['variant']['id']==identity for x in row['world_gains']]
        raw=interval(gains,[identity,'population-curve',len(gains)])
        index=identities.index(identity) if selected else None
        curve.append(dict(variant=v,raw_mean=raw['mean'],lower=raw['lower'],upper=raw['upper'],
            shrunk_mean=decision['posterior']['mean'][index] if selected else None,
            probability_above_margin=decision['posterior']['probability_above_margin'][index] if selected else None,
            selection_eligible=selected,independent_unit='whole population'))
    return dict(decision,active_ids=active,best_id=identities[decision['best']],family='N1',curve=curve)


def precision_candidates(summaries,half_width=.02):
    """Use retained worlds, never a count of makers, to request more map precision."""
    if len(summaries)<2:raise ValueError('two complete map batches precede precision extension')
    values={v['id']:[] for v in registry.variants('N1') if not registry.selection_eligible(v)}
    for summary in summaries:
        for row in summary['variants']:
            if row['variant']['id'] in values:values[row['variant']['id']].extend(row['world_gains'])
    result=[]
    for identity,gains in values.items():
        if not gains:raise ValueError('map coverage missing')
        estimate=interval(gains,[identity,'map-precision',len(gains)])
        if not estimate['finite'] or max(estimate['upper']-estimate['mean'],estimate['mean']-estimate['lower'])>half_width:
            result.append(identity)
    return result


def controls(root,pulse):
    """Retained S-id convergence and record-shuffle diagnostics, fixed in advance.

    Sixteen whole populations of ten makers at R=512. These are apparatus
    diagnostics; observed failure blocks N1 interpretation, never changes the bar.
    """
    root=Path(root);root.mkdir(parents=True,exist_ok=False);rows=[]
    recipes,weights,_=makers.recipe_library()
    for world in range(16):
        v=deepcopy(next(v for v in registry.variants('N1') if v['settings']==dict(makers=10,records=512,concentration=.5,structure='id')))
        p,t=makers.n1_world(v,'production-controls',world,pulse=pulse)
        pred,detail=makers.n1_infer(v,p,pulse)
        checker_population.verify_n1(v,p,t,pred,detail,makers.cultures(.5),pulse)
        expected=np.array([makers.probabilities(menu,weights,recipes,beta)[type_]
            for menu,beta,type_ in zip(p['other'],t['beta'],t['type'],strict=True)])
        risk=lambda forecast:np.sum(-expected*np.log(forecast),axis=1)
        convergence=float(np.mean(risk(pred['individual-only'])-risk(pred['law-marginal'])))
        shuffled=deepcopy(p)
        from .common import rng
        assignment=rng('N1-shuffle-control',world).permutation(len(p['choices']))
        shuffled['menus']=p['menus'][assignment];shuffled['choices']=p['choices'][assignment]
        other,diagnostic=makers.n1_infer(v,shuffled,pulse)
        checker_population.verify_n1(v,shuffled,t,other,diagnostic,makers.cultures(.5),pulse)
        shuffle=float(np.mean(risk(other['population-only'])-risk(other['individual-only'])))
        np.savez_compressed(root/f'world-{world:04d}_points.npz',expected=expected,
            individual=pred['individual-only'],exact=pred['law-marginal'],
            shuffled_individual=other['individual-only'],shuffled_population=other['population-only'],assignment=assignment)
        rows.append(dict(world=world,convergence_gain=convergence,shuffled_personal_gain=shuffle))
        pulse(phase='population-production-controls',world=world+1,worlds=16)
    convergence=interval([row['convergence_gain'] for row in rows],'N1-convergence-control')
    shuffle=interval([row['shuffled_personal_gain'] for row in rows],'N1-shuffle-control')
    result=dict(state='POPULATION_CONTROLS_EVALUATED',worlds=rows,known_answers=pipe.known_answers(),
        convergence=convergence,shuffle=shuffle,convergence_passed=convergence['upper']<=.02,
        shuffle_passed=shuffle['upper']<=.02,margin=.02,
        limitations='Fixed finite apparatus controls; bootstrap intervals are whole-population diagnostics, not a theorem or a human result.')
    write(root/'SUMMARY.json',result)


def run_fits(root,plan,pulse):
    from .pipeline_new import run_fits as verified_fits
    return verified_fits(root,plan,pulse,fit_builder=lambda directory,family,heartbeat:pipe.fit_family(directory,heartbeat))


def profile(root,plan,pulse):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin']);costs=[]
    # Largest observed support first; a cap failure cannot hide behind cheap cells.
    population=sorted(registry.variants('N1'),key=lambda v:(v['settings']['makers']*v['settings']['records'],
        v['settings']['structure'].startswith('drift'),v['settings']['concentration']),reverse=True)
    for v in population:
        start=time.process_time()
        pipe.retained_world(root/v['id'],v,'complete-grid-cost-profile',0,schema,plan['canary_key'],pulse=pulse)
        costs.append(dict(variant=v['id'],cpu_seconds=time.process_time()-start,whole_populations=1,makers=v['settings']['makers'],records=v['settings']['records']))
        write(root/'profile-cost.json',dict(profile=costs),immutable=False)
        pulse(phase='population-grid-profile',variant=v['id'],completed=len(costs),total=240)
    started=time.process_time();x,y=pipe.training(root/'training-profile',registry.variants('N1'),'cost-profile',240,pulse)
    training=time.process_time()-started
    model=Direct(x,y,x,y,3,'N1-profile',0,max_epochs=2,pulse=pulse);model.save(root/'pilot-fit.npz')
    write(root/'pilot-cost.json',dict(profile=costs,training_generation_cpu_seconds=training,training_labels=240,
        fit_cpu_seconds=model.cpu_seconds,fit_epochs=model.metadata['epochs']))
    write(root/'SUMMARY.json',dict(state='PILOT_VALID',family='N1',complete_grid_profile=True,
        variants=240,largest_cells_included=True,scientific_claim=False))
