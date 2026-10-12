"""C05 candidate omission: exact finite-unknown mass versus full-universe loss."""
import time,hashlib,importlib,types,sys,json
from pathlib import Path
import numpy as np
from . import world as W
from .routes import retain
from ..v18_3.io import write,file_digest,digest

READERS=('forced-choice','finite-unknown','uniform-expansion','evidence-ranked-expansion','truth-supplied')
UNIFORM_ORDER=np.random.default_rng(W.seed('v21-c05-training-expansion',71001000)).permutation(100)

def likelihood(cue,reliability=(.8,.6)):
    labels=np.array([np.arange(100)//10,np.arange(100)%10]).T
    return np.prod(np.where(labels==np.asarray(cue),reliability,(1-np.asarray(reliability))/9),axis=1)

def posterior(prior,cue,reliability=(.8,.6)):
    mass=np.asarray(prior)*likelihood(cue,reliability);return mass/mass.sum()

def predict(prior,cue,keep,budget,reader,truth=None):
    exact=posterior(prior,cue);included=np.arange(100)<keep;missing=np.flatnonzero(~included)
    if reader=='uniform-expansion':add=UNIFORM_ORDER[UNIFORM_ORDER>=keep][:budget];included[add]=True
    elif reader=='evidence-ranked-expansion':add=missing[np.argsort(-likelihood(cue)[missing],kind='stable')[:budget]];included[add]=True
    elif reader=='truth-supplied':included[truth]=True
    elif reader not in ('forced-choice','finite-unknown'):raise ValueError(reader)
    unknown=float(exact[~included].sum());q=exact.copy()
    if reader=='forced-choice' and (~included).any():q[~included]=0;q/=q.sum()
    elif (~included).any():q[~included]=unknown/(~included).sum()
    return q,included,unknown

def controls():
    from .omission_check import posterior as scalar,predict as reference
    prior=np.arange(1,101,dtype=float);prior/=prior.sum();cue=[3,4];exact=posterior(prior,cue)
    q,mask,mass=predict(prior,cue,50,64,'uniform-expansion')
    forced,_,_=predict(prior,cue,50,8,'forced-choice')
    actual=predict(prior,cue,75,8,'evidence-ranked-expansion');ref=reference(prior,cue,75,8,'evidence-ranked-expansion',UNIFORM_ORDER.tolist())
    return dict(live_forced_truth_exclusion=bool(forced[90]==0),placebo_no_information=bool(np.allclose(posterior(prior,cue,(.1,.1)),prior,atol=1e-15)),positive_full_expansion=bool(np.array_equal(q,exact) and mask.all() and mass==0),positive_no_omission=all(np.array_equal(predict(prior,cue,100,8,r,90)[0],exact) for r in READERS),positive_scalar=bool(np.allclose(exact,scalar(prior,cue),atol=1e-15) and np.allclose(actual[0],ref[0],atol=1e-15) and np.array_equal(np.flatnonzero(actual[1]),sorted(ref[1]))))

def fixtures(root,sl):
    from ..v20.integration import load
    module=load('v21_omission_process',sl/'soundingline/process_record.py');pkg=types.ModuleType('v21_omission_stage13');pkg.__path__=[str(sl/'runners/stage13')];sys.modules[pkg.__name__]=pkg
    exporter=importlib.import_module(pkg.__name__+'.export_tompathy');common=importlib.import_module(pkg.__name__+'.common')
    names=('soundingline/process_record.py','runners/stage13/export_tompathy.py','runners/stage13/common.py','runners/stage13/scoring.py');before={n:file_digest(sl/n) for n in names};cases=[]
    rng=np.random.default_rng(W.seed('v21-c05-fixtures',71000816));prior=np.full(100,.01)
    for i in range(12):
        keep=(90,75,50)[i%3];truth=int(rng.integers(100));cue=[int(rng.integers(10)),int(rng.integers(10))];identity=digest(['v21-c05-fixture',i])[:24];lineage=digest(['v21-c05-group',i//3])[:24]
        observed=dict(cue=cue,initial_candidates=list(range(keep)),unknown_option='remaining finite catalog',catalog_size=100);text='Observed family cue %d and parameter cue %d'%tuple(cue)
        case=module.ProcessCase(case_id=identity,lineage_id=lineage,domain='constructed',medium='finite-mechanism-cue',brief_id='synthetic',declared_context=observed,participants={'human':'constructed'},route_family='candidate-omission',events=[],artifact_final=text);case.validate()
        art=dict(id=identity,revision=hashlib.sha256(text.encode()).hexdigest(),regions=[dict(id='region-0',anchor=dict(exact=text,index=0,start=0,prefix='',suffix=''))]);reading={'region-0':dict(visibility='I2',goals=[],processes=[dict(label='Unresolved finite mechanism',probability=.5)])}
        excerpt=json.dumps(observed,sort_keys=True,separators=(',',':'));source='Ghost V21 constructed candidate observations';ctx=[dict(id=common.digest([source,excerpt]),source=source,excerpt=excerpt,selected=True)]
        outer=dict(case_id=identity,scope='constructed observations; truth and law evaluator-only',observed_events=[],unknown_review_scope=True,governing_purpose='finite omitted mass is not concept invention')
        exported,sidecar=exporter.export(art,reading,context=ctx,outer=outer);base=root/'reader'/identity
        for name,value in [('observed.json',observed),('offline-packet.json',exported),('sidecar.json',sidecar),('candidates.json',dict(choices=list(range(keep))+['unknown'],full_catalog=list(range(100))))]:write(base/name,value)
        write(root/'evaluator'/f'{identity}.json',dict(actual_type=truth,truth_omitted=truth>=keep,supplied_posterior=posterior(prior,cue).tolist(),scope='evaluator-only; placeholder reading is not inference'))
        cases.append(dict(case_id=identity,canonical_process_case=True,offline_export=True))
    assert before=={n:file_digest(sl/n) for n in names};write(root/'CONSUMER.json',dict(passed=True,sources=before,cases=cases,coverage='12 finite missing-candidate fixtures; no retraction coverage',browser_tested=False));return cases

def run(root,d,pulse):
    from .omission_check import predict as reference,event_loss
    population=[];timings=[];maxerr=0.;(root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
    for world in d['lineages']:
        start=time.process_time();rng=np.random.default_rng(W.seed('v21-c05-law',world));prior=rng.dirichlet(np.ones(100)*2);rng=np.random.default_rng(W.seed('v21-c05-cases',world))
        truth=rng.choice(100,size=d['test_cases'],p=prior);cues=[]
        for y in truth:
            row=[]
            for value,reliability in zip(divmod(int(y),10),(.8,.6)):
                probs=np.full(10,(1-reliability)/9);probs[value]=reliability;row.append(int(rng.choice(10,p=probs)))
            cues.append(row)
        cues=np.asarray(cues);identity=f"case-{W.seed('v21-c05-blind-id',world):016x}"
        np.savez_compressed(root/'reader'/f'{identity}_points.npz',cue=cues,catalog=np.arange(100),initial_keep_counts=np.array([100,90,75,50]))
        np.savez_compressed(root/'evaluator'/f'{identity}_points.npz',world=world,truth=truth,prior=prior)
        for keep in d['keep_counts']:
          for budget in d['budgets']:
            for reader in READERS:
                predictions=[];masks=[];unknown=[];event=[]
                for cue,y in zip(cues,truth):
                    q,mask,mass=predict(prior,cue,keep,budget,reader,int(y));ref,ids,refmass=reference(prior,cue,keep,budget,reader,UNIFORM_ORDER.tolist(),int(y));err=float(np.max(abs(q-ref)));maxerr=max(maxerr,err)
                    assert err<1e-12 and np.array_equal(np.flatnonzero(mask),sorted(ids)) and abs(mass-refmass)<1e-12
                    predictions.append(q);masks.append(mask);unknown.append(mass);event.append(event_loss(ref,ids,int(y)))
                pred=np.array(predictions);masks=np.array(masks);coverage=masks[np.arange(len(truth)),truth].astype(float);unknown=np.array(unknown);event=np.array(event)
                metrics=retain(root,f'{world}-{keep}-{budget}-{reader}',pred,truth,included=masks,unknown_mass=unknown,candidate_event_log_loss=event,explicit_coverage=coverage)
                population.append(dict(lineage=world,keep=keep,omission=1-keep/100,budget=budget,reader=reader,metrics=metrics,explicit_coverage=float(coverage.mean()),mean_unknown_mass=float(unknown.mean()),candidate_event_log_loss=float(event.mean()) if np.isfinite(event).all() else None,candidate_event_infinite_fraction=float(np.isinf(event).mean()),included_count=float(masks.sum(1).mean())))
                pulse(phase='omission-consumer',lineage=world,keep=keep,budget=budget,reader=reader)
        timings.append(dict(lineage=world,total_cpu_seconds=time.process_time()-start))
    cases=fixtures(root/'fixtures',Path(d['soundingline'])) if d.get('export_fixtures') else []
    return dict(card='G21-C05',population=population,timings=timings,controls=controls(),independent_max_probability_error=maxerr,fixtures=cases,uniform_expansion_order=UNIFORM_ORDER.tolist(),limitations=['Exact finite catalog, law and omitted aggregate mass supplied; unknown-mass calibration is privileged.','Unknown aggregate is uniformly spread for full-universe scoring; candidate-event score is coarser and kept separate.','Evidence ranking uses supplied noisy-cue likelihood, never test truth; truth-supplied is a separately labeled oracle.','All grids share cases; duplicate baselines and budget saturations are controls, not independent worlds.','No learned fitting, unknown-law integration, execution shift, architecture-superiority screen or confirmation.','Miniature — architecture untested.'])
