"""E01 selection without rewriting: finite supplied-program intervention ruler."""
import time,hashlib,importlib,types,sys,json
from pathlib import Path
import numpy as np
from . import world as W
from .routes import retain
from ..v18_3.io import write,file_digest,digest

WEIGHTS=('character-edits','uniform-events','content-intervention')

def execute(proposals,choice,edits):
    return np.array(proposals[int(choice)],copy=True),int(edits%2)

def posterior(proposals,artifact,prior,record):
    mass=np.array([1-prior,prior])*np.all(proposals==artifact,axis=1)
    if record is not None:mass*=np.arange(2)==record
    if not mass.sum():raise ValueError('inconsistent selection evidence')
    return mass/mass.sum()

def effects(proposals,choice,edits):
    return np.r_[np.count_nonzero(proposals[choice]!=proposals[1-choice]),np.zeros(edits,dtype=int)]

def controls():
    proposals=np.array([[0,0],[1,1]])
    from .selection_check import execute as scalar_execute,posterior as scalar_posterior,effects as scalar_effects
    return dict(live_selection_changes_content=effects(proposals,0,4).tolist()==[2,0,0,0,0],placebo_cosmetics_leave_content=bool(np.array_equal(execute(proposals,0,0)[0],execute(proposals,0,16)[0])),positive_identifiable_selection=bool(posterior(proposals,proposals[1],.3,None)[1]==1),live_identical_proposals_alias=bool(np.array_equal(posterior(np.zeros((2,2)),np.zeros(2),.3,None),[.7,.3])),positive_independent=scalar_execute(proposals,1,3)==([1,1],1) and scalar_effects(proposals,0,4)==[2,0,0,0,0] and scalar_posterior(proposals,[1,1],.3,None)==[0.,1.])

def fixtures(root,sl):
    from ..v20.integration import load
    module=load('v21_selection_process',sl/'soundingline/process_record.py');pkg=types.ModuleType('v21_selection_stage13');pkg.__path__=[str(sl/'runners/stage13')];sys.modules[pkg.__name__]=pkg
    exporter=importlib.import_module(pkg.__name__+'.export_tompathy');common=importlib.import_module(pkg.__name__+'.common')
    names=('soundingline/process_record.py','runners/stage13/export_tompathy.py','runners/stage13/common.py','runners/stage13/scoring.py');before={n:file_digest(sl/n) for n in names};cases=[]
    for i in range(12):
        n=(2,4,8)[i%3];choice=i%2;edits=(0,1,4,16)[i//3];proposals=np.array([[0]*n,[1]*n]);artifact,style=execute(proposals,choice,edits)
        identity=digest(['v21-e01-fixture',i])[:24];lineage=digest(['v21-e01-group',i//3])[:24];text=' '.join(map(str,artifact.tolist()))
        observed=dict(proposals=proposals.tolist(),artifact=artifact.tolist(),cosmetic_style=style,cosmetic_edit_count=edits,selection_record=choice if i%3==0 else None)
        case=module.ProcessCase(case_id=identity,lineage_id=lineage,domain='constructed',medium='selected-binary-plan',brief_id='synthetic',declared_context=observed,participants={'human':'constructed'},route_family='decisive-selection',events=[],artifact_final=text);case.validate()
        art=dict(id=identity,revision=hashlib.sha256(text.encode()).hexdigest(),regions=[dict(id='region-0',anchor=dict(exact=text,index=0,start=0,prefix='',suffix=''))]);reading={'region-0':dict(visibility='I2',goals=[],processes=[dict(label='Selected one complete proposal',probability=.5)])}
        excerpt=json.dumps(observed,sort_keys=True,separators=(',',':'));source='Ghost V21 constructed selection observations';ctx=[dict(id=common.digest([source,excerpt]),source=source,excerpt=excerpt,selected=True)]
        outer=dict(case_id=identity,scope='constructed observations; interventions evaluator-only',observed_events=[],unknown_review_scope=True,governing_purpose='not inferred from character-edit count')
        exported,sidecar=exporter.export(art,reading,context=ctx,outer=outer);base=root/'reader'/identity
        write(base/'observed.json',observed);write(base/'offline-packet.json',exported);write(base/'sidecar.json',sidecar);write(base/'candidates.json',dict(target='selected proposal identity',choices=[0,1]))
        write(root/'evaluator'/f'{identity}.json',dict(actual_choice=choice,intervention_effects=effects(proposals,choice,edits).tolist(),supplied_program_posterior=posterior(proposals,artifact,.5,observed['selection_record']).tolist(),scope='evaluator-only; offline placeholder reading is not inference'))
        cases.append(dict(case_id=identity,canonical_process_case=True,offline_export=True))
    assert before=={n:file_digest(sl/n) for n in names};write(root/'CONSUMER.json',dict(passed=True,sources=before,cases=cases,coverage='12 decisive-selection fixtures; no omission/retraction coverage',browser_tested=False));return cases

def run(root,d,pulse):
    from .selection_check import posterior as scalar_posterior,execute as scalar_execute,effects as scalar_effects
    (root/'raw').mkdir(parents=True,exist_ok=True)
    population=[];timings=[];maxerr=0.;interventions=0;aliases=0
    for world in d['lineages']:
        start=time.process_time();rng=np.random.default_rng(W.seed('v21-e01-law',world));prior=float(rng.uniform(.2,.8));rng=np.random.default_rng(W.seed('v21-e01-cases',world))
        proposals=rng.integers(2,size=(d['test_cases'],2,max(d['descendants'])));choices=(rng.random(d['test_cases'])<prior).astype(int)
        # Nested units, edits and witness conditions reuse identical cases.
        identity=f"case-{W.seed('v21-e01-blind-id',world):016x}";(root/'evaluator').mkdir(parents=True,exist_ok=True);(root/'reader').mkdir(exist_ok=True)
        np.savez_compressed(root/'evaluator'/f'{identity}_points.npz',world=world,choice=choices,prior=prior)
        for n in d['descendants']:
          for edits in d['cosmetic_edits']:
            plans=proposals[:,:,:n];artifacts=np.array([execute(p,c,edits)[0] for p,c in zip(plans,choices)]);effect=np.array([effects(p,c,edits) for p,c in zip(plans,choices)]);interventions+=len(choices)*(edits+1)
            for p,c,a,eff in zip(plans,choices,artifacts,effect):assert scalar_execute(p,int(c),edits)==(a.tolist(),edits%2) and scalar_effects(p,int(c),edits)==eff.tolist()
            norm=effect.sum(1);has_effect=norm>0;alias=np.all(plans[:,0]==plans[:,1],axis=1);assert np.array_equal(alias,~has_effect);aliases+=int(alias.sum())
            weights={}
            weights['character-edits']=np.tile(np.r_[0.,np.ones(edits)/edits if edits else np.array([])],(len(choices),1))
            weights['uniform-events']=np.full((len(choices),edits+1),1/(edits+1))
            weights['content-intervention']=np.divide(effect,norm[:,None],out=np.zeros(effect.shape,float),where=norm[:,None]>0)
            for weighting,w in weights.items():
                np.savez_compressed(root/'raw'/f'{world}-{n}-{edits}-{weighting}_points.npz',weights=w,intervention_effects=effect,has_content_effect=has_effect)
                population.append(dict(lineage=world,descendants=n,cosmetic_edits=edits,weighting=weighting,selection_weight=float(w[:,0].mean()),selection_weight_when_effect=float(w[has_effect,0].mean()) if has_effect.any() else None,mean_changed_units=float(norm.mean()),no_content_effect_fraction=float(alias.mean()),mean_absolute_weight_error=float(np.abs(w-weights['content-intervention']).sum(1).mean()),scope='Target-specific causal weighting, not a human contribution share.'))
            for witness in ('none','complete'):
                observed=np.full(len(choices),-1) if witness=='none' else choices
                np.savez_compressed(root/'reader'/f'{identity}-{n}-{edits}-{witness}_points.npz',proposals=plans,artifact=artifacts,cosmetic_style=np.full(len(choices),edits%2),selection_record=observed)
                q=[];direct=[]
                for p,a,record in zip(plans,artifacts,observed):
                    rec=None if record<0 else int(record);q.append(posterior(p,a,prior,rec));direct.append(scalar_posterior(p,a,prior,rec))
                err=float(np.max(abs(np.array(q)-direct)));assert err<1e-12;maxerr=max(maxerr,err)
                for reader,pred in [('historical',q),('direct-enumeration',direct)]:
                    metrics=retain(root,f'{world}-{n}-{edits}-{witness}-{reader}',np.asarray(pred),choices)
                    population.append(dict(lineage=world,descendants=n,cosmetic_edits=edits,witness=witness,reader=reader,metrics=metrics))
                pulse(phase='selection-consumer',lineage=world,descendants=n,cosmetic_edits=edits,witness=witness)
        timings.append(dict(lineage=world,total_cpu_seconds=time.process_time()-start))
    cases=fixtures(root/'fixtures',Path(d['soundingline'])) if d.get('export_fixtures') else []
    return dict(card='G21-E01',population=population,timings=timings,controls=controls(),independent_max_probability_error=maxerr,interventions=interventions,alias_cases_across_repeated_edit_conditions=aliases,fixtures=cases,limitations=['Supplied executable operation graph and choice prior; no learned causal attribution.','One selection copies a complete plan; cosmetic edits change style only. Content-specific influence is not total aesthetic or human contribution.','Zero intervention effect is explicitly left unnormalized; uniform/edit weighting then measures a different target.','Identical proposals alias the hidden choice unless its record is supplied.','Direct enumeration has identical evidence and program knowledge.','Native only; no shifted or duplicate-product reader. Nested units, cosmetics and direct parity are controls, not independent worlds.','Miniature — architecture untested.'])
