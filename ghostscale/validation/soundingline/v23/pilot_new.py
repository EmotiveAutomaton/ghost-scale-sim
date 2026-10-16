"""Bounded largest-cell and complete-grid engineering profiles for N families."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from . import registry, dispatch, ledger, pipeline_new, support_new
from .common import Direct, digest, write, read, losses


def controls(family):
    v=deepcopy(registry.variants(family)[0]);v['settings']['records']=2
    if family=='N2':v['settings'].update(mark_class='preference',nondiagnostic=0,rt_slope=0)
    if family=='N3':v['settings']['local_share']=1.
    if family=='N4':v['settings'].update(meaning='additive',recipes=1)
    if family=='N5':v['settings']['offset']=0
    if family=='N8':v['settings']['play_share']=0
    p,t=dispatch.sample(v,'pilot-known-answer',7,0);pred,detail=dispatch.evaluate(v,p,t)
    pipeline_new.verify_case(v,p,t,pred,detail)
    if family in ('N2','N4','N5','N8'):
        left,right=registry.PRIMARY[family][:2]
        if not np.allclose(pred[left],pred[right],atol=1e-12,rtol=0):raise ValueError('known-answer identity failed')
    if family=='N3':
        if not np.allclose(pred['law-marginal'],pred['final-choice-only'],atol=1e-12,rtol=0):raise ValueError('all-local corrections leaked preference')
    if family=='N6':
        changed=deepcopy(v);changed['settings']['stakes']=1.
        if not np.allclose(pred['law-marginal'],dispatch.evaluate(changed,p,t)[0]['law-marginal'],atol=1e-12,rtol=0):
            raise ValueError('stakes leaked into forecasting information')
    if family=='N7':
        from .checker_recipes import n7
        _,training=n7(v,p,'training-choices',None)
        changed=deepcopy(p);changed['statement']=[100.]*len(p['statement'])
        _,again=n7(v,changed,'training-choices',None)
        if not np.allclose(training['expected_weights'],again['expected_weights'],atol=1e-12,rtol=0):
            raise ValueError('shift fitting used statement as its own value target')
    # Every checker must reject a corrupted reference prediction.
    broken=deepcopy(pred);key='approximate-hierarchical' if family=='N5' else 'law-marginal'
    broken[key]=np.zeros_like(broken[key]);rejected=False
    try:pipeline_new.verify_case(v,p,t,broken,detail)
    except ValueError:rejected=True
    if not rejected:raise ValueError('independent checker accepted corrupt prediction')
    return dict(known_answer=True,mutation_refused=True,empirical_performance_not_a_gate=True)


def run(root,plan,pulse):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);family=plan['family']
    if family not in pipeline_new.BUILT:raise ValueError('pipeline not built')
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    known=controls(family);profiles=[];validity=[];auxiliary_cost=0.
    stream=pipeline_new.Stream(root/'pilot_points.jsonl.gz')
    started=time.process_time()
    try:
        population=pipeline_new.variants(family)
        if family=='N4':population=sorted(population,key=lambda v:(v['settings']['records'],v['settings']['recipes'],v['settings']['drives']),reverse=True)
        for v in population:
            auxiliary=root/v['id'];auxiliary.mkdir();aux_start=time.process_time()
            pipeline_new.retain_auxiliary(auxiliary,v)
            auxiliary_cost+=time.process_time()-aux_start
            cost_start=time.process_time();error=0.;cases=plan.get('profile_cases',1)
            for case in range(cases):
                public,truth=dispatch.sample(v,'cost-pilot',case%16,case)
                pred,detail=dispatch.evaluate(v,public,truth)
                error=max(error,pipeline_new.verify_case(v,public,truth,pred,detail))
                bundle=ledger.emit(v,public,schema,[family,v['id'],'pilot',case%16,case])
                decoded=ledger.decode(v,bundle,schema);again=dispatch.evaluate(v,decoded)[0]
                for name,probability in again.items():
                    if probability is None:
                        if pred[name] is not None:raise ValueError('availability changed across ledger')
                    elif not np.allclose(probability,pred[name],atol=1e-12,rtol=0):raise ValueError('ledger/native forecast mismatch')
                stream.write(dict(variant=v,public=public,truth=truth,predictions=pred,detail=detail,ledger=bundle))
            profiles.append(dict(variant=v['id'],cpu_seconds=time.process_time()-cost_start,cases=cases))
            validity.append(dict(variant=v['id'],independent_max_error=error,ledger_match=True))
            pulse(phase='new-family-cost-profile',family=family,variant=v['id'])
    finally:stream.close()
    inference=time.process_time()-started
    start=time.process_time();x,y=pipeline_new.training(family,'pilot-direct-training',128,pulse)
    vx,vy=pipeline_new.training(family,'pilot-direct-validation',64,pulse)
    generation=time.process_time()-start
    model=Direct(x,y,vx,vy,pipeline_new.CLASSES[family],family+'-pilot-only',0,max_epochs=2,pulse=pulse)
    model.save(root/'pilot-fit.npz');np.savez_compressed(root/'pilot-training.npz',x=x,y=y,vx=vx,vy=vy)
    write(root/'PILOT_FIT.json',model.metadata)
    write(root/'pilot-cost.json',dict(profile=profiles,inference_cpu_seconds=inference-auxiliary_cost,
        auxiliary_cpu_seconds=auxiliary_cost,cases=sum(row['cases'] for row in profiles),
        training_generation_cpu_seconds=generation,training_generation_labels=192,
        fit_cpu_seconds=model.cpu_seconds,fit_labels=128,fit_epochs=model.metadata['epochs'],
        note='Separate upper-bound training projection must cover six full fits and fresh generation before admission.'))
    write(root/'SUMMARY.json',dict(family=family,state='PILOT_VALID',controls=known,
        complete_grid_profile=True,largest_cells_included=True,variants=validity,
        support=support_new.bound(family),scientific_claim='none; engineering admission evidence only'))
