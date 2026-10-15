"""Pre-outcome largest-cell controls/cost pilot for each independently built repair."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from . import repairs,registry,checker,ledger,pipeline
from .common import Direct,digest,records,write


def largest(family):
    candidates=registry.variants(family)
    if family=='R1': key=lambda v:(v['settings']['bits'],v['settings']['strength'],v['settings']['prevalence'])
    elif family=='R2': key=lambda v:(v['settings']['prefix'],v['settings']['coupled'])
    elif family=='R3': key=lambda v:(v['settings']['records'],v['settings']['dimensions'],v['settings']['shift'])
    elif family=='R4': key=lambda v:(v['settings']['branching'],v['settings']['coverage'])
    else: raise ValueError('largest-cell pilot not implemented for '+family)
    return max(candidates,key=key)


def known_answers(family):
    v=deepcopy(largest(family))
    if family=='R1':
        v['settings']['strength']=.5;p,t=repairs.sample(v,'pilot-controls',0,0)
        a=repairs.r1_posterior(v,p);b=repairs.r1_posterior(v,p,masked=True)
        negative=float(np.max(np.abs(a-b)))
        v['settings']['strength']=.9;p['selection']=[1]*32
        positive=float(repairs.r1_posterior(v,p)[1,1])>.99
    elif family=='R2':
        v['settings']['benign_share']=.5;p=dict(flags=[1]*32,observations=[[1,1,1]]*32)
        a,path=repairs.r2_trajectory(v,p,'withdraw-all-0-0.5-none')
        negative=abs(a[0,1]-.5)
        positive=repairs.r2_trajectory(v,p,'no-gate')[0][0,1]>.99
    elif family=='R3':
        v['settings']['cost_role']='irrelevant';p,t=repairs.sample(v,'pilot-controls',0,0)
        a=repairs.r3_infer(v,p,role=0.)[0];q=deepcopy(p)
        for row in q['history']+[q['current']]: row['costs']=[100,200,300]
        negative=float(np.max(np.abs(a-repairs.r3_infer(v,q,role=0.)[0])))
        # Positive is sensitivity to genuinely relevant costs in a known menu,
        # not an empirical expectation that the questioned weighting wins.
        known=repairs.choice3(np.zeros(4),np.zeros(4),np.zeros((3,4)),np.array([0,10,20]),0,1.)[0]
        positive=known[0]>.99
    elif family=='R4':
        v['settings'].update(cue_reliability=.5,coverage=0)
        parent,_=repairs.tree(v['settings']['branching']);p=dict(evidence=[-1]*len(parent),cue=1,cue_node=0)
        negative=float(np.max(np.abs(repairs.r4_infer(v,p)-repairs.r4_infer(v,p,cue=False))))
        v['settings']['cue_reliability']=.9
        positive=abs(repairs.r4_infer(v,p)[0,1]-.9)<1e-12
    else: raise ValueError(family)
    if negative>1e-12 or not positive:raise ValueError('known-answer family control failed')
    return dict(negative_max_error=negative,positive=bool(positive),passed=True)


def run(root,plan,pulse):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    family=plan['family'];v=largest(family)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    controls=known_answers(family);write(root/'CONTROLS.json',controls)
    rows=[];started=time.process_time();max_error=0.
    for world in range(16):
        p,t=repairs.sample(v,'cost-pilot',world,0)
        predictions,details=repairs.evaluate(v,p,t)
        max_error=max(max_error,getattr(checker,'verify_'+family.lower())(v,p,t,predictions,details))
        bundle=ledger.emit(v,p,schema,[family,'pilot',world]);decoded=ledger.decode(v,bundle,schema)
        if decoded!=p:raise ValueError('pilot ledger roundtrip differs')
        rows.append(dict(world=world,public=p,truth=t,predictions=predictions,details=details,ledger=bundle))
        pulse(phase='largest-cell-pilot',family=family,world=world+1)
    inference_seconds=time.process_time()-started
    records(root/'pilot_points.jsonl.gz',rows)
    started=time.process_time()
    x,y=pipeline.training(family,'cost-pilot-fit',128,pulse=pulse)
    vx,vy=pipeline.training(family,'cost-pilot-validation',32,pulse=pulse)
    model=Direct(x,y,vx,vy,3 if family=='R3' else 2,'pilot-'+family,0,max_epochs=2,pulse=pulse)
    model.save(root/'pilot-fit.npz')
    training_seconds=time.process_time()-started
    # Timing remains separate and is never a replayed scientific outcome.
    profile=[];profile_rows=[]
    for index,variant in enumerate(registry.variants(family)):
        clock=time.process_time()
        public,truth=repairs.sample(variant,'cost-profile',index%16,0)
        pred,detail=repairs.evaluate(variant,public,truth)
        error=getattr(checker,'verify_'+family.lower())(variant,public,truth,pred,detail)
        bundle=ledger.emit(variant,public,schema,[family,'cost-profile',index])
        if ledger.decode(variant,bundle,schema)!=public:raise ValueError('profile ledger differs')
        profile.append(dict(variant=variant['id'],cpu_seconds=time.process_time()-clock,cases=1))
        profile_rows.append(dict(variant=variant,public=public,truth=truth,predictions=pred,details=detail,
            ledger=bundle,independent_error=error))
        pulse(phase='grid-cost-profile',family=family,variant=variant['id'])
    records(root/'cost_profile_points.jsonl.gz',profile_rows)
    write(root/'pilot-cost.json',dict(inference_cpu_seconds=inference_seconds,cases=16,
        training_cpu_seconds=training_seconds,training_labels=128,validation_labels=32,
        optimizer_epochs=model.metadata['epochs'],profile=profile,
        scope='largest-cell safety pilot plus one retained control case per grid cell; timing uncertainty and training remain separate'))
    write(root/'SUMMARY.json',dict(state='PILOT_VALID',family=family,largest_variant=v,
        laws=16,cases=16,controls=controls,independent_error=max_error,
        raw_sha256=digest(root/'pilot_points.jsonl.gz'),fit_sha256=digest(root/'pilot-fit.npz'),
        profile_sha256=digest(root/'cost_profile_points.jsonl.gz'),
        scientific_claim='none; known-answer and cost instrumentation only',selection_using_pilot_effects=False))
