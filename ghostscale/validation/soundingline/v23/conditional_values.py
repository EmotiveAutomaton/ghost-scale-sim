"""N4b's separate binary probe: aliased history and observed contingency change.

The hidden mechanism is excluded from all reader inputs. The probe label is
first scored as an unseen choice, then becomes evidence for mechanism recovery;
it is never used in its own forecast. No N4b statistic selects the N4 main family.
"""
from pathlib import Path
import hashlib,hmac,math,time
import numpy as np
from . import values,registry,ledger,checker_recipes,fit_rivals
from .common import Direct,canonical,digest,interval,read,write
from .pipeline import Stream
from .pipeline_new import score


def variants():return [v for v in registry.variants('N4') if v['settings']['extension']]


def features(v,p):
    x=np.zeros((32,3))
    for i,row in enumerate(p['history']):x[i]=[row[0],row[1],1]
    return np.r_[x.ravel(),float(p['changed'])]


def sample(v,phase,world,case):return values.sample_extension(v,phase,world,case,world%16)


def evaluate(v,p,t):
    pred,detail=values.evaluate_extension(v,p)
    error=checker_recipes.verify_n4(v,p,t,pred,detail)
    before=np.array(detail['mechanism_posterior']);observed=t['y'][0]
    likelihood=np.array([pred[name][0,observed] for name in ('transition','habit','intrinsic')])
    after=before*likelihood;after/=after.sum()
    # Independent conditional-mass derivation, after the held-out probe is scored.
    rels=np.linspace(.6,.95,16);mass=np.ones((3,16))/48
    for row in p['history']:
        mass*=rels if row[0] else 1-rels;mass/=mass.sum()
    prob=np.array([1-rels,np.full(16,.5),rels]) if p['changed'] else np.tile(rels,(3,1))
    mass*=prob if observed else 1-prob;mass/=mass.sum()
    if not np.allclose(after,mass.sum(1),atol=1e-12,rtol=0):raise ValueError('conditional mechanism reconstruction differs')
    return pred,dict(detail,after_probe_posterior=after.tolist(),independent_max_error=error),dict(
        pre_probe_mechanism_loss=-math.log(before[t['mechanism']]),post_probe_mechanism_loss=-math.log(after[t['mechanism']]),
        probe_label_used_for_forecast=False)


def controls():
    a=next(v for v in variants() if v['settings']['changed']);b=dict(a,settings=dict(a['settings'],changed=False))
    p,t=sample(b,'known-answer',0,0);pred,d,_=evaluate(b,p,t)
    if not np.allclose(d['after_probe_posterior'],1/3,atol=1e-12,rtol=0):raise ValueError('unchanged probe identifies aliased mechanism')
    p,t=sample(a,'known-answer',15,0);_,d,_=evaluate(a,p,t)
    if np.allclose(d['after_probe_posterior'],1/3,atol=1e-3,rtol=0):raise ValueError('changed probe contains no planted information')
    altered=dict(a,settings=dict(a['settings'],mechanism='habit'))
    if not np.array_equal(features(a,p),features(altered,p)):raise ValueError('hidden mechanism leaked into features')
    return dict(passed=True,unchanged_probe_aliased=True,changed_probe_informative=True,hidden_mechanism_excluded=True)


def run(root,plan,pulse):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    write(root/'CONTROLS.json',controls());population=variants();fitted={};fitmeta={}
    if plan['task']=='conditional-fits':
        fitted,fitmeta=fit_rivals.fit(root/'fits','N4b',population,sample,features,2,pulse,'N4b-training')
        write(root/'SUMMARY.json',dict(state='FITS_VERIFIED',family='N4b',fits=fitmeta,discovery_exposure=False));return
    pilot=plan['task']=='conditional-pilot'
    if not pilot:
        if plan['worlds']!=16 or plan['cases']!=64:raise ValueError('conditional full allocation changed')
        fitroot=Path(plan['fit_directory'])
        for name,expected in plan['fit_hashes'].items():
            if digest(fitroot/name)!=expected:raise ValueError('conditional frozen fit changed')
        fitmeta=read(fitroot/'FITS.json');fitted={name:Direct.load(fitroot/(name+'.npz')) for name in fitmeta}
    summaries=[];costs=[]
    for v in population:
        start=time.process_time();folder=root/v['id'];folder.mkdir();worlds=[]
        streams={name:Stream(folder/(name+'_points.jsonl.gz')) for name in ('public','truth','prediction','posterior','ledger')}
        try:
            for world in range(1 if pilot else plan['worlds']):
                gains=[]
                for case in range(1 if pilot else plan['cases']):
                    identity=['N4b',v['id'],'pilot' if pilot else 'discovery',world,case]
                    p,t=sample(v,identity[2],world,case);pred,detail,diagnostic=evaluate(v,p,t)
                    bundle=ledger.emit(v,p,schema,identity);decoded=ledger.decode(v,bundle,schema)
                    if canonical(decoded)!=canonical(p):raise ValueError('conditional ledger changed observations')
                    canary=hmac.new(bytes.fromhex(plan['canary_key']),canonical(identity),hashlib.sha256).hexdigest()
                    if canary.encode() in canonical(bundle):raise ValueError('conditional truth canary leaked')
                    for name,model in fitted.items():pred[name]=model.predict(features(v,p)[None,:])[0]
                    scores={name:score(prob,t) for name,prob in pred.items()}
                    streams['public'].write(dict(identity=identity,public=p));streams['truth'].write(dict(identity=identity,truth=t,canary=canary))
                    streams['ledger'].write(dict(identity=identity,ledger=bundle));streams['posterior'].write(dict(identity=identity,detail=detail))
                    streams['prediction'].write(dict(identity=identity,probabilities=pred,scores=scores,diagnostic=diagnostic))
                    gains.append(diagnostic['pre_probe_mechanism_loss']-diagnostic['post_probe_mechanism_loss'])
                worlds.append(float(np.mean(gains)));pulse(phase='conditional-values',variant=v['id'],world=world+1)
        finally:
            for stream in streams.values():stream.close()
        summaries.append(dict(variant=v,world_gains=worlds,probe_information_gain=interval(worlds,[v['id'],'N4b']),
            mechanism_label_not_observed_before_probe=True))
        costs.append(dict(variant=v['id'],cpu_seconds=time.process_time()-start,cases=1 if pilot else plan['worlds']*plan['cases']))
    training=fit_rivals.profile(root/'training-profile','N4b',population,sample,features,2,pulse,'N4b-training') if pilot else {}
    write(root/'pilot-cost.json' if pilot else root/'batch-cost.json',dict(profile=costs,**training))
    write(root/'SUMMARY.json',dict(state='PILOT_VALID' if pilot else 'CONDITIONAL_BATCH_CHECKED',family='N4b',
        variants=summaries,fits=fitmeta,selection_eligible=False,main_N4_contrast_unchanged=True))


def reconstruct(root,plan):
    from .family import rows
    from . import checker
    root=Path(root);summary=read(root/'SUMMARY.json');checked={}
    for item in summary['variants']:
        v=item['variant'];folder=root/v['id'];worlds=[[] for _ in range(plan['worlds'])];seen=set()
        paths=[folder/(name+'_points.jsonl.gz') for name in ('public','truth','prediction','posterior')]
        for public,truth,pred,post in zip(*(rows(p) for p in paths),strict=True):
            identity=public['identity'];world,case=identity[-2:]
            if identity!=truth['identity'] or identity!=pred['identity'] or identity!=post['identity']:
                raise ValueError('conditional record identity mismatch')
            if (world,case) in seen or not 0<=world<plan['worlds'] or not 0<=case<plan['cases']:
                raise ValueError('conditional coverage mismatch')
            seen.add((world,case));expected,detail,diagnostic=evaluate(v,public['public'],truth['truth'])
            if canonical(detail)!=canonical(post['detail']) or canonical(diagnostic)!=canonical(pred['diagnostic']):
                raise ValueError('conditional posterior reconstruction differs')
            for name,prob in pred['probabilities'].items():
                rebuilt=checker.scores(np.asarray(prob).reshape(-1,2),np.asarray(truth['truth']['y']))
                for metric,values in rebuilt.items():
                    if not math.isclose(float(np.mean(values)),pred['scores'][name][metric],abs_tol=1e-11,rel_tol=0):
                        raise ValueError('conditional proper score differs')
            for name,prob in expected.items():
                if not np.allclose(prob,pred['probabilities'][name],atol=1e-12,rtol=0):raise ValueError('conditional native forecast differs')
            worlds[world].append(diagnostic['pre_probe_mechanism_loss']-diagnostic['post_probe_mechanism_loss'])
        if len(seen)!=plan['worlds']*plan['cases']:raise ValueError('conditional coverage incomplete')
        gains=[math.fsum(x)/len(x) for x in worlds]
        if not np.allclose(gains,item['world_gains'],atol=1e-12,rtol=0):raise ValueError('conditional whole-world aggregation differs')
        checked[v['id']]=dict(worlds=len(worlds),cases=len(seen))
    return dict(state='passed',variants=checked,selection_eligible=False)
