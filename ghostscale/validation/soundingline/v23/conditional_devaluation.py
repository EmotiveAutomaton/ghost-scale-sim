"""N4c source-frozen conditional execution and complete saved-record review."""
from pathlib import Path
import hashlib,hmac,math,time
import numpy as np
from . import ledger,fit_rivals
from .common import Direct,canonical,digest,interval,read,write
from .pipeline import Stream
from .pipeline_new import score
from .devaluation import variants,sample,features,evaluate,controls,emit,decode


def run(root,plan,pulse):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    write(root/'CONTROLS.json',controls());population=variants();fitted={};fitmeta={}
    if plan['task']=='conditional-fits':
        fitted,fitmeta=fit_rivals.fit(root/'fits','N4c',population,sample,features,6,pulse,'N4c-training')
        write(root/'SUMMARY.json',dict(state='FITS_VERIFIED',family='N4c',fits=fitmeta,discovery_exposure=False));return
    pilot=plan['task']=='conditional-pilot'
    if not pilot:
        if plan['worlds']!=16 or plan['cases']!=64:raise ValueError('conditional full allocation changed')
        fitroot=Path(plan['fit_directory'])
        for name,expected in plan['fit_hashes'].items():
            if digest(fitroot/name)!=expected:raise ValueError('conditional frozen fit changed')
        fitmeta=read(fitroot/'FITS.json');fitted={name:Direct.load(fitroot/(name+'.npz')) for name in fitmeta}
    summaries=[];costs=[]
    for v in population:
        start=time.process_time();folder=root/v['id'];folder.mkdir();worlds=[];secondary={};forecast={}
        streams={name:Stream(folder/(name+'_points.jsonl.gz')) for name in ('public','truth','prediction','posterior','ledger')}
        try:
            for world in range(1 if pilot else plan['worlds']):
                gains=[];secondary_cases={};forecast_cases={}
                for case in range(1 if pilot else plan['cases']):
                    identity=['N4c',v['id'],'pilot' if pilot else 'discovery',world,case]
                    p,t=sample(v,identity[2],world,case);pred,detail,diagnostic=evaluate(v,p,t)
                    bundle=emit(p,schema,identity);decoded=decode(bundle,schema)
                    if canonical(decoded)!=canonical(p):raise ValueError('conditional ledger changed observations')
                    canary=hmac.new(bytes.fromhex(plan['canary_key']),canonical(identity),hashlib.sha256).hexdigest()
                    if canary.encode() in canonical(bundle):raise ValueError('conditional truth canary leaked')
                    for name,model in fitted.items():pred[name]=model.predict(features(v,p)[None,:])[0]
                    scores={name:score(prob,t) for name,prob in pred.items()}
                    streams['public'].write(dict(identity=identity,public=p));streams['truth'].write(dict(identity=identity,truth=t,canary=canary))
                    streams['ledger'].write(dict(identity=identity,ledger=bundle));streams['posterior'].write(dict(identity=identity,detail=detail))
                    streams['prediction'].write(dict(identity=identity,probabilities=pred,scores=scores,diagnostic=diagnostic))
                    gains.append(diagnostic['pre_probe_mechanism_loss']-diagnostic['post_probe_mechanism_loss'])
                    for name in ('A','B','sham','action-only'):
                        secondary_cases.setdefault(name,[]).append(diagnostic[name+'_gain'])
                    for name,metrics in scores.items():forecast_cases.setdefault(name,[]).append(metrics['log_loss'])
                worlds.append(float(np.mean(gains)));pulse(phase='conditional-values',variant=v['id'],world=world+1)
                for name,items in secondary_cases.items():secondary.setdefault(name,[]).append(float(np.mean(items)))
                for name,items in forecast_cases.items():forecast.setdefault(name,[]).append(float(np.mean(items)))
        finally:
            for stream in streams.values():stream.close()
        summaries.append(dict(variant=v,world_gains=worlds,probe_information_gain=interval(worlds,[v['id'],'N4c']),
            secondary={name:dict(world_values=items,interval=interval(items,[v['id'],'N4c',name])) for name,items in secondary.items()},
            forecast_log_loss={name:dict(world_values=items,interval=interval(items,[v['id'],'N4c','forecast',name])) for name,items in forecast.items()},
            mechanism_label_not_observed_before_probe=True))
        costs.append(dict(variant=v['id'],cpu_seconds=time.process_time()-start,cases=1 if pilot else plan['worlds']*plan['cases']))
    training=fit_rivals.profile(root/'training-profile','N4c',population,sample,features,6,pulse,'N4c-training') if pilot else {}
    write(root/'pilot-cost.json' if pilot else root/'batch-cost.json',dict(profile=costs,**training))
    write(root/'SUMMARY.json',dict(state='PILOT_VALID' if pilot else 'CONDITIONAL_BATCH_CHECKED',family='N4c',
        variants=summaries,fits=fitmeta,selection_eligible=False,main_N4_contrast_unchanged=True, miniature_architecture_untested=True))


def reconstruct(root,plan):
    from .family import rows
    from . import checker
    root=Path(root);summary=read(root/'SUMMARY.json');checked={}
    fitroot=Path(plan['fit_directory'])
    for name,expected in plan['fit_hashes'].items():
        if digest(fitroot/name)!=expected:raise ValueError('N4c fit changed before reconstruction')
    fitted={name:Direct.load(fitroot/(name+'.npz')) for name in read(fitroot/'FITS.json')}
    for item in summary['variants']:
        v=item['variant'];folder=root/v['id'];worlds=[[] for _ in range(plan['worlds'])];seen=set();secondary={};forecast={}
        paths=[folder/(name+'_points.jsonl.gz') for name in ('public','truth','prediction','posterior')]
        for public,truth,pred,post in zip(*(rows(p) for p in paths),strict=True):
            identity=public['identity'];world,case=identity[-2:]
            if identity!=truth['identity'] or identity!=pred['identity'] or identity!=post['identity']:
                raise ValueError('conditional record identity mismatch')
            if (world,case) in seen or not 0<=world<plan['worlds'] or not 0<=case<plan['cases']:
                raise ValueError('conditional coverage mismatch')
            seen.add((world,case));expected,detail,diagnostic=evaluate(v,public['public'],truth['truth'])
            for name,model in fitted.items():expected[name]=model.predict(features(v,public['public'])[None,:])[0]
            if set(expected)!=set(pred['probabilities']):raise ValueError('N4c reader coverage changed')
            if canonical(detail)!=canonical(post['detail']) or canonical(diagnostic)!=canonical(pred['diagnostic']):
                raise ValueError('conditional posterior reconstruction differs')
            for name,prob in pred['probabilities'].items():
                rebuilt=checker.scores(np.asarray(prob).reshape(-1,6),np.asarray(truth['truth']['y']))
                for metric,values in rebuilt.items():
                    if not math.isclose(float(np.mean(values)),pred['scores'][name][metric],abs_tol=1e-11,rel_tol=0):
                        raise ValueError('conditional proper score differs')
            for name,prob in expected.items():
                if not np.allclose(prob,pred['probabilities'][name],atol=1e-12,rtol=0):raise ValueError('conditional native forecast differs')
            worlds[world].append(diagnostic['pre_probe_mechanism_loss']-diagnostic['post_probe_mechanism_loss'])
            for name in ('A','B','sham','action-only'):
                secondary.setdefault(name,[[] for _ in range(plan['worlds'])])[world].append(diagnostic[name+'_gain'])
            for name,metrics in pred['scores'].items():
                forecast.setdefault(name,[[] for _ in range(plan['worlds'])])[world].append(metrics['log_loss'])
        if len(seen)!=plan['worlds']*plan['cases']:raise ValueError('conditional coverage incomplete')
        gains=[math.fsum(x)/len(x) for x in worlds]
        if not np.allclose(gains,item['world_gains'],atol=1e-12,rtol=0):raise ValueError('conditional whole-world aggregation differs')
        for kind,groups in (('secondary',secondary),('forecast_log_loss',forecast)):
            for name,blocks in groups.items():
                means=[math.fsum(x)/len(x) for x in blocks]
                if not np.allclose(means,item[kind][name]['world_values'],atol=1e-12,rtol=0):
                    raise ValueError('N4c secondary whole-world aggregation differs')
        checked[v['id']]=dict(worlds=len(worlds),cases=len(seen))
    return dict(state='passed',variants=checked,selection_eligible=False)
