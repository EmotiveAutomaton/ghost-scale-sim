"""Retained execution for explicitly declared October 17 finite miniatures."""
from pathlib import Path
import hashlib,hmac,math,time
from functools import lru_cache
import numpy as np
from . import ledger,fit_rivals,checker
from .common import Direct,canonical,digest,interval,read,write,rng
from .pipeline import Stream
from .pipeline_new import score


@lru_cache(maxsize=256)
def _law_offset(family,phase_bytes):
    return int(rng('conditional-fresh-law',family,phase_bytes.decode()).integers(16))


def training_sample(model,v,phase,world,case):
    # The inherited fitter resets world to zero per 256-label minibatch. Twelve
    # variants visit all 16 laws but overweight the early ones. Randomize the
    # offset independently by fresh minibatch identity; expected law weights match.
    label=phase[1]
    if label=='cost-profile' or (isinstance(label,list) and label and label[0]=='fresh'):
        world += _law_offset(model.FAMILY,canonical(phase))
    return model.sample(v,phase,world,case)


def run(root,plan,pulse,model):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    write(root/'CONTROLS.json',model.controls());population=model.variants();fitted={};meta={}
    family=model.FAMILY;namespace=family+'-training';classes=model.CLASSES
    sampler=lambda v,phase,world,case:training_sample(model,v,phase,world,case)
    if plan['task']=='conditional-fits':
        fitted,meta=fit_rivals.fit(root/'fits',family,population,sampler,model.features,classes,pulse,namespace)
        write(root/'SUMMARY.json',dict(state='FITS_VERIFIED',family=family,fits=meta,discovery_exposure=False));return
    pilot=plan['task']=='conditional-pilot'
    if not pilot:
        if (plan['worlds'],plan['cases'])!=(16,64):raise ValueError('conditional allocation changed')
        fitroot=Path(plan['fit_directory'])
        for name,value in plan['fit_hashes'].items():
            if digest(fitroot/name)!=value:raise ValueError('conditional fit changed')
        meta=read(fitroot/'FITS.json');fitted={name:Direct.load(fitroot/(name+'.npz')) for name in meta}
    summaries=[];costs=[]
    for v in population:
        started=time.process_time();folder=root/v['id'];folder.mkdir();blocks={};forecasts={};confusion={}
        streams={name:Stream(folder/(name+'_points.jsonl.gz')) for name in ('public','truth','prediction','posterior','ledger')}
        try:
            for world in range(1 if pilot else 16):
                metrics={};scores_by_model={};world_confusion={}
                for case in range(1 if pilot else 64):
                    identity=[family,v['id'],'pilot' if pilot else 'discovery',world,case]
                    p,t=model.sample(v,identity[2],world,case);pred,detail,diagnostic=model.evaluate(v,p,t)
                    bundle=model.emit(p,schema,identity)
                    if canonical(model.decode(bundle,schema))!=canonical(p):raise ValueError('conditional ledger round trip')
                    canary=hmac.new(bytes.fromhex(plan['canary_key']),canonical(identity),hashlib.sha256).hexdigest()
                    if canary.encode() in canonical(bundle):raise ValueError('evaluator canary leaked')
                    for name,fit in fitted.items():pred[name]=fit.predict(model.features(v,p)[None,:])[0]
                    scores={name:score(prob,t) for name,prob in pred.items()}
                    for name,value in diagnostic.items():metrics.setdefault(name,[]).append(value)
                    for name,value in scores.items():scores_by_model.setdefault(name,[]).append(value['log_loss'])
                    for name,matrix in detail.get('confusion',{}).items():world_confusion.setdefault(name,[]).append(matrix)
                    for name,row in [('public',dict(public=p)),('truth',dict(truth=t,canary=canary)),
                        ('posterior',dict(detail=detail)),('ledger',dict(ledger=bundle)),
                        ('prediction',dict(probabilities=pred,scores=scores,diagnostic=diagnostic))]:streams[name].write(dict(identity=identity,**row))
                for name,values in metrics.items():blocks.setdefault(name,[]).append(float(np.mean(values)))
                for name,values in scores_by_model.items():forecasts.setdefault(name,[]).append(float(np.mean(values)))
                for name,values in world_confusion.items():confusion.setdefault(name,[]).append(np.sum(values,axis=0).tolist())
                pulse(phase='conditional-small',family=family,variant=v['id'],world=world+1)
        finally:
            for stream in streams.values():stream.close()
        summaries.append(dict(variant=v,world_gains=blocks['primary'],
            metrics={name:dict(world_values=values,interval=interval(values,[family,v['id'],name])) for name,values in blocks.items()},
            forecast_log_loss={name:dict(world_values=values,interval=interval(values,[family,v['id'],'forecast',name])) for name,values in forecasts.items()},
            confusion_counts_by_world=confusion))
        costs.append(dict(variant=v['id'],cpu_seconds=time.process_time()-started,cases=1 if pilot else 1024))
    training=fit_rivals.profile(root/'training-profile',family,population,sampler,model.features,classes,pulse,namespace) if pilot else {}
    write(root/('pilot-cost.json' if pilot else 'batch-cost.json'),dict(profile=costs,**training))
    write(root/'SUMMARY.json',dict(state='PILOT_VALID' if pilot else 'CONDITIONAL_BATCH_CHECKED',family=family,
        variants=summaries,fits=meta,selection_eligible=False,miniature_architecture_untested=True))


def reconstruct(root,plan,model):
    from .family import rows
    root=Path(root);summary=read(root/'SUMMARY.json');fitroot=Path(plan['fit_directory']);checked={};intervals=0
    for name,value in plan['fit_hashes'].items():
        if digest(fitroot/name)!=value:raise ValueError('fit binding changed')
    fitted={name:Direct.load(fitroot/(name+'.npz')) for name in read(fitroot/'FITS.json')}
    for item in summary['variants']:
        v=item['variant'];folder=root/v['id'];seen=set();metrics={};forecasts={};confusion={}
        paths=[folder/(name+'_points.jsonl.gz') for name in ('public','truth','prediction','posterior')]
        for p,t,saved,post in zip(*(rows(p) for p in paths),strict=True):
            identity=p['identity'];w,c=identity[-2:]
            if any(row['identity']!=identity for row in (t,saved,post)) or identity[:3]!=[model.FAMILY,v['id'],'discovery']:
                raise ValueError('identity mismatch')
            if (w,c) in seen or not 0<=w<16 or not 0<=c<64:raise ValueError('coverage mismatch')
            seen.add((w,c));pred,detail,diagnostic=model.evaluate(v,p['public'],t['truth'])
            if canonical(detail)!=canonical(post['detail']) or canonical(diagnostic)!=canonical(saved['diagnostic']):raise ValueError('posterior reconstruction differs')
            for name,fit in fitted.items():pred[name]=fit.predict(model.features(v,p['public'])[None,:])[0]
            if set(pred)!=set(saved['probabilities']):raise ValueError('reader membership differs')
            for name,prob in pred.items():
                if not np.allclose(prob,saved['probabilities'][name],atol=1e-12,rtol=0):raise ValueError('forecast differs')
                independent=checker.scores(np.asarray(prob).reshape(-1,model.CLASSES),np.array(t['truth']['y']))
                for metric,values in independent.items():
                    if not math.isclose(float(np.mean(values)),saved['scores'][name][metric],abs_tol=1e-11,rel_tol=0):raise ValueError('proper score differs')
                forecasts.setdefault(name,[[] for _ in range(16)])[w].append(float(np.mean(independent['log_loss'])))
            for name,value in diagnostic.items():metrics.setdefault(name,[[] for _ in range(16)])[w].append(value)
            for name,matrix in detail.get('confusion',{}).items():confusion.setdefault(name,[[] for _ in range(16)])[w].append(matrix)
        if len(seen)!=1024:raise ValueError('incomplete batch')
        for kind,groups in (('metrics',metrics),('forecast_log_loss',forecasts)):
            for name,values in groups.items():
                means=[math.fsum(x)/len(x) for x in values]
                if not np.allclose(means,item[kind][name]['world_values'],atol=1e-12,rtol=0):raise ValueError('world means differ')
                key=[model.FAMILY,v['id'],name] if kind=='metrics' else [model.FAMILY,v['id'],'forecast',name]
                # Separate aggregate/quantile implementation using the frozen draw identities.
                indices=rng('bootstrap',key).integers(16,size=(4096,16))
                boot=sorted(math.fsum(means[i] for i in sample)/16 for sample in indices)
                def quantile(q):
                    pos=q*(len(boot)-1);lo=int(pos);return boot[lo]+(pos-lo)*(boot[min(lo+1,len(boot)-1)]-boot[lo])
                center=math.fsum(boot)/len(boot)
                rebuilt=dict(mean=math.fsum(means)/16,lower=quantile(.025),upper=quantile(.975),
                    se=math.sqrt(math.fsum((x-center)**2 for x in boot)/(len(boot)-1)),finite=True,worlds=16)
                if any(not math.isclose(x,item[kind][name]['interval'][k],abs_tol=1e-11,rel_tol=0) for k,x in rebuilt.items() if isinstance(x,(float,int))):raise ValueError('world interval differs')
                intervals+=1
        for name,values in confusion.items():
            if not np.allclose([np.sum(x,axis=0) for x in values],item['confusion_counts_by_world'][name],atol=1e-12,rtol=0):raise ValueError('confusion reconstruction differs')
        checked[v['id']]=dict(worlds=16,cases=len(seen))
    return dict(state='passed',variants=checked,intervals_reconstructed=intervals,selection_eligible=False)
