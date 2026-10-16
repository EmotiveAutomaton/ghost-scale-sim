"""Retained N1 population worlds and training, with independent reconstruction.

This builds the world-level consumer and largest-cell pilot. It does not admit
the 240-cell campaign or implement the separate N1 precision-extension ladder.
"""
import hashlib
import hmac
from pathlib import Path
import time
import numpy as np
from . import makers,checker_population,checker,ledger,registry
from .common import Direct,canonical,digest,losses,read,rng,write
from .pipeline import Stream


def save_public(path,public):
    np.savez_compressed(path,**{k:np.asarray(v) for k,v in public.items() if v is not None})


def load_public(path):
    with np.load(path,allow_pickle=False) as data:
        public={k:data[k] for k in data.files}
    public.setdefault('alias_pairs',None)
    return public


def features(v,public):
    # Only the declared design is appended. Culture, type, precision and labels
    # remain in the evaluator sidecar, never in this prediction-time interface.
    x=makers.n1_features(v,public)
    return np.c_[x,np.tile(registry.variant_features(v),(len(x),1))]


def same_public(a,b):
    if set(a)!=set(b):return False
    return all((a[k] is None and b[k] is None) or
               (a[k] is not None and b[k] is not None and np.array_equal(a[k],b[k])) for k in a)


def retain_ledger(root,v,public,schema,identity,canary,pulse):
    """Stream canonical per-maker partitions; reconstruct the complete population.

    Partitions have unique identities. The numeric population order is explicit
    in the transport envelope; no personal forecast is evaluated in isolation.
    This avoids a multimillion-row Python ledger object at the largest cell.
    """
    decoded={key:[] for key in public};stream=Stream(root/'ledger_points.jsonl.gz')
    try:
        for i in range(len(public['choices'])):
            part={key:None if value is None else value[i:i+1] for key,value in public.items()}
            bundle=ledger.emit(v,part,schema,[identity,'member',i])
            ledger.validate(bundle,schema,canaries=[canary])
            back=ledger.decode(v,bundle,schema)
            if not same_public(part,back):raise ValueError('population ledger partition differs')
            stream.write(dict(population_identity=identity,population_index=i,ledger=bundle))
            for key,value in back.items():
                if value is not None:decoded[key].append(value)
            pulse(phase='population-ledger',maker=i+1,makers=len(public['choices']))
    finally:stream.close()
    result={key:np.concatenate(value) if value else None for key,value in decoded.items()}
    if not same_public(public,result):raise ValueError('whole-population ledger reconstruction differs')
    # Exact input equality supplies a stronger deterministic round-trip identity
    # than a second numerical evaluation on the same already-checked arrays.
    return dict(partitions=len(public['choices']),public_arrays_equal=True,
                prediction_agreement='deterministic identity from complete decoded input equality')


def retained_world(root,v,phase,world,schema,canary_key,fitted=None,pulse=lambda **kw:None,law=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);start=time.process_time()
    public,truth=makers.n1_world(v,phase,world,law=law,pulse=pulse)
    identity=['N1',v['id'],phase,world]
    canary=hmac.new(bytes.fromhex(canary_key),canonical(identity),hashlib.sha256).hexdigest()
    save_public(root/'public_points.npz',public)
    np.savez_compressed(root/'truth_points.npz',**truth)
    write(root/'IDENTITY.json',dict(identity=identity,canary=canary,evaluator_only=True))
    prediction,detail=makers.n1_infer(v,public,pulse)
    error=checker_population.verify_n1(v,public,truth,prediction,detail,
        makers.cultures(v['settings']['concentration']),pulse)
    roundtrip=retain_ledger(root,v,public,schema,identity,canary,pulse)
    for name,model in (fitted or {}).items():prediction[name]=model.predict(features(v,public))[:,0,:]
    np.savez_compressed(root/'prediction_points.npz',**prediction)
    posterior={name+'__'+key:value for name,row in detail['readers'].items() for key,value in row.items()}
    posterior.update(culture_posterior=detail['culture_posterior'],population_types=detail['population_types'])
    np.savez_compressed(root/'posterior_points.npz',**posterior)
    scored={};summary={};y=np.asarray(truth['y'])
    for name,p in prediction.items():
        a=losses(p,y);b=checker.scores(p,y)
        for metric,value in a.items():
            if not np.allclose(value,b[metric],atol=1e-12,rtol=0):raise ValueError('independent population score differs')
            scored[name+'__outside__'+metric]=value
        if name in detail['readers']:
            d=detail['readers'][name]
            for metric,value in losses(d['in_context'],truth['in_context']).items():scored[name+'__inside__'+metric]=value
            presence=np.asarray(d['recipe_presence']);target=np.asarray(truth['presence'])
            binary=np.stack([1-presence,presence],axis=-1)
            # Preserve zero-support recipe losses; never smooth to make a metric finite.
            for metric,value in losses(binary,target).items():scored[name+'__presence__'+metric]=value
            top=presence.argmax(1);confidence=presence[np.arange(len(top)),top]
            observed=target[np.arange(len(top)),top]
            bins=[]
            for lo in np.arange(0,1,.1):
                mask=(confidence>=lo)&(confidence<(lo+.1) if lo<.9 else confidence<=1)
                bins.append(dict(lower=float(lo),count=int(mask.sum()),
                    confidence=float(confidence[mask].mean()) if mask.any() else None,
                    observed=float(observed[mask].mean()) if mask.any() else None))
            summary[name]=dict(top_recipe_probability_90=float(np.quantile(confidence,.9)),
                top_type_probability_90=float(np.quantile(d['top_type'],.9)),calibration=bins,
                capped_display_mean_error=float(np.mean(np.minimum(confidence,.5)-observed)))
    base=scored['individual-only__outside__log_loss'];pooled=scored['law-marginal__outside__log_loss']
    pi=makers.cultures(v['settings']['concentration'])[truth['law']]
    rarity=pi[truth['type']];thresholds=np.quantile(rarity,[.25,.5,.75])
    groups=np.searchsorted(thresholds,rarity,side='right');harm=[]
    for g in range(4):
        mask=groups==g;harm.append(dict(rarity_quartile=g,count=int(mask.sum()),
            harm_share=float(np.mean(pooled[mask]>base[mask])) if mask.any() else None))
    np.savez_compressed(root/'score_points.npz',**scored)
    gain=base-pooled
    result=dict(variant=v,world=world,law=truth['law'],makers=len(y),independent_max_error=error,
        ledger=roundtrip,primary_mean=float(gain.mean()),raw_gain_min=float(gain.min()),
        total_risk_ratio=float(pooled.sum()/base.sum()),harm_share=float(np.mean(pooled>base)),
        harm_by_rarity=harm,readers=summary,
        scope='one constructed population; makers are not independent worlds; no scientific acceptance')
    write(root/'SUMMARY.json',result)
    write(root/'RAW_INDEX.json',{p.name:digest(p) for p in root.iterdir() if p.is_file()})
    write(root/'world-cost.json',dict(cpu_seconds=time.process_time()-start))
    return result


def training(root,variants,phase,count,pulse=lambda **kw:None):
    """Retain every generated training population and selected maker identity.

    Label counts describe selected training makers. All generated labels, including
    unselected makers needed for public population summaries, are also recorded.
    """
    root=Path(root);root.mkdir(parents=True,exist_ok=False);x=[];y=[];index=[];generated=0
    if count<len(variants):raise ValueError('training budget must represent every design')
    for j,v in enumerate(variants):
        wanted=count//len(variants)+int(j<count%len(variants));remaining=wanted;world=0
        while remaining:
            p,t=makers.n1_world(v,['training',phase],world,pulse=pulse)
            folder=root/(v['id']+'-'+str(world));folder.mkdir()
            save_public(folder/'public_points.npz',p);np.savez_compressed(folder/'truth_points.npz',**t)
            take=min(remaining,len(t['y']));selected=rng('N1-training-selection',phase,v['id'],world).permutation(len(t['y']))[:take]
            xx=features(v,p)[selected];yy=np.asarray(t['y'])[selected]
            np.savez_compressed(folder/'selected_points.npz',maker=selected,x=xx,y=yy)
            x.extend(xx);y.extend(yy);generated+=len(t['y']);remaining-=take;world+=1
            index.append(dict(directory=folder.name,selected_labels=take,generated_labels=len(t['y']),
                files={f.name:digest(f) for f in folder.iterdir()}))
            pulse(phase='population-training',variant=v['id'],selected_labels=len(y),generated_labels=generated)
    write(root/'INDEX.json',dict(phase=phase,selected_labels=len(y),generated_labels=generated,populations=index))
    return np.asarray(x),np.asarray(y)


def fit_family(root,pulse=lambda **kw:None,variant=None,namespace=None):
    """Frozen fixed/fresh rivals with retained population training evidence.

    This expensive production entry point is gated by the independent resource
    pilot. Its presence does not authorize starting it or imply convergence.
    """
    root=Path(root);root.mkdir(parents=True,exist_ok=False);variants=[variant] if variant is not None else registry.variants('N1')
    phase=lambda value: value if namespace is None else [namespace,value]
    vx,vy=training(root/'validation',variants,phase('validation'),4096,pulse)
    np.savez_compressed(root/'validation.npz',x=vx,y=vy)
    fitted={};metadata={}
    for labels in (2048,8192):
        train_root=root/f'training-{labels}'
        x,y=training(train_root,variants,phase('fixed-'+str(labels)),labels,pulse)
        generated=read(train_root/'INDEX.json')['generated_labels']
        for fit in (0,1):
            for fresh in (False,True) if labels==8192 else (False,):
                name=f'direct-{labels}-fit{fit}'+('-fresh' if fresh else '')
                epoch_cache={};fresh_generated=[]
                def provider(epoch,offset,size):
                    if epoch_cache.get('epoch')!=epoch:
                        target=root/(name+'-epoch-'+str(epoch))
                        xx,yy=training(target,variants,phase([name,epoch]),labels,pulse)
                        epoch_cache.update(epoch=epoch,x=xx,y=yy)
                        fresh_generated.append(read(target/'INDEX.json')['generated_labels'])
                    return epoch_cache['x'][offset:offset+size],epoch_cache['y'][offset:offset+size]
                model=Direct(x,y,vx,vy,3,'N1' if namespace is None else canonical([namespace,'N1']).decode(),fit,fresh=provider if fresh else None,pulse=pulse)
                model.save(root/(name+'.npz'));fitted[name]=model
                metadata[name]=dict(model.metadata,selected_training_labels=model.metadata['generated_training_labels'],
                    all_generated_training_labels=generated+sum(fresh_generated),
                    unselected_future_labels_used_for_fit=False,
                    retained_training_population_index=str(train_root.name+'/INDEX.json'))
                write(root/(name+'-cost.json'),dict(cpu_seconds=model.cpu_seconds))
    write(root/'FITS.json',metadata)
    return fitted,metadata


def known_answers():
    # A no-record matched-information fixture, not a performance threshold at R=512.
    v=dict(registry.variants('N1')[0]);v['settings']=dict(v['settings'],makers=2,records=0)
    p=dict(menus=np.empty((2,0,3,4)),choices=np.empty((2,0),int),
        own=np.array([[[1,0,0,0],[0,0,0,0],[0,0,0,0]]]*2),
        other=np.array([[[1,0,0,0],[0,0,0,0],[0,0,0,0]]]*2),alias_pairs=None)
    pred,detail=makers.n1_infer(v,p)
    negative=float(np.max(np.abs(pred['law-marginal']-pred['population-only'])))
    menu=np.zeros((32,200));menu[:,0]=1
    positive,diag=checker_population.reconstruct(v,p,menu)
    expected=np.mean([np.array([np.exp(b),1,1])/(np.exp(b)+2) for b in (1,2,4)],axis=0)
    positive_error=float(np.max(np.abs(positive['law-marginal']-expected)))
    if max(negative,positive_error)>1e-12:raise ValueError('population known-answer control failed')
    return dict(passed=True,matched_information_max_error=negative,planted_single_type_max_error=positive_error,
                no_empirical_convergence_requirement=True)


def pilot(root,plan,pulse):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    write(root/'CONTROLS.json',known_answers())
    # True largest history/population/latent support. All three concentration
    # levels have the same support; diffuse culture avoids a sparse-case shortcut.
    v=next(v for v in registry.variants('N1') if v['settings']==dict(makers=10000,records=512,concentration=5,structure='drift-.05'))
    result=retained_world(root/'largest-world',v,'cost-pilot',0,schema,plan['canary_key'],pulse=pulse)
    write(root/'SUMMARY.json',dict(state='PILOT_VALID',family='N1',largest_variant=v,
        independent_max_error=result['independent_max_error'],makers=result['makers'],
        scientific_claim='none; one largest-cell cost/control pilot; no production admission',
        full_grid_profile=False,training_cost_profile=False))
