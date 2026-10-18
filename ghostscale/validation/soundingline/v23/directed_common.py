"""October 18 finite analytic studies; separate public/evaluator evidence and costs."""
from pathlib import Path
import math,time
import numpy as np
from . import ledger
from .common import canonical,digest,interval,read,write,rng
from .pipeline import Stream
from .family import rows

FAMILIES=('N16','N16b','N17','N18')
CAPS={'N16':43200,'N16b':10800,'N17':10800,'N18':5400}

def model_for(family):
    import importlib
    return importlib.import_module('.'+{'N16':'directed_probes','N16b':'probed_falsifier',
        'N17':'lean_taste','N18':'play_information'}[family],__package__)

def selected(model,plan):
    all_variants=model.variants()
    ids=plan.get('directed_design',{}).get('variant_ids',[v['id'] for v in all_variants])
    if len(ids)!=len(set(ids)) or not set(ids)<=set(v['id'] for v in all_variants):raise ValueError('invalid directed settings')
    return [v for v in all_variants if v['id'] in ids]

def summarize(model,plan,blocks):
    summaries=[]
    for v in selected(model,plan):
        entries=blocks[v['id']];worlds=sorted(entries)
        keys=set(entries[worlds[0]])
        if any(set(entries[w])!=keys for w in worlds):raise ValueError('directed metric coverage')
        metrics={k:dict(world_values=[entries[w][k] for w in worlds],
            interval=interval([entries[w][k] for w in worlds],[model.FAMILY,v['id'],k])) for k in sorted(keys)}
        summaries.append(dict(variant=v,world_indices=worlds,metrics=metrics))
    result=dict(state='CONDITIONAL_BATCH_CHECKED',family=model.FAMILY,variants=summaries,
        selection_eligible=False,miniature_architecture_untested=True,
        inference_unit='complete reliability/noise law block; never cases',
        reader_roster=model.READERS,directed_design=plan.get('directed_design',{}))
    if hasattr(model,'study_readout'):result['study_readout']=model.study_readout(summaries)
    return result

def freeze(model,plan,pulse):
    return model.fit(plan,pulse) if hasattr(model,'fit') else dict(family=model.FAMILY,
        reader_roster=model.READERS,estimated_parameters=False,
        method='fixed analytic definitions; no learned-rival fit is claimed',
        directed_design=plan.get('directed_design',{}))

def run(root,plan,pulse):
    model=model_for(plan['family']);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    controls=model.controls();write(root/'CONTROLS.json',controls)
    if not controls.get('passed'):raise ValueError('directed controls failed')
    pilot=plan['task']=='conditional-pilot';fitting=plan['task']=='conditional-fits'
    start=time.process_time()
    if pilot or fitting:fixed=freeze(model,plan,pulse)
    else:
        for name,value in plan['fit_hashes'].items():
            if digest(Path(plan['fit_directory'])/name)!=value:raise ValueError('directed fitted definitions changed')
        fixed=read(Path(plan['fit_directory'])/'FROZEN.json')
    fit_seconds=time.process_time()-start
    if fitting:
        write(root/'fits/FROZEN.json',fixed)
        write(root/'SUMMARY.json',dict(state='FITS_VERIFIED',family=model.FAMILY,
            fitted_definition=fixed,discovery_exposure=False));return
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    blocks={};profile=[]
    worlds=[15] if pilot else plan.get('world_indices',list(range(plan.get('worlds',16))))
    cases=1 if pilot else 64
    for v in selected(model,plan):
        start=time.process_time();folder=root/v['id'];folder.mkdir();blocks[v['id']]={}
        streams={kind:Stream(folder/(kind+'_points.jsonl.gz')) for kind in ('public','truth','evidence','ledger')}
        try:
            for w in worlds:
                metrics=[]
                for c in range(cases):
                    identity=[model.FAMILY,v['id'],'cost-pilot' if pilot else 'discovery',w,c]
                    public,truth,evidence,values=model.case(v,identity[2],w,c,fixed,plan.get('directed_design',{}))
                    emission=ledger.Emission(schema,identity)
                    emission.decision('public',0,[0],0,observed=public);bundle=emission.finish()
                    ledger.validate(bundle,schema)
                    decoded=bundle['rows']['Decision'][0]['context']['observed']
                    if canonical(public)!=canonical(decoded):raise ValueError('directed ledger round trip')
                    if any(not math.isfinite(float(x)) for x in values.values()):raise ValueError('nonfinite directed score')
                    for kind,payload in [('public',dict(public=public)),('truth',dict(truth=truth)),
                        ('evidence',dict(evidence=evidence,metrics=values)),('ledger',dict(ledger=bundle))]:
                        streams[kind].write(dict(identity=identity,**payload))
                    metrics.append(values)
                keys=metrics[0]
                blocks[v['id']][w]={k:float(np.mean([m[k] for m in metrics])) for k in keys}
                # N16b's decision uses within-block cases, never treats them as laws.
                if hasattr(model,'block_readout'):blocks[v['id']][w].update(model.block_readout(v,metrics))
                pulse(phase='directed-conditional',family=model.FAMILY,variant=v['id'],world=w,cases=cases)
        finally:
            for stream in streams.values():stream.close()
        profile.append(dict(variant=v['id'],cases=cases*len(worlds),cpu_seconds=time.process_time()-start))
    write(root/('pilot-cost.json' if pilot else 'batch-cost.json'),dict(profile=profile,fit_cpu_seconds=fit_seconds))
    write(root/'WORLD_BLOCKS.json',{v:{str(w):x for w,x in b.items()} for v,b in blocks.items()})
    result=summarize(model,plan,blocks)
    if pilot:result.update(state='PILOT_VALID',discovery_exposure=False)
    write(root/'SUMMARY.json',result)

def estimate(packet,design=None):
    packet=Path(packet);plan=read(packet/'PLAN.json');model=model_for(plan['family'])
    variants=selected(model,dict(plan,directed_design=design or {}));ids={v['id'] for v in variants}
    profiles=[read(packet/label/'pilot-cost.json') for label in ('original','replay')]
    for p in profiles:
        if not ids<=set(r['variant'] for r in p['profile']):raise ValueError('incomplete directed cost coverage')
    # Two complete passes, independent retained-data review, 1/3 timing margin.
    worlds=(design or {}).get('worlds',16)
    batch=max(300,4*max(sum(max(r['cpu_seconds'],.001)/r['cases']*worlds*64
        for r in p['profile'] if r['variant'] in ids) for p in profiles))
    fits=max(60,4*max(p['fit_cpu_seconds'] for p in profiles))
    from .partition import layout
    split=layout(worlds,batch,assembly_cap=max(300,batch/4)) if batch>14400 else None
    return dict(fits=fits,batch=split['total_cpu_cap'] if split else batch,
        unpartitioned_batch=batch,partition_layout=split,two_burnins=False)

def reconstruct(root,plan,pulse=lambda:None):
    model=model_for(plan['family']);root=Path(root)
    fixed=read(Path(plan['fit_directory'])/'FROZEN.json');blocks={};count=0
    for name,value in plan['fit_hashes'].items():
        if digest(Path(plan['fit_directory'])/name)!=value:raise ValueError('directed fit binding')
    worlds=plan.get('world_indices',list(range(plan.get('worlds',16))))
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    for v in selected(model,plan):
        folder=root/v['id'];seen=set();by_world={w:[] for w in worlds}
        for p,t,e,l in zip(*(rows(folder/(s+'_points.jsonl.gz')) for s in ('public','truth','evidence','ledger')),strict=True):
            identity=p['identity'];w,c=identity[-2:]
            if any(item['identity']!=identity for item in (t,e,l)) or identity[:3]!=[model.FAMILY,v['id'],'discovery']:
                raise ValueError('directed identity mismatch')
            if (w,c) in seen or w not in worlds or not 0<=c<64:raise ValueError('directed duplicate/coverage')
            seen.add((w,c));ledger.validate(l['ledger'],schema)
            if canonical(l['ledger']['rows']['Decision'][0]['context']['observed'])!=canonical(p['public']):raise ValueError('directed public transport mismatch')
            values=model.check(v,p['public'],t['truth'],e['evidence'],fixed,plan.get('directed_design',{}))
            if set(values)!=set(e['metrics']) or any(not math.isclose(float(x),e['metrics'][k],abs_tol=1e-10,rel_tol=1e-10) for k,x in values.items()):
                raise ValueError('directed independent score mismatch')
            by_world[w].append(values);count+=1
            if count%64==0:pulse()
        if seen!={(w,c) for w in worlds for c in range(64)}:raise ValueError('directed missing cases')
        blocks[v['id']]={}
        for w,metrics in by_world.items():
            blocks[v['id']][w]={k:math.fsum(m[k] for m in metrics)/len(metrics) for k in metrics[0]}
            if hasattr(model,'block_readout'):blocks[v['id']][w].update(model.block_readout(v,metrics))
    # JSON object keys are strings even when a readout indexes a budget by int.
    # Normalize only transport representation; no numeric result is changed.
    import json
    rebuilt=json.loads(canonical(summarize(model,plan,blocks)));saved=read(root/'SUMMARY.json')
    def compare(a,b):
        if isinstance(a,dict):
            if set(a)!=set(b):raise ValueError('directed summary keys')
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            if len(a)!=len(b):raise ValueError('directed summary length')
            for x,y in zip(a,b,strict=True):compare(x,y)
        elif isinstance(a,float):
            if not math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10):raise ValueError('directed aggregate differs')
        elif a!=b:raise ValueError('directed summary differs')
    compare(rebuilt,saved)
    # Independent bootstrap implementation, same prescribed resample identities.
    for item in saved['variants']:
        for name,metric in item['metrics'].items():
            vals=np.asarray(metric['world_values']);ix=rng('bootstrap',[model.FAMILY,item['variant']['id'],name]).integers(len(vals),size=(4096,len(vals)))
            means=np.sum(vals[ix],axis=1)/len(vals)
            lo,hi=np.percentile(means,[2.5,97.5])
            if not np.allclose([lo,hi],[metric['interval']['lower'],metric['interval']['upper']],atol=1e-10,rtol=0):raise ValueError('directed interval differs')
    return dict(passed=True,cases=count,law_blocks=worlds,all_raw_scores_rebuilt=True,
        law_block_intervals_independently_checked=True,canonical_ledger_round_trip=True,
        scientific_gate=saved.get('study_readout',{}))

def assemble(root,parts,pulse):
    import shutil
    from . import partition,runtime
    from .family import verify_packet
    root=Path(root);plan=read(root/'PLAN.json');model=model_for(plan['family'])
    if list(parts)!=partition.part_paths(root,plan):raise ValueError('directed partition paths')
    for i,part in enumerate(parts):
        child,_=verify_packet(part);partition.verify_part_binding(root,plan,part,child,i)
        for key in ('family','batch','worlds','cases','variants','fit_hashes','sources','directed_design'):
            if child.get(key)!=plan.get(key):raise ValueError('directed partition definition differs')
        pulse()
    for label in ('original','replay'):
        dest=root/label;dest.mkdir();blocks={v['id']:{} for v in selected(model,plan)}
        for part in parts:
            saved=read(part/label/'WORLD_BLOCKS.json')
            for variant,worlds in saved.items():
                for world,values in worlds.items():
                    if int(world) in blocks[variant]:raise ValueError('overlapping directed partition')
                    blocks[variant][int(world)]=values
        if any(sorted(b)!=list(range(plan['worlds'])) for b in blocks.values()):raise ValueError('incomplete directed partitions')
        for variant in blocks:
            folder=dest/variant;folder.mkdir()
            for kind in ('public','truth','evidence','ledger'):
                name=kind+'_points.jsonl.gz'
                with (folder/name).open('xb') as output:
                    for part in parts:
                        with (part/label/variant/name).open('rb') as source:shutil.copyfileobj(source,output)
            pulse()
        control_hash=digest(parts[0]/label/'CONTROLS.json')
        if any(digest(p/label/'CONTROLS.json')!=control_hash for p in parts):raise ValueError('directed controls changed')
        shutil.copyfile(parts[0]/label/'CONTROLS.json',dest/'CONTROLS.json')
        write(dest/'WORLD_BLOCKS.json',{v:{str(w):x for w,x in b.items()} for v,b in blocks.items()})
        write(dest/'SUMMARY.json',summarize(model,plan,blocks))
    write(root/'REPLAY.json',runtime.compare(root/'original',root/'replay'))
    review=reconstruct(root/'original',plan,pulse);write(root/'NUMERICAL_REVIEW.json',review)
    return dict(state='verified',at=runtime.now(),plan_sha256=digest(root/'PLAN.json'),
        replay_sha256=digest(root/'REPLAY.json'),summary_sha256=digest(root/'original/SUMMARY.json'),
        partitions={p.name:digest(p/'COMPLETE.json') for p in parts})
