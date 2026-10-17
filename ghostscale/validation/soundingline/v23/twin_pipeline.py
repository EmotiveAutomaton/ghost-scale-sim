"""Whole-testbed twin execution, measured realism and reusable spike-in exports.

Empirical aggregates and their source bytes are required, never inferred from
commissioning prose. Complete worlds, not individual makers, are independent
replications. Native references and the separately pinned engine remain distinct.
"""
from collections import Counter,defaultdict
from datetime import timedelta
from pathlib import Path
import hashlib,hmac,json,sqlite3,time
import numpy as np
from . import checker,engine_bridge,fit_rivals,ledger,registry,twin_contract,twin_engineering,twin_ledger
from . import twin_writing,twin_chess,twin_wiki,twin_panel
from .common import Direct,canonical,digest,interval,read,write
from .pipeline import Stream
from .pipeline_new import score


def reference(plan):
    path=Path(plan['empirical_reference'])
    if digest(path)!=plan['empirical_reference_sha256']:raise ValueError('empirical aggregates changed')
    record=read(path);family=plan['family']
    twin_contract.require_realism(twin_contract.realism(family,record,record['aggregates']))
    if digest(plan['empirical_source'])!=record['source']['sha256']:
        raise ValueError('empirical source bytes do not authenticate aggregates')
    if record['source'].get('kind')=='wikipedia_content_sha1':
        source=read(plan['empirical_source']);identity=record['source']
        anchors=source.get('article_anchors',[])
        if source.get('source_identity_kind')!='Wikipedia committed content SHA-1, not Git':
            raise ValueError('Wikipedia source identity kind differs')
        if not any(a.get('pageid')==identity['pageid'] and a.get('latest_visible_revision')==identity['revision_id']
                and a.get('sha1')==identity['revision'] for a in anchors):
            raise ValueError('Wikipedia content revision anchor differs')
        if canonical(source.get('aggregates'))!=canonical(record['aggregates']):
            raise ValueError('Wikipedia aggregate source differs')
        import re
        hashes=source.get('request_outputs_sha256',{})
        if not hashes or not all(re.fullmatch('[0-9a-f]{64}',h) for h in [source.get('intake_terminal_sha256',''),*hashes.values()]):
            raise ValueError('Wikipedia snapshot hash chain missing')
    return record


def training(root,variants,phase,count,shape,pulse):
    """Retain complete generated worlds and select makers without test labels.

    Training worlds have their own namespace. All generated outcomes, including
    those not selected for the fit, stay in the accounting and raw archive.
    """
    from .common import rng
    root=Path(root);root.mkdir(parents=True,exist_ok=False);x=[];y=[];manifest=[]
    if count<len(variants):raise ValueError('training must cover every twin design')
    for index,v in enumerate(variants):
        remaining=count//len(variants)+int(index<count%len(variants));world=0
        while remaining:
            folder=root/(v['id']+'-'+str(world));folder.mkdir()
            size=v['settings'].get('makers',v['settings'].get('players'))
            selected=set(map(int,rng('twin-training-selection',v['id'],phase,world).permutation(size)[:min(size,remaining)]))
            generated=0;stream=Stream(folder/'training_population_points.jsonl.gz')
            try:
                for p,t in population(v,['rival-training',phase],world,shape,folder,pulse):
                    stream.write(dict(public=p,evaluator=t,used_for_fit=p['maker'] in selected));generated+=1
                    if p['maker'] in selected:x.append(features(v,p));y.append(t['y'])
            finally:stream.close()
            if generated!=size:raise ValueError('incomplete twin training population')
            manifest.append(dict(variant=v['id'],world=world,selected_labels=len(selected),generated_labels=generated,
                raw_sha256=digest(folder/'training_population_points.jsonl.gz')))
            remaining-=len(selected);world+=1
    write(root/'INDEX.json',dict(populations=manifest,selected_labels=len(y),generated_labels=sum(r['generated_labels'] for r in manifest)))
    return np.asarray(x),np.asarray(y)


def run_fits(root,plan,pulse):
    ref=reference(plan);root=Path(root);root.mkdir(parents=True,exist_ok=False);family=plan['family']
    variants=registry.variants(family);groups=('3','5') if family=='T2' else ('shared',);metadata={}
    fitroot=root/'fits';fitroot.mkdir()
    for group in groups:
        members=[v for v in variants if family!='T2' or str(v['settings']['branching'])==group]
        if not members:raise ValueError('empty twin class group')
        classes=int(group) if family=='T2' else 4 if family=='T1' else 3
        builder=lambda folder,label,count:training(folder,members,[family,group,label],count,ref['aggregates'],pulse)
        models,meta=fit_rivals.fit(fitroot/group,family,members,None,features,classes,pulse,'twin-'+family+'-'+group,training_builder=builder)
        for name,model in models.items():
            identity=group+'-'+name;model.save(fitroot/(identity+'.npz'));metadata[identity]=dict(meta[name],class_group=group)
    write(fitroot/'FITS.json',metadata)
    write(root/'SUMMARY.json',dict(state='FITS_VERIFIED',family=family,models=len(metadata),
        validation_independently_scored=True,discovery_exposure=False,training_complete_populations_retained=True))


def population(v,phase,world,shape,folder,pulse):
    """Stream complete assigned makers. Wikipedia's global chronology is spooled."""
    f,s=v['family'],v['settings']
    if f=='T1':
        if len(shape['records_per_maker'])!=s['makers'] or len(shape['sessions_per_maker'])!=s['makers']:
            raise ValueError('writing shape does not cover every declared maker')
        for maker in range(s['makers']):yield twin_writing.maker(v,phase,world,maker,shape)
    elif f=='T2':
        levels=twin_chess.solve(s['depth'],s['branching'],str((v['id'],phase,world)))
        twin_chess.verify_tree(levels,s['branching'])
        for maker in range(s['players']):
            yield twin_chess.player(v,phase,world,maker,clock_quantiles=shape['move_clock_quantiles_s'])
    elif f=='T4':
        for maker in range(s['makers']):yield twin_panel.rater(v,phase,world,maker,semantics=shape['rating_semantics'])
    else:
        # A disk index retains shared article chronology without a many-million
        # row Python list. It contains only this constructed world's records.
        db=sqlite3.connect(folder/'article_spool.sqlite')
        try:
            db.execute('CREATE TABLE revisions (sequence INTEGER PRIMARY KEY, maker INTEGER, public TEXT, evaluator TEXT)')
            for index,(p,t) in enumerate(twin_wiki.world(v,phase,world)):
                db.execute('INSERT INTO revisions VALUES (?,?,?,?)',(index,p['maker'],canonical(p).decode(),canonical(t).decode()))
                if index%1024==0:db.commit();pulse(phase='twin-article-construction',records=index)
            db.commit();db.execute('CREATE INDEX maker_records ON revisions(maker,sequence)');db.commit()
            for maker in range(s['makers']):
                rows=list(db.execute('SELECT public,evaluator FROM revisions WHERE maker=? ORDER BY sequence',(maker,)))
                p=[json.loads(a) for a,_ in rows];t=[json.loads(b) for _,b in rows]
                if not p:raise ValueError('assigned editor has no retained history')
                yield dict(maker=maker,history=p),dict(marks=t,y=[t[0]['purpose']])
        finally:db.close()


def features(v,p):
    """Fixed public summary representation; its information loss is explicit.

    This is a direct diagnostic, never an exact-ceiling/headroom certificate.
    The native reference retains the complete histories and shared-world state.
    """
    rows=p['history'];f=v['family'];out=np.zeros(128)
    out[0]=len(rows)
    if f=='T1':
        if rows:
            out[1:5]=np.bincount([r['action'] for r in rows],minlength=4)/len(rows)
            out[5:9]=np.mean([np.asarray(r['options'])[r['selected']] for r in rows],axis=0)
            out[9:13]=np.mean([r['edit_delta'] for r in rows],axis=0)
            out[13:17]=[np.mean([r[k] for r in rows]) for k in ('corrected','latency_bin','phase','momentum')]
        out[17:21]=p['statement'];q=p['query'];x=np.asarray(q['options']).ravel()
        if len(x)>64:raise ValueError('writing option shape exceeds declared representation')
        out[21:21+len(x)]=x;out[85:89]=p['handling_base'];out[89:93]=[q['selected'],q['phase'],q['momentum'],len(q['options'])]
    elif f=='T2':
        own=[r for r in rows if r['actor']==p['maker']]
        if own:
            out[1:6]=np.bincount([r['taken'] for r in own],minlength=5)/len(own)
            out[6:9]=[np.mean([r['values'][r['taken']] for r in own]),np.mean([r['style_features'][r['taken']] for r in own]),np.mean([r['clock_s'] for r in own])]
        q=p['query'];out[10:10+len(q['values'])]=q['values'];out[15:15+len(q['style_features'])]=q['style_features'];out[20]=q['clock_s']
    elif f=='T3':
        if rows:
            out[1:5]=np.mean([r['diff'] for r in rows],axis=0);out[5:9]=np.mean([r['state'] for r in rows],axis=0)
            cues=[r['summary'] for r in rows if r['summary'] is not None]
            if cues:out[9:12]=np.bincount(cues,minlength=3)/len(cues)
            out[12]=np.mean([r['reverted'] for r in rows]);out[13]=len({r['article'] for r in rows})
    else:
        if rows:
            out[1:4]=np.bincount([r['rating'] for r in rows],minlength=3)/len(rows)
            out[4:8]=np.mean([np.asarray(r['features'])*r['rating'] for r in rows],axis=0)
        out[8:12]=p['query'];out[12:18]=p['reason'];out[18]=p['group']
        out[19]=p['semantics']=='ordinal_3'
    # Exclude all generating-effect, hidden-type and dirty-label declarations.
    return out


def quantiles(values):
    if not values:raise ValueError('missing constructed timing observations')
    probabilities=[0,.1,.25,.5,.75,.9,1]
    return dict(probabilities=probabilities,values=np.quantile(values,probabilities).tolist())


class Shape:
    def __init__(self,family):
        self.family=family;self.counts=[];self.sessions=[];self.options=[];self.latency=[];self.keys=[]
        self.labels=Counter();self.articles=defaultdict(set);self.item_raters=defaultdict(set);self.total=0;self.cues=0;self.reverts=0;self.games=[]
    def add(self,p,t):
        rows=p['history'];self.counts.append(len(rows));self.total+=len(rows)
        for row in rows:
            if self.family=='T1':
                self.options.append(len(row['options']));self.latency.append(row['latency_s']);self.keys.append(row['keystrokes']);self.labels[row['action']]+=1
            elif self.family=='T2':self.latency.append(row['clock_s'])
            elif self.family=='T3':
                self.articles[row['article']].add(row['maker']);self.cues+=row['summary'] is not None;self.reverts+=row['reverted']
            else:self.item_raters[row['item']].add(p['maker']);self.labels[row['rating']]+=1
        if self.family=='T1':self.sessions.append(len({r['session'] for r in rows}))
        if self.family=='T2':self.games.append(t['complete_games'])
        if self.family=='T4':self.semantics=p['semantics']
    def result(self):
        n=len(self.counts)
        if self.family=='T1':return dict(makers=n,sessions=sum(self.sessions),decisions=self.total,records_per_maker=self.counts,
            sessions_per_maker=self.sessions,options_per_decision=dict(Counter(self.options)),latency_quantiles_s=quantiles(self.latency),
            keystroke_quantiles=quantiles(self.keys),handling_marginals=[self.labels[i]/self.total for i in range(4)])
        if self.family=='T2':return dict(players=n,games_per_player=self.games,move_clock_quantiles_s=quantiles(self.latency),
            moves_per_game=[count/games for count,games in zip(self.counts,self.games,strict=True)])
        if self.family=='T3':return dict(editors=n,edits_per_editor=self.counts,editors_per_article=sorted(len(x) for x in self.articles.values()),
            summary_coverage=self.cues/self.total,revert_fraction=self.reverts/self.total)
        return dict(raters=n,items=len(self.item_raters),ratings_per_rater=self.counts,
            raters_per_item=[len(x) for _,x in sorted(self.item_raters.items())],rating_semantics=self.semantics,
            rating_marginals=[self.labels[i]/self.total for i in range(3)])


def evaluate_world(folder,v,phase,world,reference_,schema,key,pulse,fitted=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    shape=Shape(v['family']);gains=[];scores=[];posteriors=[];counts=[];exports=[]
    stream=Stream(folder/'maker_scores_points.jsonl.gz')
    try:
        for p,t in population(v,phase,world,reference_['aggregates'],folder,pulse):
            maker=p['maker'];unit=folder/f'maker-{maker:05d}';unit.mkdir()
            pred,detail,validity=twin_engineering.evaluate(v,p)
            identity=[v['family'],v['id'],phase,world,maker]
            canary=hmac.new(bytes.fromhex(key),canonical(identity),hashlib.sha256).hexdigest()
            bundle=twin_ledger.emit(v,p,schema,identity,world_identity=identity[:-1]);ledger.validate(bundle,schema,canaries=[canary])
            decoded=twin_ledger.decode(v,bundle,schema)
            if canonical(decoded)!=canonical(p):raise ValueError('whole-twin ledger loses public evidence')
            for name,model in (fitted or {}).items():pred[name]=model.predict(features(v,p)[None,:])[0]
            measured={name:score(prob,t) for name,prob in pred.items()}
            mechanism='joint' if v['family']=='T3' else 'individual'
            baseline='summary-only' if v['family']=='T3' else 'group-population' if v['family']=='T4' else 'population'
            a,b=measured[baseline]['log_loss'],measured[mechanism]['log_loss']
            gains.append(a-b if a is not None and b is not None else None)
            write(unit/'PUBLIC.json',p);write(unit/'TRUTH.json',dict(t,canary=canary))
            write(unit/'PREDICTION.json',dict(probabilities=pred,detail=detail,scores=measured))
            write(unit/'VALIDITY.json',validity)
            bridge=engine_bridge.verify_export(unit/'engine',v,bundle,schema,canaries=[canary]);exports.append(dict(maker=maker,reader_hash=bridge['reader_hash']))
            stream.write(dict(maker=maker,scores=measured,baseline=baseline,mechanism=mechanism))
            shape.add(p,t)
            if v['family']=='T4':posteriors.append(detail['type_posterior']);counts.append(detail['unique_ratings'])
            pulse(phase='whole-twin',variant=v['id'],world=world,maker=maker+1)
    finally:stream.close()
    realism=twin_contract.realism(v['family'],reference_,shape.result());twin_contract.require_realism(realism)
    write(folder/'REALISM.json',realism)
    from .twin_graph import evaluate as engine_baseline
    engine=engine_baseline(folder,v,[v['family'],v['id'],phase,world],schema,pulse)
    if posteriors:write(folder/'FLOOD.json',twin_panel.population(posteriors,counts))
    write(folder/'SPIKE_INS.json',dict(world=world,namespace=phase,synthetic=True,reader_exports=exports,
        evaluator_separate=True,public_records='maker-*/engine/reader.json',evaluator_records='maker-*/TRUTH.json',
        whole_world_reader='WHOLE_WORLD_READER.json',engine_module_admission=False,prohibition='Do not train a reader on evaluator sidecars.'))
    result=dict(variant=v,world=world,makers=len(gains),world_gain=float(np.mean(gains)) if all(g is not None for g in gains) else None,
        realism_sha256=digest(folder/'REALISM.json'),spike_ins_sha256=digest(folder/'SPIKE_INS.json'),
        direct_representation='128 public summary features; not an information-matched ceiling comparison',
        engine_baseline=engine,engine_scope='Pinned pooled choice backbone on supported observed targets; separate product-module gates remain required')
    write(folder/'SUMMARY.json',result);return result


def run(root,plan,pulse):
    ref=reference(plan);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    variants=registry.variants(plan['family']);pilot=plan['task']=='twin-profile'
    if pilot:
        variants=sorted(variants,key=lambda v: (v['settings'].get('makers',v['settings'].get('players',1))*
            v['settings'].get('max_records',v['settings'].get('games',v['settings'].get('items',1))),
            v['settings'].get('depth',1)*v['settings'].get('branching',1)),reverse=True)
    if not pilot and plan.get('worlds')!=16:raise ValueError('fixed twin replication allocation changed')
    fitted={};fitmeta={}
    if not pilot:
        fitroot=Path(plan['fit_directory'])
        for name,expected in plan['fit_hashes'].items():
            if digest(fitroot/name)!=expected:raise ValueError('twin fitted rival changed')
        fitmeta=read(fitroot/'FITS.json')
    summaries=[];costs=[]
    for v in variants:
        start=time.process_time();worlds=[]
        if not pilot:
            group=str(v['settings']['branching']) if v['family']=='T2' else 'shared'
            fitted={name:Direct.load(fitroot/(name+'.npz')) for name,row in fitmeta.items() if row['class_group']==group}
            if len(fitted)!=6:raise ValueError('six fixed/fresh twin rivals required for this target support')
        from .partition import world_indices
        for world in ([0] if pilot else world_indices(plan)):
            worlds.append(evaluate_world(root/v['id']/f'world-{world:04d}',v,'cost-profile' if pilot else 'twin-discovery',world,ref,schema,plan['canary_key'],pulse,fitted))
        gains=[row['world_gain'] for row in worlds]
        estimate=interval(gains,[v['id'],'detectability']) if all(g is not None for g in gains) else dict(finite=False,mean=None,lower=None,upper=None,se=None)
        summaries.append(dict(variant=v,worlds=worlds,gain=estimate,
            detected=bool(estimate['finite'] and estimate['lower']>.02),
            interpretation='fixed-grid constructed detectability; no post-hoc smaller-effect search',real_shape_comparisons=True))
        costs.append(dict(variant=v['id'],cpu_seconds=time.process_time()-start,whole_worlds=len(worlds)))
        write(root/'profile-cost.json',dict(profile=costs),immutable=False)
    if pilot:
        started=time.process_time()
        # One full population per setting, retaining all generated training data.
        # T2 class supports are separately fitted and timed.
        profiles=[]
        for group in (('3','5') if plan['family']=='T2' else ('shared',)):
            members=[v for v in variants if plan['family']!='T2' or str(v['settings']['branching'])==group]
            folder=root/('training-profile-'+group);begin=time.process_time()
            x,y=training(folder,members,'whole-twin-cost-profile',max(128,len(members)),ref['aggregates'],pulse)
            generation=time.process_time()-begin
            model=Direct(x,y,x,y,int(group) if plan['family']=='T2' else 4 if plan['family']=='T1' else 3,
                'twin-training-profile-'+plan['family']+'-'+group,0,max_epochs=2,pulse=pulse)
            profiles.append(dict(class_group=group,training_generation_cpu_seconds=generation,training_labels=len(y),
                fit_cpu_seconds=model.cpu_seconds,fit_epochs=model.metadata['epochs']))
        write(root/'pilot-cost.json',dict(profile=costs,training=profiles))
    write(root/'SUMMARY.json',dict(state='PILOT_VALID' if pilot else 'TWIN_BATCH_CHECKED',family=plan['family'],variants=summaries,
        real_shape_reference_sha256=plan['empirical_reference_sha256'],engine_scientific_admission=False,
        limitation='Native finite references and public summary direct rivals. Product modules retain separate gates.'))


def reconstruct(root,plan):
    """Independent proper scoring and whole-world replication, no maker pseudo-n."""
    from .family import rows
    from .partition import world_indices
    root=Path(root);checked={};items=read(root/'SUMMARY.json')['variants']
    if [item['variant']['id'] for item in items]!=[v['id'] for v in registry.variants(plan['family'])]:
        raise ValueError('incomplete twin variant coverage')
    for item in items:
        v=item['variant'];means=[]
        if [world['world'] for world in item['worlds']]!=world_indices(plan):
            raise ValueError('incomplete twin world coverage')
        for world in item['worlds']:
            folder=root/v['id']/f"world-{world['world']:04d}";gains=[]
            for unit in sorted(folder.glob('maker-*')):
                p=read(unit/'PUBLIC.json');t=read(unit/'TRUTH.json');saved=read(unit/'PREDICTION.json')
                native=twin_engineering.evaluate(v,p)[0]
                rebuilt={name:checker.scores(np.asarray(prob).reshape(-1,len(prob[0])),t['y']) for name,prob in saved['probabilities'].items()}
                for name,prob in native.items():
                    if not np.allclose(prob,saved['probabilities'][name],atol=1e-12,rtol=0):raise ValueError('retained twin forecast differs')
                for name,metrics in rebuilt.items():
                    for key,values in metrics.items():
                        value=float(np.mean(values));actual=saved['scores'][name][key]
                        if not (actual is None and not np.isfinite(value)) and not np.isclose(value,actual,atol=1e-11,rtol=0):
                            raise ValueError('retained twin proper score differs')
                rival='summary-only' if v['family']=='T3' else 'group-population' if v['family']=='T4' else 'population'
                mechanism='joint' if v['family']=='T3' else 'individual'
                delta=rebuilt[rival]['log_loss']-rebuilt[mechanism]['log_loss'];gains.append(float(np.mean(delta)))
            if len(gains)!=world['makers']:raise ValueError('incomplete twin maker coverage')
            observed=float(np.mean(gains)) if np.isfinite(gains).all() else None
            if observed is None and world['world_gain'] is not None or observed is not None and not np.isclose(observed,world['world_gain'],atol=1e-11,rtol=0):
                raise ValueError('whole-world twin aggregate differs')
            means.append(observed)
        checked[v['id']]=dict(worlds=len(means),independent_unit='whole testbed',gains=means)
    return dict(state='passed',variants=checked,engine_module_admission=False)
