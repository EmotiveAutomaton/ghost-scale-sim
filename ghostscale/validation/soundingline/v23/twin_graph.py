"""Combine constructed maker exports into one shared, chronological world.

Only public fields enter this graph. Product-module admission remains separate
from executing an ordinary pooled choice baseline on the supported targets.
"""
from copy import deepcopy
from datetime import datetime,timedelta
from pathlib import Path
import hashlib
import numpy as np
from . import ledger,engine_bridge
from .common import canonical,read,write


def build(root,v,identity,schema):
    root=Path(root);tables={kind:{} for kind in schema.RECORDS if kind!='Provenance'};blobs={};query_ids=[]
    revisions=[];latest=ledger.BASE_TIME
    for folder in sorted(root.glob('maker-*')):
        public=read(folder/'PUBLIC.json');maker=public['maker'];local_identity=[*identity,maker]
        original=read(folder/'engine/reader.json');bundle=engine_bridge.native_view(original,schema)
        actors={maker:ledger._identity(local_identity,'maker')}
        for row in public['history']:
            actor=row.get('actor',row.get('maker',maker))
            actors.setdefault(actor,ledger._identity(local_identity,'actor',actor))
        replacements={old:ledger._identity(identity,'maker',actor) for actor,old in actors.items()}
        # Sessions retain their original identities; actor identity is shared.
        replacements[ledger._identity(local_identity,'artifact')]=ledger._identity(identity,'artifact')
        def replace(value):
            if isinstance(value,str):return replacements.get(value,value)
            if isinstance(value,list):return [replace(x) for x in value]
            if isinstance(value,dict):return {k:replace(x) for k,x in value.items()}
            return value
        for kind,rows in bundle['rows'].items():
            if kind=='Provenance':continue
            for original_row in rows:
                row=replace(deepcopy(original_row));key=row[schema.ID_FIELDS[kind]]
                if kind=='Maker':row['display_label']='Synthetic participant'
                if kind=='Artifact':
                    row['source_uri']='ghost://v23/'+ledger._identity(identity)
                    if key in tables[kind]:
                        row['maker_ids']=sorted(set(row['maker_ids'])|set(tables[kind][key]['maker_ids']))
                        row['revisions']=sorted(set(row['revisions'])|set(tables[kind][key]['revisions']))
                if kind=='Decision':
                    if row['context']['channel']=='query':query_ids.append(key)
                    else:latest=max(latest,datetime.fromisoformat(row['t_resolve']))
                if kind=='Revision':revisions.append(key)
                if kind not in ('Maker','Artifact') and key in tables[kind] and tables[kind][key]!=row:
                    raise ValueError('shared-world public identity collision')
                tables[kind][key]=row
        for key,value in bundle['blobs'].items():
            if key in blobs and blobs[key]!=value:raise ValueError('shared content hash collision')
            blobs[key]=value
    if not tables['Maker']:raise ValueError('empty whole-world graph')
    cutoff=latest+timedelta(seconds=1)
    for key in query_ids:
        row=tables['Decision'][key];row['t_open']=row['t_resolve']=cutoff.isoformat()
    for row in tables['Session'].values():row['t_end']=max(cutoff,datetime.fromisoformat(row['t_end'])).isoformat()
    # T3's observed article identifiers/chronology remain in each public context.
    # The graph is a testbed artifact; it does not invent latent discrete choices.
    rows={kind:list(values.values()) for kind,values in tables.items()};rows['Provenance']=[]
    for kind,table in tables.items():
        for row in table.values():
            provenance=schema.RECORDS['Provenance'].model_validate(dict(record_id=row[schema.ID_FIELDS[kind]],record_type=kind,
                adapter='ghost-v23-whole-twin',adapter_version='1',source_file_sha256=hashlib.sha256(canonical(row)).hexdigest(),created_at=ledger.BASE_TIME))
            rows['Provenance'].append(provenance.model_dump(mode='json'))
    result=dict(schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash(),transport=ledger.TRANSPORT,rows=rows,blobs=blobs)
    ledger.validate(result,schema);export=engine_bridge.export(result,schema)
    write(root/'WHOLE_WORLD_READER.json',export)
    write(root/'WORLD_GRAPH.json',dict(state='PUBLIC_WORLD_GRAPH_BUILT',makers=len(rows['Maker']),queries=len(query_ids),
        reader_hash=export['reader_hash'],cutoff=cutoff.isoformat(),evaluator_loaded=False,
        purpose='Shared maker identities and one future-query boundary; full original maker exports remain retained.'))
    return export,cutoff


def evaluate(root,v,identity,schema,pulse):
    from tompathy_engine.ghost.bridge import load_reader,reader_choices
    from tompathy_engine.models.backbone import Backbone
    root=Path(root);export,cutoff=build(root,v,identity,schema);rows,_=load_reader(root/'WHOLE_WORLD_READER.json')
    if v['family']=='T3':
        result=dict(state='TARGET_UNAVAILABLE',reason='The committed engine choice backbone does not infer unobserved revision purposes. No latent intent is fabricated as a chosen alternative.',
            engine_module_admission=False,reader_hash=export['reader_hash'])
        write(root/'ENGINE_BASELINE.json',result);return result
    choices=reader_choices(rows);pulse(phase='twin-engine-fitting',choices=len(choices))
    if not choices:raise ValueError('supported twin target has no observed history')
    model=Backbone.fit(choices,until=cutoff);names=sorted(next(r for r in rows if type(r).__name__=='Decision').options[0].features)
    predicted={}
    for row in rows:
        if type(row).__name__!='Decision' or row.taken is not None:continue
        x=np.array([[o.features[name] for name in names] for o in row.options])
        p=model.predict(row.maker_id,x)
        if not np.isfinite(p).all() or np.any(p<=0) or not np.isclose(np.sum(p),1):raise ValueError('invalid engine twin forecast')
        predicted[row.maker_id]=p
    # Evaluator labels are read only after the fitted model and forecasts exist.
    scores=[]
    for folder in sorted(root.glob('maker-*')):
        p=read(folder/'PUBLIC.json');t=read(folder/'TRUTH.json');key=ledger._identity(identity,'maker',p['maker'])
        from .checker import scores as independent_scores
        rebuilt=independent_scores(predicted[key][None,:],t['y'])
        scores.append(dict(maker=p['maker'],probabilities=predicted[key],scores={name:float(x[0]) for name,x in rebuilt.items()}))
    write(root/'ENGINE_BASELINE.json',dict(state='ENGINE_BASELINE_EXECUTED',implementation_hash=export['implementation_hash'],
        rows=scores,train_ids=list(model.train_ids),reader_hash=export['reader_hash'],engine_module_admission=False,
        target='Constructed handling choice' if v['family']=='T1' else 'Constructed move choice' if v['family']=='T2' else 'Constructed nominal rating; no ordered preference interpretation',
        information='Public numeric option features only; native latent support and evaluator labels excluded from fitting.'))
    return dict(state='ENGINE_BASELINE_EXECUTED',engine_module_admission=False,makers=len(scores))
