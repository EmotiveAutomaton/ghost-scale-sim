"""Derived reader export for the committed ToMpathy engine, without sidecars.

The native Ghost transport remains immutable. This explicit boundary projection
supplies observable option features to the ordinary engine baseline. It does not
give that baseline the native reader's latent law menu or scientific admission.
"""
from copy import deepcopy
import hashlib,json,math
from . import ledger
from .common import canonical,write


def numeric_features(value,prefix='option'):
    if isinstance(value,bool):return {prefix:float(value)}
    if isinstance(value,(int,float)):
        if not math.isfinite(value):raise ValueError('nonfinite observed option feature')
        return {prefix:float(value)}
    if value is None:return {}
    if isinstance(value,list):items=enumerate(value)
    elif isinstance(value,dict):items=sorted(value.items())
    else:raise ValueError('only constructed numeric option content is supported')
    out={}
    for key,child in items:out.update(numeric_features(child,prefix+'.'+str(key)))
    return out


def export(bundle,schema,*,canaries=()):
    from tompathy_engine.admission import implementation_hash
    from tompathy_engine.ledger.ids import digest as engine_digest
    from tompathy_engine.ledger.schema import schema_hash
    from tompathy_engine.ghost.bridge import reject_truth_fields
    ledger.validate(bundle,schema,canaries=canaries)
    if schema_hash()!=bundle['schema_hash']:raise ValueError('engine/native schema mismatch')
    rows=deepcopy(bundle['rows']);names=set()
    for decision in rows['Decision']:
        decision['context']={'observed_context':decision['context']}
        for option in decision['options']:
            features=numeric_features(json.loads(bundle['blobs'][option['content_ref']]))
            features['menu_position_control']=float(option['position'])
            option['features']=features;names.update(features)
    for decision in rows['Decision']:
        for option in decision['options']:option['features']={name:option['features'].get(name,0.) for name in sorted(names)}
    for event in rows['ContextEvent']:event['payload']={'observation':event['payload']}
    by_id={(kind,row[schema.ID_FIELDS[kind]]):row for kind,table in rows.items() if kind!='Provenance' for row in table}
    for provenance in rows['Provenance']:
        source=by_id[(provenance['record_type'],provenance['record_id'])]
        provenance['source_file_sha256']=hashlib.sha256(canonical(source)).hexdigest()
    result=dict(ledger_schema_hash=schema_hash(),implementation_hash=implementation_hash(),
        records=[dict(type=kind,record=row) for kind,table in rows.items() for row in table],texts=deepcopy(bundle['blobs']))
    reject_truth_fields(result)
    if any(token.encode() in canonical(result) for token in canaries):raise ValueError('canary reached engine export')
    result['reader_hash']=engine_digest(result)
    return result


def native_view(reader,schema):
    """Recover native public observations, preserving every observed field."""
    rows={kind:[] for kind in schema.RECORDS}
    for item in reader['records']:
        row=deepcopy(item['record']);kind=item['type']
        if kind=='Decision':
            if set(row['context'])!={'observed_context'}:raise ValueError('unexpected engine context')
            row['context']=row['context']['observed_context']
            for option in row['options']:option['features']={}
        if kind=='ContextEvent':
            if set(row['payload'])!={'observation'}:raise ValueError('unexpected engine event')
            row['payload']=row['payload']['observation']
        rows[kind].append(row)
    by_id={(kind,row[schema.ID_FIELDS[kind]]):row for kind,table in rows.items() if kind!='Provenance' for row in table}
    for provenance in rows['Provenance']:
        provenance['source_file_sha256']=hashlib.sha256(canonical(by_id[(provenance['record_type'],provenance['record_id'])])).hexdigest()
    result=dict(schema_version=schema.SCHEMA_VERSION,schema_hash=reader['ledger_schema_hash'],
        transport=ledger.TRANSPORT,rows=rows,blobs=deepcopy(reader['texts']))
    ledger.validate(result,schema)
    return result


def verify_export(root,v,bundle,schema,*,canaries=()):
    from tompathy_engine.ghost.bridge import load_reader,reader_choices
    root.mkdir(parents=True,exist_ok=False)
    exported=export(bundle,schema,canaries=canaries);write(root/'reader.json',exported)
    rows,texts=load_reader(root/'reader.json');choices=reader_choices(rows)
    native=native_view(exported,schema)
    if v['family'].startswith('T'):
        from .twin_ledger import decode
    else:decode=ledger.decode
    if canonical(decode(v,native,schema))!=canonical(decode(v,bundle,schema)):
        raise ValueError('engine projection changed native observations')
    ledger.export(root/'canonical',native)
    receipt=dict(state='ENGINE_TRANSPORT_VERIFIED',implementation_hash=exported['implementation_hash'],
        ledger_schema_hash=exported['ledger_schema_hash'],reader_hash=exported['reader_hash'],
        records=len(rows),resolved_baseline_choices=len(choices),blob_count=len(texts),
        native_observations_recovered=True,scientific_engine_admission=False,
        limitation='Observable option-feature baseline only; native exact inference and hidden labels are not engine inputs.')
    write(root/'CONFORMANCE.json',receipt)
    return receipt
