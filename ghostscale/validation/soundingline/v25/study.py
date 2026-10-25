"""Full blocks, deterministic replay, independent arithmetic and decimal aggregation."""
from pathlib import Path
import gzip,importlib,json,math
from collections import defaultdict
from decimal import Decimal,localcontext
import numpy as np
from .common import canonical,read,write,digest,close

MODULES=dict(W1='pieces',W2='setpoints',W3='uptake',W4='population',W5='misspec',W6='emotion',W7='attention',W8='volatile',W9='betrayal',W10='absorption',W11='care',W12='sacred',W13='persistence')

def module(family):return importlib.import_module(__package__+'.'+MODULES[family])

def aggregate(blocks,independent=False):
    # First aggregate cases inside each world; worlds are the only inference units.
    cells=defaultdict(dict);raw_values=defaultdict(list)
    for item in blocks:
        within=defaultdict(list)
        for row in item['data']['readouts']:
            for metric,value in row['metrics'].items():
                key=canonical(dict(setting=item['setting'],labels=row['labels'],metric=metric)).decode()
                # Maker/truth identifiers stratify raw evidence but never multiply n.
                labels={k:v for k,v in row['labels'].items() if k not in ('maker','truth')}
                key=canonical(dict(setting=item['setting'],labels=labels,metric=metric)).decode()
                if value is not None:within[key].append(float(value))
                else:within.setdefault(key,[])
        for key,values in within.items():
            if independent:
                with localcontext() as ctx:
                    ctx.prec=50
                    mean=float(sum((Decimal(str(v)) for v in values),Decimal(0))/len(values)) if values else None
            else:mean=math.fsum(values)/len(values) if values else None
            cells[key][item['block']]=mean;raw_values[key].extend(values)
    result=[]
    for key,worlds in sorted(cells.items()):
        vals=[v for v in worlds.values() if v is not None]
        result.append(dict(**json.loads(key),whole_world_values=[worlds[k] for k in sorted(worlds)],
            blocks=len(worlds),available_blocks=len(vals),mean=math.fsum(vals)/len(vals) if vals else None,
            median=float(np.median(raw_values[key])) if raw_values[key] else None,
            standard_error=float(np.std(vals,ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else None))
    return result

def execute(root,plan,pulse,independent=False):
    root=Path(root);implementation=module(plan['family'])
    if plan['kind']=='followup':implementation=importlib.import_module(__package__+'.followups')
    controls=implementation.controls()
    if not controls.get('passed'):raise ValueError('known-answer failure')
    if not independent:root.mkdir(exist_ok=False);write(root/'CONTROLS.json',controls)
    rows=[]
    for setting_id,setting in enumerate(plan['settings']):
        for block in range(plan['blocks']):
            identity=[plan['science_identity'],plan['family'],setting_id,block]
            data=implementation.block(setting,identity,independent=independent)
            row=dict(setting=setting_id,configuration=setting,block=block,data=data)
            path=root/f'world-{setting_id:04d}-{block:04d}_points.json.gz'
            if independent:
                with gzip.open(path,'rt',encoding='utf-8') as f:saved=json.load(f)
                close(row,saved,tol=1e-9)
                rows.append(dict(setting=saved['setting'],block=saved['block'],data=dict(readouts=saved['data']['readouts'])))
            else:
                with path.open('xb') as raw:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:f.write(canonical(row)+b'\n')
                rows.append(dict(setting=row['setting'],block=row['block'],data=dict(readouts=data['readouts'])))
            pulse(setting=setting_id,block=block,phase='world_complete')
    summary=dict(family=plan['family'],kind=plan.get('scientific_kind',plan['kind']),state='DESCRIPTIVE_CONSTRUCTED_STUDY',
        architecture='miniature, architecture untested',independent_blocks_per_setting=plan['blocks'],
        settings=plan['settings'],cells=aggregate(rows,independent),confirmation=False,
        reader='exact finite-library; specified likelihood model; independent arithmetic shares declared generative model',
        scientific_identity=plan['science_identity'])
    if independent:
        close(summary,read(root/'SUMMARY.json'),tol=1e-9)
        return dict(state='INDEPENDENT_RECONSTRUCTION_PASSED',worlds=len(rows),all_declared_cells=True,
            independently_recomputed_likelihood_arithmetic=True,independent_decimal_aggregation=True,
            limitation='Shared declared generative law; model realism is not validated.',summary_sha256=digest(root/'SUMMARY.json'))
    write(root/'SUMMARY.json',summary)
    write(root/'RAW_INDEX.json',{p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(root.glob('*_points.json.gz'))})

def reconstruct(root,plan,pulse):
    root=Path(root)
    for name,binding in read(root/'RAW_INDEX.json').items():
        if digest(root/name)!=binding['sha256']:raise ValueError('raw evidence changed')
    return execute(root,plan,pulse,True)
