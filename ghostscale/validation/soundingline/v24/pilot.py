"""Complete balanced N16 pilot, immutable streams and separate reconstruction."""
import math
from pathlib import Path
from ..v23 import ledger
from ..v23.common import canonical, read, write, digest
from ..v23.pipeline import Stream
from ..v23.family import rows
from . import n16


def validate_design(plan):
    if plan['family']!='N16' or plan['kind']!='pilot':raise ValueError('N16 pilot only')
    if plan['identity']!='v24-n16-pilot-1':raise ValueError('fresh frozen pilot identity required')
    if plan['blocks']!=4 or plan['cases_per_cell']!=5:raise ValueError('four blocks with complete five-mechanism cycles required')
    if plan['epsilon']!=.1:raise ValueError('frozen retained-data epsilon changed')
    if plan['variant_ids']!=[v['id'] for v in n16.model.variants()]:raise ValueError('full original grid required')


def emit(schema,identity,public):
    emission=ledger.Emission(schema,['v24',*identity])
    # Keep the proven transport while explicitly declaring V24 provenance.
    emission.rows['Maker'][0]['source_dataset']='ghost-v24-synthetic'
    emission.rows['Artifact'][0]['source_uri']=emission.rows['Artifact'][0]['source_uri'].replace('ghost://v23/','ghost://v24/')
    emission.decision('public',0,[0],0,observed=public)
    bundle=emission.finish();ledger.validate(bundle,schema)
    if canonical(bundle['rows']['Decision'][0]['context']['observed'])!=canonical(public):
        raise ValueError('V24 canonical ledger round trip')
    return bundle


def summary(records,plan):
    values=n16.aggregate(records,plan['cases_per_cell'])
    return dict(state='PILOT_MEASUREMENT_ONLY',family='N16',confirmation=False,
        blocks=len(values),cases=len(records),balanced_block_means=values,
        epsilon=plan['epsilon'],lower_bound=math.log(plan['epsilon']),
        mean=math.fsum(values)/len(values),
        limitations=['Four pilot blocks do not establish confirmation power.',
                    'All misspecification, random-share and tempering traces remain in the raw streams.',
                    'No human, language-model or engine-admission inference follows.'])


def execute(root,plan,pulse):
    validate_design(plan);root=Path(root);root.mkdir(exist_ok=False)
    controls=n16.model.controls();write(root/'CONTROLS.json',controls)
    if not controls['passed']:raise ValueError('N16 controls failed')
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    streams={k:Stream(root/(k+'_points.jsonl.gz')) for k in ('public','truth','evidence','ledger')}
    scores=[]
    try:
        for block in range(plan['blocks']):
            for law in range(16):
                for variant in n16.model.variants():
                    for case in range(plan['cases_per_cell']):
                        identity=[block,law,variant['id'],case]
                        public,truth,record,score=n16.case(variant,plan['identity'],block,law,case,plan['epsilon'])
                        bundle=emit(schema,[plan['identity'],*identity],public)
                        for kind,payload in [('public',dict(public=public)),('truth',dict(truth=truth)),
                            ('evidence',dict(evidence=record,score=score)),('ledger',dict(ledger=bundle))]:
                            streams[kind].write(dict(identity=identity,**payload))
                        scores.append(dict(identity=identity,**score))
                pulse(block=block,law=law,cases=len(scores))
    finally:
        for stream in streams.values():stream.close()
    write(root/'SUMMARY.json',summary(scores,plan))
    write(root/'RAW_INDEX.json',{p.name:dict(sha256=digest(p),bytes=p.stat().st_size)
          for p in sorted(root.glob('*_points.jsonl.gz'))})


def reconstruct(root,plan,pulse):
    validate_design(plan);root=Path(root)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    variants={v['id']:v for v in n16.model.variants()};scores=[]
    raw_index=read(root/'RAW_INDEX.json')
    if set(raw_index)!={k+'_points.jsonl.gz' for k in ('public','truth','evidence','ledger')}:
        raise ValueError('incomplete raw manifest')
    for name,record in raw_index.items():
        path=root/name
        if digest(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise ValueError('raw hash mismatch')
    for public,truth,ev,transport in zip(*(rows(root/(k+'_points.jsonl.gz')) for k in ('public','truth','evidence','ledger')),strict=True):
        identity=public['identity']
        if any(r['identity']!=identity for r in (truth,ev,transport)):raise ValueError('stream identity mismatch')
        block,law,variant_id,case=identity;variant=variants[variant_id]
        mechanism=(16*block+law+case)%5 if variant['source']=='N9' else variant['original']['settings']['mechanism']
        if truth['truth']!={'law':law,'mechanism':mechanism}:raise ValueError('planted evaluator identity mismatch')
        ledger.validate(transport['ledger'],schema)
        if canonical(transport['ledger'])!=canonical(emit(schema,[plan['identity'],*identity],public['public'])):
            raise ValueError('independent ledger reconstruction mismatch')
        score=n16.check(variant,public['public'],truth['truth'],ev['evidence'],plan['epsilon'])
        for key in score:
            if not math.isclose(score[key],ev['score'][key],abs_tol=1e-11,rel_tol=0):raise ValueError('independent mixture score mismatch')
        scores.append(dict(identity=identity,**score))
        if len(scores)%48==0:pulse(cases=len(scores))
    rebuilt=summary(scores,plan);saved=read(root/'SUMMARY.json')
    if rebuilt['cases']!=4*16*48*5 or saved['cases']!=rebuilt['cases']:raise ValueError('incomplete pilot')
    for a,b in zip(rebuilt['balanced_block_means'],saved['balanced_block_means'],strict=True):
        if not math.isclose(a,b,abs_tol=1e-11,rel_tol=0):raise ValueError('balanced mean mismatch')
    if not math.isclose(rebuilt['mean'],saved['mean'],abs_tol=1e-11,rel_tol=0):raise ValueError('pilot mean mismatch')
    for key in set(rebuilt)|set(saved):
        if key not in ('mean','balanced_block_means') and rebuilt.get(key)!=saved.get(key):
            raise ValueError('pilot summary metadata mismatch')
    return dict(state='INDEPENDENT_PILOT_RECONSTRUCTION_PASSED',cases=len(scores),blocks=4,
        summary_sha256=digest(root/'SUMMARY.json'),confirmation=False)
