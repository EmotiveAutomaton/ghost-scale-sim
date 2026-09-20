"""Supplied finite-law decoders of frozen old-question prediction banks.

This is extra generator knowledge, not extra target labels or a learned PSR.
The retained parent test cases are reused for paired decoder comparisons.
"""
import gzip
import json
from pathlib import Path
import shutil
import numpy as np
from ..v18_3 import world as W
from ..v18_3.io import read,write,file_digest
from . import neural_data as D

MODES={'bank-full':(5,1e-10),'bank-truncated':(5,1e-3),'bank-passive':(1,1e-10)}


def linear_map(bank,target,rcond):
    u,s,vh=np.linalg.svd(bank,full_matrices=False)
    retained=s>rcond*s[0]
    inverse=(vh[retained].T/s[retained])@u[:,retained].T
    mapping=inverse@target
    return mapping,dict(rank=int(retained.sum()),smallest_retained=float(s[retained][-1]),
        condition_number=float(s[0]/s[retained][-1]),map_norm=float(np.linalg.norm(mapping,2)),
        span_residual=float(np.max(abs(bank@mapping-target))))


def repair(raw):
    """Declared numerical scoring arm; raw invalidity is retained separately."""
    raw=np.asarray(raw,float)
    if not np.isfinite(raw).all():raise ValueError('nonfinite bank output')
    out=np.maximum(raw,1e-8);return out/out.sum(axis=-1,keepdims=True)


def prepare(root,parent,pulse=lambda **kw:None):
    root=Path(root);public=root/'reader';public.mkdir(parents=True,exist_ok=True)
    proof=read(parent/'INDEPENDENT_REPLAY.json')
    if not proof['passed'] or proof['summary_sha256']!=file_digest(parent/'SUMMARY.json') or proof['plan_sha256']!=file_digest(parent/'PLAN.json'):raise ValueError('bank requires verified parent')
    completed=read(parent/'COMPLETE.json')
    for name,sha in completed['files'].items():
        if file_digest(parent/name)!=sha:raise ValueError('bank parent changed')
    if (public/'INPUTS.json').exists():
        manifest=read(public/'INPUTS.json')
        for entry in [*manifest['tests'].values(),*manifest['encoders'].values(),*manifest['maps'].values()]:
            if file_digest(public/entry['name'])!=entry['sha256']:raise ValueError('bank capsule changed')
        return manifest
    original=read(parent/'data/reader/INPUTS.json');evaluator=read(parent/'data/EVALUATOR.json')
    manifest=dict(tests={},maps={},encoders={},parent_complete_sha256=file_digest(parent/'COMPLETE.json'),
        scope='paired reuse of verified parent tests; supplied intervention law; no new training or target labels',
        history_features=original['history_features'],query_features=original['query_features'],cases=original['cases'])
    diagnostics=[]
    for condition,entry in original['tests'].items():
        target=public/entry['name'];shutil.copyfile(parent/'data/reader'/entry['name'],target)
        manifest['tests'][condition]=dict(name=target.name,sha256=file_digest(target))
        for name in (evaluator['truth'][condition]['name'],f'{condition}-points.json.gz'):
            shutil.copyfile(parent/'data'/name,root/name)
        points=json.loads(gzip.decompress((root/f'{condition}-points.json.gz').read_bytes()))
        banks=[];maps={mode:[] for mode in MODES}
        for i,case in enumerate(points):
            # Only public world coefficients enter these maps, never state/history/targets.
            w=case['world'];old=[W.artifact_matrix(w,c) for c in D.TRAIN_QUERIES]
            banks.append([D.context_features(c)+D.world_features(w) for c in D.TRAIN_QUERIES])
            for mode,(count,rcond) in MODES.items():
                bank=np.concatenate([np.ones((len(W.STATES),1)),*old[:count]],axis=1)
                per_query=[]
                for q,c in enumerate(D.TEST_QUERIES[condition]):
                    mapping,check=linear_map(bank,W.artifact_matrix(w,c),rcond);per_query.append(mapping)
                    diagnostics.append(dict(condition=condition,history=i,query=q,mode=mode,**check))
                maps[mode].append(per_query)
            if i%64==63:pulse(phase='bank-law-maps',condition=condition,histories=i+1)
        path=public/f'{condition}-MAPS.npz'
        np.savez_compressed(path,bank_query=np.asarray(banks,np.float32),**{k:np.asarray(v) for k,v in maps.items()})
        manifest['maps'][condition]=dict(name=path.name,sha256=file_digest(path))
    child=read(parent/'neural/COMPLETE.json')
    for name,tests in child['predictions'].items():
        selected={e['selected'] for e in tests.values()}
        if len(selected)!=1:raise ValueError('test-selected parent weights')
        fit=selected.pop();source=parent/'neural'/fit/'BEST.pt';target=public/f'{name}-ENCODER.pt'
        if file_digest(source)!=child['fits'][fit]['best_sha256']:raise ValueError('parent weight changed')
        shutil.copyfile(source,target);manifest['encoders'][name]=dict(name=target.name,sha256=file_digest(target))
    write(root/'EVALUATOR.json',evaluator)
    summary={}
    for condition in manifest['tests']:
        for mode in MODES:
            rows=[r for r in diagnostics if r['condition']==condition and r['mode']==mode]
            summary[condition+'|'+mode]={key:dict(minimum=float(min(r[key] for r in rows)),maximum=float(max(r[key] for r in rows)),mean=float(np.mean([r[key] for r in rows]))) for key in ('rank','smallest_retained','condition_number','map_norm','span_residual')}
    write(root/'BANK_DIAGNOSTICS.json',dict(rows=diagnostics,summary=summary,scope='span/conditioning outcomes; no scientific ranking is a gate'))
    write(public/'INPUTS.json',manifest);return manifest
