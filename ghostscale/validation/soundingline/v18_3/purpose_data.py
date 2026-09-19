"""Evaluator-owned labels for a separately supervised change of reader purpose."""
import gzip
import json
from pathlib import Path
import shutil
import numpy as np
from . import world as W,neural_data as D,purpose_readout as R
from .io import read,write,file_digest,canonical


def roles(posterior):
    values=[]
    for axis,size in enumerate((3,2,2,2)):
        for value in range(size):values.append(sum(float(posterior[i]) for i,s in enumerate(W.STATES) if s[axis]==value)/4)
    return np.asarray(values,float)


def prepare(root,parent,pulse=lambda **kw:None):
    root.mkdir(parents=True,exist_ok=True);public=root/'reader';public.mkdir(exist_ok=True)
    if (public/'INPUTS.json').exists():return read(public/'INPUTS.json')
    parent_inputs=read(parent/'data/reader/INPUTS.json');parent_complete=read(parent/'neural/COMPLETE.json')
    train_payload={}
    with np.load(parent/'data/reader/TRAIN.npz',allow_pickle=False) as z:
        for split in ('train','dev'):
            cases=json.loads(gzip.decompress((parent/'data'/f'{split}-points.json.gz').read_bytes()))
            for key in ('history','length','world'):train_payload[f'{split}_{key}']=z[f'{split}_{key}'].copy()
            train_payload[f'{split}_target']=np.asarray([roles(np.eye(len(W.STATES))[case['state']]) for case in cases])
    np.savez_compressed(public/'TRAIN.npz',**train_payload)
    tests={};truths={}
    for condition in ('in-support','new-combinations'):
        entry=parent_inputs['tests'][condition];cases=json.loads(gzip.decompress((parent/'data'/f'{condition}-points.json.gz').read_bytes()))
        with np.load(parent/'data/reader'/entry['name'],allow_pickle=False) as z:features={k:z[k].copy() for k in ('history','length','world')}
        with np.load(parent/'data'/f'{condition}-TRUTH.npz',allow_pickle=False) as z:ids=z['ids'][::5,:2].copy()
        truth=[];exact=[];passive=[];intervention=[]
        for case in cases:
            p=W.posterior(W.packet(case['world'],case['history']));truth.append(roles(np.eye(len(W.STATES))[case['state']]));exact.append(roles(p))
            for bank,target in (('passive-summary',passive),('intervention-summary',intervention)):
                summary,_=D.summary_state(case['world'],p,bank);target.append(roles(summary))
        np.savez_compressed(public/f'{condition}-INPUT.npz',**features)
        np.savez_compressed(root/f'{condition}-TRUTH.npz',target=truth,exact=exact,passive=passive,intervention=intervention,ids=ids)
        (root/f'{condition}-points.json.gz').write_bytes(gzip.compress(canonical(cases),mtime=0))
        tests[condition]=dict(name=f'{condition}-INPUT.npz',sha256=file_digest(public/f'{condition}-INPUT.npz'))
        truths[condition]=dict(name=f'{condition}-TRUTH.npz',sha256=file_digest(root/f'{condition}-TRUTH.npz'),points_sha256=file_digest(root/f'{condition}-points.json.gz'))
        pulse(phase='purpose-labels',condition=condition)
    encoders={}
    for name,predictions in parent_complete['predictions'].items():
        selected={receipt['selected'] for receipt in predictions.values()}
        if len(selected)!=1:raise ValueError('encoder selection used test condition')
        path=parent/'neural'/selected.pop()/'BEST.pt';target=public/f'{name}-ENCODER.pt';shutil.copyfile(path,target)
        encoders[name]=dict(name=target.name,sha256=file_digest(target))
    manifest=dict(schema='v18.3.purpose-input.1',train=dict(name='TRAIN.npz',sha256=file_digest(public/'TRAIN.npz')),
        tests=tests,encoders=encoders,supervision='equal additional training/development historical-role labels; behavior encoders frozen',
        role_class_sizes=[3,2,2,2],new_query_conditions='Historical role questions replace behavioral questions; same histories are reused.')
    write(root/'EVALUATOR.json',dict(truth=truths,parent_complete_sha256=file_digest(parent/'COMPLETE.json')))
    write(public/'INPUTS.json',manifest);return manifest


def score(root,pulse=lambda **kw:None):
    from .neural_runtime import proper_scores
    from .runtime import stats
    evaluator=read(root/'data/EVALUATOR.json');complete=read(root/'neural/COMPLETE.json');rows=[];references=0
    for condition,entry in evaluator['truth'].items():
        path=root/'data'/entry['name']
        if file_digest(path)!=entry['sha256']:raise ValueError('purpose truth changed')
        with np.load(path,allow_pickle=False) as z:truth=z['target'];ids=z['ids'];predictions={name:z[key].copy() for name,key in (('exact','exact'),('passive-summary','passive'),('intervention-summary','intervention'))}
        cases=json.loads(gzip.decompress((root/'data'/f'{condition}-points.json.gz').read_bytes()))
        if file_digest(root/'data'/f'{condition}-points.json.gz')!=entry['points_sha256']:raise ValueError('purpose cases changed')
        for i,case in enumerate(cases):
            expected=np.zeros(9);offset=0
            for value,size in zip(W.STATES[case['state']],(3,2,2,2)):expected[offset+value]=.25;offset+=size
            if not np.array_equal(expected,truth[i]):raise ValueError('historical role target changed')
            references+=1
        for name,outputs in complete['predictions'].items():
            receipt=outputs[condition];path=root/'neural'/receipt['file']
            if file_digest(path)!=receipt['sha256']:raise ValueError('purpose forecast changed')
            with np.load(path,allow_pickle=False) as z:predictions[name]=z['probabilities'].copy()
        for name,p in predictions.items():
            losses,brier,tv=proper_scores(truth,p);losses-=np.log(4)
            expected=np.stack([truth[:,columns].argmax(1) for columns in R.SLICES],axis=1)
            actual=np.stack([p[:,columns].argmax(1) for columns in R.SLICES],axis=1)
            method,seed=name.rsplit('-seed',1) if '-seed' in name else (name,None)
            for i,(cell,lineage) in enumerate(ids):
                rows.append(dict(condition=condition,method=method,seed=None if seed is None else int(seed),cell=int(cell),lineage=int(lineage),
                    expected_loss=W.loss_record(float(losses[i])),brier=float(brier[i]),total_variation=float(tv[i]),
                    role_accuracy=float(np.mean(actual[i]==expected[i])),whole_state_accuracy=float(np.all(actual[i]==expected[i]))))
        pulse(phase='purpose-scoring',condition=condition)
    clusters=[];cells={}
    for condition in evaluator['truth']:
        for method in sorted({r['method'] for r in rows}):
            for lineage in sorted({r['lineage'] for r in rows}):
                selected=[r for r in rows if r['condition']==condition and r['method']==method and r['lineage']==lineage]
                values={m:float(np.mean([r[m] for r in selected])) for m in ('brier','total_variation','role_accuracy','whole_state_accuracy')}
                values['expected_loss']=None if any(r['expected_loss']['infinite'] for r in selected) else float(np.mean([r['expected_loss']['value'] for r in selected]))
                clusters.append(dict(condition=condition,method=method,lineage=lineage,**values))
            selected=[r for r in clusters if r['condition']==condition and r['method']==method]
            cells[condition+'|'+method]={m:stats([r[m] for r in selected],('purpose',condition,method,m)) for m in ('expected_loss','brier','total_variation','role_accuracy','whole_state_accuracy')}
    path=root/'neural_points.json.gz';path.write_bytes(gzip.compress(canonical(dict(rows=rows,clusters=clusters)),mtime=0))
    write(root/'SUMMARY.json',dict(family='E-purpose',cells=cells,raw_sha256=file_digest(path),fits=complete['fits'],checks=dict(independent_role_labels=references),
        independent_unit='same parent E coefficient-draw lineages; new questions, fit seeds and architecture cells do not add independence',
        scope='additional equally supervised linear historical-role readouts from frozen memory; no encoder retraining or causal-identification claim'))
