"""Reanalyze all retained N2 predictions without another generative exposure."""
from pathlib import Path
import math,shutil
import numpy as np
from . import adaptive_numerics,family,registry
from .common import digest,interval,read,write
from .pipeline import Stream,headroom_groups
from .pipeline_new import score
from .adaptive_numerics import interval


def run(root,plan,pulse):
    adaptive_numerics.validate(plan)
    source=Path(plan['rescore_source']);old,summary=family.verify_packet(source)
    if digest(source/'COMPLETE.json')!=plan['rescore_source_complete_sha256']:raise ValueError('retained N2 source changed')
    if any(old[k]!=plan[k] for k in ('family','batch','worlds','cases','variants','fit_hashes','canary_key')):
        raise ValueError('reanalysis cannot change the original experiment')
    if old['family']!='N2' or old['batch']!=1:raise ValueError('only first retained N2 batch is rescored')
    root=Path(root);root.mkdir(parents=True,exist_ok=False);results=[];index={}
    rival,mechanism=registry.PRIMARY['N2'][:2]
    for original in summary['variants']:
        v=original['variant'];folder=root/v['id'];folder.mkdir();prior=source/'original'/v['id']
        for path in prior.iterdir():
            if path.name not in ('SUMMARY.json','WORLD_AGGREGATES.json','prediction_points.jsonl.gz'):
                if not path.is_file():raise ValueError('unexpected nested N2 scientific record')
                shutil.copyfile(path,folder/path.name)
        by_world={w:[] for w in range(plan['worlds'])};reader={};seen=set();stream=Stream(folder/'prediction_points.jsonl.gz')
        try:
            for pred,truth in zip(family.rows(prior/'prediction_points.jsonl.gz'),family.rows(prior/'truth_points.jsonl.gz'),strict=True):
                world,case=pred['world'],pred['case']
                if (world,case) in seen or world not in by_world or case not in range(plan['cases']):raise ValueError('N2 raw coverage changed')
                if truth['identity']!=['N2',v['id'],plan['batch'],world,case]:raise ValueError('N2 raw identity changed')
                seen.add((world,case))
                scored={name:score(prob,truth['truth'],stable=True) for name,prob in pred['probabilities'].items()}
                stream.write(dict(pred,scores=scored))
                by_world[world].append(scored[rival]['log_loss']-scored[mechanism]['log_loss'])
                for name,row in scored.items():
                    for metric in ('log_loss','half_brier','accuracy'):
                        reader.setdefault(name,{}).setdefault(metric,{}).setdefault(world,[]).append(row[metric])
            if len(seen)!=plan['worlds']*plan['cases']:raise ValueError('incomplete N2 reanalysis')
        finally:stream.close()
        mean=lambda xs:math.fsum(xs)/len(xs)
        gains=[mean(by_world[w]) for w in range(plan['worlds'])]
        readers={name:{metric:interval([mean(worlds[w]) for w in range(plan['worlds'])],[v['id'],plan['batch'],name,metric])
            for metric,worlds in metrics.items()} for name,metrics in reader.items()}
        headroom={name:[mean(reader[name]['log_loss'][w])-mean(reader['law-marginal']['log_loss'][w]) for w in range(plan['worlds'])]
            for name in summary['fits']}
        result=dict(original,primary=interval(gains,[v['id'],plan['batch'],'primary']),world_gains=gains,readers=readers,
            matched_headroom_groups=headroom_groups(headroom,summary['fits'],v,plan['batch']),
            numerical_reanalysis='stable normalized log score; no generated, excluded or replaced case')
        write(folder/'SUMMARY.json',result);results.append(result)
        index[v['id']]={p.name:digest(p) for p in folder.iterdir() if p.is_file()}
        pulse(phase='retained-N2-reanalysis',variant=v['id'])
    write(root/'DESIGN.json',{k:plan[k] for k in ('family','batch','variants','worlds','cases','schema_pin')})
    write(root/'RAW_INDEX.json',index);write(root/'SUMMARY.json',dict(summary,variants=results))
    write(root/'REANALYSIS.json',dict(original_complete_sha256=plan['rescore_source_complete_sha256'],
        original_replay_sha256=digest(source/'REPLAY.json'),numerical_validation_complete_sha256=plan['numerical_validation_complete_sha256'],
        original_preserved=True,new_sampling=False,changed_predictions=False,changed_primary=False))
