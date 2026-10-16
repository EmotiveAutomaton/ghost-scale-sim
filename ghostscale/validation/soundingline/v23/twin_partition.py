"""Assemble complete twin worlds without splitting makers or shared graphs."""
from pathlib import Path
import shutil
from . import family,partition,runtime,twin_pipeline
from .common import digest,read,write,interval,canonical


def assemble(root,parts,pulse=lambda:None):
    root=Path(root);plan=read(root/'PLAN.json');parts=[Path(p) for p in parts]
    if plan['task']!='twin-batch' or not plan.get('scheduler_only') or plan.get('world_indices') is not None:
        raise ValueError('complete logical twin batch required')
    if parts!=partition.part_paths(root,plan):raise ValueError('twin partition paths differ')
    verified=[]
    for index,part in enumerate(parts):
        child,summary=family.verify_packet(part)
        partition.verify_part_binding(root,plan,part,child,index)
        for key in ('task','family','batch','worlds','fit_hashes','canary_key','sources','empirical_reference_sha256'):
            if child[key]!=plan[key]:raise ValueError('twin partition scientific contract differs')
        if summary['state']!='TWIN_BATCH_CHECKED':raise ValueError('unverified twin partition')
        verified.append((child,summary));pulse()
    if [w for child,_ in verified for w in partition.world_indices(child)]!=list(range(plan['worlds'])):
        raise ValueError('twin world coverage overlaps or is incomplete')
    variants=[item['variant'] for item in verified[0][1]['variants']]
    if any([item['variant'] for item in summary['variants']]!=variants for _,summary in verified):
        raise ValueError('twin variant coverage differs')
    for label in ('original','replay'):
        destination=root/label;destination.mkdir();summaries=[]
        for index,variant in enumerate(variants):
            identity=variant['id'];worlds=[]
            for part,(child,_) in zip(parts,verified,strict=True):
                item=read(part/label/'SUMMARY.json')['variants'][index]
                if [world['world'] for world in item['worlds']]!=partition.world_indices(child):
                    raise ValueError('twin part omitted or duplicated a world')
                for world in item['worlds']:
                    name=f"world-{world['world']:04d}"
                    shutil.copytree(part/label/identity/name,destination/identity/name)
                    if canonical(read(destination/identity/name/'SUMMARY.json'))!=canonical(world):
                        raise ValueError('twin world summary differs')
                    worlds.append(world);pulse()
            gains=[world['world_gain'] for world in worlds]
            estimate=interval(gains,[identity,'detectability']) if all(x is not None for x in gains) else dict(finite=False,mean=None,lower=None,upper=None,se=None)
            summaries.append(dict(verified[0][1]['variants'][index],worlds=worlds,gain=estimate,
                detected=bool(estimate['finite'] and estimate['lower']>.02)))
        write(destination/'SUMMARY.json',dict(verified[0][1],variants=summaries))
        twin_pipeline.reconstruct(destination,plan);pulse()
    replay=runtime.compare(root/'original',root/'replay');write(root/'REPLAY.json',replay)
    review=twin_pipeline.reconstruct(root/'original',plan);write(root/'NUMERICAL_REVIEW.json',review)
    return dict(state='verified',at=runtime.now(),plan_sha256=digest(root/'PLAN.json'),
        replay_sha256=digest(root/'REPLAY.json'),summary_sha256=digest(root/'original/SUMMARY.json'),
        partitions={p.name:digest(p/'COMPLETE.json') for p in parts})
