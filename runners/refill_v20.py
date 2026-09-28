"""Admit only predeclared unused V20 successors after an explicit review receipt."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now,digest
from ghostscale.validation.soundingline.v20.runtime import validate_design,fingerprint,consumed

def refill(campaign,selection,review):
    campaign=campaign.resolve();r=read(review)
    if not r.get('passed'):raise ValueError('verified completed-result review required')
    queue=read(campaign/'QUEUE.json');forest=read(campaign/'FOREST.json');a=read(campaign/'ACCEPTANCE.json')
    if (campaign/'STOP').exists():raise ValueError('campaign stopping')
    selected=read(selection);ids=selected['design_sha256s']
    if not selected.get('rationale') or not ids:raise ValueError('scientific pursuit rationale required')
    by={digest(d):d for d in forest['designs']}
    already={digest(read(Path(j['root'])/'PLAN.json')['design']) for j in queue['jobs']}
    if len(ids)!=len(set(ids)) or any(k not in by or k in already for k in ids):raise ValueError('duplicate or unregistered successor')
    source=Path(forest['source']);archive=Path(forest['archive']);manifest=read(Path(queue['jobs'][0]['root'])/'PLAN.json')
    added=[];clock=0
    for key in ids:
        d=by[key];validate_design(d);jid=f"v20-{len(queue['jobs']):04d}-{d['branch'].lower()}";root=Path(a['artifact_root'])/jid;root.mkdir(exist_ok=False);os.link(archive,root/'SOURCE.zip')
        plan=dict(manifest,design=d,successor_review_sha256=file_digest(review),selection_sha256=file_digest(selection));write(root/'PLAN.json',plan)
        write(root/'CARD.json',dict(id=jid,design=d,plan_sha256=file_digest(root/'PLAN.json'),source_archive_sha256=file_digest(archive),environment=fingerprint()))
        clock+=d['estimated_wall_seconds'];boundary=clock>=4*3600 or key==ids[-1]
        if boundary:clock=0
        job=dict(id=jid,root=str(root),source=str(source),plan_sha256=file_digest(root/'PLAN.json'),requires=['v20-0000-g0'],not_before=now(),reserve_eligible=d.get('reserve_eligible',False),cpu_estimate_seconds=d['estimated_cpu_seconds'],wall_estimate_seconds=d['estimated_wall_seconds'],review_boundary=boundary)
        queue['jobs'].append(job);added.append(jid)
    # Keep every queue revision; atomically replace only the mutable dispatch index.
    queue['revision']+=1;write(campaign/'queue-revisions'/f"{queue['revision']:04d}.json",queue);write(campaign/'QUEUE.json',queue,immutable=False)
    write(campaign/'refills'/f"{queue['revision']:04d}.json",dict(at=now(),added=added,rationale=selected['rationale'],review_sha256=file_digest(review),selection_sha256=file_digest(selection)))
    return added
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();print(refill(a.campaign,a.selection,a.review))
