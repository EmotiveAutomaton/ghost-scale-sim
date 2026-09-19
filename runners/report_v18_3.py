"""Portable packet exports and coefficient-lineage summaries, never live rankings."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from ghostscale.validation.soundingline.v16.records import read,write,canonical,file_digest
from ghostscale.validation.soundingline.v18_3.runtime import load,stats
from ghostscale.validation.soundingline.v18_3.verify import metrics


def export(root,destination):
    proof=read(root/'INDEPENDENT_REPLAY.json')
    if not proof['passed']:raise ValueError('replay not passed')
    if proof['plan_sha256']!=file_digest(root/'PLAN.json') or proof['summary_sha256']!=file_digest(root/'SUMMARY.json'):
        raise ValueError('replay binding differs')
    destination.mkdir(parents=True,exist_ok=False)
    names=('PLAN.json','SOURCE.zip','SUMMARY.json','COMPLETE.json','INDEPENDENT_REPLAY.json')
    for name in names:shutil.copyfile(root/name,destination/name)
    with zipfile.ZipFile(destination/'RAW.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('raw','blocks'):
            for p in sorted((root/folder).glob('*')):archive.write(p,p.relative_to(root).as_posix())
    auditor=Path(__file__).with_name('replay_v18_3.py')
    if file_digest(auditor)!=proof['verifier_sha256']:raise ValueError('replay driver changed')
    shutil.copyfile(auditor,destination/'REPLAY_DRIVER.py')
    write(destination/'EXPORT.json',dict(files={p.name:file_digest(p) for p in destination.iterdir() if p.is_file()},
        scope='complete retained raw blocks and original source; selected eight-unit replay, not complete source regeneration'))


def rollup(roots):
    grouped={};count=0
    for root in roots:
        complete=read(root/'COMPLETE.json')
        for name in complete['blocks']:
            for unit in load(root,name):
                count+=1
                for tags,values in metrics(unit):
                    tags.pop('cell',None);key=canonical(tags).decode();lineage=unit['index']
                    for metric,value in values.items():grouped.setdefault(key,{}).setdefault(metric,{}).setdefault(lineage,[]).append(value)
    result={};clusters={}
    for key,metrics_ in grouped.items():
        result[key]={};clusters[key]={}
        for metric,draws in metrics_.items():
            means={str(i):None if any(v is None for v in a) else float(np.mean(a)) for i,a in draws.items()}
            clusters[key][metric]=means;result[key][metric]=stats(list(means.values()),('rollup',key,metric))
    return dict(assigned_units=count,cells=result,clusters=clusters,
        independent_unit='coefficient-draw lineage averaged across paired architecture cells; source lineage for C',
        scope='descriptive discovery; no fresh replication from reused draws or repairs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--rollup',type=Path,required=True)
    p.add_argument('--export',type=Path);a=p.parse_args()
    write(a.rollup,rollup(a.roots))
    if a.export:
        for root in a.roots:export(root,a.export/root.name)
