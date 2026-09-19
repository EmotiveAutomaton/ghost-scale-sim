"""Finite source-bound V18.3 packets with immutable raw blocks and honest resume."""
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid
import zipfile
import numpy as np
from ..v16.records import canonical,read,write,file_digest,now,digest
from ..v16.runtime import local_owner
from . import world as W
from .verify import check_unit,metrics

REPO=Path(__file__).resolve().parents[4]


def source_files():
    names=['ghostscale/__init__.py','ghostscale/validation/__init__.py','ghostscale/validation/soundingline/__init__.py',
           'runners/launch_background.py','runners/run_v18_3.py','runners/watch_v18_3.py',
           'docs/versions/v18-selective-acquisition/research-extension/README.md']
    for version in ('v16','v18_3'):
        names += [p.relative_to(REPO).as_posix() for p in (REPO/'ghostscale/validation/soundingline'/version).glob('*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'tests').glob('test_v18_3*.py')]
    return sorted(set(names))


def fingerprint():
    return dict(python=platform.python_version(),numpy=np.__version__)


def freeze(source,root,design,admission):
    files={name:file_digest(REPO/name) for name in source_files()}
    if not admission.get('passed') or admission.get('sources')!=files:raise ValueError('current source admission required')
    root.mkdir(parents=True,exist_ok=False);source.mkdir(parents=True,exist_ok=False)
    for name in files:
        destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((REPO/name).read_bytes())
    with zipfile.ZipFile(root/'SOURCE.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in files:archive.write(REPO/name,name)
    plan=dict(schema='v18.3.plan.1',design=design,sources=files,environment=fingerprint(),admission=admission,
              source_archive_sha256=file_digest(root/'SOURCE.zip'),scope='descriptive discovery; finite miniature architectures')
    write(root/'PLAN.json',plan)
    return plan


def keep(root,name,units,cpu,wall):
    path=root/'raw'/f'{name}_points.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    payload=gzip.compress(canonical(units),mtime=0)
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('retained orphan differs; new repair identity required')
    else:path.write_bytes(payload)
    write(root/'blocks'/f'{name}.json',dict(raw_sha256=file_digest(path),units=len(units),cpu_seconds=cpu,wall_seconds=wall))


def load(root,name):
    path=root/'raw'/f'{name}_points.json.gz';receipt=read(root/'blocks'/f'{name}.json')
    if file_digest(path)!=receipt['raw_sha256']:raise ValueError('raw block corruption')
    units=json.loads(gzip.decompress(path.read_bytes()))
    if len(units)!=receipt['units']:raise ValueError('block unit count changed')
    plan=read(root/'PLAN.json');first=int(name.split('-')[-1])
    expected=plan['design']['units'][first:first+plan['design'].get('block_size',8)]
    if [u.get('request') for u in units]!=expected:raise ValueError('raw input roster changed')
    for unit,spec in zip(units,expected):
        if any(unit.get(k)!=v for k,v in spec.items() if k!='split'):raise ValueError('raw unit identity changed')
    return units


def stats(values,key):
    if any(x is None or not np.isfinite(x) for x in values):
        return dict(n=len(values),mean=None,infinite_or_missing=sum(x is None or not np.isfinite(x) for x in values))
    a=np.array(values,float);r=W.rng('interval',key)
    means=a[r.integers(len(a),size=(500,len(a)))].mean(axis=1)
    return dict(n=len(a),mean=float(a.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),
                interval='descriptive lineage bootstrap, 500 resamples')


def aggregate(root,limited=lambda:False):
    cells={};paired={};checks=dict(executions=0,distributions=0,reference_units=0);count=0
    for block in sorted((root/'blocks').glob('*.json')):
        for i,unit in enumerate(load(root,block.stem)):
            if limited():raise TimeoutError('verification reached absolute resource cutoff')
            checked=check_unit(unit,reference=i==0)
            checks['executions']+=checked['executions'];checks['distributions']+=checked['distributions']
            checks['reference_units']+=int(checked['reference']);count+=1
            group={}
            for tags,values in metrics(unit):
                key=canonical(tags).decode()
                for metric,value in values.items():cells.setdefault(key,{}).setdefault(metric,[]).append(value)
                baseline={'A':'fixed','B':'static','C':'independent','D':'fixed'}[unit['family']]
                comparable={k:v for k,v in tags.items() if k!='method'}
                group.setdefault(canonical(comparable).decode(),{})[tags['method']]=values
            for key,arms in group.items():
                if baseline not in arms:continue
                for method,values in arms.items():
                    if method==baseline:continue
                    target=paired.setdefault(key+'|'+method+' minus '+baseline,{})
                    for metric,value in values.items():
                        other=arms[baseline].get(metric)
                        if value is not None and other is not None:
                            target.setdefault(metric,[]).append(float(value)-float(other))
    result=dict(schema='v18.3.summary.1',units=count,independent_unit='maker/world lineage or source-network lineage',checks=checks,
                cells={k:{m:stats(v,(k,m)) for m,v in x.items()} for k,x in cells.items()},
                paired={k:{m:stats(v,(k,m)) for m,v in x.items()} for k,x in paired.items()},
                scope='descriptive constructed mechanisms and methods; declared finite architectures only')
    write(root/'SUMMARY.json',result)
    return result


def dispatch(spec):
    spec=dict(spec);family=spec.pop('family')
    if family=='A':
        from .active import unit
    elif family=='B':
        from .dynamics import unit
    elif family=='C':
        from .provenance import unit
    elif family=='D':
        from .revision import unit
    else:raise ValueError('unimplemented family')
    result=unit(**spec);result['request']=dict(family=family,**spec)
    return result


def run(root,campaign,max_blocks=None):
    root=Path(root).resolve();campaign=Path(campaign).resolve()
    with local_owner(campaign/'scientific-worker-owner'),local_owner(root):
        plan=read(root/'PLAN.json');acceptance=read(campaign/'ACCEPTANCE.json')
        if plan['environment']!=fingerprint():raise ValueError('runtime fingerprint changed')
        if any(file_digest(REPO/name)!=value for name,value in plan['sources'].items()):raise ValueError('source identity changed')
        if file_digest(root/'SOURCE.zip')!=plan['source_archive_sha256']:raise ValueError('source archive changed')
        if (root/'COMPLETE.json').exists():
            complete=read(root/'COMPLETE.json')
            if complete['plan_sha256']!=file_digest(root/'PLAN.json') or complete['summary_sha256']!=file_digest(root/'SUMMARY.json'):
                raise ValueError('complete packet identity changed')
            for block in complete['blocks']:load(root,block)
            return 'complete'
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
        start=time.monotonic();cpu=time.process_time();attempt=uuid.uuid4().hex
        previous=[read(p) for p in campaign.glob('attempts/*.json')]
        if any(x['state']=='running' for x in previous):raise ValueError('unreconciled interrupted CPU attempt')
        old=acceptance['prior_cpu_seconds']+sum(max(x['cpu_seconds'],x.get('native_cpu_seconds',0),x.get('uncertainty_cpu_seconds',0))+x.get('child_cpu_seconds',0) for x in previous)
        def emit(state,**detail):
            write(root/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),attempt=attempt,heartbeat=now(),state=state,**detail),immutable=False)
            write(campaign/'attempts'/f'{attempt}.json',dict(packet=root.name,state=state,cpu_seconds=time.process_time()-cpu,
                  child_cpu_seconds=0,wall_seconds=time.monotonic()-start),immutable=False)
        def limited():
            return ((campaign/'STOP').exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start'])
                    or old+time.process_time()-cpu>=acceptance['cumulative_cpu_ceiling_seconds'])
        emit('running');completed_this_attempt=0
        try:
            units=plan['design']['units'];block_size=plan['design'].get('block_size',8)
            for first in range(0,len(units),block_size):
                name=f'block-{first:06d}'
                if (root/'blocks'/f'{name}.json').exists():load(root,name);continue
                if limited():emit('resource_cutoff');return 'resource_cutoff'
                block_cpu=time.process_time();block_wall=time.monotonic();data=[]
                for spec in units[first:first+block_size]:
                    if limited():emit('resource_cutoff',retained_complete_blocks=first//block_size);return 'resource_cutoff'
                    data.append(dispatch(spec))
                    emit('running',completed_units=first+len(data),planned_units=len(units))
                keep(root,name,data,time.process_time()-block_cpu,time.monotonic()-block_wall)
                completed_this_attempt+=1
                if first+len(data)==16:
                    receipts=[read(p) for p in (root/'blocks').glob('*.json')]
                    elapsed=sum(x['wall_seconds'] for x in receipts)
                    write(root/'FIRST16_TIMING.json',dict(units=16,cpu_seconds=sum(x['cpu_seconds'] for x in receipts),wall_seconds=elapsed,
                        observed_remaining_wall_seconds=elapsed/16*(len(units)-16),forecast_only=True))
                if max_blocks is not None and completed_this_attempt>=max_blocks:
                    emit('checkpointed');return 'checkpointed'
            if limited():emit('resource_cutoff');return 'resource_cutoff'
            summary=aggregate(root,limited)
            emit('verified',units=summary['units'])
            write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),summary_sha256=file_digest(root/'SUMMARY.json'),
                blocks=[p.stem for p in sorted((root/'blocks').glob('*.json'))],completed_at=now(),validity='scoped checks passed'))
            emit('complete');return 'complete'
        except BaseException as exc:
            emit('failed',error=repr(exc));raise
