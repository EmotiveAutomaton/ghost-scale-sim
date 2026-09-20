"""Admitted V18.4 packets with immutable raw blocks and honest resume."""
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
from ..v18_3.io import canonical,read,write,file_digest,now,digest
from ..v16.runtime import local_owner
from ..v18_3 import world as W


REPO=Path(__file__).resolve().parents[4]


def source_files():
    names=['ghostscale/__init__.py','ghostscale/validation/__init__.py','ghostscale/validation/soundingline/__init__.py',
           'runners/launch_background.py','runners/run_v18_4.py','runners/watch_v18_4.py',
           'docs/versions/v18-selective-acquisition/exploratory-loop/README.md',
           'docs/versions/v18-selective-acquisition/exploratory-loop/PROTOCOLS.md']
    for version in ('v16','v18_3','v18_4'):
        names += [p.relative_to(REPO).as_posix() for p in (REPO/'ghostscale/validation/soundingline'/version).glob('*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'tests').glob('test_v18_4*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'runners').glob('*v18_4*.py')]
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
    plan=dict(schema='v18.4.plan.1',design=design,sources=files,environment=fingerprint(),admission=admission,
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
    groups={};count=0
    for block in sorted((root/'blocks').glob('*.json')):
        for unit in load(root,block.stem):
            if limited():raise TimeoutError('verification cutoff')
            if unit['family']=='P':
                from .compression import verify
                dimensions=('family','cell','rule','tilt');metrics=('old_loss','new_loss','entropy_nats','old_optimum_ties')
            elif unit['family']=='U':
                from .adaptation import verify
                dimensions=('family','cell','condition','length','copy_span');metrics=('expected_loss','pre_change_loss','post_change_loss','expected_match','abstention_loss')
            elif unit['family']=='U2':
                from .matched_prefix import verify, DIMENSIONS, METRICS
                dimensions=DIMENSIONS;metrics=METRICS
            else:raise ValueError('unimplemented verifier')
            verify(unit);count+=1
            for row in unit['rows']:
                tags={k:unit.get(k) for k in dimensions}
                tags.update(method=row['method'])
                if 'cardinality' in row:tags['cardinality']=row['cardinality']
                key=canonical(tags).decode()
                for metric in metrics:
                    groups.setdefault(key,{}).setdefault(metric,[]).append(row[metric])
    summary=dict(schema='v18.4.summary.1',units=count,checks=dict(independent_unit_reconstructions=count),
        cells={k:{m:stats(v,(k,m)) for m,v in x.items()} for k,x in groups.items()},
        independent_unit='paired coefficient draw within fixed architecture cell; no confirmatory inference',
        scope='adaptive exploratory constructed methods; finite declared purposes')
    write(root/'SUMMARY.json',summary);return summary


def dispatch(spec):
    request=dict(spec);spec=dict(spec);family=spec.pop('family')
    if family=='P':
        from .compression import unit
    elif family=='U':
        from .adaptation import unit
    elif family=='U2':
        from .matched_prefix import unit
    else:raise ValueError('unimplemented V18.4 family')
    result=unit(**spec);result['request']=request;return result


def run(root,campaign,max_blocks=None):
    root=Path(root).resolve();campaign=Path(campaign).resolve()
    if read(root/'PLAN.json')['design'].get('engine')=='verification':
        from .verification import run as run_verification
        return run_verification(root,campaign)
    if read(root/'PLAN.json')['design'].get('engine')=='neural':
        from .neural_runtime import run as run_neural
        return run_neural(root,campaign)
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
            from .priority import below_normal
            below_normal()
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
