"""Immutable discovery blocks and single-owner execution with absolute ceilings."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import zipfile
from ..v16.records import write,read,file_digest,now
from ..v16.runtime import local_owner
from .model import canonical,make_case,evaluate
from .verify import check_case,interval

REPO=Path(__file__).resolve().parents[4]


def sources():
    names=['ghostscale/__init__.py','ghostscale/validation/__init__.py',
           'ghostscale/validation/soundingline/__init__.py','runners/launch_background.py','runners/run_v18_2.py']
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'runners').glob('*v18_2*.py')]
    for version in ('v16','v18_2'):
        names += [p.relative_to(REPO).as_posix() for p in (REPO/'ghostscale/validation/soundingline'/version).glob('*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'tests').glob('test_v18_2*.py')]
    return sorted(set(names))


def freeze(destination,root,design,admission):
    hashes={p:file_digest(REPO/p) for p in sources()}
    if admission.get('sources')!=hashes or not admission.get('passed'): raise ValueError('current-source admission required')
    destination.mkdir(parents=True,exist_ok=False)
    for p in hashes:
        out=destination/p;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((REPO/p).read_bytes())
    with zipfile.ZipFile(root/'SOURCE.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in hashes: z.write(REPO/p,p)
    plan=dict(schema='v18.2.plan.1',sources=hashes,design=design,admission=admission,
              source_archive_sha256=file_digest(root/'SOURCE.zip'),discovery=True)
    write(root/'PLAN.json',plan)
    return plan


def keep(root,name,data,cpu,wall):
    payload=gzip.compress(canonical(data),mtime=0)
    raw=root/'raw'/f'{name}_points.json.gz';raw.parent.mkdir(parents=True,exist_ok=True)
    if raw.exists():
        if raw.read_bytes()!=payload: raise ValueError('orphan differs; preserve for repair')
    else: raw.write_bytes(payload)
    write(root/'blocks'/f'{name}.json',dict(raw_sha256=file_digest(raw),worker_cpu_seconds=cpu,
          child_cpu_seconds=0,wall_seconds=wall,units=len(data),completed_at=now()))


def load(root,name):
    receipt=read(root/'blocks'/f'{name}.json');raw=root/'raw'/f'{name}_points.json.gz'
    if file_digest(raw)!=receipt['raw_sha256']:raise ValueError('raw mismatch')
    return json.loads(gzip.decompress(raw.read_bytes()))


def aggregate(root):
    cells={};checked=0;cases=0;dispositions={};rows_total=0
    for path in sorted((root/'blocks').glob('*.json')):
        for unit in load(root,path.stem):
            check=check_case(unit['case'])
            if not check['passed']:raise ValueError('independent execution mismatch')
            checked+=check['executions'];cases+=1
            per_unit={}
            for row in unit['rows']:
                key='|'.join([unit['case']['split'],row.get('condition','base'),row['tier'],row['method']])
                counts=dispositions.setdefault(key,{})
                state=row.get('instrument','missing');counts[state]=counts.get(state,0)+1;rows_total+=1
                evaluator=row.get('counterfactual_evaluator',{})
                from .verify import replay
                for obs in evaluator.get('history',[])+[p['observed'] for p in evaluator.get('probes',[])]+([evaluator['observed']] if 'observed' in evaluator else []):
                    if replay(obs['program'])!=obs['artifact']:raise ValueError('counterfactual execution mismatch')
                    checked+=1
                if state!='valid':continue
                for metric,value in row['scores'].items():per_unit.setdefault((key,metric),[]).append(value)
            for (key,metric),values in per_unit.items():cells.setdefault(key,{}).setdefault(metric,[]).append(sum(values)/len(values))
    summary=dict(schema='v18.2.summary.1',units=cases,rows=rows_total,independent_executions=checked,dispositions=dispositions,
                 cells={k:{m:interval(v) for m,v in metrics.items()} for k,metrics in cells.items()},
                 scope='discovery constructed mechanism; miniature — architecture untested')
    write(root/'SUMMARY.json',summary,immutable=False)
    return summary


def run(root,campaign):
    root=root.resolve();campaign=campaign.resolve()
    with local_owner(campaign/'scientific-worker-owner'),local_owner(root):
        plan=read(root/'PLAN.json');acceptance=read(campaign/'ACCEPTANCE.json')
        if {p:file_digest(REPO/p) for p in plan['sources']}!=plan['sources']:raise ValueError('source mismatch')
        if (root/'COMPLETE.json').exists():
            complete=read(root/'COMPLETE.json')
            if file_digest(root/'SUMMARY.json')!=complete['summary_sha256']:raise ValueError('completed summary mismatch')
            for name in complete['blocks']:load(root,name)
            return read(root/'SUMMARY.json')
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
        attempt=uuid.uuid4().hex;start=time.monotonic();cpu=time.process_time()
        old=sum(read(p)['cpu_seconds'] for p in campaign.glob('*/attempts/*.json'))
        def emit(state,**extra):
            write(root/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=now(),state=state,
                  source_root=str(REPO),**extra),immutable=False)
            write(root/'attempts'/f'{attempt}.json',dict(cpu_seconds=time.process_time()-cpu,
                  wall_seconds=time.monotonic()-start,state=state,child_cpu_seconds=0),immutable=False)
        def limited():
            return datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start']) or old+time.process_time()-cpu>=acceptance['worker_cpu_ceiling_seconds']
        emit('running')
        try:
            design=plan['design'];branch=design['branch']
            # No scientific children: fits, evaluation and all verification share this worker.
            if branch=='assembly':
                from .assembly_maker import run as assembly_run
                assembly_run(root,design,limited,lambda **kw:emit('running',**kw))
            elif branch=='learned':
                from .learned import run_comparison
                run_comparison(root,design,limited,lambda **kw:emit('running',**kw))
            elif branch=='g6':
                from .branches import population_check
                result=population_check(design,limited)
                write(root/'SUMMARY.json',result)
            else:
                specs=[('dev',64),('test',128)] if branch=='g0' else [('test',design.get('histories',128))]
                for split,count in specs:
                    for first in range(0,count,8):
                        name=f'{split}-{first:05d}'
                        if (root/'blocks'/f'{name}.json').exists():load(root,name);continue
                        if limited():emit('resource_cutoff');return
                        block_cpu=time.process_time();block_wall=time.monotonic();data=[]
                        for index in range(first,min(first+8,count)):
                            case=make_case(design['namespace'],index,split,length=design.get('length',8),
                                           change=design.get('change',False),family=design.get('family','board'),probe_mode=design.get('probe_mode','standard'))
                            if branch=='g0': rows=evaluate(case)
                            else:
                                from .branches import evaluate_branch
                                rows=evaluate_branch(case,design)
                            data.append(dict(case=case,rows=rows))
                        keep(root,name,data,time.process_time()-block_cpu,time.monotonic()-block_wall)
                        emit('running',branch=branch,split=split,completed=first+len(data))
                        if first==8:
                            elapsed=time.monotonic()-start
                            write(root/f'FORECAST-{split}.json',dict(completed_units=16,elapsed_seconds=elapsed,
                                  observed_remaining_seconds=elapsed/16*(count-16),
                                  twice_as_fast_remaining_seconds=elapsed/32*(count-16),
                                  rule='forecast only; next conceptual branch pre-admitted'))
                aggregate(root)
            if limited():emit('resource_cutoff');return
            write(root/'COMPLETE.json',dict(completed_at=now(),plan_sha256=file_digest(root/'PLAN.json'),
                  summary_sha256=file_digest(root/'SUMMARY.json'),blocks=[p.stem for p in sorted((root/'blocks').glob('*.json'))]))
            emit('complete')
        except BaseException as exc:
            emit('failed',error=repr(exc));raise
