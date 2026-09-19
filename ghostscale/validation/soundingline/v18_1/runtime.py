"""Immutable branch plans and resumable raw blocks, one native scientific owner."""
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import time
import uuid
import zipfile

from ..v16.records import canonical, digest, file_digest, read, write, now
from ..v16.runtime import local_owner
from . import g0,g1,g2,g3,g4

REPO = Path(__file__).resolve().parents[4]


def live_write(path,value):
    """A brief Windows reader may deny atomic replacement; bounded operational retry."""
    for attempt in range(20):
        try:return write(path,value,immutable=False)
        except PermissionError:
            if attempt==19:raise
            time.sleep(0.05)


def source_files():
    paths = ['ghostscale/__init__.py','ghostscale/validation/__init__.py','ghostscale/validation/soundingline/__init__.py',
             'runners/launch_background.py','runners/run_v18_1.py','runners/watch_v18_1.py','runners/replay_v18_1.py','tests/test_v18_1.py',
             'runners/read_v18_1_history.py','runners/watch_v18_1_events.py','runners/verify_v18_1_targets.py',
             'ghostscale/validation/soundingline/v17/__init__.py','ghostscale/validation/soundingline/v17/stitch_adapter.py',
             'docs/versions/v18-selective-acquisition/continuation/CODING_PACKAGE.md',
             'docs/versions/v18-selective-acquisition/continuation/README.md']
    for version in ('v16','v18','v18_1'):
        paths.extend(str(p.relative_to(REPO)).replace('\\','/') for p in (REPO/'ghostscale/validation/soundingline'/version).glob('*.py'))
    paths.extend(str(p.relative_to(REPO)).replace('\\','/') for p in (REPO/'tests').glob('test_v18_1*.py'))
    return sorted(set(paths))


def freeze(root, acceptance, archive, admission, *, branch='g0-original', namespace='v18.1-g0-exposed-1', constructors=64):
    if not admission.get('passed') or admission.get('sources') != {p:file_digest(REPO/p) for p in source_files()}:
        raise ValueError('passing current-source admission required')
    if constructors not in (8,64):
        raise ValueError('unexpected original-case prefix')
    plan = dict(schema='v18.1.branch-plan.1', branch=branch, namespace=namespace,
                started_at=acceptance['started_at'], admission_cutoff=acceptance['report_start'],
                delivery_deadline=acceptance['delivery_deadline'], cpu_ceiling_seconds=acceptance['worker_cpu_ceiling_seconds'],
                archive_sha256=file_digest(archive), constructors=list(range(constructors)), block_size=8,
                sources=admission['sources'], admission=admission, budgets=[32,128],
                sampling='original exposed V18 cases; descriptive diagnostic, no new independent units',
                worker_limit=1, child_compute=False, original_campaign_unchanged=True)
    write(root/'PLAN.json',plan)
    return plan


def archive_cases(archive):
    cases=[]
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if name.startswith('run/raw/core-') and name.endswith('.json.gz'):
                cases.extend(unit['case'] for unit in json.loads(gzip.decompress(z.read(name)))['units'])
    return cases


def freeze_cases(root,acceptance,admission,cases,design,branch,namespace):
    if not admission.get('passed') or admission.get('sources')!={p:file_digest(REPO/p) for p in source_files()}:
        raise ValueError('passing current-source admission required')
    identities=[c['case_id'] for c in cases]
    if len(set(identities))!=len(cases):raise ValueError('duplicate case identity')
    units=sorted(set(c['structural_unit'] for c in cases))
    other_units=set()
    for path in root.parent.glob('*/PLAN.json'):
        other_units.update(read(path).get('structural_units',[]))
    if len(other_units|set(units))>acceptance['new_law_context_unit_ceiling']:
        raise ValueError('campaign structural-unit ceiling')
    root.mkdir(parents=True,exist_ok=True)
    payload=gzip.compress(canonical(cases),mtime=0)
    path=root/'INPUTS.json.gz'
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('frozen cases differ')
    else:
        with path.open('xb') as stream:stream.write(payload)
    plan=dict(schema='v18.1.branch-plan.1',branch=branch,namespace=namespace,
              started_at=acceptance['started_at'],admission_cutoff=acceptance['report_start'],
              delivery_deadline=acceptance['delivery_deadline'],cpu_ceiling_seconds=acceptance['worker_cpu_ceiling_seconds'],
              structural_units=units,cases=len(cases),block_size=design.get('block_size',24),design=design,
              inputs_sha256=file_digest(path),sources=admission['sources'],admission=admission,
              sampling=design.get('sampling','declared finite support or frozen stratified discovery; not confirmation'),
              worker_limit=1,child_compute=bool(design.get('child_compute',False)))
    write(root/'PLAN.json',plan)
    return plan


def validate(root, archive):
    plan=read(root/'PLAN.json')
    if {p:file_digest(REPO/p) for p in plan['sources']} != plan['sources']:
        raise ValueError('frozen source mismatch')
    if 'archive_sha256' in plan and file_digest(archive) != plan['archive_sha256']:
        raise ValueError('original archive mismatch')
    if 'inputs_sha256' in plan and file_digest(root/'INPUTS.json.gz')!=plan['inputs_sha256']:
        raise ValueError('frozen cases mismatch')
    return plan


def load_block(root,name):
    receipt=read(root/'blocks'/(name+'.json'))
    raw=root/'raw'/(name+'_points.json.gz')
    if file_digest(raw) != receipt['raw_sha256']:
        raise ValueError('raw checksum mismatch')
    data=json.loads(gzip.decompress(raw.read_bytes()))
    if digest(data) != receipt['content_sha256']:
        raise ValueError('content checksum mismatch')
    return data,receipt


def retain(root,name,data,wall,cpu,child_cpu=0):
    raw=gzip.compress(canonical(data),mtime=0)
    path=root/'raw'/(name+'_points.json.gz')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:
            raise ValueError('orphan output differs; preserve and investigate')
    else:
        temporary=path.with_suffix('.tmp')
        with temporary.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path);temporary.unlink()
    receipt=dict(name=name,completed_at=now(),raw_sha256=file_digest(path),content_sha256=digest(data),
                 histories=len(data['units']),rows=sum(len(u['rows']) for u in data['units']),
                 wall_seconds=wall,worker_cpu_seconds=cpu,child_cpu_seconds=child_cpu)
    write(root/'blocks'/(name+'.json'),receipt)


def run(root,archive,*,stop_after_blocks=None):
    root=root.resolve()
    with local_owner(root.parent/'scientific-worker-owner'),local_owner(root):
        plan=validate(root,archive)
        if (root/'COMPLETE.json').exists():
            for name in read(root/'COMPLETE.json')['blocks']:
                load_block(root,name)
            return read(root/'COMPLETE.json')
        original=plan['branch']=='g0-original'
        cases=archive_cases(archive) if original else json.loads(gzip.decompress((root/'INPUTS.json.gz').read_bytes()))
        attempt=uuid.uuid4().hex
        wall,cpu=time.monotonic(),time.process_time()
        def measured_children():
            records=[read(p) for p in (root/'child-attempts').glob('*.json')]
            if any(not r['cpu_measured'] for r in records):raise ValueError('unmeasured child CPU retained')
            return sum(r['child_cpu_seconds'] for r in records)
        child_baseline=measured_children()
        previous_cpu=sum(read(p)['worker_cpu_seconds']+read(p).get('child_cpu_seconds',0) for p in root.parent.glob('*/attempts/*.json'))
        status=dict(pid=os.getpid(),source_root=str(REPO),started_at=now(),state='running',blocks=[])
        def emit(**changes):
            status.update(changes);status['heartbeat']=now()
            live_write(root/'STATUS.json',status)
            live_write(root/'attempts'/(attempt+'.json'),dict(started_at=status['started_at'],state=status['state'],
                  wall_seconds=time.monotonic()-wall,worker_cpu_seconds=time.process_time()-cpu,child_cpu_seconds=measured_children()-child_baseline))
        emit()
        completed=[];new=0
        try:
            population=len(plan['constructors']) if original else len(cases)
            for start in range(0,population,plan['block_size']):
                name=f"{plan['branch']}-{start//plan['block_size']:03d}"
                if (root/'blocks'/(name+'.json')).exists():
                    load_block(root,name);completed.append(name);continue
                if stop_after_blocks is not None and new>=stop_after_blocks:
                    emit(state='checkpointed',blocks=completed);return status
                if datetime.now(timezone.utc)>=datetime.fromisoformat(plan['admission_cutoff']) or previous_cpu+time.process_time()-cpu+measured_children()-child_baseline>=plan['cpu_ceiling_seconds']:
                    emit(state='resource_cutoff',blocks=completed);return status
                emit(active_block=name)
                block_wall,block_cpu=time.monotonic(),time.process_time()
                block_child=measured_children()
                units=[]
                selected=set(plan['constructors'][start:start+plan['block_size']]) if original else None
                selected_cases=[c for c in cases if c['constructor_index'] in selected] if original else cases[start:start+plan['block_size']]
                for case in selected_cases:
                    if original:
                        rows=g0.original_matrix(case,plan['budgets']);diagnostics=None
                    elif plan['branch']=='g1-native':
                        rows=g1.evaluate(case,plan['design']['budgets']);diagnostics=g1.gate_diagnostics(case)
                    elif plan['branch'] in ('g2-native','g2-transfer'):
                        rows=g2.evaluate(case,plan['design']['budgets'],plan['design']['query_counts']);diagnostics=None
                    elif plan['branch']=='g0-common':
                        rows=g0.common_matrix(case,plan['design']['budgets'],plan['design']['boundary_budgets'],plan['design']['balanced']);diagnostics=None
                    elif plan['branch']=='g3-representation':
                        rows=g3.evaluate(case,plan['design']['budgets'],plan['design']['storage_caps']);diagnostics=None
                    elif plan['branch']=='g3-stitch':
                        from . import g3_stitch
                        rows,diagnostics=g3_stitch.evaluate(case,root,plan['design']['budgets'],plan['design']['storage_caps'])
                    elif plan['branch']=='g4-history':
                        rows=g4.evaluate(case);diagnostics=None
                    elif plan['branch']=='structural-direct':
                        from . import direct
                        rows=direct.evaluate(case,case['budgets']);diagnostics=None
                    elif plan['branch']=='g2-mask':
                        from . import g2_mask
                        rows=g2_mask.evaluate(case,root.parent);diagnostics=None
                    elif plan['branch']=='g2-permuted':
                        from . import permuted
                        rows=permuted.evaluate(case,plan['design']['budgets'],plan['design']['query_counts']);diagnostics=None
                    elif plan['branch']=='g2-cyclic':
                        from . import cyclic_union
                        rows=cyclic_union.evaluate(case,plan['design']['budgets'],plan['design']['query_counts']);diagnostics=None
                    elif plan['branch']=='g2-cyclic-cost':
                        from . import cyclic_union
                        rows=cyclic_union.evaluate_cached_decision(
                            case,plan['design']['online_budget'],plan['design']['selector_budget']);diagnostics=None
                    else:
                        raise ValueError('unimplemented branch cannot be admitted')
                    units.append(dict(case=case,rows=rows,diagnostics=diagnostics))
                    emit(active_case=case['case_id'])
                data=dict(plan_sha256=file_digest(root/'PLAN.json'),branch=plan['branch'],units=units)
                retain(root,name,data,time.monotonic()-block_wall,time.process_time()-block_cpu,measured_children()-block_child)
                completed.append(name);new+=1;emit(blocks=completed)
            result=dict(execution_state='completed',instrument_state='pending_reconstruction',blocks=completed,
                        completed_at=now(),plan_sha256=file_digest(root/'PLAN.json'),campaign_state='other_commissioned_branches_pending')
            write(root/'COMPLETE.json',result);emit(state='completed');return result
        except BaseException as exc:
            emit(state='failed',error=f'{type(exc).__name__}: {exc}')
            write(root/'failures'/(attempt+'.json'),status)
            raise
        finally:
            emit()
