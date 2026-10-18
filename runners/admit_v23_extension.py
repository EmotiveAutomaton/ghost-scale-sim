"""Bounded V23 continuation admission, preserving the original allocation.

Task-specific prerequisites are validated before freezing any scientific source.
This does not retry identities, waive validity, or reset clocks and accounting.
"""
import os
from pathlib import Path
import secrets
import shutil
import zipfile
from ghostscale.validation.soundingline.v23 import runtime, ledger, resource_policy
from ghostscale.validation.soundingline.v23.common import read, write, digest
from runners import queue_v23_families as queue
from runners.watch_v23_events import owner


def prerequisites(campaign, task, family, extras):
    from ghostscale.validation.soundingline.v23.family import verify_packet
    if task == 'numerical-validation':
        if family != 'N2':raise ValueError('numerical repair belongs to N2')
        root = Path(extras['calibration_reference'])
        _, summary = verify_packet(root)
        if summary['state'] != 'CALIBRATION_PASSED':raise ValueError('calibration unresolved')
        if digest(root/'COMPLETE.json') != extras['calibration_reference_complete_sha256']:
            raise ValueError('calibration reference changed')
    elif task=='n2-robust':
        from ghostscale.validation.soundingline.v23.n2_robust import validate
        validate(dict(extras,family=family))
        from ghostscale.validation.soundingline.v23 import resource_policy
        if resource_policy.binding(campaign).get(resource_policy.BINDING,{}).get('file')!=resource_policy.SUPPLEMENT_17:
            raise ValueError('D10 requires October 17 authority')
    elif task=='n2-rescore':
        from ghostscale.validation.soundingline.v23.adaptive_numerics import validate
        if family!='N2':raise ValueError('retained score repair belongs to N2')
        validate(extras)
        root=Path(extras['rescore_source']);prior,_=verify_packet(root)
        if digest(root/'COMPLETE.json')!=extras['rescore_source_complete_sha256']:raise ValueError('N2 source binding changed')
        for key in ('family','batch','worlds','cases','variants','fit_hashes','canary_key'):
            if prior[key]!=(family if key=='family' else extras[key]):raise ValueError('N2 reanalysis changed original design')
    elif task=='directed-review':
        from ghostscale.validation.soundingline.v23.directed_review import validate
        validate(extras,family,Path(__file__).resolve().parents[1])
    elif task.startswith(('population-','conditional-')):
        from ghostscale.validation.soundingline.v23.extension_contract import validate
        validate(task,family,extras,Path(__file__).resolve().parents[1])
    elif task in ('twin-profile','twin-fits','twin-batch'):
        from ghostscale.validation.soundingline.v23.twin_admission import validate
        validate(task,family,extras,Path(__file__).resolve().parents[1])
        engine=Path(extras['engine_reference_packet']);ep,es=verify_packet(engine)
        if ep['task']!='twin-engineering' or es['state']!='TWIN_ENGINEERING_VERIFIED' or digest(engine/'COMPLETE.json')!=extras['engine_reference_complete_sha256']:
            raise ValueError('verified pinned engine dependency required')
    else:
        raise ValueError('extension task is not implemented and checked: '+task)
    authority = campaign/'CONTINUATION_AUTHORITY_20261016.json'
    if not read(authority).get('n3_additional_reporting_repair_approved'):
        raise ValueError('continuation authority unavailable')
    return digest(authority)


def admit(campaign, identity, task, family, cap, extras, *, group='setup', launch=True):
    campaign = Path(campaign).resolve()
    if Path(identity).name != identity or not identity.startswith('v23-'):
        raise ValueError('unsafe packet identity')
    if not isinstance(extras, dict):raise ValueError('explicit task contract required')
    directed=family in ('N16','N16b','N17','N18')
    partitioned=(task=='twin-batch' and extras.get('partitioned_twin',False)) or (directed and task=='conditional-batch' and extras.get('partitioned_directed',False))
    if not 0 < cap or (cap>14400 and not partitioned):raise ValueError('bounded packet requires at most four CPU hours')
    extras=dict(extras)
    reserved_keys = {'task','family','budget_group','cpu_cap_seconds','wall_cap_seconds','protected',
        'acceptance_sha256','sources','source_archive','source_archive_sha256','dependency_files',
        'schema_directory','schema_pin','continuation_authority_sha256','external_source_files','extra_pythonpath',
        'scheduler_only','partition_layout','max_parallel_parts','world_indices',resource_policy.BINDING}
    if reserved_keys.intersection(extras):raise ValueError('task extras cannot override admission authority')
    with owner(campaign/'admission'):
        acceptance = resource_policy.resolve(campaign)
        if family=='N1' and resource_policy.binding(campaign):
            raise ValueError('N1 remains resource-deferred under the addendum; no readmission')
        if not acceptance['opened'] <= runtime.now() < acceptance['report_start']:
            raise ValueError('outside immutable scientific window')
        if runtime.environment() != read(campaign/'ENVIRONMENT.json'):
            raise ValueError('scientific environment changed')
        authority = prerequisites(campaign, task, family, extras)
        if directed:
            from ghostscale.validation.soundingline.v23.directed_common import CAPS
            directed_authority=read(campaign/'DIRECTED_AUTHORITY_20261018.json')
            repo=Path(__file__).resolve().parents[1]
            if digest(repo/directed_authority['source'])!=directed_authority['source_sha256'] or directed_authority['family_caps_seconds']!=CAPS:
                raise ValueError('October 18 source authority differs')
            prior=sum(float(read(p/'STATUS.json').get('cpu_seconds',0)) if (p/'STATUS.json').exists() and read(p/'STATUS.json').get('state') in ('failed','complete','verified')
                else float(read(p/'PLAN.json')['cpu_cap_seconds']) for p in (campaign/'packets').iterdir()
                if (p/'PLAN.json').exists() and read(p/'PLAN.json').get('family')==family and not read(p/'PLAN.json').get('logical_parent'))
            # Completed attempts are charged by the canonical cost ledger below;
            # this conservative family check additionally reserves the full next batch.
            from ghostscale.validation.soundingline.v23.extension_contract import validate as directed_validate
            _,future=directed_validate(task,family,extras,repo) if task!='directed-review' else (0,0)
            if prior+cap+future>CAPS[family]:raise ValueError('DIRECTED_FAMILY_CAP')
            extras['directed_authority_sha256']=digest(campaign/'DIRECTED_AUTHORITY_20261018.json')
        if (task.startswith('conditional-') or task=='directed-review') and family not in ('N4b','N4c'):
            if resource_policy.binding(campaign).get(resource_policy.BINDING,{}).get('file')!=resource_policy.SUPPLEMENT_17:
                raise ValueError('October 17 conditional authority required')
            if group!='conditional':raise ValueError('new conditional work must charge conditional group')
        if task=='n2-robust' and (group not in ('n2','conditional') or cap>5400):raise ValueError('D10 allocation changed')
        reservation=0.
        if task.startswith(('population-','conditional-')):
            from ghostscale.validation.soundingline.v23.extension_contract import validate
            minimum,reservation=validate(task,family,extras,Path(__file__).resolve().parents[1])
            if cap<minimum:raise ValueError('measured full replay cost exceeds extension cap')
        if task in ('twin-profile','twin-fits','twin-batch'):
            from ghostscale.validation.soundingline.v23.twin_admission import validate
            minimum,reservation=validate(task,family,extras,Path(__file__).resolve().parents[1])
            if cap<minimum:raise ValueError('measured full replay cost exceeds twin cap')
        if partitioned:
            if not resource_policy.binding(campaign):raise ValueError('twin partition resource supplement required')
            if directed:
                from ghostscale.validation.soundingline.v23.directed_common import estimate
                split=estimate(Path(extras['pilot_packet']),extras.get('directed_design',{})).get('partition_layout')
            else:
                from ghostscale.validation.soundingline.v23.twin_admission import estimate
                split=estimate(Path(extras['pilot_packet']),family,partitioned=True).get('partition_layout')
            if split:
                extras.update(scheduler_only=True,partition_layout=split,max_parallel_parts=3)
        state = queue.capacity(campaign, queue.inventory(campaign), cap+reservation, budget_group=group,
            needs_slot=not extras.get('scheduler_only'),
            replacing_burnin=(family,extras.get('batch')) if task in ('population-batch','conditional-batch','twin-batch') else None)
        if state != 'AVAILABLE':raise ValueError('continuation admission held: '+state)
        pin = read(campaign/'SCHEMA_PIN.json')
        if not pin.get('committed_files_verified'):raise ValueError('committed canonical schema required')
        ledger.canonical_schema(pin['directory'], pin)
        root = campaign/'packets'/identity
        if root.exists():raise ValueError('preserve existing packet identity')
        repo = Path(__file__).resolve().parents[1]
        names = ['ghostscale/__init__.py','ghostscale/validation/__init__.py',
            'ghostscale/validation/soundingline/__init__.py','runners/run_v23.py','runners/launch_background.py']
        names += [p.relative_to(repo).as_posix() for p in sorted((repo/'ghostscale/validation/soundingline/v23').glob('*.py'))]
        if task=='n2-robust':names.append('docs/versions/v23-research-division/N2_ROBUST_PROTOCOL.md')
        if (task.startswith('conditional-') or task=='directed-review') and family not in ('N4b','N4c'):
            protocol='docs/versions/v23-research-division/'+family.upper()+'_PROTOCOL.md'
            if not (repo/protocol).is_file():raise ValueError('declared protocol required before sampling')
            names.append(protocol)
        if directed:names.append(directed_authority['source'])
        if family=='N17':names.append('docs/versions/v23-research-division/N17_AGGREGATE_TARGETS.json')
        source = campaign/'sources'/identity;source.mkdir(parents=True, exist_ok=False)
        for name in names:
            target = source/name;target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/name, target)
        archive = source.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as zipped:
            for name in names:zipped.write(source/name, name)
        canary=extras.pop('canary_key',None) or secrets.token_hex(32)
        plan = dict(task=task, family=family, budget_group=group, cpu_cap_seconds=cap,
            wall_cap_seconds=4*cap, protected=False, acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
            sources={name:digest(source/name) for name in names}, source_archive=str(archive),
            source_archive_sha256=digest(archive), schema_directory=pin['directory'], schema_pin=pin,
            dependency_files={str(p):digest(p) for p in (campaign/'dependencies').rglob('*')
                if p.is_file() and p.suffix in ('.py','.pyd','.dll')},
            continuation_authority_sha256=authority, canary_key=canary, **extras, **resource_policy.binding(campaign))
        if task in ('twin-profile','twin-fits','twin-batch'):
            reference=read(Path(extras['engine_reference_packet'])/'PLAN.json')
            for path,value in reference['external_source_files'].items():
                if digest(path)!=value:raise ValueError('verified engine snapshot changed')
            plan.update(external_source_files=reference['external_source_files'],extra_pythonpath=reference['extra_pythonpath'])
            plan['dependency_files'].update({str(p):digest(p) for p in (campaign/'bridge-dependencies').rglob('*')
                if p.is_file() and p.suffix in ('.py','.pyd','.dll')})
        write(root/'PLAN.json', plan)
        write(root/'ADMISSION.json', dict(at=runtime.now(), plan_sha256=digest(root/'PLAN.json'),
            continuation_authority_sha256=authority, charged_cpu_seconds=runtime.cost(campaign),
            new_cap_seconds=cap, reserved_successor_seconds=reservation,capacity=state))
        if launch and not plan.get('scheduler_only'):queue.launch(root, campaign, source)
        return root
