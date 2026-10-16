"""Bounded V23 continuation admission, preserving the original allocation.

Task-specific prerequisites are validated before freezing any scientific source.
This does not retry identities, waive validity, or reset clocks and accounting.
"""
import os
from pathlib import Path
import secrets
import shutil
import zipfile
from ghostscale.validation.soundingline.v23 import runtime, ledger
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
    elif task=='n2-rescore':
        from ghostscale.validation.soundingline.v23.adaptive_numerics import validate
        if family!='N2':raise ValueError('retained score repair belongs to N2')
        validate(extras)
        root=Path(extras['rescore_source']);prior,_=verify_packet(root)
        if digest(root/'COMPLETE.json')!=extras['rescore_source_complete_sha256']:raise ValueError('N2 source binding changed')
        for key in ('family','batch','worlds','cases','variants','fit_hashes','canary_key'):
            if prior[key]!=(family if key=='family' else extras[key]):raise ValueError('N2 reanalysis changed original design')
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
    if not 0 < cap <= 14400:raise ValueError('bounded packet requires at most four CPU hours')
    if not isinstance(extras, dict):raise ValueError('explicit task contract required')
    extras=dict(extras)
    reserved_keys = {'task','family','budget_group','cpu_cap_seconds','wall_cap_seconds','protected',
        'acceptance_sha256','sources','source_archive','source_archive_sha256','dependency_files',
        'schema_directory','schema_pin','continuation_authority_sha256','external_source_files','extra_pythonpath'}
    if reserved_keys.intersection(extras):raise ValueError('task extras cannot override admission authority')
    with owner(campaign/'admission'):
        acceptance = read(campaign/'ACCEPTANCE.json')
        if not acceptance['opened'] <= runtime.now() < acceptance['report_start']:
            raise ValueError('outside immutable scientific window')
        if runtime.environment() != read(campaign/'ENVIRONMENT.json'):
            raise ValueError('scientific environment changed')
        authority = prerequisites(campaign, task, family, extras)
        reservation=0.
        if task.startswith(('population-','conditional-')):
            from ghostscale.validation.soundingline.v23.extension_contract import validate
            minimum,reservation=validate(task,family,extras,Path(__file__).resolve().parents[1])
            if cap<minimum:raise ValueError('measured full replay cost exceeds extension cap')
        if task in ('twin-profile','twin-fits','twin-batch'):
            from ghostscale.validation.soundingline.v23.twin_admission import validate
            minimum,reservation=validate(task,family,extras,Path(__file__).resolve().parents[1])
            if cap<minimum:raise ValueError('measured full replay cost exceeds twin cap')
        state = queue.capacity(campaign, queue.inventory(campaign), cap+reservation, budget_group=group,
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
            continuation_authority_sha256=authority, canary_key=canary, **extras)
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
        if launch:queue.launch(root, campaign, source)
        return root
