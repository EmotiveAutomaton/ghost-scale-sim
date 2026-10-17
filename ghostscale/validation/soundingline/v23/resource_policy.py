"""Hash-bound resource amendments; scientific definitions and old plans stay fixed.

New admissions resolve ACTIVE_RESOURCE_POLICY.json once and copy its binding into
the plan. Execution resolves that binding, never whichever policy is active later.
The original acceptance and October 15 reallocations remain independently checked.
"""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import math
import shutil
import zipfile
from .common import digest, read


LIMITS = dict(setup=8, repairs=14, n1=16, n2=7, n3=6, n4=14, n5=3,
              n6=6, n7=7, n8=2, twins=14, conditional=24)
SUPPLEMENT = 'ACCEPTANCE_SUPPLEMENT_2026-10-16.json'
SUPPLEMENT_17 = 'ACCEPTANCE_SUPPLEMENT_2026-10-17.json'
BINDING = 'acceptance_supplement'
PRIORITY = ('N4c', 'N4', 'N7', 'T1', 'T4', 'T3', 'T2', 'R3', 'N8')


def legacy_caps(campaign):
    result = dict(LIMITS)
    for path in sorted(Path(campaign).glob('RESOURCE_REALLOCATION*.json')):
        amendment = read(path)
        if (amendment.get('total_cpu_hours') != 90 or amendment.get('protected_cpu_hours') != 18
                or amendment.get('max_scientific_workers') != 3
                or amendment.get('authority') != 'owner-approved-2026-10-15'):
            raise ValueError('resource amendment lacks the accepted unchanged total and gear')
        result.update(amendment['group_caps_cpu_hours'])
    if set(result) != set(LIMITS) or any(not 0 < x <= 72 for x in result.values()):
        raise ValueError('invalid group capacity')
    return result


def binding(campaign):
    pointer = Path(campaign)/'ACTIVE_RESOURCE_POLICY.json'
    if not pointer.exists():
        return {}
    record = read(pointer)
    if record.get('schema') != 'v23-resource-policy-pointer-v1':
        raise ValueError('unknown active resource policy')
    result = {BINDING: record[BINDING]}
    resolve(campaign, result)
    return result


def supplement(campaign, bound):
    campaign = Path(campaign)
    if set(bound) != {'file', 'sha256'} or bound['file'] not in (SUPPLEMENT, SUPPLEMENT_17):
        raise ValueError('invalid acceptance supplement binding')
    path = campaign/bound['file']
    if digest(path) != bound['sha256']:
        raise ValueError('acceptance supplement hash changed')
    row = read(path)
    acceptance = read(campaign/'ACCEPTANCE.json')
    if (row.get('schema') != 'v23-resource-supplement-v1'
            or row.get('acceptance_sha256') != digest(campaign/'ACCEPTANCE.json')
            or row.get('scope') != 'resources_and_unambiguous_engineering_only'
            or not row.get('authority', {}).get('user_message')):
        raise ValueError('resource supplement authority or original acceptance changed')
    source = row['source_document']
    if digest(Path(source['path'])) != source['sha256']:
        raise ValueError('resource supplement source document changed')
    if (row['cpu_seconds'] != 285*3600 or row['protected_cpu_seconds'] != 18*3600
            or row['protected_cpu_seconds'] != acceptance['protected_cpu_seconds']
            or row['packet_cpu_cap_seconds'] != 14400
            or row['deadline'] != acceptance['deadline']
            or row['report_start'] != acceptance['report_start']):
        raise ValueError('resource supplement changes an unapproved bound')
    expected_caps = dict(n7=45, n4=44, twins=40, repairs=48, confirmation=40)
    if bound['file'] == SUPPLEMENT_17:
        expected_caps.update(conditional=24, n8=8)
    if row['group_caps_cpu_hours'] != expected_caps:
        raise ValueError('resource supplement group limits differ from authorization')
    expected = dict(timezone='America/Los_Angeles', day_start=8, night_start=22,
                    day_workers=8, night_workers=10, restricted_date='2026-10-21', restricted_workers=3)
    if row['worker_schedule'] != expected or row['unverified_interference_worker_ceiling'] != 3:
        raise ValueError('resource worker schedule changed')
    return row


def scheduled_workers(row, at=None):
    at = at or datetime.now(timezone.utc)
    if at.tzinfo is None:
        raise ValueError('resource scheduling requires an aware timestamp')
    schedule = row['worker_schedule']
    local = at.astimezone(ZoneInfo(schedule['timezone']))
    if local.date().isoformat() == schedule['restricted_date']:
        return schedule['restricted_workers']
    return schedule['day_workers'] if schedule['day_start'] <= local.hour < schedule['night_start'] else schedule['night_workers']


def resolve(campaign, plan=None, *, at=None):
    campaign = Path(campaign)
    original = read(campaign/'ACCEPTANCE.json')
    caps = legacy_caps(campaign)
    bound = binding(campaign).get(BINDING) if plan is None else plan.get(BINDING)
    if bound is None:
        return dict(original, group_caps_cpu_hours=dict(caps, confirmation=15))
    row = supplement(campaign, bound)
    caps.update(row['group_caps_cpu_hours'])
    scheduled = scheduled_workers(row, at)
    # No unmeasured claim that a busy sibling's throughput changed by <5%.
    # The expansion gate remains closed until a separate measured monitoring
    # contract is implemented and checked. Budget expansion works at three slots.
    ceiling = row['unverified_interference_worker_ceiling']
    return dict(original, cpu_seconds=row['cpu_seconds'],
                protected_cpu_seconds=row['protected_cpu_seconds'],
                max_scientific_workers=min(scheduled, ceiling),
                scheduled_scientific_workers=scheduled,
                worker_expansion_state='HELD_FOR_MEASURED_INTERFERENCE_CONTRACT',
                group_caps_cpu_hours=caps, **{BINDING: bound})


def group_caps(campaign):
    if not (Path(campaign)/'ACTIVE_RESOURCE_POLICY.json').exists():
        return legacy_caps(campaign)
    return resolve(campaign)['group_caps_cpu_hours']


def worker_limit(campaign, plan=None, *, at=None):
    current = resolve(campaign, at=at)['max_scientific_workers']
    # A legacy logical packet cannot silently acquire the new concurrency limit.
    return min(current, resolve(campaign, plan, at=at)['max_scientific_workers']) if plan is not None else current


def ordered(names):
    return sorted(names, key=lambda name: (PRIORITY.index(name) if name in PRIORITY else len(PRIORITY), name))


def rebind_successor(campaign, identity, plan, repository):
    """New immutable runtime capsule; every scientific predecessor byte survives."""
    bound = binding(campaign)
    if not bound:
        return plan
    original = Path(plan['source_archive']).with_suffix('')
    if digest(plan['source_archive']) != plan['source_archive_sha256']:
        raise ValueError('predecessor source archive changed')
    result = dict(plan)
    for relative, expected in plan['sources'].items():
        if digest(original/relative) != expected:
            raise ValueError('predecessor source capsule changed')
    source = Path(campaign)/'sources'/identity
    source.mkdir(parents=True, exist_ok=False)
    for relative in plan['sources']:
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original/relative, target)
    replacements = ['ghostscale/validation/soundingline/v23/'+name+'.py'
                    for name in ('runtime', 'resource_policy')]
    for relative in replacements:
        shutil.copyfile(Path(repository)/relative, source/relative)
    names = sorted(set(plan['sources']) | set(replacements))
    hashes = {name: digest(source/name) for name in names}
    preserved = {name: value for name, value in plan['sources'].items() if name not in replacements}
    if any(hashes[name] != value for name, value in preserved.items()):
        raise ValueError('resource migration altered scientific source')
    archive = source.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as zipped:
        for name in names:
            zipped.write(source/name, name)
    result.update(bound, sources=hashes, source_archive=str(archive), source_archive_sha256=digest(archive),
                  resource_runtime_migration=dict(predecessor_archive_sha256=plan['source_archive_sha256'],
                    changed_files=replacements, preserved_scientific_sources=preserved))
    return result


def projected_finish(cpu_seconds, workers, start, *, serial_cpu_seconds=0):
    """Scheduling lower bound, not a promise: no measured contention speedup."""
    from datetime import timedelta
    if workers < 1 or any(not math.isfinite(x) or x < 0 for x in (cpu_seconds, serial_cpu_seconds)):
        raise ValueError('invalid critical path inputs')
    return start + timedelta(seconds=serial_cpu_seconds + cpu_seconds/workers)
