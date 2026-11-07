"""One owner-approved G3-T09 map; preserve the closed queue and held attempt.

DESIGN CHECK: original G3-T09 protocol, exact case check and known-answer controls
are unchanged. Out-of-case status is retained, never interpreted as a null.
Resource controls refuse undeclared cards, larger caps, shared-slot overflow and
expired windows; the approved held card must pass within its one-hour ceiling.
Lessons consulted: statistic/model/infrastructure sections 3-5, especially
immutable dependency receipts and actual producer-to-admission validation.
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from ghostscale.validation.soundingline.v26 import day, day_reporting
from ghostscale.validation.soundingline.v26.common import read, write, digest
from ghostscale.validation.soundingline.v23 import runtime
from runners import run_v26 as base, run_v26_day as parent, run_v26_friday as friday
from runners.run_v26_followups import verify_complete
from runners.watch_v23_events import alive, event

REL = 'g3-t09-20261107'
HELD = 'v26-day-20261106-g3-t09-probe'


def prepare_extension(campaign, predecessor, repo):
    started = time.process_time()
    if campaign.exists():
        raise ValueError('extension already exists; do not reprepare')
    if read(predecessor/'QUEUE_STATUS.json')['state'] != 'DRAINED':
        raise ValueError('predecessor must be drained')
    held = predecessor/'packets'/HELD
    if read(held/'STATUS.json')['state'] != 'ADMISSION_HELD' or (held/'ADMISSION.json').exists():
        raise ValueError('expected unexecuted resource hold')
    old = read(held/'PLAN.json')
    parent.verify_bindings(predecessor, held, old)
    for name in ('ACCEPTANCE.json', 'ENVIRONMENT.json', 'NUMERICAL_ENVIRONMENT.json'):
        write(campaign/name, read(predecessor/name))
    write(campaign/'LOCAL_PATHS.json', dict(repo=str(repo), predecessor=str(predecessor)))
    write(campaign/'PARENT_CHARGES.json', dict(charges=parent.charges(predecessor),
        bound_records={str(predecessor/name):digest(predecessor/name)
            for name in ('ACCEPTANCE.json','FINAL_VALIDITY_20261107.json')}))
    accepted = read(campaign/'ACCEPTANCE.json')
    supplement = dict(schema='v26-day-g3-t09-allocation-20261107', card='G3-T09',
        at=base.now().isoformat(), owner_decision='Run the declared map within the existing total',
        original_acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
        held_plan_sha256=digest(held/'PLAN.json'), held_status_sha256=digest(held/'STATUS.json'),
        previous_texture_seconds=1800, texture_seconds=3600,
        total_cpu_seconds=110*3600, pools_seconds=day.POOLS,
        new_starts_close=accepted['new_starts_close'], drain=accepted['drain'],
        sample_and_scientific_seed_unchanged=True, no_descendants=True)
    write(campaign/'RESOURCE_SUPPLEMENT.json', supplement)
    write(repo/'results/v26'/day.REL/REL/'RESOURCE_SUPPLEMENT.json', supplement)
    write(campaign/'HELD_PLAN.json', old)
    for directory, name in [('protocols',old['protocol_file']),('designs',old['design_file'])]:
        write(campaign/directory/name, read(predecessor/directory/name))
    write(campaign/'controls/G3.json', read(held/'CONTROLS_BEFORE_DATA.json'))
    source = base.frozen_source(campaign, parent.EXTRAS+('runners/run_v26_day_map.py',))
    write(campaign/'CONTROLLER_SOURCE.json', dict(source=str(source), manifest_sha256=digest(source/'SOURCE_MANIFEST.json')))
    write(campaign/'accounting/preparation.json',dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started))
    prepare_packet(campaign)
    export(campaign, 'PREPARED')


def prepare_packet(campaign, probe=None):
    old = read(campaign/'HELD_PLAN.json')
    plan = dict(old)
    if probe is not None:
        verify_complete(probe)
        caps = parent.measured_caps(probe,16)
        plan.update(kind='texture', scientific_kind='texture', blocks=16,
            pass_caps=caps, cpu_cap_seconds=sum(caps.values())+10,
            projected_wall_seconds=1.5*read(campaign/'accounting'/(probe.name+'.json'))['wall_seconds']*16+900,
            cost_probe=probe.name,cost_probe_complete_sha256=digest(probe/'COMPLETE.json'))
    # Held probe never drew data: preserve its prospective science seed, with
    # a distinct execution identity and allocation in the separate campaign.
    science = old['science_identity'] if probe is None else old['science_identity'].removesuffix('-probe')+'-study'
    plan.update(identity=science+'-allocation-20261107',science_identity=science,
        parent_charges_sha256=digest(campaign/'PARENT_CHARGES.json'),
        resource_supplement_sha256=digest(campaign/'RESOURCE_SUPPLEMENT.json'))
    packet = campaign/'packets'/plan['identity']
    if packet.exists():
        raise ValueError('packet already exists; no restart authorized')
    write(packet/'PLAN.json',plan)
    write(packet/'CONTROLS_BEFORE_DATA.json',read(campaign/'controls/G3.json'))
    write(packet/'STATUS.json',dict(state='PREPARED',at=base.now().isoformat()))
    return packet


def verify(campaign, packet):
    plan = read(packet/'PLAN.json')
    parent.verify_bindings(campaign,packet,plan)
    s = read(campaign/'RESOURCE_SUPPLEMENT.json')
    if digest(campaign/'RESOURCE_SUPPLEMENT.json') != plan['resource_supplement_sha256'] or s['original_acceptance_sha256'] != digest(campaign/'ACCEPTANCE.json'):
        raise ValueError('resource supplement binding changed')
    old = read(campaign/'HELD_PLAN.json')
    if digest(campaign/'HELD_PLAN.json') != s['held_plan_sha256']:
        raise ValueError('held plan changed')
    if plan['settings'] != old['settings'] or plan['source_manifest_sha256'] != old['source_manifest_sha256']:
        raise ValueError('declared science changed')
    return plan, s


def export(campaign, state, active=None):
    repo = Path(read(campaign/'LOCAL_PATHS.json')['repo'])
    out = repo/'results/v26'/day.REL
    rows=[]
    for path in sorted((campaign/'packets').glob('*/PLAN.json')):
        packet=path.parent; p=read(path); status=read(packet/'STATUS.json')
        rows.append(dict(packet=packet.name,card=p['card'],kind=p['kind'],state=status['state']))
        if (packet/'COMPLETE.json').exists() and not (packet/'AGGREGATE_EXPORTED.json').exists():
            verify_complete(packet)
            for name,relative in [('SUMMARY.json','original/SUMMARY.json'),('RAW_INDEX.json','original/RAW_INDEX.json'),('COMPLETE.json','COMPLETE.json'),('REPLAY.json','REPLAY.json'),('INDEPENDENT_REVIEW.json','INDEPENDENT_REVIEW.json'),('PASS_COSTS.json','PASS_COSTS.json'),('CONTROLS.json','CONTROLS_BEFORE_DATA.json')]:
                write(out/packet.name/name,read(packet/relative))
            write(out/packet.name/'PUBLIC_PLAN.json',friday.public_plan(p))
            day_reporting.land(campaign,packet,repo)
            write(packet/'AGGREGATE_EXPORTED.json',dict(at=base.now().isoformat()))
    status=dict(at=base.now().isoformat(),state=state,pid=os.getpid(),gear_limit=2,
        active=active or [],packets=rows,pools_charged_seconds=parent.charges(campaign),
        pool_ceilings_seconds=day.POOLS,no_descendants=True)
    write(campaign/'QUEUE_STATUS.json',status,immutable=False)
    write(out/REL/'CURRENT_STATUS.json',status,immutable=False)


def queue(campaign):
    runtime.below_normal(); started=time.process_time(); lock=campaign/'QUEUE_OWNER.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    state='FAILED'
    try:
        for phase in ('probe','texture'):
            candidates=[p.parent for p in (campaign/'packets').glob('*/PLAN.json')
                if read(p.parent/'STATUS.json')['state']=='PREPARED' and read(p)['kind']==phase]
            if len(candidates)!=1:raise ValueError('exactly one declared candidate required')
            packet=candidates[0];p,s=verify(campaign,packet)
            external=parent.outside(campaign,[])
            predecessor=Path(read(campaign/'LOCAL_PATHS.json')['predecessor'])
            for path in (predecessor/'packets').glob('*/STATUS.json'):
                row=read(path);pid=row.get('parent_pid',row.get('launcher_pid'))
                if row['state'] in ('RUNNING','LAUNCHING') and pid and alive(pid):external+=1
            try:
                day.admission(p,read(campaign/'ACCEPTANCE.json'),base.now(),parent.charges(campaign),external,s)
            except ValueError as exc:
                write(packet/'STATUS.json',dict(state='ADMISSION_HELD',reason=str(exc),at=base.now().isoformat()),immutable=False)
                raise
            write(packet/'ADMISSION.json',dict(at=base.now().isoformat(),plan_sha256=digest(packet/'PLAN.json'),
                resource_supplement_sha256=digest(campaign/'RESOURCE_SUPPLEMENT.json'),charges_before=parent.charges(campaign)))
            write(campaign/'accounting'/(packet.name+'.json'),dict(pool=p['pool'],kind=p['kind'],family=p['family'],lane='texture',state='RESERVED',reserved_cpu_seconds=p['cpu_cap_seconds']))
            with (packet/'parent.log').open('xb') as log:
                proc=subprocess.Popen([p['interpreter'],'-B','-m','runners.run_v26','--packet',str(packet)],
                    cwd=p['source_root'],env=base.env_for(p['source_root'])|dict(NUMBA_NUM_THREADS='1'),
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                while proc.poll() is None:
                    export(campaign,'RUNNING',[packet.name]);time.sleep(5)
            cost=campaign/'accounting'/(packet.name+'.json');row=read(cost);row['lane']='texture';write(cost,row,immutable=False)
            if proc.returncode or read(packet/'STATUS.json')['state']!='COMPLETE':
                raise ValueError('packet failed; retained for diagnosed review')
            verify_complete(packet)
            export(campaign,'BETWEEN_PASSES')
            event(campaign,packet.name+'-complete','batch_complete',dict(packet=packet.name,complete_sha256=digest(packet/'COMPLETE.json')))
            if phase=='probe':prepare_packet(campaign,packet)
        state='DRAINED'
    except BaseException as exc:
        write(campaign/'FAILURE.json',dict(at=base.now().isoformat(),error=str(exc)))
        event(campaign,'map-failed','review_boundary',dict(reason=str(exc)))
        raise
    finally:
        write(campaign/'accounting/controller.json',dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='STOPPED'))
        export(campaign,state)
        event(campaign,'map-terminal','review_boundary',dict(state=state,no_descendants=True))
        lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--predecessor',type=Path)
    parser.add_argument('--repo',type=Path);parser.add_argument('--queue',action='store_true');args=parser.parse_args()
    if args.prepare:prepare_extension(args.campaign.resolve(),args.predecessor.resolve(),args.repo.resolve())
    elif args.queue:queue(args.campaign.resolve())
    else:parser.error('choose --prepare or --queue')
