"""Finite November 3 follow-ups; the closed V26 primary queue is never reopened."""
import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from ghostscale.validation.soundingline.v26 import contract, study
from ghostscale.validation.soundingline.v26.common import read, write, digest
from runners import run_v26 as base
from runners.watch_v23_events import event

FAMILIES = ('E1', 'U1')
LIMITS = {'E1': 9000., 'U1': 1800.}


def charges(campaign):
    totals = dict(read(campaign / 'PARENT_CHARGES.json')['charges'])
    for key, value in base.charges(campaign).items():
        totals[key] = totals.get(key, 0.) + value
    return totals


def family_charged(campaign, family):
    return sum(max(r.get('cpu_seconds', 0), r.get('reserved_cpu_seconds', 0))
               for r in (read(p) for p in (campaign / 'accounting').glob('*.json'))
               if r.get('family') == family and r.get('kind') == 'followup')


def prepare(campaign, family, source, probe=None):
    started = time.process_time()
    protocol = read(campaign / 'protocols' / (family + '.json'))
    settings = protocol['settings']
    is_probe = probe is None
    identity = f'v26-{family.lower()}-followup-20261103-' + ('probe' if is_probe else 'study')
    packet = campaign / 'packets' / identity
    if packet.exists():
        raise ValueError('existing packet requires explicit recovery review')
    if is_probe:
        rows = [settings[0], settings[-1]]
        blocks = 1
        caps = dict(original=590., replay=590., review=590.)
        projection = 1800.
    else:
        complete = read(probe / 'COMPLETE.json')
        verify_complete(probe, complete)
        measured = read(probe / 'PASS_COSTS.json')
        rows, blocks = settings, protocol['blocks']
        caps = {}
        for label, costs in measured['setting_seconds'].items():
            overhead = max(0., measured['pass_cpu_seconds'][label] - sum(costs))
            caps[label] = 1.5 * (overhead + max(max(costs), 1/64) * len(rows) * blocks) + 10
        used = read(campaign / 'accounting' / (probe.name + '.json'))
        projection = 1.5 * used['wall_seconds'] * len(rows) * blocks / len(read(probe / 'PLAN.json')['settings'])
    # Charge probe, original execution, replay and independent review to the same
    # follow-up allowance. Parent-family half ceilings and the total three hours hold.
    if family_charged(campaign, family) + sum(caps.values()) + 10 > LIMITS[family]:
        raise ValueError('follow-up family reservation exceeds declared allocation')
    controls = study.module(family).controls()
    if not controls['passed'] or not read(campaign / 'designs' / (family + '.json'))['passed']:
        raise ValueError('control/design check failed')
    packet.mkdir(parents=True)
    write(packet / 'CONTROLS_BEFORE_DATA.json', controls)
    plan = dict(family=family, kind='followup', scientific_kind='probe' if is_probe else 'followup',
        generation=1, pool='followup_healing', identity=identity, science_identity=identity,
        settings=rows, blocks=blocks, source_root=str(source),
        source_manifest_sha256=digest(source / 'SOURCE_MANIFEST.json'), source_archive_sha256=digest(source / 'SOURCE.zip'),
        acceptance_sha256=digest(campaign / 'ACCEPTANCE.json'), environment_sha256=digest(campaign / 'ENVIRONMENT.json'),
        protocol_sha256=digest(campaign / 'protocols' / (family + '.json')),
        design_check_sha256=digest(campaign / 'designs' / (family + '.json')),
        controls_sha256=digest(packet / 'CONTROLS_BEFORE_DATA.json'),
        numerical_threads=1, priority='below-normal', cpu_cap_seconds=sum(caps.values())+10,
        pass_caps=caps, wall_cap_seconds=6*3600, projected_wall_seconds=projection,
        interpreter=protocol['interpreter'], confirmation=False,
        primary_complete_sha256=protocol['parent_complete_sha256'],
        cost_probe=probe.name if probe else None,
        cost_probe_complete_sha256=digest(probe / 'COMPLETE.json') if probe else None)
    write(packet / 'PLAN.json', plan)
    write(packet / 'STATUS.json', dict(state='PREPARED', at=base.now().isoformat()))
    write(campaign / 'accounting' / (identity + '-preparation.json'),
          dict(pool='preparation', kind='preparation', family=family, cpu_seconds=time.process_time()-started))
    return packet


def verify_complete(packet, complete=None):
    complete = complete or read(packet / 'COMPLETE.json')
    for key, name in [('plan_sha256','PLAN.json'), ('summary_sha256','original/SUMMARY.json'),
                      ('replay_sha256','REPLAY.json'), ('review_sha256','INDEPENDENT_REVIEW.json')]:
        if digest(packet / name) != complete[key]:
            raise ValueError('completed packet binding changed: ' + name)


def verify_bindings(campaign, packet, plan):
    base.verify_admission_bindings(campaign, packet, plan)
    parent = read(campaign / 'PARENT_CHARGES.json')
    root = Path(parent['parent_campaign'])
    if digest(root / 'CLOSEOUT.json') != parent['closeout_sha256'] or base.charges(root) != parent['charges']:
        raise ValueError('closed parent/accounting changed; reconciliation required')
    primary = root / 'packets' / ('v26-' + plan['family'].lower() + '-study-1')
    verify_complete(primary)
    if digest(primary / 'COMPLETE.json') != plan['primary_complete_sha256']:
        raise ValueError('primary binding changed')
    if plan.get('cost_probe'):
        probe = campaign / 'packets' / plan['cost_probe']
        verify_complete(probe)
        if digest(probe / 'COMPLETE.json') != plan['cost_probe_complete_sha256']:
            raise ValueError('cost probe binding changed')
    if family_charged(campaign, plan['family']) + plan['cpu_cap_seconds'] > LIMITS[plan['family']]:
        raise ValueError('follow-up cumulative family ceiling')


def export(campaign):
    repo = Path(read(campaign / 'LOCAL_PATHS.json')['repo'])
    out = repo / 'results/v26/followups-20261103'
    out.mkdir(parents=True, exist_ok=True)
    packets = []
    for path in sorted((campaign / 'packets').glob('*/PLAN.json')):
        p, plan = path.parent, read(path)
        state = read(p / 'STATUS.json')
        packets.append(dict(packet=p.name, family=plan['family'], scientific_kind=plan['scientific_kind'],
                            state=state['state'], phase=state.get('phase'), cap_seconds=plan['cpu_cap_seconds']))
        if (p / 'COMPLETE.json').exists() and not (p / 'AGGREGATE_EXPORTED.json').exists():
            verify_complete(p)
            dest = out / p.name
            dest.mkdir(exist_ok=True)
            for name, relative in [('SUMMARY.json','original/SUMMARY.json'), ('RAW_INDEX.json','original/RAW_INDEX.json'),
                ('COMPLETE.json','COMPLETE.json'), ('REPLAY.json','REPLAY.json'), ('INDEPENDENT_REVIEW.json','INDEPENDENT_REVIEW.json'),
                ('PASS_COSTS.json','PASS_COSTS.json'), ('CONTROLS.json','CONTROLS_BEFORE_DATA.json')]:
                write(dest / name, read(p / relative))
            # Public plan excludes machine-specific paths; exact original plan hash
            # remains in COMPLETE and its immutable local packet.
            public = {k:v for k,v in plan.items() if k not in ('source_root','interpreter')}
            write(dest / 'PUBLIC_PLAN.json', public)
            write(p / 'AGGREGATE_EXPORTED.json', dict(at=base.now().isoformat(), destination=dest.relative_to(repo).as_posix()))
    q = read(campaign / 'QUEUE_STATUS.json')
    write(out / 'CURRENT_STATUS.json', dict(at=base.now().isoformat(), state=q['state'], active=q.get('active',[]),
        packets=packets, gear=2, workers_maximum=2, pools_charged_seconds=charges(campaign),
        pool_ceilings_seconds=contract.POOLS, extension_charged_seconds=base.charges(campaign),
        followup_family_limits_seconds=LIMITS, new_starts_close=contract.NEW_STARTS_CLOSE, drain=contract.DRAIN,
        primary_queue='closed and preserved', fully_built=(campaign/'BUILD_COMPLETE.json').exists()), immutable=False)


def queue(campaign):
    if os.name != 'nt':
        raise RuntimeError('native Windows ownership/accounting required')
    started = time.process_time()
    lock = campaign / 'QUEUE_OWNER.lock'
    account = campaign / 'accounting' / f'controller-{os.getpid()}-{time.time_ns()}.json'
    active = {}
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        while True:
            for name, (proc, log) in list(active.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del active[name]
                packet = campaign / 'packets' / name
                state, plan = read(packet/'STATUS.json'), read(packet/'PLAN.json')
                if state['state'] == 'COMPLETE':
                    event(campaign, name+'-complete', 'batch_complete', dict(packet=name, complete_sha256=digest(packet/'COMPLETE.json')))
                    if plan['scientific_kind'] == 'probe':
                        try:
                            prepare(campaign, plan['family'], Path(plan['source_root']), packet)
                        except Exception as exc:
                            write(packet/'CONTINUATION_FAILURE.json', dict(error=f'{type(exc).__name__}: {exc}'))
                            event(campaign, name+'-continuation-failure', 'admission_failure', dict(packet=name,error=str(exc)))
                else:
                    if state['state'] not in ('FAILED','ADMISSION_HELD'):
                        write(packet/'STATUS.json',dict(state='FAILED',reason='parent exited; reservation retained',exit_code=proc.returncode),immutable=False)
                    event(campaign, name+'-failure', 'packet_failure', dict(packet=name,status_sha256=digest(packet/'STATUS.json')))
            candidates = [p.parent for p in sorted((campaign/'packets').glob('*/PLAN.json'))
                          if read(p.parent/'STATUS.json')['state']=='PREPARED']
            stopping = (campaign/'STOP').exists() or base.now()>=datetime.fromisoformat(contract.NEW_STARTS_CLOSE)
            if not stopping and not (campaign/'PAUSE_ADMISSIONS').exists():
                for packet in candidates:
                    outside = base.outside_scientists(campaign)
                    if len(active)+outside >= 2:
                        break
                    plan = read(packet/'PLAN.json')
                    try:
                        verify_bindings(campaign, packet, plan)
                        contract.admission(plan, read(campaign/'ACCEPTANCE.json'), base.now(), charges(campaign),running=len(active)+outside)
                    except ValueError as exc:
                        write(packet/'STATUS.json',dict(state='ADMISSION_HELD',reason=str(exc),at=base.now().isoformat()),immutable=False)
                        event(campaign,packet.name+'-held','admission_failure',dict(packet=packet.name,reason=str(exc)))
                        continue
                    write(packet/'ADMISSION.json',dict(at=base.now().isoformat(),plan_sha256=digest(packet/'PLAN.json'),charges_before=charges(campaign)))
                    write(campaign/'accounting'/(packet.name+'.json'),dict(pool=plan['pool'],kind=plan['kind'],family=plan['family'],state='RESERVED',reserved_cpu_seconds=plan['cpu_cap_seconds']))
                    log=(packet/'parent.log').open('xb')
                    proc=subprocess.Popen([plan['interpreter'],'-B','-m','runners.run_v26','--packet',str(packet)],
                        cwd=plan['source_root'],env=base.env_for(plan['source_root']),stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                    active[packet.name]=(proc,log)
                    write(packet/'STATUS.json',dict(state='LAUNCHING',launcher_pid=proc.pid,at=base.now().isoformat()),immutable=False)
            remaining=[p for p in candidates if read(p/'STATUS.json')['state']=='PREPARED']
            drained=not active and (stopping or ((campaign/'BUILD_COMPLETE.json').exists() and not remaining))
            write(campaign/'QUEUE_STATUS.json',dict(state='DRAINED' if drained else 'RUNNING' if active else 'WAITING',
                pid=os.getpid(),heartbeat=base.now().isoformat(),active=list(active),prepared=[p.name for p in remaining],gear=2),immutable=False)
            write(account,dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
            export(campaign)
            if drained:
                event(campaign,'v26-followups-queue-drained','review_boundary',dict(finite_scope=True))
                break
            time.sleep(10)
    finally:
        write(account,dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='STOPPED'),immutable=False)
        lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args()
    queue(args.campaign.resolve())
