"""Separate, bounded Friday exploration; immutable cards, measured admission, Gear 2."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ghostscale.validation.soundingline.v26 import friday, registry, study
from ghostscale.validation.soundingline.v26.common import read, write, digest
from ghostscale.validation.soundingline.v23 import runtime
from runners import run_v26 as base
from runners.run_v26_followups import verify_complete
from runners.watch_v23_events import event, alive

REL='friday-20261104'
EXTRAS=('runners/run_v26_friday.py','runners/run_v26_followups.py')


def charges(campaign):
    totals=dict(read(campaign/'PARENT_CHARGES.json')['charges'])
    for k,v in base.charges(campaign).items():totals[k]=totals.get(k,0)+v
    return totals


def measured_study_caps(cost, settings, blocks):
    """Reserve deterministic execution passes against the slower measured twin."""
    caps={}
    for label,units in cost['setting_seconds'].items():
        overhead=max(0.,cost['pass_cpu_seconds'][label]-sum(units))
        caps[label]=1.5*(overhead+max(max(units),1/64)*len(settings)*blocks)+10
    # Original and replay execute the same frozen worlds.  A one-world probe can
    # make either twin look slightly faster, so reserving them independently can
    # stop a valid replay even when the original fits.  Bind both to the slower
    # prospective estimate; review remains separately measured.
    execution=max(caps['original'],caps['replay'])
    caps['original']=execution;caps['replay']=execution
    return caps


def prepare(campaign, card, source, probe=None):
    started=time.process_time();cid=card['id'];family=card['family'];is_probe=probe is None
    identity=card['seed_domain']+('-probe' if is_probe else '-study')
    packet=campaign/'packets'/identity
    if packet.exists():raise ValueError('existing packet requires recovery review')
    if not read(campaign/'designs'/(cid+'.json'))['passed']:raise ValueError('design cannot produce the case')
    settings=card['settings'];blocks=1 if is_probe else card['blocks']
    if is_probe:
        caps=dict(original=590.,replay=590.,review=590.);projection=1800.
    else:
        verify_complete(probe);cost=read(probe/'PASS_COSTS.json');caps={}
        caps=measured_study_caps(cost,settings,blocks)
        prior=read(campaign/'accounting'/(probe.name+'.json'))
        projection=1.5*prior['wall_seconds']*blocks+900
    controls=read(campaign/'controls'/(family+'.json'))
    if not controls['passed']:raise ValueError('known-answer failure')
    packet.mkdir(parents=True);write(packet/'CONTROLS_BEFORE_DATA.json',controls)
    protocol=campaign/'protocols'/(cid+'.json')
    plan=dict(family=family,card=cid,kind='probe' if is_probe else 'study',scientific_kind='probe' if is_probe else 'exploratory',
        pool='studies',identity=identity,science_identity=identity,settings=settings,blocks=blocks,
        source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),source_archive_sha256=digest(source/'SOURCE.zip'),
        acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),environment_sha256=digest(campaign/'ENVIRONMENT.json'),
        supplement_sha256=digest(campaign/'SUPPLEMENT.json'),parent_charges_sha256=digest(campaign/'PARENT_CHARGES.json'),
        protocol_sha256=digest(protocol),design_check_sha256=digest(campaign/'designs'/(cid+'.json')),
        controls_sha256=digest(packet/'CONTROLS_BEFORE_DATA.json'),numerical_threads=1,priority='below-normal',
        cpu_cap_seconds=sum(caps.values())+10,pass_caps=caps,wall_cap_seconds=18*3600,
        projected_wall_seconds=projection,interpreter=read(protocol)['interpreter'],confirmation=False,
        cost_probe=probe.name if probe else None,cost_probe_complete_sha256=digest(probe/'COMPLETE.json') if probe else None,
        queue_order=card['queue_order'])
    write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='PREPARED',at=base.now().isoformat()))
    write(campaign/'accounting'/(identity+'-preparation.json'),dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started))
    return packet


def prepare_verification_healing(campaign, failed, source, healing_number=1):
    """Freeze replay/review healing around a complete retained original pass."""
    started=time.process_time();failed=failed.resolve();parent=read(failed/'PLAN.json');status=read(failed/'STATUS.json')
    if status['state']!='FAILED' or (failed/'COMPLETE.json').exists():raise ValueError('healing requires one incomplete failed packet')
    if not (failed/'original-cost.json').exists() or not (failed/'original/RAW_INDEX.json').exists() or not (failed/'original/SUMMARY.json').exists():raise ValueError('failed packet has no complete retained original')
    expected=len(parent['settings'])*parent['blocks']
    if len(read(failed/'original/RAW_INDEX.json'))!=expected:raise ValueError('retained original is incomplete')
    if healing_number not in (1,2):raise ValueError('healing limit')
    identity=failed.name+f'-healing-{healing_number}';packet=campaign/'packets'/identity
    if packet.exists():raise ValueError('existing healing packet requires recovery review')
    original_cpu=read(failed/'original-cost.json')['cpu_seconds'];execution_cap=1.5*original_cpu+30
    caps=dict(original=1.,replay=execution_cap,review=execution_cap)
    packet.mkdir(parents=True);shutil.copytree(failed/'original',packet/'original');shutil.copy2(failed/'original-cost.json',packet/'original-cost.json')
    shutil.copy2(failed/'CONTROLS_BEFORE_DATA.json',packet/'CONTROLS_BEFORE_DATA.json')
    plan=parent|dict(kind='healing',scientific_kind='exploratory',pool='followup_healing',identity=identity,
        source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),source_archive_sha256=digest(source/'SOURCE.zip'),
        healing_parent=failed.name,verification_healing_parent=failed.name,healing_number=healing_number,accounting_root=str(campaign),
        failed_status_sha256=digest(failed/'STATUS.json'),retained_original_raw_index_sha256=digest(failed/'original/RAW_INDEX.json'),
        retained_original_summary_sha256=digest(failed/'original/SUMMARY.json'),retained_original_cost_sha256=digest(failed/'original-cost.json'),
        cpu_cap_seconds=sum(caps.values())+10,pass_caps=caps,projected_wall_seconds=2*read(campaign/'accounting'/(failed.name+'.json'))['wall_seconds']+900)
    write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='HEALING_QUEUED',at=base.now().isoformat(),reason='verification-only healing; retained original scientific draw is not repeated'))
    write(campaign/'accounting'/(identity+'-preparation.json'),dict(pool='preparation',kind='preparation',family=plan['family'],cpu_seconds=time.process_time()-started))
    return packet


def run_verification_healing(campaign,packet):
    plan=read(packet/'PLAN.json')
    if read(packet/'STATUS.json')['state']!='HEALING_QUEUED':raise ValueError('healing packet is not queued')
    external=outside(campaign)
    friday.admission(plan,read(campaign/'ACCEPTANCE.json'),read(campaign/'SUPPLEMENT.json'),base.now(),charges(campaign),external)
    verify_bindings(campaign,packet,plan)
    write(packet/'ADMISSION.json',dict(at=base.now().isoformat(),plan_sha256=digest(packet/'PLAN.json'),charges_before=charges(campaign)))
    write(campaign/'accounting'/(packet.name+'.json'),dict(pool=plan['pool'],kind=plan['kind'],family=plan['family'],state='RESERVED',reserved_cpu_seconds=plan['cpu_cap_seconds']))
    try:
        base.execute(packet)
    except BaseException:
        event(campaign,packet.name+'-failed','packet_failure',dict(packet=packet.name,status_sha256=digest(packet/'STATUS.json')))
        raise
    event(campaign,packet.name+'-complete','batch_complete',dict(packet=packet.name,complete_sha256=digest(packet/'COMPLETE.json')))
    export(campaign)
    from ghostscale.validation.soundingline.v26.friday_reporting import land
    land(campaign,packet,Path(read(campaign/'LOCAL_PATHS.json')['repo']))


def build(campaign):
    runtime.below_normal();started=time.process_time()
    if (campaign/'BUILD_COMPLETE.json').exists():raise ValueError('already built; new cards require separate prospective registration')
    source=base.frozen_source(campaign,EXTRAS)
    repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);out=repo/'docs/versions/v26-top-and-ends'/REL
    (out/'protocols').mkdir(exist_ok=True);(campaign/'controls').mkdir(exist_ok=True)
    cards=friday.cards()
    # Interleave families, so a long map cannot starve every other question.
    order=['H1','E1','H2','H3','E2','E3','U1','R1']
    cards.sort(key=lambda c:(int(c['id'].split('X')[1]),order.index(c['family'])))
    write(campaign/'CARDS.json',cards)
    dispositions=[]
    try:
        for i,card in enumerate(cards):
            card=card|dict(queue_order=i);f=card['family'];cid=card['id']
            if not (campaign/'controls'/(f+'.json')).exists():write(campaign/'controls'/(f+'.json'),study.module(f).controls())
            protocol=card|dict(interpreter=str(Path(sys.executable).resolve()),module_sha256=digest(source/'ghostscale/validation/soundingline/v26'/f'{registry.MODULES[f]}.py'),
                source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),supplement_sha256=digest(campaign/'SUPPLEMENT.json'))
            write(campaign/'protocols'/(cid+'.json'),protocol)
            check=friday.design_check(card);write(campaign/'designs'/(cid+'.json'),check)
            write(out/'protocols'/(cid+'.json'),{k:v for k,v in protocol.items() if k!='interpreter'})
            write(out/'protocols'/(cid+'-DESIGN.json'),check)
            if check['passed']:
                packet=prepare(campaign,card,source);dispositions.append(dict(card=cid,state='PROBE_PREPARED',packet=packet.name))
            else:
                dispositions.append(dict(card=cid,state='DESIGN_UNAVAILABLE',failed_settings=[r['setting'] for r in check['checks'] if not r['passed']]))
            write(campaign/'BUILD_PROGRESS.json',dict(at=base.now().isoformat(),source=str(source),dispositions=dispositions),immutable=False)
        write(campaign/'BUILD_COMPLETE.json',dict(at=base.now().isoformat(),source=str(source),dispositions=dispositions,registered_cards=len(cards)))
        write(out/'CARDS.json',cards)
    finally:
        write(campaign/'accounting'/'build-1.json',dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started))
    print(str(source),flush=True)


def register_refill(campaign):
    """Append the prospectively frozen post-depletion cards and prepare probes."""
    runtime.below_normal();started=time.process_time()
    if not (campaign/'BUILD_COMPLETE.json').exists():raise ValueError('initial build is incomplete')
    if (campaign/'QUEUE_OWNER.lock').exists():raise ValueError('queue owner is still live or requires recovery review')
    if read(campaign/'QUEUE_STATUS.json')['state']!='DRAINED':raise ValueError('refill requires a drained queue')
    registration=campaign/'REFILL_REGISTRATION.json'
    if registration.exists():raise ValueError('refill already registered')
    source=base.frozen_source(campaign,EXTRAS)
    repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);out=repo/'docs/versions/v26-top-and-ends'/REL
    existing=list(read(campaign/'CARDS.json'));ids={card['id'] for card in existing}
    cards=friday.refill_cards()
    if ids.intersection(card['id'] for card in cards):raise ValueError('refill card id already registered')
    order=max((card.get('queue_order',i) for i,card in enumerate(existing)),default=-1)+1
    dispositions=[];registered=[]
    try:
        for offset,card in enumerate(cards):
            card=card|dict(queue_order=order+offset);f=card['family'];cid=card['id']
            if not (campaign/'controls'/(f+'.json')).exists():raise ValueError('missing inherited controls: '+f)
            protocol=card|dict(
                interpreter=str(Path(sys.executable).resolve()),
                module_sha256=digest(source/'ghostscale/validation/soundingline/v26'/f'{registry.MODULES[f]}.py'),
                source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
                supplement_sha256=digest(campaign/'SUPPLEMENT.json'),
            )
            write(campaign/'protocols'/(cid+'.json'),protocol)
            check=friday.design_check(card);write(campaign/'designs'/(cid+'.json'),check)
            write(out/'protocols'/(cid+'.json'),{k:v for k,v in protocol.items() if k!='interpreter'})
            write(out/'protocols'/(cid+'-DESIGN.json'),check)
            if check['passed']:
                packet=prepare(campaign,card,source)
                dispositions.append(dict(card=cid,state='PROBE_PREPARED',packet=packet.name))
            else:
                dispositions.append(dict(card=cid,state='DESIGN_UNAVAILABLE',failed_settings=[r['setting'] for r in check['checks'] if not r['passed']]))
            registered.append(card)
        combined=existing+registered
        write(campaign/'CARDS.json',combined,immutable=False)
        write(out/'CARDS.json',combined,immutable=False)
        receipt=dict(
            at=base.now().isoformat(),source=str(source),
            source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
            source_archive_sha256=digest(source/'SOURCE.zip'),
            trigger='genuine queue depletion before Friday review',
            previous_registered_cards=len(existing),registered_cards=len(registered),
            dispositions=dispositions,
        )
        write(registration,receipt)
        write(repo/'results/v26'/REL/'REFILL_REGISTRATION.json',{k:v for k,v in receipt.items() if k!='source'})
        prepared=[row['packet'] for row in dispositions if row['state']=='PROBE_PREPARED']
        write(campaign/'QUEUE_STATUS.json',dict(
            state='WAITING',pid=None,heartbeat=base.now().isoformat(),active=[],
            prepared=prepared,gear=2,needs_refill=False,
        ),immutable=False)
        export(campaign)
    finally:
        write(campaign/'accounting'/'refill-registration-1.json',dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started),immutable=False)
    print(str(source),flush=True)


def register_continuation_refill(campaign):
    """Append two prospective R1 evidence-dose cards beside a live drained queue."""
    runtime.below_normal();started=time.process_time()
    registration=campaign/'REFILL_REGISTRATION_2.json'
    if registration.exists():raise ValueError('continuation refill already registered')
    if not (campaign/'REFILL_REGISTRATION.json').exists():raise ValueError('first refill is not registered')
    if not (campaign/'PAUSE_ADMISSIONS').exists():raise ValueError('admission pause required')
    queue_status=read(campaign/'QUEUE_STATUS.json')
    if queue_status.get('prepared'):raise ValueError('continuation refill requires a depleted prepared queue')
    if not queue_status.get('active'):raise ValueError('live-drain registration requires an active packet')
    if not (campaign/'QUEUE_OWNER.lock').exists() or not alive(queue_status.get('pid')):
        raise ValueError('live queue owner is absent or requires recovery review')
    source=base.frozen_source(campaign,EXTRAS)
    repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);out=repo/'docs/versions/v26-top-and-ends'/REL
    existing=list(read(campaign/'CARDS.json'));ids={card['id'] for card in existing}
    cards=friday.continuation_refill_cards()
    if ids.intersection(card['id'] for card in cards):raise ValueError('continuation card id already registered')
    order=max((card.get('queue_order',i) for i,card in enumerate(existing)),default=-1)+1
    dispositions=[];registered=[]
    try:
        for offset,card in enumerate(cards):
            card=card|dict(queue_order=order+offset);family=card['family'];card_id=card['id']
            if not (campaign/'controls'/(family+'.json')).exists():raise ValueError('missing inherited controls: '+family)
            protocol=card|dict(
                interpreter=str(Path(sys.executable).resolve()),
                module_sha256=digest(source/'ghostscale/validation/soundingline/v26'/f'{registry.MODULES[family]}.py'),
                source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
                supplement_sha256=digest(campaign/'SUPPLEMENT.json'),
            )
            write(campaign/'protocols'/(card_id+'.json'),protocol)
            check=friday.design_check(card);write(campaign/'designs'/(card_id+'.json'),check)
            write(out/'protocols'/(card_id+'.json'),{k:v for k,v in protocol.items() if k!='interpreter'})
            write(out/'protocols'/(card_id+'-DESIGN.json'),check)
            if check['passed']:
                packet=prepare(campaign,card,source)
                dispositions.append(dict(card=card_id,state='PROBE_PREPARED',packet=packet.name))
            else:
                dispositions.append(dict(card=card_id,state='DESIGN_UNAVAILABLE',failed_settings=[r['setting'] for r in check['checks'] if not r['passed']]))
            registered.append(card)
        combined=existing+registered
        write(campaign/'CARDS.json',combined,immutable=False)
        write(out/'CARDS.json',combined,immutable=False)
        receipt=dict(
            at=base.now().isoformat(),source=str(source),
            source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
            source_archive_sha256=digest(source/'SOURCE.zip'),
            trigger='depleted prepared queue with useful commissioned time remaining',
            previous_registered_cards=len(existing),registered_cards=len(registered),
            active_packets_inspected=list(queue_status['active']),dispositions=dispositions,
        )
        write(registration,receipt)
        write(repo/'results/v26'/REL/'REFILL_REGISTRATION_2.json',{k:v for k,v in receipt.items() if k!='source'})
        export(campaign)
    finally:
        write(campaign/'accounting'/'refill-registration-2.json',dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started),immutable=False)
    print(str(source),flush=True)


def register_final_refill(campaign):
    """Append the two prospective H1 endpoint maps after a genuine drain."""
    runtime.below_normal();started=time.process_time()
    registration=campaign/'REFILL_REGISTRATION_3.json'
    if registration.exists():raise ValueError('final refill already registered')
    if not (campaign/'REFILL_REGISTRATION_2.json').exists():raise ValueError('second refill is not registered')
    queue_status=read(campaign/'QUEUE_STATUS.json')
    if queue_status.get('state')!='DRAINED' or queue_status.get('active') or queue_status.get('prepared'):
        raise ValueError('final refill requires a genuinely drained queue')
    if not queue_status.get('needs_refill'):
        raise ValueError('drained queue does not request useful refill')
    if (campaign/'QUEUE_OWNER.lock').exists():
        raise ValueError('queue owner is still live or requires recovery review')
    source=base.frozen_source(campaign,EXTRAS)
    repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);out=repo/'docs/versions/v26-top-and-ends'/REL
    existing=list(read(campaign/'CARDS.json'));ids={card['id'] for card in existing}
    cards=friday.final_refill_cards()
    if ids.intersection(card['id'] for card in cards):raise ValueError('final refill card id already registered')
    order=max((card.get('queue_order',i) for i,card in enumerate(existing)),default=-1)+1
    dispositions=[];registered=[]
    try:
        for offset,card in enumerate(cards):
            card=card|dict(queue_order=order+offset);family=card['family'];card_id=card['id']
            if not (campaign/'controls'/(family+'.json')).exists():raise ValueError('missing inherited controls: '+family)
            protocol=card|dict(
                interpreter=str(Path(sys.executable).resolve()),
                module_sha256=digest(source/'ghostscale/validation/soundingline/v26'/f'{registry.MODULES[family]}.py'),
                source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
                supplement_sha256=digest(campaign/'SUPPLEMENT.json'),
            )
            write(campaign/'protocols'/(card_id+'.json'),protocol)
            check=friday.design_check(card);write(campaign/'designs'/(card_id+'.json'),check)
            write(out/'protocols'/(card_id+'.json'),{k:v for k,v in protocol.items() if k!='interpreter'})
            write(out/'protocols'/(card_id+'-DESIGN.json'),check)
            if check['passed']:
                packet=prepare(campaign,card,source)
                dispositions.append(dict(card=card_id,state='PROBE_PREPARED',packet=packet.name))
            else:
                dispositions.append(dict(card=card_id,state='DESIGN_UNAVAILABLE',failed_settings=[r['setting'] for r in check['checks'] if not r['passed']]))
            registered.append(card)
        combined=existing+registered
        write(campaign/'CARDS.json',combined,immutable=False)
        write(out/'CARDS.json',combined,immutable=False)
        receipt=dict(
            at=base.now().isoformat(),source=str(source),
            source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
            source_archive_sha256=digest(source/'SOURCE.zip'),
            trigger='genuine queue depletion after the second prospective refill',
            previous_registered_cards=len(existing),registered_cards=len(registered),
            drained_status_sha256=digest(campaign/'QUEUE_STATUS.json'),
            dispositions=dispositions,
        )
        write(registration,receipt)
        write(repo/'results/v26'/REL/'REFILL_REGISTRATION_3.json',{k:v for k,v in receipt.items() if k!='source'})
        prepared=[row['packet'] for row in dispositions if row['state']=='PROBE_PREPARED']
        write(campaign/'QUEUE_STATUS.json',dict(
            state='WAITING',pid=None,heartbeat=base.now().isoformat(),active=[],
            prepared=prepared,gear=2,needs_refill=False,
        ),immutable=False)
        export(campaign)
    finally:
        write(campaign/'accounting'/'refill-registration-3.json',dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started),immutable=False)
    print(str(source),flush=True)


def register_rare_event_refill(campaign):
    """Append two prospective rare-event maps after the endpoint refill drains."""
    runtime.below_normal();started=time.process_time()
    registration=campaign/'REFILL_REGISTRATION_4.json'
    if registration.exists():raise ValueError('rare-event refill already registered')
    if not (campaign/'REFILL_REGISTRATION_3.json').exists():raise ValueError('endpoint refill is not registered')
    queue_status=read(campaign/'QUEUE_STATUS.json')
    if queue_status.get('state')!='DRAINED' or queue_status.get('active') or queue_status.get('prepared'):
        raise ValueError('rare-event refill requires a genuinely drained queue')
    if not queue_status.get('needs_refill'):
        raise ValueError('drained queue does not request useful refill')
    if (campaign/'QUEUE_OWNER.lock').exists():
        raise ValueError('queue owner is still live or requires recovery review')
    source=base.frozen_source(campaign,EXTRAS)
    repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);out=repo/'docs/versions/v26-top-and-ends'/REL
    existing=list(read(campaign/'CARDS.json'));ids={card['id'] for card in existing}
    cards=friday.rare_event_refill_cards()
    if ids.intersection(card['id'] for card in cards):raise ValueError('rare-event card id already registered')
    order=max((card.get('queue_order',i) for i,card in enumerate(existing)),default=-1)+1
    dispositions=[];registered=[]
    try:
        for offset,card in enumerate(cards):
            card=card|dict(queue_order=order+offset);family=card['family'];card_id=card['id']
            if not (campaign/'controls'/(family+'.json')).exists():raise ValueError('missing inherited controls: '+family)
            protocol=card|dict(
                interpreter=str(Path(sys.executable).resolve()),
                module_sha256=digest(source/'ghostscale/validation/soundingline/v26'/f'{registry.MODULES[family]}.py'),
                source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
                supplement_sha256=digest(campaign/'SUPPLEMENT.json'),
            )
            write(campaign/'protocols'/(card_id+'.json'),protocol)
            check=friday.design_check(card);write(campaign/'designs'/(card_id+'.json'),check)
            write(out/'protocols'/(card_id+'.json'),{k:v for k,v in protocol.items() if k!='interpreter'})
            write(out/'protocols'/(card_id+'-DESIGN.json'),check)
            if check['passed']:
                packet=prepare(campaign,card,source)
                dispositions.append(dict(card=card_id,state='PROBE_PREPARED',packet=packet.name))
            else:
                dispositions.append(dict(card=card_id,state='DESIGN_UNAVAILABLE',failed_settings=[r['setting'] for r in check['checks'] if not r['passed']]))
            registered.append(card)
        combined=existing+registered
        write(campaign/'CARDS.json',combined,immutable=False)
        write(out/'CARDS.json',combined,immutable=False)
        receipt=dict(
            at=base.now().isoformat(),source=str(source),
            source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
            source_archive_sha256=digest(source/'SOURCE.zip'),
            trigger='genuine queue depletion after the hierarchy endpoint refill',
            previous_registered_cards=len(existing),registered_cards=len(registered),
            drained_status_sha256=digest(campaign/'QUEUE_STATUS.json'),
            dispositions=dispositions,
        )
        write(registration,receipt)
        write(repo/'results/v26'/REL/'REFILL_REGISTRATION_4.json',{k:v for k,v in receipt.items() if k!='source'})
        prepared=[row['packet'] for row in dispositions if row['state']=='PROBE_PREPARED']
        write(campaign/'QUEUE_STATUS.json',dict(
            state='WAITING',pid=None,heartbeat=base.now().isoformat(),active=[],
            prepared=prepared,gear=2,needs_refill=False,
        ),immutable=False)
        export(campaign)
    finally:
        write(campaign/'accounting'/'refill-registration-4.json',dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started),immutable=False)
    print(str(source),flush=True)


def register_lower_rate_refill(campaign):
    """Append two prospective lower-rate maps after the 0.05 queue drains."""
    runtime.below_normal();started=time.process_time()
    registration=campaign/'REFILL_REGISTRATION_5.json'
    if registration.exists():raise ValueError('lower-rate refill already registered')
    if not (campaign/'REFILL_REGISTRATION_4.json').exists():raise ValueError('rare-event refill is not registered')
    queue_status=read(campaign/'QUEUE_STATUS.json')
    if queue_status.get('state')!='DRAINED' or queue_status.get('active') or queue_status.get('prepared'):
        raise ValueError('lower-rate refill requires a genuinely drained queue')
    if not queue_status.get('needs_refill'):
        raise ValueError('drained queue does not request useful refill')
    if (campaign/'QUEUE_OWNER.lock').exists():
        raise ValueError('queue owner is still live or requires recovery review')
    source=base.frozen_source(campaign,EXTRAS)
    repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);out=repo/'docs/versions/v26-top-and-ends'/REL
    existing=list(read(campaign/'CARDS.json'));ids={card['id'] for card in existing}
    cards=friday.lower_rate_refill_cards()
    if ids.intersection(card['id'] for card in cards):raise ValueError('lower-rate card id already registered')
    order=max((card.get('queue_order',i) for i,card in enumerate(existing)),default=-1)+1
    dispositions=[];registered=[]
    try:
        for offset,card in enumerate(cards):
            card=card|dict(queue_order=order+offset);family=card['family'];card_id=card['id']
            if not (campaign/'controls'/(family+'.json')).exists():raise ValueError('missing inherited controls: '+family)
            protocol=card|dict(
                interpreter=str(Path(sys.executable).resolve()),
                module_sha256=digest(source/'ghostscale/validation/soundingline/v26'/f'{registry.MODULES[family]}.py'),
                source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
                supplement_sha256=digest(campaign/'SUPPLEMENT.json'),
            )
            write(campaign/'protocols'/(card_id+'.json'),protocol)
            check=friday.design_check(card);write(campaign/'designs'/(card_id+'.json'),check)
            write(out/'protocols'/(card_id+'.json'),{k:v for k,v in protocol.items() if k!='interpreter'})
            write(out/'protocols'/(card_id+'-DESIGN.json'),check)
            if check['passed']:
                packet=prepare(campaign,card,source)
                dispositions.append(dict(card=card_id,state='PROBE_PREPARED',packet=packet.name))
            else:
                dispositions.append(dict(card=card_id,state='DESIGN_UNAVAILABLE',failed_settings=[r['setting'] for r in check['checks'] if not r['passed']]))
            registered.append(card)
        combined=existing+registered
        write(campaign/'CARDS.json',combined,immutable=False)
        write(out/'CARDS.json',combined,immutable=False)
        receipt=dict(
            at=base.now().isoformat(),source=str(source),
            source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
            source_archive_sha256=digest(source/'SOURCE.zip'),
            trigger='genuine queue depletion after the 0.05 rare-event maps',
            previous_registered_cards=len(existing),registered_cards=len(registered),
            drained_status_sha256=digest(campaign/'QUEUE_STATUS.json'),
            dispositions=dispositions,
        )
        write(registration,receipt)
        write(repo/'results/v26'/REL/'REFILL_REGISTRATION_5.json',{k:v for k,v in receipt.items() if k!='source'})
        prepared=[row['packet'] for row in dispositions if row['state']=='PROBE_PREPARED']
        write(campaign/'QUEUE_STATUS.json',dict(
            state='WAITING',pid=None,heartbeat=base.now().isoformat(),active=[],
            prepared=prepared,gear=2,needs_refill=False,
        ),immutable=False)
        export(campaign)
    finally:
        write(campaign/'accounting'/'refill-registration-5.json',dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started),immutable=False)
    print(str(source),flush=True)


def verify_bindings(campaign,packet,plan):
    base.verify_source(plan)
    for key,name in [('acceptance_sha256','ACCEPTANCE.json'),('environment_sha256','ENVIRONMENT.json'),('supplement_sha256','SUPPLEMENT.json'),
                     ('parent_charges_sha256','PARENT_CHARGES.json'),('protocol_sha256','protocols/'+plan['card']+'.json'),('design_check_sha256','designs/'+plan['card']+'.json')]:
        if digest(campaign/name)!=plan[key]:raise ValueError('binding changed: '+name)
    if read(campaign/'SUPPLEMENT.json')['original_acceptance_sha256']!=digest(campaign/'ACCEPTANCE.json'):raise ValueError('original acceptance binding')
    for path,sha in read(campaign/'PARENT_CHARGES.json')['closeout_hashes'].items():
        if digest(Path(path)/'CLOSEOUT.json')!=sha:raise ValueError('closed parent changed')
    check=read(campaign/'designs'/(plan['card']+'.json'))
    if not check['passed'] or digest(packet/'CONTROLS_BEFORE_DATA.json')!=plan['controls_sha256'] or not read(packet/'CONTROLS_BEFORE_DATA.json')['passed']:raise ValueError('controls/design failed')
    protocol=read(campaign/'protocols'/(plan['card']+'.json'))
    if plan['settings']!=protocol['settings'] or plan['blocks']!=(1 if plan['kind']=='probe' else protocol['blocks']):raise ValueError('undeclared setting or sample')
    if digest(Path(plan['source_root'])/'ghostscale/validation/soundingline/v26'/f'{registry.MODULES[plan["family"]]}.py')!=protocol['module_sha256']:raise ValueError('module changed')
    if plan['cost_probe']:
        probe=campaign/'packets'/plan['cost_probe'];verify_complete(probe)
        if digest(probe/'COMPLETE.json')!=plan['cost_probe_complete_sha256']:raise ValueError('cost probe changed')


def outside(campaign, owned=()):
    total=base.outside_scientists(campaign)
    owned=set(owned)
    for root in (campaign,campaign.parent/'v26',campaign.parent/'v26-followups-20261103'):
        for path in (root/'packets').glob('*/STATUS.json'):
            if root==campaign and path.parent.name in owned:continue
            s=read(path);pid=s.get('parent_pid') if s.get('state')=='RUNNING' else s.get('launcher_pid') if s.get('state')=='LAUNCHING' else None
            if pid and alive(pid):total+=1
    return total


def export(campaign):
    repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);out=repo/'results/v26'/REL;rows=[]
    for path in sorted((campaign/'packets').glob('*/PLAN.json')):
        p=path.parent;plan=read(path);s=read(p/'STATUS.json')
        rows.append(dict(packet=p.name,card=plan['card'],family=plan['family'],kind=plan['scientific_kind'],state=s['state'],phase=s.get('phase'),cap_seconds=plan['cpu_cap_seconds'],projected_wall_seconds=plan['projected_wall_seconds']))
        if (p/'COMPLETE.json').exists() and not (p/'AGGREGATE_EXPORTED.json').exists():
            verify_complete(p);dest=out/p.name;dest.mkdir(parents=True,exist_ok=True)
            for name,relative in [('SUMMARY.json','original/SUMMARY.json'),('RAW_INDEX.json','original/RAW_INDEX.json'),('COMPLETE.json','COMPLETE.json'),('REPLAY.json','REPLAY.json'),('INDEPENDENT_REVIEW.json','INDEPENDENT_REVIEW.json'),('PASS_COSTS.json','PASS_COSTS.json'),('CONTROLS.json','CONTROLS_BEFORE_DATA.json')]:write(dest/name,read(p/relative))
            write(dest/'PUBLIC_PLAN.json',public_plan(plan))
            write(p/'AGGREGATE_EXPORTED.json',dict(at=base.now().isoformat(),destination=dest.relative_to(repo).as_posix()))
    q=read(campaign/'QUEUE_STATUS.json');supplement=read(campaign/'SUPPLEMENT.json')
    write(out/'CURRENT_STATUS.json',dict(at=base.now().isoformat(),state=q['state'],active=q.get('active',[]),prepared=q.get('prepared',[]),packets=rows,
        gear=2,workers_maximum=2,pools_charged_seconds=charges(campaign),pool_ceilings_seconds=supplement['pools_seconds'],
        registered_cards=len(read(campaign/'CARDS.json')) if (campaign/'CARDS.json').exists() else 0,
        fully_built=(campaign/'BUILD_COMPLETE.json').exists(),cost_admission='Each study requires its full-setting excluded probe, measured cost and remaining-cap check.',
        review_target=supplement['review_target'],prior_queues='closed and preserved',needs_refill=q.get('needs_refill',False)),immutable=False)


def public_plan(plan):
    """Remove machine-local execution paths from a repository export."""
    private = {'source_root', 'interpreter', 'accounting_root'}
    return {key: value for key, value in plan.items() if key not in private}


def queue(campaign):
    if os.name!='nt':raise RuntimeError('native Windows required')
    runtime.below_normal();started=time.process_time();active={};lock=campaign/'QUEUE_OWNER.lock'
    account=campaign/'accounting'/f'controller-{os.getpid()}-{time.time_ns()}.json'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        while True:
            for name,(proc,log) in list(active.items()):
                if proc.poll() is None:continue
                log.close();del active[name];packet=campaign/'packets'/name;s=read(packet/'STATUS.json');plan=read(packet/'PLAN.json')
                if s['state']=='COMPLETE':
                    event(campaign,name+'-complete','batch_complete',dict(packet=name,complete_sha256=digest(packet/'COMPLETE.json')))
                    if plan['kind']=='probe':
                        try:prepare(campaign,read(campaign/'protocols'/(plan['card']+'.json')),Path(plan['source_root']),packet)
                        except Exception as exc:
                            write(packet/'CONTINUATION_FAILURE.json',dict(error=f'{type(exc).__name__}: {exc}'))
                            event(campaign,name+'-continuation-failed','admission_failure',dict(packet=name,error=str(exc)))
                else:
                    if s['state'] not in ('FAILED','ADMISSION_HELD'):write(packet/'STATUS.json',dict(state='FAILED',reason='parent exited; reservation retained',exit_code=proc.returncode),immutable=False)
                    event(campaign,name+'-failed','packet_failure',dict(packet=name,status_sha256=digest(packet/'STATUS.json')))
            candidates=[p.parent for p in (campaign/'packets').glob('*/PLAN.json') if read(p.parent/'STATUS.json')['state']=='PREPARED']
            candidates.sort(key=lambda p:(read(p/'PLAN.json')['queue_order'],read(p/'PLAN.json')['kind']=='probe'))
            supplement=read(campaign/'SUPPLEMENT.json');stopping=(campaign/'STOP').exists() or base.now()>=datetime.fromisoformat(supplement['review_target'])
            if not stopping and not (campaign/'PAUSE_ADMISSIONS').exists():
                for packet in candidates:
                    external=outside(campaign,active)
                    if len(active)+external>=2:break
                    plan=read(packet/'PLAN.json')
                    try:
                        verify_bindings(campaign,packet,plan)
                        friday.admission(plan,read(campaign/'ACCEPTANCE.json'),supplement,base.now(),charges(campaign),len(active)+external)
                    except ValueError as exc:
                        # Existing reservations may release on completion. Defer a
                        # resource-only hold until active packets finish before final disposition.
                        if active and str(exc) in ('family ceiling','pool ceiling'):continue
                        write(packet/'STATUS.json',dict(state='ADMISSION_HELD',reason=str(exc),at=base.now().isoformat()),immutable=False)
                        event(campaign,packet.name+'-held','admission_failure',dict(packet=packet.name,reason=str(exc)));continue
                    write(packet/'ADMISSION.json',dict(at=base.now().isoformat(),plan_sha256=digest(packet/'PLAN.json'),charges_before=charges(campaign)))
                    write(campaign/'accounting'/(packet.name+'.json'),dict(pool='studies',kind=plan['kind'],family=plan['family'],state='RESERVED',reserved_cpu_seconds=plan['cpu_cap_seconds']))
                    log=(packet/'parent.log').open('xb')
                    proc=subprocess.Popen([plan['interpreter'],'-B','-m','runners.run_v26','--packet',str(packet)],cwd=plan['source_root'],env=base.env_for(plan['source_root']),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                    active[packet.name]=(proc,log);write(packet/'STATUS.json',dict(state='LAUNCHING',launcher_pid=proc.pid,at=base.now().isoformat()),immutable=False)
            remaining=[p for p in candidates if read(p/'STATUS.json')['state']=='PREPARED']
            drained=not active and (stopping or ((campaign/'BUILD_COMPLETE.json').exists() and not remaining))
            write(campaign/'QUEUE_STATUS.json',dict(state='DRAINED' if drained else 'RUNNING' if active else 'WAITING',pid=os.getpid(),heartbeat=base.now().isoformat(),active=list(active),prepared=[p.name for p in remaining],gear=2,needs_refill=drained and not stopping),immutable=False)
            write(account,dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
            export(campaign)
            from ghostscale.validation.soundingline.v26.friday_reporting import land
            repo=Path(read(campaign/'LOCAL_PATHS.json')['repo'])
            for complete in (campaign/'packets').glob('*/COMPLETE.json'):
                land(campaign,complete.parent,repo)
            if drained:
                event(campaign,'friday-drained-'+str(os.getpid()),'review_boundary',dict(needs_useful_refill=not stopping,friday_review=stopping));break
            time.sleep(10)
    finally:
        write(account,dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='STOPPED'),immutable=False)
        lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--build',action='store_true');parser.add_argument('--register-refill',action='store_true');parser.add_argument('--register-continuation-refill',action='store_true');parser.add_argument('--register-final-refill',action='store_true');parser.add_argument('--register-rare-event-refill',action='store_true');parser.add_argument('--register-lower-rate-refill',action='store_true');parser.add_argument('--queue',action='store_true');parser.add_argument('--prepare-verification-healing',type=Path);parser.add_argument('--run-verification-healing',type=Path);parser.add_argument('--healing-number',type=int,default=1)
    args=parser.parse_args();campaign=args.campaign.resolve()
    if args.build:build(campaign)
    elif args.register_refill:register_refill(campaign)
    elif args.register_continuation_refill:register_continuation_refill(campaign)
    elif args.register_final_refill:register_final_refill(campaign)
    elif args.register_rare_event_refill:register_rare_event_refill(campaign)
    elif args.register_lower_rate_refill:register_lower_rate_refill(campaign)
    elif args.queue:queue(campaign)
    elif args.prepare_verification_healing:
        source=base.frozen_source(campaign,EXTRAS);print(prepare_verification_healing(campaign,args.prepare_verification_healing,source,args.healing_number),flush=True)
    elif args.run_verification_healing:run_verification_healing(campaign,args.run_verification_healing.resolve())
    else:parser.error('choose build, register-refill, register-continuation-refill, register-final-refill, register-rare-event-refill, register-lower-rate-refill or queue')
