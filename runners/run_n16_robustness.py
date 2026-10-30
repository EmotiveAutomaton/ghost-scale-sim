"""Finite two-slot N16 extension using the established isolated V25 executor."""
import argparse
from datetime import datetime, timezone, timedelta
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ghostscale.validation.soundingline.v23.common import read, write, digest
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v24 import production
from ghostscale.validation.soundingline.v25 import n16_robustness as science
from runners.run_v25 import frozen_source, verify_source
from runners.watch_v23_events import alive, event

def now(): return datetime.now(timezone.utc)

def used(root):
    return math.fsum(max(x.get('cpu_seconds',0),x.get('reserved_cpu_seconds',0)) for x in (read(p) for p in (root/'accounting').glob('*.json')))

def environment(source):
    result=dict(os.environ,PYTHONPATH=str(source),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): result[key]='1'
    return result

def prepare(root, prior, repair_from=None):
    started=time.process_time(); root.mkdir(parents=True,exist_ok=False)
    terminal=read(prior/'TERMINAL_AFTER_REPAIR.json')
    if terminal['state']!='CONFIRMED_CONSTRUCTED_SCHEDULER': raise ValueError('prior confirmation not terminal')
    old=read(prior/'ACCEPTANCE.json'); carry=production.charged(prior)
    repair_charge=0.
    if repair_from is not None:
        previous=read(repair_from/'ACCEPTANCE.json')
        if previous['identity']!=science.IDENTITY or previous.get('repair_from') or (repair_from/'BUILD_COMPLETE.json').exists():raise ValueError('only first excluded engineering repair is admitted here')
        if not (repair_from/'TERMINAL.json').exists() or (repair_from/'QUEUE_OWNER.lock').exists():raise ValueError('prior failed attempt still owned')
        repair_charge=used(repair_from)
        if read(repair_from/'TERMINAL.json')['state']!='STOPPED_REQUIRES_REVIEW':raise ValueError('repair requires retained failed attempt')
        carry+=repair_charge
    if old['cpu_seconds']-carry<86400: raise ValueError('insufficient inherited ceiling')
    (root/'accounting').mkdir(); (root/'packets').mkdir(); (root/'notifications').mkdir()
    write(root/'PRIOR_TERMINAL.json',terminal)
    acceptance=dict(identity=science.IDENTITY,opened=now().isoformat(),drain=old['drain'],new_starts_close=old['drain'],blocks=64,cases_per_block=3840,scientific_workers=2,numerical_threads=1,priority='below-normal',extension_cpu_seconds=86400,protected_cpu_seconds=7200,total_ceiling_seconds=old['cpu_seconds'],prior_charged_cpu_seconds=carry,prior_terminal_sha256=digest(root/'PRIOR_TERMINAL.json'),confirmation=False,authorization='Owner October30 explicitly approved separate fresh-seed robustness extension in Gear 2 within existing budget and deadline')
    if repair_from is not None:
        acceptance.update(extension_cpu_seconds=86400-repair_charge,opened=read(repair_from/'ACCEPTANCE.json')['opened'],repair_from=repair_from.name,prior_extension_charge=repair_charge)
        write(root/'REPAIR.json',dict(parent=repair_from.name,prior_acceptance_sha256=digest(repair_from/'ACCEPTANCE.json'),prior_terminal_sha256=digest(repair_from/'TERMINAL.json'),failed_status_sha256=digest(repair_from/'packets/n16r-excluded-000/STATUS.json'),charged_cpu_seconds=repair_charge,reason='Retained probability 1.0000000000000002 failed an exact <=1 guard. Validate normalization within 1e-12 without clipping; preserve the seed identity, grid and observations.',scientific_setting_change=False,prior_raw_files={p.name:digest(p) for p in (repair_from/'packets/n16r-excluded-000/original').glob('*_points.jsonl.gz')}))
    write(root/'ACCEPTANCE.json',acceptance)
    protocol=dict(identity=science.IDENTITY,blocks=64,seed_identity=science.IDENTITY,excluded_identity=science.EXCLUDED,worlds_per_block=16,variants=48,cases_per_cell=5,epsilon=.1,primary='Correct chooser P4 versus correct-model P1 after two probes; descriptive replication only',policies=list(science.POLICIES),probes=list(science.BUDGETS),chooser_conditions=list(science.CONDITIONS),sources=list(science.SOURCES),reference='Always correct-model fixed-order P1 at the same probe count; every scoring posterior retains full true-mechanism support',weighting='Within each case average omitted-other chooser scenarios; then equal cases, variants, laws and source families; uncertainty unit is whole balanced block',interval='Pointwise descriptive 95 percent t interval over the 64 independent whole blocks; no multiplicity-adjusted or confirmatory claim',milestones=[16,32,48,64],stopping='Fixed 64 complete verified blocks; only resource/failure/deadline stops; no outcome-dependent expansion or descendants',validation='Pre-data known-answer controls, excluded full-block original/replay/independent reconstruction, complete replay and independent reconstruction for every scientific block')
    write(root/'PROTOCOL.json',protocol)
    write(root/'ENVIRONMENT.json',runtime.environment())
    controls=science.controls(); write(root/'CONTROLS.json',controls)
    source=frozen_source(root)
    template=read(prior/'TEMPLATE_R3.json')
    keep={k:template[k] for k in ('dependency_directory','dependency_files','schema_directory','schema_pin','schema_digest') if k in template}
    keep.update(family='N16R',pool='extension',n16_robustness=True,production=False,confirmation=False,source_revision=3,settings=[{}],blocks=1,epsilon=.1,cases_per_cell=5,variant_ids=[v['id'] for v in science.n16.model.variants()],source_root=str(source),source_archive_sha256=digest(source/'SOURCE.zip'),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),accounting_root=str(root),acceptance_sha256=digest(root/'ACCEPTANCE.json'),environment_sha256=digest(root/'ENVIRONMENT.json'),protocol_sha256=digest(root/'PROTOCOL.json'),controls_sha256=digest(root/'CONTROLS.json'),numerical_threads=1,priority='below-normal',interpreter=str(Path(sys.executable).resolve()),pass_caps=dict(original=360.,replay=360.,review=1080.),cpu_cap_seconds=1810.,wall_cap_seconds=3600.,measured_wall_seconds=900.)
    write(root/'TEMPLATE.json',keep)
    packet=prepare_packet(root,keep,0,excluded=True)
    write(root/'accounting/build-initial.json',dict(kind='preparation',cpu_seconds=time.process_time()-started,state='COMPLETE'))
    return packet

def prepare_packet(root, template, number, excluded=False):
    identity='n16r-excluded-000' if excluded else f'n16r-block-{number:03d}'
    p=root/'packets'/identity; p.mkdir(exist_ok=False)
    plan=dict(template,identity=identity,science_identity=science.EXCLUDED if excluded else science.IDENTITY,kind='excluded' if excluded else 'robustness',block_index=number)
    write(p/'CONTROLS_BEFORE_DATA.json',read(root/'CONTROLS.json'))
    write(p/'PLAN.json',plan); science.validate(plan)
    write(p/'STATUS.json',dict(state='PREPARED',at=now().isoformat()))
    return p

def verify_packet(p):
    receipt=read(p/'COMPLETE.json')
    for key,name in [('plan_sha256','PLAN.json'),('summary_sha256','original/SUMMARY.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
        if digest(p/name)!=receipt[key]: raise ValueError('completed packet binding changed')
    if read(p/'REPLAY.json')['state']!='passed' or read(p/'INDEPENDENT_REVIEW.json')['state']!='INDEPENDENT_RECONSTRUCTION_PASSED':
        raise ValueError('incomplete review')
    return read(p/'original/SUMMARY.json')

def build_science(root):
    started=time.process_time(); p=root/'packets/n16r-excluded-000'
    verify_packet(p); caps=read(p/'PASS_COSTS.json')['pass_cpu_seconds']; a=read(root/'ACCEPTANCE.json')
    template=read(root/'TEMPLATE.json')
    template['pass_caps']={k:1.5*v+10 for k,v in caps.items()}
    template['cpu_cap_seconds']=sum(template['pass_caps'].values())+10
    template['measured_wall_seconds']=read(root/'accounting/n16r-excluded-000.json')['wall_seconds']
    template['excluded_proof_sha256']=digest(p/'COMPLETE.json')
    projection=used(root)+64*template['cpu_cap_seconds']+a['protected_cpu_seconds']+3600
    if projection>a['extension_cpu_seconds'] or a['prior_charged_cpu_seconds']+projection>a['total_ceiling_seconds']:
        raise ValueError('measured complete extension exceeds inherited budget')
    write(root/'EXCLUDED_VERIFIED.json',dict(complete_sha256=digest(p/'COMPLETE.json'),actual_pass_cpu_seconds=caps,per_block_reservation_seconds=template['cpu_cap_seconds'],projected_all_in_seconds=projection,forecast_science_cpu_seconds=64*sum(caps.values())))
    write(root/'SCIENCE_TEMPLATE.json',template)
    for number in range(64): prepare_packet(root,template,number)
    write(root/'BUILD_COMPLETE.json',dict(prepared=64,seed_identity=science.IDENTITY,excluded_sha256=digest(root/'EXCLUDED_VERIFIED.json')))
    write(root/'accounting/build-science.json',dict(kind='preparation',cpu_seconds=time.process_time()-started,state='COMPLETE'))

def outside_workers(main):
    total=0
    for p in (main/'packets').glob('*/STATUS.json'):
        status=read(p)
        pid=status.get('parent_pid') if status['state']=='RUNNING' else status.get('launcher_pid') if status['state']=='LAUNCHING' else None
        if pid and alive(pid): total+=1
    return total

def admission(root, plan, active, current):
    science.validate(plan); a=read(root/'ACCEPTANCE.json')
    if active>=2 or plan['priority']!='below-normal' or plan['numerical_threads']!=1:
        raise ValueError('Gear 2 limit')
    if current+timedelta(seconds=2*plan['measured_wall_seconds'])>=datetime.fromisoformat(a['drain']):
        raise ValueError('last safe block start')
    projected=used(root)+plan['cpu_cap_seconds']+a['protected_cpu_seconds']
    if projected>a['extension_cpu_seconds'] or a['prior_charged_cpu_seconds']+projected>a['total_ceiling_seconds']:
        raise ValueError('extension or shared ceiling')
    if plan['kind']!='excluded' and digest(root/'packets/n16r-excluded-000/COMPLETE.json')!=plan['excluded_proof_sha256']:
        raise ValueError('excluded proof changed')
    verify_source(plan)

def queue(root, main):
    runtime.below_normal(); started=time.process_time(); active={}; failure=False
    lock=root/'QUEUE_OWNER.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    account=root/'accounting'/f'controller-{os.getpid()}-{time.time_ns()}.json'
    try:
        while True:
            for name,(proc,log) in list(active.items()):
                if proc.poll() is None: continue
                log.close(); del active[name]; p=root/'packets'/name
                status=read(p/'STATUS.json')
                if status['state']!='COMPLETE':
                    if status['state']!='FAILED': write(p/'STATUS.json',dict(state='FAILED',reason='Parent exited without terminal evidence; reservation retained',exit_code=proc.returncode),immutable=False)
                    failure=True; event(root,name+'-failed','packet_failure',dict(packet=name,status_sha256=digest(p/'STATUS.json')))
                else: verify_packet(p)
            completed=[]
            for p in sorted((root/'packets').glob('n16r-block-*')):
                if not (p/'COMPLETE.json').exists(): break
                completed.append(p)
            for milestone in (16,32,48,64):
                path=root/'aggregates'/f'PREFIX_{milestone:03d}.json'
                if len(completed)>=milestone and not path.exists():
                    aggregate=science.aggregate_blocks([verify_packet(p) for p in completed[:milestone]])
                    aggregate['complete_bindings']={p.name:digest(p/'COMPLETE.json') for p in completed[:milestone]}
                    write(path,aggregate)
                    event(root,f'n16r-prefix-{milestone:03d}','robustness_milestone',dict(blocks=milestone,aggregate_sha256=digest(path),file=path.name))
            all_packets=sorted((root/'packets').glob('*/PLAN.json'))
            if any(read(p.parent/'STATUS.json')['state'] in ('FAILED','ADMISSION_HELD') for p in all_packets): failure=True
            candidates=[p.parent for p in all_packets if read(p.parent/'STATUS.json')['state']=='PREPARED']
            stop=(root/'STOP').exists() or now()>=datetime.fromisoformat(read(root/'ACCEPTANCE.json')['drain'])
            if not failure and not stop:
                other=outside_workers(main)
                for p in candidates:
                    if len(active)+other>=2: break
                    plan=read(p/'PLAN.json')
                    try: admission(root,plan,len(active)+other,now())
                    except ValueError as exc:
                        write(p/'STATUS.json',dict(state='ADMISSION_HELD',reason=str(exc)),immutable=False)
                        event(root,p.name+'-held','admission_failure',dict(packet=p.name,reason=str(exc))); failure=True; break
                    if (p/'ADMISSION.json').exists(): raise ValueError('Prior admission without terminal evidence; inspect ownership before resume')
                    if digest(p/'CONTROLS_BEFORE_DATA.json')!=plan['controls_sha256'] or not read(p/'CONTROLS_BEFORE_DATA.json')['passed']: raise ValueError('control binding')
                    write(p/'ADMISSION.json',dict(at=now().isoformat(),plan_sha256=digest(p/'PLAN.json'),prior_extension_charge=used(root)))
                    write(root/'accounting'/(p.name+'.json'),dict(kind=plan['kind'],state='RESERVED',reserved_cpu_seconds=plan['cpu_cap_seconds']))
                    log=(p/'parent.log').open('xb')
                    proc=subprocess.Popen([plan['interpreter'],'-B','-m','runners.run_v25','--packet',str(p)],cwd=plan['source_root'],env=environment(plan['source_root']),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                    active[p.name]=(proc,log)
                    write(p/'STATUS.json',dict(state='LAUNCHING',launcher_pid=proc.pid,at=now().isoformat()),immutable=False)
            write(root/'QUEUE_STATUS.json',dict(state='RUNNING' if active else 'WAITING',pid=os.getpid(),heartbeat=now().isoformat(),active=list(active),verified_prefix=len(completed),target=64,failure=failure,extension_charged_cpu_seconds=used(root)),immutable=False)
            write(account,dict(kind='controller',cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
            if not active and (failure or stop or not candidates): break
            time.sleep(5)
        verified=len(completed)
        state='COMPLETE_FIXED_SIZE_DESCRIPTIVE' if verified==64 else 'EXCLUDED_PROBE_VERIFIED' if not (root/'BUILD_COMPLETE.json').exists() and (root/'packets/n16r-excluded-000/COMPLETE.json').exists() and not failure else 'STOPPED_REQUIRES_REVIEW'
        if state!='EXCLUDED_PROBE_VERIFIED':
            write(root/'TERMINAL.json',dict(at=now().isoformat(),state=state,verified_prefix=verified,target=64,confirmation=False,extension_charged_cpu_seconds=used(root)))
            event(root,'n16r-terminal','review_boundary',dict(terminal_sha256=digest(root/'TERMINAL.json')))
    finally:
        write(root/'QUEUE_STATUS.json',dict(state='STOPPED_WITH_ACTIVE_CHILDREN' if active else 'DRAINED',pid=os.getpid(),active=list(active),at=now().isoformat()),immutable=False)
        write(account,dict(kind='controller',cpu_seconds=time.process_time()-started,state='STOPPED'),immutable=False)
        lock.unlink()

def report(root, repo):
    out=repo/'results/v25/n16-robustness-20261030'; out.mkdir(exist_ok=True,parents=True)
    for name in ('ACCEPTANCE.json','PROTOCOL.json','EXCLUDED_VERIFIED.json','BUILD_COMPLETE.json','TERMINAL.json','REPAIR.json'):
        if (root/name).exists(): write(out/name,read(root/name),immutable=False)
    for p in (root/'aggregates').glob('PREFIX_*.json'):
        write(out/p.name,read(p),immutable=False)
    counts={}
    for p in (root/'packets').glob('*/STATUS.json'):
        state=read(p)['state']; counts[state]=counts.get(state,0)+1
    a=read(root/'ACCEPTANCE.json'); q=read(root/'QUEUE_STATUS.json') if (root/'QUEUE_STATUS.json').exists() else {}
    status=dict(at=now().isoformat(),identity=science.IDENTITY,packet_states=counts,queue=q,extension_charged_cpu_seconds=used(root),prior_charged_cpu_seconds=a['prior_charged_cpu_seconds'],total_ceiling_seconds=a['total_ceiling_seconds'],prior_confirmation_unchanged=True,gear=2,workers_maximum=2,target_blocks=64,confirmation=False)
    write(out/'CURRENT_STATUS.json',status,immutable=False)
    return status

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path);p.add_argument('--repair-from',type=Path);p.add_argument('--main-campaign',type=Path);p.add_argument('--repo',type=Path)
    p.add_argument('--action',choices=('prepare','build-science','queue','report'),required=True);a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare': print(prepare(root,a.prior.resolve(),a.repair_from.resolve() if a.repair_from else None))
    elif a.action=='build-science': build_science(root)
    elif a.action=='queue': queue(root,a.main_campaign.resolve())
    else: print(report(root,a.repo.resolve()))

if __name__=='__main__': main()
