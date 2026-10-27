"""Freeze W5's optional WB1 calibration only after an eligible aggregate arrives."""
import argparse,shutil,sys,time
from pathlib import Path
from datetime import datetime,timezone
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v25 import wb1,misspec,contract
from runners.run_v25 import frozen_source,verify_source,charges


def prepare_probe(campaign,incoming,evidence):
    started=time.process_time();campaign=Path(campaign).resolve();incoming=Path(incoming);evidence=Path(evidence)
    payload=read(incoming);observed=datetime.now(timezone.utc)
    library=wb1.calibrated_library(payload,observed)
    if digest(evidence)!=payload['source']['result_sha256']:raise ValueError('WB1 source result hash mismatch')
    accepted=read(campaign/'ACCEPTANCE.json');controls=misspec.controls()
    if not controls['passed']:raise ValueError('known-answer failure')
    source=frozen_source(campaign);identity='v25-w5-calibrated-probe-1';packet=campaign/'packets'/identity
    if packet.exists():raise ValueError('calibration probe already prepared; preserve existing packet')
    settings=[dict(beta=b,calibrated_library=library) for b in (1,3)]
    protocol=dict(family='W5',variant='wb1-calibrated',source_result_sha256=digest(evidence),
        incoming_sha256=digest(incoming),observed_at=observed.isoformat(),controls=controls,
        provisional=True,library=library,settings=settings,blocks=16,
        hypothesis='Descriptive true-ends log-score cost of distorted libraries at128 records, in WB1-calibrated constructed worlds.',
        choices='Empirical classes calibrate the world library; compare true/merged/missing/split readers within it. Uniform15 weights x2 rationalities x(K+1) habits; eight makers per world. Project W1 training menus to K, retain unique nonempty menus. With K=1 merge is unavailable, not an extra class. No source effects are treated as human end weights.',
        receipt_review='Operator must verify complete upstream scientific and control receipts before invoking; this tool additionally binds the supplied source receipt by hash.',
        verification='Known-answer controls, cheapest/costliest complete probe, measured per-pass reserves, full byte replay and independent scalar reconstruction/Decimal aggregation.')
    protocol_path=campaign/'protocols/W5_CALIBRATED.json'
    if protocol_path.exists():raise ValueError('calibration already frozen')
    plan=dict(family='W5',kind='probe',pool='studies',identity=identity,science_identity=identity,
        settings=settings,blocks=1,shed_level=0,calibration_variant='wb1-calibrated',
        source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),source_archive_sha256=digest(source/'SOURCE.zip'),
        acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),environment_sha256=digest(campaign/'ENVIRONMENT.json'),
        protocol_sha256=None,controls_sha256=None,numerical_threads=1,priority='below-normal',
        cpu_cap_seconds=1800.,pass_caps=dict(original=600.,replay=600.,review=600.),wall_cap_seconds=18*3600,
        interpreter=str(Path(sys.executable).resolve()),parent=None,production=False,confirmation=False,
        calibration_input_sha256=digest(incoming),calibration_source_result_sha256=digest(evidence))
    contract.admission(plan,accepted,observed,charges(campaign),running=0)
    write(protocol_path,protocol);plan['protocol_sha256']=digest(protocol_path)
    packet.mkdir(parents=True);write(packet/'CONTROLS_BEFORE_DATA.json',controls);plan['controls_sha256']=digest(packet/'CONTROLS_BEFORE_DATA.json')
    shutil.copyfile(incoming,packet/'CALIBRATION_INPUT.json');shutil.copyfile(evidence,packet/'SOURCE_RESULT.json')
    write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='PREPARED',at=observed.isoformat()))
    write(campaign/'accounting'/(identity+'-preparation.json'),dict(pool='preparation',kind='preparation',family='W5',cpu_seconds=time.process_time()-started,state='COMPLETE'))
    return packet


def prepare_study(campaign,parent,source=None):
    started=time.process_time();campaign=Path(campaign);parent=Path(parent);prior=read(parent/'PLAN.json')
    if prior.get('calibration_variant')!='wb1-calibrated' or prior['kind']!='probe':raise ValueError('calibrated probe required')
    complete=read(parent/'COMPLETE.json')
    for key,name in [('plan_sha256','PLAN.json'),('summary_sha256','original/SUMMARY.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
        if complete[key]!=digest(parent/name):raise ValueError('verified probe binding changed')
    verify_source(prior)
    if source is not None and Path(source).resolve()!=Path(prior['source_root']).resolve():raise ValueError('calibration source mismatch')
    costs=read(campaign/'accounting'/(parent.name+'.json'))['pass_cpu_seconds'];timing=read(parent/'PASS_COSTS.json')
    if not timing.get('includes_world_io'):raise ValueError('complete world I/O cost required')
    units={k:max(max(timing['setting_seconds'][k]),1/64) for k in costs}
    overhead={k:max(0,costs[k]-sum(timing['setting_seconds'][k])) for k in costs}
    settings=prior['settings'];blocks=16;shed=0
    projection=sum(overhead.values())+sum(units.values())*len(settings)*blocks
    if projection>contract.FAMILY_HOURS['W5']*3600:
        settings=[dict(s,omit_split=True) for s in settings];shed=1
        # Retain the measured full-reader cost as a conservative bound after shedding.
        if projection>contract.FAMILY_HOURS['W5']*3600:blocks=8;shed=2
    caps={k:1.5*(overhead[k]+units[k]*len(settings)*blocks)+10 for k in costs}
    identity='v25-w5-calibrated-study-1';packet=campaign/'packets'/identity
    plan=dict(prior,identity=identity,science_identity=identity,kind='study',settings=settings,blocks=blocks,
        shed_level=shed,cpu_cap_seconds=sum(caps.values()),pass_caps=caps,parent=parent.name,parent_complete_sha256=digest(parent/'COMPLETE.json'))
    contract.admission(plan,read(campaign/'ACCEPTANCE.json'),datetime.now(timezone.utc),charges(campaign),running=0)
    packet.mkdir();shutil.copyfile(parent/'CONTROLS_BEFORE_DATA.json',packet/'CONTROLS_BEFORE_DATA.json')
    write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='PREPARED',at=datetime.now(timezone.utc).isoformat()))
    write(campaign/'accounting'/(identity+'-preparation.json'),dict(pool='preparation',kind='preparation',family='W5',cpu_seconds=time.process_time()-started,state='COMPLETE'))
    return packet


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('campaign','input','evidence'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();print(prepare_probe(args.campaign,args.input,args.evidence))
