"""V27 prospective build, bounded dispatch and result-driven commissioned lanes.

DESIGN CHECK: source/protocol/control hashes precede every start; failed attempts
are retained. Only prepared card files may run. Independent pools and host-load
hysteresis are tested separately; source capsules isolate all running science.
"""
import argparse,ctypes,json,math,os,subprocess,sys,time,threading
from pathlib import Path
from datetime import datetime
from ghostscale.validation.soundingline.v27 import contract,registry,study,reporting,continuation
from ghostscale.validation.soundingline.v27.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime
from runners import run_v27 as native
from ghostscale.validation.soundingline.v27.common import verify_complete
from runners.watch_v23_events import event,alive

EXTRAS=('runners/queue_v27.py','runners/watch_v27.py','runners/watch_v26.py','runners/fixture_v23.py')


def charges(c):
    total={}
    for p in (c/'accounting').glob('*.json'):
        row=read(p);value=contract.accounted(row);pool=row.get('pool','preparation')
        total[pool]=total.get(pool,0)+value
        for key in ([row['kind']] if row.get('kind') in ('healing','texture','followup','precision','reread') else []):total[key]=total.get(key,0)+value
        if pool=='studies':
            key='family:'+row.get('family','none');total[key]=total.get(key,0)+value
    return total


def settings_cost(s,f):
    cost=s.get('records',s.get('structures',32))*s.get('makers',s.get('worlds_per_block',1)*s.get('sources',1))
    if f=='K2':cost*=s['targets']
    if f=='K3':cost*=21**s['needs']*(8 if s['dynamic'] in ('S','P-','G','P+') else 1)
    if f=='K5':cost*=len(s.get('strengths',[.5]))
    if f=='K6':cost*=s['goals']*s['rounds']
    return cost


def protocol_choices(f):
    return dict(
      K1='Finite personal-parameter grid over the declared ranges; exact binned first-passage likelihoods. Locked-choice interpretation recorded separately before admission.',
      K2='Cyclic balanced target encounters; independent Bernoulli contexts and observed payment. Exact temporal reader integrates RAGE. BP has separate conditional-RAGE and act factors, damping 0.5, tolerance 1e-12, 200 iterations. Circular messages have exponent two. Extended Kalman cue learner starts at zero with identity covariance and maps target weight using logit(0.3)+4w. Scores use final dislikes.',
      K3='All actions available; ordering and exact hidden-state recursion declared in the module and card-specific numerical preflight. Unknown stocks are integrated, never supplied to hidden readers.',
      K4='Uniform prior on 25 log-spaced stress scales, also the generating scale distribution. Designed and observational mixes retain their stated frequencies. S3 is the reduction of Bernoulli variance from a binary diagnostic of accuracy 0.6, 0.8 or 0.95. All-channel and arousal-only readers are both retained.',
      K5='RR weakens closed-source log odds by fraction x; RT discounts the same log odds with exponential fading and saved independent tag retrieval after record 48. W receives a noisy harmful-intent diagnostic before observations. IG pools a common latent agenda over independent source truths; outsider library has strength 0.5, insider library includes 1 with honesty prior 0.5+b. This library/prior contrast is not a human mechanism. Partial tag loss is compared with its no-switch counterfactual.',
      K6='H2 habit/proposal-redraw law with four goals. Hard pause classes use goal/habit prior averages; iteration replaces classes and recomputes each datum once. Ic uses the maximum-posterior goal/habit. The gain-share denominator is retained, and a denominator interval crossing zero makes the ratio verdict unresolved.',
      K7='The world shares one top goal across repeated variable-length structures; each structure conceals that goal until its final piece. Readers infer boundaries, including hidden state for M3. Listener and marker laws are declared before excluded horizon calibration.')[f]


def initialize(c):
    repo=Path(read(c/'LOCAL_PATHS.json')['repo'])
    if not (c/'CARDS.json').exists():
        cards=registry.cards();write(c/'CARDS.json',cards);write(repo/'docs/versions/v27-motives-and-set-points/CARDS.json',cards)
        for card in cards:write(c/'cards'/(card['id']+'.json'),card)
    if not (c/'ENVIRONMENT.json').exists():write(c/'ENVIRONMENT.json',runtime.environment())
    if not (c/'NUMERICAL_ENVIRONMENT.json').exists():write(c/'NUMERICAL_ENVIRONMENT.json',study.extra_environment())


def build_card(c,card,source):
    cid=card['id'];f=card['family'];repo=Path(read(c/'LOCAL_PATHS.json')['repo'])
    if (c/'protocols'/(cid+'.json')).exists():return
    if f=='K1':
        decision=read(c/'DECISIONS.json')['K1']
        if decision['choice_law']!='locked_choices':raise ValueError('K1 choice law requires a recorded decision')
        card=card|dict(settings=[s|dict(choice_law='locked_choices') for s in card['settings']],owner_decision=decision)
    if f=='K1' and 'type_mix' in card.get('changes',{}):
        card=card|dict(settings=[s|dict(type_mix=card['changes']['type_mix']) for s in card['settings']])
    impl=study.module(f,card.get('analysis_module'));controls=impl.controls()
    if f=='K7':
        calibration=read(c/'K7_CALIBRATION.json');chosen=calibration['chosen']
        card=card|dict(settings=[s|dict(structures=chosen['structures']) if s['structures']==32 else s for s in card['settings']],calibration_sha256=digest(c/'K7_CALIBRATION.json'))
        if card['lane']=='study':card['settings']=[s|dict(qc=chosen['qc']) for s in card['settings']]
    if not controls['passed']:raise ValueError('controls failed: '+cid)
    checks=[]
    for i,s in enumerate(card['settings']):
        check=impl.design_check(s,['excluded-case',cid,i])
        if f=='K4' and card['lane']=='study' and s['mix']=='observational':
            check=check|dict(registered_out_of_case_arm=True,passed=True,actual_case_passed=False)
        checks.append(check)
    passed=all(x['passed'] for x in checks);out_of_case=card['declared_out_of_case'] or (card['lane']!='study' and not passed)
    # Every rival is evaluated on excluded data before scientific admission.
    # One small full block per distinct setting is a precheck, never a result.
    rival=[]
    for i,s in enumerate(card['settings']):
        fixture=dict(s,makers=1,worlds_per_block=1)
        data=impl.precheck(card,i) if card.get('analysis_module')=='closure_reread' else impl.block(fixture,['excluded-rivals',cid,i])
        rival.append(dict(setting=i,measures=data['measures']))
    check=dict(passed=passed,checks=checks,excluded_rival_prechecks=rival,scientific_draws=0)
    protocol=card|dict(modelling_choices=card.get('analysis_description',protocol_choices(f)),provisional=True,out_of_case=out_of_case,
        source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
        module_sha256=digest(source/'ghostscale/validation/soundingline/v27'/((card.get('analysis_module') or registry.MODULES[f])+'.py')),
        precheck_under_every_rival=[x.get('expected_signs',{}) for x in checks])
    write(c/'controls'/(cid+'.json'),controls);write(c/'designs'/(cid+'.json'),check)
    public=repo/'docs/versions/v27-motives-and-set-points/protocols'
    write(public/(cid+'.json'),{k:v for k,v in protocol.items() if k not in ('source_root','retained_manifest')});write(public/(cid+'-DESIGN.json'),check)
    if not passed and not out_of_case:
        write(c/'dispositions'/(cid+'.json'),dict(state='DESIGN_CANNOT_PRODUCE_CASE',card=cid));write(c/'protocols'/(cid+'.json'),protocol);return
    write(c/'dispositions'/(cid+'.json'),dict(state='BUILT',card=cid))
    # The protocol is the publication marker seen by the live continuation.
    # All dependent files must already exist when it appears.
    write(c/'protocols'/(cid+'.json'),protocol)
    if card['lane']=='study':prepare(c,cid)


def build(c,family,primary_only=False):
    runtime.below_normal();start=time.process_time();initialize(c)
    source=native.frozen_source(c,EXTRAS)
    try:
        if family=='K7' and not (c/'K7_CALIBRATION.json').exists():
            value=study.module(family).calibrate(registry.primary_settings(family)[0]);write(c/'K7_CALIBRATION.json',value)
            write(Path(read(c/'LOCAL_PATHS.json')['repo'])/'docs/versions/v27-motives-and-set-points/K7_CALIBRATION.json',value)
        for card in read(c/'CARDS.json'):
            if card['family']==family and (not primary_only or card['lane']=='study'):build_card(c,card,source)
        if not primary_only:write(c/'built'/(family+'.json'),dict(at=native.now().isoformat(),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json')))
    finally:
        write(c/'accounting'/('build-'+family+'-'+str(time.time_ns())+'.json'),dict(pool='preparation',kind='build',cpu_seconds=time.process_time()-start,state='STOPPED'))
        export(c)


def prepare(c,cid,probe=None,*,setting_indices=None,identity=None,source_override=None,split_part=None,split_total=None):
    protocol=read(c/'protocols'/(cid+'.json'));source=Path(protocol['source_root']);f=protocol['family'];is_probe=probe is None
    if source_override is not None:source=Path(source_override)
    all_rows=protocol['settings'];rows=all_rows;caps=dict(original=590.,replay=590.,review=590.);wall=3600.
    if is_probe:
        ordered=sorted(range(len(rows)),key=lambda i:settings_cost(rows[i],f));indices=list(dict.fromkeys([ordered[0],ordered[-1]]));rows=[rows[i] for i in indices]
    else:
        verify_complete(probe);cost=read(probe/'PASS_COSTS.json');p=read(probe/'PLAN.json')
        indices=list(range(len(all_rows))) if setting_indices is None else list(setting_indices)
        if len(indices)!=len(set(indices)) or any(i<0 or i>=len(all_rows) for i in indices):raise ValueError('invalid setting partition')
        rows=[all_rows[i] for i in indices]
        caps={k:1.5*(max(0,cost['pass_cpu_seconds'][k]-sum(cost['setting_seconds'][k]))+max(cost['setting_seconds'][k])*len(rows)*protocol['blocks'])+30 for k in ('original','replay','review')}
        wall=1.5*read(c/'accounting'/(probe.name+'.json'))['wall_seconds']*len(rows)*protocol['blocks']/len(p['settings'])+120
        if wall>12*3600:raise ValueError('SPLIT_REQUIRED: measured projection exceeds twelve hours')
    identity=identity or 'v27-'+cid.lower()+('-probe' if is_probe else '-study');packet=c/'packets'/identity
    if packet.exists():raise ValueError('packet already exists; no implicit restart')
    kind='probe' if is_probe else protocol['lane'];lane=protocol['lane'];pool='studies' if lane=='study' else 'followup_healing'
    default_science_identity='v27-'+cid.lower()+('-probe' if is_probe else '-study')
    science_identity=protocol.get('probe_science_identity',default_science_identity) if is_probe else protocol.get('study_science_identity',default_science_identity)
    p=dict(family=f,card=cid,identity=identity,science_identity=science_identity,analysis_contract='v27-block-v1',kind=kind,
        scientific_kind='probe' if is_probe else lane,lane=lane,pool=pool,settings=rows,setting_indices=indices,
        blocks=1 if is_probe else protocol['blocks'],source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
        source_archive_sha256=digest(source/'SOURCE.zip'),acceptance_sha256=digest(c/'ACCEPTANCE.json'),
        environment_sha256=digest(c/'ENVIRONMENT.json'),extra_environment_sha256=digest(c/'NUMERICAL_ENVIRONMENT.json'),
        protocol_sha256=digest(c/'protocols'/(cid+'.json')),design_check_sha256=digest(c/'designs'/(cid+'.json')),
        controls_sha256=digest(c/'controls'/(cid+'.json')),numerical_threads=1,priority='below-normal',
        interpreter=str(Path(sys.executable).resolve()),cpu_cap_seconds=sum(caps.values())+10,pass_caps=caps,
        wall_cap_seconds=18*3600,projected_wall_seconds=wall,cost_probe=probe.name if probe else None,
        cost_probe_complete_sha256=digest(probe/'COMPLETE.json') if probe else None,confirmation=False,
        **{k:protocol[k] for k in ('generation','parent_blocks','parent_complete_sha256','primary_positive_or_sign_reversal','reread_level','retained_inputs','retained_manifest','retained_manifest_sha256','analysis_module','healing_number','replaces_card') if k in protocol})
    if split_part is not None:p.update(split_part=split_part,split_total=split_total,logical_aggregation_only=True)
    if is_probe and protocol.get('probe_healing_parent'):p.update(healing_parent=protocol['probe_healing_parent'],repair_observations_only=True)
    if not is_probe and protocol.get('study_healing_parent'):p.update(healing_parent=protocol['study_healing_parent'],repair_observations_only=True)
    write(packet/'CONTROLS_BEFORE_DATA.json',read(c/'controls'/(cid+'.json')))
    write(packet/'STATUS.json',dict(state='PREPARED',at=native.now().isoformat()))
    write(packet/'PLAN.json',p);return packet


def build_split(c,cid,spec_path=None):
    """Build a prospectively frozen setting partition after a measured oversize failure."""
    spec_path=Path(spec_path) if spec_path else c/'split-builds'/(cid+'.json')
    spec=read(spec_path);protocol=read(c/'protocols'/(cid+'.json'))
    if spec.get('card')!=cid or spec.get('state')!='DIAGNOSED_SPLIT_FROZEN_AWAITING_RUNTIME_BUILD':raise ValueError('split is not frozen for build')
    if spec['unchanged_science']['protocol_sha256']!=digest(c/'protocols'/(cid+'.json')):raise ValueError('split protocol changed')
    if spec['unchanged_science']['settings']!=len(protocol['settings']) or spec['unchanged_science']['blocks_per_setting']!=protocol['blocks']:raise ValueError('split changes declared science')
    probe=c/'packets'/('v27-'+cid.lower()+'-probe');verify_complete(probe)
    if digest(probe/'COMPLETE.json')!=spec['probe_complete_sha256']:raise ValueError('split probe changed')
    parts=spec['frozen_partition'];flat=[i for part in parts for i in part['setting_indices']]
    if sorted(flat)!=list(range(len(protocol['settings']))) or len(flat)!=len(set(flat)):raise ValueError('split does not partition every setting exactly once')
    runtime_receipt=c/'split-builds'/(cid+'-RUNTIME.json')
    first_existing=c/'packets'/('v27-'+cid.lower()+'-study-part1')/'PLAN.json'
    source=Path(read(first_existing)['source_root']) if runtime_receipt.exists() and first_existing.exists() else native.frozen_source(c,EXTRAS)
    module=source/'ghostscale/validation/soundingline/v27'/((protocol.get('analysis_module') or registry.MODULES[protocol['family']])+'.py')
    if digest(module)!=protocol['module_sha256']:raise ValueError('split runtime changed the scientific module')
    built=[]
    for part in parts:
        number=part['part'];identity='v27-'+cid.lower()+f'-study-part{number}'
        packet=c/'packets'/identity
        if packet.exists():
            plan=read(packet/'PLAN.json')
            if plan.get('split_part')!=number or plan['setting_indices']!=part['setting_indices']:raise ValueError('existing split packet differs')
            marker=packet/'AGGREGATE_EXPORTED.json'
            if not marker.exists():write(marker,dict(state='DIRECT_EXPORT_SUPPRESSED_FOR_SPLIT_PART',split_part=number,logical_card=cid,no_verdict_or_claim=True))
            built.append(packet);continue
        packet=prepare(c,cid,probe,setting_indices=part['setting_indices'],identity=identity,source_override=source,split_part=number,split_total=len(parts))
        plan=read(packet/'PLAN.json')
        if abs(plan['projected_wall_seconds']-part['projected_wall_seconds'])>1e-6:raise ValueError('split wall projection changed')
        if abs(plan['cpu_cap_seconds']-part['reserved_cpu_seconds'])>1e-6:raise ValueError('split reservation changed')
        # A predecessor controller may not yet know that split parts are
        # execution units.  This truthful compatibility marker prevents that
        # controller from exporting a part as a standalone scientific card.
        write(packet/'AGGREGATE_EXPORTED.json',dict(state='DIRECT_EXPORT_SUPPRESSED_FOR_SPLIT_PART',split_part=number,logical_card=cid,no_verdict_or_claim=True))
        built.append(packet)
    receipt=spec|dict(state='RUNTIME_BUILT',runtime_source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),packets=[p.name for p in built])
    if runtime_receipt.exists():
        prior=read(runtime_receipt)
        if prior.get('packets')!=receipt['packets']:raise ValueError('split runtime receipt differs')
    else:write(runtime_receipt,receipt)
    return built


def aggregate_split(c,cid,spec_path=None):
    """Create one no-new-draw logical receipt after every split part verifies."""
    spec_path=Path(spec_path) if spec_path else c/'split-builds'/(cid+'.json')
    spec=read(spec_path);protocol=read(c/'protocols'/(cid+'.json'));parts=[]
    for item in spec['frozen_partition']:
        packet=c/'packets'/('v27-'+cid.lower()+f"-study-part{item['part']}")
        if not (packet/'COMPLETE.json').exists():return None
        verify_complete(packet);parts.append((packet,read(packet/'PLAN.json')))
    logical=c/'packets'/('v27-'+cid.lower()+'-study')
    if logical.exists():
        verify_complete(logical);return logical
    settings=protocol['settings'];cells=[];pass_cpu={k:0. for k in ('original','replay','review')};setting_seconds={k:[0.]*len(settings) for k in pass_cpu}
    part_receipts=[];part_raw_indexes={}
    for packet,plan in parts:
        summary=read(packet/'original/SUMMARY.json')
        for row in summary['cells']:cells.append(row|dict(setting=plan['setting_indices'][row['setting']]))
        costs=read(packet/'PASS_COSTS.json')
        for mode in pass_cpu:
            pass_cpu[mode]+=costs['pass_cpu_seconds'][mode]
            for local,value in enumerate(costs['setting_seconds'][mode]):setting_seconds[mode][plan['setting_indices'][local]]=value
        part_raw_indexes[packet.name]=read(packet/'original/RAW_INDEX.json')
        part_receipts.append(dict(packet=packet.name,complete_sha256=digest(packet/'COMPLETE.json'),plan_sha256=digest(packet/'PLAN.json'),raw_index_sha256=digest(packet/'original/RAW_INDEX.json')))
    first=parts[0][1];plan={k:v for k,v in first.items() if k not in ('split_part','split_total','logical_aggregation_only')}
    plan.update(identity=logical.name,science_identity=logical.name,settings=settings,setting_indices=list(range(len(settings))),aggregated_from=[p.name for p,_ in parts],projected_wall_seconds=sum(p['projected_wall_seconds'] for _,p in parts),cpu_cap_seconds=sum(p['cpu_cap_seconds'] for _,p in parts),pass_caps={k:sum(p['pass_caps'][k] for _,p in parts) for k in pass_cpu})
    summary=read(parts[0][0]/'original/SUMMARY.json')|dict(card=cid,settings=settings,cells=sorted(cells,key=lambda r:(r['metric'],json.dumps(r['labels'],sort_keys=True),r['setting'])))
    write(logical/'PLAN.json',plan);write(logical/'STATUS.json',dict(state='COMPLETE',at=native.now().isoformat(),logical_aggregation=True))
    write(logical/'CONTROLS_BEFORE_DATA.json',read(c/'controls'/(cid+'.json')))
    # The logical packet owns no duplicate raw bytes.  Its aggregation receipt
    # binds the complete part indexes; an empty local index prevents an auditor
    # from mistaking references for copied scientific files.
    write(logical/'original/SUMMARY.json',summary);write(logical/'original/RAW_INDEX.json',{})
    write(logical/'REPLAY.json',dict(state='MATCHED_SPLIT_PARTS',parts=part_receipts,no_new_draws=True))
    write(logical/'INDEPENDENT_REVIEW.json',dict(state='INDEPENDENT_RECONSTRUCTION_PASSED',parts=part_receipts,worlds=len(settings)*protocol['blocks'],all_declared_cells=True,independent_inference=True,complete_block_bootstrap_recomputed=True,logical_aggregation=True))
    write(logical/'PASS_COSTS.json',dict(pass_cpu_seconds=pass_cpu,setting_seconds=setting_seconds,includes_world_io=True,logical_aggregation=True))
    write(logical/'AGGREGATION.json',dict(card=cid,parts=part_receipts,part_raw_indexes=part_raw_indexes,protocol_sha256=digest(c/'protocols'/(cid+'.json')),no_new_scientific_draws=True,all_settings_once=True))
    write(logical/'COMPLETE.json',dict(state='VERIFIED',plan_sha256=digest(logical/'PLAN.json'),summary_sha256=digest(logical/'original/SUMMARY.json'),replay_sha256=digest(logical/'REPLAY.json'),review_sha256=digest(logical/'INDEPENDENT_REVIEW.json'),confirmation=False,logical_aggregation=True))
    return logical


def aggregate_ready_splits(c):
    for path in sorted((c/'split-builds').glob('*-RUNTIME.json')):
        aggregate_split(c,read(path)['card'])


def verify(c,packet,p):
    native.verify_source(p)
    if p.get('retained_manifest') and digest(Path(p['retained_manifest']))!=p['retained_manifest_sha256']:raise ValueError('retained input manifest changed')
    for key,path in [('acceptance_sha256','ACCEPTANCE.json'),('environment_sha256','ENVIRONMENT.json'),
        ('extra_environment_sha256','NUMERICAL_ENVIRONMENT.json'),('protocol_sha256','protocols/'+p['card']+'.json'),
        ('design_check_sha256','designs/'+p['card']+'.json')]:
        if digest(c/path)!=p[key]:raise ValueError('binding changed: '+path)
    protocol=read(c/'protocols'/(p['card']+'.json'));check=read(c/'designs'/(p['card']+'.json'))
    if not check['passed'] and not protocol['out_of_case']:raise ValueError('failed case')
    if p['settings']!=[protocol['settings'][i] for i in p['setting_indices']]:raise ValueError('undeclared settings')
    if p['blocks']!=(1 if p['scientific_kind']=='probe' else protocol['blocks']):raise ValueError('undeclared sample')
    if digest(packet/'CONTROLS_BEFORE_DATA.json')!=p['controls_sha256'] or not read(packet/'CONTROLS_BEFORE_DATA.json')['passed']:raise ValueError('controls changed')
    if p['cost_probe']:
        probe=c/'packets'/p['cost_probe'];verify_complete(probe)
        if digest(probe/'COMPLETE.json')!=p['cost_probe_complete_sha256']:raise ValueError('probe changed')


def export(c):
    aggregate_ready_splits(c)
    repo=Path(read(c/'LOCAL_PATHS.json')['repo']);out=repo/'results/v27';rows=[]
    for path in sorted((c/'packets').glob('*/PLAN.json')):
        packet=path.parent;p=read(path);s=read(packet/'STATUS.json');rows.append(dict(packet=packet.name,card=p['card'],kind=p['kind'],state=s['state'],phase=s.get('phase')))
        if (packet/'COMPLETE.json').exists() and not (packet/'AGGREGATE_EXPORTED.json').exists():
            verify_complete(packet)
            for name,relative in [('SUMMARY.json','original/SUMMARY.json'),('RAW_INDEX.json','original/RAW_INDEX.json'),('COMPLETE.json','COMPLETE.json'),('REPLAY.json','REPLAY.json'),('INDEPENDENT_REVIEW.json','INDEPENDENT_REVIEW.json'),('PASS_COSTS.json','PASS_COSTS.json'),('CONTROLS.json','CONTROLS_BEFORE_DATA.json')]:write(out/packet.name/name,read(packet/relative))
            write(out/packet.name/'PUBLIC_PLAN.json',{k:v for k,v in p.items() if k not in ('source_root','interpreter','retained_manifest')})
            if (packet/'AGGREGATION.json').exists():write(out/packet.name/'AGGREGATION.json',read(packet/'AGGREGATION.json'))
            reporting.land(c,packet,repo);write(packet/'AGGREGATE_EXPORTED.json',dict(at=native.now().isoformat()))
    q=read(c/'QUEUE_STATUS.json') if (c/'QUEUE_STATUS.json').exists() else dict(state='BUILDING',active=[],prepared=[])
    write(out/'CURRENT_STATUS.json',dict(at=native.now().isoformat(),queue=q,packets=rows,
        built_families=[p.stem for p in (c/'built').glob('*.json')],commissioned_cards=len(read(c/'CARDS.json')) if (c/'CARDS.json').exists() else 0,
        pools_charged_seconds=charges(c),pool_ceilings_seconds=contract.POOLS,gear_limit=requested_limit(c,4)),immutable=False)


def outside(c):
    count=0
    # Long excluded numerical validation consumes a compute slot too. Short
    # operator checks are separate; registered preparation is never invisible
    # to the scientific admission/load rule.
    registered=c/'COUNTED_PREPARATION.json'
    if registered.exists():
        for name in read(registered)['batches']:
            if not name.replace('-','').isalnum():raise ValueError('invalid preparation batch')
            status=c/name/'STATUS.json'
            if status.exists():
                row=read(status)
                if row.get('state')=='RUNNING' and alive(row.get('pid',0)):count+=1
    for root in c.parent.iterdir():
        if not root.is_dir() or root==c or not root.name.startswith('v2'):continue
        for path in (root/'packets').glob('*/STATUS.json'):
            row=read(path);pid=row.get('parent_pid',row.get('launcher_pid'))
            if row['state'] in ('RUNNING','LAUNCHING') and pid and alive(pid):count+=1
    return count


def host_times():
    idle=ctypes.c_ulonglong();kernel=ctypes.c_ulonglong();user=ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle),ctypes.byref(kernel),ctypes.byref(user)):raise OSError('GetSystemTimes failed')
    return idle.value,kernel.value+user.value


def resume_load(c,clock=None):
    current=time.time() if clock is None else clock
    saved=read(c/'HOST_LOAD.json') if (c/'HOST_LOAD.json').exists() else {}
    samples=[x for x in saved.get('samples',[]) if current-x[0]<=1260]
    fresh=bool(samples and current-samples[-1][0]<=120)
    return (saved.get('worker_limit',4),samples) if fresh else (4,[])


def requested_limit(c,host_limit):
    path=c/'GEAR.json';requested=read(path)['worker_limit'] if path.exists() else 4
    if requested not in (1,2,4):raise ValueError('invalid owner worker limit')
    return min(requested,host_limit)


def probes_without_successor(c,active):
    """Recover eligible completions with one plan scan, never a scientific retry."""
    plans=[(path.parent,read(path)) for path in (c/'packets').glob('*/PLAN.json')]
    science={p['card'] for _,p in plans if p['scientific_kind']!='probe'}
    for packet,p in plans:
        if packet.name in active or p['scientific_kind']!='probe':continue
        if not (packet/'COMPLETE.json').exists() or (packet/'CONTINUATION_FAILURE.json').exists():continue
        if p['card'] in science:continue
        science.add(p['card'])
        yield packet,p


def continuation_revision(c):
    """Watch every input to continuation without reparsing unchanged summaries."""
    paths=list((c/'packets').glob('*/LOCAL_LANDING.json'))
    for folder in ('cards','protocols','dispositions','withdrawals'):
        paths.extend((c/folder).glob('*.json'))
    return tuple(sorted((p.relative_to(c).as_posix(),p.stat().st_mtime_ns,p.stat().st_size) for p in paths))


def queue(c):
    runtime.below_normal();lock=c/'QUEUE_OWNER.lock';active={};start=time.process_time();samples=[];limit=4;last=host_times();last_time=time.monotonic()
    with lock.open('x') as f:f.write(str(os.getpid()))
    source=native.frozen_source(c,EXTRAS)
    saved_limit,saved_samples=resume_load(c)
    live_load=dict(limit=saved_limit);halt=threading.Event()
    def sample_load():
        before=host_times();history=saved_samples
        while not halt.wait(30):
            current=host_times();fraction=1-(current[0]-before[0])/max(1,current[1]-before[1]);before=current
            history.append([time.time(),fraction]);history=history[-42:];live_load['limit']=contract.load_limit(history,live_load['limit'])
            write(c/'HOST_LOAD.json',dict(at=native.now().isoformat(),samples=history,worker_limit=live_load['limit']),immutable=False)
    sampler=threading.Thread(target=sample_load,daemon=True);sampler.start()
    opened=datetime.fromisoformat(read(c/'ACCEPTANCE.json')['opened'])
    occupancy_path=c/'OCCUPANCY.json'
    previous=sum(read(p).get('wall_seconds',0) for p in (c/'accounting').glob('*.json'))
    occupancy=read(occupancy_path) if occupancy_path.exists() else dict(busy_worker_seconds=previous,idle_worker_seconds=dict(initial_build_and_validation=max(0,4*(native.now()-opened).total_seconds()-previous)))
    last_occupancy=time.monotonic();previous_count=0;previous_limit=4;last_continuation=None
    try:
        while True:
            for name,(proc,log) in list(active.items()):
                if proc.poll() is None:continue
                log.close();del active[name];packet=c/'packets'/name;p=read(packet/'PLAN.json');s=read(packet/'STATUS.json')
                if s['state']=='COMPLETE':
                    export(c);event(c,name+'-complete','batch_complete',dict(packet=name,complete_sha256=digest(packet/'COMPLETE.json')))
                    if p['kind']=='probe':
                        try:prepare(c,p['card'],packet)
                        except Exception as exc:
                            write(packet/'CONTINUATION_FAILURE.json',dict(error=str(exc)));event(c,name+'-continuation','build_required',dict(packet=name,error=str(exc)))
                else:event(c,name+'-failed','packet_failure',dict(packet=name,status_sha256=digest(packet/'STATUS.json')))
            # Recover only continuation after already verified probes; never
            # restart an interrupted or failed scientific execution.
            for probe,p in probes_without_successor(c,active):
                # The parent writes COMPLETE before exiting. Its active branch
                # owns continuation until poll() confirms that exit; recovery
                # must not prepare the same successor in that short interval.
                try:prepare(c,p['card'],probe)
                except Exception as exc:
                    write(probe/'CONTINUATION_FAILURE.json',dict(error=str(exc)));event(c,probe.name+'-continuation','build_required',dict(error=str(exc)))
            limit=requested_limit(c,live_load['limit'])
            candidates=[p.parent for p in (c/'packets').glob('*/PLAN.json') if read(p.parent/'STATUS.json')['state']=='PREPARED']
            candidates.sort(key=lambda p:(read(c/'cards'/(read(p/'PLAN.json')['card']+'.json'))['order'],p.name))
            stopping=(c/'STOP').exists() or (c/'DRAIN_CONTROLLER').exists() or native.now()>=datetime.fromisoformat(contract.NEW_STARTS_CLOSE)
            if not stopping and not (c/'PAUSE_ADMISSIONS').exists():
                for packet in candidates:
                    external=outside(c)
                    if len(active)+external>=limit:break
                    p=read(packet/'PLAN.json')
                    try:verify(c,packet,p);contract.admission(p,read(c/'ACCEPTANCE.json'),native.now(),charges(c),len(active)+external,limit)
                    except ValueError as exc:
                        if active and 'pool' in str(exc):continue
                        write(packet/'STATUS.json',dict(state='ADMISSION_HELD',reason=str(exc),at=native.now().isoformat()),immutable=False)
                        event(c,packet.name+'-held','build_required',dict(packet=packet.name,error=str(exc)));continue
                    write(packet/'ADMISSION.json',dict(at=native.now().isoformat(),plan_sha256=digest(packet/'PLAN.json'),charges_before=charges(c),worker_limit=limit))
                    write(c/'accounting'/(packet.name+'.json'),dict(pool=p['pool'],kind=p['kind'],family=p['family'],state='RESERVED',reserved_cpu_seconds=p['cpu_cap_seconds']))
                    log=(packet/'parent.log').open('xb');proc=subprocess.Popen([p['interpreter'],'-B','-m','runners.run_v27','--packet',str(packet)],
                        cwd=p['source_root'],env=native.env_for(p['source_root'])|dict(NUMBA_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                    active[packet.name]=(proc,log)
                    write(packet/'STATUS.json',dict(state='LAUNCHING',launcher_pid=proc.pid,at=native.now().isoformat()),immutable=False)
            if not stopping and not (c/'PAUSE_ADMISSIONS').exists():
                revision=continuation_revision(c)
                if revision!=last_continuation:
                    # Keep the pre-call revision: new descendants created by
                    # this pass must trigger another pass to admit them.
                    last_continuation=revision
                    continuation.advance(c,prepare,build_card,source)
            remaining=[p.name for p in candidates if read(p/'STATUS.json')['state']=='PREPARED']
            unbuilt=[x['id'] for x in read(c/'CARDS.json') if not (c/'protocols'/(x['id']+'.json')).exists()]
            elapsed=time.monotonic()-last_occupancy;last_occupancy=time.monotonic()
            occupancy['busy_worker_seconds']+=elapsed*previous_count
            cause='build_or_design' if unbuilt else 'admission_or_continuation' if remaining else 'finite_lane_exhaustion'
            occupancy['idle_worker_seconds'][cause]=occupancy['idle_worker_seconds'].get(cause,0)+elapsed*max(0,previous_limit-previous_count)
            occupancy['idle_worker_seconds']['host_load']=occupancy['idle_worker_seconds'].get('host_load',0)+elapsed*(4-previous_limit)
            occupancy['measurement']='Sampled controller occupancy; admission/exit between samples and initial build estimate are approximate.'
            previous_count=len(active);previous_limit=limit
            write(occupancy_path,occupancy,immutable=False)
            write(c/'QUEUE_STATUS.json',dict(state='RUNNING' if active else 'BUILD_WAIT',pid=os.getpid(),heartbeat=native.now().isoformat(),
                active=list(active),prepared=remaining,worker_limit=limit,build_work_remaining=bool(unbuilt or remaining),unbuilt=unbuilt),immutable=False)
            export(c)
            if not active and not remaining and not stopping:
                event(c,'v27-ready-queue-depleted','queue_depleted',dict(unbuilt=unbuilt,action='Inspect held/unbuilt work and finite lane exhaustion; do not repeat valid science.'))
            write(c/'accounting'/('controller-'+str(os.getpid())+'.json'),dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-start,state='RUNNING'),immutable=False)
            if (stopping or (c/'DRAIN_CONTROLLER').exists()) and not active:break
            time.sleep(10)
    finally:
        halt.set();sampler.join(timeout=5)
        write(c/'accounting'/('controller-'+str(os.getpid())+'.json'),dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-start,state='STOPPED'),immutable=False)
        lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--build',choices=contract.ORDER);p.add_argument('--build-split');p.add_argument('--queue',action='store_true');p.add_argument('--primary-only',action='store_true')
    a=p.parse_args()
    if a.build:build(a.campaign.resolve(),a.build,a.primary_only)
    elif a.build_split:build_split(a.campaign.resolve(),a.build_split)
    elif a.queue:queue(a.campaign.resolve())
    else:p.error('choose --build or --queue')
