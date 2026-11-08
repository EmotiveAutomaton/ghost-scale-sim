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


def prepare(c,cid,probe=None):
    protocol=read(c/'protocols'/(cid+'.json'));source=Path(protocol['source_root']);f=protocol['family'];is_probe=probe is None
    rows=protocol['settings'];caps=dict(original=590.,replay=590.,review=590.);wall=3600.
    if is_probe:
        ordered=sorted(range(len(rows)),key=lambda i:settings_cost(rows[i],f));indices=list(dict.fromkeys([ordered[0],ordered[-1]]));rows=[rows[i] for i in indices]
    else:
        verify_complete(probe);cost=read(probe/'PASS_COSTS.json');p=read(probe/'PLAN.json');indices=list(range(len(rows)))
        caps={k:1.5*(max(0,cost['pass_cpu_seconds'][k]-sum(cost['setting_seconds'][k]))+max(cost['setting_seconds'][k])*len(rows)*protocol['blocks'])+30 for k in ('original','replay','review')}
        wall=1.5*read(c/'accounting'/(probe.name+'.json'))['wall_seconds']*len(rows)*protocol['blocks']/len(p['settings'])+120
        if wall>12*3600:raise ValueError('SPLIT_REQUIRED: measured projection exceeds twelve hours')
    identity='v27-'+cid.lower()+('-probe' if is_probe else '-study');packet=c/'packets'/identity
    if packet.exists():raise ValueError('packet already exists; no implicit restart')
    kind='probe' if is_probe else protocol['lane'];lane=protocol['lane'];pool='studies' if lane=='study' else 'followup_healing'
    science_identity=protocol.get('probe_science_identity',identity) if is_probe else protocol.get('study_science_identity',identity)
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
    if is_probe and protocol.get('probe_healing_parent'):p.update(healing_parent=protocol['probe_healing_parent'],repair_observations_only=True)
    if not is_probe and protocol.get('study_healing_parent'):p.update(healing_parent=protocol['study_healing_parent'],repair_observations_only=True)
    write(packet/'CONTROLS_BEFORE_DATA.json',read(c/'controls'/(cid+'.json')))
    write(packet/'STATUS.json',dict(state='PREPARED',at=native.now().isoformat()))
    write(packet/'PLAN.json',p);return packet


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
    repo=Path(read(c/'LOCAL_PATHS.json')['repo']);out=repo/'results/v27';rows=[]
    for path in sorted((c/'packets').glob('*/PLAN.json')):
        packet=path.parent;p=read(path);s=read(packet/'STATUS.json');rows.append(dict(packet=packet.name,card=p['card'],kind=p['kind'],state=s['state'],phase=s.get('phase')))
        if (packet/'COMPLETE.json').exists() and not (packet/'AGGREGATE_EXPORTED.json').exists():
            verify_complete(packet)
            for name,relative in [('SUMMARY.json','original/SUMMARY.json'),('RAW_INDEX.json','original/RAW_INDEX.json'),('COMPLETE.json','COMPLETE.json'),('REPLAY.json','REPLAY.json'),('INDEPENDENT_REVIEW.json','INDEPENDENT_REVIEW.json'),('PASS_COSTS.json','PASS_COSTS.json'),('CONTROLS.json','CONTROLS_BEFORE_DATA.json')]:write(out/packet.name/name,read(packet/relative))
            write(out/packet.name/'PUBLIC_PLAN.json',{k:v for k,v in p.items() if k not in ('source_root','interpreter','retained_manifest')})
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
    last_occupancy=time.monotonic();previous_count=0;previous_limit=4
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
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--build',choices=contract.ORDER);p.add_argument('--queue',action='store_true');p.add_argument('--primary-only',action='store_true')
    a=p.parse_args()
    if a.build:build(a.campaign.resolve(),a.build,a.primary_only)
    elif a.queue:queue(a.campaign.resolve())
    else:p.error('choose --build or --queue')
