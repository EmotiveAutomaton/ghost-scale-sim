"""Prospective finite theory-day queue; two native scientists, no descendants."""
import argparse,os,shutil,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from ghostscale.validation.soundingline.v26 import day,day_study
from ghostscale.validation.soundingline.v26.common import read,write,digest,canonical
from ghostscale.validation.soundingline.v23 import runtime
from runners import run_v26 as base,run_v26_friday as friday
from runners.run_v26_followups import verify_complete
from runners.watch_v23_events import event,alive

EXTRAS=('runners/run_v26_day.py','runners/run_v26_friday.py','runners/run_v26_followups.py')

def charges(campaign):
    out=dict(read(campaign/'PARENT_CHARGES.json')['charges'])
    for k,v in base.charges(campaign).items():out[k]=out.get(k,0)+v
    for p in (campaign/'accounting').glob('*.json'):
        row=read(p)
        if row.get('lane')=='texture':
            key='texture:'+row['family'];out[key]=out.get(key,0)+max(row.get('cpu_seconds',0),row.get('reserved_cpu_seconds',0))
    return out

def design(card):
    impl=day_study.module(card['family']);seen={};checks=[]
    for s in card['settings']:
        key={k:v for k,v in s.items() if k not in ('truth','makers','worlds_per_block','sources','theta','delta','offgrid','tau_known','rho')}
        if card['family']=='G3':key={k:v for k,v in s.items() if k not in ('theta','delta')}
        identity=canonical(key).decode()
        if identity not in seen:seen[identity]=impl.design_check(s,[card['id'],key])
        checks.append(seen[identity])
    return dict(passed=all(x['passed'] for x in checks),checks=checks,seed_domain=card['case_check_seed_domain'],scientific_draws=0)

def build(campaign):
    runtime.below_normal();started=time.process_time()
    if (campaign/'BUILD_COMPLETE.json').exists():raise ValueError('already built')
    repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);out=repo/'docs/versions/v26-top-and-ends'/day.REL
    metadata=read(campaign/'PROTOCOL_METADATA.json');cards=day.cards();source=base.frozen_source(campaign,EXTRAS);dispositions=[]
    write(campaign/'CARDS.json',cards);write(out/'CARDS.json',cards)
    try:
        for index,card in enumerate(cards):
            f=card['family'];cid=card['id'];protocol=card|metadata[f]|dict(interpreter=str(Path(sys.executable).resolve()),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),module_sha256=digest(source/'ghostscale/validation/soundingline/v26'/(day.MODULES[f]+'.py')),provisional_modelling_choices=True)
            write(campaign/'protocols'/(cid+'.json'),protocol)
            check=design(card);write(campaign/'designs'/(cid+'.json'),check)
            # Only explicit bounded grid revision is allowed; the protocol and failed check stay intact.
            if not check['passed'] and card['lane']=='primary' and f in ('G1','G2'):
                candidates=([dict(menu=m,restore=r,urgency_floor=u) for m in (.25,.35,.5) for r in (4,5) for u in (.1,1.)] if f=='G1' else [dict(specialists=p,cue=q) for p in (.15,.25,.5) for q in (.15,.2)])
                for revision,change in enumerate(candidates,1):
                    revised=card|dict(settings=[s|change for s in card['settings']]);rp=protocol|dict(settings=revised['settings'],revision=revision,revision_of=cid+'.json')
                    name=cid+'-GRID-'+str(revision)+'.json';write(campaign/'protocols'/name,rp);rc=design(revised);write(campaign/'designs'/name,rc)
                    if rc['passed']:protocol=rp;check=rc;card=revised;protocol['protocol_file']=name;protocol['design_file']=name;break
            protocol.setdefault('protocol_file',cid+'.json');protocol.setdefault('design_file',cid+'.json')
            protocol['out_of_case']=bool(card['declared_out_of_case'] or (card['lane']=='texture' and not check['passed']))
            # Separate execution protocol includes the already recorded case classification.
            execution=cid+'-EXECUTION.json';write(campaign/'protocols'/execution,protocol);protocol['execution_file']=execution
            write(out/'protocols'/(cid+'.json'),{k:v for k,v in protocol.items() if k!='interpreter'})
            write(out/'protocols'/(cid+'-DESIGN.json'),check)
            if not (campaign/'controls'/(f+'.json')).exists():
                control=day_study.module(f).controls();write(campaign/'controls'/(f+'.json'),control)
                if not control['passed']:raise ValueError('known-answer failure: '+f)
            if check['passed'] or card['lane']=='texture':
                if card['lane']=='primary':p=prepare(campaign,protocol,source);state='PROBE_PREPARED'
                else:p=None;state='TEXTURE_BUILT_WAITING_PRIMARY'
            else:p=None;state='DESIGN_CANNOT_PRODUCE_CASE'
            dispositions.append(dict(card=cid,family=f,lane=card['lane'],state=state,packet=p.name if p else None,execution_file=execution))
            write(campaign/'BUILD_PROGRESS.json',dict(at=base.now().isoformat(),dispositions=dispositions),immutable=False)
            write(campaign/'accounting/build.json',dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
        write(campaign/'BUILD_COMPLETE.json',dict(at=base.now().isoformat(),source=str(source),dispositions=dispositions))
        if sum(x['state']=='DESIGN_CANNOT_PRODUCE_CASE' for x in dispositions)>=2:event(campaign,'two-design-failures','owner_trigger',dict(reason='At least two registered case checks exhausted their declared grids.'))
        write(campaign/'QUEUE_STATUS.json',dict(state='PREPARED',active=[],prepared=[x['packet'] for x in dispositions if x['packet']],gear=2),immutable=False);export(campaign)
    finally:write(campaign/'accounting/build.json',dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started,state='STOPPED'),immutable=False)

def measured_caps(probe,blocks):
    cost=read(probe/'PASS_COSTS.json');result={}
    for label in ('original','replay','review'):
        elapsed=cost['pass_cpu_seconds'][label];units=cost['setting_seconds'][label]
        result[label]=1.5*(max(0.,elapsed-sum(units))+sum(units)*blocks)+30
    result['original']=result['replay']=max(result['original'],result['replay'])
    return result

def prepare(campaign,protocol,source,probe=None):
    started=time.process_time();cid=protocol['id'];f=protocol['family'];is_probe=probe is None;settings=protocol['settings'];caps={x:min(570.,day.STUDY_HOURS[f]*3600/6) for x in ('original','replay','review')};projection=1800.;shed=[]
    if not is_probe:
        verify_complete(probe);caps=measured_caps(probe,16);projection=1.5*read(campaign/'accounting'/(probe.name+'.json'))['wall_seconds']*16+900
        # Re-price only declared shedding, before any scientific study draw.
        if protocol['lane']=='primary':
            old=charges(campaign).get('family:'+f,0);ceiling=day.STUDY_HOURS[f]*3600
            if old+sum(caps.values())+10>ceiling:
                original_records=sum(s['records']*s.get('makers',s.get('worlds_per_block',1)*s.get('sources',1)) for s in settings)
                if f in ('G1','G2'):settings=[s|dict(records=256) for s in settings];shed.append('512-record level')
                elif f=='G3':settings=[s for s in settings if s['theta']!=.5];shed.append('theta 0.5 cell')
                elif f=='G4':settings=[s for s in settings if s['rho']!=1/32];shed.append('rho 1/32 cell')
                elif f=='G5':settings=[s|dict(makers=12) if s['truth']=='SP' and s['offgrid'] else s for s in settings];shed.append('off-grid SP worlds beyond 192')
                ratio=sum(s['records']*s.get('makers',s.get('worlds_per_block',1)*s.get('sources',1)) for s in settings)/original_records
                caps={k:30+(v-30)*ratio for k,v in caps.items()};projection=900+(projection-900)*ratio
            if old+sum(caps.values())+10>ceiling and f in ('G1','G3','G4'):
                before=len(settings)
                if f=='G1':settings=[s|dict(omit_firings_matrix=True) for s in settings];shed.append('firings truth-by-rival matrix; P2 and arm retained')
                elif f=='G3':settings=[s|dict(delta_honest=False) for s in settings];shed.append('honest-source half of delta grid')
                else:settings=[s for s in settings if not s['tau_known']];shed.append('tau-known cell');caps={k:30+(v-30)*len(settings)/before for k,v in caps.items()}
        if shed:
            protocol=protocol|dict(settings=settings,shed=shed,execution_file=cid+'-SHED.json');write(campaign/'protocols'/protocol['execution_file'],protocol)
            write(Path(read(campaign/'LOCAL_PATHS.json')['repo'])/'docs/versions/v26-top-and-ends'/day.REL/'protocols'/(cid+'-SHED.json'),{k:v for k,v in protocol.items() if k!='interpreter'})
    identity=protocol['seed_domain']+('-probe' if is_probe else '-study');packet=campaign/'packets'/identity
    if packet.exists():raise ValueError('packet exists; recovery review required')
    packet.mkdir(parents=True);shutil.copy2(campaign/'controls'/(f+'.json'),packet/'CONTROLS_BEFORE_DATA.json')
    plan=dict(family=f,card=cid,kind='probe' if is_probe else 'study' if protocol['lane']=='primary' else 'texture',scientific_kind='probe' if is_probe else protocol['lane'],lane=protocol['lane'],
        identity=identity,science_identity=identity,settings=settings,blocks=1 if is_probe else 16,
        analysis_contract='v26-day-block-v1',source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),source_archive_sha256=digest(source/'SOURCE.zip'),
        acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),environment_sha256=digest(campaign/'ENVIRONMENT.json'),extra_environment_sha256=digest(campaign/'NUMERICAL_ENVIRONMENT.json'),parent_charges_sha256=digest(campaign/'PARENT_CHARGES.json'),
        protocol_file=protocol['execution_file'],protocol_sha256=digest(campaign/'protocols'/protocol['execution_file']),design_file=protocol['design_file'],design_check_sha256=digest(campaign/'designs'/protocol['design_file']),controls_sha256=digest(packet/'CONTROLS_BEFORE_DATA.json'),
        pool='studies' if protocol['lane']=='primary' else 'followup_healing',cpu_cap_seconds=sum(caps.values())+10,pass_caps=caps,projected_wall_seconds=projection,wall_cap_seconds=18*3600,
        numerical_threads=1,priority='below-normal',interpreter=protocol['interpreter'],cost_probe=probe.name if probe else None,cost_probe_complete_sha256=digest(probe/'COMPLETE.json') if probe else None,
        queue_order=protocol['queue_order'],confirmation=False,shed=shed,shedding_exhausted=bool(shed))
    write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='PREPARED',at=base.now().isoformat()))
    write(campaign/'accounting'/(identity+'-preparation-'+str(time.time_ns())+'.json'),dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started));return packet

def verify_bindings(campaign,packet,p):
    base.verify_source(p)
    for key,name in [('acceptance_sha256','ACCEPTANCE.json'),('environment_sha256','ENVIRONMENT.json'),('extra_environment_sha256','NUMERICAL_ENVIRONMENT.json'),('parent_charges_sha256','PARENT_CHARGES.json'),('protocol_sha256','protocols/'+p['protocol_file']),('design_check_sha256','designs/'+p['design_file'])]:
        if digest(campaign/name)!=p[key]:raise ValueError('binding changed: '+name)
    for path,sha in read(campaign/'PARENT_CHARGES.json')['bound_records'].items():
        if digest(path)!=sha:raise ValueError('parent record changed')
    protocol=read(campaign/'protocols'/p['protocol_file']);check=read(campaign/'designs'/p['design_file'])
    if not check['passed'] and not protocol['out_of_case']:raise ValueError('case failed')
    if p['settings']!=protocol['settings'] or p['blocks']!=(1 if p['kind']=='probe' else 16):raise ValueError('undeclared sample')
    if digest(packet/'CONTROLS_BEFORE_DATA.json')!=p['controls_sha256'] or not read(packet/'CONTROLS_BEFORE_DATA.json')['passed']:raise ValueError('controls changed')
    if p['cost_probe']:
        probe=campaign/'packets'/p['cost_probe'];verify_complete(probe)
        if digest(probe/'COMPLETE.json')!=p['cost_probe_complete_sha256']:raise ValueError('probe changed')

def export(campaign):
    repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);out=repo/'results/v26'/day.REL;rows=[]
    for path in sorted((campaign/'packets').glob('*/PLAN.json')):
        p=path.parent;plan=read(path);s=read(p/'STATUS.json');rows.append(dict(packet=p.name,card=plan['card'],kind=plan['kind'],lane=plan['lane'],state=s['state'],phase=s.get('phase'),shed=plan.get('shed',[]),reserved_cpu_seconds=plan['cpu_cap_seconds'],projected_wall_seconds=plan['projected_wall_seconds']))
        if (p/'COMPLETE.json').exists() and not (p/'AGGREGATE_EXPORTED.json').exists():
            verify_complete(p);dest=out/p.name
            for name,relative in [('SUMMARY.json','original/SUMMARY.json'),('RAW_INDEX.json','original/RAW_INDEX.json'),('COMPLETE.json','COMPLETE.json'),('REPLAY.json','REPLAY.json'),('INDEPENDENT_REVIEW.json','INDEPENDENT_REVIEW.json'),('PASS_COSTS.json','PASS_COSTS.json'),('CONTROLS.json','CONTROLS_BEFORE_DATA.json')]:write(dest/name,read(p/relative))
            write(dest/'PUBLIC_PLAN.json',friday.public_plan(plan));write(p/'AGGREGATE_EXPORTED.json',dict(at=base.now().isoformat()))
    q=read(campaign/'QUEUE_STATUS.json') if (campaign/'QUEUE_STATUS.json').exists() else dict(state='BUILDING',active=[],prepared=[])
    write(out/'CURRENT_STATUS.json',dict(at=base.now().isoformat(),**q,packets=rows,fully_built=(campaign/'BUILD_COMPLETE.json').exists(),gear_limit=2,pools_charged_seconds=charges(campaign),pool_ceilings_seconds=day.POOLS,new_starts_close=day.contract.NEW_STARTS_CLOSE,drain=day.contract.DRAIN,followups_commissioned=False),immutable=False)

def outside(campaign,owned):
    total=friday.outside(campaign,owned)
    for path in (campaign.parent/'v26-friday-20261104/packets').glob('*/STATUS.json'):
        s=read(path);pid=s.get('parent_pid',s.get('launcher_pid'))
        if s['state'] in ('RUNNING','LAUNCHING') and pid and alive(pid):total+=1
    return total

def primary_terminal(campaign,family,built):
    study=campaign/'packets'/('v26-day-20261106-'+family.lower()+'-study')
    if study.exists():return read(study/'STATUS.json')['state'] in ('COMPLETE','FAILED','ADMISSION_HELD')
    probe=campaign/'packets'/('v26-day-20261106-'+family.lower()+'-probe')
    if (probe/'CONTINUATION_FAILURE.json').exists():return True
    if probe.exists() and read(probe/'STATUS.json')['state'] in ('FAILED','ADMISSION_HELD'):return True
    return any(x['family']==family and x['lane']=='primary' and x['state']=='DESIGN_CANNOT_PRODUCE_CASE' for x in built['dispositions'])

def texture_streak(campaign,family,cards):
    """Apply amendment F only to in-case variants, never to out-of-case maps."""
    primary=campaign/'packets'/('v26-day-20261106-'+family.lower()+'-study')
    if not (primary/'COMPLETE.json').exists():return 0
    def measures(summary):
        result={}
        for row in summary['cells']:
            config=summary['settings'][row['setting']];stratum={k:config[k] for k in ('rho','tau_known','theta','delta','offgrid') if k in config}
            key=canonical(dict(metric=row['metric'],labels=row['labels'],stratum=stratum)).decode();result[key]=row
        return result
    baseline=measures(read(primary/'original/SUMMARY.json'));streak=0
    for card in [c for c in cards if c['family']==family and c['lane']=='texture']:
        packet=campaign/'packets'/(card['seed_domain']+'-study')
        if not (packet/'COMPLETE.json').exists():break
        p=read(packet/'PLAN.json');protocol=read(campaign/'protocols'/p['protocol_file'])
        if protocol['out_of_case']:continue
        current=measures(read(packet/'original/SUMMARY.json'));unchanged=True
        for key,row in current.items():
            prior=baseline.get(key);bar=1. if row['statistic'] in ('median','median_difference') else .05
            if prior is None or row['estimate'] is None or prior['estimate'] is None or abs(row['estimate']-prior['estimate'])>bar:unchanged=False;break
        streak=streak+1 if unchanged else 0
        if streak>=5:break
    return streak

def queue(campaign):
    runtime.below_normal();started=time.process_time();lock=campaign/'QUEUE_OWNER.lock';active={};account=campaign/'accounting'/f'controller-{os.getpid()}-{time.time_ns()}.json'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        while True:
            for name,(proc,log) in list(active.items()):
                if proc.poll() is None:continue
                log.close();del active[name];packet=campaign/'packets'/name;p=read(packet/'PLAN.json');s=read(packet/'STATUS.json')
                if s['state']=='COMPLETE':
                    a=campaign/'accounting'/(name+'.json');row=read(a);row['lane']=p['lane'];write(a,row,immutable=False)
                    event(campaign,name+'-complete','batch_complete',dict(packet=name,complete_sha256=digest(packet/'COMPLETE.json')))
                    if p['kind']=='probe':
                        try:prepare(campaign,read(campaign/'protocols'/p['protocol_file'])|dict(execution_file=p['protocol_file']),Path(p['source_root']),packet)
                        except Exception as exc:
                            write(packet/'CONTINUATION_FAILURE.json',dict(error=str(exc)));event(campaign,name+'-prepare-failed','admission_failure',dict(packet=name,error=str(exc)))
                else:
                    account_path=campaign/'accounting'/(name+'.json')
                    if account_path.exists():
                        charge=read(account_path);charge['lane']=p['lane'];write(account_path,charge,immutable=False)
                    if s['state'] not in ('FAILED','ADMISSION_HELD'):write(packet/'STATUS.json',dict(state='FAILED',reason='native parent exited; reservation retained'),immutable=False)
                    event(campaign,name+'-failed','packet_failure',dict(packet=name,status_sha256=digest(packet/'STATUS.json')))
            # One sequential lane per family; new texture requires its verified primary.
            built=read(campaign/'BUILD_COMPLETE.json');cards=read(campaign/'CARDS.json');primaries_terminal=True
            for f in day.ORDER:
                if not primary_terminal(campaign,f,built):primaries_terminal=False
            if primaries_terminal:
                event(campaign,'day-primary-results','primary_review_boundary',dict(all_registered_primaries_terminal=True))
                for f in day.ORDER:
                    primary=campaign/'packets'/('v26-day-20261106-'+f.lower()+'-study')
                    if not (primary/'COMPLETE.json').exists():continue
                    if not (campaign/'TEXTURE_STOP'/f).exists() and texture_streak(campaign,f,cards)>=5:write(campaign/'TEXTURE_STOP'/f,dict(reason='five consecutive in-case variants without a practical estimate change',at=base.now().isoformat()))
                    for card in [c for c in cards if c['family']==f and c['lane']=='texture']:
                        base_name=card['seed_domain'];p=campaign/'packets'/(base_name+'-study');probe=campaign/'packets'/(base_name+'-probe')
                        if (p/'COMPLETE.json').exists():continue
                        if p.exists() or probe.exists():break
                        if (campaign/'TEXTURE_STOP'/f).exists():break
                        meta=next(x for x in built['dispositions'] if x['card']==card['id']);protocol=read(campaign/'protocols'/meta['execution_file'])|dict(execution_file=meta['execution_file'])
                        prepare(campaign,protocol,Path(built['source']));break
            candidates=[p.parent for p in (campaign/'packets').glob('*/PLAN.json') if read(p.parent/'STATUS.json')['state']=='PREPARED'];candidates.sort(key=lambda p:(read(p/'PLAN.json')['queue_order'],read(p/'PLAN.json')['kind']=='probe'))
            stopping=(campaign/'STOP').exists() or base.now()>=datetime.fromisoformat(day.contract.NEW_STARTS_CLOSE)
            if not stopping and not (campaign/'PAUSE_ADMISSIONS').exists():
                for packet in candidates:
                    if len(active)+outside(campaign,active)>=2:break
                    p=read(packet/'PLAN.json')
                    if p['kind']=='study' and p['lane']=='primary':
                        earlier=day.ORDER[:day.ORDER.index(p['family'])]
                        if any(not (campaign/'packets'/('v26-day-20261106-'+f.lower()+'-study')/'ADMISSION.json').exists() and not primary_terminal(campaign,f,built) for f in earlier):continue
                    try:verify_bindings(campaign,packet,p);day.admission(p,read(campaign/'ACCEPTANCE.json'),base.now(),charges(campaign),len(active)+outside(campaign,active))
                    except ValueError as exc:
                        if active and str(exc) in ('pool ceiling','texture ceiling','shedding required'):continue
                        write(packet/'STATUS.json',dict(state='ADMISSION_HELD',reason=str(exc),at=base.now().isoformat()),immutable=False);event(campaign,packet.name+'-held','admission_failure',dict(packet=packet.name,reason=str(exc)))
                        if p['lane']=='texture':write(campaign/'TEXTURE_STOP'/p['family'],dict(reason=str(exc),at=base.now().isoformat()))
                        continue
                    write(packet/'ADMISSION.json',dict(at=base.now().isoformat(),plan_sha256=digest(packet/'PLAN.json'),charges_before=charges(campaign)))
                    write(campaign/'accounting'/(packet.name+'.json'),dict(pool=p['pool'],kind=p['kind'],family=p['family'],lane=p['lane'],state='RESERVED',reserved_cpu_seconds=p['cpu_cap_seconds']))
                    log=(packet/'parent.log').open('xb');proc=subprocess.Popen([p['interpreter'],'-B','-m','runners.run_v26','--packet',str(packet)],cwd=p['source_root'],env=base.env_for(p['source_root'])|dict(NUMBA_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                    active[packet.name]=(proc,log);write(packet/'STATUS.json',dict(state='LAUNCHING',launcher_pid=proc.pid,at=base.now().isoformat()),immutable=False)
            if stopping:
                for p in candidates:
                    if read(p/'STATUS.json')['state']=='PREPARED':write(p/'STATUS.json',dict(state='ADMISSION_HELD',reason='new-start window closed or stop requested'),immutable=False)
            remaining=[p for p in candidates if read(p/'STATUS.json')['state']=='PREPARED'];drained=not active and not remaining
            write(campaign/'QUEUE_STATUS.json',dict(state='DRAINED' if drained else 'RUNNING' if active else 'WAITING',pid=os.getpid(),heartbeat=base.now().isoformat(),active=list(active),prepared=[p.name for p in remaining],gear=2),immutable=False)
            export(campaign)
            from ghostscale.validation.soundingline.v26.day_reporting import land
            for complete in (campaign/'packets').glob('*/COMPLETE.json'):land(campaign,complete.parent,Path(read(campaign/'LOCAL_PATHS.json')['repo']))
            write(account,dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='RUNNING'),immutable=False)
            if drained:event(campaign,'day-drained-'+str(os.getpid()),'review_boundary',dict(finite_queue=True,needs_refill=False));break
            time.sleep(10)
    finally:
        write(account,dict(pool='preparation',kind='controller',cpu_seconds=time.process_time()-started,state='STOPPED'),immutable=False)
        lock.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--build',action='store_true');parser.add_argument('--queue',action='store_true');args=parser.parse_args();campaign=args.campaign.resolve()
    if args.build:build(campaign)
    elif args.queue:queue(campaign)
    else:parser.error('choose build or queue')
