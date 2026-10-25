"""Separately authorized N16 production; immutable design and complete blocks.

The V24 pilot contract and pilot kernel remain unchanged. Stopping is on
complete balanced blocks only, with a predictable, unclipped e-process.
"""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import itertools,math,sys,time
import numpy as np
from ..v23.common import read,write,digest,canonical
from ..v23 import ledger
from ..v23.pipeline import Stream
from ..v23.family import rows
from . import n16,evidence,pilot
from contextlib import contextmanager

DRAIN='2026-11-02T06:00:00+00:00'
IDENTITY='v24-n16-confirmation-1'
EXCLUDED='v24-n16-production-excluded-1'

@contextmanager
def repeated_distribution_cache():
    """Memoize byte-identical inputs within one case; never approximate a key."""
    original=n16.model.distributions;cache={}
    def cached(mass,laws,available,scenario):
        key=(mass.shape,mass.dtype.str,mass.tobytes(),laws.shape,laws.dtype.str,laws.tobytes(),tuple(available),scenario)
        if key not in cache:cache[key]=original(mass,laws,available,scenario)
        return cache[key]
    n16.model.distributions=cached
    try:yield
    finally:n16.model.distributions=original

def acceptance(opened):
    return dict(version='v24-n16-production-1',opened=opened,drain=DRAIN,cpu_seconds=64*3600,
        builder_hours=8,maximum_blocks=260,cases_per_cell=5,epsilon=.1,margin=.02,
        alpha=.1,gamma=.5,log_threshold=math.log(20),order=['N16'],identity=IDENTITY,
        excluded_identity=EXCLUDED,priority='below-normal',numerical_threads=1,
        scientific_slots_shared_with_v25=2,independent_unit='complete balanced block of all 16 laws, 48 settings and five cases',
        calibration='results/v24/preparation/N16_DESIGN_CALIBRATION.json',power_guarantee=False)

def controls():
    native=n16.model.controls()
    assert native['passed']
    wealth=[];crossings=0
    for path in itertools.product((-1.,1.),repeat=8):
        bet=evidence.adaptive_bets(path,-1,margin=0);wealth.append(math.exp(bet['log_e']))
        crossings+=any(evidence.adaptive_bets(path[:k],-1,margin=0)['log_e']>=math.log(20) for k in range(1,9))
    assert abs(math.fsum(wealth)/len(wealth)-1)<1e-12
    assert crossings/len(wealth)<=.05
    assert evidence.adaptive_bets([1.]*30,-1,margin=0)['log_e']>math.log(20)
    a=evidence.adaptive_bets([.2,.3,-1],-1);b=evidence.adaptive_bets([.2,-1,10],-1)
    assert a['stakes'][:2]==b['stakes'][:2]
    return dict(passed=True,native=native,exhaustive_fair_sign_null=True,positive_fixture=True,past_only_stakes=True)

def dependency_check(plan):
    for path,sha in plan['dependency_files'].items():
        if digest(path)!=sha:raise ValueError('production dependency byte changed')
    sys.path.insert(0,plan['dependency_directory'])
    return ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])[0]

def validate(plan):
    a=read(Path(plan['accounting_root'])/'ACCEPTANCE.json')
    if a!=acceptance(a['opened']):raise ValueError('production acceptance changed')
    if plan['family']!='N16' or plan['kind'] not in ('production','excluded') or plan['epsilon']!=.1 or plan['cases_per_cell']!=5:raise ValueError('production design mismatch')
    if plan['science_identity']!=(EXCLUDED if plan['kind']=='excluded' else IDENTITY):raise ValueError('production phase identity mismatch')
    if type(plan['block_index']) is not int or not 0<=plan['block_index']<260:raise ValueError('invalid production block')
    if plan['variant_ids']!=[v['id'] for v in n16.model.variants()]:raise ValueError('incomplete variant grid')

def block_summary(records,plan):
    # Normalize only the aggregation coordinate; seed and raw identity remain global.
    normal=[dict(identity=[0,*r['identity'][1:]],mixture_gain=r['mixture_gain']) for r in records]
    mean=n16.aggregate(normal,5)[0]
    return dict(state='COMPLETE_BALANCED_BLOCK',block_index=plan['block_index'],excluded=plan['kind']=='excluded',
        science_identity=plan['science_identity'],cases=len(records),mean=mean,lower_bound=math.log(.1),confirmation=False)

def execute(root,plan,pulse):
    validate(plan);root=Path(root);root.mkdir(exist_ok=False);schema=dependency_check(plan)
    write(root/'CONTROLS.json',dict(passed=True,pre_data_controls_sha256=plan['controls_sha256']))
    streams={k:Stream(root/(k+'_points.jsonl.gz')) for k in ('public','truth','evidence','ledger')};scores=[];block=plan['block_index']
    try:
        for law in range(16):
            for variant in n16.model.variants():
                for case in range(5):
                    identity=[block,law,variant['id'],case]
                    with repeated_distribution_cache():public,truth,ev,score=n16.case(variant,plan['science_identity'],block,law,case,.1)
                    bundle=pilot.emit(schema,[plan['science_identity'],*identity],public)
                    for kind,payload in [('public',dict(public=public)),('truth',dict(truth=truth)),('evidence',dict(evidence=ev,score=score)),('ledger',dict(ledger=bundle))]:streams[kind].write(dict(identity=identity,**payload))
                    scores.append(dict(identity=identity,**score))
                pulse(setting=0,block=block,law=law,cases=len(scores))
    finally:
        for stream in streams.values():stream.close()
    write(root/'SUMMARY.json',block_summary(scores,plan));write(root/'RAW_INDEX.json',{p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(root.glob('*_points.jsonl.gz'))})

def reconstruct(root,plan,pulse):
    validate(plan);root=Path(root);schema=dependency_check(plan);scores=[];variants={v['id']:v for v in n16.model.variants()}
    index=read(root/'RAW_INDEX.json')
    if set(index)!={k+'_points.jsonl.gz' for k in ('public','truth','evidence','ledger')}:raise ValueError('incomplete raw roster')
    for name,binding in index.items():
        if digest(root/name)!=binding['sha256'] or (root/name).stat().st_size!=binding['bytes']:raise ValueError('raw binding changed')
    for public,truth,ev,transport in zip(*(rows(root/(k+'_points.jsonl.gz')) for k in ('public','truth','evidence','ledger')),strict=True):
        identity=public['identity'];block,law,vid,case=identity
        if block!=plan['block_index'] or any(r['identity']!=identity for r in (truth,ev,transport)):raise ValueError('stream identity mismatch')
        variant=variants[vid];mechanism=(16*block+law+case)%5 if variant['source']=='N9' else variant['original']['settings']['mechanism']
        if truth['truth']!={'law':law,'mechanism':mechanism}:raise ValueError('truth identity mismatch')
        ledger.validate(transport['ledger'],schema)
        if canonical(transport['ledger'])!=canonical(pilot.emit(schema,[plan['science_identity'],*identity],public['public'])):raise ValueError('canonical ledger mismatch')
        with repeated_distribution_cache():score=n16.check(variant,public['public'],truth['truth'],ev['evidence'],.1)
        if any(not math.isclose(score[k],ev['score'][k],abs_tol=1e-11,rel_tol=0) for k in score):raise ValueError('independent score mismatch')
        scores.append(dict(identity=identity,**score))
        if len(scores)%48==0:pulse(setting=0,block=block,cases=len(scores))
    result=block_summary(scores,plan);saved=read(root/'SUMMARY.json')
    if result['cases']!=3840 or not math.isclose(result['mean'],saved['mean'],abs_tol=1e-11,rel_tol=0):raise ValueError('incomplete or inconsistent balanced block')
    if {k:v for k,v in result.items() if k!='mean'}!={k:v for k,v in saved.items() if k!='mean'}:raise ValueError('summary metadata mismatch')
    return dict(state='INDEPENDENT_RECONSTRUCTION_PASSED',cases=3840,global_block=plan['block_index'],summary_sha256=digest(root/'SUMMARY.json'))

def charged(root):
    return math.fsum(max(r.get('cpu_seconds',0),r.get('reserved_cpu_seconds',0)) for r in (read(p) for p in (root/'accounting').glob('*.json')))

def admission(plan,current,running):
    validate(plan);root=Path(plan['accounting_root']);a=read(root/'ACCEPTANCE.json')
    if running>=2 or not datetime.fromisoformat(a['opened'])<=current<datetime.fromisoformat(DRAIN):raise ValueError('production window/shared slot')
    if plan['numerical_threads']!=1 or plan['priority']!='below-normal':raise ValueError('production resource mismatch')
    if charged(root)+plan['cpu_cap_seconds']>a['cpu_seconds']:raise ValueError('production total ceiling')
    if current+timedelta(seconds=2*plan['measured_wall_seconds'])>=datetime.fromisoformat(DRAIN):raise ValueError('last safe block start')
    if plan['kind']=='production' and not (root/'EXCLUDED_VERIFIED.json').exists():raise ValueError('excluded replay required')
    if plan.get('source_revision',1)>1 and plan['kind']=='production':
        if digest(plan['excluded_proof'])!=plan['excluded_proof_sha256']:raise ValueError('optimized excluded proof changed')
    return True

def prepare_block(campaign,template,number,excluded=False):
    plan=dict(template);identity='v24-n16-excluded-1' if excluded else f'v24-n16-confirmation-block-{number:03d}'
    if plan.get('source_revision',1)>1:identity+='-r'+str(plan['source_revision'])
    plan.update(identity=identity,science_identity=EXCLUDED if excluded else IDENTITY,block_index=number,kind='excluded' if excluded else 'production')
    packet=campaign/'packets'/identity
    if packet.exists():return packet
    packet.mkdir();write(packet/'PLAN.json',plan);write(packet/'CONTROLS_BEFORE_DATA.json',read(Path(plan['accounting_root'])/'CONTROLS.json'));write(packet/'STATUS.json',dict(state='PREPARED',at=datetime.now(timezone.utc).isoformat()))
    return packet

def advance(campaign,packet):
    plan=read(packet/'PLAN.json');root=Path(plan['accounting_root']);state=root/'checkpoints'/packet.name
    if (root/'TERMINAL_AFTER_REPAIR.json').exists():return
    terminal=read(root/'TERMINAL.json') if (root/'TERMINAL.json').exists() else None
    repairing=bool(plan.get('source_revision',1)>1 and plan['kind']=='excluded' and terminal and terminal['state']=='NOT_CONFIRMED_WITHIN_BUDGET')
    if terminal and not repairing and not (root/'RESOURCE_REPAIR_ACCEPTED.json').exists():return
    complete=read(packet/'COMPLETE.json')
    for key,name in [('plan_sha256','PLAN.json'),('summary_sha256','original/SUMMARY.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
        if digest(packet/name)!=complete[key]:raise ValueError('production checkpoint binding changed')
    summary=read(packet/'original/SUMMARY.json');cost=read(root/'accounting'/(packet.name+'.json'))
    revision=plan.get('source_revision',1);template=read(root/('TEMPLATE.json' if revision==1 else f'TEMPLATE_R{revision}.json'));template['pass_caps']={k:max(10.,1.5*v) for k,v in cost['pass_cpu_seconds'].items()};template['cpu_cap_seconds']=sum(template['pass_caps'].values())+10
    template.pop('healing_parent',None)
    if plan['kind']=='production' and plan.get('excluded_proof'):
        template['excluded_proof']=plan['excluded_proof'];template['excluded_proof_sha256']=plan['excluded_proof_sha256']
    template['measured_wall_seconds']=max(1.,cost['wall_seconds'])
    checkpoint=dict(packet=packet.name,at=datetime.now(timezone.utc).isoformat(),summary=summary,complete_sha256=digest(packet/'COMPLETE.json'),cpu_seconds=cost['cpu_seconds'],wall_seconds=cost['wall_seconds'])
    state.parent.mkdir(exist_ok=True)
    if not state.exists():write(state,checkpoint)
    if plan['kind']=='excluded':
        proof=root/('EXCLUDED_VERIFIED.json' if revision==1 else f'EXCLUDED_VERIFIED_R{revision}.json')
        if not proof.exists():write(proof,checkpoint)
        template.pop('healing_parent',None);template['excluded_proof']=str(proof);template['excluded_proof_sha256']=digest(proof)
        completed=list((root/'checkpoints').glob('v24-n16-confirmation-block-*'));next_block=len(completed)
        if repairing:
            if not (packet/'REPAIR_CONTINUITY.json').exists():raise ValueError('exact old/new excluded replay required before resource repair')
            projection=charged(root)+(260-next_block)*template['cpu_cap_seconds']
            repair=dict(state='ADMITTED' if projection<=64*3600 else 'COST_STILL_EXCEEDS_CEILING',source_revision=revision,
                original_terminal_sha256=digest(root/'TERMINAL.json'),excluded_complete_sha256=digest(packet/'COMPLETE.json'),
                exact_old_new_continuity_sha256=digest(packet/'REPAIR_CONTINUITY.json'),projected_cpu_seconds=projection,
                source_archive_sha256=plan['source_archive_sha256'],authorization='V25 standing automatic resource healing; unchanged design, retained first block, same64-hour ceiling, no scientific setting change.')
            if not (root/'RESOURCE_REPAIR_EVALUATION.json').exists():write(root/'RESOURCE_REPAIR_EVALUATION.json',repair)
            if repair['state']!='ADMITTED':return
            if not (root/'RESOURCE_REPAIR_ACCEPTED.json').exists():write(root/'RESOURCE_REPAIR_ACCEPTED.json',repair)
        prepare_block(campaign,template,next_block);return
    points=[read(p) for p in sorted((root/'checkpoints').glob('v24-n16-confirmation-block-*'))]
    if [p['summary']['block_index'] for p in points]!=list(range(len(points))):raise ValueError('nonconsecutive confirmation checkpoints')
    values=[p['summary']['mean'] for p in points];bet=evidence.adaptive_bets(values,math.log(.1),.02);used=charged(root)
    current_source=[p for p in points if read(campaign/'packets'/p['packet']/'PLAN.json')['source_archive_sha256']==plan['source_archive_sha256']]
    projection=used+(260-len(values))*1.5*math.fsum(p['cpu_seconds'] for p in current_source)/len(current_source)
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(read(root/'ACCEPTANCE.json')['opened'])).total_seconds()
    throughput=math.fsum(p['cpu_seconds'] for p in points)/elapsed if elapsed>0 else None
    status='RUNNING'
    if bet['log_e']>=math.log(20):status='CONFIRMED_CONSTRUCTED_SCHEDULER'
    elif projection>64*3600:status='NOT_CONFIRMED_WITHIN_BUDGET'
    elif len(values)>=260:status='NOT_CONFIRMED_AT_BLOCK_CAP'
    elif datetime.now(timezone.utc)+timedelta(seconds=2*template['measured_wall_seconds'])>=datetime.fromisoformat(DRAIN):status='NOT_CONFIRMED_BY_DRAIN'
    result=dict(state=status,blocks=len(values),block_means=values,e_process=bet,charged_cpu_seconds=used,projected_full_cpu_seconds=projection,
        elapsed_seconds=elapsed,cpu_per_wall_second=throughput,threshold=math.log(20),alpha=.1,gamma=.5,confirmation=status=='CONFIRMED_CONSTRUCTED_SCHEDULER',
        limitations=['Constructed scheduler under the declared model; no human or engine admission.', 'Discovery-based sample forecast does not guarantee power.'])
    write(root/'E_VALUE.json',result,immutable=False)
    if elapsed>=86400 and not (root/'THROUGHPUT_24H.json').exists():write(root/'THROUGHPUT_24H.json',result)
    if status=='RUNNING':prepare_block(campaign,template,len(values))
    else:
        terminal_path=root/('TERMINAL.json' if not (root/'RESOURCE_REPAIR_ACCEPTED.json').exists() else 'TERMINAL_AFTER_REPAIR.json')
        if not terminal_path.exists():write(terminal_path,result)

def reconcile(campaign):
    completed=[]
    for packet in (campaign/'packets').glob('v24-n16-*'):
        if (packet/'COMPLETE.json').exists():completed.append((read(packet/'STATUS.json')['at'],packet))
    if completed:advance(campaign,max(completed)[1])
