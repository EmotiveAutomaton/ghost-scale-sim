"""Immutable V26 window, pool reservations and two-slot admission."""
from datetime import datetime,timedelta
import math

OPEN = '2026-11-02T06:00:00+00:00'
SCIENCE_OPEN = '2026-11-02T16:00:00+00:00'
NEW_STARTS_CLOSE = '2026-11-09T04:00:00+00:00'
DRAIN = '2026-11-09T06:00:00+00:00'
HEALING_RELEASE = '2026-11-05T20:00:00+00:00'
ORDER = ('H1','H2','H3','E1','E2','E3','U1','R1','R4','R13')
FAMILY_HOURS = dict(H1=6,H2=4,H3=4,E1=6,E2=4,E3=3,U1=3,R1=3,R4=1.5,R13=1.5)
POOLS = dict(studies=36*3600,preparation=3*3600,followup_healing=15*3600,protected=6*3600)
SHED_ORDER = ('R13','U1','E3','R4')

def acceptance(opened,specification_sha256):
    t=datetime.fromisoformat(opened)
    if t.tzinfo is None or not datetime.fromisoformat(OPEN)<=t<datetime.fromisoformat(NEW_STARTS_CLOSE):
        raise ValueError('invalid immutable opening')
    return dict(version='v26-1',opened=opened,science_open=SCIENCE_OPEN,new_starts_close=NEW_STARTS_CLOSE,
        drain=DRAIN,specification_sha256=specification_sha256,pools_seconds=POOLS,
        family_seconds={k:v*3600 for k,v in FAMILY_HOURS.items()},scientific_workers=2,
        numerical_threads=1,priority='below-normal',gpu=False,environment_synchronization=False,
        healing_per_family=2,followup_generation_limit=1,texture_stop_consecutive=5,
        n16_charged_separately=True)

def admission(plan,accepted,current,charges,*,running=0):
    if current.tzinfo is None:raise ValueError('aware clock required')
    if accepted!=acceptance(accepted['opened'],accepted['specification_sha256']):raise ValueError('unknown acceptance')
    if any(charges.get(k,0)>v for k,v in POOLS.items()):raise ValueError('existing pool overrun')
    if not max(datetime.fromisoformat(accepted['opened']),datetime.fromisoformat(SCIENCE_OPEN))<=current<datetime.fromisoformat(NEW_STARTS_CLOSE):raise ValueError('outside scientific start window')
    if current+timedelta(seconds=plan.get('projected_wall_seconds',3600))>=datetime.fromisoformat(DRAIN):raise ValueError('insufficient drain margin')
    if running>=2:raise ValueError('two shared scientific slots')
    if plan.get('numerical_threads')!=1 or plan.get('priority')!='below-normal':raise ValueError('resource mismatch')
    family,kind,pool,cap=(plan[k] for k in ('family','kind','pool','cpu_cap_seconds'))
    if family not in ORDER or kind not in ('probe','study','healing','followup','texture'):raise ValueError('unknown packet')
    if not math.isfinite(cap) or cap<=0:raise ValueError('invalid reservation')
    expected='studies' if kind in ('probe','study') else 'followup_healing'
    if pool!=expected:raise ValueError('pool mismatch')
    if charges.get(pool,0)+cap>POOLS[pool]:raise ValueError('pool ceiling')
    if sum(charges.get(k,0) for k in POOLS)+cap>sum(POOLS.values()):raise ValueError('total ceiling')
    if pool=='studies' and charges.get('family:'+family,0)+cap>1.5*FAMILY_HOURS[family]*3600:raise ValueError('family ceiling after shedding; trigger 1')
    if family in ('R4','R13') and pool=='studies' and sum(charges.get('family:'+f,0) for f in ('R4','R13'))+cap>3*3600:raise ValueError('shared repair ceiling')
    if kind=='probe' and cap>1800:raise ValueError('probe ceiling')
    if kind=='healing' and plan.get('healing_number') not in (1,2):raise ValueError('healing limit')
    if kind=='followup' and (plan.get('generation')!=1 or cap>FAMILY_HOURS[family]*1800):raise ValueError('follow-up bound')
    if current<datetime.fromisoformat(HEALING_RELEASE):
        if kind=='followup' and charges.get('followup',0)+cap>3*3600:raise ValueError('follow-up subtotal')
        if kind in ('followup','texture') and charges.get(pool,0)+cap>POOLS[pool]-max(0,4*3600-charges.get('healing',0)):raise ValueError('protected healing share')
    passes=plan['pass_caps']
    if set(passes)!= {'original','replay','review'} or any(not math.isfinite(x) or x<=0 for x in passes.values()) or sum(passes.values())>cap+1e-8:raise ValueError('pass reservation')
    if not plan.get('design_check_sha256'):raise ValueError('missing pre-mortem')
    return True
