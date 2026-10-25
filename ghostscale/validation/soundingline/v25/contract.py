"""Finite separately charged allocation. Dates are UTC across the DST change."""
from datetime import datetime
import math

NEW_STARTS_CLOSE='2026-11-02T04:00:00+00:00'
DRAIN='2026-11-02T06:00:00+00:00'
HEALING_RELEASE='2026-10-31T19:00:00+00:00'
FAMILY_HOURS=dict(W1=4,W2=2,W3=4,W4=3,W5=3,W6=2,W7=3,W8=2,W9=1,W10=2,W11=2,W12=2,W13=2)
POOLS=dict(studies=32*3600,preparation=2*3600,followup_healing=6*3600,protected=6*3600)
ORDER=('W1','W2','W6','W7','W3','W10','W8','W12','W9','W4','W5','W11','W13')
SHED_ORDER=('W13','W11','W5','W4','W9')

def acceptance(opened,specification_sha256):
    stamp=datetime.fromisoformat(opened)
    if stamp.tzinfo is None or stamp>=datetime.fromisoformat(NEW_STARTS_CLOSE):raise ValueError('invalid opening')
    return dict(version='v25-1',opened=opened,new_starts_close=NEW_STARTS_CLOSE,drain=DRAIN,
        original_specification_open='2026-10-26T15:00:00+00:00',
        opening_amendment='Owner October 25: start today; retain original finish and budgets.',
        specification_sha256=specification_sha256,pools_seconds=POOLS,
        family_seconds={k:v*3600 for k,v in FAMILY_HOURS.items()},scientific_workers=2,
        numerical_threads=1,priority='below-normal',gpu=False,environment_synchronization=False,
        healing_per_family=2,followup_generation_limit=1,production_charged_separately=True)

def admission(plan,accepted,current,charges,*,running=0):
    if current.tzinfo is None:raise ValueError('aware clock required')
    if accepted!=acceptance(accepted['opened'],accepted['specification_sha256']):raise ValueError('unknown acceptance')
    if not datetime.fromisoformat(accepted['opened'])<=current<datetime.fromisoformat(NEW_STARTS_CLOSE):raise ValueError('outside new-start window')
    if running>=2:raise ValueError('Gear 2 has two shared scientific slots')
    if plan.get('numerical_threads')!=1 or plan.get('priority')!='below-normal':raise ValueError('resource mismatch')
    family=plan['family'];kind=plan['kind'];pool=plan['pool'];cap=plan['cpu_cap_seconds']
    if family not in FAMILY_HOURS or kind not in ('probe','study','healing','followup'):raise ValueError('unknown scientific packet')
    if not math.isfinite(cap) or cap<=0:raise ValueError('invalid reservation')
    if pool not in ('studies','followup_healing'):raise ValueError('science cannot spend preparation/protected pool')
    if kind in ('probe','study') and pool!='studies':raise ValueError('study pool mismatch')
    if kind=='followup' and (pool!='followup_healing' or plan.get('generation')!=1 or cap>FAMILY_HOURS[family]*1800):raise ValueError('follow-up bound')
    if kind=='probe' and cap>1800:raise ValueError('probe ceiling')
    if kind=='healing' and plan.get('healing_number',0) not in (1,2):raise ValueError('healing limit')
    if charges.get(pool,0)+cap>POOLS[pool]:raise ValueError('pool ceiling')
    if pool=='studies' and charges.get('family:'+family,0)+cap>1.5*FAMILY_HOURS[family]*3600:raise ValueError('family ceiling after shedding')
    if kind=='followup' and current<datetime.fromisoformat(HEALING_RELEASE) and charges.get('followup',0)+cap>3*3600:raise ValueError('healing protection')
    passes=plan['pass_caps']
    if set(passes)!= {'original','replay','review'} or any(not math.isfinite(x) or x<=0 for x in passes.values()) or sum(passes.values())>cap:raise ValueError('invalid measured pass reservations')
    return True
