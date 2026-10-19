"""Separate finite V24 pilot allocation; preparation cannot open production."""
from datetime import datetime
import math

OPEN='2026-10-22T00:00:00-07:00'
CLOSE='2026-10-25T22:00:00-07:00'
PRODUCTION_OPEN='2026-10-26T08:00:00-07:00'
TOTAL_SECONDS=40*3600
VERIFICATION_RESERVE=8*3600
FAMILY_SECONDS={
    'N16':6*3600,'D5':4*3600,'T1':3*3600,'T3':2*3600,'T4':2*3600,
    'N1':3*3600,'PYMDP':3600,'NEEDS_DRIVES':3600,'SETPOINTS':3600,
    'PLAY_UNCERTAINTY':3600,'HIGHER_GOAL_EDITING':3600,
    'CONTEXT_CONVERGENCE':3600,'DISTURBANCE':3600,'PROMOTIONS':3*3600}


def acceptance():
    return dict(version='v24-pilot-1',pilot_open=OPEN,pilot_close=CLOSE,
        production_not_before=PRODUCTION_OPEN,total_cpu_seconds=TOTAL_SECONDS,
        verification_reserve_seconds=VERIFICATION_RESERVE,
        preparation_seconds=2*3600,family_seconds=FAMILY_SECONDS,
        scientific_workers=2,numerical_threads=1,priority='below-normal',
        production_authorized_by_this_contract=False)


def admission(plan,current,charged_seconds,reserved_seconds=0,family_charged_seconds=0):
    if current.tzinfo is None:raise ValueError('timezone-aware clock required')
    if not datetime.fromisoformat(OPEN)<=current<datetime.fromisoformat(CLOSE):
        raise ValueError('outside the authorized V24 pilot window')
    if plan.get('kind')!='pilot' or plan.get('family') not in FAMILY_SECONDS:
        raise ValueError('this allocation admits pilots only')
    amounts=(charged_seconds,reserved_seconds,family_charged_seconds,plan.get('cpu_cap_seconds',0))
    if any(not math.isfinite(x) or x<0 for x in amounts) or amounts[-1]<=0:
        raise ValueError('invalid accounting')
    if plan.get('numerical_threads')!=1 or plan.get('priority')!='below-normal':
        raise ValueError('pilot resource contract mismatch')
    if charged_seconds+reserved_seconds+amounts[-1]>TOTAL_SECONDS-VERIFICATION_RESERVE:
        raise ValueError('protected verification reserve would be spent')
    if family_charged_seconds+amounts[-1]>FAMILY_SECONDS[plan['family']]:
        raise ValueError('family pilot cap would be spent')
    return True
