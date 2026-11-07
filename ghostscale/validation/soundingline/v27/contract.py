"""V27 authority, measured completed costs, four slots and independent pools.

DESIGN CHECK: allocations and kinds come from packet 0.1-0.5. Refusal fixtures
cover changed authority, exhausted own pool, invalid descendants and late starts;
an overrun in an unrelated pool must not block otherwise eligible work.
"""
from datetime import datetime,timedelta
import math

NEW_STARTS_CLOSE='2026-11-16T04:00:00+00:00'
DRAIN='2026-11-16T06:00:00+00:00'
HEALING_RELEASE='2026-11-11T20:00:00+00:00'
ORDER=('K1','K3','K2','K5','K4','K6','K7')
FAMILY_HOURS=dict(K1=40,K2=40,K3=50,K4=30,K5=30,K6=30,K7=20)
POOLS=dict(studies=240*3600,preparation=20*3600,followup_healing=480*3600,protected=20*3600)
KINDS=('probe','study','texture','healing','followup','precision','reread')


def acceptance(opened,packet_sha256):
    t=datetime.fromisoformat(opened)
    if t.tzinfo is None or not t<datetime.fromisoformat(NEW_STARTS_CLOSE):raise ValueError('invalid opening')
    return dict(version='v27-1',opened=opened,new_starts_close=NEW_STARTS_CLOSE,drain=DRAIN,
        packet_sha256=packet_sha256,total_cpu_seconds=sum(POOLS.values()),pools_seconds=POOLS,
        family_seconds={k:v*3600 for k,v in FAMILY_HOURS.items()},scientific_workers=4,
        numerical_threads=1,priority='below-normal',gpu=False,environment_synchronization=False,
        followup_generations=2,reread_levels=2,texture_stop_consecutive=7,
        healing_per_family=2,healing_protection_seconds=10*3600,healing_release=HEALING_RELEASE)


def accounted(row):
    actual=float(row.get('cpu_seconds',0));reserved=float(row.get('reserved_cpu_seconds',0))
    if row.get('pool')=='followup_healing' and row.get('state') in ('COMPLETE','FAILED','STOPPED'):
        return actual
    return max(actual,reserved)


def admission(p,a,now,charges,running=0,worker_limit=4):
    if a!=acceptance(a['opened'],a['packet_sha256']):raise ValueError('acceptance changed')
    if now.tzinfo is None or not datetime.fromisoformat(a['opened'])<=now<datetime.fromisoformat(NEW_STARTS_CLOSE):raise ValueError('outside start window')
    if worker_limit not in (2,4) or running>=worker_limit:raise ValueError('shared worker slots')
    if p.get('numerical_threads')!=1 or p.get('priority')!='below-normal':raise ValueError('resource mismatch')
    kind=p['kind'];f=p['family'];pool=p['pool'];cap=p['cpu_cap_seconds'];wall=p['projected_wall_seconds']
    if f not in ORDER or kind not in KINDS:raise ValueError('uncommissioned kind or family')
    if not all(math.isfinite(v) and v>=0 for v in charges.values()):raise ValueError('invalid charges')
    if not math.isfinite(cap) or cap<=0:raise ValueError('invalid reservation')
    if not math.isfinite(wall) or not 0<wall<=12*3600:raise ValueError('split above twelve-hour projection')
    if now+timedelta(seconds=wall)>=datetime.fromisoformat(DRAIN):raise ValueError('drain margin')
    lane=p.get('lane',kind)
    expected='studies' if lane=='study' else 'followup_healing'
    if pool!=expected:raise ValueError('pool mismatch')
    # A logged completed overrun is local to its pool (amendment I).
    if charges.get(pool,0)+cap>POOLS[pool]:raise ValueError('own pool ceiling')
    if pool=='studies':
        total=charges.get('family:'+f,0)+cap
        if total>1.5*FAMILY_HOURS[f]*3600:raise ValueError('trigger 1: family ceiling after shedding')
        if total>FAMILY_HOURS[f]*3600 and not p.get('shedding_exhausted'):raise ValueError('shed before family overrun')
    if kind=='probe' and cap>1800:raise ValueError('probe ceiling')
    if kind=='healing' and p.get('healing_number') not in (1,2):raise ValueError('healing limit')
    if lane=='followup':
        if p.get('generation') not in (1,2):raise ValueError('followup generation')
        if p['generation']==2 and not p.get('primary_positive_or_sign_reversal'):raise ValueError('generation two primary condition')
        if cap>FAMILY_HOURS[f]*1800:raise ValueError('followup per-card ceiling')
    if lane=='reread' and (p.get('reread_level') not in (1,2) or not p.get('retained_inputs')):raise ValueError('reread binding or level')
    if lane=='precision' and (not p.get('parent_complete_sha256') or (kind!='probe' and p['blocks']!=2*p['parent_blocks'])):raise ValueError('precision doubles blocks')
    if pool=='followup_healing' and kind!='healing' and now<datetime.fromisoformat(HEALING_RELEASE):
        protected=max(0,10*3600-charges.get('healing',0))
        if charges.get(pool,0)+cap>POOLS[pool]-protected:raise ValueError('protected healing share')
    passes=p['pass_caps']
    if set(passes)!= {'original','replay','review'} or any(not math.isfinite(v) or v<=0 for v in passes.values()) or sum(passes.values())>cap:raise ValueError('pass reservation')
    for field in ('design_check_sha256','controls_sha256','protocol_sha256','source_manifest_sha256'):
        if not p.get(field):raise ValueError('missing admission evidence')
    return True


def load_limit(samples,previous=4):
    """Timestamped host fractions; require a full uninterrupted ten minutes."""
    if len(samples)<2:return previous
    end=samples[-1][0];before=[s for s in samples if s[0]<=end-600]
    if not before:return previous
    window=[before[-1]]+[s for s in samples if end-600<s[0]<=end]
    if any(b[0]-a[0]>60 for a,b in zip(window,window[1:])):return previous
    if previous==4 and all(v>.90 for _,v in window):return 2
    if previous==2 and all(v<.80 for _,v in window):return 4
    return previous
