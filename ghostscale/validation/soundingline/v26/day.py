"""Finite theory-day admission; no Friday admissions or implicit descendants."""
from datetime import datetime, timedelta
import math
from . import contract

REL = 'day-20261106'
ORDER = ('G1', 'G2', 'G3', 'G4', 'G5')
STUDY_HOURS = dict(G1=2.6, G2=.5, G3=.25, G4=1.25, G5=1.)
TEXTURE_HOURS = dict(G1=6., G2=2., G3=.5, G4=1.4, G5=.75)
POOLS = dict(studies=86*3600, preparation=3*3600, followup_healing=15*3600, protected=6*3600)
MODULES = dict(G1='day_velocity', G2='day_identity', G3='day_closure', G4='day_prices', G5='day_security')
QUESTIONS = dict(G1='Can hidden-stock evidence distinguish direction-relative from need-balance-relative triggers, and locate set points?',
    G2='How many records reveal preemptive identity with and without the frame?',
    G3='When does closing a source model protect against a con or analyst-defined contamination?',
    G4='How many well-priced offers distinguish a strong terminal weight, and how many records do offers save?',
    G5='Can observed security choices distinguish an asymmetric set point from a smooth preference?')
ROUTES = dict(
    G1=['Sounding Line: a real-record trigger-law test needs dated context for stocks, losses and contacts, not cues alone.', 'ToMpathy: set-point marks require an explicit stock-observation and trigger model; this miniature does not license a display.'],
    G2=['Sounding Line: retain prompts and first-mentioned considerations beside choices in a reframing test.', 'ToMpathy: test whether identity inference survives unknown frames and frame distributions before using a core-values mark.'],
    G3=['Sounding Line: compare the theory-days examined-rejection test with subsequent uptake of offers.', 'ToMpathy: a closure state needs validated source models; a steady omitted manipulator may never look harmful.'],
    G4=['Sounding Line: signature-commitment refusals need priced alternatives and ordinary choices; this finite ladder cannot establish refusal at every price.', 'ToMpathy: distinguish lexicographic choice rules from terminal weights only under a tested choice-noise model.'],
    G5=['Sounding Line: measure a reserve such as time or finished pages together with choices on both sides of it.', 'ToMpathy: infer defence and surplus slopes rather than assuming them; constructed recovery does not validate real-record marks.'])

def supplement(opened, packet_sha256, parent_sha256, prior):
    return dict(version='v26-day-1', opened=opened, new_starts_close=contract.NEW_STARTS_CLOSE,
        drain=contract.DRAIN, packet_sha256=packet_sha256, parent_charges_sha256=parent_sha256,
        total_cpu_seconds=110*3600, pools_seconds=POOLS,
        studies_seconds={k:v*3600 for k,v in STUDY_HOURS.items()},
        texture_seconds={k:v*3600 for k,v in TEXTURE_HOURS.items()},
        frozen_predecessor_family_charges={k:v for k,v in prior.items() if k.startswith('family:')},
        workers=2, numerical_threads=1, priority='below-normal', gpu=False,
        healing_per_card=1, followups=False)

def admission(plan, accepted, current, charges, running=0, texture_supplement=None):
    if current.tzinfo is None: raise ValueError('aware clock required')
    if accepted['pools_seconds'] != POOLS or accepted['total_cpu_seconds'] != 110*3600: raise ValueError('invalid allocation')
    if accepted['new_starts_close'] != contract.NEW_STARTS_CLOSE or accepted['drain'] != contract.DRAIN: raise ValueError('window changed')
    if not datetime.fromisoformat(accepted['opened']) <= current < datetime.fromisoformat(contract.NEW_STARTS_CLOSE): raise ValueError('outside new-start window')
    wall = plan['projected_wall_seconds']
    if not math.isfinite(wall) or wall <= 0 or current+timedelta(seconds=wall) >= datetime.fromisoformat(contract.DRAIN): raise ValueError('insufficient drain margin')
    if running >= 2: raise ValueError('two shared scientific slots')
    if plan['numerical_threads'] != 1 or plan['priority'] != 'below-normal': raise ValueError('resource mismatch')
    f, kind, pool, cap = (plan[k] for k in ('family','kind','pool','cpu_cap_seconds'))
    if f not in ORDER or kind not in ('probe','study','texture','healing'): raise ValueError('uncommissioned packet')
    texture = plan.get('lane') == 'texture'
    expected = 'followup_healing' if texture or kind == 'healing' else 'studies'
    if pool != expected: raise ValueError('pool mismatch')
    if any(not math.isfinite(v) or v < 0 for v in charges.values()): raise ValueError('invalid charges')
    if any(charges.get(k,0) > v for k,v in POOLS.items()): raise ValueError('existing pool overrun')
    if not math.isfinite(cap) or cap <= 0: raise ValueError('invalid reservation')
    passes = plan['pass_caps']
    if set(passes) != {'original','replay','review'} or any(not math.isfinite(v) or v <= 0 for v in passes.values()) or sum(passes.values()) > cap: raise ValueError('pass reservation')
    if kind == 'probe' and cap > 1800: raise ValueError('probe ceiling')
    if kind == 'healing' and plan.get('healing_number') != 1: raise ValueError('healing limit')
    if charges.get(pool,0)+cap > POOLS[pool] or sum(charges.get(k,0) for k in POOLS)+cap > 110*3600: raise ValueError('pool ceiling')
    texture_ceiling = TEXTURE_HOURS[f]*3600
    if texture_supplement is not None:
        # Owner's November 7 allocation changes only the already-declared G3-T09.
        # The caller also verifies the supplement's immutable file/acceptance bindings.
        s = texture_supplement
        if (s.get('schema') != 'v26-day-g3-t09-allocation-20261107'
                or s.get('card') != 'G3-T09' or plan.get('card') != 'G3-T09'
                or f != 'G3' or not texture or kind not in ('probe','texture')
                or s.get('previous_texture_seconds') != 1800
                or s.get('texture_seconds') != 3600
                or s.get('total_cpu_seconds') != accepted['total_cpu_seconds']
                or s.get('pools_seconds') != POOLS
                or s.get('new_starts_close') != accepted['new_starts_close']
                or s.get('drain') != accepted['drain']
                or not plan.get('resource_supplement_sha256')):
            raise ValueError('invalid G3-T09 resource supplement')
        texture_ceiling = s['texture_seconds']
    if texture and charges.get('texture:'+f,0)+cap > texture_ceiling: raise ValueError('texture ceiling')
    if pool == 'studies':
        projected = charges.get('family:'+f,0)+cap
        if projected > 1.5*STUDY_HOURS[f]*3600: raise ValueError('trigger 1: studies ceiling after shedding')
        if projected > STUDY_HOURS[f]*3600 and not plan.get('shedding_exhausted'): raise ValueError('shedding required')
    if not plan.get('design_check_sha256') or not plan.get('controls_sha256'): raise ValueError('missing design or controls')
    return True

def cards():
    result=[]
    def add(f, cell, settings, texture=False, out=False):
        n=sum(c['family']==f for c in result)
        cid=f if not texture else f+'-T'+str(n).zfill(2)
        result.append(dict(id=cid,family=f,cell=cell,settings=settings,blocks=16,
            lane='texture' if texture else 'primary',declared_out_of_case=out,
            queue_order=(5 if texture else 0)+ORDER.index(f)*100+n if texture else ORDER.index(f),
            question=QUESTIONS[f],routes=ROUTES[f],seed_domain='v26-day-20261106-'+cid.lower(),
            case_check_seed_domain='v26-day-excluded-'+cid.lower(),
            interpretation='Constructed method; miniature, architecture untested; no human inference or confirmation.'))
    g1=dict(records=512,makers=8,menu=.35,restore=5,urgency_floor=.1,urgency_power=1.,laws=['a','b','c','d'])
    def velocity(s):return [s|dict(truth=l) for l in s['laws']]
    add('G1','registered',velocity(g1))
    g2=dict(records=512,makers=8,specialists=.25,cue=.15,options=4,pi=None)
    def identity(s):return [s|dict(truth=t) for t in ('A','L','Ps' if s['pi'] is not None else 'P')]
    add('G2','registered',identity(g2))
    g3=dict(records=64,worlds_per_block=4,sources=32,rho=.05,switch=16,overpush=1.,expanded=False,pull_away=False)
    def closure(s):return [s|dict(theta=t,delta=d) for t in (.7,.5,.9) for d in (0.,.01,.02,.05,.1)]
    add('G3','registered',closure(g3))
    g4=dict(records=512,makers=4,rho=1/16,specialists=.5,ladder=[1,2,4,8],precision=3.,tau_known=False)
    def prices(s,all_types=False):return [s|dict(truth=t) for t in (['A75','A9','A1','C'] if all_types else ['A9','A1','C'])]
    add('G4','registered',prices(g4,True)+prices(g4|dict(rho=.25))+prices(g4|dict(rho=1/32))+prices(g4|dict(tau_known=True)))
    g5=dict(records=256,makers=4,precision=3.,prices=[.25,16.],offgrid=False)
    def security(s):return [s|dict(truth=t) for t in ('SP','LV','HO')]
    offgrid=security(g5|dict(offgrid=True));offgrid[0]['makers']=60
    add('G5','registered',security(g5)+offgrid)
    for name,change,out in [('menu-quarter',dict(menu=.25),False),('urgency-squared',dict(urgency_power=2.),False),('drive-law',dict(laws=['a','b','c','d','k']),False),('menu-half',dict(menu=.5),True)]:
        add('G1',name,velocity(g1|dict(makers=4)|change),True,out)
    for name,change,out in [('specialists-half',dict(specialists=.5),False),('cue-quarter',dict(cue=.25),False),('five-options',dict(options=5),False),('all-specialists',dict(specialists=1.),True),('cue-002',dict(cue=.02),True),('cue-005',dict(cue=.05),True),('soft-05',dict(pi=.5),True),('soft-1',dict(pi=1.),True),('soft-2',dict(pi=2.),True)]:
        add('G2',name,identity(g2|dict(makers=4)|change),True,out)
    for name,change,out in [('unfittable-001',dict(rho=.01),False),('unfittable-02',dict(rho=.2),False),('switch-8',dict(switch=8),False),('switch-32',dict(switch=32),False),('overpush-075',dict(overpush=.75),False),('claims-128',dict(records=128),False),('pull-away',dict(pull_away=True),False),('claims-16',dict(records=16),True),('expanded-library',dict(expanded=True),True)]:
        add('G3',name,closure(g3|change),True,out)
    for name,change in [('specialists-075',dict(specialists=.75)),('ladder-16',dict(ladder=[1,2,4,8,16])),('records-1024',dict(records=1024)),('precision-6',dict(precision=6.))]:
        add('G4',name,prices(g4|change),True)
    for name,change in [('precision-15',dict(precision=1.5)),('precision-1',dict(precision=1.)),('records-512',dict(records=512)),('prices-narrow',dict(prices=[.5,8.]))]:
        add('G5',name,security(g5|change),True)
    return result
