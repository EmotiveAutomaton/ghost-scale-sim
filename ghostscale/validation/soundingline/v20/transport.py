"""Compatible observed-event outer packet; strict reader/truth separation."""
import hashlib
import numpy as np
from ..v18_3.io import digest
from .world import FEATURES,events,project

FIELDS={'schema','case_id','source_lineage_id','workflow_family','artifact','regions','evidence','observed_events','epistemic_status','downstream_scope','trace_support'}
EVENT_FIELDS={'event_id','order','actor_id','operation','target','parent_event_ids','primary_goal_id','secondary_goal_candidates','constraint_ids','alternatives','perceptual_access','noticed','visible_in_final','ground_truth_source','payload'}
def packet(row,tier,identity,observed=False):
    rendered='Content: '+str(int(row[11]))+'; linked evidence: '+str(int(row[12]))+'; presentation: '+str(int(row[13]))
    p=dict(schema='ghost.v20.observed-outer.1',case_id=digest(['case',identity])[:24],source_lineage_id=digest(['lineage',identity[0]])[:24],workflow_family='bounded-selection-revision',
        artifact=rendered,regions=[{'id':'region-0','start':0,'end':len(rendered),'unit':'UTF-16','exact':rendered}],
        evidence={'tier':tier,'fields':dict(zip(FEATURES,project(np.asarray([row]),tier,observed)[0].tolist())),'source_group':'constructed-observation'},
        observed_events=events(row,observed) if tier=='complete' else [],epistemic_status='constructed observations; aims and values unknown',
        downstream_scope='Sounding Line retrospective contribution and offline export',trace_support='constructed known-answer record; no human correspondence')
    validate(p);return p

def validate(p):
    if set(p)!=FIELDS:raise ValueError('unknown/private field at reader boundary')
    if set(p['evidence'])!={'tier','fields','source_group'} or set(p['evidence']['fields'])!=set(FEATURES):raise ValueError('private evidence field')
    if any(type(x) is not int or x not in (-1,0,1) for x in p['evidence']['fields'].values()):raise ValueError('invalid observation')
    if p['evidence']['tier']!='complete' and p['observed_events']:raise ValueError('events leaked across tier')
    for e in p['observed_events']:
        if set(e)!=EVENT_FIELDS or e['primary_goal_id'] is not None or e['secondary_goal_candidates']:raise ValueError('private aims are not observed events')
        if set(e['payload'])-{'proposal','recorded_scope'}:raise ValueError('private event payload')
    if any(set(r)!={'id','start','end','unit','exact'} for r in p['regions']):raise ValueError('private region field')
    return True

def offline_input(p,probability):
    """Arguments to the actual Sounding Line exporter, not a second exporter."""
    validate(p);text=p['artifact'];revision=hashlib.sha256(text.encode()).hexdigest();region='region-0'
    artifact=dict(id=p['case_id'],revision=revision,regions=[dict(id=region,anchor=dict(exact=text,index=0,start=0,prefix='',suffix=''))])
    from .world import LABELS
    marginal=np.asarray(probability)@LABELS
    visibility='I1' if p['evidence']['tier']=='artifact' else 'I2'
    reading={region:dict(visibility=visibility,goals=[dict(label='Stipulated content aim',support=float(marginal[5])),dict(label='Stipulated presentation aim',support=float(marginal[6]))],
        processes=[dict(label='Tool proposed',probability=float(marginal[0])),dict(label='Human proposed',probability=float(1-marginal[0]))])}
    outer=dict(case_id=p['case_id'],scope=p['trace_support'],observed_events=p['observed_events'],unknown_review_scope=True,governing_purpose='unmeasured; retained as a hypothesis slot')
    return artifact,reading,outer
