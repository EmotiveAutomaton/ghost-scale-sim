"""Exactly enumerable three-unit dependency editing with goals chosen before acts.

The full process state is (maker, artifact, undo buffer, step, selected local goal).
Persistent maker states alone are NOT a Markov representation of the process.
"""
from collections import defaultdict
from itertools import product
import gzip
import numpy as np
from ..v18_3.io import canonical, digest, write
from ..v18_3.world import rng

MAKERS=tuple(product(range(2),repeat=4))  # governing purpose, skill, belief error, routine
GOALS=('meaning','dependency','presentation')
CONTEXTS=tuple(dict(initial=list(a),requested_purpose=b) for a in ((0,1,0),(1,0,1)) for b in (0,1))
OPERATIONS=('edit-claim','repair-evidence','replace-presentation','accept-tool','inspect','undo')
TIERS=('E0','E1','E2-sparse','E2-full','E1-corrected')


def law(lineage):
    random=rng('v19-local-world',lineage)
    return dict(lineage=lineage,goal_strength=float(random.uniform(.8,1.2)),
        action_rate=float(random.uniform(.65,.85)),routine_strength=float(random.uniform(.1,.3)))


def execute(artifact,previous,operation,maker):
    purpose,skill,belief,routine=maker
    a=list(artifact)
    if operation=='edit-claim':a[0]=1-a[0]
    elif operation=='repair-evidence':a[1]=(a[0]^belief) if skill else 1-a[1]
    elif operation=='replace-presentation':a[2]=1-a[2]
    elif operation=='accept-tool':
        if not skill:raise ValueError('unreachable tool use')
        a[0]=1-a[0];a[1]=a[0]  # tool proposal flips claim and repairs its dependent evidence
    elif operation=='undo':a=list(previous)
    elif operation!='inspect':raise ValueError('unknown operation')
    return tuple(a)


def choices(world,maker,artifact,step,context):
    purpose,skill,belief,routine=maker
    perceived=artifact[0]^belief
    strengths=np.array([1.5 if purpose==0 else .6,1.4 if perceived!=artifact[1] else .5,1.8 if purpose==1 else .5])
    strengths[2 if routine else 0]+=world['routine_strength']
    strengths[2 if context['requested_purpose'] else 0]+=.2  # a request need not be adopted
    strengths=np.power(strengths,world['goal_strength']);strengths/=strengths.sum()
    for goal,weight in zip(GOALS,strengths):
        action={'meaning':'accept-tool' if skill else 'edit-claim','dependency':'repair-evidence','presentation':'replace-presentation'}[goal]
        alternative='undo' if goal=='dependency' and step==2 else 'inspect'
        yield goal,action,float(weight*world['action_rate'])
        yield goal,alternative,float(weight*(1-world['action_rate']))


def enumerate_world(world):
    records=[]
    for context_index,context in enumerate(CONTEXTS):
        for maker_index,maker in enumerate(MAKERS):
            def visit(artifact,previous,steps,probability):
                if len(steps)==3:
                    records.append(dict(context_index=context_index,maker_index=maker_index,maker=list(maker),
                        initial=context['initial'],requested_purpose=context['requested_purpose'],
                        final=list(artifact),steps=steps,probability=probability/len(MAKERS)/len(CONTEXTS)))
                    return
                step=len(steps)
                for goal,operation,weight in choices(world,maker,artifact,step,context):
                    after=execute(artifact,previous,operation,maker)
                    event=dict(step=step,goal=goal,operation=operation,before=list(artifact),after=list(after),undo_buffer=list(previous),
                        dependency_edges=[[0,1]],tool_proposal=list(after) if operation=='accept-tool' else None,
                        perceived_claim=artifact[0]^maker[2])
                    visit(after,artifact,steps+[event],probability*weight)
            visit(tuple(context['initial']),tuple(context['initial']),[],1.)
    return records


def project(record,tier):
    if tier not in TIERS:raise ValueError('unknown evidence tier')
    inputs=dict(artifact=list(record['final']))
    if tier!='E0':
        inputs.update(initial=record['initial'],requested_purpose=record['requested_purpose'])
    if tier in ('E2-sparse','E2-full'):
        steps=record['steps'][:1] if tier=='E2-sparse' else record['steps']
        inputs['observations']=[{k:event[k] for k in ('step','operation','before','after','tool_proposal')} for event in steps]
    if tier=='E1-corrected':
        inputs['reports']=[dict(status='retracted',requested_purpose=1-record['requested_purpose']),
            dict(status='correction',requested_purpose=record['requested_purpose'])]
    return dict(schema='v19.local.public.1',tier=tier,inputs=inputs)


def validate_public(packet):
    if set(packet)!={'schema','tier','inputs'} or packet['schema']!='v19.local.public.1':raise ValueError('private top-level field')
    tier=packet['tier']
    allowed={'artifact'}
    if tier!='E0':allowed|={'initial','requested_purpose'}
    if tier in ('E2-sparse','E2-full'):allowed.add('observations')
    if tier=='E1-corrected':allowed.add('reports')
    if tier not in TIERS or set(packet['inputs'])!=allowed:raise ValueError('private or missing input field')
    for event in packet['inputs'].get('observations',[]):
        if set(event)!={'step','operation','before','after','tool_proposal'}:raise ValueError('private observation field')
    return True


def infer(packet,records,weighted=True):
    validate_public(packet)
    matched=[r for r in records if project(r,packet['tier'])==packet]
    if not matched:return dict(candidate_count=0,unknown=True,goal_support=[],process_support=[],governing_support=[],normalizer=0.)
    weights=np.array([r['probability'] if weighted else 1. for r in matched]);normalizer=float(weights.sum());weights/=normalizer
    goals=defaultdict(float);processes=defaultdict(float);governing=defaultdict(float);joint=defaultdict(float)
    for r,p in zip(matched,weights):
        for event in r['steps']:goals[(event['step'],event['goal'])]+=float(p)
        processes[tuple(event['operation'] for event in r['steps'])]+=float(p)
        governing[r['maker'][0]]+=float(p)
        joint[(r['maker'][0],tuple(e['goal'] for e in r['steps']),tuple(e['operation'] for e in r['steps']))]+=float(p)
    return dict(candidate_count=len(matched),unknown=False,normalizer=normalizer,
        goal_support=[dict(step=k[0],goal=k[1],passage_anchor='unit-'+str(GOALS.index(k[1])),support=v) for k,v in sorted(goals.items())],
        process_support=[dict(operations=list(k),support=v) for k,v in sorted(processes.items())],
        governing_support=[dict(purpose=k,support=v) for k,v in sorted(governing.items())],
        complete_hypotheses=[dict(compatibility_id=digest(k),governing_purpose=k[0],local_goals=list(k[1]),operations=list(k[2]),support=v) for k,v in sorted(joint.items())],
        support_kind='exact conditional on supplied finite generator' if weighted else 'uniform compatible complete templates; uncalibrated',
        access='oracle-family reference' if weighted else 'legal finite template baseline',
        human_values='unknown',endorsement='unknown')


def controls():
    maker=(0,1,0,0)
    repaired=execute((1,0,0),(0,0,0),'repair-evidence',maker)
    undone=execute((1,1,0),(0,1,0),'undo',maker)
    options=list(choices(law(0),maker,(0,1,0),0,CONTEXTS[0]))
    forbidden=False
    try:execute((0,1,0),(0,1,0),'accept-tool',(0,0,0,0))
    except ValueError:forbidden=True
    return dict(live_dependency_repair=repaired==(1,1,0),positive_undo=undone==(0,1,0),
        placebo_same_operation_different_goals=len({g for g,op,p in options if op=='inspect'})==3,
        normalized_choice=abs(sum(p for g,op,p in options)-1)<1e-12,skill_changes_reachability=forbidden)


def admission(root,plan,pulse):
    raw=[];public=[];evaluators=[];reference=[]
    for lineage in plan['design']['lineages']:
        pulse(phase='local-enumeration',lineage=lineage)
        world=law(lineage);records=enumerate_world(world)
        mass=sum(r['probability'] for r in records)
        if len(records)!=13824 or abs(mass-1)>1e-10:raise ValueError('enumeration denominator or mass failed')
        # Predetermined all-inspect path: genuine endpoint alias with different selected local goals.
        case=next(r for r in records if r['maker_index']==0 and r['context_index']==0 and all(e['operation']=='inspect' for e in r['steps']))
        examples=[]
        for tier in TIERS:
            pulse(phase='exact-reader',lineage=lineage,tier=tier)
            packet=project(case,tier);validate_public(packet)
            exact=infer(packet,records);baseline=infer(packet,records,False)
            actual=[e['operation'] for e in case['steps']]
            if not any(p['operations']==actual for p in exact['process_support']):raise ValueError('true process not covered')
            ident=f'{lineage}-{tier}'
            public.append(dict(case_id=ident,packet=packet,input_sha256=digest(packet),
                passages=[dict(id=f'unit-{i}',rendering=(('claim-off','claim-on'),('evidence-off','evidence-on'),('plain','display'))[i][v]) for i,v in enumerate(case['final'])],
                dependency_edges=[[0,1]],request_status='observed request; adoption unknown'))
            reference.append(dict(case_id=ident,exact=exact,template_baseline=baseline,
                metrics=dict(candidate_coverage=True,contradictions=0,unsupported_assertions=0,
                    explanation_policy='set-valued complete candidates; no single historical process asserted')))
            evaluators.append(dict(case_id=ident,record=case,world=world))
            examples.append(dict(tier=tier,candidates=exact['candidate_count'],process_alternatives=len(exact['process_support'])))
        # All combinations of three visible bits are representable, but illegal renderings are outside-family.
        unknown=dict(schema='v19.local.public.1',tier='E0',inputs=dict(artifact=[2,0,0]))
        if not infer(unknown,records)['unknown']:raise ValueError('outside-family absence not recognized')
        # Keep full raw trajectories, including unobserved goals, in evaluator evidence only.
        path=root/'raw'/f'lineage-{lineage}_points.json.gz';path.parent.mkdir(exist_ok=True)
        payload=gzip.compress(canonical(records),mtime=0)
        if path.exists() and path.read_bytes()!=payload:raise ValueError('immutable local trajectory block differs')
        if not path.exists():path.write_bytes(payload)
        raw.append(dict(lineage=lineage,trajectories=len(records),mass=mass,world=world,
            trajectory_sha256=__import__('hashlib').sha256(payload).hexdigest(),examples=examples))
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.local.export.1',cases=public))
    write(root/'EVALUATOR_ONLY.json',dict(schema='v19.local.evaluator.1',cases=evaluators))
    write(root/'REFERENCE_OUTPUTS.json',dict(schema='v19.local.references.1',cases=reference))
    write(root/'ENUMERATION.json',dict(lineages=raw))
    return dict(lineages=len(raw),persistent_maker_states=16,contexts=4,steps=3,trajectories_per_lineage=13824,
        total_trajectories=sum(r['trajectories'] for r in raw),evidence_cases=len(public),controls=controls(),
        full_state=['maker','artifact','undo buffer','step','selected local goal'],
        pursuit='exact local-goal and process apparatus ready for learned access and jointness tests',
        warrant='exact finite constructed reference; no learned-reader result; miniature — architecture untested')
