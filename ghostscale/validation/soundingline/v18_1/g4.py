"""Outcome-blind, finite-hypothesis maker-history benchmark and same-evidence rivals.

The known method family is supplied, not inferred. Neither task goals nor stable
priorities are latent scores. Actual alternatives and execution truth are retained
only in the evaluator packet; the consumer accepts one serialized public tier.
"""
from collections import Counter
from copy import deepcopy
import json
import random

from ..v16.records import canonical,digest,seed_for
from .common import execute,identity

SCHEMA='v18.1.process-reader.1'
STRATEGIES=('routine','primitive','episode','adaptation')
TIERS=('endpoint','context','process')
FIELDS={'schema','case_id','tier','world','initial','endpoint','roles','method_family',
        'evidence','context_channel','allowed_queries','questions','scope'}


def programs(family):
    """Public procedural semantics; actual choice is never an input."""
    if family['kind']=='assembly':
        return [[4,5,6,1,2,9],[5,4,6,1,2,9],[7,4,5,6,1,2,9],[8,7,5,4,6,1,2,9]]
    a,b,c,d=family['cells']
    return [[a,b,c],[c,b,a],[b,a,c],[a,d,16+d,b,c]]


def actor(action,roles):
    owners=[r['actor'] for r in roles if action in r['available_actions']]
    if len(owners)!=1:raise ValueError('each physical action needs one available actor')
    return owners[0]


def enact(world,initial,program,roles):
    # Execution requires the named actor to possess each operator; dropping an
    # actor removes those operations and invalidates histories that require them.
    assignments=[actor(a,roles) for a in program]
    result=execute(world,initial,program,12)
    if not result['legal'] or not result['stopped']:raise ValueError('maker history not executable')
    trace=[dict(t,actor=owner,index=i) for i,(t,owner) in enumerate(zip(result['trace'],assignments))]
    return dict(program=program,trace=trace,endpoint=result['state'])


def relationship(program,source):
    longest=max((length for i in range(len(program)) for j in range(len(source))
                 for length in range(1,min(len(program)-i,len(source)-j)+1)
                 if program[i:i+length]==source[j:j+length]),default=0)
    return dict(exact_source_sequence=program==source,
                source_contiguous_fraction=longest/len(source))


def make_packets(namespace,replicates=3):
    """Strata and prevalence fixed before scoring: 72 cases at default allocation."""
    pending=[]
    for kind in ('assembly','graphic'):
        for condition in ('positive','misleading','ambiguous'):
            for replicate in range(replicates):
                rng=random.Random(seed_for(namespace,kind,condition,replicate,'world'))
                if kind=='assembly':
                    defaults=[rng.randrange(2) for _ in range(3)]
                    world=dict(kind=kind,parents=[-1,0,0],defaults=defaults,forbidden=[])
                    initial=defaults.copy();family=dict(kind=kind,cells=[]);action_count=10
                else:
                    world=dict(kind=kind,cells=16,forbidden=[]);initial=0
                    family=dict(kind=kind,cells=rng.sample(range(16),4));action_count=32
                roles=[dict(actor=name,available_actions=list(range(parity,action_count,2)))
                       for parity,name in enumerate(('tool-A','tool-B'))]
                alternatives=[enact(world,initial,p,roles) for p in programs(family)]
                assert len({str(identity(h['endpoint'])) for h in alternatives})==1
                unit=digest([world,initial,family,condition])
                sampling_cluster=digest([namespace,world,initial,family,condition,replicate])
                for selected,history in enumerate(alternatives):
                    earlier=(history['endpoint'] if condition=='ambiguous' else
                             alternatives[(selected+(condition=='misleading'))%4]['trace'][0]['after'])
                    common_action=(6 if kind=='assembly' else family['cells'][0])
                    process=([dict(index=None,action=common_action,actor=actor(common_action,roles))]
                             if condition=='ambiguous' else
                             [{k:t[k] for k in ('index','action','actor')} for t in history['trace'][:2]])
                    pending.append(dict(structural_unit=unit,sampling_cluster=sampling_cluster,history=selected,n=3 if kind=='assembly' else 16,
                        family=kind,condition=condition,public_base=dict(schema=SCHEMA,world=world,initial=initial,
                        endpoint=history['endpoint'],roles=roles,method_family=family,
                        context_channel=('uninformative endpoint artifact' if condition=='ambiguous' else
                                         'equal mixture of own first artifact and next method first artifact'),
                        allowed_queries=['first-two-indexed-actions'] if condition!='ambiguous' else ['one-unindexed-common-action'],
                        questions=['production_strategy','first_actor','first_operation','source_sequence_correspondence'],
                        scope='constructed known four-method family; supplied task endpoint, no latent goal/value/priority score'),
                        evidence=dict(earlier_artifact=earlier,process=process),
                        private=dict(strategy=STRATEGIES[selected],selected=selected,actual=history,alternatives=alternatives,
                                     relationship=relationship(history['program'],alternatives[0]['program']),sampling_condition=condition)))
    # Opaque IDs and ordering do not encode strategy or condition.
    random.Random(seed_for(namespace,'packet-order')).shuffle(pending)
    reader=[];evaluator=[]
    for i,case in enumerate(pending):
        case_id=digest([namespace,'opaque',i]);tiers=[]
        for tier in TIERS:
            evidence={}
            if tier!='endpoint':evidence['earlier_artifact']=case['evidence']['earlier_artifact']
            if tier=='process':evidence['process']=case['evidence']['process']
            tiers.append(json.loads(canonical(dict(case['public_base'],case_id=case_id,tier=tier,evidence=evidence))))
        reader.append(dict(case_id=case_id,tiers=tiers))
        evaluator.append(dict(case_id=case_id,structural_unit=case['structural_unit'],history=case['history'],
            sampling_cluster=case['sampling_cluster'],
            n=case['n'],family=case['family'],condition=case['condition'],private=case['private']))
    return reader,evaluator


def validate_public(payload):
    p=json.loads(payload)
    if set(p)!=FIELDS or p['schema']!=SCHEMA or p['tier'] not in TIERS:raise ValueError('process reader schema')
    if set(p['method_family'])!={'kind','cells'}:raise ValueError('method semantics schema')
    world_fields={'kind','parents','defaults','forbidden'} if p['world']['kind']=='assembly' else {'kind','cells','forbidden'}
    if set(p['world'])!=world_fields or any(set(r)!={'actor','available_actions'} for r in p['roles']):
        raise ValueError('world/actor schema')
    expected=({'earlier_artifact','process'} if p['tier']=='process' else
              {'earlier_artifact'} if p['tier']=='context' else set())
    if set(p['evidence'])!=expected or any(set(o)!={'index','action','actor'} for o in p['evidence'].get('process',[])):
        raise ValueError('tier evidence schema')
    return p


def compile_features(p):
    histories=[enact(p['world'],p['initial'],program,p['roles']) for program in programs(p['method_family'])]
    table=[]
    for i,h in enumerate(histories):
        table.append(dict(strategy=STRATEGIES[i],endpoint=h['endpoint'],first_artifact=h['trace'][0]['after'],
                          sequence=[{k:t[k] for k in ('index','action','actor')} for t in h['trace']],
                          relationship=relationship(h['program'],histories[0]['program'])))
    return table,sum(len(h['program']) for h in histories)


def likelihood(p,table,index):
    feature=table[index]
    if identity(feature['endpoint'])!=identity(p['endpoint']):return 0.,1
    weight=1.;cost=1
    if 'earlier_artifact' in p['evidence']:
        evidence=identity(p['evidence']['earlier_artifact']);cost+=2
        if p['context_channel']=='uninformative endpoint artifact':
            weight*=float(evidence==identity(feature['endpoint']))
        elif p['context_channel']=='equal mixture of own first artifact and next method first artifact':
            weight*=0.5*(evidence==identity(feature['first_artifact']))+0.5*(evidence==identity(table[(index+1)%4]['first_artifact']))
        else:raise ValueError('unknown context channel')
    for observation in p['evidence'].get('process',[]):
        choices=feature['sequence'] if observation['index'] is None else feature['sequence'][observation['index']:observation['index']+1]
        cost+=len(choices)
        if not any(t['action']==observation['action'] and t['actor']==observation['actor'] for t in choices):weight=0.
    return weight,cost


def predict(payload,method='inverse-history'):
    p=validate_public(payload)
    if method not in ('inverse-history','direct-template','finite-history-ceiling'):raise ValueError('unknown history reader')
    table,build=compile_features(p)
    weights=[];cost=0
    for i in range(4):
        weight,work=likelihood(p,table,i);weights.append(weight);cost+=work
    total=sum(weights)
    posterior=[w/total for w in weights] if total else [0.25]*4
    best=max(range(4),key=lambda i:posterior[i]);actors=Counter();operations=Counter()
    for probability,feature in zip(posterior,table):
        actors[feature['sequence'][0]['actor']]+=probability
        operations[str(feature['sequence'][0]['action'])]+=probability
    # Direct template and inverse reconstruction have the same finite likelihood.
    # The direct table's build cost is explicit, not a free hidden training oracle.
    return dict(method=method,posterior=posterior,prediction=STRATEGIES[best],compatible=[STRATEGIES[i] for i,w in enumerate(weights) if w],
        abstained=not total or max(posterior)<=0.5,first_actor_probabilities=dict(actors),first_operation_probabilities=dict(operations),
        predicted_source_contiguous_fraction=sum(p*f['relationship']['source_contiguous_fraction'] for p,f in zip(posterior,table)),
        costs=dict(physical_model_build=build,feature_comparisons=cost,total_per_case=build+cost,
                   online_after_reusable_table=cost if method=='direct-template' else build+cost),
        scope='known-family posterior; direct rival matches the exact ceiling, no claim of inverse algorithm superiority')


def evaluate(case):
    truth=case['private'];rows=[]
    for p in case['public']['tiers']:
        for method in ('inverse-history','direct-template','finite-history-ceiling'):
            result=predict(canonical(p),method);selected=truth['selected']
            brier=sum((value-int(i==selected))**2 for i,value in enumerate(result['posterior']))
            first=truth['actual']['trace'][0]
            rows.append(dict(tier=p['tier'],method=method,success=result['prediction']==truth['strategy'],
                strategy_brier=brier,first_actor_probability=result['first_actor_probabilities'].get(first['actor'],0),
                first_operation_probability=result['first_operation_probabilities'].get(str(first['action']),0),
                source_fraction_absolute_error=abs(result['predicted_source_contiguous_fraction']-truth['relationship']['source_contiguous_fraction']),
                missing_output=False,invalid=False,**{k:v for k,v in result.items() if k!='method'}))
    return rows
