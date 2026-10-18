"""N16: public-history-only adaptive choice on unchanged finite probe laws."""
import math
import numpy as np
from scipy.special import logsumexp
from . import sacrifice,consideration,precision_habits,trust_attention
from .common import rng,interval

FAMILY='N16';READERS=['P1','P2','P3','P4','P5','unchanged-full-exact-scoring-reader']
SOURCES={'N4d':sacrifice,'N9':consideration,'N11':precision_habits,'N15':trust_attention}

def variants():
    return [dict(id='N16-'+v['id'],family=FAMILY,source=s,original=v)
        for s,model in SOURCES.items() for v in model.variants()]

def state(source,public):
    p=public['baseline'];menu=public['menu'];model=SOURCES[source]
    if source=='N4d':
        prior=model.posterior(p,public['baseline_responses'])
        laws=np.stack([model.laws(tuple(q)) for q in menu])
    elif source=='N9':
        prior=model.posterior(p);laws=model.laws(p['width'],menu).transpose(2,0,1,3)
        if not p['timing']:laws=laws.reshape(6,5,16,2,2).sum(-1)
    elif source=='N11':
        prior,e=model.history_state(p);laws=np.stack([model.laws(e,q) for q in menu])
    else:
        prior,seen,failed=model.history_state(p);laws=np.stack([model.probe_law(seen,failed,q) for q in menu])
    return prior,laws

def information(mass,laws):
    """Exact I(mechanism; next outcome), integrating the nuisance law."""
    joint=np.einsum('ml,qmly->qmy',mass,laws)
    expected=mass.sum(1)[None,:,None]*joint.sum(1)[:,None,:]
    mask=joint>0
    terms=np.zeros_like(joint);terms[mask]=joint[mask]*np.log(joint[mask]/expected[mask])
    return terms.sum((1,2))

def policy_mass(full,scenario,exponent=1.):
    mass=full**exponent
    if scenario.startswith('omit-'):mass[int(scenario.split('-')[1])]=0
    elif scenario=='high-reliability':mass[:,:12]=0
    mass/=mass.sum();return mass

def distributions(full,laws,available,scenario):
    available=np.asarray(available,int);random=np.zeros(len(laws));random[available]=1/len(available)
    fixed=np.zeros(len(laws));fixed[available[0]]=1
    gain=information(policy_mass(full,scenario),laws)
    tempered=information(policy_mass(full,scenario,.5),laws)
    greedy=np.zeros(len(laws));greedy[next(i for i in available if gain[i]>=max(gain[available])-1e-12)]=1
    soft=np.zeros(len(laws));soft[next(i for i in available if tempered[i]>=max(tempered[available])-1e-12)]=1
    return np.stack([fixed,random,greedy,.8*greedy+.2*random,soft]),gain

def update(mass,factor):
    result=mass*factor;return result/result.sum()

def metrics_for(v,public,truth,evidence,design):
    initial,_=state(v['source'],public);m=truth['mechanism'];before=math.log(float(initial[m].sum()))
    budgets=design.get('budgets',[1,2,3,4,6]);values={};type_values={}
    for trace,saved in zip(public['traces'],evidence['posteriors'],strict=True):
        if trace['scenario']!=saved['scenario'] or trace['policy']!=saved['policy']:raise ValueError('N16 trace identity')
        prefix=trace['scenario']+'|'+trace['policy']
        for k in budgets:values[prefix+'|k'+str(k)]=math.log(saved['mass'][k-1][m])-before
        for row in trace['steps']:
            key='expected-info|'+trace['scenario']+'|'+trace['policy']+'|'+str(public['menu'][row['choice']])
            type_values.setdefault(key,[]).append(row['expected_information'][row['choice']])
    # Missing types have zero total information/count, not selectively dropped cases.
    for scenario in sorted(set(t['scenario'] for t in public['traces'])):
        for policy in READERS[:5]:
            types={'N4d':[[1.,3.],[3.,1.]],'N9':[[s,o] for s in range(3) for o in (0,1)],
                'N11':precision_habits.PROBES[:3],'N15':[[s,o] for s in (0,1) for o in (0,1)]}[v['source']]
            for context in sorted(set(str(x) for x in types)):
                key='expected-info|'+scenario+'|'+policy+'|'+context
                vals=type_values.get(key,[]);values[key+'|total']=math.fsum(vals);values[key+'|count']=float(len(vals))
    values['primary']=values['correct|P4|k2']-values['correct|P1|k2']
    for scenario in sorted(set(t['scenario'] for t in public['traces'])):
        for k in budgets:
            values[scenario+'|P3-P2|k'+str(k)]=values[scenario+'|P3|k'+str(k)]-values[scenario+'|P2|k'+str(k)]
            values[scenario+'|P5-P3|k'+str(k)]=values[scenario+'|P5|k'+str(k)]-values[scenario+'|P3|k'+str(k)]
        if scenario.startswith('omit-'):
            omitted=int(scenario.split('-')[1])==m
            values[scenario+'|true-omitted-count']=float(omitted)
            values[scenario+'|true-omitted-P3-P2-k2-total']=float(omitted)*(values[scenario+'|P3-P2|k2'])
    for k in budgets:values['match|k'+str(k)]=values['correct|P4|k'+str(k)]-values['correct|P1|k6']
    return values

def case(v,phase,world,case,fixed,design):
    source=v['source'];model=SOURCES[source]
    baseline,t=model.sample(v['original'],['N16',phase,'baseline'],world,case)
    public=dict(baseline=baseline,menu=baseline['query'])
    if source=='N4d':
        public['baseline_responses']=t['y'][:baseline['design']['probes']]
        fresh,_=model.sample(v['original'],['N16',phase,'additional'],world,case)
        public['menu']=fresh['query']
    prior,laws=state(source,public);m=t['mechanism'];law=t['law']
    random=rng('N16',phase,v['id'],world,case)
    outcomes=[int(np.searchsorted(np.cumsum(p),u,side='right')) for p,u in zip(laws[:,m,law],random.random(6),strict=True)]
    scenarios=['correct']+['omit-'+str(i) for i in range(len(prior))]
    if design.get('reliability_misspecification',True):scenarios.append('high-reliability')
    traces=[];posteriors=[]
    for scenario in scenarios:
        # Common random numbers pair policies/scenarios without exposing outcomes.
        picks=rng('N16-picks',phase,v['id'],world,case).random(6)
        for policy in range(5):
            mass=prior.copy();available=list(range(6));steps=[];posterior=[]
            for step in range(6):
                probs,gain=distributions(mass,laws,available,scenario)
                choice=int(np.searchsorted(np.cumsum(probs[policy]),picks[step],side='right'))
                outcome=outcomes[choice]
                steps.append(dict(choice=choice,outcome=outcome,selection_probabilities=probs.tolist(),
                    expected_information=gain.tolist(),realized_probability=float(probs[policy,choice])))
                mass=update(mass,laws[choice,:,:,outcome]);posterior.append(mass.sum(1).tolist());available.remove(choice)
            traces.append(dict(scenario=scenario,policy='P'+str(policy+1),steps=steps))
            posteriors.append(dict(scenario=scenario,policy='P'+str(policy+1),mass=posterior))
    public['traces']=traces;truth=dict(mechanism=m,law=law)
    evidence=dict(posteriors=posteriors)
    return public,truth,evidence,metrics_for(v,public,truth,evidence,design)

def check(v,public,truth,evidence,fixed,design):
    prior,laws=state(v['source'],public)
    for trace,saved in zip(public['traces'],evidence['posteriors'],strict=True):
        logmass=np.log(prior);available=list(range(6))
        for i,row in enumerate(trace['steps']):
            mass=np.exp(logmass-logsumexp(logmass))
            probs,gain=distributions(mass,laws,available,trace['scenario'])
            if not np.allclose(probs,row['selection_probabilities'],atol=1e-12,rtol=0):raise ValueError('N16 selection propensity')
            if not np.allclose(gain,row['expected_information'],atol=1e-12,rtol=0):raise ValueError('N16 information')
            choice=row['choice'];outcome=row['outcome'];policy=int(trace['policy'][1:])-1
            if choice not in available or not 0<=outcome<laws.shape[-1] or not probs[policy,choice]>0:raise ValueError('N16 invalid action/response')
            if abs(row['realized_probability']-probs[policy,choice])>1e-12:raise ValueError('N16 realized propensity')
            logmass+=np.log(laws[choice,:,:,outcome]);logmass-=logsumexp(logmass)
            if not np.allclose(np.exp(logmass).sum(1),saved['mass'][i],atol=1e-11,rtol=0):raise ValueError('N16 independent log posterior')
            available.remove(choice)
    return metrics_for(v,public,truth,evidence,design)

def study_readout(items):
    groups={s:[item for item in items if item['variant']['source']==s] for s in SOURCES}
    def combined(metric):
        values=np.mean([np.mean([x['metrics'][metric]['world_values'] for x in group],axis=0) for group in groups.values()],axis=0)
        return dict(world_values=values.tolist(),interval=interval(values,['N16','equal-study',metric]))
    result=dict(primary=combined('primary'),equal_study_weight=True,confirmation=False)
    matches={k:combined('match|k'+str(k)) for k in (1,2,3,4,6) if 'match|k'+str(k) in items[0]['metrics']}
    result['matches']=matches;result['smallest_descriptive_match']=next((k for k,x in matches.items() if x['interval']['mean'] is not None and x['interval']['mean']>=0),None)
    result['limitations']=['finite supplied laws','no human generalization','zero propensities for deterministic policies',
        'additional-probe budgets condition on original evidence','law grid bootstrap is descriptive']
    return result

def controls():
    prior=np.ones((2,16))/32
    laws=np.full((2,2,16,2),.5);laws[1,0,:,0]=.9;laws[1,0,:,1]=.1;laws[1,1,:,0]=.1;laws[1,1,:,1]=.9
    p,g=distributions(prior,laws,[0,1],'correct')
    assert p[2,1]==1 and g[1]>0 and abs(g[0])<1e-14
    assert np.allclose(p.sum(1),1) and np.all(p>=0) and np.all(p[3]>0)
    assert np.max(np.abs(information(prior,np.full_like(laws,.5))))<1e-14
    assert np.all(policy_mass(prior,'omit-0')[0]==0) and np.all(prior>0)
    for source,model in SOURCES.items():
        v=next(v for v in variants() if v['source']==source)
        public,truth,evidence,values=case(v,'known-answer',0,0,{},dict(reliability_misspecification=False))
        rebuilt=check(v,public,truth,evidence,{},dict(reliability_misspecification=False))
        assert values==rebuilt
    return dict(passed=True,unique_informative_probe_first=True,all_alias_zero_information=True,
        probability_product_and_log_posteriors_agree=True,selection_probabilities_normalized=True,
        policy_omission_does_not_remove_scoring_support=True,original_generators_unmodified=True)

def run(root,plan,pulse):
    from .directed_common import run as execute
    return execute(root,plan,pulse)
def reconstruct(root,plan):
    from .directed_common import reconstruct as verify
    return verify(root,plan)
