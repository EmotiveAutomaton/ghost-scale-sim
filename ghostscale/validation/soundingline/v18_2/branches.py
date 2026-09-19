"""Predeclared discovery interventions; conditions never enter ordinary readers."""
import copy
import json
import math
import random
import time
import numpy as np
from ..v16.learning import learn,encoding_cost
from ..v16.craft import construct
from ..v16.world import execute
from . import model as m
from .verify import interval


def row(case,tier,j,method,result,condition='base',truth=None,observed=None):
    probe=case['probes'][j]
    truth=truth if truth is not None else m.artifacts(m.policy(case['world'],case['truth']['future_state'],probe['context']))
    observed=probe['observed']['artifact'] if observed is None else observed
    r=dict(tier=tier,probe=j,method=method,condition=condition,result=result,
           instrument='model_mismatch' if result.get('mismatch') else 'valid')
    if not result.get('mismatch'):r['scores']=m.score(result['probabilities'],truth,observed)
    return r


def persistence(case,design):
    rows=[]
    # Same lineage, four paired cells. A policy change alters preference only;
    # goal intervention alters current goal only, while skill remains acquired.
    for changed_pref in (False,True):
        for changed_goal in (False,True):
            c=copy.deepcopy(case)
            state=c['truth']['initial_state'][:]
            rng=random.Random(m.seed(case['case_id'],'paired-persistence'))
            for t,obs in enumerate(c['history']):
                s=state[:]
                if changed_pref and t>=len(c['history'])//2:s[1]=(s[1]+1)%3
                c['history'][t],choice=m.draw(c['world'],s,obs['context'],rng)
                c['truth']['history'][t]=dict(state=s,choice=choice)
            if changed_pref:state[1]=(state[1]+1)%3
            c['truth']['future_state']=state
            for j,probe in enumerate(c['probes']):
                probe['context']['goal']=2 if design.get('novel_goal') and changed_goal else int(changed_goal)
                probe['observed'],probe['choice']=m.draw(c['world'],state,probe['context'],rng)
            for r in m.evaluate(c,('direct','raw','persistent','adaptive','rebuilt','oracle'),('process-history',)):
                r['condition']=f'goal-{int(changed_goal)}-preference-{int(changed_pref)}'
                r['counterfactual_evaluator']=dict(state=state,history=c['history'],probes=c['probes'])
                rows.append(r)
    return rows


def perspective(case,design):
    rows=[];rng=random.Random(m.seed(case['case_id'],'perspective'))
    # Actual routing truth is 1 in all siblings; maker sees stale 0 or corrected 1.
    conditions=[('shared-true',1,1),('false-belief',0,0),('reader-only-correction',0,1),('maker-correction',1,1)]
    for name,signal,fact in conditions:
        for j in range(4):
            c=copy.deepcopy(case);ctx=m.context(j%2,signal,fact,.5 if design.get('unknown_access') else 1.)
            c['probes'][j]['context']=ctx
            obs,choice=m.draw(c['world'],c['truth']['future_state'],ctx,rng)
            c['probes'][j].update(observed=obs,choice=choice)
            payload=m.public_packet(c,'process-history',j)
            truth=m.artifacts(m.policy(c['world'],c['truth']['future_state'],ctx))
            for method in ('direct','raw','persistent','oracle','shared-world'):
                if method=='oracle':result=dict(probabilities=truth.tolist(),mismatch=False,evaluations=len(m.PROGRAMS))
                elif method=='shared-world':
                    public=json.loads(payload);public['current']['signal']=fact
                    # Deliberately wrong integration; receives the SAME public fields.
                    result=m.infer(m.canonical(public))
                else:result=m.infer(payload,method)
                r=row(c,'process-history',j,method,result,name,truth)
                r['counterfactual_evaluator']=dict(actual_routing=1,context=ctx,observed=obs,true_distribution=truth.tolist())
                rows.append(r)
    return rows


def family_revision(case,design):
    rows=[];fixed=tuple(s for s in m.STATES if s[0]==0)
    # Hypothesis vocabulary is frozen here: acquired skill, goal default, belief
    # prior. Context testimony is one fallible source, not three observations.
    for j in range(4):
        payload=m.public_packet(case,'process-history',j);public=json.loads(payload)
        fixed_result=m.infer(payload,states=fixed)
        evidence_seen=len(public['history'])
        disconfirmed=fixed_result['mismatch']
        for method in ('fixed','safe-abstention','bounded-expansion','raw','wrong-context'):
            result=copy.deepcopy(fixed_result)
            proposals=0;access=0
            if method=='raw':result=m.infer(payload,'raw')
            elif method=='bounded-expansion' and disconfirmed:
                proposals=len(m.STATES)-len(fixed)
                result=m.infer(payload)
                access=sum(len(h['program']) for h in case['history']) if design.get('paid_access') else 0
            elif method=='wrong-context':
                # Attractive report asserts no acquired skill. Two copies retain
                # the same source id, so they supply exactly one likelihood factor.
                result=m.infer(payload)
                if not result['mismatch']:
                    weights=np.array(result['posterior']);weights*=np.array([.8 if s[0]==0 else .1 for s in m.STATES]);weights/=weights.sum()
                    q=weights@np.array([m.artifacts(m.policy(public['world'],s,public['current'])) for s in m.STATES])
                    result['probabilities']=q.tolist()
            r=row(case,'process-history',j,method,result,'paid-setup' if design.get('paid_access') else 'supplied-setup')
            r['revision']=dict(disconfirmed=disconfirmed,revised=method=='bounded-expansion' and disconfirmed,
                proposal_cost=proposals,evaluation_cost=result['evaluations'],access_primitives=access,
                latest_evidence_index=evidence_seen-1,future_probe_used_for_proposal=False,
                context_sources=['source-A','source-A','source-A'],independent_context_sources=1,
                abstained=method=='safe-abstention' and disconfirmed,
                unresolved=bool(result['mismatch']),
                setup_scope='replayed observed commands from supplied fresh empty boards; observation execution charged, fresh-object provisioning excluded')
            if access:
                programs=[h['program'] for h in case['history']]
                r['revision']['setup_programs']=programs
                r['revision']['setup_artifacts']=[execute(p).artifact for p in programs]
            rows.append(r)
    return rows


def uptake(case,design):
    rng=random.Random(m.seed(case['case_id'],'uptake'));rows=[]
    motif=case['world']['groups'][0];other=case['world']['groups'][1]
    dependency=design.get('dependency',False)
    # Demonstration's extra action expresses the teacher's unwanted goal. In the
    # dependency arm it is part of each available successful procedure trace.
    useful=list(motif);full=useful+[other[0]]
    records=[]
    for i in range(12):
        program=(full if dependency else useful) if i%2==0 else [other[0]]
        if rng.random()<.1:program=[rng.randrange(4)]
        records.append(dict(program=program,artifact=execute(program).artifact,source=i))
    own_target=sum(1<<x for x in useful)
    for attend in ('all','procedure'):
        for weight in (0.,.25,1.):
            for apply in (False,True):
                selected=records if attend=='all' else records[::2]
                acquisition=learn([x['program'] for x in selected],[x['artifact'] for x in selected],capacity=2)
                # A symmetric Beta update on observed goal feature; weight is
                # independent of information selection and procedure application.
                successes=sum(bool(x['artifact']&(1<<other[0])) for x in selected)
                a=1+weight*successes;b=9+weight*(len(selected)-successes)
                unwanted=a/(a+b)
                library=acquisition.library if apply else ()
                # Same total envelope, attention and fit charged before search.
                acquisition_cost=sum(len(x['program']) for x in selected)+len(selected)
                remaining=max(0,128-acquisition_cost-acquisition.processing_cost)
                output=construct(own_target,library,primitive_budget=remaining)
                execution=execute(output['program'])
                value=float(execution.legal and execution.artifact==own_target)
                rows.append(dict(tier='process-history',probe=0,method=f'{attend}-weight-{weight}-apply-{int(apply)}',
                    condition='dependency' if dependency else 'separable',instrument='valid',
                    result=dict(acquired=selected,library=list(map(list,acquisition.library)),output=output,
                                acquisition_cost=acquisition_cost,fit_cost=acquisition.processing_cost,
                                beta_parameters=[a,b],total_envelope=128),
                    scores=dict(task_transfer=value,unwanted_goal_uptake=unwanted,
                                retained_variance=a*b/((a+b)**2*(a+b+1)),
                                construction_cost=execution.primitive_cost,query_cost=len(selected))))
    return rows


def evaluate_branch(case,design):
    return {'g1':persistence,'g2':perspective,'g3':family_revision,'g5':uptake}[design['branch']](case,design)


def population_check(design,limited):
    start=time.process_time();results=[]
    # Gaussian hierarchical generator has exact normal means and variances.
    # The population estimand is the superpopulation mean 0, individual is theta0.
    for persons in (4,64):
        for within in (2,64):
            for heterogeneity in (.1,1.):
                for bias in (0.,1.):
                    for duplicates in (1,8):
                        errors=[];covered=[];naive_covered=[];individual_errors=[];individual_covered=[];widths=[];iwidths=[]
                        for k in range(256):
                            if limited() or time.process_time()-start>1800:
                                return dict(state='resource_cutoff',cells=results,cpu_seconds=time.process_time()-start)
                            rng=np.random.default_rng(m.seed(design['namespace'],persons,within,heterogeneity,bias,k))
                            theta=rng.normal(bias*heterogeneity,heterogeneity,size=persons)
                            means=theta+rng.normal(0,1/math.sqrt(within),size=persons)
                            estimate=float(means.mean());se=math.sqrt((heterogeneity**2+1/within)/persons)
                            # Dependence-aware count versus naive duplicate-counting.
                            for label,scale in (('source-aware',1.),('duplicate-naive',math.sqrt(duplicates))):
                                radius=1.96*se/scale
                                if label=='source-aware':
                                    errors.append(estimate**2);covered.append(abs(estimate)<=radius);widths.append(2*radius)
                                else:naive_covered.append(abs(estimate)<=radius)
                            target=float(theta[0]);personal=float(means[0]);iradius=1.96/math.sqrt(within)
                            individual_errors.append((personal-target)**2);individual_covered.append(abs(personal-target)<=iradius);iwidths.append(2*iradius)
                        results.append(dict(persons=persons,within=within,heterogeneity=heterogeneity,selection_bias=bias,
                            duplicates=duplicates,replicates=256,population_mse=float(np.mean(errors)),
                            population_coverage=float(np.mean(covered)),population_width=float(np.mean(widths)),
                            individual_mse=float(np.mean(individual_errors)),individual_coverage=float(np.mean(individual_covered)),
                            individual_width=float(np.mean(iwidths)),naive_duplicate_width=float(np.mean(widths))/math.sqrt(duplicates),
                            naive_duplicate_coverage=float(np.mean(naive_covered))))
    return dict(state='complete',cells=results,cpu_seconds=time.process_time()-start,
                estimands=['superpopulation mean','first sampled individual parameter'],
                scope='method counterexamples; neither interval is an alignment guarantee')
