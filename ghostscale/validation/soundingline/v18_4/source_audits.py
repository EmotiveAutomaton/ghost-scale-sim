"""Exact paid noisy-provenance decisions in a finite binary source model.

All report strings and policy outcomes are integrated, not sampled. Truth and
copy graphs are latent; only an evaluator receives their joint probability mass.
"""
from itertools import product
import math
import numpy as np
from ..v18_3 import world as W

GRAPHS=((0,0,0,0,0,0),(0,0,0,1,1,1),(0,1,0,1,0,1),(0,1,2,3,4,5))
PAIRS=((0,1),(0,5))
REPORTS=tuple(product((0,1),repeat=6))
METHODS=('stop','naive-stop','common','naive-common','evidence','audit','myopic','lookahead','graph-info')
DIMENSIONS=('family','shared','audit_true','audit_assumed','stake','audit_price')
METRICS=('expected_loss','expected_match','net_utility','audit_count','evidence_count','cost','graph_brier','false_confidence')


def parameters(index):
    r=W.rng('v18.4-source-audits-parameters',index)
    return dict(root_reliability=float(r.uniform(.65,.9)),copy_fidelity=float(r.uniform(.9,.99)),
                fresh_reliability=float(r.uniform(.75,.95)))


def report_table(params,shared,scalar=False):
    """Conditional P(report string | truth,graph), including common root error."""
    table=[]
    for reports in REPORTS:
        row=[]
        for truth in (0,1):
            for graph in GRAPHS:
                total=0.
                for flip,weight in ((0,1-shared),(1,shared)):
                    prob1=params['root_reliability'] if truth^flip else 1-params['root_reliability']
                    if scalar:
                        # Independent latent-root enumeration, not the factored product below.
                        subtotal=0.
                        for roots in product((0,1),repeat=max(graph)+1):
                            term=math.prod(prob1 if x else 1-prob1 for x in roots)
                            for value,g in zip(reports,graph):
                                term*=params['copy_fidelity'] if value==roots[g] else 1-params['copy_fidelity']
                            subtotal+=term
                    else:
                        terms=[]
                        for g in range(max(graph)+1):
                            seen=[v for v,h in zip(reports,graph) if h==g]
                            one=math.prod(params['copy_fidelity'] if v else 1-params['copy_fidelity'] for v in seen)
                            zero=math.prod(1-params['copy_fidelity'] if v else params['copy_fidelity'] for v in seen)
                            terms.append(prob1*one+(1-prob1)*zero)
                        subtotal=math.prod(terms)
                    total+=weight*subtotal
                row.append(total)
        table.append(row)
    return np.asarray(table)


def channels(params,quality):
    result={'evidence':np.array([1-params['fresh_reliability']]*4+[params['fresh_reliability']]*4)}
    for i,(a,b) in enumerate(PAIRS):
        result[f'audit{i}']=np.array([quality if g[a]==g[b] else 1-quality for _ in (0,1) for g in GRAPHS])
    return result


def update(p,likelihood,outcome):
    mass=p*(likelihood if outcome else 1-likelihood);prob=float(mass.sum())
    if prob<=0:raise ValueError('impossible outcome')
    return mass/prob,prob


def truth_prob(p):return p.reshape(2,4).sum(axis=1)
def graph_prob(p):return p.reshape(2,4).sum(axis=0)
def entropy(p):return -math.fsum(float(x)*math.log(float(x)) for x in p if x>0)
def available(history):return tuple(a for a in ('evidence','audit0','audit1') if a=='evidence' or all(h[0]!=a for h in history))


def choose(p,history,method,channel,stake,prices,remaining):
    stop=stake*float(max(truth_prob(p)))
    if remaining==0 or method.endswith('stop'):return 'stop',stop
    if method.endswith('common'):return ('audit0' if not history else 'evidence'),stop
    choices=available(history)
    if method=='evidence':choices=('evidence',)
    if method=='audit':choices=tuple(a for a in choices if a.startswith('audit'))
    if method=='graph-info':
        scores=[]
        for action in choices:
            expected=0.
            for outcome in (0,1):
                after,prob=update(p,channel[action],outcome);expected+=prob*entropy(graph_prob(after))
            scores.append(((entropy(graph_prob(p))-expected)/prices[action],action))
        # This is a declared graph-information-per-price heuristic, not truth utility.
        best=max(scores,key=lambda x:x[0])
        return (best[1] if best[0]>1e-12 else 'stop'),stop
    best_action='stop';best=stop
    depth=1 if method=='myopic' else remaining
    for action in choices:
        value=-prices[action]
        for outcome in (0,1):
            after,prob=update(p,channel[action],outcome)
            future=choose(after,history+((action,outcome),),method,channel,stake,prices,depth-1)[1]
            value+=prob*future
        if value>best+1e-12:best_action,best=action,value
    return best_action,best


def leaves(p,mass,method,assumed,actual,stake,prices,history=(),remaining=2):
    action,_=choose(p,history,method,assumed,stake,prices,remaining)
    if action=='stop':
        return [dict(history=[list(h) for h in history],posterior=p.tolist(),mass=mass.tolist())]
    result=[]
    for outcome in (0,1):
        after,_=update(p,assumed[action],outcome)
        next_mass=mass*(actual[action] if outcome else 1-actual[action])
        result+=leaves(after,next_mass,method,assumed,actual,stake,prices,history+((action,outcome),),remaining-1)
    return result


def metrics(cases,stake,prices):
    sums={k:0. for k in METRICS};by_graph=[dict(sums) for _ in GRAPHS]
    for case in cases:
        for leaf in case['leaves']:
            p=np.asarray(leaf['posterior']);mass=np.asarray(leaf['mass']);t=truth_prob(p);g=graph_prob(p)
            chosen=int(t[1]>t[0]);audits=sum(a.startswith('audit') for a,_ in leaf['history'])
            evidence=sum(a=='evidence' for a,_ in leaf['history']);cost=math.fsum(prices[a] for a,_ in leaf['history'])
            for state,weight in enumerate(mass):
                truth=state//4;graph=state%4
                values=dict(expected_loss=-math.log(max(float(t[truth]),1e-300)),expected_match=float(chosen==truth),
                    net_utility=stake*float(chosen==truth)-cost,audit_count=audits,evidence_count=evidence,cost=cost,
                    graph_brier=math.fsum((float(value)-float(j==graph))**2 for j,value in enumerate(g)),false_confidence=float(t[1-truth]>.95))
                for key,value in values.items():
                    sums[key]+=float(weight)*value;by_graph[graph][key]+=4*float(weight)*value
    return sums,by_graph


def unit(index,shared=.2,audit_true=.75,audit_assumed=.75,stake=1.,audit_price=.05):
    params=parameters(index);table=report_table(params,shared)
    actual=channels(params,audit_true);assumed=channels(params,audit_assumed)
    prices=dict(evidence=.1,audit0=audit_price,audit1=audit_price);rows=[]
    for method in METHODS:
        cases=[]
        for reports,lik in zip(REPORTS,table):
            p=lik/lik.sum()
            if method.startswith('naive'):
                p=np.array([0.,0.,0.,lik[3],0.,0.,0.,lik[7]]);p/=p.sum()
            cases.append(dict(reports=list(reports),leaves=leaves(p,lik/8,method,assumed,actual,stake,prices)))
        summary,by_graph=metrics(cases,stake,prices)
        rows.append(dict(method=method,**summary,by_graph=by_graph,cases=cases))
    return dict(family='S1',index=index,shared=shared,audit_true=audit_true,audit_assumed=audit_assumed,stake=stake,
                audit_price=audit_price,parameters=params,initial_likelihood=table.tolist(),rows=rows,
                scope='exact finite binary truth/graph inquiry; supplied channels; no native uptake or general provenance identification')


def verify(u):
    params=parameters(u['index'])
    if params!=u['parameters']:raise ValueError('parameter identity differs')
    table=report_table(params,u['shared'],scalar=True)
    if not np.allclose(table,u['initial_likelihood'],rtol=0,atol=1e-12) or not np.allclose(table.sum(0),1,rtol=0,atol=1e-12):
        raise ValueError('independent initial likelihood differs')
    assumed=channels(params,u['audit_assumed']);actual=channels(params,u['audit_true'])
    prices=dict(evidence=.1,audit0=u['audit_price'],audit1=u['audit_price'])
    if [r['method'] for r in u['rows']]!=list(METHODS):raise ValueError('reader roster differs')
    for row in u['rows']:
        if [tuple(c['reports']) for c in row['cases']]!=list(REPORTS):raise ValueError('report support differs')
        for case,lik in zip(row['cases'],table):
            start=lik/lik.sum()
            if row['method'].startswith('naive'):
                start=np.array([0.,0.,0.,lik[3],0.,0.,0.,lik[7]]);start/=start.sum()
            for leaf in case['leaves']:
                history=leaf['history']
                if len(history)>2 or len({a for a,_ in history if a!='evidence'})!=sum(a!='evidence' for a,_ in history):raise ValueError('purchase budget differs')
                weights=[];masses=[]
                for state in range(8):
                    weights.append(float(start[state])*math.prod(float(assumed[a][state] if v else 1-assumed[a][state]) for a,v in history))
                    masses.append(float(lik[state])/8*math.prod(float(actual[a][state] if v else 1-actual[a][state]) for a,v in history))
                p=[x/math.fsum(weights) for x in weights]
                if not np.allclose(p,leaf['posterior'],atol=1e-12,rtol=0) or not np.allclose(masses,leaf['mass'],atol=1e-12,rtol=0):raise ValueError('independent path product differs')
            if not np.allclose(np.sum([x['mass'] for x in case['leaves']],axis=0),lik/8,atol=1e-12,rtol=0):raise ValueError('policy path coverage differs')
        values,graphs=metrics(row['cases'],u['stake'],prices)
        if any(not math.isclose(values[k],row[k],abs_tol=1e-10,rel_tol=0) for k in METRICS):raise ValueError('cost or score differs')
        if any(not math.isclose(graphs[g][k],row['by_graph'][g][k],abs_tol=1e-10,rel_tol=0) for k in METRICS for g in range(4)):raise ValueError('conditional graph score differs')
    return True


def controls():
    p=np.ones(8)/8;params=dict(root_reliability=.8,copy_fidelity=.97,fresh_reliability=1.)
    uniform=channels(params,.5);perfect=channels(params,1.)
    posterior,_=update(p,perfect['evidence'],1)
    audit,_=update(p,uniform['audit0'],1)
    prices=dict(evidence=.1,audit0=.05,audit1=.05)
    return {'live:perfect_binary_evidence':bool(np.array_equal(truth_prob(posterior),[0,1])),
            'placebo:uninformative_audit':bool(np.array_equal(audit,p)),
            'positive:costly_useless_audit_stops':choose(p,(),'audit',uniform,1.,prices,2)[0]=='stop'}
