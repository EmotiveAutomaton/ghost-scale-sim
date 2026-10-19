"""Promoted N16 pilot kernel; fresh V24 identities, same declared observation law."""
import math
import numpy as np
from ..v23 import directed_probes as model
from .evidence import mixture,mixture_gain


def score(public,truth,evidence,epsilon):
    entries={(e['scenario'],e['policy']):e for e in evidence['posteriors']}
    p=np.asarray(entries['correct','P4']['mass'][1]);q=np.asarray(entries['correct','P1']['mass'][1])
    candidate=mixture(p,q,epsilon);m=truth['mechanism']
    if q[m]<=0 or candidate[m]<=0:raise ValueError('unavailable true-label support')
    value=math.log(candidate[m])-math.log(q[m])
    raw=math.log(p[m])-math.log(q[m])
    if not math.isclose(value,float(mixture_gain(raw,epsilon)),abs_tol=1e-12,rel_tol=0):raise ValueError('mixture score identity')
    if value<math.log(epsilon)-1e-12:raise ValueError('mixture support violation')
    return dict(raw_gain=raw,mixture_gain=value,lower_bound=math.log(epsilon))


def case(variant,identity,block,law,case_index,epsilon):
    if not identity.startswith('v24-') or any(type(x) is not int for x in (block,law,case_index)) or block<0 or not 0<=law<16 or case_index<0:raise ValueError('fresh V24 identity and balanced law coordinates required')
    p,t,e,_=model.case(variant,identity,16*block+law,case_index,{}, {})
    if t['law']!=law:raise ValueError('law coordinate changed')
    return p,t,e,score(p,t,e,epsilon)


def check(variant,public,truth,evidence,epsilon):
    prior,_=model.state(variant['source'],public)
    scenarios=['correct']+['omit-'+str(i) for i in range(len(prior))]+['high-reliability']
    expected=[(scenario,'P'+str(policy)) for scenario in scenarios for policy in range(1,6)]
    for records,length_key in ((public['traces'],'steps'),(evidence['posteriors'],'mass')):
        if [(r['scenario'],r['policy']) for r in records]!=expected or any(len(r[length_key])!=6 for r in records):
            raise ValueError('incomplete policy/scenario trace roster')
    model.check(variant,public,truth,evidence,{}, {})
    # Separate probability-product calculation of the true-label mixture ratio.
    entries={(e['scenario'],e['policy']):e for e in evidence['posteriors']}
    p=entries['correct','P4']['mass'][1][truth['mechanism']]
    q=entries['correct','P1']['mass'][1][truth['mechanism']]
    return dict(raw_gain=math.log(p/q),mixture_gain=math.log((1-epsilon)*(p/q)+epsilon),lower_bound=math.log(epsilon))


def aggregate(records,cases_per_cell):
    if type(cases_per_cell) is not int or cases_per_cell<1:raise ValueError('positive integer cases required')
    variants=model.variants();required={v['id'] for v in variants};seen=set();by_block={}
    for row in records:
        block,law,variant,case_index=row['identity']
        key=(block,law,variant,case_index)
        if any(type(x) is not int for x in (block,law,case_index)) or block<0 or key in seen or variant not in required or not 0<=law<16 or not 0<=case_index<cases_per_cell or not math.isfinite(row['mixture_gain']):raise ValueError('duplicate or invalid balanced cell')
        seen.add(key);by_block.setdefault(block,[]).append(row['mixture_gain'])
    if not by_block or set(by_block)!=set(range(len(by_block))):raise ValueError('nonconsecutive independent blocks')
    for block in by_block:
        expected={(block,l,v,c) for l in range(16) for v in required for c in range(cases_per_cell)}
        if {k for k in seen if k[0]==block}!=expected:raise ValueError('incomplete balanced block')
    return [math.fsum(by_block[b])/len(by_block[b]) for b in sorted(by_block)]
