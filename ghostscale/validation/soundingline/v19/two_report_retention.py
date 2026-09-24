"""Two independent retrospective reports: joint versus separate past tables.

The supplied future grouping is exact. Singleton groups contribute identical
numerators to both accounts and may be combined when measuring total variation.
Every report context and endpoint is enumerated; symmetric time pairs retain
their ordered multiplicity. This kernel alone is not a native admitted job.
"""
from itertools import combinations_with_replacement
import numpy as np
from .reachable_retrospective import TOL


def pair_state(weight, st, law, first, second):
    w = np.asarray(weight, float); law = np.asarray(law, float)
    if w.shape != st['mapping'].shape or not np.isfinite(w).all() or (w < 0).any() or abs(w.sum()-1) > TOL:
        raise ValueError('weights')
    if law.shape != (16,4,8) or not np.isfinite(law).all() or (law < 0).any() or np.max(abs(law.sum(-1)-1)) > TOL:
        raise ValueError('law')
    if not 1 <= first <= len(st['past']) or not 1 <= second <= len(st['past']):
        raise ValueError('times')
    mapping = st['mapping']; groups = len(st['signatures'])
    mixed = np.flatnonzero(st['counts'] > 1)
    index = np.full(groups,-1,dtype=int); index[mixed] = np.arange(len(mixed))
    selected = index[mapping] >= 0
    a,b = st['past'][first-1],st['past'][second-1]
    code = 16*a+b
    joint = np.bincount(index[mapping[selected]]*256+code[selected],weights=w[selected],minlength=len(mixed)*256).reshape(len(mixed),16,16)
    fixed = np.bincount(code[~selected],weights=w[~selected],minlength=256).reshape(16,16)
    channels = law.reshape(16,32)
    true_mixed = (channels.T @ joint) @ channels
    same = (channels.T @ fixed) @ channels
    q = joint.sum(axis=(1,2))
    one = joint.sum(-1) @ channels; two = joint.sum(-2) @ channels
    # A repeated time names the same variable. Its single table already contains
    # the needed dependence; do not manufacture an independent-copy rival.
    separate = (true_mixed.copy() if first == second else np.divide(
        one[:,:,None]*two[:,None,:],q[:,None,None],out=np.zeros_like(true_mixed),where=q[:,None,None]>0))
    truth = same+true_mixed.sum(0); rival = same+separate.sum(0)
    if np.max(abs(truth.reshape(4,8,4,8).sum(axis=(1,3))-1)) > TOL or np.max(abs(rival.reshape(4,8,4,8).sum(axis=(1,3))-1)) > TOL:
        raise ValueError('report mass')
    possible = truth>0; rival_possible = rival>0; both = possible & rival_possible
    exact_post = np.divide(true_mixed,truth[None],out=np.zeros_like(true_mixed),where=possible[None])
    separate_post = np.divide(separate,rival[None],out=np.zeros_like(separate),where=rival_possible[None])
    inv_true = np.divide(1.,truth,out=np.zeros_like(truth),where=possible)
    inv_rival = np.divide(1.,rival,out=np.zeros_like(rival),where=rival_possible)
    tv = .5*(abs(exact_post-separate_post).sum(0)+same*abs(inv_true-inv_rival))
    # Zero posterior groups remain structurally present; they are not new laws.
    return dict(joint=joint,fixed_joint=fixed,mixed_groups=mixed,true_mixed=true_mixed,
                separate_mixed=separate,singleton_numerator=same,report_probability=truth,
                separate_report_probability=rival,possible=possible,separate_possible=rival_possible,
                group_total_variation=np.where(both,tv,np.nan))


def evaluate(weight,st,law,times):
    times = tuple(sorted(int(t) for t in times))
    if not times or len(times) != len(set(times)): raise ValueError('source times')
    pairs=[];multiplicity=[];support=[];rival_support=[];mean=[];maximum=[];unsupported=[]
    for a,b in combinations_with_replacement(times,2):
        r=pair_state(weight,st,law,a,b)
        pairs.append((a,b));multiplicity.append(1 if a==b else 2)
        support.append(r['possible']);rival_support.append(r['separate_possible'])
        comparable=r['possible'] & r['separate_possible']
        mean.append(float(np.sum(r['report_probability']*np.nan_to_num(r['group_total_variation']))/16))
        maximum.append(float(np.max(r['group_total_variation'][comparable],initial=0)))
        unsupported.append(float(np.sum(r['report_probability']*~r['separate_possible'])/16))
    return dict(pair_times=np.asarray(pairs,np.int32),ordered_multiplicity=np.asarray(multiplicity,np.int32),
                possible=np.asarray(support),separate_possible=np.asarray(rival_support),
                expected_group_total_variation=np.asarray(mean),max_group_total_variation=np.asarray(maximum),
                unsupported_report_mass=np.asarray(unsupported))
