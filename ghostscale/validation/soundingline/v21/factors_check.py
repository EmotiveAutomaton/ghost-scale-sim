"""Scalar generative enumeration independent of B06 array likelihoods."""
import itertools,math
import numpy as np

def reference(y,cues,success,reader):
    states=list(itertools.product((0,1),repeat=4));m=[]
    for g,k,s,c in states:
        if any((g,k,s,c)[b]!=v for b,v in cues.items()):m.append(0.);continue
        intended=1 if g and k and c else 0
        mass=0.
        for worked in (0,1):
            outcome=intended if worked else 1-intended
            if outcome==y:mass+=(float(success[s]) if worked else 1-float(success[s]))/16
        m.append(mass)
    total=math.fsum(m);q=[p/total for p in m]
    if reader in ('joint','direct-table'):return np.array(q)
    selected=0 if reader=='goal-only' else 2;marginal=math.fsum(q[i] for i,st in enumerate(states) if st[selected])
    out=[]
    for st in states:
        p=1.
        for b in range(4):
            if b in cues:p*=float(st[b]==cues[b])
            elif b==selected:p*=marginal if st[b] else 1-marginal
            else:p*=.5
        out.append(p)
    return np.array(out)/math.fsum(out)

def endpoint_reference(q,success):
    out=[0.,0.]
    for i,(g,k,s,c) in enumerate(itertools.product((0,1),repeat=4)):
        intended=int(bool(g and k and c))
        out[intended]+=float(q[i])*float(success[s]);out[1-intended]+=float(q[i])*(1-float(success[s]))
    return np.array(out)
