"""Independent elementary-row tables and scalar entropy/posterior reference."""
import math
import numpy as np
from .checker import key
from . import world as W


def check_tables(rows,table):
    current={};history=[{} for _ in range(8)]
    for row in rows:
        m,c,y=int(row[0]),int(row[1]),int(row[10]);weight=float(row[-1])
        ck=key(row,W.CONTEXTS[c],'context',False);hk=key(row,W.CONTEXTS[c],'complete',True)
        current.setdefault(ck,np.zeros((32,128)))[m,y]+=weight
        history[c].setdefault(hk,np.zeros(32))[m]+=weight
    for i,k in enumerate(table['keys']):assert np.allclose(current[tuple(k)],table['current'][i],atol=1e-14)
    for c,(keys,matrix) in table['histories'].items():
        denominators=[sum(v[m] for v in history[c].values()) for m in range(32)]
        expected=np.array([[history[c][tuple(k)][m]/denominators[m] for k in keys] for m in range(32)])
        assert np.allclose(expected,matrix,atol=1e-14)
    return True


def reference(joint):
    targets,reports=joint.shape
    py=[sum(float(joint[y,r]) for r in range(reports)) for y in range(targets)]
    pr=[sum(float(joint[y,r]) for y in range(targets)) for r in range(reports)]
    info=0.;posterior=np.zeros_like(joint)
    for y in range(targets):
        for r in range(reports):
            p=float(joint[y,r])
            if p:info+=p*math.log(p/(py[y]*pr[r]));posterior[y,r]=p/pr[r]
    return info,-sum(p*math.log(p) for p in pr if p),posterior
