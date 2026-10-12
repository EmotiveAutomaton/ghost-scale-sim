"""Scalar set enumeration and independent policy-score identities."""
import math
import numpy as np


def reference_frontier(probability):
    masks=[0]*8;values=[0]*8;mass=[1.]+[0.]*7
    for mask in range(1,128):
        k=mask.bit_count();options=sorted({target & mask for target in range(128)})
        weights={value:math.fsum(float(probability[t]) for t in range(128) if t & mask==value) for value in options}
        best=max(weights.values());value=min(v for v in options if weights[v]>=best-1e-12)
        if best>mass[k]+1e-12:masks[k]=mask;values[k]=value;mass[k]=weights[value]
    return np.array(masks),np.array(values),np.array(mass)


def check(probability,truth,answer,ratio):
    for i,t in enumerate(truth):
        mask=int(answer['mask'][i]);value=int(answer['value'][i]);k=mask.bit_count()
        assert answer['asserted'][i]==k and value & mask==value
        mass=math.fsum(float(p) for target,p in enumerate(probability[i]) if target & mask==value)
        assert abs(mass-answer['confidence'][i])<1e-12
        error=bool(int(t) & mask!=value)
        assert bool(answer['false_specificity'][i])==error
        assert abs(answer['loss'][i]-((7-k)/7+ratio*error))<1e-12
        assert abs(answer['expected_loss'][i]-((7-k)/7+ratio*(1-mass)))<1e-12
        assert answer['alternatives'][i]==sum(target & mask==value for target in range(128))

