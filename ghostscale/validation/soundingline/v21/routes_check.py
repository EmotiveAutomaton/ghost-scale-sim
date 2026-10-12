"""Scalar E04 maker/route integration and independent program interpreter."""
import math
import numpy as np

def execute(program):
    value=None
    for op in program:
        if op=='zero':value=0
        elif op=='one':value=1
        elif op=='flip':
            assert value in (0,1);value=1-value
        elif op=='inspect':assert value in (0,1)
        else:raise ValueError(op)
    return value

def reference(policy,history,goal,witness):
    # Direct sums over maker/current/future routes, rather than producer log updates.
    weights=[]
    for maker in range(4):
        likelihood=.25
        for past_goal,method in history:likelihood*=float(policy[int(past_goal),maker,int(method)])
        weights.append(likelihood)
    allowed=list(range(4)) if witness is None else [r for r in range(4) if (r==3)==bool(witness[1])] if witness[0]=='inspection' else [witness[1]]
    joint=[[weights[m]*float(policy[goal,m,r]) if r in allowed else 0. for r in range(4)] for m in range(4)]
    total=math.fsum(x for row in joint for x in row)
    current=np.array([math.fsum(joint[m][r] for m in range(4))/total for r in range(4)])
    future=np.array([math.fsum(joint[m][r]*float(policy[1-goal,m,f]) for m in range(4) for r in range(4))/total for f in range(4)])
    return current,future
