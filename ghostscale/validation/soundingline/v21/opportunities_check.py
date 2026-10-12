"""Scalar reference for fixed-maker inference across constrained contexts."""
import math

def success(goal,skill,context,law):
    allowed,cost=context
    if allowed>=0:choice=float(allowed)
    else:choice=1/(1+math.exp(-(law['preference']*(2*goal-1)-cost)))
    accuracy=law['accuracy'][skill]
    return choice*accuracy+(1-choice)*(1-accuracy)

def reference(contexts,outcomes,law):
    mass=[]
    for goal in (0,1):
        for skill in (0,1):
            factors=[]
            for context,observed in zip(contexts,outcomes):
                p=success(goal,skill,context,law);factors.append(p if observed else 1-p)
            mass.append(.25*math.prod(factors))
    total=math.fsum(mass)
    return [m/total for m in mass]

def information(prior,context,law):
    channels=[success(g,s,context,law) for g in (0,1) for s in (0,1)]
    out=0.
    for y in (0,1):
        joint=[q*(p if y else 1-p) for q,p in zip(prior,channels)];prob=math.fsum(joint)
        for q,j in zip(prior,joint):
            if j:out+=j*math.log(j/(q*prob))
    return out
