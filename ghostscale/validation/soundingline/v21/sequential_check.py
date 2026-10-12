"""Scalar enumeration reference; independent of the vector planner and its cache."""
import math


def states(prior,history):
    result=[]
    for order in (0,1):
      for pad in (0,1):
        mass=float(prior[order])/2
        for action,bit in history:
            if action==0:mass*=int((order^pad)==bit)
            elif action==1:mass*=int(pad==bit)
            else:mass*=.8 if order==bit else .2
        result.append((order,pad,mass))
    total=sum(x[2] for x in result);assert total>0
    return [(o,p,m/total) for o,p,m in result]


def posterior(prior,history):
    rows=states(prior,history)
    return [sum(m for o,p,m in rows if o==g) for g in (0,1)]


def entropy(p):return -sum(x*math.log(x) for x in p if x>0)


def transitions(prior,history,action):
    rows=states(prior,history);out=[]
    for bit in (0,1):
        total=0.
        for order,pad,mass in rows:
            chance=int((order^pad)==bit) if action==0 else int(pad==bit) if action==1 else (.8 if order==bit else .2)
            total+=mass*chance
        if total>1e-15:out.append((total,tuple(sorted(history+((action,bit),)))))
    return out


def choice(prior,history,remaining,fee,policy,threshold):
    available=[a for a in range(8) if a not in dict(history)]
    if not available or remaining==0:return -1
    if policy=='fixed-pair-first':return available[0]
    stop=entropy(posterior(prior,history))
    if policy=='uncertainty-threshold' and stop<=threshold:return -1
    scores=[]
    for action in available:
        score=fee
        for chance,updated in transitions(prior,history,action):
            value=entropy(posterior(prior,updated))
            if policy=='two-step-lookahead' and remaining>=2:
                for next_action in available:
                    if next_action==action:continue
                    candidate=fee+sum(c*entropy(posterior(prior,h)) for c,h in transitions(prior,updated,next_action))
                    value=min(value,candidate)
            score+=chance*value
        scores.append(score)
    best=min(scores);index=next(i for i,v in enumerate(scores) if abs(v-best)<1e-14)
    return available[index] if best<stop-1e-12 else -1
