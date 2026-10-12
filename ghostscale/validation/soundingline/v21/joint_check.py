"""Independent scalar probability reference, including both bit orderings."""
import math
import numpy as np
from . import world as W


def reference(outputs,kind):
    result=[];pairs=[(i,j) for i in range(7) for j in range(i+1,7)]
    for values in outputs:
        probabilities=[]
        for target in W.LABELS:
            bits=target[::-1] if kind.endswith('reverse') else target
            if kind=='direct':value=math.exp(values[len(probabilities)]-max(values))
            elif kind=='pairwise':
                energy=sum(float(values[i])*bits[i] for i in range(7))
                energy+=sum(float(values[7+k])*bits[i]*bits[j] for k,(i,j) in enumerate(pairs))
                value=math.exp(energy)
            elif kind.startswith('autoregressive'):
                value=1.
                for j in range(7):
                    logit=values[j]+sum(values[7+k]*bits[i] for k,(i,jj) in enumerate(pairs) if jj==j)
                    chance=1/(1+math.exp(-float(logit)));value*=chance if bits[j] else 1-chance
            elif kind=='low-rank':
                mix=np.exp(values[:4]-max(values[:4]));mix/=sum(mix);value=0.
                for r in range(4):
                    part=float(mix[r])
                    for j in range(7):
                        chance=1/(1+math.exp(-float(values[4+7*r+j])));part*=chance if bits[j] else 1-chance
                    value+=part
            else:raise ValueError(kind)
            probabilities.append(value)
        probabilities=np.array(probabilities);result.append(probabilities/probabilities.sum())
    return np.array(result)


def check_saved(model,x):
    from ..v20.readers import features
    a,b,c,d=model.params;output=np.tanh(features(x)@a+b)@c+d
    error=float(np.max(np.abs(np.exp(model.predict(x))-reference(output,model.kind))))
    assert error<1e-10
    return error
