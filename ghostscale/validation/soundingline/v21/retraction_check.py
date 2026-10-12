"""Independent probability-domain recomputation from source-linked raw reports."""
import math

def posterior(prior, records):
    mass=[]
    for target in (0,1):
        factors=[1. if value<0 else reliability if value==target else 1-reliability
                 for value,reliability in records.values()]
        mass.append(float(prior[target])*math.prod(factors))
    total=math.fsum(mass)
    return ([v/total for v in mass],False) if total else (list(prior),True)

def updated(records, update):
    result=dict(records);source,value,reliability=update
    if value is None:result.pop(source,None)
    else:result[source]=(value,reliability)
    return result
