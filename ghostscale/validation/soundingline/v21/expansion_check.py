"""Scalar library, posterior and exhaustive policy reference without producer imports."""
from itertools import combinations
from math import comb,log

def bit(model,context):
    first=context//2;second=context%2
    if model==1:return (first+second)%2
    if model==2:return (first+int(context==3))%2
    if model==3:return (first+int(context in (1,2)))%2
    return first

def reference(xs,ys,reliability,noise,budget):
    prior=[.5,1/6,1/6,1/6];rate=.5+(reliability-.5)*(1-2*noise);mass=[]
    for model in range(4):
        value=prior[model]
        for x,y in zip(xs,ys):value*=rate if bit(model,int(x))==y else 1-rate
        mass.append(value)
    def posterior(selected):
        ids={0}|{k for k in selected if k<4};total=sum(mass[k] for k in ids)
        return [mass[k]/total if k in ids else 0. for k in range(4)]
    def prediction(p):return [sum(p[k]*(reliability if bit(k,x) else 1-reliability) for k in range(4)) for x in range(4)]
    full=prediction(posterior([1,2,3]));best=None
    for count in range(budget+1):
        for subset in combinations((1,2,3,4),count):
            pred=prediction(posterior(subset));risk=sum(-a*log(b)-(1-a)*log(1-b) for a,b in zip(full,pred))/4+.01*count
            if best is None or risk<best[0]-1e-12:best=(risk,subset)
    hits=sum(bit(0,int(x))==y for x,y in zip(xs,ys));n=len(xs);tail=sum(comb(n,j)*rate**j*(1-rate)**(n-j) for j in range(hits+1))
    fixed=tuple(range(1,budget+1));selections={'base':(),'confidence':fixed if max(reliability,1-reliability)<.75 else (), 'surprise':fixed if tail<=.05 else (), 'expected-gain':best[1], 'fixed':fixed,'full-library':(1,2,3)}
    return {name:dict(probability_one=prediction(posterior(selected)),selected=[int(k in selected) for k in range(1,5)]) for name,selected in selections.items()}
