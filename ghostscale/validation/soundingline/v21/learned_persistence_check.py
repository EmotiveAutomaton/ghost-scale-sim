"""Probability-domain per-case reference, independently aggregating contexts."""
import math
import numpy as np


def reference(model,x,support):
    rates=(1.,.9,.5,0.);states=len(model.prior);out=[];posterior=[];by=[[] for _ in rates]
    cm=np.zeros((states,8))
    for z in range(states):
        for cx in range(8):
            cm[z,cx]=math.fsum([float(model.emission[z,i]) for key,i in model.history.items()
                if int(np.frombuffer(key,dtype=np.int8)[3:6]@np.array([4,2,1]))==cx]+[float(model.emission[z,-1])/8])
    for i,row in enumerate(x):
        ci=model.current.get(row[:13].tobytes(),len(model.current));masses=[]
        for rate in rates:
            belief=model.prior.copy()
            for episode in row[13:].reshape(-1,13):
                hi=model.history.get(episode.tobytes(),len(model.history));cx=int(episode[3:6]@np.array([4,2,1]))
                bg=math.fsum(float(model.prior[z])*float(model.emission[z,hi]) for z in range(states))
                if hi==len(model.history):bg/=8
                den=math.fsum(float(model.prior[z])*float(cm[z,cx]) for z in range(states))
                for z in range(states):
                    own=float(model.emission[z,hi])/(8 if hi==len(model.history) else 1)
                    belief[z]*=rate*own+(1-rate)*float(cm[z,cx])*bg/den
            masses.append(np.where(support[i],belief@model.joint[:,ci,:],0.))
        totals=np.array([math.fsum(p) for p in masses]);posterior.append(totals/math.fsum(totals))
        total=np.sum(masses,axis=0);out.append(total/math.fsum(total))
        for j in range(4):by[j].append(masses[j]/totals[j])
    return dict(inferred=np.array(out),posterior_rates=np.array(posterior),by_rate={str(r):np.array(by[j]) for j,r in enumerate(rates)})
