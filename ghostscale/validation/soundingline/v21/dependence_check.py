"""Probability-domain references independent of vectorized log likelihoods."""
import math
import numpy as np


def latent_reference(model,x,support,keep=None):
    output=[]
    for i,row in enumerate(x):
        ci=model.current.get(row[:13].tobytes(),len(model.current));belief=model.prior.copy()
        for step,episode in enumerate(row[13:].reshape(-1,13)):
            if keep is None or keep[i,step]:
                hi=model.history.get(episode.tobytes(),len(model.history));belief*=model.emission[:,hi]
                belief/=math.fsum(belief)
        weights=[math.fsum(float(belief[z])*float(model.joint[z,ci,t]) for z in range(len(belief))) if support[i,t] else 0. for t in range(128)]
        total=math.fsum(weights);output.append([v/total for v in weights])
    return np.asarray(output)


def grouped_reference(model,x,support,power):
    output=[]
    for i,row in enumerate(x):
        ci=model.current.get(row[:13].tobytes(),len(model.current));weights=[]
        for target in range(128):
            value=float(model.prior[target])*float(model.c[target,ci]) if support[i,target] else 0.
            for episode in row[13:].reshape(-1,13):
                hi=model.history.get(episode.tobytes(),len(model.history));value*=float(model.h[target,hi])**power
            weights.append(value)
        total=math.fsum(weights);output.append([v/total for v in weights])
    return np.asarray(output)
