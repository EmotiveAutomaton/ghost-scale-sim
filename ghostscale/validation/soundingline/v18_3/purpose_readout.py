"""Fixed linear readouts for a new purpose; no encoder or test-label fitting."""
import numpy as np

SLICES=(slice(0,3),slice(3,5),slice(5,7),slice(7,9))


def probabilities(scores,temperature):
    out=np.zeros_like(scores,dtype=float)
    for columns in SLICES:
        values=scores[:,columns]/temperature;values-=values.max(1,keepdims=True)
        p=np.exp(values);p/=p.sum(1,keepdims=True);out[:,columns]=p/4
    return out


def loss(truth,p):
    if np.any((truth>0)&(p==0)):return float('inf')
    logs=np.zeros_like(p);np.log(p,out=logs,where=p>0)
    return float(-(truth*logs).sum(1).mean()-np.log(4))


def fit(x,y,dev,dev_y,alphas=(.1,1.,10.),temperatures=(.25,.5,1.,2.)):
    center=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    x=(x-center)/scale;v=(dev-center)/scale;intercept=y.mean(0)*4
    # One deterministic factorization services the entire fixed regularization ladder.
    u,s,vt=np.linalg.svd(x,full_matrices=False);target=y*4-intercept;projected=u.T@target
    best=None;curve=[]
    for alpha in alphas:
        coefficients=vt.T@((s/(s*s+alpha))[:,None]*projected)
        scores=v@coefficients+intercept
        for temperature in temperatures:
            value=loss(dev_y,probabilities(scores,temperature));curve.append(dict(alpha=alpha,temperature=temperature,dev_loss=value))
            if best is None or value<best[0]:best=(value,alpha,temperature,coefficients)
    if not np.isfinite(best[0]):raise ValueError('no finite role readout')
    return dict(center=center,scale=scale,intercept=intercept,coefficients=best[3],temperature=np.array(best[2])),dict(
        selected_alpha=best[1],selected_temperature=best[2],best_dev_loss=best[0],curve=curve,
        feature_floats=x.shape[1],readout_parameters=int(best[3].size+len(intercept)),
        scope='linear supervised purpose readout; unequal feature/storage dimensions are explicit')


def predict(model,x):return probabilities(((x-model['center'])/model['scale'])@model['coefficients']+model['intercept'],float(model['temperature']))


def accuracies(truth,p,tolerance=1e-10):
    """Expected credit under uniform, independent resolution of numerical ties."""
    credits=[]
    for columns in SLICES:
        actual=truth[:,columns].argmax(1);values=p[:,columns]
        tied=values>=values.max(1,keepdims=True)-tolerance
        credits.append(tied[np.arange(len(values)),actual]/tied.sum(1))
    credits=np.stack(credits,axis=1)
    return credits.mean(1),credits.prod(1)
