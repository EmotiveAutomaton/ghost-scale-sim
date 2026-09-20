"""Fixed supervised decoder ladder; test labels never select a readout."""
import math
import numpy as np


def probabilities(scores, temperature):
    logits=scores/float(temperature);logits=logits-logits.max(1,keepdims=True)
    p=np.exp(logits);p/=p.sum(1,keepdims=True)
    return p


def loss(y,p):
    if y.shape!=p.shape or not np.all(np.isfinite(p)) or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-8):
        raise ValueError('invalid decoder forecast')
    if np.any((y>0)&(p==0)):return float('inf')
    logs=np.zeros_like(p);np.log(p,out=logs,where=p>0)
    return float(-np.sum(y*logs)/len(y))


def transform(model,x):
    x=(np.asarray(x,float)-model['center'])/model['scale']
    return np.tanh(x@model['projection']+model['bias']) if model['projection'].size else x


def fit(x,y,dev,dev_y,kind,seed=180404,width=128,alphas=(.1,10.,1000.),temperatures=(.05,.1,.25,.5,1.),check=lambda:None):
    if kind not in ('linear','nonlinear'):raise ValueError('undeclared decoder')
    if len(x)!=len(y) or len(dev)!=len(dev_y):raise ValueError('decoder supervision alignment')
    for a in (y,dev_y):
        if np.any(a<0) or not np.allclose(a.sum(1),1,atol=1e-6):raise ValueError('invalid decoder targets')
    center=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    r=np.random.default_rng(seed)
    projection=r.normal(0,1/math.sqrt(x.shape[1]),(x.shape[1],width)) if kind=='nonlinear' else np.empty((0,0))
    model=dict(center=center,scale=scale,projection=projection,bias=r.normal(0,.5,width) if kind=='nonlinear' else np.empty(0))
    a=transform(model,x);v=transform(model,dev);feature_mean=a.mean(0);a-=feature_mean;v-=feature_mean
    intercept=y.mean(0);cross=a.T@(y-intercept);gram=a.T@a;best=None;curve=[]
    for alpha in alphas:
        check();coefficients=np.linalg.solve(gram+alpha*np.eye(gram.shape[0]),cross)
        scores=v@coefficients+intercept
        for temperature in temperatures:
            value=loss(dev_y,probabilities(scores,temperature));curve.append(dict(alpha=alpha,temperature=temperature,dev_loss=value))
            if best is None or value<best[0]:best=(value,alpha,temperature,coefficients)
    if not np.isfinite(best[0]):raise ValueError('no finite decoder')
    model.update(feature_mean=feature_mean,intercept=intercept,coefficients=best[3],temperature=np.asarray(best[2]))
    return model,dict(kind=kind,selected_alpha=best[1],selected_temperature=best[2],best_dev_loss=best[0],curve=curve,
        input_features=x.shape[1],fitted_parameters=int(best[3].size+len(intercept)),
        fixed_projection_parameters=int(projection.size+model['bias'].size),training_rows=len(x),
        scope='ridge probability readout; nonlinear means a fixed random tanh map, not a trained neural encoder')


def predict(model,x):
    return probabilities((transform(model,x)-model['feature_mean'])@model['coefficients']+model['intercept'],model['temperature'])


def permute_targets(target,query,context_features,seed):
    """Preserve the exact target multiset within each declared question context."""
    result=target.copy();r=np.random.default_rng(seed)
    contexts=query[:,:context_features]
    for context in np.unique(contexts,axis=0):
        rows=np.flatnonzero(np.all(contexts==context,axis=1));result[rows]=target[r.permutation(rows)]
    return result


def controls():
    x=np.eye(16);y=x.copy()
    model,_=fit(x,y,x,y,'linear',alphas=(.1,),temperatures=(.05,))
    p=predict(model,x)
    uniform=np.full((16,16),1/16)
    null,_=fit(x,uniform,x,uniform,'nonlinear',width=32,alphas=(1,),temperatures=(.1,))
    blank=predict(null,x)
    scalar=math.fsum(-float(q)*math.log(float(v)) for q,v in zip(y[0],p[0]) if q)/1
    return dict(live_identity_recovery=bool(np.all(p.argmax(1)==np.arange(16)) and loss(y,p)<.01),
        placebo_uniform=bool(np.allclose(blank,uniform,atol=1e-12)),
        independent_scalar=bool(abs(scalar-loss(y[:1],p[:1]))<1e-12))
