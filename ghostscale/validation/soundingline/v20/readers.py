"""Label-matched finite readers; all arms receive identical legal support."""
import time
import numpy as np
from .world import LABELS,seed

def normalize(a):
    a=np.asarray(a,float);s=a.sum(axis=-1,keepdims=True)
    if np.any(s<=0) or np.any(a<0) or not np.isfinite(a).all():raise ValueError('invalid mass')
    return a/s

def features(x):
    return np.concatenate([(x==v).astype(float) for v in [-1,0,1]],axis=1)

class Readers:
    def __init__(self,x,y,epochs=120,fit_seed=0,pulse=lambda **kw:None):
        start=time.process_time();self.width=64;self.epochs=epochs
        self.prior=np.bincount(y,minlength=128)+.5;self.prior=normalize(self.prior)
        self.table={}
        for row,t in zip(x,y):
            key=row.tobytes();self.table.setdefault(key,np.zeros(128))[t]+=1
        self.conditionals=np.ones((128,x.shape[1],3))*.5
        for j in range(x.shape[1]):np.add.at(self.conditionals[:,j,:],(y,x[:,j]+1),1)
        self.conditionals=normalize(self.conditionals)
        z=features(x);rng=np.random.default_rng(seed('v20-reader',fit_seed));self.w1=rng.normal(0,.1,(z.shape[1],64));self.b1=np.zeros(64)
        self.w2=rng.normal(0,.03,(64,128));self.b2=np.log(self.prior)
        # Fixed full-batch supervised MLP. No optimizer/model selection by test results.
        for ep in range(epochs):
            hidden=np.tanh(z@self.w1+self.b1);logits=hidden@self.w2+self.b2
            pp=np.exp(logits-logits.max(1,keepdims=True));pp=normalize(pp);pp[np.arange(len(y)),y]-=1;pp/=len(y)
            grad=pp@self.w2.T*(1-hidden**2)
            self.w2-=.7*(hidden.T@pp+1e-4*self.w2);self.b2-=.7*pp.sum(0)
            self.w1-=.7*(z.T@grad+1e-4*self.w1);self.b1-=.7*grad.sum(0)
            if ep%10==0:pulse(phase='fit',epoch=ep)
        width=max(1,round((self.w1.size+self.b1.size+self.w2.size+self.b2.size-7)/(z.shape[1]+8)))
        self.f1=rng.normal(0,.1,(z.shape[1],width));self.fb1=np.zeros(width)
        self.f2=rng.normal(0,.03,(width,7));self.fb2=np.zeros(7);target=LABELS[y]
        for ep in range(epochs):
            h=np.tanh(z@self.f1+self.fb1);p=1/(1+np.exp(-np.clip(h@self.f2+self.fb2,-40,40)))
            delta=(p-target)/len(y);grad=delta@self.f2.T*(1-h*h)
            self.f2-=.7*(h.T@delta+1e-4*self.f2);self.fb2-=.7*delta.sum(0)
            self.f1-=.7*(z.T@grad+1e-4*self.f1);self.fb1-=.7*grad.sum(0)
            if ep%10==0:pulse(phase='factorized-fit',epoch=ep)
        self.fit_cpu=time.process_time()-start

    def predict(self,x,method,support=None):
        if method=='prior':p=np.tile(self.prior,(len(x),1))
        elif method=='template':p=np.ones((len(x),128))
        elif method=='direct-table':p=np.array([self.table.get(row.tobytes(),np.zeros(128))+.5*self.prior for row in x])
        elif method in ('direct-mlp','factorized-mlp'):
            if method=='direct-mlp':
                h=np.tanh(features(x)@self.w1+self.b1);log=h@self.w2+self.b2
            else:
                fh=np.tanh(features(x)@self.f1+self.fb1);logit=fh@self.f2+self.fb2
                # Stable log-sigmoid probabilities: computing 1-sigmoid(40)
                # rounds to zero and can erase all legally allowed candidates.
                one=-np.logaddexp(0,-logit);zero=-np.logaddexp(0,logit)
                log=np.sum(np.where(LABELS[None,:,:],one[:,None,:],zero[:,None,:]),axis=2)
            if support is not None:log=np.where(support,log,-np.inf)
            p=np.exp(log-log.max(1,keepdims=True))
        elif method=='structured-joint':
            log=np.tile(np.log(self.prior),(len(x),1))
            for j in range(x.shape[1]):log+=np.log(self.conditionals[:,j,x[:,j]+1].T)
            if support is not None:log=np.where(support,log,-np.inf)
            p=np.exp(log-log.max(1,keepdims=True))
        else:raise ValueError('unknown method')
        if support is not None:p=p*support
        return normalize(p)

    def metadata(self):
        return dict(setting='categorical-mlp64-120-v1',epochs=self.epochs,parameters=int(self.w1.size+self.b1.size+self.w2.size+self.b2.size),
            table_cells=len(self.table),generative_parameters=int(self.conditionals.size+128),fit_cpu_seconds=self.fit_cpu,
            factorized_parameters=int(self.f1.size+self.fb1.size+self.f2.size+self.fb2.size),
            factorization='separately trained independent-bit MLP; width chosen to match total joint-MLP parameter count before fitting',
            fairness='Same labels and evidence; all candidate support equal. Parameter/storage/compute differ and are reported; no capacity-matched advantage asserted.')

def score(p,y):
    # Score the exact forecast that is retained and independently checked.
    # Renormalizing here can move a value by one ULP across the fixed 0.9
    # confidence boundary even when its row already sums to one within rounding.
    p=np.asarray(p,float)
    if p.ndim!=2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-10):
        raise ValueError('score requires normalized forecast')
    q=p[np.arange(len(y)),y];mode=p.argmax(1)
    marginal=p@LABELS;target=LABELS[y]
    return dict(log_loss=-np.log(np.maximum(q,1e-12)),brier=(p*p).sum(1)-2*q+1,correct=(mode==y).astype(float),
        tool_brier=(marginal[:,0]-target[:,0])**2,goal_brier=((marginal[:,-2:]-target[:,-2:])**2).mean(1),
        order_brier=(marginal[:,3]-target[:,3])**2,inspection_brier=(marginal[:,4]-target[:,4])**2,
        unsupported_attribution=((p.max(1)>=.9)&(mode!=y)).astype(float),confidence=p.max(1),zero_truth_probability=(q==0).astype(float))
