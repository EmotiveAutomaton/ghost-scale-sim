"""Finite C02 parameter-frontier comparison; no generator identities in fitting."""
import time
import numpy as np
from . import world as W, oracle as O
from ..v20.readers import features

KINDS=('direct','autoregressive-forward','autoregressive-reverse','pairwise','low-rank')
PAIRS=tuple((i,j) for i in range(7) for j in range(i+1,7))
PHI=np.column_stack([W.LABELS]+[W.LABELS[:,i]*W.LABELS[:,j] for i,j in PAIRS])


def head(output,kind,y=None):
    """Return normalized joint log probabilities and mean-loss output gradient."""
    n=len(output);bits=W.LABELS[:,::-1] if kind.endswith('reverse') else W.LABELS
    if kind=='direct':
        lp=O.normalize_log(output)
        grad=np.exp(lp)
        if y is not None:grad[np.arange(n),y]-=1
    elif kind=='pairwise':
        lp=O.normalize_log(output@PHI.T)
        grad=np.exp(lp)@PHI
        if y is not None:grad-=PHI[y]
    elif kind.startswith('autoregressive'):
        logits=np.broadcast_to(output[:,None,:7],(n,128,7)).copy()
        for k,(i,j) in enumerate(PAIRS):logits[:,:,j]+=output[:,None,7+k]*bits[None,:,i]
        lp=np.sum(bits[None,:,:]*logits-np.logaddexp(0,logits),axis=2)
        grad=np.zeros_like(output)
        if y is not None:
            chosen=logits[np.arange(n),y];delta=np.exp(-np.logaddexp(0,-chosen))-bits[y]
            grad[:,:7]=delta
            for k,(i,j) in enumerate(PAIRS):grad[:,7+k]=delta[:,j]*bits[y,i]
    elif kind=='low-rank':
        logits=output[:,4:].reshape(n,4,7)
        components=np.einsum('nrb,kb->nrk',logits,W.LABELS)-np.logaddexp(0,logits).sum(2)[:,:,None]
        mixture=O.normalize_log(output[:,:4]);lp=O.logsumexp(components+mixture[:,:,None],1)
        grad=np.zeros_like(output)
        if y is not None:
            evidence=components[np.arange(n),:,y]+mixture
            responsibilities=np.exp(O.normalize_log(evidence))
            grad[:,:4]=np.exp(mixture)-responsibilities
            grad[:,4:]=(responsibilities[:,:,None]*(np.exp(-np.logaddexp(0,-logits))-W.LABELS[y,None,:])).reshape(n,28)
    else:raise ValueError(kind)
    return lp,grad/n if y is not None else None


class JointReader:
    def __init__(self,dimension,kind,budget,fit_seed=0):
        self.kind=kind;self.budget=budget
        outputs=128 if kind=='direct' else 32 if kind=='low-rank' else 28
        self.width=max(1,(budget-outputs)//(dimension+outputs+1))
        rng=np.random.default_rng(W.seed('v21-c02',kind,budget,fit_seed))
        self.params=[rng.normal(0,.1,(dimension,self.width)),np.zeros(self.width),rng.normal(0,.03,(self.width,outputs)),np.zeros(outputs)]
        self.fit_seed=fit_seed

    def forward(self,z):
        a,b,c,d=self.params;hidden=np.tanh(z@a+b)
        return head(hidden@c+d,self.kind)[0]

    def gradient(self,z,y,penalty=1e-4):
        a,b,c,d=self.params;hidden=np.tanh(z@a+b);lp,delta=head(hidden@c+d,self.kind,y)
        dh=(delta@c.T)*(1-hidden*hidden)
        loss=-lp[np.arange(len(y)),y].mean()+penalty/2*((a*a).sum()+(c*c).sum())
        return loss,[z.T@dh+penalty*a,dh.sum(0),hidden.T@delta+penalty*c,delta.sum(0)]

    def fit(self,x,y,vx,vy,max_epochs=120,pulse=lambda **k:None):
        started=time.process_time();z=features(x);vz=features(vx)
        rng=np.random.default_rng(W.seed('v21-c02-batches',self.fit_seed))
        first=[p.copy() for p in self.params];best=float('inf');best_params=first;stale=0;steps=0;curve=[]
        m=[np.zeros_like(p) for p in self.params];v=[np.zeros_like(p) for p in self.params]
        for epoch in range(max_epochs):
            order=rng.permutation(len(y))
            for offset in range(0,len(y),256):
                ids=order[offset:offset+256];_,grads=self.gradient(z[ids],y[ids]);steps+=1
                for i,(p,g) in enumerate(zip(self.params,grads)):
                    m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
                    p-=.003*(m[i]/(1-.9**steps))/(np.sqrt(v[i]/(1-.999**steps))+1e-8)
            # Bound autoregressive prediction memory independently of validation size.
            losses=[]
            for offset in range(0,len(vy),256):
                labels=vy[offset:offset+256];lp=self.forward(vz[offset:offset+256]);losses.extend(-lp[np.arange(len(labels)),labels])
            loss=float(np.mean(losses));curve.append(dict(epoch=epoch+1,validation_log_loss=loss))
            if loss<best-1e-4:best=loss;best_params=[p.copy() for p in self.params];stale=0
            else:stale+=1
            pulse(phase='joint-architecture-fit',kind=self.kind,budget=self.budget,fit=self.fit_seed,epoch=epoch+1)
            if stale>=10:break
        self.params=best_params
        self.metadata=dict(kind=self.kind,target_parameter_budget=self.budget,width=self.width,parameters=sum(p.size for p in self.params),fit_seed=self.fit_seed,fit_cpu_seconds=time.process_time()-started,validation_curve=curve,early_stopped=stale>=10,convergence_disposition='VALIDATION STOP' if stale>=10 else 'CAP REACHED; unresolved',validation_log_loss=best,training_labels=len(y),validation_labels=len(vy),max_epochs=max_epochs,optimizer='Adam',learning_rate=.003,penalty=1e-4,batch_size=256,patience=10,minimum_improvement=1e-4)
        return self

    def predict(self,x,support=None):
        z=features(x);lp=np.concatenate([self.forward(z[o:o+256]) for o in range(0,len(x),256)])
        return O.normalize_log(np.where(support,lp,-np.inf)) if support is not None else lp

    def save(self,path):np.savez_compressed(path,**{f'parameter_{i}':p for i,p in enumerate(self.params)})
