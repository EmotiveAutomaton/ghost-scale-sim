"""Disjoint-validation direct readers and a learned shared-episode latent model."""
import time
import numpy as np
from .world import seed
from .oracle import normalize_log,logsumexp
from ..v20.readers import features
from .audit import counts,forecast

class Direct:
    def __init__(self,x,y,vx,vy,fit_seed=0,width=64,max_epochs=120,patience=10,
                 pulse=lambda **kw:None,fresh=None):
        started=time.process_time();rng=np.random.default_rng(seed('v21-direct',fit_seed,width))
        z=features(x).astype(np.float32);vz=features(vx).astype(np.float32);prior=np.bincount(y,minlength=128)+.5;prior/=prior.sum()
        self.params=[rng.normal(0,.1,(z.shape[1],width)).astype(np.float32),np.zeros(width,np.float32),
                     rng.normal(0,.03,(width,128)).astype(np.float32),np.log(prior).astype(np.float32)]
        m=[np.zeros_like(p) for p in self.params];v=[np.zeros_like(p) for p in self.params]
        best=float('inf');best_params=None;stale=0;steps=0;generated=0;self.curve=[]
        for epoch in range(max_epochs):
            indices=rng.permutation(len(y))
            for offset in range(0,len(y),256):
                ids=indices[offset:offset+256];zz=z[ids];yy=y[ids]
                if fresh is not None:
                    xx,yy=fresh(len(ids),steps);zz=features(xx).astype(np.float32);generated+=len(yy)
                a,b,c,d=self.params;h=np.tanh(zz@a+b);lp=normalize_log((h@c+d).astype(float));pp=np.exp(lp).astype(np.float32)
                pp[np.arange(len(yy)),yy]-=1;pp/=len(yy);dh=(pp@c.T)*(1-h*h)
                grads=[zz.T@dh+1e-4*a,dh.sum(0),h.T@pp+1e-4*c,pp.sum(0)]
                steps+=1
                for i,(param,g) in enumerate(zip(self.params,grads)):
                    m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
                    param-=.003*(m[i]/(1-.9**steps))/(np.sqrt(v[i]/(1-.999**steps))+1e-8)
            loss=float(-self._logits(vz)[np.arange(len(vy)),vy].mean());self.curve.append(dict(epoch=epoch+1,validation_log_loss=loss))
            pulse(phase='converged-direct-fit' if fresh is None else 'fresh-law-direct-fit',epoch=epoch+1,validation_log_loss=loss)
            if loss<best-1e-4:best=loss;best_params=[p.copy() for p in self.params];stale=0
            else:stale+=1
            if stale>=patience:break
        self.params=best_params
        self.metadata=dict(reader='converged-direct' if fresh is None else 'fresh-law-direct',width=width,optimizer='minibatch Adam',
            learning_rate=.003,batch_size=256,max_epochs=max_epochs,patience=patience,early_stopped=stale>=patience,
            convergence_disposition='VALIDATION STOP' if stale>=patience else 'CAP REACHED; convergence unresolved',
            validation_curve=self.curve,validation_log_loss=best,fit_cpu_seconds=time.process_time()-started,
            unique_finite_labels=len(y),fresh_generated_labels=generated,parameters=sum(p.size for p in self.params))
    def _logits(self,z):
        a,b,c,d=self.params;return normalize_log((np.tanh(z@a+b)@c+d).astype(float))
    def predict(self,x,support=None):
        lp=self._logits(features(x).astype(np.float32))
        if support is not None:lp=normalize_log(np.where(support,lp,-np.inf))
        return lp
    def save(self,path):np.savez_compressed(path,**{f'parameter_{i}':p for i,p in enumerate(self.params)})

class SharedMaker:
    """Unsupervised episode clusters; latent identities are NEVER generator maker IDs.

    A finite mixture shares one state across all historical episodes and the joint
    current-observation/target. Different projections have separate vocabularies.
    """
    def __init__(self,x,y,fit_seed=0,states=32,iterations=40,pulse=lambda **kw:None):
        self.current={k:i for i,k in enumerate(sorted({r.tobytes() for r in x[:,:13]}))}
        h=x[:,13:].reshape(len(x),-1,13)
        self.history={k:i for i,k in enumerate(sorted({r.tobytes() for r in h.reshape(-1,13)}))}
        ci=self._indices(x[:,:13],self.current);hi=self._indices(h.reshape(-1,13),self.history).reshape(len(x),-1)
        nc=len(self.current)+1;nh=len(self.history)+1;atom=ci*128+y
        rng=np.random.default_rng(seed('v21-em',fit_seed));resp=rng.dirichlet(np.ones(states),len(y));self.curve=[]
        start=time.process_time()
        for iteration in range(iterations):
            self.prior=(resp.sum(0)+.5)/(len(y)+.5*states)
            self.joint=np.full((states,nc*128),.01);self.emission=np.full((states,nh),.05)
            for z in range(states):
                self.joint[z]+=np.bincount(atom,resp[:,z],minlength=nc*128)
                for step in range(hi.shape[1]):self.emission[z]+=np.bincount(hi[:,step],resp[:,z],minlength=nh)
            self.joint/=self.joint.sum(1,keepdims=True);self.emission/=self.emission.sum(1,keepdims=True)
            lp=np.log(self.prior)[None,:]+np.log(self.joint[:,atom].T)
            for step in range(hi.shape[1]):lp+=np.log(self.emission[:,hi[:,step]].T)
            self.curve.append(float(logsumexp(lp,1).mean()));resp=np.exp(normalize_log(lp))
            if iteration%5==0:pulse(phase='shared-maker-EM',iteration=iteration)
        self.joint=self.joint.reshape(states,nc,128)
        self.metadata=dict(reader='learned-shared-maker',latent_states=states,latent_truth_supplied=False,iterations=iterations,
            effective_states=float(np.exp(-np.sum(self.prior*np.log(self.prior)))),training_log_likelihood=self.curve,
            fit_cpu_seconds=time.process_time()-start,parameters=int(self.joint.size+self.emission.size+states))
    @staticmethod
    def _indices(rows,lookup):return np.array([lookup.get(r.tobytes(),len(lookup)) for r in rows])
    def predict(self,x,support=None):
        ci=self._indices(x[:,:13],self.current);h=x[:,13:].reshape(len(x),-1,13)
        hi=self._indices(h.reshape(-1,13),self.history).reshape(len(x),-1)
        lp=np.tile(np.log(self.prior),(len(x),1))
        for step in range(hi.shape[1]):lp+=np.log(self.emission[:,hi[:,step]].T)
        out=logsumexp(np.log(self.joint[:,ci,:].transpose(1,0,2))+lp[:,:,None],1)
        if support is not None:out=np.where(support,out,-np.inf)
        return normalize_log(out)

class GroupedTable:
    def __init__(self,x,y):
        self.prior=(np.bincount(y,minlength=128)+.5);self.prior/=self.prior.sum();self.table={}
        for row,label in zip(x,y):self.table.setdefault(self.key(row),np.zeros(128))[label]+=1
    @staticmethod
    def key(row):return (row[:13].tobytes(),tuple(sorted(r.tobytes() for r in row[13:].reshape(-1,13))))
    def predict(self,x,support=None):
        p=np.array([self.table.get(self.key(r),np.zeros(128))+.5*self.prior for r in x])
        if support is not None:p*=support
        with np.errstate(divide='ignore'):return normalize_log(np.log(p))
