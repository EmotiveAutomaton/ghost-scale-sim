"""Frozen full-record approximation checks, with exact finite-state restrictions."""
import time
import numpy as np
from .common import rng,write,save_npz
from .w_settings import simulate,hidden_path,softmax,BASE

def exact_memoryless(path,s,theta):
    assert theta[0]==1 and (s['view']=='V1' or (theta[2]==0 and theta[3]==0 and theta[4]==0))
    v=s['volatility'];r=s['reliability'];ts=theta[1];posterior=np.array([1.,0,0,0]);old=0;ll=0.;values=[];plan=0
    for t,row in enumerate(path):
        prior=posterior*(1-v)+(1-posterior)*v/3;emission=np.zeros(4)
        for context in range(4):
            for cue in range(4):
                like=np.full(4,(1-r)/3);like[cue]=r;b=like**2;b/=b.sum();z=np.log(b);z[old]+=1
                if t%500<s.get('blackout',0):z=np.zeros(4);z[old]=1.
                emission[context]+=(r if cue==context else (1-r)/3)*softmax(z/ts)[int(row[4])]
        mass=prior@emission;ll+=np.log(mass);posterior=prior*emission/mass;old=int(row[4])
        if s['view']=='V3':
            if plan==0:
                prob=.1*s['actionable'];ll+=np.log(prob if row[7] else 1-prob)
                if row[7]:plan=4
            elif not row[7]:raise ValueError('impossible plan observation')
            ll-=np.logaddexp(0,-1 if row[5] else 1)
            if row[5]:
                ll-=.5*(row[6]/.3)**2
                if row[7]:ll-=np.logaddexp(0,3 if row[8] else -3)
                if row[8]:plan=0
                elif plan:plan-=1
        values.append(ll)
    return np.array(values)

def validate(s,dest,pulse=lambda **kw:None):
    dest.mkdir(parents=True,exist_ok=True);rows=[];start=time.process_time();lo=s['low_particles'];hi=s['high_particles']
    # Stop on a failed stage, retaining it. Full-length admission requires every
    # listed stage, never extrapolation from a successful short prefix.
    for steps in (128,1024,5000):
        for fixture,theta in [('exact',[1.,.35,0.,0.,0.]),('base',BASE.tolist()),('perturbed',[.6,.5,1.5,1.2,1.])]:
            g=rng('W-hidden-validation',s,steps,fixture);z=g.random((steps,11));z[:,10]=g.normal(size=steps)
            path=simulate(z,s['volatility'],s['reliability'],s['actionable'],np.array(theta),s.get('blackout',0));save_npz(dest/f'{steps}-{fixture}_points.npz',dict(records=path))
            a=hidden_path(path,dict(s,particles=lo),theta,31031);b=hidden_path(path,dict(s,particles=hi),theta,41031)
            ref=hidden_path(path,dict(s,particles=lo),theta,31031,True)
            row=dict(steps=steps,fixture=fixture,resolution_error=float(np.max(abs(a-b))),reconstruction_error=float(np.max(abs(a-ref))))
            if fixture=='exact':row['exact_error']=float(np.max(abs(b-exact_memoryless(path,s,theta))))
            row['passed']=row['resolution_error']<=.25 and row['reconstruction_error']<=1e-8 and row.get('exact_error',0)<=.25;rows.append(row)
            write(dest/'PROGRESS_VALIDATION.json',rows,immutable=False);pulse(phase='hidden_validation',steps=steps,fixture=fixture)
            if not row['passed']:return dict(passed=False,rows=rows,cpu_seconds=time.process_time()-start,production_admitted=False)
    return dict(passed=True,rows=rows,production_particles=hi,cpu_seconds=time.process_time()-start)
