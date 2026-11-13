"""Matched cue alarm policies, including explicitly declared finite learners."""
from pathlib import Path
import math
import numpy as np
from numba import njit
from .common import rng,logistic,mean_blocks,censored_blocks,conditional_blocks,write,digest,audit_controls
NAMES=('TRIGGER','BRAKE','BOTH','TRIGGER learner','BRAKE learner','BOTH learner')


def settings():
    return [dict(truth=t,rho=r,steps=512,worlds=200,blocks=8) for t in range(6) for r in (.9,.7,.5,0.)]+[dict(truth=t,rho=1.,steps=512,worlds=200,blocks=8,known_answer=True) for t in range(3)]


def cue_law(rho):return np.array([.49+.21*rho,.21*(1-rho),.21*(1-rho),.09+.21*rho])


def probabilities(safety_absent,forecast,counts,harm,s):
    n=len(forecast);w=np.arange(n)
    trigger=(1+harm[w,0,forecast])/(10+counts[w,0,forecast])
    brake=(9+harm[w,1,safety_absent])/(10+counts[w,1,safety_absent])
    signal=np.column_stack([forecast,safety_absent,np.minimum(1,forecast+safety_absent),np.minimum(1,2*trigger),np.minimum(1,2*brake),np.minimum(1,2*trigger+2*brake)])
    return logistic(s.get('slope',4.)*(signal-.5),s.get('tremble',.02))


def update(counts,harm,safety_absent,forecast,outcome):
    w=np.arange(len(forecast))
    for view,cue in ((0,forecast),(1,safety_absent)):
        counts[w,view,cue]+=1;harm[w,view,cue]+=outcome


def generate(s,lane='study',worlds=None):
    n=worlds or s['worlds'];r=rng('B2',lane,{k:v for k,v in s.items() if k!='reminder_step'} if s.get('paired_reminder') else s);steps=s['steps'];cues=r.choice(4,size=(n,steps),p=cue_law(s['rho']));safety=cues//2;forecast=cues%2
    alarms=np.empty((n,steps),np.int8);outcomes=np.empty_like(alarms);counts=np.zeros((n,2,2));harm=np.zeros_like(counts)
    for t in range(steps):
        p=probabilities(safety[:,t],forecast[:,t],counts,harm,s)[:,s['truth']];alarms[:,t]=r.random(n)<p
        rate=np.where(forecast[:,t],.5,.02)
        if s.get('extinction_start') is not None and t>=s['extinction_start']:rate=np.zeros(n)
        outcome=r.random(n)<rate
        if s.get('reminder_step')==t:outcome=np.ones(n,bool)
        outcomes[:,t]=outcome;update(counts,harm,safety[:,t],forecast[:,t],outcome)
    return dict(cues=cues.astype(np.int8),alarms=alarms,harm=outcomes)


@njit(cache=True)
def reference(cues,alarms,outcomes,truth,slope,floor):
    n,steps=cues.shape;ll=np.zeros((n,6));last=np.full((n,6),-1);early=np.full((n,6),min(100,steps)+1);same=np.ones((n,6),np.bool_)
    for w in range(n):
        counts=np.zeros((2,2));harm=np.zeros((2,2))
        for t in range(steps):
            safety=cues[w,t]//2;forecast=cues[w,t]%2
            trigger=(1+harm[0,forecast])/(10+counts[0,forecast]);brake=(9+harm[1,safety])/(10+counts[1,safety])
            signal=np.array([float(forecast),float(safety),float(min(1,forecast+safety)),min(1.,2*trigger),min(1.,2*brake),min(1.,2*(trigger+brake))]);p=np.empty(6)
            for j in range(6):
                p[j]=min(1-floor,max(floor,1/(1+math.exp(-slope*(signal[j]-.5)))))
                ll[w,j]+=math.log(p[j] if alarms[w,t] else 1-p[j])
            for j in range(6):
                if ll[w,truth]-ll[w,j]<3:last[w,j]=t
                if abs(p[truth]-p[j])>=1e-12:same[w,j]=False
            if t==min(100,steps)-1:early[w]=last[w]+2
            counts[0,forecast]+=1;counts[1,safety]+=1;harm[0,forecast]+=outcomes[w,t];harm[1,safety]+=outcomes[w,t]
        same[w,truth]=False
    return ll,last+2,early,same.astype(np.float64)


def infer(raw,s,reference_mode=False):
    n,steps=raw['cues'].shape;truth=s['truth']
    if reference_mode:ll,pair,early,same=reference(raw['cues'],raw['alarms'],raw['harm'],truth,s.get('slope',4.),s.get('tremble',.02))
    else:
        counts=np.zeros((n,2,2));harm=np.zeros_like(counts);ll=np.zeros((n,6));last=np.full((n,6),-1);same=np.ones((n,6),bool)
        for t in range(steps):
            safety=raw['cues'][:,t]//2;forecast=raw['cues'][:,t]%2;p=probabilities(safety,forecast,counts,harm,s)
            ll+=np.where(raw['alarms'][:,t,None],np.log(p),np.log1p(-p));last=np.where(ll[:,truth,None]-ll<3,t,last);same &= abs(p[:,truth,None]-p)<1e-12
            if t==min(100,steps)-1:early=last+2
            update(counts,harm,safety,forecast,raw['harm'][:,t])
        pair=last+2;same[:,truth]=False;same=same.astype(float)
    state_counts=np.stack([(raw['cues']==j).sum(1) for j in range(4)],axis=1);alarms=np.stack([((raw['cues']==j)*raw['alarms']).sum(1) for j in range(4)],axis=1)
    return dict(separation=np.max(np.delete(pair,truth,axis=1),axis=1),pair_separation=pair,early_separation=np.max(np.delete(early,truth,axis=1),axis=1),early_pair_separation=early,final_likelihood=ll,same_probability=same,early_alarm_rate=raw['alarms'][:,:min(100,steps)].mean(1),alarm_rate=raw['alarms'].mean(1),telling_state_share=np.mean((raw['cues']==1)|(raw['cues']==2),axis=1),state_counts=state_counts,state_alarms=alarms)


def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);roster=settings() if selected is None else selected;rows=[];files={}
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:raw=generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        out=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**out)
        rows.append(dict(setting=s,measure='separation',**censored_blocks(out['separation'],s['steps'],s['blocks'])))
        rows.append(dict(setting=s,measure='first_hundred_record_separation',**censored_blocks(out['early_separation'],min(100,s['steps']),s['blocks'])))
        for j in range(6):
            if j!=s['truth']:
                rows.append(dict(setting=s,measure='pair_separation',rival=NAMES[j],**censored_blocks(out['pair_separation'][:,j],s['steps'],s['blocks'])))
                rows.append(dict(setting=s,measure='early_pair_separation',rival=NAMES[j],**censored_blocks(out['early_pair_separation'][:,j],min(100,s['steps']),s['blocks'])))
        for key in ('early_alarm_rate','alarm_rate','telling_state_share'):rows.append(dict(setting=s,measure=key,**mean_blocks(out[key],s['blocks'])))
        for j in range(4):rows.append(dict(setting=s,measure='alarm_rate_given_cue_world_present',cue=j,**conditional_blocks(out['state_alarms'][:,j]/np.maximum(1,out['state_counts'][:,j]),out['state_counts'][:,j]>0,s['blocks'])))
        if s.get('extinction_start') is not None:
            for phase,lo,hi in (('before_extinction',128,256),('late_extinction',384,512),('after_reminder',513,640)):
                values=raw['alarms'][:,lo:min(hi,s['steps'])].mean(1)
                rows.append(dict(setting=s,measure='phase_alarm_rate',phase=phase,**mean_blocks(values,s['blocks'])))
        pulse(family='B2',cell=i+1,cells=len(roster))
    result=dict(family='B2',settings=roster,rows=rows,no_registered_claim=True,learner_note='Declared Beta outcome learners; calm/alarmed priors and cue partition differ by construction.')
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result


def controls(destination,pulse=lambda **kw:None):
    checks=[]
    for rho in (0.,.5,1.):
        p=cue_law(rho);checks.append(dict(name=str(rho)+' exact joint cue marginals',passed=abs(p.sum()-1)<1e-14 and abs(p[1]+p[3]-.3)<1e-14 and abs(p[2]+p[3]-.3)<1e-14 and abs((p[3]-.09)/.21-rho)<1e-14))
    cue=np.array([0,1]);zeros=np.zeros((2,2,2));p=probabilities(cue,cue,zeros,zeros,{})
    checks.append(dict(name='rho-one fixed laws exactly equal',passed=bool(np.array_equal(p[:,0],p[:,1])) and bool(np.array_equal(p[:,1],p[:,2]))))
    checks.append(dict(name='initial trigger calm and brake alarmed',passed=bool(np.all(p[:,3]<.5)) and bool(np.all(p[:,4]>.5))))
    for truth in (0,3,4,5):
        s=dict(truth=truth,rho=.5,steps=128,worlds=8,blocks=8);raw=generate(s,'known-answer');a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=str(truth)+' independent complete path',passed=all(np.allclose(a[k],b[k],rtol=0,atol=1e-9) for k in a)))
    s=dict(truth=0,rho=1.,steps=32,worlds=8,blocks=8);raw=generate(s,'alias');a=infer(raw,s)
    checks.append(dict(name='exact fixed aliases remain unresolved',passed=bool(np.all(a['pair_separation'][:,1:3]==33)) and bool(np.all(a['same_probability'][:,1:3]==1))))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result