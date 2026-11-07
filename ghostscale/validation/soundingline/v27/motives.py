"""K2: exact temporal noisy-OR inference versus static loops and cue learners.

DESIGN CHECK: contexts must vary within targets; constant-context maps are labelled.
Zero RAGE removes the graph's loop and BP must equal exact static inference.
The independent exact route enumerates all joint dislike states; the fast route
uses their conditional independence. No scientific seed enters design checks.
"""
import itertools,math
import numpy as np
from numba import njit
from scipy.special import expit
from .common import rng,measure,sustained,close


def act_probability(d,r,context,pay=0,rho=.4):
    return 1-(1-.02)*(1-.6*context)*(1-.3*d)*(1-rho*r)*(1-.4*pay)


def likelihood(context,act,pay,rho=.4):
    rage=np.array([.1,.6]);p=np.array([(1-rage[d])*act_probability(d,0,context,pay,rho)+rage[d]*act_probability(d,1,context,pay,rho) for d in (0,1)])
    return p if act else 1-p


def generate(s,identity):
    r=rng('K2-world',identity);j=s['targets'];d=r.random(j)<.3;initial=d.copy();events=[]
    for t in range(s['records']):
        target=t%j;context=int(r.random()<s['tournament']);pay=int(s.get('payment',False) and r.random()<.5)
        rage=int(r.random() < (.6 if d[target] else .1));act=int(r.random()<act_probability(int(d[target]),rage,context,pay,s.get('rho',.4)))
        changed=bool(act and not d[target] and r.random()<s['phi']);d[target]|=changed
        events.append([target,context,act,pay,int(rage),int(d[target])])
    return dict(initial=initial.astype(int).tolist(),final=d.astype(int).tolist(),events=events)


def exact(events,j,phi,rho=.4,independent=False):
    out=[]
    if independent:
        states=np.array(list(itertools.product((0,1),repeat=j)),int)
        p=np.prod(np.where(states,.3,.7),axis=1)
        for target,context,act,pay,*_ in events:
            p*=likelihood(context,act,pay,rho)[states[:,target]];p/=p.sum()
            if act and phi:
                nxt=p.copy()
                for i,state in enumerate(states):
                    if state[target]==0:
                        k=i+(1<<(j-target-1));nxt[k]+=phi*p[i];nxt[i]-=phi*p[i]
                p=nxt
            out.append(p@states)
    else:
        p=np.full(j,.3)
        for target,context,act,pay,*_ in events:
            l=likelihood(context,act,pay,rho);q=p[target]*l[1]/((1-p[target])*l[0]+p[target]*l[1])
            p[target]=q+(1-q)*phi if act else q;out.append(p.copy())
    return np.array(out)


@njit
def bp_target(contexts,acts,pay,rho,power,maxiter=200,tolerance=1e-12):
    n=len(acts)
    if n==0:return .3,True,0
    # Two pair factors per encounter: P(r|d) and P(act|r,d).
    factors=np.empty((2*n,2,2))
    for k in range(n):
        for d in range(2):
            pr=.6 if d else .1
            for rage in range(2):
                factors[2*k,d,rage]=pr if rage else 1-pr
                p=1-.98*(1-.6*contexts[k])*(1-.3*d)*(1-rho*rage)*(1-.4*pay[k])
                factors[2*k+1,d,rage]=p if acts[k] else 1-p
    fd=np.full((2*n,2),.5);fr=fd.copy();converged=False
    for iteration in range(maxiter):
        logprod=np.array([math.log(.7),math.log(.3)])
        for f in range(2*n):
            for d in range(2):logprod[d]+=power*math.log(max(fd[f,d],1e-300))
        nd=np.empty_like(fd);nr=np.empty_like(fr);change=0.
        for f in range(2*n):
            other=f+1 if f%2==0 else f-1
            vd=np.empty(2);vr=np.empty(2)
            for d in range(2):vd[d]=logprod[d]-power*math.log(max(fd[f,d],1e-300))
            top=max(vd[0],vd[1]);vd=np.exp(vd-top);vd/=vd.sum()
            for z in range(2):vr[z]=max(fr[other,z],1e-300)**power
            vr/=vr.sum()
            for d in range(2):nd[f,d]=factors[f,d,0]*vr[0]+factors[f,d,1]*vr[1]
            for z in range(2):nr[f,z]=factors[f,0,z]*vd[0]+factors[f,1,z]*vd[1]
            nd[f]/=nd[f].sum();nr[f]/=nr[f].sum()
            for z in range(2):
                nd[f,z]=.5*nd[f,z]+.5*fd[f,z];nr[f,z]=.5*nr[f,z]+.5*fr[f,z]
                change=max(change,abs(nd[f,z]-fd[f,z]),abs(nr[f,z]-fr[f,z]))
        fd=nd;fr=nr
        if change<tolerance:converged=True;break
    logodds=math.log(.3/.7)
    for f in range(2*n):logodds+=power*math.log(max(fd[f,1],1e-300)/max(fd[f,0],1e-300))
    p=1/(1+math.exp(-max(-700.,min(700.,logodds))))
    return p,converged,iteration+1


def analyze(raw,s,independent=False):
    events=raw['events'];j=s['targets'];truth=np.array(raw['final']);n=len(events)
    paths={'E':exact(events,j,s['phi'],s.get('rho',.4),independent)}
    paths.update({k:np.empty((n,j)) for k in ('B','C','H','K')})
    failures={'B':[],'C':[]};counts=np.zeros(j);punches=np.zeros(j);mean=np.zeros(j+3);cov=np.eye(j+3)
    for t,event in enumerate(events):
        target,context,act,pay,*_=event;counts[target]+=1;punches[target]+=act
        paths['H'][t]=(punches+.5)/(counts+1)
        x=np.zeros(j+3);x[:3]=[1,context,pay];x[3+target]=1
        pred=float(expit(x@mean));h=pred*(1-pred)*x
        variance=max(.05,pred*(1-pred));gain=cov@h/(variance+h@cov@h)
        mean+=gain*(act-pred);cov-=np.outer(gain,h@cov)
        paths['K'][t]=expit(math.log(.3/.7)+4*mean[3:])
        for k in range(j):
            observations=np.array([e[1:4] for e in events[:t+1] if e[0]==k],dtype=np.int64).reshape(-1,3)
            for reader,power in [('B',1.),('C',2.)]:
                p,ok,it=bp_target(observations[:,0],observations[:,1],observations[:,2],s.get('rho',.4),power)
                paths[reader][t,k]=p
                if t==n-1:failures[reader].append(int(not ok))
    stats={};log={}
    for reader,path in paths.items():
        correct=np.where(truth,path[-1],1-path[-1]);log[reader]=float(np.log(np.maximum(correct,1e-300)).sum())
        odds=(2*truth-1)[None,:]*(np.log(np.maximum(path,1e-300))-np.log(np.maximum(1-path,1e-300)))
        stats[reader]=dict(log_score=log[reader],half_brier=float(np.mean((path[-1]-truth)**2)),
            false_dislike=[int(np.sum((truth==0)&(path[-1]>.5))),int(np.sum(truth==0))],
            records=[sustained(odds[:,k]) for k in range(j)],nonconvergence=float(np.mean(failures[reader])) if reader in failures else 0.)
    stats['contrasts']=dict(primary=log['E']-log['H'],K_gap=log['E']-log['K'],
        K_gap_share=[log['E']-log['K'],log['E']-log['H']],
        false_dislike_excess=[stats['H']['false_dislike'][0]-stats['E']['false_dislike'][0],stats['E']['false_dislike'][1]])
    return stats


def block(s,identity,independent=False):
    raw=[];vals={};out=[]
    for k in range(s['makers']):
        o=generate(s,[identity,k]);raw.append(o)
        for reader,stats in analyze(o,s,independent).items():
            for metric,value in stats.items():
                values=vals.setdefault((reader,metric),[])
                if metric=='records':values.extend(value)
                else:values.append(value)
    for (reader,metric),values in vals.items():
        kind='median' if metric=='records' else 'ratio_of_sums' if metric in ('false_dislike','K_gap_share','false_dislike_excess') else 'mean'
        measure(out,metric,values,kind,reader=reader)
    return dict(observations=raw,measures=out)


def controls():
    events=[[0,0,a,0,0,0] for a in (1,0,1,0,0)]
    for phi in (0.,.3):close(exact(events,2,phi).tolist(),exact(events,2,phi,independent=True).tolist(),tol=1e-12)
    e=exact(events,1,0.,0.)[-1,0];b,ok,_=bp_target(np.zeros(5,dtype=int),np.array([1,0,1,0,0]),np.zeros(5,dtype=int),0.,1.)
    assert ok and abs(e-b)<1e-12
    l=np.prod([likelihood(0,a,0) for a in (1,0,1,0,0)],axis=0);closed=.3*l[1]/(.7*l[0]+.3*l[1])
    assert abs(exact(events,1,0)[-1,0]-closed)<1e-12
    return dict(passed=True,joint_enumeration_equals_factorized=True,zero_rage_tree_BP_equal=True,street_closed_form=True)


def design_check(s,identity):
    shares=[]
    for k in range(64):
        o=generate(s,[identity,'excluded',k]);yes=[]
        for j in range(s['targets']):
            contexts=[e[1] for e in o['events'] if e[0]==j];p=np.mean(contexts);yes.append(.25<=p<=.75)
        shares.append(np.mean(yes)>=.5)
    return dict(passed=float(np.mean(shares))>=.25,case_share=float(np.mean(shares)),floor=.25,worlds=64,
        expected_signs={'E_minus_H':'nonnegative in expectation under the declared true model; finite bar not a theorem',
            'E_minus_B':'nonnegative in expectation; loops may be inaccurate','E_minus_K':'nonnegative in expectation; magnitude open',
            'E_minus_C':'nonnegative in expectation; nonconvergence separately recorded'},scientific_draws=0)
