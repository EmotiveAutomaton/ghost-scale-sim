"""U1: exact selection-biased evidence with a shared latent source policy."""
import math
import numpy as np
from .common import rng,posterior,record,close

def presented(p,s,size=8):return (1-s)*p+s*(1-(1-p)**size)
def conditional(k,s,size=8):return (1-s)*k/size+s*float(k>0)

def block(setting,identity,independent=False):
    r=rng(*identity);s=setting['s'];tau=setting['tau'];claims=setting.get('claims',32);grid=np.array([0.,.5,1.]);lengths=setting.get('training_lengths',[1,2,4,8,16,32,64]);out=[];raw=[]
    if not lengths or len(set(lengths))!=len(lengths) or any(not isinstance(n,int) or not 1<=n<=64 for n in lengths):raise ValueError('invalid training grid')
    for maker in range(32):
        truth=maker%2;ptruth=.7 if truth else .3;track=[];lls=np.zeros(3);priors={}
        for t in range(64):
            k=int(r.binomial(8,ptruth));y=int(r.random()<conditional(k,s));prob=np.array([conditional(k,v) for v in grid])
            with np.errstate(divide='ignore'):lls+=np.log(np.where(y,prob,1-prob)) # exact impossible source strengths remain excluded
            track.append([k,y])
            if t+1 in lengths:priors[t+1]=posterior(lls,independent=independent)
        p=np.array([[presented(.3,v),presented(.7,v)] for v in grid]);trans=np.array([presented(.3,s),presented(.7,s)]);ll=np.zeros((3,2));lt=np.zeros(2);lo=np.zeros(2);seen=[];lm=np.zeros(2)
        for t in range(claims):
            k=int(r.binomial(8,ptruth));y=int(r.random()<conditional(k,s));seen.append([k,y]);ll+=np.log(p if y else 1-p);lt+=np.log(trans if y else 1-trans);lo+=tau*np.log(np.array([.3,.7]) if y else np.array([.7,.3]))
            # Menu arm: disclose four random pool items sampled without replacement.
            j=int(r.hypergeometric(k,8-k,4));lik=[]
            if setting.get('omit_menu'):seen[-1].append(j);continue
            for ph in (.3,.7):
                total=0.
                for kk in range(9):
                    if j<=kk and 4-j<=8-kk:
                        pool=math.comb(8,kk)*ph**kk*(1-ph)**(8-kk);sample=math.comb(kk,j)*math.comb(8-kk,4-j)/math.comb(8,4);py=conditional(kk,s);total+=pool*sample*(py if y else 1-py)
                lik.append(total)
            lm+=np.log(lik);seen[-1].append(j)
        pt=posterior(lt,independent=independent);po=posterior(lo,independent=independent);pm=posterior(lm,independent=independent)
        for n in lengths:
            joint=posterior(ll.reshape(-1),np.repeat(priors[n]/2,2),independent).reshape(3,2).sum(0)
            out.append(record(dict(capture=float(joint[1]),transparent_capture=float(pt[1]),opaque_capture=float(po[1]),menu_capture=float(pm[1]) if not setting.get('omit_menu') else None,excess_capture=float(joint[1]-pt[1]),label_difference=0.),maker=maker,truth=truth,reader='track-record',history=n))
        raw.append(dict(truth=truth,training=track,claims=seen,source_posteriors={str(k):v.tolist() for k,v in priors.items()}))
    return dict(public=dict(choice_model='8 iid .7/.3 items; positive-selection policy; source strength shared across claims',source_grid=grid.tolist(),menu='four random pool items plus selected claim; exact hypergeometric marginal'),truth_and_observations=raw,readouts=out)

def controls():
    for p in (.3,.7):
        assert presented(p,0)==p
        for s in (0,.5,1):
            enumerated=sum(math.comb(8,k)*p**k*(1-p)**(8-k)*conditional(k,s) for k in range(9));assert abs(enumerated-presented(p,s))<1e-12
    ll=np.log([.3,.7])*3+np.log([.7,.3])*2;close(posterior(ll).tolist(),posterior(ll,independent=True).tolist(),tol=1e-12)
    return dict(passed=True,unbiased_tau1_equality=True,zero_strength_identity=True,exact_pool_enumeration=True)

def design_check(settings):
    rows=[]
    for s in (.5,1.):
        a,b=presented(.7,s),presented(.3,s);share=sum(prob for ratio,prob in [(a/b,.5*(a+b)),((1-a)/(1-b),1-.5*(a+b))] if max(ratio/(.7/.3),(.7/.3)/ratio)>2)
        # Compare each observed sign to its own unbiased likelihood ratio.
        share=.5*(a+b)*float(max((a/b)/(.7/.3),(.7/.3)/(a/b))>2)+(1-.5*(a+b))*float(max(((1-a)/(1-b))/(.3/.7),(.3/.7)/((1-a)/(1-b)))>2)
        rows.append(dict(s=s,share=share,passed=share>=.25))
    share=sum(x['share'] for x in rows)/len(rows)
    return dict(passed=share>=.25,checks=rows,pooled_biased_source_share=share,seed_domain='exact-enumeration-U1',interpretation='Equal shares of the two declared biased strengths; the floor applies to the grid, not each strength.')
