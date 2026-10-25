"""W9: exact hierarchical beta integration; beliefs and exit decisions separated."""
import math
import numpy as np
from scipy.special import roots_jacobi
from .common import rng,record,close

def beta_integral(a,b,likelihoods):
    # Positive Bernstein expansion; independent of Gaussian quadrature.
    coeff=[1.]
    for row in likelihoods:
        left=.75*float(row[0])+.25*float(row[2]);right=float(row[1]);next_coeff=[0.]*(len(coeff)+1)
        for i,c in enumerate(coeff):next_coeff[i]+=c*left;next_coeff[i+1]+=c*right
        coeff=next_coeff
    n=len(likelihoods);base=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    return math.fsum(c*math.exp(math.lgamma(a+i)+math.lgamma(b+n-i)-math.lgamma(a+b+n)-base) for i,c in enumerate(coeff))

def hierarchical(likelihoods,hyper,rates,independent=False):
    ll=np.asarray(likelihoods,float);ll=ll/ll.max(1)[:,None];a,b=hyper
    if independent:
        denominator=beta_integral(a,b,ll);pred=[]
        for i in range(len(ll)):
            num=ll.copy();num[i]*=rates;pred.append(beta_integral(a,b,num)/denominator)
        return np.array(pred)
    x,w=roots_jacobi(16,b-1,a-1);theta=(x+1)/2;w=w/w.sum();prior=np.stack([.75*(1-theta),theta,.25*(1-theta)],axis=1)
    local=prior[:,None,:]*ll[None,:,:];norm=local.sum(2);weight=w*np.prod(norm,axis=1);weight/=weight.sum()
    return np.sum(weight[:,None]*np.sum(local*rates,axis=2)/norm,axis=0)

def recovery(post,rates):
    p=post/post.sum()
    for n in range(201):
        if float(p@rates)<.2:return n
        p*=1-rates;p/=p.sum()
    return None

def block(setting,identity,independent=False):
    r=rng(*identity);eps=setting['epsilon'];rates=np.array([eps,1-eps,.5]);a,b=setting['hyperprior'];k=setting['k'];m=setting['m'];n=setting['n'];prior=np.array([.6,.2,.2])
    # Condition on the stipulated observed betrayal. Low-stake cooperations are
    # equally likely under every type and therefore cancel from this draw.
    for attempt in range(10000):
        theta=float(r.beta(a,b));def_theta=theta if setting['typical'] else float(r.beta(a,b));def_type=int(r.choice(3,p=[.75*(1-def_theta),def_theta,.25*(1-def_theta)]))
        if r.random()<rates[def_type]:break
    else:raise RuntimeError('conditional betrayal sampling bound')
    types=r.choice(3,k,p=[.75*(1-theta),theta,.25*(1-theta)]);observations=[(r.random(m)<rates[t]).astype(int).tolist() for t in types]
    local=np.array([rates**sum(obs)*(1-rates)**(len(obs)-sum(obs)) for obs in observations]);before=np.vstack([np.ones(3),local]);after=before.copy();after[0]=rates
    pre=hierarchical(before,[a,b],rates,independent);post=hierarchical(after,[a,b],rates,independent)
    individual=local*prior;individual/=individual.sum(1)[:,None];type_pred=individual@rates;count_pred=np.array([(1+sum(o))/(2+len(o)) for o in observations])
    out=[record(dict(primary=float(np.mean(post[1:]>.5)-np.mean(pre[1:]>.5)),before_fraction=float(np.mean(pre[1:]>.5)),after_fraction=float(np.mean(post[1:]>.5)),expected_cooperative_partners=float(np.sum(1-post[1:]))),reader='hierarchical')]
    for name,p in [('hierarchical',post[1:]),('individual_type',type_pred),('counting',count_pred)]:
        out.append(record(dict(expected_cooperative_partners=float(np.sum(1-p)),mean_defection=float(np.mean(p))),reader=name))
    def_post=prior*rates;def_post/=def_post.sum();count=(2/(n+3));type_probability=float(def_post@rates)
    out.append(record(dict(type_minus_count=type_probability-count,type_prediction=type_probability,count_prediction=count,cooperations_to_recover=recovery(def_post,rates)),reader='defector'))
    if not setting.get('omit_deception'):
        announced=np.array([.9,.05,.05]);audited_false=np.array([.01,.9,.5]);deception=prior*announced*rates*audited_false;deception/=deception.sum()
        out.append(record(dict(defection_prediction=float(deception@rates),cooperations_to_recover=recovery(deception,rates)),reader='announced_and_audited_deception'))
    for dependence in (0.,.5,.9):
        motivated=prior*rates**(1-dependence);motivated/=motivated.sum()
        out.append(record(dict(exact_prediction=type_probability,motivated_prediction=float(motivated@rates),exact_stay=float(1-2*type_probability>=-dependence),motivated_stay=float(1-2*float(motivated@rates)>=-dependence)),reader='dependence',dependence=dependence))
    return dict(public=dict(types=['benevolent','conditional_exploiter','mixed'],high_defection_rates=rates.tolist(),low_defection_rate=eps,hyperprior=[a,b],conditional_sampling='group and defector conditioned on one observed high-stakes defection; low-stakes cooperations uninformative about type',integrator='16-node Gauss-Jacobi exactly integrates polynomial degree at most9; independent positive Bernstein beta moments'),truth=dict(group_share=theta,defector_type=def_type,partner_types=types.tolist(),typical=setting['typical']),observations=dict(defector_low_cooperations=n,defector_high_defection=1,partners=observations),readouts=out)

def controls():
    # Rate-library quadrature matches a uniform Beta counting prior exactly.
    x,w=roots_jacobi(16,0,0);rate=(x+1)/2;w=w/w.sum()
    for defects,coops in ((0,0),(1,5),(3,12),(0,20)):
        weight=w*rate**defects*(1-rate)**coops;p=float(weight@rate/weight.sum())
        assert abs(p-(defects+1)/(defects+coops+2))<1e-12
    ll=np.array([[.05,.95,.5],[.95**2,.05**2,.5**2],[.05*.95,.95*.05,.25]])
    for hyper in ((.1,.4),(1,4),(4,16)):
        close(hierarchical(ll,hyper,np.array([.05,.95,.5])).tolist(),hierarchical(ll,hyper,np.array([.05,.95,.5]),True).tolist(),tol=1e-12)
    return dict(passed=True,stake_independent_beta_count_equivalence=True,independent_analytic_beta_integral=True)
