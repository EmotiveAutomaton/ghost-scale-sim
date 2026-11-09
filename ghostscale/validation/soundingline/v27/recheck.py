"""K5: retained influence, fading tags and explicit hierarchical group pooling.

DESIGN CHECK: x=1 returns the prior and tau=0 returns frozen D. G3 trajectories
remain identical at delta=0. Group pooling integrates a common latent agenda and
independent source truths, never pretending alternating truths are one truth.
"""
import math
from fractions import Fraction
from functools import lru_cache
from decimal import Decimal,localcontext
import numpy as np
from scipy.special import expit,logsumexp
from scipy.stats import binom
from ..v26 import day_closure as old
from .common import rng,measure,close


def library(strengths,honest=.5):
    source=[(-1,0.,honest)]+[(a,b,(1-honest)/(2*len(strengths))) for b in strengths for a in (1,0)]
    z=np.array([v for v in (0,1) for _ in source]);agenda=np.array([a for _ in (0,1) for a,_,_ in source])
    p=np.array([old.rate(v,a,b) for v in (0,1) for a,b,_ in source]);prior=np.array([w/2 for _ in (0,1) for _,_,w in source])
    return z,agenda,p,prior


def weakened(prob,fraction):
    if fraction==1:return .5
    if prob in (0.,1.):return prob
    return float(expit((1-fraction)*(math.log(prob)-math.log1p(-prob))))


@lru_cache(maxsize=8192)
def exact_unbiased_harm(ones,zeros,strengths):
    """Rational count-state check before the first non-Bayesian boost."""
    total=Fraction(0);harm=Fraction(0)
    for truth in (0,1):
        q=Fraction(7 if truth else 3,10)
        for agenda,b,weight in [(-1,Fraction(0),Fraction(1,4))]+[
                (a,Fraction(str(b)),Fraction(1,8*len(strengths))) for b in strengths for a in (1,0)]:
            p=q if agenda<0 else (1-b)*q+b*((1-(1-q)**8) if agenda==1 else q**8)
            value=weight*p**ones*(1-p)**zeros
            total+=value
            if agenda>=0 and agenda!=truth:harm+=value
    return harm/total


def no_boost_before_end(items,strengths,theta):
    ones=zeros=0;threshold=Fraction(str(theta));strengths=tuple(strengths)
    for item in items:
        ones+=item==1;zeros+=item==0
        if exact_unbiased_harm(int(ones),int(zeros),strengths)>=threshold:return False
    return True


def exact_balanced_bias(items,strengths,theta,delta):
    """Count signed boosts with high precision; equal boosts cancel exactly.

    Balanced observations alone do not imply a tie after bias. The existing
    agenda tie rule (difference > 1e-12) is retained without widening it.
    """
    with localcontext() as ctx:
        ctx.prec=70
        D=lambda value:Decimal(str(value))
        states=[];weights=[]
        for truth in (0,1):
            q=D('.7') if truth else D('.3')
            states.append((truth,-1,q));weights.append(D('.25'))
            for strength in strengths:
                b=D(strength)
                for agenda in (1,0):
                    p=(1-b)*q+b*((1-(1-q)**8) if agenda==1 else q**8)
                    states.append((truth,agenda,p));weights.append(D(1)/D(8*len(strengths)))
        signed=0;boost=D(delta).exp()
        for item in items:
            if item>=0:weights=[w*(p if item else 1-p) for w,(_,_,p) in zip(weights,states)]
            total=sum(weights);weights=[w/total for w in weights]
            harm=sum(w for w,(truth,agenda,_) in zip(weights,states) if agenda>=0 and agenda!=truth)
            if delta>0 and harm>=D(theta):
                diff=sum(w for w,(_,a,_) in zip(weights,states) if a==1)-sum(w for w,(_,a,_) in zip(weights,states) if a==0)
                favored=1 if diff>D('1e-12') else 0
                signed+=1 if favored else -1
                weights=[w*boost if truth==favored else w for w,(truth,_,_) in zip(weights,states)]
        return signed==0


def pooled(items,strengths,honest=.5,independent=False):
    # Conditional on one shared agenda, each source has its own truth and kind.
    n=len(items);conditional=np.empty((2,n,2));evidence=np.empty((2,n))
    for agenda in (0,1):
        for i,row in enumerate(items):
            ones=int(np.sum(row==1));zeros=int(np.sum(row==0));ll=[]
            for truth in (0,1):
                weights=[honest]+[(1-honest)/len(strengths)]*len(strengths)
                ps=[old.rate(truth)]+[old.rate(truth,agenda,b) for b in strengths]
                terms=np.array([math.log(w)+ones*math.log(p)+zeros*math.log1p(-p) for w,p in zip(weights,ps)])
                if independent:
                    top=float(max(terms));value=top+math.log(math.fsum(math.exp(float(x)-top) for x in terms))
                else:value=float(logsumexp(terms))
                ll.append(value+math.log(.5))
            norm=float(logsumexp(ll));conditional[agenda,i]=np.exp(np.array(ll)-norm);evidence[agenda,i]=norm
    total=evidence.sum(1);a=np.exp(total-logsumexp(total))
    return np.einsum('a,aiz->iz',a,conditional)


def pooled_worlds(items,sources,strengths,honest=.5,independent=False):
    if len(items)%sources:raise ValueError('incomplete world group')
    return np.vstack([pooled(items[i:i+sources],strengths,honest,independent) for i in range(0,len(items),sources)])


def generate(s,identity):
    r=rng('K5-world',identity);rows=[]
    legacy=s|dict(overpush=1.,expanded=False,pull_away=False)
    for source in ('H','M','Mx','Mc','HU'):
        truth,items=old.observations(legacy,r,source)
        warning=r.random(len(truth))<np.where((source in ('M','Mx','Mc'))&(truth==0),s['warning'],1-s['warning'])
        retrieval=r.random((len(truth),s['records']))<.5
        rows.append(dict(source=source,truth=truth.tolist(),items=items.tolist(),warning=warning.astype(int).tolist(),retrieval=retrieval.astype(int).tolist()))
    return rows


def analyze(raw,s,independent=False):
    vals={};lib=library(s['strengths']);kernel=old.review_trajectories if independent else old.trajectory
    for row in raw:
        name=row['source'];code=['H','M','Mx','Mc','HU'].index(name);truth=np.array(row['truth']);items=np.array(row['items'],dtype=np.int8)
        v,gate=kernel(items,truth,code,s['theta'],s['delta'],s['switch'],1.,False,*lib)
        # Exact symmetry, not a tolerance around the scientific threshold:
        # equal 0/1 counts and the symmetric library give P(truth)=1/2.
        for i in range(len(items)):
            retained=items[i,:gate[i]] if gate[i]>=0 else items[i]
            if np.sum(retained==0)==np.sum(retained==1):v[i,1]=math.log(.5);v[i,5]=.5
            if (np.sum(items[i]==0)==np.sum(items[i]==1) and
                    (s['delta']==0 or exact_balanced_bias(items[i],s['strengths'],s['theta'],s['delta']))):
                v[i,0]=math.log(.5);v[i,4]=.5
        z,agenda,prob,prior=lib;wout=np.empty_like(v)
        for warning in (0,1):
            mask=np.array(row['warning'])==warning
            if not np.any(mask):continue
            reliability=s['warning'];harm=(agenda>=0)&(agenda!=z)
            diagnostic=np.where(harm,reliability,1-reliability) if warning else np.where(harm,1-reliability,reliability)
            wp=prior*diagnostic;wp/=wp.sum()
            wout[mask],_=kernel(items[mask],truth[mask],code,s['theta'],s['delta'],s['switch'],1.,False,z,agenda,prob,wp)
        for true in (0,1):
            mask=truth==true;rr=[];rt=[];rt_before=[];cost=[];r_reset=[]
            for i in np.flatnonzero(mask):
                d=float(math.exp(v[i,1]));when=int(gate[i]);closed=when>=0
                rr.append(1-weakened(d,s['x']) if closed else 1-d);cost.append(when if closed else 0)
                age=max(0,s['records']-when);strength=s['tau']*math.exp(-s['zeta']*age)
                active=not s['context_switch'] or s['records']<=48 or bool(row['retrieval'][i][-1])
                rt.append(1-weakened(d,strength if active else 0.) if closed else 1-d)
                rt_before.append(1-weakened(d,strength) if closed else 1-d)
                r_reset.append(.5 if closed else 1-d)
            rr=np.array(rr);rt=np.array(rt);dd=v[mask,5];kk=v[mask,4]
            stats=dict(K_capture=kk,D_capture=dd,R_capture=r_reset,RR_capture=rr,RT_capture=rt,
                RT_minus_RR=rt-rr,RT_context_loss=rt-np.array(rt_before),RR_reread_cost=cost,
                RR_captured=(rr>.5).astype(float),RT_captured=(rt>.5).astype(float),
                W_capture=wout[mask,4],W_capture_saved=kk-wout[mask,4],W_closed=wout[mask,6],
                D_closed=v[mask,6],K_captured=(kk>.5).astype(float),D_captured=(dd>.5).astype(float))
            for metric,value in stats.items():vals[(name,true,metric)]=np.asarray(value).tolist()
        if name=='Mx':
            # Each world has exactly `sources` members. Whole-block grouping is
            # only an uncertainty unit, never an extra common-agenda group.
            outsider=pooled_worlds(items,s['sources'],[.5],.5,independent)
            insider=pooled_worlds(items,s['sources'],s['strengths'] if len(s['strengths'])>1 else [.5,1.],.5+s['bias'],independent)
            wrong_out=1-outsider[np.arange(len(truth)),truth];wrong_in=1-insider[np.arange(len(truth)),truth]
            vals[('IG',0,'outsider_minus_insider')]=(wrong_out[truth==0]-wrong_in[truth==0]).tolist()
            vals[('IG',0,'outsider_capture')]=wrong_out[truth==0].tolist();vals[('IG',0,'insider_capture')]=wrong_in[truth==0].tolist()
    return vals


def block(s,identity,independent=False):
    raw=generate(s,identity);out=[]
    for (source,truth,metric),values in analyze(raw,s,independent).items():measure(out,metric,values,source=source,truth=truth)
    return dict(observations=raw,measures=out)


def controls():
    assert weakened(.1,1)==weakened(.9,1)==.5
    for p in (.1,.5,.9):assert abs(weakened(p,0)-p)<1e-12
    a=library([.5]);b=old.library(False)
    for x,y in zip(a,b):close(x.tolist(),y.tolist(),tol=1e-12)
    fixture=np.array([[0,1,1,0],[1,1,0,1]],dtype=np.int8)
    close(pooled(fixture,[.5,1.]).tolist(),pooled(fixture,[.5,1.],independent=True).tolist(),tol=1e-12)
    two_worlds=np.vstack([fixture,np.ones_like(fixture)])
    altered=np.vstack([fixture,np.zeros_like(fixture)])
    close(pooled_worlds(two_worlds,2,[.5])[:2].tolist(),pooled_worlds(altered,2,[.5])[:2].tolist(),tol=1e-12)
    known=old.exact('Mc');honest=old.exact('H')
    # Published exact values, not rounded Monte Carlo estimates.
    # Pinned published values: V26 G3 study CONTROLS.json and EXACT_REFERENCES.json.
    assert abs(known['D_minus_K_capture']-(-.0897098994900386))<1e-12
    assert abs(honest['closed']-.08990315282392991)<1e-12
    # The original failed probe had exactly fourteen 0s and fourteen 1s at
    # closure. Both unnormalized count-state likelihoods must be equal.
    z,agenda,p,prior=library([.5]);weights=prior*p**14*(1-p)**14
    assert abs(weights[z==0].sum()/weights.sum()-.5)<1e-12
    return dict(passed=True,full_reread_prior=True,zero_tag_D_identity=True,original_library_byte_equivalence=True,
        cross_world_intervention_invariant=True,balanced_evidence_symmetry=True,independent_group_marginal=True,exact_G3_con=known['D_minus_K_capture'],exact_G3_honest_closed=honest['closed'])


def design_check(s,identity):
    z,agenda,p,prior=library(s['strengths']);h=s['records'];first=min(h,s['switch'])
    mass=np.convolve(binom.pmf(np.arange(first+1),first,.3),binom.pmf(np.arange(h-first+1),h-first,old.rate(0,1,1.)))
    captured=0.
    for ones,weight in enumerate(mass):
        log=np.log(prior)+ones*np.log(p)+(h-ones)*np.log1p(-p);post=np.exp(log-logsumexp(log))
        if post[z==0].sum()<.5:captured+=weight
    return dict(passed=captured>=.25,case_share=float(captured),floor=.25,
        method='Exact count-state programme; con captures open reader on truth-0 sources',
        expected_signs={'H':'false-closure cost open','M':'library-dependent','Mx':'library frontier open','Mc':'RR/RT mean change small, bar not rescued','IG':'library and honesty-prior contrast; magnitude open'},scientific_draws=0)
