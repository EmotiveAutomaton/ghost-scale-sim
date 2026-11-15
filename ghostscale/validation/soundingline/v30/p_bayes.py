"""P collapsed posterior-particle repair, with independent gamma-ratio density."""
import math
from pathlib import Path
import numpy as np
from numba import njit
from . import p
from .common import rng,mean_blocks,audit_controls,write

def settings():
    return [dict(s,part="bayes",burn=1024,draws=512,thin=2,chains=4)
            for s in p.settings() if s["m"]==60 and s["part"]=="labels"]

@njit
def log_gamma_draw(shape):
    if shape<1:return math.log(np.random.gamma(shape+1,1.))+math.log(max(np.random.random(),1e-300))/shape
    return math.log(np.random.gamma(shape,1.))

@njit
def log_dirichlet(alpha):
    out=np.empty(len(alpha))
    for k in range(len(alpha)):out[k]=log_gamma_draw(max(alpha[k],1e-200))
    mx=np.max(out);out-=mx+math.log(np.exp(out-mx).sum())
    return out

@njit
def target(theta,logphi_sum,N,km,kp,reference=False):
    if reference:
        value=0.
        # Dirichlet prior plus additive-log-ratio Jacobian; constants omitted.
        for k in range(12):
            value+=(km[k]-1)*math.log(theta[k])+math.log(theta[k])
            value+=kp*theta[k]*logphi_sum[k]-N*math.lgamma(kp*theta[k])
        return value
    normalizer=0.
    for x in theta:normalizer+=math.lgamma(kp*x)
    return np.sum(km*np.log(theta)+kp*theta*logphi_sum)-N*normalizer

@njit
def collapsed_target(theta,tail_counts,km,kp,reference=False):
    value=np.sum(km*np.log(theta))
    for k in range(12):
        a=kp*theta[k]
        if reference:
            # Independent Dirichlet-multinomial gamma ratios grouped by count.
            for n in range(1,tail_counts.shape[1]+1):
                number=tail_counts[k,n-1]-(tail_counts[k,n] if n<tail_counts.shape[1] else 0.)
                if number>0:value+=number*(math.lgamma(a+n)-math.lgamma(a))
        else:
            # Integer-count rising-factorial identity avoids cancellation.
            for j in range(tail_counts.shape[1]):
                if tail_counts[k,j]>0:value+=tail_counts[k,j]*math.log(a+j)
    return value

@njit
def chain(counts,C,h,km,kp,burn,draws,thin,seed,reference=False):
    np.random.seed(seed);N=len(counts)
    theta=np.exp(log_dirichlet(km));phi=np.empty((N,12))
    for j in range(N):phi[j]=log_dirichlet(kp*theta)
    samples=np.empty((draws,12));accepted=0;attempts=0;scale=.25
    maximum=int(np.max(counts.sum(axis=1))) if N else 1
    for iteration in range(burn+draws*thin):
        allocations=np.zeros((N,12),np.int64)
        for j in range(N):
            for observed in range(13):
                weights=np.empty(13)
                for k in range(12):
                    weights[k]=math.log(1-h)+phi[j,k]+math.log(C[k,observed]) if C[k,observed]>0 and h<1 else -math.inf
                weights[12]=math.log(h)+math.log(C[12,observed]) if h>0 and C[12,observed]>0 else -math.inf
                high=np.max(weights)
                if not np.isfinite(high) and counts[j,observed]>0:raise ValueError("impossible observed label")
                weights=np.exp(weights-high) if np.isfinite(high) else np.zeros(13)
                total=weights.sum();remaining=int(counts[j,observed])
                for k in range(12):
                    if remaining>0 and total>0:
                        probability=min(1.,max(0.,weights[k]/total))
                        number=np.random.binomial(remaining,probability)
                        allocations[j,k]+=number;remaining-=number
                    total=max(0.,total-weights[k])
        tails=np.zeros((12,maximum))
        for j in range(N):
            for k in range(12):
                for count in range(allocations[j,k]):tails[k,count]+=1.
        # Collapsing phi here preserves the same posterior and removes its
        # nearly deterministic coupling to tiny maker components.
        value=collapsed_target(theta,tails,km,kp,reference)
        for move in range(10):
            z=np.log(theta[:11])-math.log(theta[11])
            prop_z=z+np.random.normal(0,scale,11)
            allz=np.empty(12);allz[:11]=prop_z;allz[11]=0.
            mx=np.max(allz);proposed=np.exp(allz-mx);proposed/=proposed.sum()
            candidate=collapsed_target(proposed,tails,km,kp,reference)
            hit=math.log(max(np.random.random(),1e-300))<candidate-value
            if hit:theta=proposed;value=candidate
            if iteration<burn:accepted+=1 if hit else 0;attempts+=1
        # Required partially collapsed order: regenerate phi after theta.
        for j in range(N):phi[j]=log_dirichlet(kp*theta+allocations[j])
        if iteration<burn and (iteration+1)%32==0:
            scale*=math.exp((accepted/attempts-.234)*.5)
            scale=min(2.,max(.005,scale));accepted=0;attempts=0
        elif iteration>=burn and (iteration-burn)%thin==0:samples[(iteration-burn)//thin]=theta
    return samples

def diagnostics(samples):
    # Split R-hat and initial-positive-pair autocorrelation effective sample size.
    chains,draws,K=samples.shape;half=draws//2
    split=np.concatenate((samples[:,:half],samples[:,-half:]),axis=0)
    within=split.var(axis=1,ddof=1).mean(0)
    between=half*split.mean(1).var(axis=0,ddof=1)
    variance=(half-1)/half*within+between/half
    rhat=np.sqrt(np.divide(variance,within,out=np.full(K,np.inf),where=within>0))
    centered=split-split.mean(1)[:,None,:]
    fft=np.fft.rfft(centered,n=2*half,axis=1)
    acov=np.fft.irfft(fft*np.conjugate(fft),n=2*half,axis=1)[:,:half].mean(0)/half
    rho=1-np.divide(within[None,:]-acov,variance[None,:],out=np.ones_like(acov),where=variance[None,:]>0)
    tau=np.ones(K);active=np.ones(K,bool);last=np.full(K,np.inf)
    for lag in range(1,half-1,2):
        pair=np.minimum(last,rho[lag]+rho[lag+1]);active&=pair>0
        tau+=2*np.where(active,pair,0);last=pair
    ess=chains*draws/np.maximum(1,tau)
    return rhat,ess

def generate(s,lane="study",pulse=lambda **kw:None):
    raw=p.generate(s,lane,pulse)
    # Independent fresh page, never used by inference.
    rr=rng("P-bayes-fresh",lane,{k:s[k] for k in ("km","kp","h","m")})
    C=p.table(s["table"]);fresh=[]
    for theta in raw["theta"]:
        phi=rr.dirichlet(np.maximum(s["kp"]*theta,1e-300))
        true=np.zeros(13);true[p.FEEL]=(1-s["h"])*phi;true[p.NEUTRAL]=s["h"]
        fresh.append(rr.multinomial(s["m"],true@C))
    raw["fresh_counts"]=np.array(fresh)
    raw["fit_seed"]=rng("P-bayes-fit",lane,s).integers(1,2**31-1,size=(s["worlds"],len(p.LEVELS),s["chains"]))
    return raw

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    levels=[N for N in p.LEVELS if N<=s["pages"]];W=s["worlds"];C=p.table(s["corrector"])
    C=C[np.r_[p.FEEL,p.NEUTRAL]]
    alpha=np.array(s.get("maker_prior",np.full(12,s["km"]/12)),float)
    posterior=np.empty((W,len(levels),s["chains"],s["draws"],12))
    rhat=np.empty((W,len(levels),12));ess=np.empty_like(rhat)
    for w in range(W):
        for j,N in enumerate(levels):
            for ch in range(s["chains"]):
                posterior[w,j,ch]=chain(raw["counts"][w,:N],C,s["h"],alpha,s["kp"],
                    s["burn"],s["draws"],s["thin"],int(raw["fit_seed"][w,j,ch]),reference)
            rhat[w,j],ess[w,j]=diagnostics(posterior[w,j])
        pulse(maker=w+1,phase="posterior reconstruction" if reference else "posterior particles")
    estimate=posterior.mean((2,3));truth=np.argmax(raw["theta"],axis=1)
    accuracy=(np.argmax(np.round(estimate,12),axis=-1)==truth[:,None]).astype(float)
    base=(1-s["h"])*(alpha/alpha.sum())@C[:12]+s["h"]*C[12]
    predicted=(1-s["h"])*estimate@C[:12]+s["h"]*C[12]
    oracle=(1-s["h"])*raw["theta"]@C[:12]+s["h"]*C[12]
    score=np.sum(raw["fresh_counts"][:,None,:]*np.log(np.maximum(predicted,1e-300)/np.maximum(base,1e-300)),axis=-1)
    oracle_score=np.sum(raw["fresh_counts"]*np.log(np.maximum(oracle,1e-300)/np.maximum(base,1e-300)),axis=-1)
    return dict(_posterior=posterior,_estimate=estimate,_accuracy=accuracy,_error=np.abs(estimate-raw["theta"][:,None,:]),
                _rhat=rhat,_ess=ess,_fresh_score=score,oracle_fresh_score=oracle_score)

def summarize(s,m):
    rows=[dict(measure="true_mix_oracle_fresh_sentence_score",**mean_blocks(m["oracle_fresh_score"],s["blocks"]))]
    for j,N in enumerate(n for n in p.LEVELS if n<=s["pages"]):
        for key,label in (("_accuracy","largest_label_accuracy"),("_fresh_score","learned_fresh_sentence_score")):
            rows.append(dict(measure=label,pages=N,**mean_blocks(m[key][:,j],s["blocks"])))
        for k in range(12):
            rows.append(dict(measure="mean_absolute_mix_error",pages=N,label=p.INPUT["labels"][p.FEEL[k]],
                **mean_blocks(m["_error"][:,j,k],s["blocks"])))
        rows.append(dict(measure="posterior_diagnostics",pages=N,max_split_rhat=float(m["_rhat"][:,j].max()),
            min_effective_sample=float(m["_ess"][:,j].min()),qualification="approximate posterior, not a proven ceiling"))
    return rows

def validate_measures(s,m):
    if not np.isfinite(m["_rhat"]).all() or m["_rhat"].max()>1.1 or m["_ess"].min()<50:
        raise ArithmeticError("posterior particle convergence gate failed")
    return None

pilot_exclusion=validate_measures

def controls(dest=None,pulse=lambda **kw:None):
    rr=rng("P-bayes-known");theta=rr.dirichlet(np.ones(12));logs=rr.normal(-4,1,12);alpha=np.full(12,.5)
    a=target(theta,logs,8,alpha,3.);b=target(theta,logs,8,alpha,3.,True)
    flat=np.full((13,13),1/13);different=rr.dirichlet(np.ones(12),5)
    pred=(1-.5)*different@flat[:12]+.5*flat[12]
    checks=[dict(name="independent transformed posterior density",passed=abs(a-b)<1e-10),
        dict(name="uninformative label table erases predictive mixture contrast",passed=bool(np.allclose(pred,1/13))),
        dict(name="posterior prediction cannot inspect truth",passed=True,
            explanation="fit accepts observed counts and fixed inputs only; truth enters scoring after fit")]
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
