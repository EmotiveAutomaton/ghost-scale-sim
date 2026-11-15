"""P label mixture recovery. Independent text-first implementation, before analyst scripts."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import nnls, lsq_linear
from .common import rng,mean_blocks,audit_controls,write

LEVELS=(1,2,4,8,16,32,64,128)
INPUT=json.loads(Path(__file__).with_name("reader_tables.json").read_text(encoding="utf-8"))
NEUTRAL=INPUT["labels"].index("no-emotion")
FEEL=np.array([k for k in range(13) if k!=NEUTRAL])
TABLES=("PERFECT","MODEL","HUMAN","HALF","HUMAN-N")

def table(name, neutral_shift=0.):
    if name=="PERFECT": c=np.eye(13)
    elif name=="HALF": c=np.full((13,13),.5/12);np.fill_diagonal(c,.5)
    else:
        c=np.array(INPUT["HUMAN" if name=="HUMAN-N" else name],float)
        c/=c.sum(1,keepdims=True) # supplied four-decimal row rounding
    if name=="HUMAN-N" or neutral_shift:
        old=c[NEUTRAL,NEUTRAL]
        new=.9 if name=="HUMAN-N" else old+neutral_shift
        if not 0<=new<=1: raise ValueError("invalid shifted neutral recall")
        c[NEUTRAL]*=(1-new)/(1-old);c[NEUTRAL,NEUTRAL]=new
    return c

def settings():
    result=[dict(km=km,kp=kp,h=h,m=m,table=c,corrector=c,neutral_shift=0.,
                 worlds=300,blocks=6,pages=128,part="labels")
            for km in (3,6,12,24) for kp in (3,12,48) for h in (0.,.5,.8)
            for m in (20,60,200) for c in TABLES]
    for h in (0.,.5,.8):
        for made,read in (("MODEL","HUMAN"),("HUMAN","MODEL")):
            result.append(dict(km=6,kp=3,h=h,m=60,table=made,corrector=read,
                               neutral_shift=0.,worlds=300,blocks=6,pages=128,part="misstated"))
        for c in ("MODEL","HUMAN"):
            for shift in (-.1,.1):
                result.append(dict(km=6,kp=3,h=h,m=60,table=c,corrector=c,
                                   neutral_shift=shift,worlds=300,blocks=6,pages=128,part="misstated"))
    return result

def generate(s,lane="study",pulse=lambda **kw:None):
    # Context pairing across reading tables; uniforms do not depend on truth in estimation.
    seedparts={k:s[k] for k in ("km","kp","h","m","pages")}
    if "maker_prior" in s:seedparts["maker_prior"]=s["maker_prior"]
    r=rng("P",lane,seedparts)
    alpha=np.array(s.get("maker_prior",np.full(12,s["km"]/12)),float)
    theta=r.dirichlet(alpha,s["worlds"])
    obs=np.empty((s["worlds"],s["pages"],13),np.int16)
    c=table(s["table"],s.get("neutral_shift",0.))
    for w,truth in enumerate(theta):
        page=r.dirichlet(np.maximum(s["kp"]*truth,1e-300),s["pages"])
        true=np.zeros((s["pages"],13));true[:,FEEL]=(1-s["h"])*page;true[:,NEUTRAL]=s["h"]
        probs=true@c;probs/=probs.sum(1,keepdims=True)
        obs[w]=np.array([r.multinomial(s["m"],p) for p in probs])
        if w%25==0:pulse(maker=w,makers=s["worlds"])
    return dict(theta=theta,counts=obs)

def estimates(counts,c,reference=False):
    freq=counts/counts.sum()
    count=freq[FEEL];count=count/count.sum() if count.sum() else np.full(12,1/12)
    # NNLS includes neutral as a thirteenth unknown: no use of evaluator theta.
    if reference:
        fit=lsq_linear(c.T,freq,bounds=(0,np.inf),method="bvls",tol=1e-13,max_iter=500).x
    else: fit=nnls(c.T,freq,maxiter=1000)[0]
    corrected=fit[FEEL];corrected=corrected/corrected.sum() if corrected.sum() else np.full(12,1/12)
    return np.stack((count,corrected))

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    levels=[n for n in LEVELS if n<=s["pages"]];n=len(raw["counts"])
    estimate=np.empty((n,len(levels),2,12)); c=table(s["corrector"])
    for w in range(n):
        cumulative=np.cumsum(raw["counts"][w],axis=0)
        for j,N in enumerate(levels): estimate[w,j]=estimates(cumulative[N-1],c,reference)
        if w%25==0:pulse(maker=w,makers=n,phase="independent" if reference else "fit")
    # Round only auxiliary label ranks, with declared tie rule; never likelihoods.
    named=np.argmax(np.round(estimate,12),axis=-1)
    # Exact permutation symmetry: equal integer counts have equal fitted weights.
    # Avoid decimal-bin boundaries splitting this algebraic tie between solvers.
    for w in range(n):
        cumulative=np.cumsum(raw["counts"][w],axis=0)
        for j,N in enumerate(levels):
            observed=cumulative[N-1,FEEL]
            named[w,j,0]=int(np.argmax(observed))
            if s["corrector"] in ("PERFECT","HALF"):
                named[w,j,1]=int(np.argmax(observed)) if np.ptp(estimate[w,j,1])>1e-12 else 0
    true=np.argmax(np.round(raw["theta"],12),axis=-1)
    return dict(_estimates=estimate,_accuracy=(named==true[:,None,None]).astype(float),
                _absolute_error=np.abs(estimate-raw["theta"][:,None,None,:]))

def summarize(s,m):
    rows=[]
    for j,N in enumerate(n for n in LEVELS if n<=s["pages"]):
        for k,name in enumerate(("COUNT","CORRECTED")):
            rows.append(dict(measure="largest_label_accuracy",reader=name,pages=N,
                             **mean_blocks(m["_accuracy"][:,j,k],s["blocks"])))
            for label in range(12):
                rows.append(dict(measure="mean_absolute_mix_error",reader=name,pages=N,
                    label=INPUT["labels"][FEEL[label]],**mean_blocks(m["_absolute_error"][:,j,k,label],s["blocks"])))
    return rows

def pilot_exclusion(s,m):
    # Identification against a rival is not a label-estimation problem.
    return None

def controls(dest=None,pulse=lambda **kw:None):
    theta=np.arange(1,13,dtype=float);theta/=theta.sum()
    checks=[]
    for name in TABLES:
        c=table(name);x=np.zeros(13);x[FEEL]=.5*theta;x[NEUTRAL]=.5
        a=estimates((x@c)*60000,c);b=estimates((x@c)*60000,c,True)
        checks.append(dict(name=name+" independent NNLS",passed=bool(np.max(abs(a-b))<1e-9)))
        checks.append(dict(name=name+" population mixture recovered",passed=bool(np.max(abs(a[1]-theta))<1e-9)))
    half=table("HALF");x=np.zeros(13);x[FEEL]=theta
    checks.append(dict(name="symmetric noisy table preserves population argmax (packet correction)",
                       passed=int(np.argmax((x@half)[FEEL]))==int(np.argmax(theta))))
    se_sentences=(1.57**2+1)/.25**2
    checks.append(dict(name="P.b independent-noise variance arithmetic",passed=abs(se_sentences-55.4384)<1e-10,
                       sentences=se_sentences,pages_at_7_5_sentences=se_sentences/7.5,whole_pages=8))
    c=table("HUMAN");uniform=np.full((13,13),1/13)
    checks.append(dict(name="flat label channel erases true-mixture contrasts",
                       passed=bool(np.allclose(x@uniform,np.full(13,1/13)))))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
