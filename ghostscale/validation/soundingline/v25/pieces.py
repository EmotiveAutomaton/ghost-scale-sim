"""W1's explicitly provisional six-piece world and exact 210-hypothesis reader."""
import itertools,math
import numpy as np
from .common import simplex,softmax,posterior,scores,information,rng,record,close

WEIGHTS=simplex(4)
HYPOTHESES=[(w,b,h) for w in range(15) for b in (1,3) for h in range(-1,6)]
TRAIN=((0,1,2,3),(0,1,4,5),(2,3,4,5),(0,1,2,3,4,5))
SWITCH=((0,2,4),(1,3,5))
PREFIXES=(8,32,128,512)
MAKERS=8

def library(overlap,random):
    e=np.zeros((6,3));e[np.arange(6),np.arange(6)%3]=random.choice([.5,1.],6)
    if overlap in ('partial','heavy'):
        for k in (0,1):e[k,(k+1)%3]=random.choice([.5,1.])
    if overlap=='heavy':
        for k in (2,3):e[k,:]=random.choice([.5,1.],3)
    return e

def laws(e,available,hypotheses=HYPOTHESES,satiate=None,independent=False,weights=WEIGHTS):
    if independent:
        result=[]
        for wi,b,h in hypotheses:
            w=weights[wi].copy()
            if satiate is not None:w[satiate]=0
            utility=[b*(math.fsum(float(w[j]*e[k,j]) for j in range(3))+int(k==h)) for k in available]
            result.append(softmax(utility,True))
        return np.array(result)
    ws=np.array([weights[wi] for wi,b,h in hypotheses]);betas=np.array([b for wi,b,h in hypotheses])
    if satiate is not None:ws[:,satiate]=0
    u=ws@e[list(available)].T+np.array([[int(k==h) for k in available] for wi,b,h in hypotheses])
    return softmax(betas[:,None]*u)

def probes(e,independent=False):
    out=[]
    for k in range(6):out.append(('remove',k,tuple(i for i in range(6) if i!=k),None))
    for j in range(3):out.append(('satiate',j,tuple(range(6)),j))
    for j,a in enumerate(SWITCH):out.append(('switch',j,a,None))
    return [(kind,index,a,s,laws(e,a,satiate=s,independent=independent)) for kind,index,a,s in out]

def aliases(e,independent=False,training=TRAIN):
    distributions=[laws(e,a,independent=independent) for a in training]
    # Symmetric KL, mean over four training situations, not summed over records.
    kl=np.zeros((210,210));delta=np.zeros_like(kl)
    for p in distributions:
        lp=np.log(p);kl+=(np.sum(p*lp,1)[:,None]-p@lp.T+np.sum(p*lp,1)[None,:]-lp@p.T)/(2*len(training))
        delta=np.maximum(delta,np.max(abs(p[:,None,:]-p[None,:,:]),axis=2))
    pairs=np.argwhere(np.triu((delta<=1e-12)|(kl<.01),1));probe_laws=probes(e,independent)
    rows=[]
    for i,j in pairs:
        separate={}
        for kind in ('remove','satiate','switch'):
            values=[]
            for typ,k,a,s,p in probe_laws:
                if typ==kind:values.append(float(.5*np.sum((p[i]-p[j])*np.log(p[i]/p[j]))))
            separate[kind]=max(values)
        rows.append(dict(pair=[int(i),int(j)],exact=bool(delta[i,j]<=1e-12),symmetric_kl=float(max(0,kl[i,j])),separation=separate))
    return rows

def block(setting,identity,independent=False):
    r=rng(*identity);e=library(setting['overlap'],r);hs=HYPOTHESES
    situations=TRAIN[:setting.get('training_count',4)]
    training=[laws(e,a,independent=independent) for a in situations];probe=probes(e,independent)
    outputs=[];raw=[]
    for maker in range(MAKERS):
        eligible=[i for i,h in enumerate(hs) if h[1]==setting.get('beta',3)]
        truth=int(r.choice(eligible));ll=np.zeros(len(hs));choices=[];at32=None
        for t in range(max(setting.get('prefixes',PREFIXES))):
            si=t%len(situations);local=int(r.choice(len(situations[si]),p=training[si][truth]));choices.append(local)
            ll+=np.log(training[si][:,local])
            if t+1 in setting.get('prefixes',PREFIXES):outputs.append(record(scores(posterior(ll,independent=independent),truth),maker=maker,history=t+1,reader='passive'))
            if t+1==32:at32=ll.copy()
        p=posterior(at32,independent=independent);gains=[information(p,x[-1]) for x in probe]
        selected=int(np.flatnonzero(np.asarray(gains)>=max(gains)-1e-12)[0]);kind,k,a,s,law=probe[selected]
        y=int(r.choice(len(a),p=law[truth]));pa=posterior(at32+np.log(law[:,y]),independent=independent)
        passive=training[32%len(training)];z=int(r.choice(passive.shape[1],p=passive[truth]));pb=posterior(at32+np.log(passive[:,z]),independent=independent)
        outputs.append(record(dict(primary=float(math.log(pa[truth])-math.log(pb[truth])),selected_information=gains[selected],passive_information=information(p,passive)),maker=maker,history=32,reader='one_intervention',intervention=kind))
        raw.append(dict(truth=truth,choices=choices,selected=selected,probe_outcome=y,passive_outcome=z))
    amap=aliases(e,independent,situations)
    exact=[a for a in amap if a['exact']]
    metrics=dict(exact_pairs=len(exact),near_pairs=len(amap)-len(exact))
    for kind in ('remove','satiate','switch'):metrics[kind+'_separation_share']=sum(x['separation'][kind]>1e-12 for x in exact)/len(exact) if exact else None
    outputs.append(record(metrics,reader='alias_map'))
    return dict(public=dict(effects=e.tolist(),training=situations,switch=SWITCH),truth_and_observations=raw,alias_map=amap,readouts=outputs)

def controls():
    e=np.array([[1,1,0],[0,0,1],[.5,.5,0],[0,0,.5],[1,1,0],[0,0,1.]])
    h=[(0,3,-1),(1,3,-1)];w=np.array([[.75,.25,0],[.25,.75,0]])
    a=laws(e,tuple(range(6)),h,weights=w);assert np.max(abs(a[0]-a[1]))<1e-12
    b=laws(e,tuple(range(6)),h,weights=w,satiate=0);assert np.max(abs(b[0]-b[1]))>.01
    close(a.tolist(),laws(e,tuple(range(6)),h,weights=w,independent=True).tolist(),tol=1e-12)
    close(posterior(np.log(a[:,0])+np.log(a[:,3])).tolist(),posterior(np.log(a[:,0])+np.log(a[:,3]),independent=True).tolist(),tol=1e-12)
    habit=laws(e,(0,1),[(0,3,0)],weights=np.array([[0.,0.,0.]]),satiate=0)
    assert habit[0,0]>.9
    assert len(HYPOTHESES)==210 and all(len(set(a)^set(b))>=2 for a in TRAIN for b in SWITCH)
    return dict(passed=True,exact_alias=True,satiation_breaks_fixture=True,independent_posterior=True,habit_bonus_survives=True)
