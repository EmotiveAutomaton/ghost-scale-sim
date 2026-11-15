"""V30 pre-flight, family R: asked or watched. Why is a practice kept: HABIT, MEANS, FACE or RULE?
Keep chance on an occasion with price p, need flag n, watched flag w, standing cost 0.5:
 HABIT s(1.5*(1-0.5-p)); MEANS s(1.5*(4n-0.5-p)); FACE s(1.5*(4w-0.5-p)); RULE 0.98.
Watched: an occasion arrives with chance 0.2 a step; p from LOW {0:.5,1:.3,2:.2} or MIXED {0:.25,1:.25,2:.2,4:.15,8:.15};
 n=1 with chance .7, w=1 with chance .6; the reader sees p and the act, and sees or does not see the flags.
Asked: three one-off questions; the answer matches what the maker would do with chance q, else it is 'keep' with chance b."""
import numpy as np, zlib, json, itertools
S=lambda z:1/(1+np.exp(-z)); C=['HABIT','MEANS','FACE','RULE']
def keep(c,p,n,w):
    return {'HABIT':S(1.5*(0.5-p)),'MEANS':S(1.5*(4*n-0.5-p)),'FACE':S(1.5*(4*w-0.5-p)),'RULE':0.98}[c]
PRICES={'LOW':([0,1,2],[.5,.3,.2]),'MIXED':([0,1,2,4,8],[.25,.25,.2,.15,.15])}
PROBES={'DEVALUE':(0,0,1),'AUDIENCE':(0,1,0),'PRICE':(8,1,1)}   # (price, need flag, watched flag) of the imagined case
def asked(q,b):
    """expected log-odds for the truth against its best rival after all three answers; and share of worlds at 3 or more"""
    out={}
    for truth in C:
        tot=0.0; res=0.0
        for ans in itertools.product([0,1],repeat=3):
            pa={c:1.0 for c in C}
            for (name,(p,n,w)),a in zip(PROBES.items(),ans):
                for c in C:
                    pk=q*keep(c,p,n,w)+(1-q)*b; pa[c]*=pk if a else 1-pk
            lo=np.log(pa[truth])-max(np.log(pa[c]) for c in C if c!=truth)
            tot+=pa[truth]*lo; res+=pa[truth]*(lo>=3)
        out[truth]=(round(tot,2),round(res,2))
    return out
def watched(price,flags_seen,start=None,worlds=300,T=3000):
    out={}
    for truth in C:
        rng=np.random.default_rng(zlib.crc32(f'{price}|{flags_seen}|{truth}'.encode())); steps=[]
        ps,pp=PRICES[price]
        for _ in range(worlds):
            L=dict(start[truth]) if start else {c:0.0 for c in C}   # prior log-likelihoods from answers, if any
            done=None; t=0
            while t<T:
                t+=int(rng.geometric(0.2)); p=ps[rng.choice(len(ps),p=pp)]; n=int(rng.random()<.7); w=int(rng.random()<.6)
                a=rng.random()<keep(truth,p,n,w)
                for c in C:
                    if flags_seen: pk=keep(c,p,n,w)
                    else: pk=sum(keep(c,p,nn,ww)*(.7 if nn else .3)*(.6 if ww else .4) for nn in (0,1) for ww in (0,1))
                    L[c]+=np.log(pk if a else 1-pk)
                lo=L[truth]-max(L[c] for c in C if c!=truth)
                if lo>=3:
                    if done is None: done=t
                else: done=None
                if done is not None and t-done>400: break
            steps.append(done)
        got=[s for s in steps if s is not None]
        out[truth]=(round(len(got)/worlds,2),int(np.median(got)) if got else None)
    return out
if __name__=='__main__':
    R={'asked':{},'watched':{}}
    print('keep chance in the three imagined cases:',{n:{c:round(float(keep(c,*v)),2) for c in C} for n,v in PROBES.items()})
    for q,b in [(1.0,.5),(.9,.5),(.75,.5),(.75,.8),(.6,.5),(.6,.8)]:
        R['asked'][f'q={q},b={b}']=asked(q,b); print('ASKED q=%.2f b=%.1f: expected log-odds, share at 3 or more:'%(q,b),R['asked'][f'q={q},b={b}'])
    for price in ('LOW','MIXED'):
        for fs in (True,False):
            k=f'{price}, flags {"seen" if fs else "unseen"}'; R['watched'][k]=watched(price,fs); print('WATCHED',k,'(share resolved, median steps):',R['watched'][k])
    json.dump(R,open('r_probes_out.json','w'),indent=1)
