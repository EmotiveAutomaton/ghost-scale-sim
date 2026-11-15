"""Family L pre-flight, the exact-events reader (no grid): A is known given events and sizes."""
import numpy as np, zlib, json
from l_leftover import TYPES, XS, simulate
def ll_exact(pr,e,xi,d):
    a=0.0; out=np.empty(len(d))
    for t in range(len(d)):
        a=pr['g']*XS[xi[t]]*e[t]+pr['rho']*a; p=1-np.exp(-a)
        out[t]=np.log(max(p,1e-300)) if d[t] else np.log(max(1-p,1e-300))
    return np.cumsum(out)
def ll_events(pr,e): return np.cumsum(np.where(e,np.log(pr['p']),np.log(1-pr['p'])))
res={}
for name,pr in TYPES.items():
    rng=np.random.default_rng(zlib.crc32(f'exact|{name}'.encode())); sep=[]
    for _ in range(200):
        e,xi,d=simulate(pr,1500,rng)
        L={n:ll_exact(p2,e,xi,d)+ll_events(p2,e) for n,p2 in TYPES.items()}
        lo=L[name]-np.max([L[n] for n in TYPES if n!=name],axis=0); ok=lo>=3
        if ok[-1]:
            bad=np.where(~ok)[0]; sep.append(int(bad[-1]+2) if len(bad) else 1)
        else: sep.append(None)
    got=[s for s in sep if s is not None]
    res[name]={'separated':round(len(got)/200,3),'median_steps':int(np.median(got)) if got else None}
print('shows and events, exact (200 worlds, 1,500 steps):',json.dumps(res))
o=json.load(open('l_leftover_out.json')); o['results']['shows and events, exact']=res; json.dump(o,open('l_leftover_out.json','w'),indent=1)
