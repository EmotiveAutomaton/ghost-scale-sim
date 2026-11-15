"""V30 pre-flight, family P: pages until a maker's largest label is known.
Reader tables are measured on the public event-report corpus (Troiano, Oberlaender, Klinger 2023; CC BY 4.0):
 HUMAN = one reader against the writer's label; MODEL = the authors' released RoBERTa-large predictions; PERFECT = identity.
World: a maker has a mix over the 12 emotion labels, Dirichlet(km/12 each); a share h of every page is nothing-here.
A page's emotion mix is Dirichlet(kp * maker mix). A page has 60 sentences. The reader labels each sentence through its table.
Estimators after N pages: pooled COUNT of emotion labels; CORRECTED count (non-negative least squares through the table)."""
import sys, glob, json, zlib, numpy as np, pandas as pd
from scipy.optimize import nnls
D=sys.argv[1]
g=pd.read_csv(D+'/corpus/crowd-enVent_generation.tsv',sep='\t',quoting=3); v=pd.read_csv(D+'/corpus/crowd-enVent_validation.tsv',sep='\t',quoting=3)
L=sorted(g.emotion.unique()); ix={l:i for i,l in enumerate(L)}; K=len(L); NE=ix['no-emotion']; EM=[i for i in range(K) if i!=NE]
yt=g.set_index('text_id').emotion.map(ix)
def table(t,p):
    M=np.full((K,K),0.5)
    for a,b in zip(t,p): M[a,b]+=1
    return M/M.sum(1,keepdims=True)
Mh=table(v.text_id.map(yt).values,v.emotion.map(ix).values)
pr=pd.read_csv(sorted(glob.glob(D+'/predictions/TEmodel_classifier_*.tsv'))[0],sep='\t'); Mm=table(pr.text_id.map(yt).values,pr.predicted_emotion.map(ix).values)
T={'HUMAN':Mh,'MODEL':Mm,'PERFECT':np.eye(K)}
json.dump({'labels':L,'HUMAN':Mh.round(4).tolist(),'MODEL':Mm.round(4).tolist(),'source':'Crowd-enVENT (CC BY 4.0), doi 10.1162/coli_a_00461; rows = writer label, columns = reader label; 0.5 added to every cell'},open('reader_tables.json','w'),indent=1)
print('diagonals HUMAN',{l:round(float(Mh[i,i]),2) for l,i in ix.items()}); print('diagonals MODEL',{l:round(float(Mm[i,i]),2) for l,i in ix.items()})
def run(reader,h,km,kp,makers=300,m=60,Ns=(1,2,4,8,16,32,64)):
    rng=np.random.default_rng(zlib.crc32(f'{reader}|{h}|{km}|{kp}'.encode())); M=T[reader]; cum=np.cumsum(M,1)
    okc={N:0 for N in Ns}; oka={N:0 for N in Ns}
    for _ in range(makers):
        mk=rng.dirichlet(np.full(12,km/12)); counts=np.zeros(K); n=0
        for N in range(1,max(Ns)+1):
            pg=rng.dirichlet(np.maximum(kp*mk,1e-3)); full=np.zeros(K); full[EM]=(1-h)*pg; full[NE]=h
            true=rng.choice(K,size=m,p=full); obs=(rng.random(m)[:,None]>cum[true]).sum(1).clip(0,K-1); counts+=np.bincount(obs,minlength=K)
            if N in okc:
                q=counts/counts.sum(); x,_=nnls(M.T,q)
                okc[N]+=int(np.argmax(q[EM])==np.argmax(mk)); oka[N]+=int(np.argmax(x[EM])==np.argmax(mk))
    return {N:(round(okc[N]/makers,2),round(oka[N]/makers,2)) for N in Ns}
out={}
for km,kp in ((6,3),(24,3),(6,12)):
    for h in (0.0,0.5,0.8):
        for reader in ('PERFECT','MODEL','HUMAN'):
            r=run(reader,h,km,kp); out[f'km={km},kp={kp},h={h},{reader}']=r
            need=lambda j: next((N for N in sorted(r) if r[N][j]>=0.8),None)
            print('makers %s pages %s nothing-here %.1f %-7s | largest label right (count, corrected) by pages: %s | pages to 0.8: count %s, corrected %s'%('differ' if km==6 else 'alike','vary' if kp==3 else 'steady',h,reader,' '.join('%d:%.2f/%.2f'%(N,*r[N]) for N in sorted(r)),need(0),need(1)))
json.dump(out,open('p_pages_out.json','w'),indent=1)
