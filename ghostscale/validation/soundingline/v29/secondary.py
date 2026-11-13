"""Retained-data secondary descriptions; never redraws a primary world."""
import argparse,time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.stats import t as student
from .common import read,write,digest,rng,mean_blocks,audit_controls
from ..v23 import runtime

@njit(cache=True)
def reference_counts(d,a,success,w,length,blocks):
    n,steps=d.shape;out=np.zeros((blocks,7,2,2),np.int64);size=n//blocks
    for i in range(n):
        for t in range(1,steps):
            if t//length!=(t-1)//length or w[t] or w[t-1] or not a[i,t-1]:continue
            out[i//size,d[i,t],success[i,t-1],0]+=a[i,t]
            out[i//size,d[i,t],success[i,t-1],1]+=1
    return out

def counts(raw,s,reference=False):
    d=raw['deficits'];a=raw['actions'];success=raw['successes'];w=raw['watching'];blocks=s['blocks'];n,steps=d.shape
    if reference:return reference_counts(d,a,success,w,s['length'],blocks)
    eligible=(a[:,:-1]==1)&(w[None,1:]==0)&(w[None,:-1]==0)&(np.arange(1,steps)//s['length']==np.arange(steps-1)//s['length'])
    out=np.zeros((blocks,7,2,2),np.int64)
    for deficit in range(7):
        for won in range(2):
            mask=eligible&(d[:,1:]==deficit)&(success[:,:-1]==won)
            out[:,deficit,won,0]=(mask*a[:,1:]).reshape(blocks,-1).sum(1)
            out[:,deficit,won,1]=mask.reshape(blocks,-1).sum(1)
    return out

def ratio(x):
    num=x[:,0];den=x[:,1];total=int(den.sum())
    if not total:return dict(estimate=None,interval95=[None,None],events=0,block_counts=den.tolist())
    draws=rng('Q2-secondary-ratio',len(x)).integers(len(x),size=(4000,len(x)));d=den[draws].sum(1);v=num[draws].sum(1)
    sample=v[d>0]/d[d>0]
    return dict(estimate=float(num.sum()/total),interval95=np.quantile(sample,[.025,.975]).tolist(),events=total,block_counts=den.tolist())

def controls():
    raw=dict(deficits=np.array([[2,2,2,2]]),actions=np.array([[1,1,0,1]]),successes=np.array([[1,0,0,1]]),watching=np.zeros(4,int))
    s=dict(length=4,blocks=1);a=counts(raw,s);b=counts(raw,s,True)
    return audit_controls([dict(name='independent success and failure counts',passed=bool(np.array_equal(a,b))),dict(name='known successful predecessor and failed predecessor',passed=a[0,2,1].tolist()==[1,1] and a[0,2,0].tolist()==[0,1]),dict(name='empty deficit stays missing',passed=ratio(a[:,0,0])['estimate'] is None)])

def run(c,dest,reference=False):
    runtime.below_normal();start=time.process_time();c=Path(c);dest=Path(dest);dest.mkdir(exist_ok=False)
    gate=controls();write(dest/'CONTROLS.json',gate)
    if not gate['passed']:raise ArithmeticError('secondary known-answer failed')
    rows=[];counts_store={};episode={};bindings={}
    q2=c/'jobs/Q2-original/output';summary=read(q2/'SUMMARY.json');manifest=read(q2/'RAW_MANIFEST.json')
    bindings['Q2']=digest(q2/'SUMMARY.json')
    for i,s in enumerate(summary['settings']):
        path=q2/f'cell-{i:03d}_points.npz'
        if digest(path)!=manifest[path.name]:raise ValueError('Q2 raw changed')
        with np.load(path,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        v=raw['actions'].sum(1)/max(1,np.sum(raw['watching']==0));old=next(r for r in summary['rows'] if r['setting']==s and r['measure']=='push_rate')
        if abs(mean_blocks(v,s['blocks'])['estimate']-old['estimate'])>1e-12:raise ValueError('Q2 original mean not reproduced')
        table=counts(raw,s,reference);counts_store[str(i)]=table
        for d in range(7):
            for won in range(2):rows.append(dict(setting=s,deficit=d,previous_owned_push_succeeded=bool(won),**ratio(table[:,d,won])))
        if s['length']==150 and ((s['part']=='Q2b' and s['order']=='WU') or (s['part']=='Q2' and s['order']=='CU')):
            block=raw['actions'][:,150:300].reshape(s['blocks'],-1).mean(1)
            episode[(s['truth'],s['order'])]=block
    watched=[]
    for truth in range(8):
        w=episode[(truth,'WU')];o=episode[(truth,'CU')];vw=w.var(ddof=1)/len(w);vo=o.var(ddof=1)/len(o);se=np.sqrt(vw+vo);df=(vw+vo)**2/(vw*vw/(len(w)-1)+vo*vo/(len(o)-1));delta=float(w.mean()-o.mean());width=float(student.ppf(.975,df)*se);ci=[delta-width,delta+width]
        watched.append(dict(truth=truth,watched=float(w.mean()),owned=float(o.mean()),difference=delta,interval95=ci,practical_bar=.05,follow_on_admitted=ci[0]>=-.05 and ci[1]<=.05))
    x1=c/'jobs/X1-original/output';xs=read(x1/'SUMMARY.json');verified=read(c/'families/X1/VERIFIED.json');ladder=np.array([0,1,2,4,8,16,32,64,128,256]);floors=[]
    bindings['X1']=digest(x1/'SUMMARY.json')
    for i,s in enumerate(xs['settings']):
        path=x1/f'cell-{i:03d}_measures.npz'
        if digest(path)!=verified['replay_files'][path.name]:raise ValueError('X1 retained measures changed')
        with np.load(path,allow_pickle=False) as z:
            f=float(z['price_floor'][:,-1].mean());g=float(z['refused_temptation'][:,-1].mean());events=int(np.sum(z['refused_temptation'][:,-1]>0))
        fi=int(np.searchsorted(ladder,f,side='right')-1);gi=int(np.searchsorted(ladder,g,side='right')-1)
        floors.append(dict(setting=s,mean_floor=f,mean_maximum_refused_gain=g,floor_position=fi,refusal_position=gi,worlds_with_refusal=events,tracks=events>0 and abs(fi-gi)<=1))
    q3=read(c/'jobs/Q3-original/output/SUMMARY.json');bindings['Q3']=digest(c/'jobs/Q3-original/output/SUMMARY.json')
    pairs=[r for r in q3['rows'] if r['measure']=='pair_separation' and r['setting'].get('profile')=='even' and ((r['setting']['truth']==0 and r['rival']=='TOTAL') or (r['setting']['truth']==1 and r['rival']=='COUNT'))]
    if len(pairs)!=4:raise ValueError('Q3 two-direction trigger roster not four')
    fraction=sum(r['tracks'] for r in floors)/len(floors)
    result=dict(no_registered_claim=True,raw_bindings=bindings,Q2_conditional_rates=rows,Q2_watched_owned=watched,X1_floor_trigger=dict(cells=floors,fraction_tracking=fraction,threshold=.8,admitted=fraction>=.8),Q3_weighted_count_trigger=dict(rows=pairs,admitted=any(isinstance(r['estimate'],(int,float)) and r['estimate']<500 for r in pairs)),limits='Conditional response rates are not causal; deterministic history features cannot add independent oracle information.')
    write(dest/'SUMMARY.json',result);np.savez_compressed(dest/'counts_points.npz',**counts_store)
    write(dest/'COMPLETE.json',dict(cpu_seconds=time.process_time()-start,summary_sha256=digest(dest/'SUMMARY.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--destination',required=True);p.add_argument('--reference',action='store_true');a=p.parse_args();run(a.campaign,a.destination,a.reference)
