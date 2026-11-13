"""Full X1 map with independent scalar path reconstruction."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from . import ledger,ledger_reference
from .common import class_evidence,sustained,mean_blocks,censored_blocks,write,digest,audit_controls

EVENTS=('probe refusal','refusal under unmet need','cheap purchase declined','dear purchase made','decline within eight steps of high offer','need spending','other')
settings=ledger.settings_x1

def paths(raw,s,reference=False):
    params=ledger.candidates();records=raw['records'];n,steps,_=records.shape
    if reference:
        value=ledger_reference.kernel_paths(records,params,np.zeros(len(params),int),np.arange(len(params)),np.zeros(len(params)),params,np.arange(4),s.get('slope',1.5),s.get('tremble',.02))
        return value[3],value[4]
    state=ledger.initial(n,params);ll=np.zeros((n,len(params)));full=np.empty((n,steps,len(params)));offered=np.empty_like(full)
    for t in range(steps):
        m,d,x,o,p,a,after,spent=records[:,t,:8].T
        ledger.before_offer(state,m,d)
        prob=ledger.offer_probability(m,d,x,o,p,params,state,s.get('slope',1.5),s.get('tremble',.02))
        with np.errstate(divide='ignore'):ll+=np.log(np.where(a[:,None],prob,1-prob))
        offered[:,t]=ll;ledger.update_offer(state,o,p,a,params)
        prob=ledger.spend_probability(after,d,params,s.get('slope',1.5),s.get('tremble',.02))
        with np.errstate(divide='ignore'):ll+=np.log(np.where(spent[:,None],prob,1-prob))
        full[:,t]=ll
    return full,offered

def infer(raw,s,reference=False):
    ll,offered=paths(raw,s,reference);params=ledger.candidates();classes=params[:,0].astype(int);true=int(s['truth'][0]);n,steps,_=ll.shape
    e=class_evidence(ll,classes);eo=class_evidence(offered,classes)
    odds=e[:,:,true,None]-e;best=e[:,:,true]-np.max(np.delete(e,true,axis=2),axis=2)
    offodds=eo[:,:,true]-np.max(np.delete(eo,true,axis=2),axis=2)
    record=raw['records'];m,d,x,o,p,a,after,spent=record[:,:,:8].transpose(2,0,1)
    gain=.5*p+np.minimum(p,np.maximum(0,d-m));refused=np.maximum.accumulate(np.where((o==-1)&(a==0),gain,0),axis=1)
    prior=np.array([-np.log(4)-np.log(sum(classes==k)) for k in classes])
    posterior=np.exp(ll+prior-logsumexp(ll+prior,axis=2,keepdims=True));cum=np.cumsum(posterior[:,:,:9],axis=2)
    floor=np.max(np.where(cum<.05,params[None,None,:9,1],0),axis=2)
    last_high=np.maximum.accumulate(np.where((o==-1)&(p>=4),np.arange(steps),-999),axis=1)
    recent=np.concatenate((np.full((n,1),-999),last_high[:,:-1]),axis=1)
    event=np.full((n,steps),6,int)
    event=np.where((o!=0)&(a==0)&((np.arange(steps)-recent)<=8),4,event)
    event=np.where((o==1)&(p>=4)&(a==1),3,event)
    event=np.where((o==1)&(p<=2)&(a==0),2,event)
    event=np.where((o==-1)&(d>m)&(a==0),1,event)
    event=np.where((o==-1)&(p==32)&(a==0),0,event)
    event=np.where((best>=3)&(offodds<3),5,event)
    hit=np.argmax(best>=3,axis=1);has=np.any(best>=3,axis=1)
    same=np.all(abs(odds)<1e-12,axis=1)&np.all(abs(eo[:,:,true,None]-eo)<1e-12,axis=1);same[:,true]=False
    return dict(separation=sustained(best),pair_separation=np.stack([sustained(odds[:,:,j]) for j in range(4)],axis=1),final_class_odds=odds[:,-1],final_likelihood=ll[:,-1],false_rule=np.any(e[:,:,1]-np.max(np.delete(e,1,axis=2),axis=2)>=3,axis=1).astype(float),first_event=np.where(has,event[np.arange(n),hit],-1),event_counts=np.stack([np.sum(event==j,axis=1) for j in range(7)],axis=1),high_offers=np.sum((o==-1)&(p>=4),axis=1),price_floor=floor,refused_temptation=refused,same_probability=same.astype(float))

def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);rows=[];files={};roster=settings() if selected is None else selected
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:raw=ledger.generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        out=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**out)
        rows.append(dict(setting=s,measure='separation',**censored_blocks(out['separation'],s['steps'],s['blocks'])))
        for j in range(4):
            if j!=int(s['truth'][0]):rows.append(dict(setting=s,measure='pair_separation',rival=ledger.CLASSES[j],**censored_blocks(out['pair_separation'][:,j],s['steps'],s['blocks'])))
        for key in ('false_rule','high_offers'):rows.append(dict(setting=s,measure=key,**mean_blocks(out[key],s['blocks'])))
        for j,event in enumerate(EVENTS):
            rows.append(dict(setting=s,measure='first_three_nats_event_share',event=event,**mean_blocks(out['first_event']==j,s['blocks'])))
            rows.append(dict(setting=s,measure='event_frequency_per_step',event=event,**mean_blocks(out['event_counts'][:,j]/s['steps'],s['blocks'])))
        rows.append(dict(setting=s,measure='no_first_crossing',**mean_blocks(out['first_event']<0,s['blocks'])))
        for step in sorted(set([x for x in (16,32,64,128,256,512,1024,2048) if x<=s['steps']]+[s['steps']])):
            for key in ('price_floor','refused_temptation'):rows.append(dict(setting=s,measure=key,step=step,**mean_blocks(out[key][:,step-1],s['blocks'])))
        pulse(family='X1',cell=i+1,cells=len(roster))
    result=dict(family='X1',settings=roster,rows=rows,no_registered_claim=True,known_answers=ledger.construction_checks())
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result

def controls(destination,pulse=lambda **kw:None):
    checks=[];construction=ledger.construction_checks()
    checks.append(dict(name='approved clipping-margin counterexample',passed=abs(construction['named_counterexample']['live_sell']-.18242552380635635)<1e-13 and construction['corrected_identity']))
    for truth,regime,buy in [([0,4,0],'LOW',0.),([1,2,0],'PROBE',.5),([2,4,0],'MIXED',.1),([3,2,8],'PROBE',.5)]:
        s=dict(truth=truth,regime=regime,environment='scarce',buy_share=buy,steps=64,worlds=8,blocks=8);raw=ledger.generate(s,'known-answer');a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=str(truth)+' independent entire path',passed=all(np.allclose(a[k],b[k],rtol=0,atol=1e-9,equal_nan=True) for k in a)))
        old=ledger.infer_x1(raw,s)
        checks.append(dict(name=str(truth)+' original kernel continuity',passed=all(np.allclose(a[k],old[k],rtol=0,atol=1e-9,equal_nan=True) for k in ('separation','final_likelihood','false_rule','price_floor','refused_temptation'))))
    s=dict(truth=[1,2,0],regime='LOW',environment='scarce',buy_share=0.,steps=64,worlds=8,blocks=8);raw=ledger.generate(s,'known-alias');a=infer(raw,s)
    checks.append(dict(name='LOW no-buy RULE and BACKFIRE class laws coincide',passed=bool(np.all(a['same_probability'][:,3]==1))))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result
