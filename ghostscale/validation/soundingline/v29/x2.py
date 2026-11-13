"""Exact finite-state mixture for visible and late-entry frozen-sum readers.

Late-entry hidden streak prior: uniform over 0..n-1 and already frozen (n).
This is a declared reader prior, not the true conditional prehistory law.
"""
import numpy as np
from scipy.special import logsumexp
from . import ledger
from .common import class_evidence

def expand(start,breach=False):
    base=ledger.candidates(frozen=True,breach=breach);rows=[];initials=[];model=[];weight=[]
    for j,p in enumerate(base):
        states=range(int(p[2])+1) if start and int(p[0])==4 else (0,)
        for state in states:
            rows.append(p);initials.append(state);model.append(j);weight.append(-np.log(len(states)))
    return base,np.array(rows),np.array(initials),np.array(model),np.array(weight)

def _vector(raw,s):
    base,params,initials,model,logprior=expand(s.get('start',0),s.get('part')=='X2b')
    n=raw['records'].shape[0];start=s.get('start',0);steps=s['steps'];state=ledger.initial(n,params)
    state['streak'][:]=initials
    state['frozen'][:]=(params[:,0]==4)&(initials>=params[:,2])
    ll=np.broadcast_to(logprior,(n,len(params))).copy()
    classes=base[:,0].astype(int);available=np.unique(classes);true=int(s['truth'][0])
    true_index=int(np.flatnonzero(available==true)[0])
    trace=np.empty((n,steps,len(available)));ftrace=np.empty((n,steps))
    last=np.full((n,len(available)),-1);accepted_before=np.zeros(n);freeze_estimate=np.zeros(n)
    for t in range(steps):
        rec=raw['records'][:,start+t];m,d,x,offer,price,accepted,after,spent=rec[:,:8].T
        ledger.before_offer(state,m,d)
        p=ledger.offer_probability(m,d,x,offer,price,params,state,s.get('slope',1.5),s.get('tremble',.02))
        with np.errstate(divide='ignore'):ll+=np.log(np.where(accepted[:,None],p,1-p))
        ledger.update_offer(state,offer,price,accepted,params)
        p=ledger.spend_probability(after,d,params,s.get('slope',1.5),s.get('tremble',.02))
        with np.errstate(divide='ignore'):ll+=np.log(np.where(spent[:,None],p,1-p))
        collapsed=np.column_stack([logsumexp(ll[:,model==j],axis=1) for j in range(len(base))])
        evidence=class_evidence(collapsed,classes);odds=evidence[:,true_index,None]-evidence
        trace[:,t]=evidence
        last=np.where(odds<3,t,last)
        accepted_before+=(offer==-1)&accepted.astype(bool)&((start+t+1)<raw['freeze_time'])
        # Dating is restricted to an observed transition; pre-entry freezes are left censored.
        frozen_model=params[:,0]==4
        normalized=np.exp(ll[:,frozen_model]-logsumexp(ll[:,frozen_model],axis=1,keepdims=True))
        probability=(normalized*state['frozen'][:,frozen_model]).sum(1)
        ftrace[:,t]=probability
        freeze_estimate=np.where((freeze_estimate==0)&(probability>=.5),start+t+1,freeze_estimate)
    return dict(class_trace=trace,frozen_trace=ftrace,pair_separation=last+2,final_class_odds=odds,final_likelihood=collapsed,
                froze=(raw['freeze_time']<=s.get('life',steps)).astype(float),
                froze_before_entry=(raw['freeze_time']<=start).astype(float),
                accepted_sales_before_freeze=accepted_before,
                estimated_freeze=freeze_estimate,truth_freeze=raw['freeze_time'],
                breach=(raw['breach_time']<=s.get('life',steps)).astype(float))


from pathlib import Path
from . import ledger_reference
from .common import sustained,mean_blocks,censored_blocks,conditional_blocks,write,digest,audit_controls
settings=ledger.settings_x2

def infer(raw,s,reference=False):
    start=s.get('start',0);steps=s['steps'];life=s.get('life',steps);n=len(raw['records'])
    base=ledger.candidates(frozen=True,breach=s.get('part')=='X2b');classes=np.unique(base[:,0]).astype(int);true=int(np.flatnonzero(classes==s['truth'][0])[0])
    if reference:trace,ll,frozen=ledger_reference.likelihood(raw,s,True,s.get('part')=='X2b')
    else:
        out=_vector(raw,s);trace=out['class_trace'];ll=out['final_likelihood'];frozen=out['frozen_trace']
    odds=trace[:,:,true,None]-trace;pair=np.stack([sustained(odds[:,:,j]) for j in range(len(classes))],axis=1)
    times=np.arange(start+1,start+steps+1);rec=raw['records'][:,start:start+steps]
    accepted_before=np.sum((rec[:,:,3]==-1)&(rec[:,:,5]==1)&(times[None,:]<raw['freeze_time'][:,None]),axis=1)
    estimate=np.where(np.any(frozen>=.5,axis=1),np.argmax(frozen>=.5,axis=1)+start+1,0)
    same=np.all(abs(odds)<1e-12,axis=1);same[:,true]=False
    result=dict(separation=sustained(np.min(np.delete(odds,true,axis=2),axis=2)),pair_separation=pair,final_class_odds=odds[:,-1],final_likelihood=ll,froze=(raw['freeze_time']<=life).astype(float),froze_before_entry=(raw['freeze_time']<=start).astype(float),accepted_sales_before_freeze=accepted_before,estimated_freeze=estimate,truth_freeze=raw['freeze_time'],breach=(raw['breach_time']<=life).astype(float),same_probability=same.astype(float))
    if s.get('part')=='X2b':
        rival=int(np.flatnonzero(classes==(6 if s['truth'][0]==5 else 5))[0]);cross=pair[:,rival]
        # Re-express the sustained crossing in post-breach observations, keeping missing breaches censored.
        elapsed=np.maximum(0,cross-raw['breach_time']);available=np.maximum(0,steps-raw['breach_time'])
        result.update(post_breach_records=np.where((cross<=steps)&(raw['breach_time']<=steps),elapsed,available+1),post_breach_available=available,post_breach_resolved=((cross<=steps)&(raw['breach_time']<=steps)).astype(float))
    return result

def pilot_exclusion(s,z):
    return 'fewer than 20 percent of pilot lives froze; design exclusion, not universal LIVE identity' if s.get('part')=='X2' and float(np.mean(z['froze']))<.2 else None

def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);rows=[];files={};roster=settings() if selected is None else selected
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:raw=ledger.generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        out=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**out)
        classes=np.unique(ledger.candidates(True,s['part']=='X2b')[:,0]).astype(int)
        rows.append(dict(setting=s,measure='separation',**censored_blocks(out['separation'],s['steps'],s['blocks'])))
        for j,typ in enumerate(classes):
            if typ!=s['truth'][0]:rows.append(dict(setting=s,measure='pair_separation',rival=ledger.CLASSES[typ],**censored_blocks(out['pair_separation'][:,j],s['steps'],s['blocks'])))
        for key in ('froze','froze_before_entry','accepted_sales_before_freeze','breach'):rows.append(dict(setting=s,measure=key,**mean_blocks(out[key],s['blocks'])))
        dated=(out['truth_freeze']>s['start'])&(out['truth_freeze']<=s['start']+s['steps'])&(out['estimated_freeze']>0)
        rows.append(dict(setting=s,measure='absolute_dating_error_observed_and_dated_only',**conditional_blocks(abs(out['estimated_freeze']-out['truth_freeze']),dated,s['blocks'])))
        rows.append(dict(setting=s,measure='dating_coverage',**mean_blocks(dated,s['blocks'])))
        if s['part']=='X2b':
            rows.append(dict(setting=s,measure='resolved_after_breach_given_breach',**conditional_blocks(out['post_breach_resolved'],out['breach']>0,s['blocks'])))
            rows.append(dict(setting=s,measure='post_breach_records_resolved_only',**conditional_blocks(out['post_breach_records'],out['post_breach_resolved']>0,s['blocks'])))
            rows.append(dict(setting=s,measure='available_post_breach_records_given_breach',**conditional_blocks(out['post_breach_available'],out['breach']>0,s['blocks'])))
        pulse(family='X2',cell=i+1,cells=len(roster))
    result=dict(family='X2',settings=roster,rows=rows,no_registered_claim=True)
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result

def controls(destination,pulse=lambda **kw:None):
    checks=[dict(name='empty frozen-state sum has zero mass',passed=ledger_reference.lse(np.array([],float))==-np.inf)]
    for part,start,truth in [('X2',0,[4,4,5]),('X2',16,[4,2,10]),('X2b',0,[5,2,0]),('X2b',0,[6,4,0])]:
        s=dict(part=part,truth=truth,environment='scarce',regime='PROBE',buy_share=.1,start=start,steps=64,life=128,worlds=8,blocks=8)
        raw=ledger.generate(s,'known-answer');a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=part+str(truth)+str(start)+' independent full path',passed=all(np.allclose(a[k],b[k],atol=1e-9,rtol=0,equal_nan=True) for k in a)))
    params=np.array([[4,2,5],[1,2,0]],float);state=ledger.initial(1,params);state['frozen'][:,0]=True
    equal=True
    for m,d,o,price in [(0,6,-1,32),(10,2,1,4),(30,0,1,2)]:
        value=ledger.offer_probability(np.array([m]),np.array([d]),np.array([8]),np.array([o]),np.array([price]),params,state);equal &= value[0,0]==value[0,1]
    checks.append(dict(name='individual frozen weight two and RULE two identical after freeze',passed=bool(equal)))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result
