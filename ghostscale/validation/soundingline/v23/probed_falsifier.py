"""N16b: equal-input conflict probes through N13's unchanged categorical law."""
import math
import numpy as np
from scipy.special import logsumexp,softmax
from . import falsifier as f,directed_probes as adaptive
from .common import rng
FAMILY='N16b';READERS=['records','goal','tempered-records','personal-oracle','pooled']
MENU=[(j,a,b,sa,sb) for j in range(8) for a in range(4) for b in range(a+1,4) for sa,sb in ((1,3),(3,1))]
BRIDGE=np.array([np.stack([1-p,p],axis=-1) for p in [
    np.sum(f.LAW*np.array([float((sb if y==b else 0)>(sa if y==a else 0)) if (sb if y==b else 0)!=(sa if y==a else 0) else .5 for y in range(4)])[None,None,:],axis=-1)
    for j,a,b,sa,sb in MENU]])

def variants():return [dict(id='N16b-'+v['id'][4:],family=FAMILY,original=v) for v in f.variants()]

def infer(ll,q,exponent=1.,product=False):
    ll=np.asarray(ll)*exponent
    local=softmax(ll,axis=-1);ln=logsumexp(ll,axis=-1)-math.log(4)
    nlaw=softmax(ln.sum(-1));records=np.einsum('l,ljd,ldy->jy',nlaw,local,f.LAW)
    weighted=ll[:,None,:,:]+np.log(f.prior(q))[None]
    lg=logsumexp(weighted,axis=-1);joint=lg.sum(-1)
    mass=np.exp(joint-logsumexp(joint));post=softmax(weighted,axis=-1)
    goal=np.einsum('lg,lgjd,ldy->jy',mass,post,f.LAW)
    if product:
        scaled=np.exp(ll-ll.max(-1,keepdims=True));evidence=np.einsum('ljd,gjd->lgj',scaled,f.prior(q))
        mass2=evidence.prod(-1)*np.exp(ll.max(-1).sum(-1)-ll.max(-1).sum(-1).max())[:,None];mass2/=mass2.sum()
        if not np.allclose(mass,mass2,atol=2e-12,rtol=0):raise ValueError('N16b independent products')
    return records,goal,mass,post

def probe_distributions(ll,q,available):
    gains=[]
    for exponent in (1.,.5):
        _,_,mass,post=infer(ll,q,exponent)
        # Mechanism (goal) by law by candidate outcome, integrating personal type.
        local=post[:,:,[context[0] for context in MENU],:].transpose(2,1,0,3)
        laws=np.einsum('qgld,qldy->qgly',local,BRIDGE)
        gains.append(adaptive.information(mass.T,laws))
    p=np.zeros((5,len(MENU)));p[0,available[0]]=1;p[1,available]=1/len(available)
    for row,gain in ((2,gains[0]),(4,gains[1])):p[row,next(i for i in available if gain[i]>=max(gain[available])-1e-12)]=1
    p[3]=.8*p[2]+.2*p[1]
    return p,gains[0]

def score(v,p,t,evidence):
    setting=f.calibration(v['original']);actual=np.eye(4)[t['y']];order=[x[0] for x in p['ordinary']['query']]
    values={}
    for k in (2,4,8):
        forecasts=evidence[str(k)];loss={name:.5*float(np.mean(np.sum((np.asarray(prob)-actual)**2,axis=-1))) for name,prob in forecasts.items()}
        for name,value in dict(primary=loss['records']-loss['goal'],headroom=loss['records']-loss['personal-oracle'],
            records_over_pooled=loss['pooled']-loss['records'],tempered_gain=loss['tempered-records']-loss['goal'],
            records_loss=loss['records'],goal_loss=loss['goal'],margin=setting['margin']).items():values[f'k{k}|{name}']=value
    values['primary']=values['k2|primary'];return values

def case(v,phase,world,case,fixed,design):
    p,t=f.sample(v['original'],['N16b',phase],world,case);q=p['design']['q']
    ll=np.einsum('jy,ldy->ljd',f.count_records(p),np.log(f.LAW));available=list(range(len(MENU)))
    random=rng(FAMILY,phase,v['id'],world,case);steps=[];evidence={};order=[x[0] for x in p['query']]
    dominant=np.asarray(t['personal']).argmax(-1)
    for step in range(8):
        probs,gains=probe_distributions(ll,q,available);choice=int(random.choice(len(MENU),p=probs[3]));j=MENU[choice][0]
        response=int(random.choice(2,p=BRIDGE[choice,world%16,dominant[j]]))
        steps.append(dict(choice=choice,context=list(MENU[choice]),outcome=response,
            selection_probabilities=probs.tolist(),expected_information=gains.tolist(),realized_probability=float(probs[3,choice])))
        ll[:,j,:]+=np.log(BRIDGE[choice,:,:,response]);available.remove(choice)
        if step+1 in (2,4,8):
            records,goal,_,_=infer(ll,q);records,goal=records[order],goal[order]
            evidence[str(step+1)]={'records':records.tolist(),'goal':goal.tolist(),
                'tempered-records':f.tempered(records,f.calibration(v['original'])['exponent']).tolist(),
                'personal-oracle':np.asarray(t['personal'])[order].tolist(),'pooled':np.full((8,4),.25).tolist()}
    public=dict(ordinary=p,probes=steps)
    return public,t,evidence,score(v,public,t,evidence)

def check(v,p,t,evidence,fixed,design):
    ordinary=p['ordinary'];q=ordinary['design']['q'];ll=np.einsum('jy,ldy->ljd',f.count_records(ordinary),np.log(f.LAW))
    available=list(range(len(MENU)));order=[x[0] for x in ordinary['query']]
    for index,row in enumerate(p['probes']):
        probs,gain=probe_distributions(ll,q,available)
        if not np.allclose(probs,row['selection_probabilities'],atol=1e-12,rtol=0) or not np.allclose(gain,row['expected_information'],atol=1e-12,rtol=0):raise ValueError('N16b policy reconstruction')
        choice=row['choice'];j=MENU[choice][0]
        if choice not in available or list(MENU[choice])!=row['context'] or row['outcome'] not in (0,1):raise ValueError('N16b probe context')
        ll[:,j,:]+=np.log(BRIDGE[choice,:,:,row['outcome']]);available.remove(choice)
        if index+1 in (2,4,8):
            records,goal,_,_=infer(ll,q,product=True)
            expected={'records':records[order],'goal':goal[order],
                'tempered-records':f.tempered(records[order],f.calibration(v['original'])['exponent']),
                'personal-oracle':np.asarray(t['personal'])[order],'pooled':np.full((8,4),.25)}
            for name,prob in expected.items():
                if not np.allclose(prob,evidence[str(index+1)][name],atol=1e-11,rtol=0):raise ValueError('N16b forecast reconstruction')
    return score(v,p,t,evidence)

def block_readout(v,metrics):
    out={}
    for k in (2,4,8):
        cols={name:[m[f'k{k}|{name}'] for m in metrics] for name in ('primary','headroom','records_over_pooled')}
        result=f.block_readout(v['original'],0,cols)
        for name in ('supported','powered','falsifiable_local','low_gain'):out[f'k{k}|{name}']=float(result[name])
    return out
def study_readout(items):
    byid={x['variant']['id']:x for x in items};result={}
    calibration=f.calibration(dict(id='N13-1-81'))
    qualified=abs(calibration['independent_oracle_gain']-calibration['target'])<=3*calibration['independent_gain_se']+.1*calibration['margin']
    for k in (2,4,8):
        count=lambda item,name:int(sum(item['metrics'][f'k{k}|{name}']['world_values']))
        power=count(byid['N16b-1-81'],'powered');nulls={n:count(byid[f'N16b-0-{n}'],'supported') for n in (32,81,250)}
        passed=power>=13 and all(x==0 for x in nulls.values()) and qualified
        verdicts={}
        for identity,item in byid.items():
            flags=[item['metrics'][f'k{k}|{name}']['world_values'] for name in ('supported','falsifiable_local','low_gain')]
            verdicts[identity]=['SUPPORTED' if s else 'FALSIFIED' if passed and f0 and low else 'INCONCLUSIVE' for s,f0,low in zip(*flags,strict=True)]
        result[str(k)]=dict(powered_blocks=power,required=13,false_supported_by_history=nulls,allowed=0,
            independent_strength_calibration_qualified=qualified,P3_passed=passed,verdicts=verdicts)
    return dict(probe_budgets=result,route='V24 FX1 redesign only; real-data FX1 gates remain separate',
        stake_order_is_an_exact_alias=True,unavailable_70_percent_sensitivities_preserved=True)
def controls():
    assert np.all(BRIDGE>0) and np.allclose(BRIDGE.sum(-1),1)
    assert np.array_equal(BRIDGE[::2],BRIDGE[1::2])
    counts=np.zeros((8,4));counts[:,1]=12;ll=np.einsum('jy,ldy->ljd',counts,np.log(f.LAW))
    for q in (0.,.4):
        old=f.predictions(counts,q);new=infer(ll,q,product=True)
        assert np.allclose(old[0][0],new[0],atol=1e-12,rtol=0) and np.allclose(old[1][0],new[1],atol=1e-12,rtol=0)
    ll[:,0,:]+=3*np.log(BRIDGE[0,:,:,1]);a,b,_,_=infer(ll,0.,product=True);assert np.allclose(a,b,atol=1e-13,rtol=0)
    return dict(passed=True,zero_probe_reproduces_N13=True,habit_alias_after_probes=True,
        independent_products=True,binary_bridge_normalized=True,stake_order_alias_recorded=True,equal_reader_inputs=True)
def run(root,plan,pulse):
    from .directed_common import run as execute
    return execute(root,plan,pulse)
def reconstruct(root,plan):
    from .directed_common import reconstruct as verify
    return verify(root,plan)
