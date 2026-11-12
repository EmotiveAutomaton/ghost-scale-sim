"""Visible-stock exact finite-grid likelihoods and simple-reader symptoms."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .common import rng,softmax,mean,median,proportion,row,separation,read,write,digest

TYPES=('ONE','POINT','BAND','MOVE','DULL','SHARP')


def candidates(simple=False):
    if simple:return np.array([(1,theta,pi,0.) for theta in range(4,23) for pi in (.25,.5,1.,2.,4.)])
    rows=[]
    for typ in range(6):
        extras=(4.,8.) if typ==2 else ((.02,.1) if typ>=3 else (0.,))
        rows += [(typ,theta,pi,extra) for theta in (8,10,12,14,16,18,20) for pi in (.5,1.,2.) for extra in extras]
    return np.array(rows)


def truths():
    return [(typ,12.,1.,extra) for typ in range(6) for extra in
        ((.02,.1) if typ>=3 else ((4.,) if typ==2 else (0.,)))]


def settings():
    # Freeze the packet's one long-N cell as ONE, its stated near-alias example.
    return [dict(truth=list(truth),regime=regime,steps=256) for truth in truths()
            for regime in ('N','D','W')]+[dict(truth=[0,12.,1.,0.],regime='N',steps=1024)]


def probabilities(stock,params,lo,streak,previous,t):
    typ=params[:,0].astype(int);theta=params[:,1];precision=params[:,2];extra=params[:,3]
    s=np.asarray(stock)[:,None]
    moving=typ==3
    if t:lo[:,moving]=np.clip(lo[:,moving]+extra[moving]*(previous[:,None]-lo[:,moving]),2,20)
    dull=(typ==4);sharp=(typ==5)
    active=np.where(dull[None,:],s<theta-2,np.where(sharp[None,:],abs(s-theta)<=2,False))
    streak=np.where(active,streak+1,0)
    pi=np.broadcast_to(precision,lo.shape).copy()
    pi[:,dull]*=np.exp(-extra[dull]*streak[:,dull])
    pi[:,sharp]*=np.minimum(3,1+extra[sharp]*streak[:,sharp])
    lower=np.where(moving[None,:],lo,theta)
    upper=theta+np.where(typ==2,extra,0.)
    restore=pi*np.maximum(0,lower-s)-3
    spend=pi*np.maximum(0,s-upper)-3
    spend[:,~np.isin(typ,(1,2))]=-np.inf
    values=np.stack([np.zeros_like(restore),restore,spend],axis=2)
    return softmax(values),lo,streak


def generate(setting,lane='study',worlds=400):
    r=rng('G2',lane,setting);steps=setting['steps'];truth=np.array([setting['truth']],dtype=float)
    stocks=np.empty((worlds,steps),dtype=np.int16);actions=np.empty_like(stocks,dtype=np.int8);disturb=np.empty_like(actions)
    stock=np.full(worlds,12.);previous=stock.copy();lo=np.full((worlds,1),truth[0,1]);streak=np.zeros((worlds,1),int)
    for t in range(steps):
        stocks[:,t]=stock
        p,lo,streak=probabilities(stock,truth,lo,streak,previous,t)
        a=(r.random(worlds)[:,None]>np.cumsum(p[:,0],axis=1)).sum(axis=1)
        d=(r.random(worlds)[:,None]>np.array([.5,.85])).sum(axis=1)
        actions[:,t]=a;disturb[:,t]=d;previous=stock.copy()
        addition=np.where(a==1,3.,np.where(a==2,-3.,0.))
        if setting['regime']=='D' and 64<=t<128:addition=np.where(a==1,0.,addition)
        stock=stock+addition-d
        if setting['regime']=='W' and 64<=t<128:stock+=2
        stock=np.clip(stock,0,24)
    return dict(stocks=stocks,actions=actions,disturbances=disturb,final_stock=stock)


def infer(raw,setting):
    params=candidates();simple=candidates(True);all_params=np.vstack([params,simple])
    n=len(params);worlds,steps=raw['stocks'].shape
    lo=np.broadcast_to(all_params[:,1],(worlds,len(all_params))).copy();streak=np.zeros_like(lo,dtype=int)
    ll=np.zeros_like(lo);last=np.full((worlds,9),-1,int);final=None
    groups=[np.flatnonzero(params[:,0]==i) for i in range(6)]
    groups += [np.flatnonzero((params[:,0]==i)&(params[:,3]==.1)) for i in (3,4,5)]
    truth_index=int(setting['truth'][0]);previous=raw['stocks'][:,0].astype(float)
    theta_posterior=None
    for t in range(steps):
        s=raw['stocks'][:,t].astype(float)
        p,lo,streak=probabilities(s,all_params,lo,streak,previous,t)
        chosen=np.take_along_axis(p,raw['actions'][:,t,None,None],axis=2)[:,:,0]
        with np.errstate(divide='ignore'):ll+=np.log(chosen)
        evidence=np.column_stack([logsumexp(ll[:,:n][:,g],axis=1)-np.log(len(g)) for g in groups])
        final=evidence[:,truth_index,None]-evidence
        last=np.where(final<1,t,last);previous=s
    chosen=np.argmax(ll[:,n:],axis=1)
    exact=np.flatnonzero(np.all(params==np.array(setting['truth']),axis=1))[0]
    one=groups[0];one_log=ll[:,one];posterior=softmax(one_log)
    theta_posterior=posterior[:,params[one,1]==12].sum(axis=1)
    return dict(separation=separation(last,final),last_below=last,final_odds=final,
        fitted_theta=simple[chosen,1],fitted_pi=simple[chosen,2],
        loss_per_step=(ll[:,exact]-ll[np.arange(worlds),n+chosen])/steps,
        one_theta12_posterior=theta_posterior)


def run(destination,mode='original',original=None,pulse=lambda **kw:None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);rows=[];files={}
    from .g2_reference import infer as reconstruct
    for index,setting in enumerate(settings()):
        name=f'cell-{index:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:
            raw=generate(setting);np.savez_compressed(destination/name,**raw);files[name]=digest(destination/name)
        result=(reconstruct(raw,setting) if mode=='reconstruct' else infer(raw,setting))
        np.savez_compressed(destination/f'cell-{index:03d}_measures.npz',**result)
        typ=int(setting['truth'][0])
        for rival in range(9):
            if rival==typ or (rival>=6 and rival-3==typ):continue
            label=TYPES[rival] if rival<6 else TYPES[rival-3]+'(rate=0.1)'
            rows.append(row(setting,'last_below_plus_one',median(result['separation'][:,rival]),rival=label))
            rows.append(row(setting,'not_separated_share',proportion(np.isinf(result['separation'][:,rival])),rival=label))
        if setting['regime']=='N':
            for measure in ('fitted_theta','fitted_pi','loss_per_step'):
                rows.append(row(setting,measure,mean(result[measure]),reader='POINT maximum likelihood'))
            values,counts=np.unique(np.column_stack([result['fitted_theta'],result['fitted_pi']]),axis=0,return_counts=True)
            for values_,count in zip(values,counts):
                rows.append(row(setting,'fit_pair_share',proportion((result['fitted_theta']==values_[0])&(result['fitted_pi']==values_[1])),
                    fitted_theta=float(values_[0]),fitted_pi=float(values_[1])))
        pulse(family='G2',cell=index+1,cells=len(settings()))
    result=dict(family='G2',rows=rows,scale_only=True,independent_units='worlds',
        separation_definition='literal zero-based last-below step + 1; final-below censored; zero means above on first step')
    write(destination/'SUMMARY.json',result)
    if mode!='reconstruct':write(destination/'RAW_MANIFEST.json',files)
    return result


def controls(destination,pulse=lambda **kw:None):
    gates=[]
    from .g2_reference import infer as reconstruct
    for truth in ((0,12.,1.,0.),(1,12.,1.,0.)):
        setting=dict(truth=list(truth),regime='N',steps=256)
        raw=generate(setting,'known-answer',400);out=infer(raw,setting);other=reconstruct(raw,setting)
        gates.append(dict(name='independent_reconstruction_'+TYPES[truth[0]],passed=all(
            np.allclose(out[k],other[k],rtol=0,atol=1e-9,equal_nan=True) for k in out)))
        if truth[0]==0:
            gates.append(dict(name='ONE_never_spends',passed=not bool((raw['actions']==2).any())))
            estimate=float(out['one_theta12_posterior'].mean())
            gates.append(dict(name='ONE_reference_recovery',passed=estimate>=.9,estimate=estimate,bar=.9))
        else:
            estimate=float(((out['fitted_theta']==12)&(out['fitted_pi']==1)).mean())
            gates.append(dict(name='POINT_ML_recovery',passed=estimate>=.9,estimate=estimate,bar=.9))
        pulse(control=TYPES[truth[0]])
    params=np.array([(0,12,1,0),(4,12,1,.1),(5,12,1,.1)],float)
    p,_,_=probabilities(np.array([12.]),params,np.full((1,3),12.),np.zeros((1,3),int),np.array([12.]),0)
    gates.append(dict(name='identical_action_laws_zero_log_odds_increment',passed=bool(np.array_equal(p[:,0],p[:,1]) and np.array_equal(p[:,0],p[:,2])),
        qualification='Conditional parameter-level identity; do not infer unchanged marginal odds after differing parameter histories.'))
    result=dict(family='G2',passed=all(x['passed'] for x in gates),gates=gates)
    write(Path(destination)/'CONTROLS.json',result);return result
