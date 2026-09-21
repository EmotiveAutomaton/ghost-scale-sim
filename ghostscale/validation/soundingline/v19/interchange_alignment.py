"""Fixed supervised rank-one probes on already admitted saved reader states."""
from itertools import product
import gzip
import time
import numpy as np
from ..v18_3.io import canonical, read, write, file_digest
from ..v18_3.world import rng
from .interchange_fixtures import arrays, decode, loss, reconstruct, final_normalize
from .readout_model import save_arrays

ARMS=('unchanged','learned','random','shuffled','wrong-factor','mean-direction','full-state')
FACTORS=(0,2)


def unit(vector):
    size=float(np.linalg.norm(vector))
    if not np.isfinite(size) or size<1e-12:raise ValueError('zero or invalid direction')
    return vector/size


def fit_probe(x,y):
    center=x.mean(0);yc=float(y.mean());xc=x-center;cov=xc.T@xc/len(x)
    penalty=.001*float(np.trace(cov))/x.shape[1]+1e-8
    system=cov+np.eye(x.shape[1])*penalty;rhs=xc.T@(y-yc)/len(x)
    beta=np.linalg.solve(system,rhs);bias=yc-center@beta
    return beta,float(bias),dict(penalty=penalty,normal_residual=float(np.max(abs(system@beta-rhs))),
        training_mse=float(np.mean((x@beta+bias-y)**2)),examples=len(x),dimension=x.shape[1])


def swap(recipient,donor,direction):
    return recipient+((donor-recipient)@direction)[...,None]*direction


def controls():
    x=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    beta,_,_=fit_probe(x,x[:,0]);q=unit(beta);recipient=x[[0,0]];donor=x[[3,1]]
    got=swap(recipient,donor,q);wrong=swap(recipient,donor,np.array([0.,1.]))
    constant={'head.weight':np.zeros((32,2)),'head.bias':np.zeros(32)}
    return {'live:selective_factor':bool(np.allclose(got,[[1.,-1.],[-1.,-1.]])),
        'placebo:constant_output':bool(np.array_equal(decode(constant,got),decode(constant,recipient))),
        'positive:wrong_factor_damage':bool(np.all(wrong[:,1]!=recipient[:,1]) and np.all(wrong[:,0]==recipient[:,0])),
        'positive:no_op':bool(np.array_equal(swap(x,x,q),x)),
        'positive:proper_score':bool(np.allclose(loss(np.ones((1,8))/8,np.ones((1,8))/8),np.log(8)))}


def states_at_site(source,stem,seed,kind,parameters,site):
    saved=arrays(source/'forecasts'/f'{stem}-{seed}-{kind}.npz')
    if site=='final':return saved['hidden']
    public=arrays(source/'reader'/f'{stem}.npz')
    if set(public)!={'codes','novel'}:raise ValueError('unexpected reader fields')
    hidden,p=reconstruct(parameters,public['codes'],public['novel'],kind,site)
    normalized=final_normalize(parameters,hidden)
    if not np.allclose(normalized,saved['hidden'],atol=1e-12,rtol=0):raise ValueError('saved final-state reconstruction failed')
    if not np.allclose(p,decode(parameters,saved['hidden']),atol=1e-12,rtol=0):raise ValueError('saved output reconstruction failed')
    return hidden


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    site=cfg.get('site','final')
    if site not in ('final','pre-final-normalization') or (site!='final' and cfg['kinds']!=['transformer']):
        raise ValueError('unadmitted intervention site')
    if not all(checks.values()):raise ValueError('alignment control failed')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('frozen alignment input corrupted')
    source=root/'inputs';review=read(source/'INDEPENDENT_REVIEW.json')
    if not review['passed'] or not all(x['passed'] for x in review['per_fit_capability']):raise ValueError('challenge capability not admitted')
    rows=[];probe_rows=[];fits=[];timing=[]
    for draw,seed,kind in product(cfg['training_draws'],cfg['fit_seeds'],cfg['kinds']):
        began=time.process_time();pulse(phase='training-probes',draw=draw,seed=seed,kind=kind)
        parameters={k:v.astype(float) for k,v in arrays(source/'inputs/models'/f'{draw}-{seed}-{kind}.npz').items()}
        def output(h):return decode(parameters,final_normalize(parameters,h) if site=='pre-final-normalization' else h)
        states=[];makers=[];lineages=[]
        for lineage in cfg['train_lineages']:
            stem=f'train-{lineage}-{draw}'
            full=states_at_site(source,stem,seed,kind,parameters,site)
            states.append(full[:,31])
            if site!='final':save_arrays(root/'site_states'/f'{stem}-{seed}-{kind}.npz',hidden=full)
            makers.append(arrays(source/'evaluator'/f'{stem}.npz')['maker']);lineages.extend([lineage]*16)
        x=np.concatenate(states);labels=2*np.concatenate(makers)[:,FACTORS]-1;lineages=np.array(lineages)
        directions={};coefficients={}
        for fi,factor in enumerate(FACTORS):
            y=labels[:,fi];beta,bias,receipt=fit_probe(x,y);coefficients[factor]=(beta,bias)
            random=rng('v19-C1-probe-controls',draw,seed,kind,factor);shuffled=y.copy()
            for lineage in cfg['train_lineages']:
                ix=np.flatnonzero(lineages==lineage);shuffled[ix]=random.permutation(y[ix])
            sb,si,sr=fit_probe(x,shuffled)
            directions[factor]={'learned':unit(beta),'shuffled':unit(sb),
                'random':unit(random.normal(size=x.shape[1])),
                'mean-direction':unit(x[y==1].mean(0)-x[y==-1].mean(0))}
            fits.append(dict(draw=draw,seed=seed,kind=kind,factor=factor,probe=receipt,shuffled=sr))
            save_arrays(root/'models'/f'{draw}-{seed}-{kind}-{factor}.npz',
                beta=beta,bias=np.array(bias),shuffled_beta=sb,shuffled_bias=np.array(si),
                training_labels=y,shuffled_labels=shuffled,training_lineages=lineages,**directions[factor])
        for factor in FACTORS:directions[factor]['wrong-factor']=directions[2-factor]['learned']
        parameters={k:v.astype(float) for k,v in arrays(source/'inputs/models'/f'{draw}-{seed}-{kind}.npz').items()}
        timing.append(dict(phase='probe-fit',draw=draw,seed=seed,kind=kind,cpu_seconds=time.process_time()-began))
        for lineage in cfg['development_lineages']:
            began=time.process_time();pulse(phase='paired-interchange',lineage=lineage,draw=draw,seed=seed,kind=kind)
            stem=f'development-{lineage}-{draw}';ev=arrays(source/'evaluator'/f'{stem}.npz')
            inverse=np.argsort(ev['maker_indices']);makers=ev['maker'][inverse];hidden=states_at_site(source,stem,seed,kind,parameters,site)[inverse]
            if site!='final':save_arrays(root/'site_states'/f'{stem}-{seed}-{kind}.npz',hidden=hidden,maker_indices=np.arange(16))
            laws=arrays(source/'evaluator'/f'development-{lineage}-laws.npz')['laws']
            for length,factor in product(cfg['lengths'],FACTORS):
                h=hidden[:,length-1];beta,bias=coefficients[factor];y=makers[:,factor]
                probe_rows.append(dict(lineage=lineage,draw=draw,seed=seed,kind=kind,length=length,factor=factor,
                    accuracy=float(np.mean((h@beta+bias>=0)==y)),predicted_positive=float(np.mean(h@beta+bias>=0))))
                for change in (False,True):
                    donor=makers.copy();donor[:,1]=1-donor[:,1]
                    if change:donor[:,factor]=1-donor[:,factor]
                    hybrid=makers.copy();hybrid[:,factor]=donor[:,factor]
                    # Lexicographic enumeration of four binary maker coordinates.
                    di=donor@np.array([8,4,2,1]);hi=hybrid@np.array([8,4,2,1]);target=laws[hi]
                    original=output(h);donorp=output(h[di])
                    probabilities=[]
                    for arm in ARMS:
                        changed=h if arm=='unchanged' else h[di] if arm=='full-state' else swap(h,h[di],directions[factor][arm])
                        pred=output(changed)
                        if arm=='full-state' and not np.array_equal(pred,donorp):raise ValueError('full donor identity failed')
                        if arm=='unchanged' and not np.array_equal(pred,original):raise ValueError('recipient identity failed')
                        probabilities.append(pred)
                        losses=loss(target,pred).mean(-1);oracle=loss(target,target).mean(-1)
                        error=(.5*abs(target-pred).sum(-1)).mean(-1);movement=(.5*abs(original-pred).sum(-1)).mean(-1)
                        for ri in range(16):rows.append(dict(lineage=lineage,draw=draw,seed=seed,kind=kind,length=length,factor=factor,change=change,arm=arm,recipient=ri,donor=int(di[ri]),hybrid=int(hi[ri]),loss=float(losses[ri]),oracle_loss=float(oracle[ri]),target_tv=float(error[ri]),prediction_tv=float(movement[ri])))
                    save_arrays(root/'forecasts'/f'{lineage}-{draw}-{seed}-{kind}-{length}-{factor}-{int(change)}.npz',probabilities=np.array(probabilities))
            timing.append(dict(phase='interchange-score',draw=draw,seed=seed,kind=kind,lineage=lineage,cpu_seconds=time.process_time()-began))
    (root/'raw').mkdir(exist_ok=True)
    for n,data in [('alignment',rows),('probe',probe_rows)]:
        (root/'raw'/f'{n}_points.json.gz').write_bytes(gzip.compress(canonical(data),mtime=0))
    write(root/'FITS.json',fits);write(root/'TIMING.jsonl',dict(measurements=timing,accounting='already included in native charge'))
    return dict(controls={**checks,'positive:full_donor_identity':True,'positive:input_integrity':True},
        rows=len(rows),probe_rows=len(probe_rows),probes=len(fits),model_updates=0,new_model_settings=0,
        arms=list(ARMS),site=site,scope='supervised rank-one static factor access; exploratory constructed method, miniature — architecture untested')
