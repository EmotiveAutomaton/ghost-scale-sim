"""Equal-size old-world portfolios; selection never receives development arrays."""
from collections import defaultdict
import gzip
import time
import numpy as np
from ..v18_3 import world as W
from ..v18_3.io import write, canonical
from ..v18_4 import neural_data as N
from . import readout_data as D, readout_model as M
from .readouts import interval

CANDIDATES=N.TRAIN_QUERIES+N.NEW_QUERIES
GRID=(.001,.01,.1)

def fit(x,y,ridge):
    x=np.column_stack((np.ones(len(x)),x))
    penalty=np.eye(x.shape[1])*ridge;penalty[0,0]=0
    return np.linalg.solve(x.T@x+penalty,x.T@y)

def predict(x,head,sizes=(16,16)):
    return M.probabilities(np.column_stack((np.ones(len(x)),x))@head,list(sizes))

def columns(indices):
    return np.array([i*16+j for i in indices for j in range(16)],dtype=int)

def select_portfolios(teachers,learned_fit,targets_fit,learned_select,targets_select,seed,size=5):
    count=teachers.shape[1]//16
    if not 1<=size<=count:raise ValueError('invalid portfolio size')
    original=list(range(size));random=sorted(np.random.default_rng(seed+700).choice(count,size,replace=False).tolist())
    diversity=[0]
    while len(diversity)<size:
        choices=[j for j in range(count) if j not in diversity]
        distances=[min(float(np.mean((teachers[:,columns([j])]-teachers[:,columns([k])])**2)) for k in diversity) for j in choices]
        diversity.append(choices[int(np.argmax(distances))])
    targeted=[];search=[]
    while len(targeted)<size:
        choices=[j for j in range(count) if j not in targeted];losses=[]
        for j in choices:
            cols=columns(targeted+[j]);head=fit(learned_fit[:,cols],targets_fit,.01)
            p,_=predict(learned_select[:,cols],head)
            value=float(M.scores(targets_select,p,[16,16])[0].mean());losses.append(value)
            search.append(dict(step=len(targeted),candidate=j,loss=value))
        targeted.append(choices[int(np.argmin(losses))])
    return dict(original=original,random=random,diversity=diversity,targeted=targeted),search

def splits(ids,lineages):
    sets=[set(lineages[:64]),set(lineages[64:96]),set(lineages[96:128])]
    if len(lineages)!=128 or len(set(lineages))!=128:raise ValueError('128 distinct training lineages required')
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('overlap')
    masks=[np.isin(ids[:,0],list(s)) for s in sets]
    if not np.all(sum(masks)==1):raise ValueError('unassigned training row')
    return masks

def controls():
    x=np.tile(np.array([[1.,0.],[0.,1.]]),(10,1));y=np.tile(x,(1,16))
    # 2 categorical questions with 16 entries each, each normalized.
    y/=8
    p,_=predict(x,fit(x,y,.01));null=np.full((20,32),1/16)
    q,_=predict(x,fit(x,null,.01))
    teacher=np.zeros((20,48));teacher[:,0]=1;teacher[:,16]=1-x[:,0];teacher[:,17]=x[:,0];teacher[:,32]=1
    portfolio,_=select_portfolios(teacher,teacher,y,teacher,y,11,1)
    rejected=False
    try:splits(np.array([[999,0,0]]),list(range(128)))
    except ValueError:rejected=True
    return dict(M.controls(),**{'live:informative_portfolio':portfolio['targeted']==[1],
        'live:known_linear_response':bool(np.max(abs(p-y))<.001),
        'placebo:constant_target':bool(np.max(abs(q-null))<1e-12),
        'positive:bad_split_rejected':rejected,
        'positive:equal_size_unique':all(len(a)==len(set(a))==1 for a in portfolio.values())})

def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('portfolio controls failed')
    assert cfg['candidate_queries']==list(CANDIDATES) and cfg['ridge_grid']==list(GRID)
    assert not set(cfg['train_lineages'])&set(cfg['development_lineages'])
    times=[];begin=time.process_time();data=D.generate(root,cfg,pulse)
    import json
    records=json.loads(gzip.decompress((root/'evaluator/selection_points.json.gz').read_bytes()))
    indexed={(r['split'],r['draw'],r['lineage'],r['index']):r for r in records}
    for (split,draw,tier),a in data.items():
        teachers=[];banks=[]
        for lineage,index,_ in a['ids']:
            r=indexed[split,draw,int(lineage),int(index)];w=r['world']
            matrix=np.concatenate([W.artifact_matrix(w,q) for q in CANDIDATES],axis=1)
            teachers.append(matrix[r['state']]);banks.append(D.old_posterior(w,r['history'])@matrix)
        a['candidate_teacher']=np.array(teachers);a['candidate_exact']=np.array(banks)
        M.save_arrays(root/'evaluator'/f'{split}-{draw}-candidate.npz',teacher=a['candidate_teacher'],exact=a['candidate_exact'],ids=a['ids'])
        if split=='train':M.save_arrays(root/'training'/f'{draw}-candidates.npz',teacher=a['candidate_teacher'],x=a['x'],target=a['target'],ids=a['ids'])
    times.append(dict(phase='generation-and-references',cpu_seconds=time.process_time()-begin))
    rows=[];inventory=[];selections=[];spectra=[]
    for draw in cfg['training_draws']:
        train=data['train',draw,'artifact-history'];test=data['development',draw,'artifact-history']
        fm,sm,vm=splits(train['ids'],cfg['train_lineages'])
        assert [int(m.sum()) for m in (fm,sm,vm)]==[1024,512,512]
        for seed in cfg['fit_seeds']:
            pulse(phase='portfolio-fit',draw=draw,seed=seed);begin=time.process_time()
            h,hw,hb=M.basis(train['x'],seed);ht,_,_=M.basis(test['x'],seed)
            head=M.fit(h[fm],train['candidate_teacher'][fm])
            learned,_=M.probabilities(h@head,[16]*8);learned_test,_=M.probabilities(ht@head,[16]*8)
            portfolios,search=select_portfolios(train['candidate_teacher'][fm],learned[fm],train['target'][fm],learned[sm],train['target'][sm],seed)
            variance=np.mean((learned[fm]-train['candidate_exact'][fm])**2,axis=0)
            rng=np.random.default_rng(seed+draw+900)
            noisy,_=M.probabilities(train['candidate_exact']+rng.normal(size=learned.shape)*np.sqrt(variance),[16]*8)
            noisy_test,_=M.probabilities(test['candidate_exact']+rng.normal(size=learned_test.shape)*np.sqrt(variance),[16]*8)
            pairs={'learned':(learned,learned_test),'exact-oracle':(train['candidate_exact'],test['candidate_exact']),'noise-matched-oracle':(noisy,noisy_test)}
            stem=f'{draw}-{seed}'
            M.save_arrays(root/'models'/f'{stem}-acquisition.npz',basis_weights=hw,basis_bias=hb,old_head=head,noise_variance=variance)
            M.save_arrays(root/'evaluator'/f'{stem}-banks.npz',learned_train=learned,learned_development=learned_test,noisy_train=noisy,noisy_development=noisy_test)
            selections.append(dict(draw=draw,seed=seed,portfolios=portfolios,search=search))
            for portfolio,indices in portfolios.items():
                cols=columns(indices)
                assert len(cols)==80 and len(set(cols))==80
                for bank,(x,xt) in pairs.items():
                    pulse(phase='portfolio-readout',draw=draw,seed=seed,portfolio=portfolio,bank=bank)
                    fits=[fit(x[fm][:,cols],train['target'][fm],ridge) for ridge in GRID]
                    val=[float(M.scores(train['target'][vm],predict(x[vm][:,cols],weights)[0],[16,16])[0].mean()) for weights in fits]
                    choice=int(np.argmin(val));weights=fits[choice]
                    p,invalid=predict(xt[:,cols],weights);loss,brier=M.scores(test['target'],p,[16,16])
                    label=f'{stem}-{portfolio}-{bank}'
                    M.save_arrays(root/'models'/f'{label}.npz',target_head=weights,candidate_indices=indices,ridge_grid=GRID,validation_loss=val)
                    M.save_arrays(root/'forecasts'/f'{label}.npz',probabilities=p,invalid=invalid,ids=test['ids'])
                    inventory.append(dict(draw=draw,seed=seed,portfolio=portfolio,bank=bank,ridge=GRID[choice],target_parameters=int(weights.size),old_parameters=int(head.size),fit_histories=1024,portfolio_histories=512,ridge_histories=512,acquired_queries=8,deployed_queries=5))
                    for lineage in cfg['development_lineages']:
                        for cl in range(4):
                            mask=(test['ids'][:,0]==lineage)&(test['ids'][:,2]==cl)
                            for q in range(2):rows.append(dict(draw=draw,seed=seed,portfolio=portfolio,bank=bank,lineage=lineage,world_class=cl,query=q,histories=int(mask.sum()),loss=float(loss[mask,q].mean()),brier=float(brier[mask,q].mean()),invalid=float(invalid[mask,q].mean())))
                for lineage in cfg['development_lineages']:
                    for cell in range(16):
                        w=W.make_world(cell,lineage);matrix=np.concatenate([np.ones((24,1))]+[W.artifact_matrix(w,CANDIDATES[j]) for j in indices],axis=1)
                        singular=np.linalg.svd(matrix,compute_uv=False)
                        spectra.append(dict(draw=draw,seed=seed,portfolio=portfolio,lineage=lineage,cell=cell,singular_values=singular.tolist(),rank_1e10=int(sum(singular>1e-10)),rank_1e12=int(sum(singular>1e-12))))
            times.append(dict(phase='all-selection-and-fits',draw=draw,seed=seed,cpu_seconds=time.process_time()-begin))
    (root/'raw').mkdir(exist_ok=True);(root/'raw/portfolio_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',dict(selections=selections));write(root/'SPECTRA.json',dict(rows=spectra,scope='numerical ranks only; no exact ambiguity certificate'))
    clusters=defaultdict(list)
    for r in rows:clusters[r['portfolio'],r['bank'],r['world_class'],r['lineage']].append(r['loss'])
    means={k:float(np.mean(v)) for k,v in clusters.items()};contrasts=[]
    for bank in pairs:
        for portfolio in ('random','diversity','targeted'):
            for cl in [-1,0,1,2,3]:
                classes=range(4) if cl==-1 else [cl]
                values=[float(np.mean([means[portfolio,bank,k,l]-means['original',bank,k,l] for k in classes])) for l in cfg['development_lineages']]
                contrasts.append(dict(portfolio=portfolio,bank=bank,world_class=cl,**interval(values)))
    write(root/'TIMING.jsonl',dict(measurements=times,accounting='native attempt controls; do not charge component times twice'))
    return dict(controls=checks,rows=len(rows),contrasts=contrasts,fits=inventory,
        population='eight development lineages; four classes equally weighted; paired draws and feature seeds',
        uncertainty='conditional lineage intervals; per-fit rows retained; no confirmation or universal training uncertainty',
        scope='constructed old-world method; oracle banks privileged; repaired linear head; miniature — architecture untested')
