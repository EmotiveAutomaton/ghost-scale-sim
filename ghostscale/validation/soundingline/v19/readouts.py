"""Paired supervised readout screens, with test truth held in the evaluator role."""
from collections import defaultdict
import gzip
import time
import numpy as np
from ..v18_3.io import write,canonical
from . import readout_model as M
from . import readout_data as D

def run(root,plan,pulse):
    timings=[];start=time.process_time();wall=time.monotonic()
    cfg=plan['design'];data=D.generate(root,cfg,pulse)
    timings.append(dict(phase='data-and-exact-reference',cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall))
    sizes=[3]*3+[6]*3 if cfg['domain']=='local' else [16,16]
    old_sizes=[8]*4 if cfg['domain']=='local' else [16]*5
    tiers=D.TIERS if cfg['domain']=='local' else ('artifact-history',)
    rows=[];fit_inventory=[];bank_checks=[]
    for draw in cfg['training_draws']:
        for tier in tiers:
            train=data[('train',draw,tier)];test=data[('development',draw,tier)]
            for seed in cfg['fit_seeds']:
                pulse(phase='old-outcome-fit',draw=draw,tier=tier,seed=seed)
                start=time.process_time();wall=time.monotonic()
                h,hw,hb=M.basis(train['x'],seed);ht,_,_=M.basis(test['x'],seed)
                old_weights=M.fit(h,train['old']);latent_basis=np.linalg.svd(old_weights,full_matrices=False)[0][:,:80]
                learned,_=M.probabilities(h@old_weights,old_sizes)
                learned_test,_=M.probabilities(ht@old_weights,old_sizes)
                latent=h@latent_basis;latent_test=ht@latent_basis
                bank_loss,_=M.scores(test['old'],learned_test,old_sizes)
                bank_prior=np.tile(train['old'].mean(0),(len(test['x']),1));prior_loss,_=M.scores(test['old'],bank_prior,old_sizes)
                bank_checks.append(dict(draw=draw,tier=tier,seed=seed,old_loss=float(bank_loss.mean()),
                    old_no_history_loss=float(prior_loss.mean()),old_bank_error=float(np.mean((learned_test-test['bank'])**2))))
                stem=f'{draw}-{tier}-{seed}'
                M.save_arrays(root/'models'/(stem+'-old.npz'),history_weights=hw,history_bias=hb,
                    old_head=old_weights,latent_basis=latent_basis)
                timings.append(dict(phase='old-fit-and-encoding',draw=draw,tier=tier,seed=seed,
                    cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall))
                pairs={'raw-history':(train['x'],test['x']),'learned-bank':(learned,learned_test),
                    'frozen-latent':(latent,latent_test),'exact-bank-oracle':(train['bank'],test['bank'])}
                for arm,(x,xt) in pairs.items():
                    features,rw,rb=M.basis(x,seed+17);test_features,_,_=M.basis(xt,seed+17)
                    for budget in cfg['budgets']:
                        pulse(phase='target-fit',draw=draw,tier=tier,seed=seed,arm=arm,budget=budget)
                        start=time.process_time();wall=time.monotonic()
                        if budget>len(x):raise ValueError('infeasible shared budget')
                        weights=M.fit(features[:budget],train['target'][:budget])
                        raw=test_features@weights;p,invalid=M.probabilities(raw,sizes)
                        loss,brier=M.scores(test['target'],p,sizes)
                        name=f'{stem}-{arm}-{budget}'
                        M.save_arrays(root/'models'/(name+'.npz'),readout_weights=rw,readout_bias=rb,target_head=weights)
                        M.save_arrays(root/'forecasts'/(name+'.npz'),probabilities=p,invalid_raw=invalid,ids=test['ids'])
                        fit_inventory.append(dict(draw=draw,tier=tier,seed=seed,arm=arm,budget=budget,
                            new_head_parameters=int(weights.size),old_head_parameters=int(old_weights.size),
                            raw_input_width=int(x.shape[1]),basis_dimension=M.HIDDEN,ridge=M.RIDGE))
                        add_rows(rows,test,loss,brier,invalid,draw,tier,seed,arm,budget)
                        timings.append(dict(phase='target-fit-score-save',draw=draw,tier=tier,seed=seed,arm=arm,budget=budget,
                            cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall))
                    # Same largest-budget shuffled target control for each arm.
                    budget=max(cfg['budgets']);order=np.random.default_rng(seed+draw).permutation(budget)
                    weights=M.fit(features[:budget],train['target'][:budget][order])
                    p,invalid=M.probabilities(test_features@weights,sizes);loss,brier=M.scores(test['target'],p,sizes)
                    M.save_arrays(root/'models'/(f'{stem}-{arm}-shuffled.npz'),target_head=weights,permutation=order)
                    M.save_arrays(root/'forecasts'/(f'{stem}-{arm}-shuffled.npz'),probabilities=p,ids=test['ids'])
                    add_rows(rows,test,loss,brier,invalid,draw,tier,seed,arm+'-shuffled',budget)
                for arm,p in [('exact-reference',test['exact']),('no-history',np.tile(train['target'].mean(0),(len(test['x']),1)))]:
                    p,_=M.probabilities(p,sizes);loss,brier=M.scores(test['target'],p,sizes)
                    M.save_arrays(root/'forecasts'/(f'{stem}-{arm}.npz'),probabilities=p,ids=test['ids'])
                    for budget in cfg['budgets']:add_rows(rows,test,loss,brier,np.zeros_like(loss),draw,tier,seed,arm,budget)
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/readout_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    grouped=defaultdict(list)
    for row in rows:
        key=tuple(row[k] for k in ('tier','arm','budget','world_class','query'))
        grouped[key].append(row)
    cells=[dict(zip(('tier','arm','budget','world_class','query'),key),loss=float(np.mean([r['loss'] for r in rr])),
        brier=float(np.mean([r['brier'] for r in rr])),raw_invalid=float(np.mean([r['raw_invalid'] for r in rr])),
        lineages=len({r['lineage'] for r in rr})) for key,rr in sorted(grouped.items())]
    # Pair within lineage after seed/draw averaging. Classes stay separate.
    cluster=defaultdict(list)
    for r in rows:cluster[tuple(r[k] for k in ('tier','arm','budget','world_class','lineage'))].append(r['loss'])
    means={k:float(np.mean(v)) for k,v in cluster.items()};contrasts=[];auc=[]
    for tier in tiers:
        classes=sorted({r['world_class'] for r in rows if r['tier']==tier})
        for cl in classes:
            for baseline in ('raw-history','frozen-latent'):
                vectors=[]
                for budget in cfg['budgets']:
                    v=[means[(tier,'learned-bank',budget,cl,l)]-means[(tier,baseline,budget,cl,l)] for l in cfg['development_lineages']]
                    vectors.append(v);contrasts.append(dict(tier=tier,world_class=cl,baseline=baseline,budget=budget,**interval(v)))
                values=np.trapezoid(np.array(vectors),x=np.log(cfg['budgets']),axis=0)/(np.log(cfg['budgets'][-1])-np.log(cfg['budgets'][0]))
                auc.append(dict(tier=tier,world_class=cl,baseline=baseline,**interval(values)))
    crossings=[]
    for tier in tiers:
        for cl in sorted({r['world_class'] for r in rows if r['tier']==tier}):
            for arm in ('raw-history','learned-bank','frozen-latent','exact-bank-oracle'):
                gaps=[float(np.mean([means[(tier,arm,b,cl,l)]-means[(tier,'exact-reference',b,cl,l)] for l in cfg['development_lineages']])) for b in cfg['budgets']]
                crossing=next((b for b,gap in zip(cfg['budgets'],gaps) if gap<=.02),None)
                crossings.append(dict(tier=tier,world_class=cl,arm=arm,first_budget_within_reference_plus_002=crossing,gaps=gaps,
                    interpretation='descriptive observed crossing; no monotonicity or confirmation assumed'))
    write(root/'TIMING.jsonl',dict(measurements=timings,scope='execution-specific measurements; source-bound separately from deterministic scientific replay',
        ledger='native attempt accounting is controlling; these component times are not added again'))
    return dict(controls=dict(M.controls(),**D.controls()),domain=cfg['domain'],rows=len(rows),cells=cells,
        contrasts=contrasts,normalized_log_budget_area=auc,fits=fit_inventory,bank_capability=bank_checks,
        label_target_crossings=crossings,execution_measurements='TIMING.jsonl',
        teacher_access='equal old observable distribution labels and new target labels on training histories; local goals explicitly evaluator-supervised',
        population='development lineages only; old cells equal weighted; initial local four evidence tiers; no test/confirmation lineages consumed',
        uncertainty='paired development lineage intervals conditional on two fixed feature seeds and training draws; raw retains variation separately',
        pursuit='evaluate capability and rival explanations before further architecture or confirmation admission',
        warrant='exploratory fixed-random-feature ridge screen; matched target-head parameters and available labels, unequal representation geometry; not tiny-reader admission; miniature — architecture untested')

def add_rows(rows,test,loss,brier,invalid,draw,tier,seed,arm,budget):
    for lineage in sorted(set(test['ids'][:,0])):
        classes=sorted(set(test['ids'][:,2]))
        for cl in classes+([-1] if len(classes)>1 else []):
            mask=(test['ids'][:,0]==lineage)&((test['ids'][:,2]==cl) if cl!=-1 else True)
            if not np.any(mask):continue
            for q in range(loss.shape[1]):
                rows.append(dict(draw=draw,tier=tier,seed=seed,arm=arm,budget=budget,lineage=int(lineage),world_class=int(cl),query=q,
                    histories=int(mask.sum()),loss=float(loss[mask,q].mean()),brier=float(brier[mask,q].mean()),
                    raw_invalid=float(invalid[mask,q].mean())))

def interval(values):
    v=np.asarray(values,float);r=np.random.default_rng(190501)
    boot=v[r.integers(len(v),size=(10000,len(v)))].mean(1)
    return dict(mean=float(v.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),lineages=len(v))
