"""Prespecified probability-valid heads on retained, frozen representations."""
from collections import defaultdict
import gzip
import time
import numpy as np
from ..v18_3.io import canonical, file_digest, write
from . import readout_model as M
from .readouts import add_rows, interval


def objective_gradient(x, y, w, sizes, penalty=.01):
    logits=x@w; logp=np.empty_like(logits)
    for start,size in zip(np.cumsum([0]+sizes[:-1]),sizes):
        a=logits[:,start:start+size];a=a-a.max(axis=1,keepdims=True)
        logp[:,start:start+size]=a-np.log(np.exp(a).sum(axis=1,keepdims=True))
    regularized=w.copy();regularized[0]=0
    objective=-np.sum(y*logp)/len(x)+.5*penalty*np.sum(regularized**2)
    gradient=x.T@(np.exp(logp)-y)/len(x)+penalty*regularized
    return float(objective),gradient,logp


def fit(x,y,sizes,pulse=lambda **kw:None):
    w=np.zeros((x.shape[1],y.shape[1]));trace=[]
    for step in range(200):
        objective,g,_=objective_gradient(x,y,w,sizes)
        if step%25==0:
            pulse(optimization_step=step)
            trace.append(dict(step=step,objective=objective,gradient_norm=float(np.linalg.norm(g))))
        w-=.1*g
    objective,g,_=objective_gradient(x,y,w,sizes)
    trace.append(dict(step=200,objective=objective,gradient_norm=float(np.linalg.norm(g))))
    return w,trace


def controls():
    x=np.column_stack((np.ones(20),np.tile([-1.,1.],10)))
    y=np.column_stack((x[:,1]<0,x[:,1]>0)).astype(float)
    w,_=fit(x,y,[2]);_,_,p=objective_gradient(x,y,w,[2])
    null=np.full((20,2),.5);nw,_=fit(x,null,[2])
    probe=np.array([[.1,-.2],[.3,.4]]);_,g,_=objective_gradient(x,y,probe,[2]);errors=[]
    for index in np.ndindex(probe.shape):
        a=probe.copy();b=probe.copy();a[index]+=1e-6;b[index]-=1e-6
        difference=(objective_gradient(x,y,a,[2])[0]-objective_gradient(x,y,b,[2])[0])/2e-6
        errors.append(abs(difference-g[index]))
    return dict(live_known_separation=bool(np.min(np.exp(p)[y.astype(bool)])>.94),
        placebo_constant_target=bool(np.max(abs(nw))<1e-12),
        positive_gradient=bool(max(errors)<1e-8),
        normalized_probabilities=bool(np.max(abs(np.exp(p).sum(1)-1))<1e-12))


def load(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}


def representations(data,old,old_sizes):
    h=np.column_stack((np.ones(len(data['x'])),np.tanh(M.padded(data['x'])@old['history_weights']+old['history_bias'])))
    bank,_=M.probabilities(h@old['old_head'],old_sizes)
    return {'raw-history':data['x'],'learned-bank':bank,'frozen-latent':h@old['latent_basis'],'exact-bank-oracle':data['bank']}


def run(root,plan,pulse,fit_function=fit):
    cfg=plan['design'];rows=[];fits=[];timing=[]
    for name,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=h:raise ValueError('retained input differs: '+name)
    for packet in cfg['packets']:
        source=root/'inputs'/packet['name'];domain=packet['domain']
        sizes=[3]*3+[6]*3 if domain=='local' else [16,16]
        old_sizes=[8]*4 if domain=='local' else [16]*5
        for draw in cfg['training_draws']:
            for tier in packet['tiers']:
                train=load(source/'evaluator'/f'train-{draw}-{tier}.npz')
                test=load(source/'evaluator'/f'development-{draw}-{tier}.npz')
                for seed in cfg['fit_seeds']:
                    stem=f'{draw}-{tier}-{seed}';old=load(source/'models'/(stem+'-old.npz'))
                    tr=representations(train,old,old_sizes);te=representations(test,old,old_sizes)
                    for arm in tr:
                        frozen=load(source/'models'/f'{stem}-{arm}-{min(packet["budgets"])}.npz')
                        def encode(a):return np.column_stack((np.ones(len(a)),np.tanh(M.padded(a)@frozen['readout_weights']+frozen['readout_bias'])))
                        x=encode(tr[arm]);xt=encode(te[arm])
                        for budget in packet['budgets']:
                            started=time.process_time()
                            def heartbeat(**kw):pulse(phase='probability-head',domain=domain,draw=draw,tier=tier,seed=seed,arm=arm,budget=budget,**kw)
                            w,trace=fit_function(x[:budget],train['target'][:budget],sizes,heartbeat)
                            _,_,logp=objective_gradient(xt,test['target'],w,sizes);p=np.exp(logp)
                            loss,brier=M.scores(test['target'],p,sizes)
                            key=f'{domain}-{stem}-{arm}-{budget}'
                            M.save_arrays(root/'models'/f'{key}.npz',head=w)
                            M.save_arrays(root/'forecasts'/f'{key}.npz',probabilities=p,ids=test['ids'])
                            add_rows(rows,test,loss,brier,np.zeros_like(loss),draw,tier,seed,arm,budget)
                            fits.append(dict(domain=domain,draw=draw,tier=tier,seed=seed,arm=arm,budget=budget,parameters=int(w.size),optimization=trace))
                            timing.append(dict(fit=key,cpu_seconds=time.process_time()-started))
                    for budget in packet['budgets']:
                        for arm,raw in [('nested-label-mean',np.tile(train['target'][:budget].mean(0),(len(test['x']),1))),('exact-reference',test['exact'])]:
                            p,_=M.probabilities(raw,sizes);loss,brier=M.scores(test['target'],p,sizes)
                            M.save_arrays(root/'forecasts'/f'{domain}-{stem}-{arm}-{budget}.npz',probabilities=p,ids=test['ids'])
                            add_rows(rows,test,loss,brier,np.zeros_like(loss),draw,tier,seed,arm,budget)
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/probability-head_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in ('tier','world_class','arm','budget','query'))].append(r['loss'])
    cells=[dict(zip(('tier','world_class','arm','budget','query'),k),loss=float(np.mean(v))) for k,v in sorted(grouped.items())]
    write(root/'OPTIMIZATION.json',dict(fits=fits,schedule=cfg.get('schedule','zero weights; 200 full-batch steps; learning rate .1; L2 .01; unpenalized intercept; no checkpoint selection')))
    write(root/'TIMING.jsonl',dict(measurements=timing,scope='component measurements; not added again to native accounting'))
    return dict(controls=controls(),rows=len(rows),cells=cells,fit_count=len(fits),
        interpretation='probability-valid target-head and regularization rival on unchanged retained data/features; original ridge result preserved',
        access='same old teachers and nested new labels; exact-bank diagnostic privileged; local goal supervision evaluator-labelled on training only',
        uncertainty='two retained feature seeds and two training draws; development only; not trained initialization replication or confirmation',
        warrant='exploratory constructed-method diagnostic; miniature — architecture untested',tiny_settings=0)
