"""B2: one self-generated input pass, same teachers, frozen B1 input maps."""
from collections import defaultdict
import gzip, time
import numpy as np
from ..v18_3.io import canonical, read, write, file_digest
from . import readout_model as M, recursive_update as U


def arrays(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def features(previous,codes,counts,w,b):
    x=np.concatenate((previous,np.eye(32)[codes],counts),axis=-1)
    x=x.reshape(-1,65)
    return np.column_stack((np.ones(len(x)),np.tanh(M.padded(x)@w+b)))


def rollout(a,model,head,teacher=False):
    n,t=a['codes'].shape;counts=U.counts_features(a['codes'],a['novel'])[:,:,32:]
    bank=np.tile(model['initial_bank'],(n,1));bad=np.zeros((n,4),bool)
    before=[];pred=[];invalid=[]
    for i in range(t):
        previous=a['before'][:,i] if teacher else bank
        before.append(previous.copy())
        f=features(previous,a['codes'][:,i],counts[:,i],model['update_weights'],model['update_bias'])
        p,inv=M.probabilities(f@head,[8]*4)
        bank=np.where(a['novel'][:,i,None],p,previous if teacher else bank)
        bad=np.where(a['novel'][:,i,None],inv,False if teacher else bad)
        pred.append(bank.copy());invalid.append(bad.copy())
    return np.stack(pred,axis=1),np.stack(invalid,axis=1),np.stack(before,axis=1)


def controls():
    checks=U.controls()
    x=np.array([[1.,0.],[1.,1.],[1.,2.]])
    y=np.array([[.8,.2],[.5,.5],[.2,.8]])
    checks['positive_refit_identity']=bool(np.array_equal(M.fit(x,y),M.fit(x,y)))
    return checks


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();assert all(checks.values())
    for n,h in cfg['input_files'].items():assert file_digest(root/'inputs'/n)==h,n
    data={(split,draw,q):arrays(root/'inputs/evaluator'/f'{split}-{draw}-{q}.npz') for split in ('train','development') for draw in cfg['training_draws'] for q in ('independent','copied')}
    rows=[];fits=[];timing=[]
    for draw in cfg['training_draws']:
        train={k:np.concatenate([data['train',draw,q][k] for q in ('independent','copied')]) for k in data['train',draw,'independent']}
        mask=train['novel'].reshape(-1);target=train['target'].reshape(-1,32)[mask]
        counts=U.counts_features(train['codes'],train['novel'])[:,:,32:]
        for seed in cfg['fit_seeds']:
            pulse(phase='roll-in-fit',draw=draw,seed=seed);start=time.process_time()
            model=arrays(root/'inputs/models'/f'{draw}-{seed}.npz')
            x=features(train['before'],train['codes'],counts,model['update_weights'],model['update_bias'])
            one=M.fit(x[mask],target)
            if not np.array_equal(one,model['update_head']):raise ValueError('retained one-step fit did not reproduce')
            _,_,before=rollout(train,model,one)
            rx=features(before,train['codes'],counts,model['update_weights'],model['update_bias'])
            rolled=M.fit(rx[mask],target);control=M.fit(x[mask],target)
            if not np.array_equal(one,control):raise ValueError('exact-teacher refit differed')
            M.save_arrays(root/'models'/f'{draw}-{seed}.npz',one_step=one,rolled_in=rolled,two_pass_teacher=control)
            M.save_arrays(root/'evaluator'/f'training-roll-in-{draw}-{seed}.npz',previous_bank=before,novel=train['novel'])
            fits.append(dict(draw=draw,seed=seed,labels_per_fit=int(mask.sum()),fits=3,head_parameters=int(one.size),one_step_reproduced=True,control_identity=True))
            timing.append(dict(phase='fits-and-training-rollout',draw=draw,seed=seed,cpu_seconds=time.process_time()-start))
            for q in ('independent','copied'):
                test=data['development',draw,q]
                for method,head in [('one-step',one),('rolled-in',rolled),('two-pass-teacher',control)]:
                    for teacher in (False,True):
                        pulse(phase='roll-in-evaluate',draw=draw,seed=seed,condition=q,method=method,teacher=teacher)
                        p,invalid,_=rollout(test,model,head,teacher)
                        indices=np.array(cfg['lengths'])-1;p=p[:,indices];invalid=invalid[:,indices]
                        mode='teacher-forced' if teacher else 'free-running'
                        M.save_arrays(root/'forecasts'/f'{draw}-{seed}-{q}-{method}-{mode}.npz',probabilities=p,invalid_raw=invalid,ids=test['ids'])
                        if method=='one-step':
                            arm='teacher-forced-update' if teacher else 'bank-update'
                            reference=arrays(root/'inputs/forecasts'/f'{draw}-{seed}-{q}-{arm}.npz')
                            if not np.array_equal(reference['probabilities'],p):raise ValueError('B1 forecast mismatch')
                        for j,length in enumerate(cfg['lengths']):
                            loss,brier=M.scores(test['target'][:,length-1],p[:,j],[8]*4)
                            for lineage in cfg['development_lineages']:
                                selected=test['ids'][:,0]==lineage
                                rows.append(dict(draw=draw,seed=seed,condition=q,method=method,mode=mode,length=length,lineage=lineage,streams=int(selected.sum()),loss=float(loss[selected].mean()),brier=float(brier[selected].mean()),invalid=float(invalid[selected,j].mean())))
    (root/'raw').mkdir(exist_ok=True);(root/'raw/roll-in_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    groups=defaultdict(list)
    for r in rows:groups[r['condition'],r['method'],r['mode'],r['length']].append(r)
    cells=[dict(condition=k[0],method=k[1],mode=k[2],length=k[3],**{m:float(np.mean([r[m] for r in rr])) for m in ('loss','brier','invalid')}) for k,rr in sorted(groups.items())]
    write(root/'FITS.json',dict(fits=fits));write(root/'TIMING.jsonl',dict(measurements=timing,accounting='component timings within native accounting; no double addition'))
    write(root/'INPUT_SCHEMA.json',dict(reader='unchanged B1 public streams; fitting uses training teachers only',evaluator='retained arrays, reference forecasts, training roll-in states, models, scores',privilege='teacher-forced preceding exact bank; duplicate positions are exact pass-through'))
    return dict(controls=checks,rows=len(rows),cells=cells,fits=fits,tiny_settings=0,
        scope='single roll-in pass at original lengths; same targets, changed preceding-bank training distribution and additional fit; no development selection',
        warrant='exploratory constructed-method comparison; miniature — architecture untested')
