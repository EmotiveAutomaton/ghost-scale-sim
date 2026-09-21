"""B2 successor: fixed cached-history resets and a retained-prefix horizon test."""
from collections import defaultdict
import gzip,time
import numpy as np
from ..v18_3.io import canonical,write,file_digest
from ..v18_3.world import rng
from . import readout_model as M,recursive_update as U
from .roll_in import arrays,features


def extend(a,laws,draw,condition,length=128):
    """Preserve the complete 32-observation parent and append paired reset episodes."""
    n=len(a['codes']);codes=np.empty((n,length),int);codes[:,:32]=a['codes']
    novel=np.ones((n,length),bool)
    if condition=='copied':novel[:,1::2]=False
    before=np.empty((n,length,32));target=np.empty_like(before)
    for j,(lineage,stream) in enumerate(a['ids']):
        law=laws[int(lineage)];maker=int(a['maker'][j])
        random=rng('v19-B2-horizon-tail',int(lineage),int(draw),int(stream))
        contexts=random.integers(4,size=length-32)
        tail=np.array([int(q)*8+int(random.choice(8,p=law[maker,int(q)*8:int(q)*8+8])) for q in contexts])
        if condition=='copied':tail[1::2]=tail[::2]
        codes[j,32:]=tail;p=np.ones(16)/16
        for t,code in enumerate(codes[j]):
            before[j,t]=p@law;p=U.update(p,law,code,bool(novel[j,t]));target[j,t]=p@law
    if not np.array_equal(novel[:,:32],a['novel']):raise ValueError('parent source identity changed')
    if not np.allclose(before[:,:32],a['before'],rtol=0,atol=1e-12) or not np.allclose(target[:,:32],a['target'],rtol=0,atol=1e-12):raise ValueError('parent filter failed to reproduce')
    return dict(a,codes=codes,novel=novel,before=before,target=target)


def reset_due(count,novel,period):
    return novel & (count>0) & (count%period==0)


def evaluate(a,m,heads,period=8):
    n,t=a['codes'].shape;count=np.zeros(n);counts=np.zeros((n,32));eye=np.eye(32)
    names=['one-step','rolled-in','one-step-reset','rolled-in-reset','cached-history','exact-filter','teacher-one-step','teacher-rolled-in']
    banks={k:np.tile(m['initial_bank'],(n,1)) for k in names[:4]};bad={k:np.zeros((n,4),bool) for k in names[:4]}
    forecasts={k:[] for k in names};invalid={k:[] for k in names};resets=[];cache_before=[]
    def cached():
        x=np.column_stack((counts/np.maximum(count[:,None],1),np.log1p(count)/np.log(33)))
        f=np.column_stack((np.ones(n),np.tanh(M.padded(x)@m['direct_weights']+m['direct_bias'])))
        return M.probabilities(f@m['direct_head'],[8]*4)
    for i in range(t):
        new=a['novel'][:,i];due=reset_due(count,new,period);cp,ci=cached();cache_before.append(cp.copy());resets.append(due.copy())
        for name in banks:
            if name.endswith('-reset'):
                banks[name]=np.where(due[:,None],cp,banks[name]);bad[name]=np.where(due[:,None],ci,bad[name])
        counts+=eye[a['codes'][:,i]]*new[:,None];count+=new;scalar=(np.log1p(count)/np.log(33))[:,None]
        for name in banks:
            head=heads['rolled_in' if name.startswith('rolled-in') else 'one_step']
            f=features(banks[name],a['codes'][:,i],scalar,m['update_weights'],m['update_bias'])
            p,inv=M.probabilities(f@head,[8]*4);banks[name]=np.where(new[:,None],p,banks[name]);bad[name]=np.where(new[:,None],inv,bad[name])
            forecasts[name].append(banks[name].copy());invalid[name].append(bad[name].copy())
        cp,ci=cached();forecasts['cached-history'].append(cp);invalid['cached-history'].append(ci)
        ep,ei=M.probabilities(a['target'][:,i],[8]*4);forecasts['exact-filter'].append(ep);invalid['exact-filter'].append(np.zeros_like(ei))
        f=features(a['before'][:,i],a['codes'][:,i],scalar,m['update_weights'],m['update_bias'])
        for name,head in [('teacher-one-step',heads['one_step']),('teacher-rolled-in',heads['rolled_in'])]:
            p,inv=M.probabilities(f@head,[8]*4);p=np.where(new[:,None],p,a['before'][:,i]);inv=inv&new[:,None]
            forecasts[name].append(p);invalid[name].append(inv)
    return {k:np.stack(v,1) for k,v in forecasts.items()},{k:np.stack(v,1) for k,v in invalid.items()},np.stack(resets,1),np.stack(cache_before,1)


def controls():
    checks=U.controls()
    checks['reset_after_eight_only']=bool(np.array_equal(reset_due(np.array([0,7,8,8,16]),np.array([1,1,1,0,1],bool),8),[False,False,True,False,True]))
    return checks


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();assert all(checks.values())
    for n,h in cfg['input_files'].items():assert file_digest(root/'inputs'/n)==h,n
    laws=arrays(root/'inputs/evaluator/development-laws.npz');laws=dict(zip(laws['lineages'],laws['endpoint_laws']))
    data={};timing=[]
    for draw in cfg['training_draws']:
        for condition in ('independent','copied'):
            pulse(phase='extend-retained-stream',draw=draw,condition=condition)
            a=arrays(root/'inputs/evaluator'/f'development-{draw}-{condition}.npz')
            a=extend(a,laws,draw,condition,cfg['horizon']);data[draw,condition]=a
            M.save_arrays(root/'evaluator'/f'{draw}-{condition}.npz',**a)
            M.save_arrays(root/'reader'/f'{draw}-{condition}.npz',codes=a['codes'],novel=a['novel'])
    rows=[];costs=[]
    for draw in cfg['training_draws']:
        for seed in cfg['fit_seeds']:
            m=arrays(root/'inputs/models'/f'{draw}-{seed}.npz');heads=arrays(root/'inputs/roll-in-models'/f'{draw}-{seed}.npz')
            for condition in ('independent','copied'):
                pulse(phase='fixed-reset-evaluation',draw=draw,seed=seed,condition=condition);start=time.process_time()
                a=data[draw,condition];pred,bad,resets,cached=evaluate(a,m,heads,cfg['reset_period'])
                M.save_arrays(root/'evaluator'/f'resets-{draw}-{seed}-{condition}.npz',reset_before_observation=resets,cached_before=cached)
                for method in ('one-step','rolled-in'):
                    old=arrays(root/'inputs/forecasts'/f'{draw}-{seed}-{condition}-{method}-free-running.npz')
                    if not np.array_equal(pred[method][:,np.array([8,32])-1],old['probabilities']):raise ValueError('retained free-running forecast differs')
                for method,p in pred.items():
                    M.save_arrays(root/'forecasts'/f'{draw}-{seed}-{condition}-{method}.npz',probabilities=p,invalid_raw=bad[method],ids=a['ids'])
                    for length in cfg['lengths']:
                        loss,brier=M.scores(a['target'][:,length-1],p[:,length-1],[8]*4)
                        for lineage in cfg['development_lineages']:
                            mask=a['ids'][:,0]==lineage
                            rows.append(dict(draw=draw,seed=seed,condition=condition,method=method,length=length,lineage=lineage,streams=int(mask.sum()),loss=float(loss[mask].mean()),brier=float(brier[mask].mean()),invalid=float(bad[method][mask,length-1].mean())))
                for length in cfg['lengths']:
                    costs.append(dict(draw=draw,seed=seed,condition=condition,length=length,streams=len(a['codes']),updates_per_stream=int(a['novel'][0,:length].sum()),resets_per_stream=int(resets[0,:length].sum()),count_scalar=float(np.log1p(a['novel'][0,:length].sum())/np.log(33))))
                timing.append(dict(draw=draw,seed=seed,condition=condition,cpu_seconds=time.process_time()-start))
    (root/'raw').mkdir(exist_ok=True);(root/'raw/reset_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    groups=defaultdict(list)
    for r in rows:groups[r['condition'],r['method'],r['length']].append(r)
    cells=[dict(condition=k[0],method=k[1],length=k[2],**{m:float(np.mean([r[m] for r in rr])) for m in ('loss','brier','invalid')}) for k,rr in sorted(groups.items())]
    write(root/'COSTS.json',dict(counts=costs,note='reset requires maintained 32-bin history cache and an additional frozen direct-head call; common diagnostic calls in this run are not the deployed reset cost; no measured break-even claim'))
    write(root/'TIMING.jsonl',dict(measurements=timing,accounting='within native charge; no double addition'))
    write(root/'INPUT_SCHEMA.json',dict(reader='only reader/*.npz codes/novel; same parent training teachers available by reference',evaluator='all other files including fitted parameters, laws, makers, target probabilities, resets and forecasts'))
    return dict(controls=checks,cells=cells,rows=len(rows),fits=0,tiny_settings=0,scope='one fixed history-cache reset every eight new sources; same frozen heads; 32-prefix identity then 128-horizon extrapolation with original count scaling; stationary makers only',warrant='exploratory constructed-method diagnostic; miniature — architecture untested')
