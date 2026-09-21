"""B1 stationary/copy streams of the admitted local world's short episodes."""
from collections import defaultdict
import gzip,time
import numpy as np
from ..v18_3.io import canonical,write
from ..v18_3.world import rng
from . import local_world as L, readout_data as D, readout_model as M

def update(p,law,code,novel=True):
    if not novel:return p.copy()
    weights=p*law[:,int(code)];mass=weights.sum()
    if mass<=0:raise ValueError('impossible generated evidence')
    return weights/mass

def counts_features(codes,novel,recompute=False):
    n,t=codes.shape;out=np.empty((n,t,33));counts=np.zeros((n,32));rows=np.arange(n)
    for i in range(t):
        if recompute:
            counts[:]=0
            for j in range(i+1):counts[rows,codes[:,j]]+=novel[:,j]
        else:counts[rows,codes[:,i]]+=novel[:,i]
        total=counts.sum(1);out[:,i,:32]=counts/np.maximum(total[:,None],1)
        out[:,i,32]=np.log1p(total)/np.log(33)
    return out

def make_data(root,cfg,pulse):
    data={};timing=[]
    for split,lineages in [('train',cfg['train_lineages']),('development',cfg['development_lineages'])]:
        start=time.process_time();laws=[]
        for lineage in lineages:
            pulse(phase='update-endpoint-law',split=split,lineage=lineage)
            records=L.enumerate_world(L.law(lineage));laws.append(D.local_cache(records))
        law_array=np.stack(laws)
        M.save_arrays(root/'evaluator'/f'{split}-laws.npz',lineages=np.array(lineages),endpoint_laws=law_array)
        for draw in cfg['training_draws']:
            store={condition:defaultdict(list) for condition in ('independent','copied')}
            for lineage,law in zip(lineages,laws):
                random=rng('v19-B1-stream',split,lineage,draw)
                for stream in range(cfg['streams_per_lineage']):
                    maker=int(random.integers(16));contexts=random.integers(4,size=32)
                    fresh=np.array([int(ctx)*8+int(random.choice(8,p=law[maker,int(ctx)*8:int(ctx)*8+8])) for ctx in contexts])
                    for condition in store:
                        codes=fresh.copy();novel=np.ones(32,dtype=bool)
                        if condition=='copied':codes[1::2]=codes[::2];novel[1::2]=False
                        p=np.ones(16)/16;before=[];target=[]
                        for code,new in zip(codes,novel):
                            before.append(p@law);p=update(p,law,code,new);target.append(p@law)
                        s=store[condition];s['codes'].append(codes);s['novel'].append(novel);s['before'].append(before);s['target'].append(target)
                        s['prior'].append(law.mean(0));s['ids'].append([lineage,stream]);s['maker'].append(maker)
            for condition,s in store.items():
                a={k:np.asarray(v) for k,v in s.items()};data[split,draw,condition]=a
                name=f'{split}-{draw}-{condition}.npz'
                M.save_arrays(root/'evaluator'/name,**a)
                allowed={'codes':a['codes'],'novel':a['novel']}
                if split=='train':allowed.update(target=a['target'],before_bank_teacher=a['before'])
                M.save_arrays(root/'reader'/name,**allowed)
        timing.append(dict(phase='data',split=split,cpu_seconds=time.process_time()-start))
    return data,timing

def reservoir(codes,novel,seed):
    random=np.random.default_rng(seed);wi=random.normal(size=(33,128))/np.sqrt(33)
    wr=random.normal(size=(128,128));wr*=.5/np.linalg.norm(wr,2)
    h=np.zeros((len(codes),128));values=[];counts=np.zeros(len(codes));eye=np.eye(32)
    for t in range(32):
        counts+=novel[:,t];x=np.column_stack((eye[codes[:,t]],np.log1p(counts)/np.log(33)))
        next_h=np.tanh(x@wi+h@wr);h=np.where(novel[:,t,None],next_h,h);values.append(h.copy())
    return np.stack(values,axis=1),wi,wr

def run(root,plan,pulse):
    cfg=plan['design'];data,timing=make_data(root,cfg,pulse);rows=[];fits=[];eye=np.eye(32);sizes=[8]*4
    for draw in cfg['training_draws']:
        train={k:np.concatenate([data['train',draw,condition][k] for condition in ('independent','copied')]) for k in data['train',draw,'independent']}
        counts=counts_features(train['codes'],train['novel']);mask=train['novel'].reshape(-1)
        y=train['target'].reshape(-1,32)[mask];initial=train['prior'].mean(0);mean=y.mean(0)
        one=np.concatenate((train['before'],eye[train['codes']],counts[:,:,32:]),axis=2)
        for seed in cfg['fit_seeds']:
            pulse(phase='update-fit',draw=draw,seed=seed);start=time.process_time()
            direct,rw,rb=M.basis(counts.reshape(-1,33),seed)
            bx,bw,bb=M.basis(one.reshape(-1,65),seed+17)
            rec,wi,wr=reservoir(train['codes'],train['novel'],seed+31)
            rx=np.column_stack((np.ones(len(mask)),rec.reshape(-1,128)))
            dw=M.fit(direct[mask],y);uw=M.fit(bx[mask],y);hw=M.fit(rx[mask],y)
            M.save_arrays(root/'models'/f'{draw}-{seed}.npz',direct_weights=rw,direct_bias=rb,direct_head=dw,update_weights=bw,update_bias=bb,update_head=uw,reservoir_input=wi,reservoir_recurrent=wr,reservoir_head=hw,initial_bank=initial,mean=mean)
            fits.append(dict(draw=draw,seed=seed,training_labels=len(y),head_parameters=int(dw.size),teacher='identical target prefixes; updater has exact previous-bank training input'))
            timing.append(dict(phase='fit',draw=draw,seed=seed,cpu_seconds=time.process_time()-start))
            for condition in ('independent','copied'):
                test=data['development',draw,condition];n=len(test['codes']);start=time.process_time()
                full=counts_features(test['codes'],test['novel'],True);full_time=time.process_time()-start
                start=time.process_time();cached=counts_features(test['codes'],test['novel']);cached_time=time.process_time()-start
                if not np.array_equal(full,cached):raise ValueError('cached and reread features differ')
                pred={arm:[] for arm in ('full-history','cached-history','bank-update','teacher-forced-update','recurrent-state','no-history','exact-filter')};invalid={arm:[] for arm in pred}
                bank=np.tile(initial,(n,1));bank_bad=np.zeros((n,4),dtype=bool);h=np.zeros((n,128))
                for t in range(32):
                    pulse(phase='update-rollout',draw=draw,seed=seed,condition=condition,step=t)
                    x=np.column_stack((np.ones(n),np.tanh(M.padded(cached[:,t])@rw+rb)));direct_p,di=M.probabilities(x@dw,sizes)
                    code=eye[test['codes'][:,t]];count=cached[:,t,32:]
                    ux=np.concatenate((bank,code,count),axis=1);uf=np.column_stack((np.ones(n),np.tanh(M.padded(ux)@bw+bb)))
                    next_bank,bi=M.probabilities(uf@uw,sizes);bank=np.where(test['novel'][:,t,None],next_bank,bank)
                    bank_bad=np.where(test['novel'][:,t,None],bi,bank_bad)
                    tx=np.concatenate((test['before'][:,t],code,count),axis=1);tf=np.column_stack((np.ones(n),np.tanh(M.padded(tx)@bw+bb)))
                    teacher,ti=M.probabilities(tf@uw,sizes);teacher=np.where(test['novel'][:,t,None],teacher,test['before'][:,t])
                    next_h=np.tanh(np.column_stack((code,count))@wi+h@wr);h=np.where(test['novel'][:,t,None],next_h,h)
                    rp,ri=M.probabilities(np.column_stack((np.ones(n),h))@hw,sizes)
                    if t+1 in cfg['lengths']:
                        for arm,p,inv in [('full-history',direct_p,di),('cached-history',direct_p,di),('bank-update',bank,bank_bad),('teacher-forced-update',teacher,ti&test['novel'][:,t,None]),('recurrent-state',rp,ri),('no-history',np.tile(mean,(n,1)),np.zeros((n,4))),('exact-filter',test['target'][:,t],np.zeros((n,4)))]:
                            # Uniform repair is also applied to the two unlearned references.
                            if arm in ('no-history','exact-filter'):p,_=M.probabilities(p,sizes)
                            pred[arm].append(p.copy());invalid[arm].append(inv.copy())
                for arm,pp in pred.items():
                    forecasts=np.stack(pp,axis=1);bad=np.stack(invalid[arm],axis=1)
                    M.save_arrays(root/'forecasts'/f'{draw}-{seed}-{condition}-{arm}.npz',probabilities=forecasts,invalid_raw=bad,ids=test['ids'])
                    for j,length in enumerate(cfg['lengths']):
                        loss,brier=M.scores(test['target'][:,length-1],forecasts[:,j],sizes)
                        for lineage in cfg['development_lineages']:
                            selected=test['ids'][:,0]==lineage
                            rows.append(dict(draw=draw,seed=seed,condition=condition,arm=arm,length=length,lineage=lineage,streams=int(selected.sum()),loss=float(loss[selected].mean()),brier=float(brier[selected].mean()),invalid=float(bad[selected,j].mean())))
                timing.append(dict(phase='features',draw=draw,seed=seed,condition=condition,full_history_cpu_seconds=full_time,cached_cpu_seconds=cached_time))
    (root/'raw').mkdir(exist_ok=True);(root/'raw/update_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    groups=defaultdict(list)
    for r in rows:groups[r['condition'],r['arm'],r['length']].append(r)
    cells=[dict(condition=k[0],arm=k[1],length=k[2],loss=float(np.mean([r['loss'] for r in rr])),brier=float(np.mean([r['brier'] for r in rr])),invalid=float(np.mean([r['invalid'] for r in rr]))) for k,rr in sorted(groups.items())]
    write(root/'FITS.json',dict(fits=fits));write(root/'TIMING.jsonl',dict(measurements=timing,accounting='component measurements within native charge; do not add twice'))
    write(root/'COST_MODEL.json',dict(full_history='32 stored observation codes plus novelty flags; recompute each prefix; sum 1..32 prefix visits',cached='32 counts plus unique-count scalar; one update per new source',bank='32 forecast entries plus count and fitted parameters; one 128-coordinate update per new source',recurrent='128 state entries plus count and fixed/fitted parameters',caveat='state size excludes model parameters and public source deduplication; native timing and fit cost remain separate'))
    write(root/'INPUT_SCHEMA.json',dict(reader_train=['codes','novel','target','before_bank_teacher'],reader_development=['codes','novel'],excluded=['maker','ids','endpoint_laws','evaluation targets','evaluation exact previous bank'],task='fresh-episode endpoint probabilities; not preceding local-process recovery'))
    return dict(controls=controls(),rows=len(rows),cells=cells,fits=len(fits),population='eight development lineages, two feature seeds/two draws, eight streams per lineage; paired prefixes and copies; not confirmation',teachers='auxiliary exact observable forecast teachers, common targets; updater previous bank is extra training input; teacher-forced evaluation privileged',warrant='exploratory method diagnostic; miniature — architecture untested',scope='stationary persistent maker across independent reset episodes; no skill/purpose switch or joint local-process reconstruction',tiny_settings=0)

def controls():
    law=np.tile(np.array([[.9,.1],[.1,.9]]),(1,16));p=np.array([.5,.5])
    a=update(p,law,0);b=update(a,law,0);expected=p*law[:,0]**2;expected/=expected.sum()
    codes=np.array([[0,0,1,1]+[0]*28]);novel=np.array([[True,False,True,False]+[False]*28])
    ca=counts_features(codes,novel);fu=counts_features(codes,novel,True)
    return dict(live_informative_update=bool(a[0]>.5),placebo_uninformative=bool(np.allclose(update(p,np.ones((2,32)),0),p)),
        positive_sequential_likelihood=bool(np.allclose(b,expected)),duplicate_identity=bool(np.array_equal(update(a,law,0,False),a)),
        cached_full_identity=bool(np.array_equal(ca,fu)),copy_features_unchanged=bool(np.array_equal(ca[:,0],ca[:,1])))
