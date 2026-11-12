"""Library, own-history and repeated probabilistic reading: original implementation."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .common import rng,softmax,mean,auc_blocks,row,read,write,digest

READERS=('J','Ps','I1','B1','B2','Pp','Pe','PeL1','PeL4','PpS32','PpS128','PpSL32','PpSL128')
MARKS=('J','Pp','Pe','PeL4')
METRICS=('log_score','gap_to_J','true_goal_below_one_percent','accuracy','top_probability_minus_accuracy')


def dimensions(size):
    return (12,8,4,3,64,160,(0,1,2,4,8,16)) if size=='small' else (200,32,8,10,128,40,(0,1,4,16))


def groups():
    return [dict(size=size,rho=rho,overlap=overlap) for size,rhos in
        [('small',(.2,.3,.4)),('large',(.05,.10,.20))] for rho in rhos for overlap in ('low','high')]


def generate(setting,block,lane='study',worlds=50):
    v,types,goals,target,length,_,_=dimensions(setting['size'])
    r=rng('G1',lane,setting,block);rho=setting['rho']
    h=r.dirichlet(np.full(v,.3),size=types)
    tg=np.full((goals,v),.1/v)
    remaining=list(r.permutation(v)) if setting['overlap']=='low' else list(range(v))
    for g in range(goals):
        chosen=remaining[:target] if setting['overlap']=='low' else sorted(remaining,key=lambda x:(-h[g,x],x))[:target]
        tg[g,chosen]+=.9/target;remaining=[x for x in remaining if x not in chosen]
    maker=r.integers(types,size=worlds);truth=r.integers(goals,size=(worlds,17))
    sources=r.random((worlds,length))<rho
    tokens=np.empty((worlds,length),dtype=np.int16);counts=np.zeros((worlds,17,v),dtype=np.int16)
    for w in range(worlds):
        a=r.choice(v,size=length,p=h[maker[w]])
        b=r.choice(v,size=length,p=tg[truth[w,0]])
        tokens[w]=np.where(sources[w],b,a)
        counts[w,0]=np.bincount(tokens[w],minlength=v)
        for d in range(1,17):
            counts[w,d]=r.multinomial(length,(1-rho)*h[maker[w]]+rho*tg[truth[w,d]])
    return dict(h=h,goals=tg,counts=counts,truth=truth,maker=maker,tokens=tokens,sources=sources)


def estimated_goal(counts,h,tg,rho):
    mix=(1-rho)*h[:,None,:]+rho*tg[None,:,:]
    ll=np.einsum('wv,wgv->wg',counts,np.log(mix))
    return softmax(ll),mix


def refine(counts,h,tg,rho,prior,rounds):
    for _ in range(rounds):
        mix=(1-rho)*h[:,None,:]+rho*tg[None,:,:]
        posterior=softmax(np.einsum('wdv,wgv->wdg',counts,np.log(mix)))
        weighted=np.einsum('wdv,wdg->wgv',counts,posterior)
        habit_counts=np.sum(weighted*((1-rho)*h[:,None,:]/mix),axis=1)
        h=habit_counts+prior;h/=h.sum(axis=1,keepdims=True)
    return h


def preserve_count_symmetry(raw,k,probabilities):
    """Project only provably equal empirical goals; never merge nearby scores.

    Permuting target moves with identical document-count vectors leaves the
    uniform pseudo-count empirical model invariant. Equal multisets therefore
    imply equal goal probabilities. Lowest goal index breaks the exact tie.
    """
    targets=raw['goals'];counts=raw['counts'][:,:k+1]
    for w in range(len(counts)):
        classes={}
        for g,target in enumerate(targets):
            signature=tuple(sorted(tuple(x) for x in counts[w][:,target>target.min()].T))
            classes.setdefault(signature,[]).append(g)
        for indices in classes.values():
            if len(indices)>1:
                for reader in ('Pe','PeL1','PeL4'):
                    p=probabilities[reader]
                    p[w,indices]=p[w,indices].mean()


def readers(raw,k,rho,full_beam=False):
    h,tg=raw['h'],raw['goals'];counts=raw['counts'][:,:k+1].astype(float)
    page=counts[:,0];mix=(1-rho)*h[:,None,:]+rho*tg[None,:,:]
    library_ll=np.einsum('wdv,igv->wdig',counts,np.log(mix))
    history=logsumexp(library_ll[:,1:],axis=3).sum(axis=1)-k*np.log(len(tg))
    history-=logsumexp(history,axis=1,keepdims=True)
    page_ll=library_ll[:,0]
    joint=softmax((history[:,:,None]+page_ll).reshape(len(page),-1)).reshape(page_ll.shape)
    p={'J':joint.sum(axis=1)}
    q=np.einsum('wv,iv->wi',counts.sum(axis=1),np.log(h));q-=logsumexp(q,axis=1,keepdims=True)
    ps=logsumexp(q[:,:,None]+page_ll,axis=1);ps-=logsumexp(ps,axis=1,keepdims=True)
    p['Ps']=np.exp(ps)
    qi=history+logsumexp(ps[:,None,:]+page_ll,axis=2);qi-=logsumexp(qi,axis=1,keepdims=True)
    p['I1']=softmax(logsumexp(qi[:,:,None]+page_ll,axis=1))
    ordering=np.argsort(-(history+page@np.log(h).T),axis=1,kind='stable')
    for beam in ((1,2,len(h)) if full_beam else (1,2)):
        selected=ordering[:,:beam];prior=np.take_along_axis(history,selected,axis=1)
        prior-=logsumexp(prior,axis=1,keepdims=True)
        selected_ll=np.take_along_axis(page_ll,selected[:,:,None],axis=1)
        p['B'+str(beam)]=softmax(logsumexp(prior[:,:,None]+selected_ll,axis=1))
    total=counts.sum(axis=1);typical=h.mean(axis=0)
    habits={'Pp':np.broadcast_to(typical,total.shape),'Pe':(total+.5)/(total.sum(axis=1,keepdims=True)+.5*h.shape[1])}
    habits['PeL1']=refine(counts,habits['Pe'],tg,rho,.5,1)
    habits['PeL4']=refine(counts,habits['PeL1'],tg,rho,.5,3)
    for a in (32,128):
        estimate=total+a*typical;estimate/=estimate.sum(axis=1,keepdims=True)
        habits['PpS'+str(a)]=estimate
        habits['PpSL'+str(a)]=refine(counts,estimate,tg,rho,a*typical,4)
    for label,estimate in habits.items():p[label]=estimated_goal(page,estimate,tg,rho)[0]
    preserve_count_symmetry(raw,k,p)
    # One mark score per move and world, looked up for every retained page token.
    depart=np.einsum('wig,igv->wv',joint,rho*tg[None,:,:]/mix)
    marks={'J':np.take_along_axis(depart,raw['tokens'],axis=1)}
    for label in MARKS[1:]:
        target=tg[np.argmax(p[label],axis=1)]
        depart=rho*target/((1-rho)*habits[label]+rho*target)
        marks[label]=np.take_along_axis(depart,raw['tokens'],axis=1)
    return p,marks


def block_measures(raw,probabilities):
    truth=raw['truth'][:,0];j=np.log(probabilities['J'][np.arange(len(truth)),truth]);out=[]
    for reader in READERS:
        p=probabilities[reader];true=p[np.arange(len(truth)),truth];score=np.log(true)
        accuracy=(p.argmax(axis=1)==truth).astype(float)
        out.append([score.mean(),(j-score).mean(),(true<.01).mean(),accuracy.mean(),(p.max(axis=1)-accuracy).mean()])
    value=np.asarray(out)
    if not np.isfinite(value).all():raise ArithmeticError('nonfinite G1 score; no flooring permitted')
    return value


def run(destination,mode='original',original=None,pulse=lambda **kw:None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    all_rows=[];files={}
    from . import g1_reference
    for group_index,setting in enumerate(groups()):
        _,_,_,_,_,blocks,ks=dimensions(setting['size'])
        measurements={k:[] for k in ks};marked={k:{r:[] for r in MARKS} for k in ks};labels=[]
        for b in range(blocks):
            name=f'g{group_index:02d}-b{b:03d}_points.npz'
            if mode=='reconstruct':
                with np.load(Path(original)/name,allow_pickle=False) as saved:raw={key:saved[key] for key in saved.files}
            else:
                raw=generate(setting,b);np.savez_compressed(destination/name,**raw);files[name]=digest(destination/name)
            labels.append(raw['sources'])
            for k in ks:
                probs,marks=(g1_reference.readers(raw,k,setting['rho']) if mode=='reconstruct' else readers(raw,k,setting['rho']))
                measurements[k].append(block_measures(raw,probs))
                for label in MARKS:marked[k][label].append(marks[label])
            pulse(family='G1',group=group_index+1,groups=12,block=b+1,blocks=blocks)
        for k in ks:
            cell=setting|dict(K=k);units=np.asarray(measurements[k])
            for ri,reader in enumerate(READERS):
                for mi,metric in enumerate(METRICS):
                    all_rows.append(row(cell,metric,mean(units[:,ri,mi]),reader in ('J','Pp'),reader=reader))
            for reader in MARKS:
                scores=np.asarray(marked[k][reader])
                # A declared numerical representation makes exact midranks
                # reproducible across independent floating-point contractions.
                statistic=auc_blocks(np.round(scores,12),np.asarray(labels))
                coarse=auc_blocks(np.round(scores,10),np.asarray(labels))
                statistic['ranking_decimal_places']=12
                statistic['resolution_10_vs_12_absolute_difference']=abs(coarse['estimate']-statistic['estimate'])
                all_rows.append(row(cell,'departure_auc',statistic,reader=='Pp',reader=reader))
            contrast=units[:,READERS.index('PeL4'),0]-units[:,READERS.index('Pe'),0]
            all_rows.append(row(cell,'PeL4_minus_Pe_log_score',mean(contrast),False,reader='contrast'))
    result=dict(family='G1',rows=all_rows,interpretation='descriptive; no registered claim',
        independent_units='library blocks of 50 worlds',paired_K=True)
    write(destination/'SUMMARY.json',result)
    if mode!='reconstruct':write(destination/'RAW_MANIFEST.json',files)
    return result


def controls(destination,pulse=lambda **kw:None):
    gates=[]
    def record(name,passed,**extra):
        gates.append(dict(name=name,passed=bool(passed),**extra));pulse(control=name)
    worst=0.;beam_error=0.;normalization=0.
    from . import g1_reference
    for size in ('small','large'):
        for overlap in ('low','high'):
            s=dict(size=size,rho=.3 if size=='small' else .1,overlap=overlap)
            raw=generate(s,0,'controls-arithmetic')
            for k in (0,4,16):
                p,m=readers(raw,k,s['rho'],True);q,n=g1_reference.readers(raw,k,s['rho'])
                beam_error=max(beam_error,float(np.max(abs(p['B'+str(len(raw['h']))]-p['J']))))
                for label in READERS:
                    worst=max(worst,float(np.max(abs(p[label]-q[label]))))
                    normalization=max(normalization,float(np.max(abs(p[label].sum(axis=1)-1))))
                for label in MARKS:worst=max(worst,float(np.max(abs(m[label]-n[label]))))
    record('full_library_beam_equals_J',beam_error<=1e-12,error=beam_error)
    record('independent_reader_and_mark_arithmetic',worst<=1e-10,error=worst)
    record('all_posteriors_normalized',normalization<=1e-12,error=normalization)
    # Separate fixed excluded blocks; no controls enter the study.
    for size in ('small','large'):
        calibration=[]
        s=dict(size=size,rho=.3 if size=='small' else .1,overlap='high')
        for b in range(160 if size=='small' else 40):
            raw=generate(s,b,'controls-calibration');p,_=readers(raw,0,s['rho'])
            calibration.append(block_measures(raw,p)[0,4])
            if b%10==0:pulse(control='calibration',size=size,block=b)
        stat=mean(calibration)
        record('exact_J_calibration_'+size,abs(stat['estimate'])<=3*stat['standard_error']+1e-12,**stat)
    for size in ('small','large'):
        for overlap in ('low','high'):
            gaps=[];s=dict(size=size,rho=.97,overlap=overlap)
            for b in range(16):
                raw=generate(s,b,'controls-goal-dominated')
                for k in (0,16):
                    p,_=readers(raw,k,.97);gaps.append(block_measures(raw,p)[:,1])
            values=np.mean(gaps,axis=0)
            record('goal_dominated_'+size+'_'+overlap,bool(np.all(abs(values)<.005)),gaps=dict(zip(READERS,values.tolist())),blocks=16)
    gaps=[];s=dict(size='small',rho=.4,overlap='low')
    for b in range(80):
        raw=generate(s,b,'controls-history');p,_=readers(raw,16,.4)
        gaps.append(block_measures(raw,p)[READERS.index('PeL4'),1])
        if b%10==0:pulse(control='history',block=b)
    stat=mean(gaps);record('own_history_loop_near_J',abs(stat['estimate'])<.02,**stat)
    result=dict(family='G1',passed=all(g['passed'] for g in gates),gates=gates)
    write(Path(destination)/'CONTROLS.json',result);return result
