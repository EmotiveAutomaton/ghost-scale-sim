"""Complete G0-G7 consumers with retained forecasts and independent arithmetic."""
from collections import defaultdict
from pathlib import Path
import gzip,json,time
import numpy as np
from ..v18_3.io import canonical,write,digest,file_digest
from . import world as W,checker as C
from .readers import Readers,normalize,score

METHODS=('prior','template','direct-table','direct-mlp','factorized-mlp','structured-joint','oracle')

def reference(rows,tier,review_observed=False):
    keys,masses,inv=W.group_reference(rows,tier,review_observed)
    joint=np.zeros((len(keys),32,128));np.add.at(joint,(inv,rows[:,0].astype(int),rows[:,10].astype(int)),rows[:,-1])
    by_maker=joint.sum(2);maker_prior=by_maker.sum(0)
    return dict(keys={tuple(k):i for i,k in enumerate(keys)},joint=joint,
        likelihood=by_maker/maker_prior[None,:],masses=masses)

def inputs(rows,chosen,tier,history,identity,review_observed=False):
    """Generate only observed old episodes from the current persistent maker.

    The generator uses a latent maker to generate data; it is never a feature.
    Complete past records do not reveal their private aims.
    """
    x=W.project(chosen,tier,review_observed);parts=[x];rng=np.random.default_rng(W.seed('history',identity))
    groups={m:np.flatnonzero(rows[:,0]==m) for m in range(32)}
    history_indices=[]
    for step in range(history):
        inds=np.empty(len(chosen),int)
        for mi,rr in groups.items():
            positions=np.flatnonzero(chosen[:,0]==mi)
            if len(positions):inds[positions]=rng.choice(rr,size=len(positions),p=normalize(rows[rr,-1]))
        history_indices.append(inds);parts.append(W.project(rows[inds],tier,review_observed))
    return np.concatenate(parts,axis=1),history_indices

def posterior(ref,current_x,history_rows):
    key_indices=np.array([ref['keys'][tuple(k)] for k in current_x]);j=ref['joint'][key_indices]
    log=np.zeros((len(current_x),32))
    for x in history_rows:
        indices=[ref['keys'][tuple(k)] for k in x]
        with np.errstate(divide='ignore'):log+=np.log(ref['likelihood'][indices])
    log-=log.max(1,keepdims=True);maker_like=np.exp(log)
    out=(j*maker_like[:,:,None]).sum(1)
    return normalize(out),out>0

def training(design,tier,pulse):
    all_x=[];all_y=[];n=design['train_cases'];lineages=design['training_lineages'];hist=design.get('history',0)
    for index,lineage in enumerate(lineages):
        rows=W.enumerate_world(lineage,design.get('training_shift','native'))
        count=n//len(lineages)+(index<n%len(lineages));sample=W.sampled(rows,count,'train',design['fit_seed'],lineage,n)
        x,_=inputs(rows,sample,tier,hist,('train',design['fit_seed'],lineage,n),design.get('review_observed',False))
        all_x.append(x);all_y.append(sample[:,10].astype(int));pulse(phase='training-data',lineage=lineage)
    return np.concatenate(all_x),np.concatenate(all_y)

def write_forecasts(root,name,p,y,details):
    metrics=score(p,y);checked=C.reaggregate(p,y,metrics)
    path=root/'raw'/f'{name}_points.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,prediction=p,truth=y,**metrics)
    write(root/'raw'/f'{name}.json',dict(details,sha256=file_digest(path),independent_reaggregation=checked))
    return {k:float(np.mean(v)) for k,v in metrics.items()}

def core(root,design,pulse):
    tier=design['tier'];history=design.get('history',0);x,y=training(design,tier,pulse)
    model=Readers(x,y,design.get('epochs',120),design['fit_seed'],pulse)
    (root/'training').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'training/examples_points.npz',evidence=x,labels=y)
    write(root/'FIT_COST.json',model.metadata());summaries=[];casebook=[]
    for lineage in design['lineages']:
        pulse(phase='evaluation',lineage=lineage);shift=design.get('shift','native');rows=W.enumerate_world(lineage,shift)
        check=C.verify_world(rows,W.law(lineage,shift),W.MAKERS,W.CONTEXTS)
        sample=W.sampled(rows,design['test_cases'],'test',lineage,shift)
        xx,hh=inputs(rows,sample,tier,history,('test',lineage,shift),design.get('review_observed',False))
        ref=reference(rows,tier,design.get('review_observed',False));oracle,support=posterior(ref,xx[:,:13],[W.project(rows[h],tier,design.get('review_observed',False)) for h in hh]);yy=sample[:,10].astype(int)
        C.verify_posterior(rows,xx,tier,W.CONTEXTS,oracle,design.get('review_observed',False))
        (root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(parents=True,exist_ok=True)
        np.savez_compressed(root/'reader'/f'{lineage}_points.npz',evidence=xx)
        np.savez_compressed(root/'evaluator'/f'{lineage}_points.npz',hidden_rows=sample,history_indices=np.asarray(hh,dtype=np.int32))
        for method in METHODS:
            started=time.process_time();p=oracle if method=='oracle' else model.predict(xx,method,support)
            detail=dict(lineage=lineage,tier=tier,method=method,history=history,fit_seed=design['fit_seed'],shift=shift)
            result=write_forecasts(root,f'{lineage}-{method}',p,yy,detail)
            summaries.append(dict(detail,metrics=result))
            write(root/'cost'/f'{lineage}-{method}.json',dict(cpu_seconds=time.process_time()-started,forecast_rows=len(yy)))
            if method=='structured-joint':
                for idx in (0,int(np.argmax(p.max(1))),int(np.argmin(p[np.arange(len(yy)),yy]))):
                    casebook.append(dict(case_id=digest([lineage,shift,idx])[:24],evidence=xx[idx].tolist(),method=method,
                        prediction=p[idx].tolist(),truth=int(yy[idx]),role='evaluator casebook; NOT reader input'))
        write(root/'checks'/f'{lineage}.json',check)
        pulse(phase='retained',lineage=lineage)
    write(root/'CASEBOOK_EVALUATOR.json',casebook)
    write(root/'WITNESSES.json',C.exact_witnesses())
    return dict(branch=design['branch'],population=summaries,scope='Constructed method; miniature architecture; coefficient lineages are not humans',
        primary='G1/G7 context joint-history log loss; structured-joint versus both direct readers, must beat the stronger direct and template rival',
        practical_margin=.02,confirmation=design.get('split')=='confirmation',controls=controls())

def controls():
    w=C.exact_witnesses();p=np.eye(128)[:2];y=np.array([0,1]);m=score(p,y);C.reaggregate(p,y,m)
    malformed=False
    try:C.probabilities([[.3,.3]])
    except ValueError:malformed=True
    corrupt=False
    try:C.reaggregate(p,y,dict(m,log_loss=np.ones(2)))
    except ValueError:corrupt=True
    prior=np.array([[.2,.8]]);independent=normalize(prior*np.array([[.5,.5]]))
    return dict(live_revealing_recovery=bool(np.all(m['correct']==1)),placebo_independent=bool(np.array_equal(prior,independent)),
        positive_joint_witness=w['same_marginals_different_joints'],corrupt_probability_rejected=malformed,corrupt_score_rejected=corrupt)

def cue_inputs(x,labels,kind,reliability,rng,copied=None):
    out=x.copy()
    if kind=='event':out[:,6]=W.LABELS[labels,0]
    elif kind in ('context','repeat'):
        target=W.LABELS[labels,5];value=np.where(rng.random(len(labels))<reliability,target,1-target)
        if copied is not None:value=copied
        out[:,11 if kind=='context' else 12]=value
    elif kind!='none':raise ValueError('unknown cue')
    return out

def cue_model(design,pulse):
    x,y=training(dict(design,history=0),'context',pulse);rng=np.random.default_rng(W.seed('cue-train',design['fit_seed']))
    # One random observation pattern per training case. All labels/cases shared.
    kinds=rng.integers(4,size=len(y))
    for j,kind in enumerate(('none','event','context','repeat')):
        mask=kinds==j;x[mask]=cue_inputs(x[mask],y[mask],kind,.8,rng)
    m=Readers(x,y,design.get('epochs',120),design['fit_seed'],pulse)
    available=x[:,11]>=0;estimated=float((x[available,11]==W.LABELS[y[available],5]).mean()) if available.any() else .5
    return m,estimated,x,y

def information(p,bit,reliability):
    m=p@W.LABELS[:,bit];q=m*reliability+(1-m)*(1-reliability)
    entropy=lambda v:-(np.where(v>0,v*np.log(np.maximum(v,1e-300)),0)+np.where(v<1,(1-v)*np.log(np.maximum(1-v,1e-300)),0))
    return entropy(q)-entropy(np.full(len(p),reliability))

def evidence_study(root,design,pulse):
    branch=design['branch'];model,reliability,train_x,train_y=cue_model(design,pulse);write(root/'FIT_COST.json',model.metadata());summaries=[]
    (root/'training').mkdir(parents=True,exist_ok=True);np.savez_compressed(root/'training/examples_points.npz',evidence=train_x,labels=train_y)
    for lineage in design['lineages']:
        pulse(phase='evidence-evaluation',lineage=lineage);rows=W.enumerate_world(lineage);sample=W.sampled(rows,design['test_cases'],'evidence',lineage)
        yy=sample[:,10].astype(int);x=W.project(sample,'context');ref=reference(rows,'context');oracle,support=posterior(ref,x,[])
        rng=np.random.default_rng(W.seed('cues',lineage));base=model.predict(x,'structured-joint',support)
        (root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(parents=True,exist_ok=True)
        np.savez_compressed(root/'reader'/f'{lineage}_points.npz',evidence=x)
        np.savez_compressed(root/'evaluator'/f'{lineage}_points.npz',hidden_rows=sample)
        if branch=='G4':
            actual=design.get('reliability',.8);copied=design.get('copied',False);irrelevant=design.get('irrelevant',False)
            if irrelevant:actual=.5
            views={kind:cue_inputs(x,yy,kind,actual,rng) for kind in ('none','event','context','repeat')}
            if copied:
                previous=views['context'][:,11].copy()
                for view in views.values():view[:,11]=previous
                views['repeat'][:,12]=previous
                base=model.predict(views['none'],'structured-joint',support)
                likelihood=np.where(W.LABELS[None,:,5]==previous[:,None],actual,1-actual)
                oracle=normalize(oracle*likelihood)
            np.savez_compressed(root/'reader'/f'{lineage}-acquired-views_points.npz',**views)
            forecasts={}
            for kind,view in views.items():
                sup=support.copy()
                if kind=='event':sup&=W.LABELS[None,:,0]==view[:,None,6]
                forecasts[kind]=model.predict(view,'structured-joint',sup)
            kinds=list(views);fee=design.get('fee',.02);cost=np.array([0,fee,fee*2,fee*3])
            gains=np.stack([np.zeros(len(x)),information(base,0,1),information(base,5,reliability),information(base,5,reliability)],axis=1)
            # A copied report never earns independent information a second time.
            if copied:gains[:,2:4]=0
            learned=np.argmax(gains-cost,axis=1)
            og=np.stack([np.zeros(len(x)),information(oracle,0,1),information(oracle,5,actual),information(oracle,5,actual)],axis=1)
            if copied:og[:,2:4]=0
            choices={'learned-selector':learned,'fixed-context':np.full(len(x),2),'random':rng.integers(4,size=len(x)),
                'cheapest-record':np.ones(len(x),int),'no-acquisition':np.zeros(len(x),int),'oracle-selector':np.argmax(og-cost,axis=1)}
            for name,index in choices.items():
                p=np.stack([forecasts[kinds[j]][i] for i,j in enumerate(index)])
                metric=write_forecasts(root,f'{lineage}-{name}',p,yy,dict(branch=branch,lineage=lineage,method=name))
                metric.update(acquisition_cost=float(cost[index].mean()),net_log_loss=float(metric['log_loss']+cost[index].mean()))
                summaries.append(dict(lineage=lineage,method=name,metrics=metric))
            write(root/'policies'/f'{lineage}.json',dict(learned=learned.tolist(),choices={name:index.tolist() for name,index in choices.items()},costs=cost.tolist(),train_estimated_reliability=reliability,
                oracle_scope='Oracle expected information selector; same learned readout after acquisition; not a perfect-reader ceiling',source_groups=['event','context','context' if copied else 'repeat']))
        elif branch=='G5':
            allowed=W.LABELS[:,0]==0;missing=~allowed[yy];budget=design.get('candidate_budget',16)
            masks={'forced-choice':np.tile(allowed,(len(x),1)),'unknown':np.tile(allowed,(len(x),1))}
            expanded=masks['unknown'].copy()
            for i in range(len(x)):
                extra=np.flatnonzero(~allowed);rank=extra[np.argsort(-base[i,extra],kind='stable')[:budget]];expanded[i,rank]=True
            masks['generated-expansion']=expanded
            supplied=masks['unknown'].copy();supplied[np.arange(len(x)),yy]=True;masks['supplied-hypothesis-control']=supplied
            for name,mask in masks.items():
                mass=base*mask;unknown=(base*(~mask)).sum(1)
                if name=='forced-choice':
                    p=normalize(mass);mass=p.copy();unknown=np.zeros(len(x))
                else:
                    # Scoring-only resolution of unknown: uniform over the declared
                    # finite omitted universe. Raw export preserves unknown separately.
                    p=mass+unknown[:,None]*(~mask)/np.maximum(1,(~mask).sum(1))[:,None]
                metric=write_forecasts(root,f'{lineage}-{name}',p,yy,dict(branch=branch,lineage=lineage,method=name))
                event_probability=np.where(mask[np.arange(len(x)),yy],mass[np.arange(len(x)),yy],unknown)
                metric.update(candidate_event_log_loss=float(-np.log(np.maximum(event_probability,1e-12)).mean()),
                    candidate_event_brier=float(((mass*mass).sum(1)+unknown*unknown-2*event_probability+1).mean()),represented_truth_fraction=float(mask[np.arange(len(x)),yy].mean()),candidate_count=float(mask.sum(1).mean()),
                    initial_omission_fraction=float(missing.mean()),unknown_mass=float(unknown.mean()))
                summaries.append(dict(lineage=lineage,method=name,metrics=metric))
                np.savez_compressed(root/'raw'/f'{lineage}-{name}-transport_points.npz',candidate_mass=mass,unknown_mass=unknown,mask=mask)
        elif branch=='G6':
            for update in ('false-to-true','true-to-true','irrelevant','unchanged','retract'):
                initial=cue_inputs(x,yy,'context',0 if update in ('false-to-true','retract') else (1 if update=='true-to-true' else .8),rng)
                final=initial.copy()
                if update=='false-to-true':final[:,11]=W.LABELS[yy,5]
                elif update=='irrelevant':final[:,12]=rng.integers(2,size=len(x))
                elif update=='retract':final[:,11]=-1
                observation_id=digest(['revision-view',lineage,update])[:24]
                np.savez_compressed(root/'reader'/f'{observation_id}_points.npz',initial=initial,final=final)
                write(root/'evaluator'/f'{lineage}-{update}-observation.json',dict(reader_file=observation_id+'_points.npz',update=update))
                old=model.predict(initial,'structured-joint',support);new=model.predict(final,'structured-joint',support)
                for name,p in [('saved-conclusion',old),('ledger-recompute',new),('bounded-raw',new)]:
                    metric=write_forecasts(root,f'{lineage}-{update}-{name}',p,yy,dict(branch=branch,lineage=lineage,method=name,update=update))
                    summaries.append(dict(lineage=lineage,method=name,update=update,metrics=metric))
        else:raise ValueError('evidence branch not implemented')
    return dict(branch=branch,population=summaries,controls=controls(),scope='Constructed probability and decision method; unknown resolution is finite-universe only; no human inference')

def retention(root,design,pulse):
    """Information ceiling for persistent-maker bank versus witnessed ledger.

    A selected proposal with no revision produces identical artifacts in both
    operation orders. Equal order likelihood for every maker is symbolic, so any
    fresh-episode bank factoring through that posterior is identical. The ledger
    retains a bit of historical truth. Learned access is tested separately in core.
    """
    records=[]
    for lineage in design['lineages']:
        rows=W.enumerate_world(lineage);w=W.law(lineage);C.verify_world(rows,w,W.MAKERS,W.CONTEXTS)
        # A genuine prediction bank: future endpoint distributions for each
        # of the eight possible task contexts, under each persistent maker.
        kernel=np.zeros((32,8,8))
        endpoint=(rows[:,11]*4+rows[:,12]*2+rows[:,13]).astype(int)
        np.add.at(kernel,(rows[:,0].astype(int),rows[:,1].astype(int),endpoint),rows[:,-1])
        kernel=kernel/kernel.sum(2,keepdims=True)
        for ci in range(8):
            for route in range(W.CONTEXTS[ci][2]+1):
                mask=(rows[:,1]==ci)&(rows[:,2]==route)&(rows[:,4]==1)&(rows[:,6]==0)
                pair=[rows[mask&(rows[:,7]==order)] for order in (0,1)]
                banks=[np.bincount(a[:,0].astype(int),weights=a[:,-1],minlength=32) for a in pair]
                if not np.array_equal(banks[0],banks[1]):raise ValueError('order bank witness lost exact computed identity')
                posterior=normalize(banks[0]);q=w['inspect']/10
                # Fresh artifact bank marginalizes review style; maker state retains it.
                prediction=(posterior@kernel.reshape(32,64)).reshape(8,8)
                size={'raw-history':len(canonical(W.events(pair[0][0],True))),
                    'maker-state':len(canonical(posterior.tolist())),'prediction-bank':len(canonical(prediction.tolist())),
                    'state-plus-ledger':len(canonical(dict(state=posterior.tolist(),order=0,inspection=0)))}
                for method,loss in [('raw-history',0),('maker-state',float(np.log(2))),('prediction-bank',float(np.log(2))),('state-plus-ledger',0)]:
                    records.append(dict(lineage=lineage,context=ci,route=route,method=method,order_log_loss=loss,
                        new_order_log_loss=float(np.log(2)),bytes=size[method],
                        recorded_inspection_log_loss=float(np.log(2)) if method=='prediction-bank' or (method=='maker-state' and q==.5) else 0.,
                        review_scope='observed inspection diagnostic; hidden inspection remains unknown',
                        inspection_warrant='Equal review-style mixture leaves artifact bank unchanged; full maker posterior distinguishes inspection unless q=1/2; supplied-law decoding only'))
        pulse(phase='retention',lineage=lineage)
    write(root/'RETENTION_ROWS.json',records);write(root/'WITNESSES.json',C.exact_witnesses())
    return dict(branch='G3',rows=records,controls=controls(),scope='Exact information counterexample; symbolic order symmetry plus all finite lineage transitions checked. Bytes measure canonical JSON payloads, excluding Python heap; no learned-memory superiority claimed.')

def run_study(root,design,pulse):
    branch=design['branch']
    if branch in ('G1','G2','G7'):return core(root,design,pulse)
    if branch in ('G4','G5','G6'):return evidence_study(root,design,pulse)
    if branch=='G3':return retention(root,design,pulse)
    if branch=='G0':
        inventories=[]
        for lineage in design['lineages']:
            rows=W.enumerate_world(lineage);inv=C.verify_world(rows,W.law(lineage),W.MAKERS,W.CONTEXTS)
            inv.update(lineage=lineage,makers=32,contexts=8,endpoints=len(np.unique(rows[:,11:14],axis=0)))
            inventories.append(inv);pulse(phase='source-ruler',lineage=lineage)
        write(root/'WITNESSES.json',C.exact_witnesses())
        return dict(branch='G0',inventories=inventories,controls=controls(),scope='Finite constructed world; source-adapter consumer separately required')
    raise ValueError('unknown branch')
