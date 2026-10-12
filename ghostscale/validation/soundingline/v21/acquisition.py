"""One-purchase D01 supplied-law ruler; evaluator makers never enter the reader."""
import time
import numpy as np
from ..v18_3.io import write
from . import world as W
from .studies import retain

QUERIES=('operation-event','contrasting-context','noisy-maker-cue','random-action-distractor')
POLICIES=('target-information','action-entropy','fixed-cheapest','random','no-acquisition')
COSTS=np.array([2.,4.,1.,1.])


def entropy(p):
    positive=p>0
    return float(-np.sum(p[positive]*np.log(p[positive])))


def describe(joint):
    """Joint target/report mass; return information and report entropy."""
    assert np.isclose(joint.sum(),1) and np.all(joint>=0)
    py=joint.sum(1);pr=joint.sum(0)
    return entropy(py)+entropy(pr)-entropy(joint),entropy(pr)


def tables(rows):
    current=W.project(rows,'context');keys,inverse=np.unique(current,axis=0,return_inverse=True)
    j=np.zeros((len(keys),32,128));np.add.at(j,(inverse,rows[:,0].astype(int),rows[:,10].astype(int)),rows[:,-1])
    histories={}
    for context in range(8):
        rr=rows[rows[:,1]==context];hk,hi=np.unique(W.project(rr,'complete',True),axis=0,return_inverse=True)
        e=np.zeros((32,len(hk)));np.add.at(e,(rr[:,0].astype(int),hi),rr[:,-1]);e/=e.sum(1)[:,None]
        histories[context]=(hk,e)
    return dict(keys=keys,current=j,index={k.tobytes():i for i,k in enumerate(keys)},histories=histories)


def channels(table,index):
    key=table['keys'][index];j=table['current'][index];j=j/j.sum();base=j.sum(0)
    op=base[:,None]*(W.LABELS[:,3,None]==np.arange(2))
    context=int(key[3:6]@np.array([4,2,1]))^2
    hist=j.T@table['histories'][context][1]
    pref=np.array(W.MAKERS)[:,2]
    cue=j.T@np.where(pref[:,None]==np.arange(2),.8,.2)
    distractor=np.repeat(base[:,None]/8,8,axis=1)
    return base,[op,hist,cue,distractor],context


def choose(info,uncertainty,fee,random_query):
    def maximum(values):
        score=values-fee*COSTS;choice=int(np.argmax(score))
        return choice if score[choice]>1e-12 else -1
    return [maximum(info),maximum(uncertainty),2,int(random_query),-1]


def controls():
    base=np.array([.25,.75]);uninformative=base[:,None]*np.array([.5,.5])
    informative=np.diag(base)
    return dict(placebo_random_action=abs(describe(uninformative)[0])<1e-12,
        positive_revealing_record=abs(describe(informative)[0]-entropy(base))<1e-12,
        live_free_information=choose(np.array([.4,0,0,0]),np.array([.4,0,0,2]),0,2)[0]==0,
        costly_record_abstention=choose(np.array([.01,0,0,0]),np.ones(4),.2,2)[0]==-1)


def run(root,d,pulse):
    from .acquisition_check import check_tables,reference
    population=[];timings=[];maximum=0.;cases=[]
    for world in d['lineages']:
        start=time.process_time();rows=W.enumerate_world(world,d['shift'],d['regime']);table=tables(rows)
        check_tables(rows,table)
        chosen=W.sampled(rows,d['test_cases'],'v21-d01',world,d['shift'],d['regime'])
        current=W.project(chosen,'context');truth=chosen[:,10].astype(int);maker=chosen[:,0].astype(int)
        rng=np.random.default_rng(W.seed('v21-d01-reports',world,d['shift'],d['regime']))
        draws=rng.random((len(chosen),3));random_queries=rng.integers(4,size=len(chosen))
        bases=[];posteriors=[];information=[];uncertainty=[];observations=[];history_keys=[];setup_end=time.process_time()
        cache={};policy_cpu=0.;decoder_cpu=0.
        for i,row in enumerate(current):
            idx=table['index'][row.tobytes()]
            if idx not in cache:
                base,masses,context=channels(table,idx)
                descriptors=[describe(m) for m in masses]
                refs=[reference(m) for m in masses]
                for mass,descriptor,ref in zip(masses,descriptors,refs):
                    assert np.allclose(descriptor,ref[:2],atol=1e-12)
                    valid=mass.sum(0)>0
                    maximum=max(maximum,float(np.max(np.abs(mass[:,valid]/mass[:,valid].sum(0)-ref[2][:,valid]))))
                cache[idx]=(base,masses,context,descriptors)
            base,masses,context,descriptors=cache[idx]
            hk,e=table['histories'][context]
            history=min(int(np.searchsorted(np.cumsum(e[maker[i]]),draws[i,0],side='right')),len(hk)-1)
            cue=int(W.MAKERS[maker[i]][2]) if draws[i,1]<.8 else 1-int(W.MAKERS[maker[i]][2])
            observed=[int(chosen[i,7]),history,cue,min(int(draws[i,2]*8),7)]
            tick=time.process_time();post=[]
            for mass,o in zip(masses,observed):
                p=mass[:,o];assert p.sum()>0;post.append(p/p.sum())
            decoder_cpu+=time.process_time()-tick
            bases.append(base);posteriors.append(post);information.append([x[0] for x in descriptors]);uncertainty.append([x[1] for x in descriptors]);observations.append(observed);history_keys.append(hk[history])
        bases=np.array(bases);posteriors=np.array(posteriors);information=np.array(information);uncertainty=np.array(uncertainty)
        (root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
        np.savez_compressed(root/'reader'/f'{world}_points.npz',evidence=current,operation_record=np.array(observations)[:,0],history=np.array(history_keys),cue=np.array(observations)[:,2],distractor=np.array(observations)[:,3])
        np.savez_compressed(root/'evaluator'/f'{world}_points.npz',truth=truth,source_rows=chosen,information=information,report_entropy=uncertainty,all_report_indices=np.array(observations),random_queries=random_queries,base=bases,posterior=posteriors)
        for fee in d['fees']:
            tick=time.process_time();choices=np.array([choose(information[i],uncertainty[i],fee,random_queries[i]) for i in range(len(current))]);policy_cpu+=time.process_time()-tick
            for column,policy in enumerate(POLICIES):
                q=choices[:,column];p=bases.copy();purchase=q>=0;p[purchase]=posteriors[np.arange(len(q))[purchase],q[purchase]]
                with np.errstate(divide='ignore'):lp=np.log(p)
                cost=np.zeros(len(q));cost[purchase]=COSTS[q[purchase]]
                metrics=retain(root,f'{world}-{fee}-{policy}',lp,truth,query=q,acquisition_cost=cost,net_loss=-lp[np.arange(len(q)),truth]+fee*cost)
                population.append(dict(lineage=world,fee=fee,policy=policy,metrics=metrics,mean_cost=float(cost.mean()),purchase_fraction=float(purchase.mean()),query_counts={str(k):int(np.sum(q==k)) for k in range(-1,4)},mean_net_loss=float((-lp[np.arange(len(q)),truth]+fee*cost).mean())))
        timings.append(dict(lineage=world,setup_cpu_seconds=setup_end-start,policy_cpu_seconds=policy_cpu,decoder_cpu_seconds=decoder_cpu,total_cpu_seconds=time.process_time()-start,scope='Total includes independent checks and export; setup and selection timings are descriptive, not converted into loss units.'))
        for i in range(min(3,len(chosen))):cases.append(dict(world=world,evidence=current[i].tolist(),truth=int(truth[i]),target_information=information[i].tolist(),report_entropy=uncertainty[i].tolist(),scope='Evaluator illustration only; no blind export.'))
        pulse(phase='acquisition-information-ruler',lineage=world)
    return dict(card='G21-D01',population=population,controls=controls(),independent_max_probability_error=maximum,timings=timings,illustrative_cases=cases,
        scope='Constructed supplied-law information ruler; one purchase, not learned acquisition or confirmation.',
        limitations=['Generator law, cue reliability and same-maker conditional episode likelihood are supplied.','No fit capacity, validation convergence or law uncertainty is tested.','Eight-way distractor is deliberately irrelevant; action-entropy selection is an explicit stress test.','CPU seconds are reported separately from fee-weighted net log loss; no empirical exchange rate is assumed.','Fees and policies share reports and worlds; they are not independent replications.','No multiplied-feature decoder, copied-evidence improvement, sequential policy or live inference claim.'])
