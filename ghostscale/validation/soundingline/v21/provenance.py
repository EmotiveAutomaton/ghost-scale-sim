"""D03 finite copying ruler with uncertain source labels, not learned provenance."""
import itertools,math,time
import numpy as np
from ..v18_3.io import write
from . import world as W
from .acquisition import tables
from .studies import retain

READERS=('supplied-roots','inferred-groups','naive-independent','conservative-envelope')
RELIABILITY=.8


def source_channel(ids,roots,mode,rate):
    ids=np.asarray(ids);labels=np.arange(roots)
    if mode=='hidden':return np.where(ids[:,None]<0,rate,(1-rate)*(ids[:,None]==labels))
    return (1-rate)*(ids[:,None]==labels)+rate/roots


def group_likelihood(reports,ids,roots,mode,rate):
    """Sum all onto labelled copying graphs by subset inclusion-exclusion."""
    y=np.asarray(reports);channel=source_channel(ids,roots,mode,rate)
    bits=np.array(list(itertools.product((0,1),repeat=roots)))
    subsets=np.array(list(itertools.product((0,1),repeat=roots)))
    signs=(-1.)**(roots-subsets.sum(1))
    weights=np.prod(np.einsum('bnr,sr->bsn',(bits[:,None,:]==y[None,:,None])*channel[None,:,:],subsets),axis=2)@signs
    assert np.min(weights)>-1e-10
    weights=np.maximum(weights,0)
    result=np.array([np.dot(weights,np.prod(np.where(bits==g,RELIABILITY,1-RELIABILITY),axis=1)) for g in (0,1)])
    assert result.sum()>0
    return result/result.sum()


def posterior(joint,likelihood):
    p=(joint*likelihood[:,None]).sum(0);return p/p.sum()


def conservative(joint,reports,roots):
    """Project current goal probability onto the feasible copying envelope.

    This is a conservative decision rule, not an inferred graph posterior.
    The root count is supplied; all label-independent feasible partitions enter.
    """
    counts=np.bincount(reports,minlength=2);values=[]
    prior=joint.sum(1);prior/=prior.sum()
    for ones in range(roots+1):
        zero=roots-ones
        if (zero==0)!=(counts[0]==0) or (ones==0)!=(counts[1]==0):continue
        if zero>counts[0] or ones>counts[1]:continue
        likelihood=np.array([RELIABILITY**zero*(1-RELIABILITY)**ones,(1-RELIABILITY)**zero*RELIABILITY**ones])
        p=prior*likelihood;p/=p.sum();values.append(p[1])
    assert values
    lo,hi=min(values),max(values);goal=np.clip(prior[1],lo,hi)
    conditional=np.divide(joint,prior[:,None],out=np.zeros_like(joint),where=prior[:,None]>0)
    return conditional[0]*(1-goal)+conditional[1]*goal,(lo,hi)


def draw_graph(rng,n,roots,kind):
    if kind=='balanced':
        a=np.arange(n)%roots;rng.shuffle(a);return a
    while True:
        a=rng.integers(roots,size=n)
        if len(set(a))==roots:return a


def controls():
    identical=group_likelihood([1,1],[0,0],1,'hidden',0)
    independent=group_likelihood([1,1],[0,1],2,'hidden',0)
    return dict(live_known_copy=bool(np.allclose(identical,[.2,.8])),positive_independent_roots=bool(independent[1]>identical[1]),placebo_label_permutation=bool(np.allclose(independent,group_likelihood([1,1],[1,0],2,'hidden',0))))


def run(root,d,pulse):
    from .provenance_check import reference
    population=[];timings=[];cache={};maxerr=0.;maxperm=0.
    for world in d['lineages']:
        start=time.process_time();rows=W.enumerate_world(world,'native',d['regime']);table=tables(rows)
        sampled=W.sampled(rows,d['test_cases'],'v21-d03',world,d['regime']);evidence=W.project(sampled,'context');truth=sampled[:,10].astype(int)
        goals=np.array([W.MAKERS[int(m)][2] for m in sampled[:,0]])
        joints=[]
        for row in evidence:
            j=table['current'][table['index'][row.tobytes()]];j=j/j.sum()
            joints.append(np.array([j[[m for m in range(32) if W.MAKERS[m][2]==g]].sum(0) for g in (0,1)]))
        joints=np.array(joints);setup=time.process_time()-start
        for n in d['report_counts']:
          for roots in d['root_counts']:
            if roots>n:continue
            for graph in d['graphs']:
                rng=np.random.default_rng(W.seed('v21-d03-reports',world,n,roots,graph,d['regime']))
                assignments=np.array([draw_graph(rng,n,roots,graph) for _ in truth])
                root_reports=np.where(rng.random((len(truth),roots))<RELIABILITY,goals[:,None],1-goals[:,None])
                reports=np.take_along_axis(root_reports,assignments,axis=1)
                corrupt_draw=rng.random((len(truth),n));replacement=rng.integers(roots,size=(len(truth),n))
                for mode in d['id_modes']:
                  for rate in d['id_rates']:
                    ids=np.where(corrupt_draw<rate,-1 if mode=='hidden' else replacement,assignments)
                    predictions={reader:[] for reader in READERS};bounds=[];tick=time.process_time()
                    for i,report in enumerate(reports):
                        key=(tuple(report),tuple(ids[i]),roots,mode,rate)
                        if key not in cache:
                            value=group_likelihood(report,ids[i],roots,mode,rate);ref=reference(report,ids[i],roots,mode,rate)
                            err=float(np.max(np.abs(value-ref)));maxerr=max(maxerr,err);assert err<1e-10
                            renamed=np.where(ids[i]<0,-1,roots-1-ids[i]);perm=group_likelihood(report,renamed,roots,mode,rate)
                            err=float(np.max(np.abs(value-perm)));maxperm=max(maxperm,err);assert err<1e-10
                            cache[key]=value
                        known=np.array([np.prod(np.where(root_reports[i]==g,RELIABILITY,1-RELIABILITY)) for g in (0,1)])
                        naive=np.array([np.prod(np.where(report==g,RELIABILITY,1-RELIABILITY)) for g in (0,1)])
                        predictions['supplied-roots'].append(posterior(joints[i],known));predictions['inferred-groups'].append(posterior(joints[i],cache[key]));predictions['naive-independent'].append(posterior(joints[i],naive))
                        conservative_p,bound=conservative(joints[i],report,roots);predictions['conservative-envelope'].append(conservative_p);bounds.append(bound)
                    decode=time.process_time()-tick;prefix=f'{world}-{n}-{roots}-{graph}-{mode}-{rate}'
                    (root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
                    np.savez_compressed(root/'reader'/f'{prefix}_points.npz',evidence=evidence,reports=reports,source_ids=ids)
                    np.savez_compressed(root/'evaluator'/f'{prefix}_points.npz',truth=truth,source_rows=sampled,assignments=assignments,root_reports=root_reports,joint=joints,goal_posterior_bounds=np.array(bounds))
                    for reader,pred in predictions.items():
                        with np.errstate(divide='ignore'):lp=np.log(np.array(pred))
                        metrics=retain(root,prefix+'-'+reader,lp,truth)
                        population.append(dict(lineage=world,reports=n,roots=roots,graph=graph,id_mode=mode,id_rate=rate,reader=reader,metrics=metrics))
                    timings.append(dict(lineage=world,reports=n,roots=roots,graph=graph,id_mode=mode,id_rate=rate,decoder_and_reference_cpu_seconds=decode))
                    pulse(phase='uncertain-source-groups',lineage=world,reports=n,roots=roots,graph=graph,id_mode=mode,id_rate=rate)
        timings.append(dict(lineage=world,setup_cpu_seconds=setup,total_cpu_seconds=time.process_time()-start))
    return dict(card='G21-D03',population=population,controls=controls(),independent_max_probability_error=maxerr,source_permutation_max_error=maxperm,timings=timings,scope='Supplied-law uncertain-source information ruler; no learned provenance or architecture superiority.',limitations=['Generator law, binary cue reliability .8, number of roots and source-ID noise rate are supplied.','Inferred groups integrate uniform onto labelled copying assignments; balanced graphs are a declared prior-mismatch challenge.','Noisy IDs are replaced uniformly, possibly by their original label; at one root replacement has no effect.','Hidden IDs are erased independently. Copy outcomes are exact copies of one root reply.','The conservative rule projects the current goal probability onto the feasible copying envelope; it is not a graph posterior or a universal minimax rule.','No trainable fit, direct convergence, law-marginal headroom, shifted execution or confirmation is tested.','Opening original-law grid only; broader regimes and learned source models remain conditional.','No extra F03 fixture category or live inference claimed.'])
