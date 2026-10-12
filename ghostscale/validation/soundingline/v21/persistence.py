"""B03 supplied-law information ruler under matched exogenous contexts.

This diagnostic knows the generator law and maker-state vocabulary. It is not a
learned-reader superiority screen. Persistence is inferred without target labels.
"""
from pathlib import Path
import numpy as np
from ..v18_3.io import write
from . import world as W, oracle as O
from .studies import retain

RATES=(1.,.9,.5,0.)


def tables(rows,tier='context'):
    current=W.project(rows,tier);history=W.project(rows,'complete',True)
    ck,ci=np.unique(current,axis=0,return_inverse=True)
    hk,hi=np.unique(history,axis=0,return_inverse=True)
    joint=np.zeros((len(ck),32,128));emission=np.zeros((32,len(hk)))
    maker=rows[:,0].astype(int);target=rows[:,10].astype(int)
    np.add.at(joint,(ci,maker,target),rows[:,-1]);np.add.at(emission,(maker,hi),rows[:,-1])
    contexts=hk[:,3:6]@np.array([4,2,1]);den=np.zeros((32,8))
    for context in range(8):den[:,context]=emission[:,contexts==context].sum(1)
    emission/=den[:,contexts]
    return dict(current={r.tobytes():i for i,r in enumerate(ck)},history={r.tobytes():i for i,r in enumerate(hk)},joint=joint,emission=emission,background=emission.mean(0))


def observations(rows,chosen,history,identity):
    """Use the same current cases, contexts and uniform draws at every rate."""
    rng=np.random.default_rng(W.seed('v21-b03-context',identity))
    contexts=rng.integers(8,size=(len(chosen),history));uniform=rng.random((len(chosen),history))
    foreign=rng.integers(32,size=(len(chosen),history));bundle=rng.integers(32,size=len(chosen))
    draw=rng.random((len(chosen),history));groups={}
    for maker in range(32):
        for context in range(8):
            ids=np.flatnonzero((rows[:,0]==maker)&(rows[:,1]==context))
            weights=rows[ids,-1];groups[maker,context]=(ids,np.cumsum(weights/weights.sum()))
    result={};truth=chosen[:,10].astype(int);current=W.project(chosen,'context')
    for condition in (*RATES,'shuffled-author'):
        owners=np.where(uniform<float(condition),chosen[:,0,None].astype(int),foreign) if condition!='shuffled-author' else np.broadcast_to(bundle[:,None],contexts.shape)
        episodes=[]
        for step in range(history):
            indices=[]
            for i in range(len(chosen)):
                ids,cdf=groups[int(owners[i,step]),int(contexts[i,step])]
                indices.append(ids[min(np.searchsorted(cdf,draw[i,step],side='right'),len(ids)-1)])
            episodes.append(W.project(rows[indices],'complete',True))
        result[str(condition)]=dict(evidence=np.concatenate([current,*episodes],axis=1),truth=truth,contexts=contexts,owners=owners,current_maker=chosen[:,0].astype(int),same_source=(uniform<float(condition)) if condition!='shuffled-author' else np.zeros(contexts.shape,bool))
    return result


def predict(table,x):
    ci=np.array([table['current'][r[:13].tobytes()] for r in x]);joint=table['joint'][ci]
    h=x[:,13:].reshape(len(x),-1,13);hi=np.array([[table['history'][r.tobytes()] for r in hist] for hist in h])
    log_weights=[]
    with np.errstate(divide='ignore'):
        log_joint=np.log(joint)
        for rate in RATES:
            evidence=np.zeros((len(x),32))
            for step in range(hi.shape[1]):
                evidence+=np.log(rate*table['emission'][:,hi[:,step]].T+(1-rate)*table['background'][hi[:,step],None])
            log_weights.append(O.logsumexp(log_joint+evidence[:,:,None],1))
    weights=np.array(log_weights)
    log_z=O.logsumexp(weights,2).T
    inferred=O.normalize_log(O.logsumexp(weights,0))
    with np.errstate(divide='ignore'):base=O.normalize_log(np.log(joint.sum(1)))
    predictions={};conflicts={}
    for i,rate in enumerate(RATES):
        conflict=~np.isfinite(log_z[:,i]);mass=weights[i].copy();mass[conflict]=base[conflict]
        predictions[str(rate)]=O.normalize_log(mass);conflicts[str(rate)]=conflict
    return dict(by_rate=predictions,conflicts=conflicts,inferred=inferred,
                persistence_probability=np.exp(O.normalize_log(log_z)),no_history=base)


def controls():
    joint=np.ones((1,32,128));joint[0,:,0]=np.arange(1,33)
    joint/=joint.sum();emission=np.tile(np.array([.2,.8]),(32,1))
    current=np.zeros(13,dtype=np.int8);a=np.ones(13,dtype=np.int8);b=np.full(13,2,dtype=np.int8)
    table=dict(current={current.tobytes():0},history={a.tobytes():0,b.tobytes():1},joint=joint,emission=emission,background=emission.mean(0))
    x=np.concatenate([current,a,b])[None,:];p=predict(table,x)
    return dict(placebo_identical_makers=all(np.allclose(v,p['no_history'],atol=1e-12) for v in p['by_rate'].values()),
                live_unrelated_history=np.allclose(p['by_rate']['0.0'],p['no_history'],atol=1e-12),
                positive_normalized_inferred=bool(np.allclose(np.exp(p['inferred']).sum(1),1)))


def run(root,d,pulse):
    from .persistence_check import reference,check_tables
    population=[];maximum=0.;rate_rows=[]
    for world in d['lineages']:
        rows=W.enumerate_world(world,'native',d.get('regime','original'))
        chosen=W.sampled(rows,d['test_cases'],'v21-b03-current',world)
        table=tables(rows);check_tables(rows,table)
        observations_by_rate=observations(rows,chosen,d['history'],world)
        for condition,obs in observations_by_rate.items():
            x=obs['evidence'];truth=obs['truth'];prediction=predict(table,x);ref=reference(table,x)
            for name in ('inferred','no_history'):
                error=float(np.max(np.abs(np.exp(prediction[name])-ref[name])));maximum=max(maximum,error);assert error<1e-10
            for rate in RATES:
                error=float(np.max(np.abs(np.exp(prediction['by_rate'][str(rate)])-ref['by_rate'][str(rate)])));maximum=max(maximum,error);assert error<1e-10
            assert np.allclose(prediction['by_rate']['0.0'],prediction['no_history'],atol=1e-10,equal_nan=True)
            informed_rate='0.0' if condition=='shuffled-author' else condition
            arms={'informed-persistence':prediction['by_rate'][informed_rate],'inferred-persistence':prediction['inferred'],
                  'rigid-same-maker':prediction['by_rate']['1.0'],'no-history':prediction['no_history']}
            for name,lp in arms.items():
                conflict=prediction['conflicts']['1.0'] if name=='rigid-same-maker' else prediction['conflicts'][informed_rate] if name=='informed-persistence' else np.zeros(len(truth),bool)
                population.append(dict(lineage=world,persistence=condition,reader=name,contradiction_fraction=float(conflict.mean()),metrics=retain(root,f'{world}-{condition}-{name}',lp,truth,contradiction=conflict)))
            path=root/'reader'/f'{world}-{condition}_points.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,evidence=x)
            path=root/'evaluator'/f'{world}-{condition}_points.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,truth=truth,contexts=obs['contexts'],history_makers=obs['owners'],current_makers=obs['current_maker'],same_source=obs['same_source'],persistence_probability=prediction['persistence_probability'])
            rate_rows.append(dict(lineage=world,persistence=condition,mean_posterior_persistence=float((prediction['persistence_probability']@np.array(RATES)).mean()),
                mean_persistence_probabilities=prediction['persistence_probability'].mean(0).tolist(),realized_same_source_fraction=float(obs['same_source'].mean())))
            pulse(phase='persistence-diagnostic',lineage=world,persistence=condition)
    return dict(card='G21-B03',population=population,persistence_inference=rate_rows,controls=controls(),independent_max_probability_error=maximum,
        scope='Supplied-law information ruler; no learned-reader superiority or new law-marginal ceiling.',
        limitations=['Generator law, maker-state vocabulary and conditional likelihoods are supplied; capacity, learning and law uncertainty are not tested.',
                    'Contexts are randomized independently of the current maker and held identical across persistence conditions.',
                    'Inferred persistence has a uniform prior on 1, 0.9, 0.5 and 0, shared within each four-episode case, without target labels.',
                    'If a rigid model assigns zero likelihood to all current/history combinations, record the contradiction and fall back to the current-only posterior. This is a declared decision fallback, not a valid conditional posterior under the contradicted model.',
                    'Shuffled-author uses one unrelated maker for the complete history bundle. The zero-persistence arm is deliberately misspecified about within-bundle dependence there.',
                    'No trainable fits, direct convergence, copied-cell headroom, shifted execution or confirmation are claimed.',
                    'Learned contamination detection and original/wide asymmetric learned-reader comparisons remain conditional B03 scope.'])
