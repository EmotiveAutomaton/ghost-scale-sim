"""A04 separable inspection-record channel on the original finite world.

The observation mechanism and false-record rate are supplied. This is an exact
information/measurement ruler, not a learned inspection detector.
"""
import numpy as np
from . import world as W
from .studies import retain
from ..v18_3.io import write

MISSING=(0.,.25,.5,.75)
FALSE=(0.,.05,.2)


def deletion(inspection,rate,mechanism):
    if mechanism=='independent':return np.full(np.shape(inspection),rate)
    if mechanism=='inspection-dependent':return np.where(inspection==1,min(1.,1.5*rate),.5*rate)
    raise ValueError('unregistered observation mechanism')


def joint(rows):
    out=np.zeros((8,128,2))
    artifact=rows[:,11:14].astype(int)@np.array([4,2,1])
    np.add.at(out,(artifact,rows[:,10].astype(int),rows[:,5].astype(int)),rows[:,-1])
    return out


def table(mass,rate,false,mechanism,reader):
    """Lookup is keyed by artifact and three possible records (-1, 0, 1)."""
    result=np.zeros((8,3,128));fallback=np.zeros((8,3),bool)
    miss=deletion(np.arange(2),rate,mechanism)
    for artifact in range(8):
        for record in (-1,0,1):
            if reader=='endpoint':likelihood=np.ones(2)
            elif reader in ('naive-negative','full-record-oracle'):likelihood=(np.arange(2)==max(record,0)).astype(float)
            elif record<0:likelihood=miss if reader in ('observation-aware','matched-direct') else np.ones(2)
            else:
                likelihood=np.where(np.arange(2)==record,1-false,false)
                if reader in ('observation-aware','matched-direct'):likelihood=likelihood*(1-miss)
            value=mass[artifact]@likelihood
            if value.sum()==0:
                # Impossible observations use the explicit endpoint-prior fallback.
                # Their model probability is zero; generated aware cases cannot hit it.
                value=mass[artifact].sum(1);fallback[artifact,record+1]=True
            result[artifact,record+1]=value/value.sum()
    return result,fallback


def controls():
    mass=np.zeros((8,128,2));mass[:,0,0]=.5/8;mass[:,4,1]=.5/8
    a,_=table(mass,.5,0.,'inspection-dependent','observation-aware')
    b,_=table(mass,.5,0.,'independent','observation-aware')
    n,_=table(mass,.5,0.,'independent','naive-negative')
    noisy,_=table(mass,0.,.2,'independent','observation-aware')
    return dict(live_selective_absence=bool(np.allclose(a[:,0,4],.75)),
        placebo_random_absence=bool(np.allclose(b[:,0,4],.5)),
        positive_exact_witness=bool(np.allclose(b[:,2,4],1)),
        false_record_keeps_truth=bool(np.allclose(noisy[:,1,4],.2)),
        naive_absence_excludes_inspection=bool(np.allclose(n[:,0,4],0)))


def run(root,d,pulse):
    from .inspection_check import reference
    # Exact zero-history law mixture: each independent coefficient occurs once.
    law=dict(route=4,select=4,revise=4,goal=4,inspect=4,skill=6.5,shift='native',presentation=0,lineage=-1)
    prior=W.enumerate_law(law);mass=joint(prior);population=[];maximum=0.;cases=[]
    readers=('endpoint','naive-negative','missingness-ignorant','observation-aware','matched-direct','full-record-oracle')
    for mechanism in d.get('mechanisms',('independent','inspection-dependent')):
      for rate in d.get('rates',MISSING):
       for false in d.get('false_rates',FALSE):
        tables={};fallbacks={}
        for reader in readers:
            tables[reader],fallbacks[reader]=table(mass,rate,false,mechanism,reader)
            ref=reference(prior,rate,false,mechanism,reader)
            error=float(abs(tables[reader]-ref).max());maximum=max(maximum,error);assert error<1e-12
        for world in d['lineages']:
            rows=W.enumerate_world(world);chosen=W.sampled(rows,d['test_cases'],'v21-a04-inspection',world)
            truth=chosen[:,10].astype(int);inspection=chosen[:,5].astype(int);artifact=chosen[:,11:14].astype(int);keys=artifact@np.array([4,2,1])
            rng=np.random.default_rng(W.seed('v21-a04-channel',world));u=rng.random(len(chosen));v=rng.random(len(chosen))
            deleted=u<deletion(inspection,rate,mechanism);flipped=v<false;record=np.where(deleted,-1,np.where(flipped,1-inspection,inspection))
            token=f'{world}-{mechanism}-{rate}-{false}'
            p=root/'reader'/f'{token}_points.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,artifact=artifact,inspection_record=record)
            p=root/'evaluator'/f'{token}_points.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,truth=truth,inspection=inspection,deleted=deleted,flipped=flipped)
            for reader in readers:
                use=inspection if reader=='full-record-oracle' else record
                probabilities=tables[reader][keys,use+1]
                with np.errstate(divide='ignore'):lp=np.log(probabilities)
                metrics=retain(root,token+'-'+reader,lp,truth)
                population.append(dict(lineage=world,mechanism=mechanism,missing_rate=rate,false_rate=false,reader=reader,metrics=metrics,fallback_fraction=float(fallbacks[reader][keys,use+1].mean())))
                if reader=='observation-aware':assert not fallbacks[reader][keys,use+1].any()
            if world==d['lineages'][0]:cases.append(dict(mechanism=mechanism,missing_rate=rate,false_rate=false,case=0,artifact=artifact[0].tolist(),record=int(record[0]),inspection=int(inspection[0]),truth=int(truth[0]),role='evaluator only'))
            pulse(phase='inspection-record-channel',lineage=world,mechanism=mechanism,missing_rate=rate,false_rate=false)
    write(root/'CASEBOOK_EVALUATOR.json',cases)
    return dict(card='G21-A04',population=population,controls=controls(),independent_max_probability_error=maximum,
        scope='Original native zero-history exact law-marginal inspection channel; no learned superiority or confirmation.',
        limitations=['Observation mechanism, deletion rate and false-record rate are supplied, not inferred from metadata.',
        'Inspection changes one historical target bit but does not causally change the artifact in this world.',
        'Matched direct lookup has identical information and exactly matches aware inference; no fitting or capacity claim.',
        'Naive negative treats every present record as exact and absence as zero; false records or missing inspections can produce infinite uncapped loss.',
        'Full-record oracle receives the true inspection bit and is a privileged information bound.',
        'All channel conditions share cases and random uniforms; eight worlds are the uncertainty units. Native only; no execution shift or product-of-factors ablation.',
        'Selective absence differs from random absence; this tests observation-process assumptions, not human thought or real log completeness.'])
