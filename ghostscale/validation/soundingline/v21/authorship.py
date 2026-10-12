"""E02 finite role-history ruler; exact records and supplied population priors.

The changing-maker category means role-conditioned aims. In aligned episodes it
can be observationally identical to a stable maker even with complete records.
"""
from itertools import product
import numpy as np
from . import world as W
from .studies import retain
from ..v18_3.io import write

READERS=('endpoint','single-person','role-mixture','matched-direct','full-record-oracle')


def catalog(roles,conflict,prior):
    states=[]
    for topology,endpoint in product(range(3),range(2)):
        # Last writer determines the final artifact. Every topology has both
        # identical possible endpoints, regardless of hidden intermediate aims.
        goals=[endpoint ^ (int(conflict) if topology and j%2!=(roles-1)%2 else 0) for j in range(roles)]
        owners=[int(topology==2 and j>0) for j in range(roles)]
        label=topology*8+sum(g<<(2-j) for j,g in enumerate(goals))
        states.append(dict(topology=topology,endpoint=endpoint,goals=goals,owners=owners,label=label,mass=float(prior[topology]/2)))
    return states


def view(s,tier):
    roles=len(s['goals']);visible=0 if tier=='none' else roles-1 if tier=='partial' else roles
    return (s['endpoint'],roles,*[s['goals'][j] if j<visible else -1 for j in range(3) if j<roles],
            *[int(s['owners'][j]==s['owners'][j-1]) if j<visible else -1 for j in range(1,roles)])


def forecast(states,evidence,tier,reader):
    values=np.zeros(128)
    for s in states:
        compatible=s['endpoint']==evidence[0] if reader=='endpoint' else view(s,tier)==tuple(evidence)
        if compatible and not (reader=='single-person' and s['topology']==2):values[s['label']]+=s['mass']
    fallback=not values.sum()
    if fallback:
        # A contradicted single-person model has no conditional posterior.
        # Its explicit fallback remains the endpoint-conditioned single-person prior.
        for s in states:
            if s['endpoint']==evidence[0] and s['topology']!=2:values[s['label']]+=s['mass']
    return values/values.sum(),fallback


def controls():
    prior=np.array([1/3]*3);aligned=catalog(3,False,prior);opposed=catalog(3,True,prior)
    q,_=forecast(aligned,view(aligned[0],'full'),'full','role-mixture')
    z,_=forecast(opposed,view(opposed[2],'full'),'full','role-mixture')
    absent=[forecast(opposed,view(s,'none'),'none','role-mixture')[0] for s in opposed[::2]]
    return dict(live_role_witness=bool(z[opposed[2]['label']]==1),placebo_endpoint_twins=bool(all(np.array_equal(absent[0],a) for a in absent)),
        positive_aligned_alias=bool(np.isclose(q[aligned[0]['label']],.5) and np.isclose(q[aligned[2]['label']],.5)),
        positive_two_maker_record=bool(forecast(opposed,view(opposed[4],'full'),'full','role-mixture')[0][opposed[4]['label']]==1))


def run(root,d,pulse):
    from .authorship_check import reference
    population=[];cases=[];maximum=0.;aliases=[]
    for world in d['lineages']:
        rng=np.random.default_rng(W.seed('v21-e02-prior',world));prior=rng.dirichlet(np.full(3,4.))
        rng=np.random.default_rng(W.seed('v21-e02-cases',world));topology=rng.choice(3,size=d['test_cases'],p=prior);endpoint=rng.integers(2,size=d['test_cases'])
        for roles,conflict in product((2,3),(False,True)):
            states=catalog(roles,conflict,prior);selected=[states[2*t+e] for t,e in zip(topology,endpoint)];truth=np.array([s['label'] for s in selected]);token=f'{world}-r{roles}-c{int(conflict)}'
            eroot=root/'evaluator';eroot.mkdir(exist_ok=True);np.savez_compressed(eroot/f'{token}_points.npz',truth=truth,topology=topology,goals=np.array([s['goals'] for s in selected]),owners=np.array([s['owners'] for s in selected]),prior=prior)
            for tier in ('none','partial','full'):
                evidence=np.array([view(s,tier) for s in selected]);rr=root/'reader';rr.mkdir(exist_ok=True);np.savez_compressed(rr/f'{token}-{tier}_points.npz',evidence=evidence)
                classes={}
                for s in states:classes.setdefault(view(s,tier),[]).append(s['label'])
                aliases.append(dict(lineage=world,roles=roles,conflict=conflict,tier=tier,classes=[dict(evidence=list(k),labels=v) for k,v in classes.items()]))
                for reader in READERS:
                    use_tier='full' if reader=='full-record-oracle' else tier
                    vals=[forecast(states,view(s,use_tier),use_tier,reader) for s in selected];prob=np.array([x[0] for x in vals]);fallback=np.array([x[1] for x in vals])
                    for s in states:
                        actual,_=forecast(states,view(s,use_tier),use_tier,reader);ref=reference(roles,conflict,prior,view(s,use_tier),use_tier,reader);err=float(abs(actual-ref).max());maximum=max(maximum,err);assert err<1e-12
                    with np.errstate(divide='ignore'):lp=np.log(prob)
                    metrics=retain(root,f'{token}-{tier}-{reader}',lp,truth)
                    population.append(dict(lineage=world,roles=roles,conflict=conflict,tier=tier,reader=reader,metrics=metrics,fallback_fraction=float(fallback.mean())))
                if world==d['lineages'][0]:cases.append(dict(roles=roles,conflict=conflict,tier=tier,evidence=evidence[0].tolist(),truth=int(truth[0]),role='evaluator only'))
                pulse(phase='authorship-topology',lineage=world,roles=roles,conflict=conflict,tier=tier)
    write(root/'ALIAS_CLASSES.json',aliases);write(root/'CASEBOOK_EVALUATOR.json',cases)
    return dict(card='G21-E02',population=population,controls=controls(),independent_max_probability_error=maximum,
        scope='Finite supplied-prior role-history information ruler; no learned authorship detection or human attribution.',
        limitations=['Final output is the last role goal; all three topologies have exactly matched endpoint distributions.',
        'Role-conditioned aims can align; full records cannot distinguish aligned stable versus role-conditioned single-maker policies.',
        'Population topology prior, role catalog and noiseless source-link equality are supplied. Source linkage is observable only in declared records.',
        'The direct table and role mixture have identical supplied information. No fitting, convergence or architecture superiority claim.',
        'Single-person inference excludes two-maker truth. Contradictory records invoke its explicit endpoint-prior fallback, not a valid posterior.',
        'Full-record oracle receives all role outputs and source links, not latent topology truth; aligned aliases remain.',
        'Eight reused development worlds are uncertainty units. Shared draws pair all conditions; cases are not independent worlds.',
        'Native only; no execution shift, duplicate-product arm, confirmation, contribution percentage or real author identification.',
        'Purpose-built miniature — architecture untested; this is a transparent exact alias ruler.'])
