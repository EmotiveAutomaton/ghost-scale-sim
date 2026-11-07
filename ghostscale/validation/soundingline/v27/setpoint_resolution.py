"""Excluded resolution diagnostics; never a scientific result or a new seed lane.

The tolerances and all fixture identities are frozen before execution. Candidate
likelihoods receive only observed actions/sizes/shocks and an inference seed key.
The checks compare complete marginalized dynamic posteriors and decision paths,
not just a single true-hypothesis likelihood.
"""
import itertools
import numpy as np
from . import setpoints, setpoint_particles
from .common import posteriors, margin, sustained

COUNTS=(2048,8192,32768)
LIMITS=dict(posterior=.05,margin=.25,crossing_records=16,static_log_likelihood=.1)


def hidden_observations(raw):
    return dict(inference_key=raw['inference_key'],records=[
        {k:r[k] for k in ('action','size','shock')} for r in raw['records']])


def paths(raw,s,count,replicate,independent=False):
    points=list(itertools.product((8,11,14),repeat=s['needs']))
    hypotheses=list(itertools.product(points,setpoints.DYNAMICS))
    observed=hidden_observations(raw);logs=[]
    exact=s|dict(inference='exact')
    for point,dynamic in hypotheses:
        if dynamic=='P0':ll,_=setpoints.likelihood(observed,exact,point,dynamic,s['width'],s['beta'],independent=independent)
        else:ll,_=setpoint_particles.likelihood(observed,s,point,dynamic,s['width'],s['beta'],independent=independent,particles=count,replicate=replicate)
        logs.append(ll)
    logs=np.asarray(logs).T;prior=np.full(len(hypotheses),1/len(hypotheses))
    p=posteriors(logs,prior,independent)
    marginal=np.column_stack([p[:,[d==law for _,d in hypotheses]].sum(axis=1) for law in setpoints.DYNAMICS])
    contrasts={}
    for law in ('S','P-','G'):
        yes=np.array([d==law for _,d in hypotheses]);no=np.array([d in ('S','P-','G') and d!=law for _,d in hypotheses])
        contrasts[law]=margin(logs,prior,yes,no)
    return dict(posterior=marginal,margins=contrasts)


def compare(reference,candidate,begin=0):
    posterior=float(np.max(np.abs(reference['posterior']-candidate['posterior'])))
    error=max(float(np.max(np.abs(reference['margins'][law]-candidate['margins'][law]))) for law in ('S','P-','G'))
    crossings={};crossing_pass=True
    for law in ('S','P-','G'):
        a=sustained(reference['margins'][law][begin:]);b=sustained(candidate['margins'][law][begin:])
        stable=(a is None and b is None) or (a is not None and b is not None and abs(a-b)<=LIMITS['crossing_records'] and (a<=128)==(b<=128))
        crossings[law]=dict(reference=a,candidate=b,passed=stable);crossing_pass &= stable
    return dict(posterior_error=posterior,margin_error=error,crossings=crossings,
        passed=posterior<=LIMITS['posterior'] and error<=LIMITS['margin'] and crossing_pass)


def fixture(s,identity,pulse=lambda **kw:None,counts=COUNTS,retained=None):
    raw=setpoints.generate(s|dict(inference='checked_particles'),identity)
    all_paths={}
    if retained is not None:
        if retained['identity']!=identity or retained['configuration']!=s:raise ValueError('retained fixture identity differs')
        for row in retained['diagnostic_paths']:
            all_paths[row['particles'],row['replicate']]=dict(posterior=np.array(row['posterior']),margins={k:np.array(v) for k,v in row['margins'].items()})
    for count in counts:
        for replicate in range(2):
            if (count,replicate) not in all_paths:all_paths[count,replicate]=paths(raw,s,count,replicate)
            pulse(particles=count,replicate=replicate)
    reference=all_paths[counts[-1],0]
    checks=[dict(particles=count,replicate=replicate,**compare(reference,all_paths[count,replicate],raw['first_long_deficit'] or 0))
        for count in counts[-2:] for replicate in range(2) if (count,replicate)!=(counts[-1],0)]
    return dict(identity=identity,configuration=s,limits=LIMITS,counts=list(counts),replicates=2,
        scientific_draws=0,checks=checks,passed=all(x['passed'] for x in checks),
        diagnostic_paths=[dict(particles=count,replicate=replicate,posterior=value['posterior'].tolist(),
            margins={k:v.tolist() for k,v in value['margins'].items()}) for (count,replicate),value in all_paths.items()])
