"""K5-R1: retained G3 observations, unchanged scoring, closure-balance covariate.

No generator is called. The original nine measures and closure times must be
reconstructed before stratification. Old whole-world blocks remain the units;
empty strata have zero denominators, never invented observations.
"""
import gzip,json
from pathlib import Path
import numpy as np
from ..v26 import day_closure as old
from .common import read,digest,close,measure


def strata(items,lengths):
    closed=lengths>=0
    balanced=np.array([when>=0 and np.sum(row[:when]==0)==np.sum(row[:when]==1)
                       for row,when in zip(items,lengths)])
    return dict(balanced_closed=balanced,unbalanced_closed=closed&~balanced,open=~closed)


def controls():
    assert old.controls()['passed']
    items=np.array([[1,0,-1,1],[1,1,0,0],[1,0,1,0]],dtype=np.int8)
    ss=strata(items,np.array([3,2,-1]))
    np.testing.assert_array_equal(sum(ss.values()),np.ones(3,dtype=int))
    assert ss['balanced_closed'].tolist()==[True,False,False]
    assert ss['unbalanced_closed'].tolist()==[False,True,False]
    return dict(passed=True,original_G3_known_answers=True,partition_covers_each_source_once=True,
                generators_called=False,empty_strata_are_zero_denominators=True)


def analyze(saved,s,independent=False):
    original=[];extra=[];kernel=old.review_trajectories if independent else old.trajectory
    names=['K_logscore','D_logscore','T_logscore','O_logscore','K_capture','D_capture','closed','D_minus_K_logscore','D_minus_K_capture']
    for row in saved['data']['observations']:
        source=row['source'];code=['H','M','Mx','Mc','HU'].index(source)
        truth=np.array(row['truth']);items=np.array(row['items'],dtype=np.int8)
        values,lengths=kernel(items,truth,code,s['theta'],s['delta'],s['switch'],s['overpush'],s['pull_away'],*old.library(s['expanded']))
        groups=strata(items,lengths)
        for true in (0,1):
            mask=truth==true;labels=dict(source=source,truth=true,theta=s['theta'],delta=s['delta'])
            for j,name in enumerate(names):measure(original,name,values[mask,j].tolist(),**labels)
            measure(original,'closure_record',[None if v<0 else int(v) for v in lengths[mask]],'median',**labels)
            for group,keep in groups.items():
                eligible=mask&keep
                measure(extra,'retained_stratum_share',keep[mask].astype(float).tolist(),stratum=group,**labels)
                for j,name in ((4,'K_capture'),(5,'D_capture'),(8,'D_minus_K_capture')):
                    measure(extra,'conditional_'+name,[[float(v) if yes else 0.,int(yes)] for v,yes in zip(values[mask,j],keep[mask])],
                            'ratio_of_sums',stratum=group,**labels)
    close(original,saved['data']['measures'],tol=1e-9)
    return dict(observations=saved['data']['observations'],measures=original+extra,
                original_measures_reproduced=True,retained_covariate='exactly balanced observed zeros and ones at the original closure')


def retained_block(s,plan,setting,block,independent=False):
    path=Path(plan['retained_manifest'])
    if digest(path)!=plan['retained_manifest_sha256']:raise ValueError('retained manifest changed')
    manifest=read(path)
    if digest(Path(old.__file__))!=manifest['old_module_sha256']:raise ValueError('earlier scoring source changed')
    mode='probe' if plan['scientific_kind']=='probe' else 'study'
    index=plan['setting_indices'][setting];bound=manifest[mode][index][block]
    raw=Path(bound['path'])
    if digest(raw)!=bound['sha256']:raise ValueError('retained raw data changed')
    with gzip.open(raw,'rt',encoding='utf-8') as f:saved=json.load(f)
    if saved['configuration']!=s:raise ValueError('earlier frozen configuration changed')
    return analyze(saved,s,independent)|dict(ancestor_raw_sha256=bound['sha256'])


def precheck(card,setting):
    # The prior excluded probe is disjoint from all retained scientific blocks.
    p=card|dict(scientific_kind='probe',setting_indices=list(range(len(card['settings']))))
    return retained_block(card['settings'][setting],p,setting,0)


def design_check(s,identity):
    return dict(passed=True,scientific_draws=0,retained_data_only=True,
                case='Original G3 case remains unchanged; this is a descriptive covariate analysis, not a new signal gate.',
                expected_signs=dict(balance='No new directional bar; original nine scores are reproduced before stratification.'))
