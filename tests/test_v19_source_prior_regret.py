from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import pytest
from ghostscale.validation.soundingline.v19.robust_source_mass import choose
from ghostscale.validation.soundingline.v19.source_prior_regret import minimize, controls


def example():
    return choose([1,2,3],[1,1,1],1,{'a':[1],'b':[2],'c':[3]},[[F(2,5),F(3,5),F(0)],[F(2,5),F(3,10),F(3,10)]])


def test_controls():assert all(controls().values())


@pytest.mark.parametrize('capacity',range(5))
def test_exhaustive_subsets_and_convex_mixtures(capacity):
    times=[1,2,3,4];priors=[[F(1,4)]*4,[F(i,10) for i in times],[F(4-i,10) for i in range(4)]]
    subsets=[list(s) for k in range(capacity+1) for s in combinations(times,k)]
    library={str(s):s for s in subsets};s=choose(times,[1]*4,capacity,library,priors);a=minimize(s)
    scores={}
    for r in s['candidates']:
        masses=[F(*v) for v in r['masses']]
        regrets=[]
        for i in range(9):
            for j in range(9-i):
                mixture=[F(i,8),F(j,8),F(8-i-j,8)]
                best=max(sum(w*F(*v) for w,v in zip(mixture,c['masses'])) for c in s['candidates'])
                regrets.append(best-sum(w*v for w,v in zip(mixture,masses)))
        scores[r['mask']]=max(regrets)
    best=min(scores.values());ties=sorted(k for k,v in scores.items() if v==best)
    assert a['regret_optimal_masks']==ties and a['regret_selected_mask']==ties[-1]
    assert F(*a['minimax_regret'])==best


def test_identical_priors_and_exact_ties():
    s=choose([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3);a=minimize(s)
    assert a['regret_optimal_masks']==[1,2] and a['regret_selected_mask']==2
    assert a['minimax_regret']==[0,1]


def test_prior_permutation():
    s=example();t=deepcopy(s)
    for k in ('best_prior_mass','prior_optimal_masks'):t[k].reverse()
    for r in t['candidates']:
        for k in ('masses','prior_regrets'):r[k].reverse()
    assert minimize(s)==minimize(t)


def test_identity_order_has_no_effect():
    s=example();t=deepcopy(s);t['candidates'].reverse()
    for r in t['candidates']:r['identities']=['unrelated name']
    assert minimize(s)==minimize(t)


@pytest.mark.parametrize('corruption',['optimum','regret','mass','robust','duplicate','empty'])
def test_corrupt_selection_fails(corruption):
    s=example()
    if corruption=='optimum':s['best_prior_mass'][0]=[1,1]
    if corruption=='regret':s['candidates'][0]['prior_regrets'][0]=[999,1]
    if corruption=='mass':s['candidates'][0]['masses'][0]=[-1,1]
    if corruption=='robust':s['selected_mask']=99
    if corruption=='duplicate':s['candidates'].append(s['candidates'][0])
    if corruption=='empty':s['candidates']=[]
    with pytest.raises(ValueError):minimize(s)
