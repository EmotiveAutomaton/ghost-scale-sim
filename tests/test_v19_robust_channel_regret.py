from fractions import Fraction as F
from itertools import product
from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v19 import robust_channel_regret as M


def fixture():
    return M.fixture([1,2,3],[1,2,1],2,{'a':[1,3],'b':[2],'c':[3]},
        [[F(1,2),F(1,3),F(1,6)],[F(1,6),F(2,3),F(1,6)],[F(0),F(0),F(1)]])


def test_known_answers():
    assert all(M.controls().values())


@pytest.mark.parametrize('counts',M.MIXTURES)
def test_endpoint_reduction_against_all_intersections(counts):
    s=fixture();r=M.solve(s,counts)
    mass={x['mask']:[F(*v) for v in x['masses']] for x in s['candidates']}
    policies=list(product(mass,repeat=3))
    def score(p,q):
        # Sum by cue first, independently of the producer's line evaluation.
        return sum(sum(F(counts[j],4)*(q if k==j else (1-q)/2)*mass[p[k]][j]
                       for j in range(3)) for k in range(3))
    intercept={p:score(p,F(0)) for p in policies}
    slope={p:score(p,F(1))-intercept[p] for p in policies}
    points={F(1,3),F(2,3),F(1)}
    for p,q in product(policies,repeat=2):
        if slope[p]!=slope[q]:
            x=(intercept[q]-intercept[p])/(slope[p]-slope[q])
            if F(1,3)<=x<=1:points.add(x)
    worst={p:max(max(score(q,x) for q in policies)-score(p,x) for x in points) for p in policies}
    assert F(*r['worst_regret'])==min(worst.values())
    assert r['optimal_policies']==[list(p) for p in policies if worst[p]==min(worst.values())]
    assert r['selected_policy']==max(r['optimal_policies'])
    assert all(F(*v['worst_regret'])==worst[tuple(v['policy'])] for v in r['policies'])
    assert F(*r['improvement_over_fixed'])>=0 and F(*r['improvement_over_nominal'])>=0
    narrow=M.solve(s,counts,((2,3),(1,1)))
    assert F(*narrow['worst_regret'])<=F(*r['worst_regret'])


def test_realized_loss_differs_from_expected_regret():
    s=M.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])
    r=M.solve(s,(3,1,0));d=r['diagnostics'][0]['arms']['robust']
    assert d['regret']==[1,6] and d['maximum_realized_coverage_loss']==[1,1]
    assert all(max(v['used_bytes'])<=s['capacity_bytes'] for v in r['policies'])


def test_joint_label_permutation_preserves_optimum():
    s=fixture();q=deepcopy(s)
    for r in q['candidates']:r['masses']=[r['masses'][i] for i in (2,0,1)]
    a=M.solve(s,(1,1,2));b=M.solve(q,(2,1,1))
    assert a['worst_regret']==b['worst_regret']
    assert {tuple(p[i] for i in (2,0,1)) for p in a['optimal_policies']}=={tuple(p) for p in b['optimal_policies']}


def test_corrupt_cost_rejected():
    s=fixture();s['candidates'][0]['used_bytes']=3
    with pytest.raises(AssertionError):M.solve(s,(1,1,2))
