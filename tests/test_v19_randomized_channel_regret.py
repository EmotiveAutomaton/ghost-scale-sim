from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import pytest
from ghostscale.validation.soundingline.v19 import randomized_channel_regret as M
from ghostscale.validation.soundingline.v19 import robust_channel_regret as D


def hull_value(points):
    # Independent lower hull, followed by intersections with the diagonal.
    points=sorted(set(points)); low=[]
    for p in points:
        if low and p[0]==low[-1][0]:
            continue
        while len(low)>1:
            a,b=low[-2:]
            cross=(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
            if cross>0: break
            low.pop()
        low.append(p)
    values=[max(p) for p in low]
    for a,b in zip(low,low[1:]):
        da=a[0]-a[1];db=b[0]-b[1]
        if da*db<0:
            t=da/(da-db)
            values.append((1-t)*a[0]+t*b[0])
    return min(values)


def fixture():
    return M.fixture([1,2,3],[1,2,1],2,{'a':[1,3],'b':[2],'c':[3]},
        [[F(1,2),F(1,3),F(1,6)],[F(1,6),F(2,3),F(1,6)],[F(0),F(0),F(1)]])


def test_known_answers():
    assert all(M.controls().values())


@pytest.mark.parametrize('counts',M.MIXTURES)
def test_convex_hull_and_intersections(counts):
    r=M.solve(fixture(),counts)
    assert r['deterministic']==D.solve(fixture(),counts)
    endpoint={tuple(x['policy']):tuple(F(*v) for v in x['regrets']) for x in r['endpoint_regrets']}
    assert F(*r['worst_expected_regret'])==hull_value(endpoint.values())
    assert F(*r['improvement_over_deterministic'])>=0
    lines={tuple(x['policy']):(F(*x['intercept']),F(*x['slope'])) for x in r['deterministic']['policies']}
    points={F(1,3),F(1),F(2,3)}
    for a,b in combinations(lines.values(),2):
        if a[1]!=b[1]:
            x=(b[0]-a[0])/(a[1]-b[1])
            if F(1,3)<=x<=1:points.add(x)
    score=lambda p,x:lines[p][0]+lines[p][1]*x
    optimum={x:max(score(p,x) for p in lines) for x in points}
    for c in r['candidate_lotteries']:
        p=tuple(c['first_policy']);q=tuple(c['second_policy']);w=F(*c['first_weight'])
        assert 0 < w <= 1
        maximum=max(optimum[x]-w*score(p,x)-(1-w)*score(q,x) for x in points)
        assert F(*c['worst_expected_regret'])==maximum


def test_duplicate_and_dominated_policy_invariance():
    endpoints={(1,):(F(1),F(0)),(2,):(F(0),F(1))}
    assert M.lotteries(endpoints)[1]==F(1,2)
    endpoints[(3,)]=endpoints[(1,)];endpoints[(4,)]=(F(2),F(2))
    assert M.lotteries(endpoints)[1]==hull_value(endpoints.values())==F(1,2)


def test_null_and_point_interval():
    s=fixture();a=M.solve(s,(1,1,2),((2,3),(2,3)))
    assert a['worst_expected_regret']==[0,1]
    for row in s['candidates']:row['masses']=[[1,2]]*3
    assert M.solve(s,(1,1,2))['worst_expected_regret']==[0,1]


def test_permutation_and_individual_feasibility():
    s=fixture();changed=deepcopy(s)
    for row in changed['candidates']:row['masses']=[row['masses'][i] for i in (2,0,1)]
    a=M.solve(s,(1,1,2));b=M.solve(changed,(2,1,1))
    assert a['worst_expected_regret']==b['worst_expected_regret']
    assert all(d['maximum_realized_bytes']<=s['capacity_bytes'] for d in a['diagnostics'])
    s['candidates'][0]['used_bytes']=s['capacity_bytes']+1
    with pytest.raises(AssertionError):M.solve(s,(1,1,2))


def test_expected_and_realized_regret_are_distinct():
    values,best,_,chosen=M.lotteries({(1,):(F(1),F(0)),(2,):(F(0),F(1))})
    assert best==F(1,2) and max(values[chosen])==best
    assert best < 1  # Every realized singleton has worst regret one.
