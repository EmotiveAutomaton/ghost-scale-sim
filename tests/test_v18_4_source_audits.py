from copy import deepcopy
from itertools import product
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_4 import source_audits as S


def brute_value(p,channels,prices,stake,depth=2):
    """Enumerate root actions and both conditional second actions, then decisions."""
    def terminal(weights):
        return stake*max(math.fsum(weights[:4]),math.fsum(weights[4:]))
    values=[terminal(p)]
    for first in channels:
        options=['stop']+[a for a in channels if a=='evidence' or a!=first]
        for second in product(options,repeat=2) if depth==2 else [('stop','stop')]:
            score=-prices[first]
            for outcome in (0,1):
                mass=[float(p[s])*(float(channels[first][s]) if outcome else 1-float(channels[first][s])) for s in range(8)]
                if second[outcome]=='stop':score+=terminal(mass)
                else:
                    action=second[outcome];score-=math.fsum(mass)*prices[action]
                    for value in (0,1):
                        weights=[mass[s]*(float(channels[action][s]) if value else 1-float(channels[action][s])) for s in range(8)]
                        score+=terminal(weights)
            values.append(score)
    return max(values)


def test_live_placebo_and_independent_likelihood_enumeration():
    assert all(S.controls().values())
    params=S.parameters(991)
    for shared in (0.,.2):
        a=S.report_table(params,shared);b=S.report_table(params,shared,scalar=True)
        assert np.allclose(a,b,atol=1e-14,rtol=0)
        assert np.allclose(a.sum(0),1,atol=1e-14,rtol=0)


def test_dynamic_program_matches_exhaustive_policy_enumeration():
    r=np.random.default_rng(144)
    for _ in range(8):
        p=r.random(8);p/=p.sum();c={a:r.uniform(.1,.9,8) for a in ('evidence','audit0','audit1')}
        prices=dict(evidence=.04,audit0=.03,audit1=.05)
        for depth in (1,2):
            actual=S.choose(p,(),'lookahead',c,2.,prices,depth)[1]
            assert actual==pytest.approx(brute_value(p,c,prices,2.,depth),abs=1e-12)


def test_joint_only_information_fixture_and_cost_reconstruction():
    p=np.ones(8)/8
    c=dict(evidence=np.array([.01,.99,.01,.99,.99,.01,.99,.01]),
           audit0=np.array([.01,.99,.01,.99]*2),audit1=np.ones(8)*.5)
    prices=dict(evidence=.05,audit0=.05,audit1=.05)
    assert S.choose(p,(),'myopic',c,1.,prices,2)[0]=='stop'
    assert S.choose(p,(),'lookahead',c,1.,prices,2)[1]==pytest.approx(.9802-.1)
    leaves=S.leaves(p,p,'lookahead',c,c,1.,prices)
    metrics,_=S.metrics([dict(leaves=leaves)],1.,prices)
    assert metrics['cost']==pytest.approx(.1)
    assert metrics['net_utility']==pytest.approx(.8802)


def test_uninformative_and_correctly_inverted_audit():
    p=np.arange(1.,9.);p/=p.sum();params=S.parameters(992)
    uninformative=S.channels(params,.5)
    for value in (0,1):assert np.array_equal(S.update(p,uninformative['audit0'],value)[0],p)
    positive=S.channels(params,.8);negative=S.channels(params,.2)
    assert np.allclose(S.update(p,positive['audit0'],1)[0],S.update(p,negative['audit0'],0)[0])
    mass=p/64
    paths=S.leaves(p,mass,'common',uninformative,uninformative,1.,dict(evidence=.1,audit0=.07,audit1=.07))
    scores,_=S.metrics([dict(leaves=paths)],1.,dict(evidence=.1,audit0=.07,audit1=.07))
    assert scores['cost']==pytest.approx(.17/64)


def test_decision_uses_only_reader_state_and_costs():
    params=S.parameters(993);p=np.arange(1.,9.);p/=p.sum();c=S.channels(params,.75)
    prices=dict(evidence=.1,audit0=.05,audit1=.05)
    a=S.leaves(p,np.ones(8)/8,'lookahead',c,c,1.,prices)
    b=S.leaves(p,np.array([1.,0,0,0,0,0,0,0]),'lookahead',c,c,1.,prices)
    assert [(x['history'],x['posterior']) for x in a]==[(x['history'],x['posterior']) for x in b]


def test_exact_case_reconstruction_and_corruption():
    u=S.unit(994,shared=.2,audit_true=.5,audit_assumed=.95)
    assert S.verify(u)
    for mutation in ('cost','posterior','mass','reports'):
        bad=deepcopy(u)
        if mutation=='cost':bad['rows'][0]['cost']+=.01
        elif mutation=='reports':bad['rows'][0]['cases'][0]['reports'][0]=1
        else:bad['rows'][0]['cases'][0]['leaves'][0][mutation][0]+=.01
        with pytest.raises(ValueError):S.verify(bad)


def test_dispatch_independent_means_and_complete_unit_replay(tmp_path):
    from ghostscale.validation.soundingline.v18_4 import runtime as R
    from ghostscale.validation.soundingline.v18_3.io import write,file_digest
    from runners.replay_v18_4 import finite
    spec=dict(family='S1',index=995,shared=0.,audit_true=.75,audit_assumed=.75,stake=.25,audit_price=.15)
    write(tmp_path/'PLAN.json',dict(design=dict(units=[spec],block_size=1)))
    u=R.dispatch(spec);R.keep(tmp_path,'block-000000',[u],0.,0.);R.aggregate(tmp_path)
    write(tmp_path/'COMPLETE.json',dict(blocks=['block-000000'],summary_sha256=file_digest(tmp_path/'SUMMARY.json')))
    proof=finite(tmp_path)
    assert proof['independent_means']==len(S.METHODS)*len(S.METRICS)
    assert proof['whole_unit_replays']==[0]
