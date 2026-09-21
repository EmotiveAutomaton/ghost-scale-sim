from itertools import product
import numpy as np
from ghostscale.validation.soundingline.v19 import practice as P,local_world as W


def test_known_control_placebo_and_replay():
    assert all(P.controls().values())


def test_native_conditional_law_and_scalar_dynamic_program():
    world=W.law(-9631);p=P.native_kernel(world)
    for step,context,a,previous,g in product(range(3),range(2),range(8),range(8),range(3)):
        rows=[(op,w) for goal,op,w in W.choices(world,P.ACTOR,P.ARTIFACTS[a],step,W.CONTEXTS[context*2]) if goal==P.GOALS[g]]
        expected=np.zeros(8)
        for op,w in rows:expected[P.INDEX[W.execute(P.ARTIFACTS[a],P.ARTIFACTS[previous],op,P.ACTOR)]]+=w/sum(v for _,v in rows)
        assert np.allclose(expected,p[step,context,a,previous,g])
    pi,v=P.policy(p);measured,truth,occupancy=P.evaluate(p,pi)
    def exact(step,context,a,previous):
        if step==3:return P.SUCCESS[a]
        return max(sum(p[step,context,a,previous,g,b]*exact(step+1,context,b,a) for b in range(8) if p[step,context,a,previous,g,b]>0) for g in range(3))
    for context,a in enumerate(P.INITIAL):assert abs(measured[context]-exact(0,context,a,a))<1e-12
    assert np.allclose(v,truth) and abs(occupancy.sum()-3)<1e-12


def test_persistent_practice_retains_replay_and_restricted_prior(tmp_path):
    cfg=dict(lineages=[-9641],training_draws=[-9642],policy_seeds=[9643],budgets=[2,4],epsilon=.2)
    out=P.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert out['rows']==12 and out['tiny_settings']==0
    for condition,episode in product(('matched-start','restricted-start'),[2,4]):
        stem=f'-9641--9642-9643-{condition}-{episode}'
        with np.load(tmp_path/'models'/f'{stem}-active.npz') as a,np.load(tmp_path/'models'/f'{stem}-replay.npz') as b:
            assert all(np.array_equal(a[k],b[k]) for k in a.files)
            assert a['counts'].sum()==np.prod(P.SHAPE)*.5+episode*3
            if condition=='restricted-start':assert np.all(a['counts'][:,1]==.5)
