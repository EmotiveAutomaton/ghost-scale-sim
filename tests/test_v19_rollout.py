import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import rollout as R
from ghostscale.validation.soundingline.v19 import local_world as L

def test_known_controls():assert all(R.controls().values())

def test_mechanics_and_roles():
    assert R.oracle((1,0,2,3,4,4))==6
    assert R.oracle((1,1,4,1,4,4))==4
    with pytest.raises(ValueError):R.oracle((0,0,2,3,4,4))
    q=(1,0,2,4,4,4);p=R.visible(q);assert R.validate_visible(p)
    with pytest.raises(ValueError):R.validate_visible(dict(p,endpoint=2))

def test_empirical_counts_and_cost_control():
    r=dict(maker=[0,1,0,0],initial=[0,1,0],final=[0,1,0],steps=[dict(operation='inspect',before=[0,1,0],undo_buffer=[0,1,0],after=[0,1,0])]*3)
    counts,d=R.fit([r]);assert counts[1,0,2,2,4,2]==4 and d[R.query(r)][2]==2
    p=R.forecasts(R.normalized(counts),d,R.query(r),np.full((16,3),.01))
    assert set(p)==set(R.ARMS) and all(np.isclose(v.sum(),1) for v in p.values())
    assert np.array_equal(p['direct-mc16'],R.estimate([0]*16))
    assert p['oracle-exact'][2]==1

def test_propagation_matches_explicit_path_sum():
    table=R.normalized(np.arange(2*2*8*8*6*8,dtype=float).reshape(2,2,8,8,6,8)+1)
    q=(1,0,2,0,5,4);expected=np.zeros(8)
    for a in range(8):
        for b in range(8):
            for d in range(8):expected[d]+=table[1,0,2,2,0,a]*table[1,0,a,2,5,b]*table[1,0,b,a,4,d]
    assert np.allclose(R.propagate(table,q),expected,atol=1e-14)

def test_full_fixture(tmp_path):
    cfg=dict(arms=list(R.ARMS),train_lineages=[190973],development_lineages=[190974],training_draws=[190301],paths_per_lineage=32,budgets=[32])
    s=R.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert all(s['controls'].values()) and s['rows']>0 and len(s['cells'])==8
    assert next(r for r in s['cells'] if r['arm']=='oracle-exact')['loss']==0
