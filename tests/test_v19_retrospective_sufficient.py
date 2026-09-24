import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import retrospective_sufficient as S,reachable_retrospective as Q
from test_v19_reachable_retrospective import fixture,law


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
def test_complete_joint_factor_reconstruction_and_forecast_bound(mode):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);W=np.array([[.6,.1,.2,.1],[0,.5,0,.5]]);p=law(mode)
    raw,structures,metrics=S.evaluate(W,st,p)
    for time,s in enumerate(structures,1):
        joint=np.empty((2,len(s['pairs'])))
        joint[:,s['mixed']]=raw[f'{time}-mixed_joint_mass']
        joint[:,~s['mixed']]=raw['group_weights'][:,s['pairs'][~s['mixed'],0]]
        for row,w in enumerate(W):
            for i,(g,state) in enumerate(s['pairs']):
                expected=sum(w[h] for h in range(len(w)) if st['mapping'][h]==g and st['past'][time-1,h]==state)
                assert joint[row,i]==pytest.approx(expected,abs=1e-15)
            for c in range(4):
                for e in range(8):
                    nu=np.array([sum(joint[row,i]*p[state,c,e] for i,(g,state) in enumerate(s['pairs']) if g==x) for x in range(3)])
                    direct=np.array([sum(w[h]*p[st['past'][time-1,h],c,e] for h in range(len(w)) if st['mapping'][h]==x) for x in range(3)])
                    if not direct.sum():assert not raw[f'{time}-possible'][row,8*c+e];continue
                    a,b=nu/nu.sum(),direct/direct.sum();tv=.5*abs(a-b).sum()
                    for offset in range(3):
                        diff=sum((a[g]-b[g])*p[path[offset]] for g,path in enumerate(st['signatures']))
                        assert abs(diff).max()<=tv+1e-15
                    assert tv<1e-12
    assert all(S.controls().values())


def test_dropped_joint_cell_changes_past_report():
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);W=np.array([[.6,.1,.2,.1]]);p=law('random');raw,structures,_=S.evaluate(W,st,p)
    s=structures[0];mass=raw['1-mixed_joint_mass'].copy();mass[0,0]=0
    assert abs(mass.sum()-raw['1-mixed_joint_mass'].sum())>.1
    # Corrupted mapping is refused before constructing a table.
    broken=fixture();broken['membership'][1]=1
    with pytest.raises(ValueError,match='schedule mapping'):Q.prepare(broken,[.2,.3,.4,.1])


def test_total_variation_bound_can_be_tight():
    a=np.array([1.,0.]);b=np.array([0.,1.]);f=np.array([1.,0.])
    assert abs((a-b)@f)==.5*abs(a-b).sum()==1
