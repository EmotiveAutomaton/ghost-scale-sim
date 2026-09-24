import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import two_report_retention as S, reachable_retrospective as Q
from test_v19_reachable_retrospective import fixture,law


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('weights',[[.6,.1,.2,.1],[0,.5,0,.5]])
@pytest.mark.parametrize('times',[(1,1),(1,2),(2,1),(2,2)])
def test_direct_hypothesis_pair_bayes_and_all_future_bounds(mode,weights,times):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);p=law(mode);w=np.array(weights);a,b=times
    r=S.pair_state(w,st,p,a,b);G=len(st['signatures']);q=np.bincount(st['mapping'],weights=w,minlength=G)
    first=p[st['past'][a-1]].reshape(len(w),32);second=p[st['past'][b-1]].reshape(len(w),32)
    full=np.zeros((G,32,32));one=np.zeros((G,32));two=np.zeros((G,32))
    for h,g in enumerate(st['mapping']):
        full[g]+=w[h]*np.outer(first[h],second[h]);one[g]+=w[h]*first[h];two[g]+=w[h]*second[h]
    separate=full if a==b else np.divide(one[:,:,None]*two[:,None,:],q[:,None,None],out=np.zeros_like(full),where=q[:,None,None]>0)
    np.testing.assert_allclose(r['report_probability'],full.sum(0),atol=2e-15,rtol=0)
    np.testing.assert_allclose(r['separate_report_probability'],separate.sum(0),atol=2e-15,rtol=0)
    for i in range(32):
        for j in range(32):
            den=full[:,i,j].sum();rd=separate[:,i,j].sum()
            if not den or not rd:
                assert np.isnan(r['group_total_variation'][i,j]);continue
            exact=full[:,i,j]/den;approx=separate[:,i,j]/rd
            assert r['group_total_variation'][i,j]==pytest.approx(.5*abs(exact-approx).sum(),abs=2e-15)
            for offset in range(st['signatures'].shape[1]):
                delta=(exact-approx)@p[st['signatures'][:,offset]].reshape(G,32)
                assert np.max(.5*abs(delta.reshape(4,8)).sum(-1))<=r['group_total_variation'][i,j]+2e-15
    np.testing.assert_allclose(r['true_mixed'],full[r['mixed_groups']],atol=2e-15,rtol=0)
    reverse=S.pair_state(w,st,p,b,a)
    np.testing.assert_allclose(r['group_total_variation'],reverse['group_total_variation'].T,atol=2e-15,rtol=0)
    if a==b or mode=='uniform':assert np.nanmax(abs(r['group_total_variation']))<1e-12


def test_real_dependence_and_deleted_joint_cell():
    # Two histories share their entire future but retain correlated past states.
    spec=dict(hypotheses=[['none',0,0],['purpose',2,8],['none',0,1]],length=4,checkpoint=3,
              signatures=[[0,0],[1,1]],membership=[0,0,1])
    w=np.array([.3,.3,.4]);st=Q.prepare(spec,w);p=law('random')
    r=S.pair_state(w,st,p,1,2)
    assert np.nanmax(r['group_total_variation'])>1e-4
    bad=r['joint'].copy();bad[0,0,0]=0
    wrong=(p.reshape(16,32).T@bad)@p.reshape(16,32)
    assert np.max(abs(wrong-r['true_mixed']))>1e-4
    raw=S.evaluate(w,st,p,[1,2,3]);assert raw['ordered_multiplicity'].sum()==9
    assert raw['pair_times'].shape==(6,2)
    copy_single=w*p[st['past'][0],0,0];copy_single/=copy_single.sum()
    # A literal copy has likelihood one given its original; it cannot add a draw.
    np.testing.assert_array_equal(copy_single*1.,copy_single)
