import unittest
import numpy as np
from ghostscale.validation.soundingline.v28 import common,g1,g1_reference,g2,g2_reference,g3,g3_reference


class V28Checks(unittest.TestCase):
    def test_grid_is_finite(self):
        self.assertEqual(sum(len(g1.dimensions(s['size'])[-1]) for s in g1.groups()),60)
        self.assertEqual(len(g2.settings()),28)
        self.assertEqual(len(g3.settings()),40)
        self.assertEqual(len(g2.candidates()),210)

    def test_midrank_auc_ties_and_block_deletion(self):
        scores=np.array([[0.,.5,.5,1.],[.2,.3,.8,.9],[.1,.3,.5,.7]])
        labels=np.array([[0,0,1,1],[0,1,0,1],[0,0,1,1]],bool)
        result=common.auc_blocks(scores,labels)
        positive=scores[labels];negative=scores[~labels]
        expected=np.mean((positive[:,None]>negative)+.5*(positive[:,None]==negative))
        self.assertAlmostEqual(result['estimate'],expected)
        leave=[]
        for b in range(3):
            keep=np.arange(3)!=b;pos=scores[keep][labels[keep]];neg=scores[keep][~labels[keep]]
            leave.append(np.mean((pos[:,None]>neg)+.5*(pos[:,None]==neg)))
        self.assertAlmostEqual(result['standard_error'],np.sqrt(2/3*np.sum((np.array(leave)-np.mean(leave))**2)))

    def test_censoring_is_not_dropped(self):
        value=common.median([1.,2.,np.inf,np.inf,np.inf])
        self.assertEqual(value['estimate'],'infinite');self.assertTrue(value['finite_horizon_aliased'])
        np.testing.assert_array_equal(common.separation([-1,3,4],[2,2,.9]),[0,4,np.inf])

    def test_g1_library_beam_and_history_blind_identity(self):
        raw=g1.generate(dict(size='small',rho=.3,overlap='high'),0,'unit',worlds=4)
        a,_=g1.readers(raw,0,.3,True);b,_=g1.readers(raw,16,.3)
        np.testing.assert_allclose(a['B8'],a['J'],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(a['Pp'],b['Pp'])
        self.assertTrue(np.allclose(raw['h'].sum(1),1))
        self.assertTrue(np.allclose(raw['goals'].sum(1),1))
        self.assertTrue(np.all(raw['counts'].sum(2)==64))

    def test_g1_independent_readers(self):
        raw=g1.generate(dict(size='small',rho=.3,overlap='high'),1,'unit',worlds=4)
        a,m=g1.readers(raw,4,.3);b,n=g1_reference.readers(raw,4,.3)
        for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=0)
        for key in m:np.testing.assert_allclose(m[key],n[key],atol=1e-12,rtol=0)

    def test_g1_exact_empirical_goal_ties(self):
        raw=g1.generate(dict(size='small',rho=.4,overlap='low'),1,'unit-ties',worlds=1)
        for target in raw['goals']:
            raw['counts'][0,0,target>target.min()]=[8,4,4]
        a,m=g1.readers(raw,0,.4);b,n=g1_reference.readers(raw,0,.4)
        for label in ('Pe','PeL1','PeL4'):
            np.testing.assert_allclose(a[label],.25,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(a[label].argmax(1),[0])
            np.testing.assert_array_equal(b[label].argmax(1),[0])
        for label in m:np.testing.assert_allclose(m[label],n[label],atol=1e-10,rtol=0)

    def test_g1_distinct_count_patterns_are_not_ties(self):
        raw=g1.generate(dict(size='small',rho=.3,overlap='low'),2,'unit-ties',worlds=1)
        for g,target in enumerate(raw['goals']):
            raw['counts'][0,0,target>target.min()]=[g,2*g,3*g]
        probabilities={label:np.array([[.1,.2,.3,.4]]) for label in ('Pe','PeL1','PeL4')}
        g1.preserve_count_symmetry(raw,0,probabilities)
        for p in probabilities.values():np.testing.assert_array_equal(p,[[.1,.2,.3,.4]])

    def test_g2_independent_changing_histories(self):
        for typ in (0,1,2,3,4,5):
            setting=dict(truth=[typ,12.,1.,4. if typ==2 else (.1 if typ>=3 else 0.)],regime='D',steps=132)
            raw=g2.generate(setting,'unit',worlds=4);a=g2.infer(raw,setting);b=g2_reference.infer(raw,setting)
            for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-9,rtol=0,equal_nan=True)
            stock=raw['stocks'][:,0].copy()
            for t in range(132):
                np.testing.assert_array_equal(stock,raw['stocks'][:,t])
                action=raw['actions'][:,t]
                stock=stock+np.where(action==1,0 if 64<=t<128 else 3,np.where(action==2,-3,0))-raw['disturbances'][:,t]
                stock=np.clip(stock,0,24)
            np.testing.assert_array_equal(stock,raw['final_stock'])

    def test_g3_independent_and_resource_order(self):
        setting=dict(q_in=.5,q_need=.3,truth=5,steps=128,initial='uniform_integer_0_30')
        raw=g3.generate(setting,'unit',worlds=5);a=g3.infer(raw,setting);b=g3_reference.infer(raw,setting)
        for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-9,rtol=0)
        stock=raw['initial_stock'].copy();need=np.zeros((5,2),int)
        for t in range(128):
            stock=np.minimum(30,stock+raw['inflows'][:,t]);need=np.minimum(3,need+raw['increments'][:,t])
            np.testing.assert_array_equal(stock,raw['stocks'][:,t]);np.testing.assert_array_equal(need,raw['needs'][:,t])
            for w,action in enumerate(raw['actions'][:,t]):
                if action:stock[w]-=1;need[w,action-1]-=1


if __name__=='__main__':unittest.main()
