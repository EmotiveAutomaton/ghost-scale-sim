import unittest
from runners.diagnose_k3_redesign import within_budget, compare_arrays, JOBS


class DiagnosticContract(unittest.TestCase):
    def test_exactly_four_predeclared_jobs(self):
        self.assertEqual(len(JOBS),4)
        self.assertEqual(sum(row[3] for row in JOBS),8400)
        self.assertEqual(sorted((row[1],row[2]) for row in JOBS),[(1024,False),(1024,True),(16384,False),(16384,True)])

    def test_four_hour_subcap_and_prior_pool_charge(self):
        prior=dict(A=9500.,B=0.,C=0.,D=300.)
        current=dict(A=8400.,B=0.,C=0.,D=6000.)
        self.assertTrue(within_budget(prior,current))
        current['D']+=.1
        self.assertFalse(within_budget(prior,current))
        current['D']=0.;prior['A']=40000.
        self.assertFalse(within_budget(prior,current))

    def test_original_reserve_cannot_be_double_spent(self):
        prior=dict(A=0.,B=0.,C=0.,D=14399.)
        current=dict(A=0.,B=0.,C=0.,D=2.)
        self.assertFalse(within_budget(prior,current))

    def test_negative_or_nonfinite_accounting_refused(self):
        with self.assertRaises(ValueError):
            within_budget(dict(A=-1.,B=0.,C=0.,D=0.),dict(A=0.,B=0.,C=0.,D=0.))

    def test_replay_is_exact(self):
        self.assertTrue(compare_arrays([[-1.,-2.]],[[-1.,-2.]],False)['passed'])
        with self.assertRaises(ArithmeticError):
            compare_arrays([[-1.,-2.]],[[-1.,-2.+1e-12]],False)

    def test_reference_tolerance_does_not_waive_replay(self):
        self.assertTrue(compare_arrays([[-1.,-2.]],[[-1.,-2.+1e-10]],True)['passed'])
        self.assertFalse(compare_arrays([[-1.,-2.]],[[-1.,-2.+1e-8]],True)['passed'])

    def test_shape_and_nonfinite_mismatches_refused(self):
        with self.assertRaises(ValueError):compare_arrays([[-1.,-2.]],[[-1.]],True)
        with self.assertRaises(ValueError):compare_arrays([[-1.,-2.]],[[-1.,float('nan')]],True)


if __name__=='__main__':unittest.main()
