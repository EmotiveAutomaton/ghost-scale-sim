import copy
from pathlib import Path
import unittest
import numpy as np
from ghostscale.validation.soundingline.v23 import profile_validation as s
from ghostscale.validation.soundingline.v23 import profile_successor as successor


class ProfileSuccessorContractTests(unittest.TestCase):
    def test_successor_is_fixed_before_sampling(self):
        cfg=successor.configuration()
        self.assertEqual((cfg['worlds'],cfg['rounds'],cfg['future_prefix']),(40,16,48))
        self.assertEqual(cfg['probability_tolerance'],1e-9)
        self.assertTrue(cfg['no_adaptive_stopping'])
        self.assertEqual(cfg['structural_worlds'],list(range(40)))


class ProfileValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sibling=Path(__file__).resolve().parents[5]/'SoundingLine/sounding-line'
        # Explicit local dependency tests are skipped outside this workstation;
        # frozen scientific runs bind archived source instead.
        if not sibling.exists():raise unittest.SkipTest('pinned sibling source unavailable')
        cls.f,cls.proof=s.exact_functions(dict(profile_source=str(sibling/'runners/next_round/profile_features.py'),
            task_source=str(sibling/'runners/stage14_midweek/baselines.py')))

    def test_exact_representation_and_function_binding(self):
        self.assertEqual(set(self.proof),{'profile_vector','choice_features','predict_all','features'})
        np.testing.assert_array_equal(self.f['profile_vector'](['ONE two one']),self.f['profile_vector'](['two ONE one']))
        self.assertEqual(self.f['choice_features'](np.ones(9),np.ones(32)).shape,(4,164))

    def test_generated_inputs_and_truth_separation(self):
        for condition in s.CONDITIONS:
            records,labels,truth=s.construct(0,condition,self.f)
            self.assertEqual(len(records),32)
            self.assertTrue(all(a!=b for a,b in enumerate(truth['donors'])))
            for row in records:
                self.assertEqual(set(row),{'key','writer','prompt','open_at','resolved_at','profile_cutoff','features'})
                self.assertLess(row['profile_cutoff'],row['open_at'])
                self.assertTrue(all(np.asarray(x).shape==(4,164) for x in row['features'].values()))
            if condition=='constant':
                self.assertTrue(all(r['features']['own']==r['features']['donor'] for r in records))

    def test_real_engine_strict_cutoff_and_future_outcome(self):
        records,labels,_=s.construct(123,'planted',self.f,pilot=True)
        records=records[:5];labels=labels[:5]
        original=self.f['predict_all'](records,labels)
        changed=copy.deepcopy(labels);changed[-1]['action']=(changed[-1]['action']+1)%4
        self.assertEqual(original,self.f['predict_all'](records,changed))
        s.validate_predictions(records,original)
        bad=copy.deepcopy(original);bad[-1]['arms']['own']['training_keys'].append(records[-1]['key'])
        with self.assertRaises(ValueError):s.validate_predictions(records,bad)

    def test_incomplete_evidence_cannot_pass(self):
        result=s.summarize([],[])
        self.assertEqual(result['status'],'INCONCLUSIVE')
        self.assertTrue(result['constructed_software_fixture'])

    def test_noisy_or_false_gain_control_prevents_pass(self):
        rows=[dict(world=w,condition=c,gain_baseline=.02 if c=='planted' else 0.,gain_donor=.02 if c=='planted' else 0.) for w in range(20) for c in s.CONDITIONS]
        controls=[dict(world=w,future_max_difference=0.,option_max_difference=0.,constant_max_difference=0.) for w in range(20)]
        self.assertEqual(s.summarize(rows,controls)['status'],'PASS')
        controls[-1]['future_max_difference']=.1
        self.assertEqual(s.summarize(rows,controls)['status'],'FAIL')
        controls[-1]['future_max_difference']=0.
        for row in rows:
            if row['condition']=='uninformative':row['gain_baseline']=.02
        self.assertEqual(s.summarize(rows,controls)['status'],'FAIL')

    def test_successor_constructs_long_histories(self):
        cfg=successor.configuration()
        records,labels,truth=s.construct(0,'constant',self.f,cfg=cfg)
        self.assertEqual(len(records),128)
        self.assertEqual(len(labels),128)
        self.assertEqual(len(truth['targets']),128)
        self.assertTrue(all(r['features']['own']==r['features']['donor'] for r in records))


if __name__=='__main__':unittest.main()
