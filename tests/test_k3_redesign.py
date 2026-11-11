"""Targeted correctness, causal-channel, admission and finite-monitor checks."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ghostscale.validation.soundingline.k3_redesign import filter as f, reference, experiment
from ghostscale.validation.soundingline.v27.common import read, write
from runners import run_k3_redesign as run, watch_k3_redesign as watch


class AdaptedControls(unittest.TestCase):
    def setUp(self):
        self.s = dict(needs=1, width=0., beta=0., lapse=.02, consumption=3.)

    def test_initial_predictive_likelihood_is_exact(self):
        observed = dict(inference_key=['unit-initial'], records=[dict(action=1, size=0., shock=-1)])
        actual, info = f.likelihood(observed, self.s, (11,), 'P+', 32)
        exact, _ = reference.exact(observed, self.s, (11,), 1)
        np.testing.assert_allclose(actual, exact, atol=1e-14, rtol=0)
        self.assertEqual(info['exact_initial_states'], 21)

    def test_causal_order_and_independent_weights(self):
        states = np.array([[0., 11., 1., 19.], [20., 11., 10., 0.]])
        losses, probability = f.branch_law(1, 3.)
        for law in range(5):
            args = (states, 0, 5., 1, 0., -1, law, 0., 0., .02, losses, probability)
            a, w = f.candidates(*args)
            b, v = reference.enumerate_candidates(*args)
            np.testing.assert_allclose(a, b, atol=1e-14, rtol=0)
            np.testing.assert_allclose(w, v, atol=1e-14, rtol=0)
        # Restore and loss clip together: 0 + 5 - 15 is zero, not five.
        self.assertEqual(a[2, 0], 0.)

    def test_duplicate_branches_sum_their_mass(self):
        states = np.array([[0., 11., 1., 19.]])
        losses, probability = f.branch_law(1, 3.)
        a, w = f.candidates(states, 1, 0., 1, 0., 0, 3, 0., 0., .02, losses, probability)
        self.assertEqual(np.count_nonzero(w), 1)
        expected, cp = reference.observe_scalar.py_func(states[0], 1, 0., 0, 3, 0., 0., .02)
        self.assertAlmostEqual(float(w.sum()), cp, places=14)
        np.testing.assert_array_equal(a[0], expected)

    def test_zero_support_is_failure_not_floor(self):
        observed = dict(inference_key=['zero'], records=[dict(action=1, size=1., shock=0)])
        with self.assertRaises(ArithmeticError):
            f.likelihood(observed, self.s, (11,), 'G', 32)

    def test_hidden_reader_rejects_truth_fields(self):
        observed = dict(inference_key=['truth'], records=[dict(action=1, size=0., shock=0, stock=[0.])])
        with self.assertRaises(ValueError):
            f.likelihood(observed, self.s, (11,), 'G', 32)

    def test_future_append_cannot_change_prefix(self):
        prefix = [dict(action=1, size=0., shock=0)]*24
        a, _ = f.likelihood(dict(inference_key=['causal'], records=prefix), self.s, (11,), 'P-', 64)
        b, _ = f.likelihood(dict(inference_key=['causal'], records=prefix+[dict(action=0, size=5., shock=-1)]),
                            self.s, (11,), 'P-', 64)
        np.testing.assert_array_equal(a, b[:len(a)])

    def test_exact_cap_is_unavailable_not_passed(self):
        obs = dict(inference_key=['cap'], records=[dict(action=1, size=0., shock=-1)]*3)
        with self.assertRaises(MemoryError):
            reference.exact(obs, self.s, (11,), 1, state_cap=2)

    def test_twentieth_deficit_and_limits(self):
        state = np.array([0., 11., .10001, 19.])
        moved, _ = f.observe_state(state, 1, 0., -1, 2, 0., 0., .02)
        self.assertEqual(moved[2], .1)
        moved, _ = f.observe_state(np.array([0., 11., 9.99, 19.]), 1, 0., -1, 4, 0., 0., .02)
        self.assertEqual(moved[2], 10.)
        moved, _ = f.observe_state(state, 1, 0., -1, 3, 0., 0., .02)
        self.assertEqual(moved[1], 10.9)

    def test_evaluator_clock_does_not_depend_on_discarded_streak(self):
        raw = dict(point=[11], records=[dict(stock=[0.], edge=[11.], streak=[0])]*22, first_long_deficit=None)
        self.assertEqual(experiment.first_deficit(raw, dict(needs=1, width=0, dynamic='P+')), 19)


class QueueControls(unittest.TestCase):
    def test_separate_budget_no_production(self):
        a = run.authority('x'*64, '2026-11-11T02:00:00+00:00')
        self.assertEqual(sum(a['pools_seconds'].values()), 48*3600)
        self.assertEqual(a['pools_seconds']['D'], 4*3600)
        self.assertFalse(a['production_authorized'])

    def test_reservations_cannot_spend_protected_pool(self):
        plan = dict(phase='A', cpu_cap_seconds=500.)
        t = datetime(2026, 11, 11, tzinfo=timezone.utc)
        totals = dict(A=run.POOLS['A']-499, B=0, C=0, D=0)
        self.assertFalse(run.eligible(plan, totals, t))
        totals['A'] -= 1
        self.assertTrue(run.eligible(plan, totals, t))
        self.assertFalse(run.eligible(plan, totals, datetime(2026, 11, 16, 5, tzinfo=timezone.utc)))

    def test_six_hour_anchor_and_terminal_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            c = Path(directory)
            anchor = datetime(2026, 11, 11, 2, tzinfo=timezone.utc)
            write(c/'ACCEPTANCE.json', run.authority('x'*64, anchor.isoformat()))
            watch.scheduled(c, anchor+timedelta(hours=5, minutes=59))
            self.assertFalse((c/'events').exists())
            watch.scheduled(c, anchor+timedelta(hours=6))
            self.assertEqual(len(list((c/'events').glob('*.json'))), 1)
            watch.scheduled(c, anchor+timedelta(hours=12))
            self.assertEqual(len(list((c/'events').glob('*.json'))), 2)
            write(c/'QUEUE_COMPLETE.json', dict(outcome='finished'))
            watch.scheduled(c, anchor+timedelta(hours=18))
            self.assertEqual(len(list((c/'events').glob('*.json'))), 2)

    def test_all_pairs_catches_nonreference_pair(self):
        rows = []
        for rep, value in enumerate((0., -.2, .2)):
            rows.append(dict(index=0, lane='old', first_long_deficit=None, particles=1024, replicate=rep,
                paths=dict(posterior=[[.2]*5]*3, margins={k: [value]*3 for k in ('S', 'P-', 'G')})))
        result = experiment.all_pairs(rows)
        self.assertEqual(len(result['pairs']), 3)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['maximum_margin_error'], .4)


if __name__ == '__main__':
    unittest.main()
