"""Checks for threshold semantics and the model comparison interface.

Run: python -m unittest analysis.test_compare
"""
import unittest
from unittest.mock import patch

import numpy as np

from analysis.compare import Comparison, crossing_record, first_exceedance, solve_pair
from backend.contract import Params, Result


class ThresholdChecks(unittest.TestCase):
    def test_strict_threshold_and_time_bracket(self):
        self.assertEqual(first_exceedance([0, 1, 2], [0, .1, .101]), (2., 1.))

    def test_missing_crossing_is_not_end_time(self):
        self.assertEqual(first_exceedance([0, 1, 2], [0, .1, .05]), (None, None))

    def test_invalid_time_series_rejected(self):
        for t, d in [([0, 0], [0, .2]), ([0, 1], [0, np.nan])]:
            with self.assertRaises(ValueError):
                first_exceedance(t, d)

    def test_first_onset_location_and_distance(self):
        delta = np.array([[0, 0, 0], [.02, .15, 0], [.4, .3, .1]])
        c = Comparison(np.array([0., 10., 20.]), np.array([0., 1., 2.]),
                       np.array([5., 3., 0.]), delta, delta.max(axis=1), 2., 0.)
        r = crossing_record(c, .5, coastal_censor=True)
        self.assertEqual((r['t10_s'], r['onset_x_m'], r['onset_depth_m'],
                          r['distance_to_shore_m']), (1., 10., 3., 10.))
        c.coastal_response = 1.
        r = crossing_record(c, .5, coastal_censor=True)
        self.assertEqual(r['status'], 'coastal_boundary_already_active')
        self.assertIsNone(r['onset_x_m'])


class ComparisonChecks(unittest.TestCase):
    def test_real_flat_pair_matches_grid_and_initial_surface(self):
        c = solve_pair(Params(a=.05, nx=80, t_end=.1, n_frames=3))
        self.assertEqual(c.difference.shape, (3, 80))
        self.assertLess(c.d[0], 1e-10)
        np.testing.assert_array_equal(c.d, c.difference.max(axis=1))

    def test_incompatible_result_grid_rejected(self):
        class FakeModel:
            dim = 1

            def solve(self, p):
                return Result([0., 1.], [0., .1], [[0., 0.], [0., 0.]], [5., 5.])

        with patch.dict('analysis.compare.MODELS', {'fake': FakeModel()}):
            with self.assertRaisesRegex(ValueError, 'different coordinates'):
                solve_pair(Params(nx=80, t_end=.1, n_frames=2), model='fake')

    def test_zero_amplitude_rejected(self):
        with self.assertRaises(ValueError):
            solve_pair(Params(a=0))


if __name__ == '__main__':
    unittest.main()
