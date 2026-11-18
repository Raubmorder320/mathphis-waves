"""Research diagnostics: python -m unittest analysis.test_third_wave."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.third_wave import align_periodic, field, historical_floods, incoming_metrics, pair_metrics
from backend.contract import Params


class PhaseChecks(unittest.TestCase):
    def test_three_real_models_share_the_research_setup(self):
        p = Params(a=.005, depth=5., wavelength=100., L=400., nx=64, t_end=.1, n_frames=3)
        initial = None
        for model in ('saint_venant', 'kdv', 'boussinesq'):
            t, eta = field(p, model, gamma=0.)
            np.testing.assert_array_equal(t, [0., .05, .1])
            if initial is not None:
                np.testing.assert_allclose(eta[0], initial, atol=1e-12)
            initial = eta[0]

    def test_translation_is_removed_without_changing_shape(self):
        x = np.arange(256) * 2 * np.pi / 256
        reference = np.sin(x) + .2 * np.sin(2 * x)
        target = np.roll(reference, 17)
        moved, shift = align_periodic(reference, target)
        self.assertAlmostEqual(shift, -17, places=9)
        np.testing.assert_allclose(moved, reference, atol=1e-12)
        metrics, _ = pair_metrics(reference, target, 1.)
        self.assertGreater(metrics['raw_linf_over_a'], .1)
        self.assertLess(metrics['aligned_rms_over_a'], 1e-12)

    def test_alignment_preserves_amplitude_difference(self):
        x = np.arange(256) * 2 * np.pi / 256
        reference = np.sin(x)
        metrics, moved = pair_metrics(reference, 2 * np.roll(reference, 10), 1.)
        self.assertAlmostEqual(metrics['aligned_rms_over_a'], np.sqrt(.5), places=10)
        self.assertAlmostEqual(np.max(moved), 2.)


class IncomingChecks(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(11, dtype=float)
        self.h = np.r_[np.full(9, 5.), .1, 0.]
        self.t = np.arange(4, dtype=float)
        e = np.zeros((4, 11))
        e[1, 4:7] = [.2, 1., .2]
        e[2, 5:8] = [.2, 1., .2]
        # Boussinesq's last active node is 8; the other walls are at node 9.
        self.fields = {'linear': e.copy(), 'saint_venant': e.copy(), 'boussinesq': e.copy()}
        self.fields['boussinesq'][3, 8] = .1
        self.fields['linear'][3, 9] = .1
        self.fields['saint_venant'][3, 9] = .1

    def test_uses_earliest_actual_model_wall_and_excludes_response(self):
        peaks, ratios, record = incoming_metrics(self.x, self.h, self.t, self.fields, 1., 1.)
        self.assertEqual(record['common_cutoff_s'], 3.)
        self.assertEqual(record['last_included_frame_s'], 2.)
        self.assertIsNone(record['front_doubling'])
        self.assertTrue(np.isnan(peaks['linear'][9]))
        self.assertTrue(np.isnan(ratios['linear'][:4]).all())
        self.fields['boussinesq'][2, 8] = .1
        _, _, record = incoming_metrics(self.x, self.h, self.t, self.fields, 1., 1.)
        self.assertEqual(record['common_cutoff_s'], 2.)
        self.assertEqual(record['last_included_frame_s'], 1.)

    def test_stronger_front_and_no_response_are_reported(self):
        self.fields['saint_venant'][:3] *= 3
        _, _, record = incoming_metrics(self.x, self.h, self.t, self.fields, 1., 1.)
        self.assertGreater(record['front_doubling']['ratio'], 2.)
        for e in self.fields.values():
            e[3] = 0
        _, _, record = incoming_metrics(self.x, self.h, self.t, self.fields, 1., 1.)
        self.assertTrue(all(v is None for v in record['coastal_response_s'].values()))
        self.assertEqual(record['last_included_frame_s'], 3.)

    def test_grid_refinements_can_use_the_same_physical_window(self):
        _, _, record = incoming_metrics(self.x, self.h, self.t, self.fields, 1., 1., stop_s=1.5)
        self.assertEqual(record['last_included_frame_s'], 1.)
        self.assertEqual(record['common_cutoff_s'], 1.5)
        self.assertEqual(record['coastal_response_s']['boussinesq'], 3.)


class SourceChecks(unittest.TestCase):
    def read_catalogue(self, table):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'catalogue.md'
            path.write_text('## 4. Floods\n' + table + '\n## 5. End\n[s]: https://example.org/source\n', encoding='utf-8')
            return historical_floods(path)

    def test_multiple_events_preserve_datum_missing_wind_and_sources(self):
        rows = self.read_catalogue('| 1824 | подъём 4,21 м | не найдено | [source][s] |\n'
                                   '| 1924 | 3,80 м выше ординара | до 34 м/с | [source][s] |')
        self.assertEqual([r['reported_level_m'] for r in rows], [4.21, 3.8])
        self.assertEqual(rows[0]['wind_as_published'], 'не найдено')
        self.assertEqual(rows[1]['datum_as_published'], '3,80 м выше ординара')
        self.assertEqual(rows[0]['sources'], ['https://example.org/source'])

    def test_missing_level_is_not_invented(self):
        with self.assertRaisesRegex(ValueError, 'Missing sourced level'):
            self.read_catalogue('| 1824 | не найдено | не найдено | [source][s] |')

    def test_uncited_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing source link'):
            self.read_catalogue('| 1824 | 4,21 м | не найдено | нет |')

    def test_project_catalogue_has_ten_sourced_events(self):
        rows = historical_floods(Path(__file__).with_name('gulf.md'))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r['sources'] for r in rows))


if __name__ == '__main__':
    unittest.main()
