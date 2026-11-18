"""Geometry and false-alarm-budget checks for the hotspot experiments."""
import unittest

import numpy as np
from PIL import Image

from cat_hotspot_study import agreement
from hotspot_study import annotate, hotspots
from report_hotspots import ranking
from sampling_study import shifted_copy


class HotspotTests(unittest.TestCase):
    def test_corner_boxes_do_not_overlap_and_use_exclusive_upper_bounds(self):
        matrix = np.zeros((100, 160), dtype=np.float32)
        matrix[0, 0], matrix[0, 32], matrix[0, 33] = 4, 3, 2
        proposals = hotspots(matrix)
        self.assertEqual(proposals[0]["box"], [0, 0, 17, 17])
        self.assertEqual(proposals[1]["center"], [33, 0])
        mask = np.zeros_like(matrix, dtype=bool)
        mask[0, 17] = True
        annotate(proposals, mask)
        self.assertFalse(proposals[0]["hits"])
        self.assertTrue(proposals[1]["hits"])
        self.assertFalse(proposals[1]["center_hit"])

    def test_spatial_agreement_distinguishes_distant_peaks(self):
        a = np.zeros((120, 180), dtype=np.float32)
        b = a.copy()
        a[20, 20] = b[100, 150] = 1
        self.assertEqual(agreement(a, b)[0]["agreement"], 0.)
        b[24, 25] = .81
        self.assertAlmostEqual(agreement(a, b)[0]["agreement"], .9, places=5)

    def test_restored_offset_matches_original_at_valid_pixels(self):
        pixels = np.arange(63, dtype=np.uint8).reshape(7, 9)
        shifted = np.asarray(shifted_copy(Image.fromarray(pixels).convert("RGB")))
        np.testing.assert_array_equal(shifted[2:, 2:, 0], pixels[:-2, :-2])

    def test_equal_scores_cannot_split_a_tied_negative_group(self):
        rows = [dict(bad=bad, model=score) for bad, score in
                [(False, 1), (False, 1), (True, 1), (True, 2)]]
        self.assertEqual(ranking(rows, "model", 1)["caught_at_budget"], 1)
        self.assertEqual(ranking(rows, "model", 2)["caught_at_budget"], 2)


if __name__ == "__main__":
    unittest.main()
