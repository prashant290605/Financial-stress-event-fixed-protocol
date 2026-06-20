import math
import unittest

import numpy as np

from src.paper_protocol import event_metrics, point_metrics, windowed_hybrid


class PaperMetricTests(unittest.TestCase):
    def test_point_metrics(self):
        out = point_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
        self.assertAlmostEqual(out["precision"], 0.5)
        self.assertAlmostEqual(out["recall"], 0.5)
        self.assertAlmostEqual(out["f1"], 0.5)
        self.assertAlmostEqual(out["fpr"], 0.5)

    def test_event_metrics_with_missing_detection(self):
        out = event_metrics(np.array([0, 1, 0, 0, 0, 0]), [(2, 0, 4), (5, 5, 5)])
        self.assertAlmostEqual(out["event_coverage"], 0.5)
        self.assertAlmostEqual(out["event_delay"], -1.0)
        self.assertAlmostEqual(out["event_alignment_error"], 1.0)

    def test_event_metrics_all_missing(self):
        out = event_metrics(np.zeros(5, dtype=int), [(2, 1, 3)])
        self.assertEqual(out["event_coverage"], 0.0)
        self.assertTrue(math.isnan(out["event_delay"]))
        self.assertTrue(math.isnan(out["event_alignment_error"]))

    def test_windowed_hybrid(self):
        vol = np.array([0, 1, 0, 1, 0])
        inst = np.array([0, 0, 1, 0, 0])
        self.assertTrue(np.array_equal(windowed_hybrid(vol, inst, 0), np.array([0, 0, 0, 0, 0])))
        self.assertTrue(np.array_equal(windowed_hybrid(vol, inst, 1), np.array([0, 1, 0, 1, 0])))


if __name__ == "__main__":
    unittest.main()
