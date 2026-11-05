"""Fast unit tests for the hybrid side track (no data files needed except where skipped)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evals")]

import devsplit  # noqa: E402
from hybrid import features as F  # noqa: E402


class TestSplit(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(devsplit.subset_of({"date": "2002-02-28"}), "dev-train")
        self.assertEqual(devsplit.subset_of({"date": "2002-03-01"}), "dev-val")


class TestFeatures(unittest.TestCase):
    def counts(self, t):
        return dict(zip(F.COUNT_NAMES, F.raw_counts(t)))

    def test_basic(self):
        c = self.counts("Herr talman! Det vet vi ju. Dessutom, vi skulle vilja säga att detta är viktigt.")
        self.assertEqual(c["p_ju"], 1)
        self.assertEqual(c["si_det_obj"], 1)
        self.assertEqual(c["si_adv_comma"], 1)
        self.assertEqual(c["skulle_vilja"], 1)
        self.assertEqual(c["dem_T"], 1)
        self.assertEqual(c["n_sents"], 2)  # vocative stripped

    def test_ju_comparative_excluded(self):
        self.assertEqual(self.counts("Ju större desto bättre.")["p_ju"], 0)

    def test_calque_prefilter(self):
        self.assertEqual(self.counts("Mötet tog plats i Bryssel på basis av förslaget.")["calque"], 2)

    def test_transform_shape(self):
        import numpy as np
        X = F.transform(np.asarray([F.raw_counts("Vi tycker att det här är bra.")], float), np)
        self.assertEqual(X.shape, (1, len(F.feature_names())))
        for n in F.feature_names():
            F.feature_meta(n)


class TestRouting(unittest.TestCase):
    def test_gran_for(self):
        from hybrid.clf import gran_for
        self.assertEqual(gran_for("a " * 20), "sent")
        self.assertEqual(gran_for("a " * 98), "100")
        self.assertEqual(gran_for("a " * 497), "500")


if __name__ == "__main__":
    unittest.main()
