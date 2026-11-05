"""Tiny stdlib unit tests: python3 -m unittest discover -s evals/tests -v"""
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import devsplit  # noqa: E402
import metrics  # noqa: E402
import run  # noqa: E402
import topicmask  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_auc_hand_checked(self):
        # positives score 0.8, 0.4, 0.35; negatives 0.1, 0.4, 0.7.
        # pairs (pos > neg): 0.8 beats all 3 -> 3; 0.4 beats 0.1, ties 0.4 -> 1.5;
        # 0.35 beats 0.1 -> 1. Total 5.5 / 9.
        labels = [1, 1, 1, 0, 0, 0]
        scores = [0.8, 0.4, 0.35, 0.1, 0.4, 0.7]
        self.assertAlmostEqual(metrics.roc_auc(labels, scores), 5.5 / 9)

    def test_auc_extremes(self):
        self.assertEqual(metrics.roc_auc([0, 0, 1, 1], [1, 2, 3, 4]), 1.0)
        self.assertEqual(metrics.roc_auc([0, 0, 1, 1], [4, 3, 2, 1]), 0.0)
        self.assertEqual(metrics.roc_auc([0, 1, 0, 1], [7, 7, 7, 7]), 0.5)
        self.assertTrue(metrics.roc_auc([1, 1], [1, 2]) != metrics.roc_auc([1, 1], [1, 2]))  # nan

    def test_accuracy_f1_confusion(self):
        labels = [1, 1, 0, 0, 0]
        scores = [1.0, -1.0, 1.0, -1.0, 0.0]  # 0.0 counts as "original"
        preds = metrics.predict(scores)
        self.assertEqual(preds, [1, 0, 1, 0, 0])
        c = metrics.confusion(labels, preds)
        self.assertEqual(c, {"tp": 1, "fn": 1, "fp": 1, "tn": 2})
        self.assertAlmostEqual(metrics.accuracy(labels, preds), 3 / 5)
        # F1(trans) = 2*1/(2+1+1) = 0.5 ; F1(orig) = 2*2/(4+1+1) = 2/3
        self.assertAlmostEqual(metrics.macro_f1(labels, preds), (0.5 + 2 / 3) / 2)

    def test_constant_classifier_macro_f1(self):
        labels = [0, 1, 0, 1]
        m = metrics.all_metrics(labels, [1.0] * 4)
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 0) / 2)
        self.assertAlmostEqual(m["roc_auc"], 0.5)

    def test_rule_diagnostics(self):
        labels = [1, 1, 0, 0]
        expl = [{"a": 1.0}, {"a": 1.0, "b": -0.5}, {}, {"b": -0.5}]
        d = metrics.rule_diagnostics(labels, expl)
        self.assertEqual(d["a"]["fire_trans"], 1.0)
        self.assertEqual(d["a"]["fire_orig"], 0.0)
        self.assertGreater(d["a"]["log_odds"], 0)
        self.assertEqual(d["a"]["sign_acc"], 1.0)
        self.assertEqual(d["b"]["fire_trans"], 0.5)
        self.assertAlmostEqual(d["b"]["mean_orig"], -0.25)
        self.assertAlmostEqual(d["b"]["log_odds"], 0.0)
        self.assertEqual(d["b"]["sign_acc"], 0.5)


class TestGuardAndMask(unittest.TestCase):
    def _main(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                run.main(argv)
        return cm.exception.code

    def test_test_split_refused_without_flag(self):
        code = self._main(["--clf", "baselines:Constant", "--split", "test", "--gran", "100"])
        self.assertIn("REFUSING", str(code))

    def test_show_errors_refused_on_test(self):
        code = self._main(["--clf", "baselines:Constant", "--split", "test", "--gran", "100",
                           "--i-mean-it", "--show-errors", "3"])
        self.assertIn("REFUSING", str(code))

    def test_guard_function(self):
        run.check_test_guard("dev", False)  # fine
        run.check_test_guard("test", True)  # fine
        with self.assertRaises(SystemExit):
            run.check_test_guard("test", False)

    def test_mask(self):
        m = topicmask.mask("Sverige och svenskarna, Storbritanniens regering och Labour i London.")
        for w in ("Sverige", "svenskarna", "Storbritanniens", "Labour", "London"):
            self.assertNotIn(w, m)
        self.assertIn("och", m)
        self.assertEqual(topicmask.mask("en moderat ökning av rom"), "en moderat ökning av rom")

    def test_load_clf_with_kwargs(self):
        c = run.load_clf("baselines:Random(seed=7)")
        self.assertEqual(c.score("x"), run.load_clf("baselines:Random(seed=7)").score("x"))
        self.assertNotEqual(c.score("x"), run.load_clf("baselines:Random(seed=8)").score("x"))


class TestDevSplit(unittest.TestCase):
    RECS = [{"id": "a", "date": "1997-01-14", "label": 0}, {"id": "b", "date": "2002-02-28", "label": 1},
            {"id": "c", "date": "2002-03-01", "label": 0}, {"id": "d", "date": "2004-05-05", "label": 1},
            {"id": "e", "date": "2002-02-28", "label": 0}, {"id": "f", "date": "2002-03-01", "label": 1}]

    def test_boundary_and_partition(self):
        self.assertEqual(devsplit.VAL_START, "2002-03-01")
        tr = devsplit.select(self.RECS, "dev-train")
        va = devsplit.select(self.RECS, "dev-val")
        self.assertEqual([r["id"] for r in tr], ["a", "b", "e"])
        self.assertEqual([r["id"] for r in va], ["c", "d", "f"])
        with self.assertRaises(ValueError):
            devsplit.select(self.RECS, "test")

    def test_deterministic_order_independent(self):
        once = [devsplit.subset_of(r) for r in self.RECS]
        rev = [devsplit.subset_of(r) for r in reversed(self.RECS)][::-1]
        self.assertEqual(once, rev)
        self.assertEqual(devsplit.select(self.RECS, "dev-val"), devsplit.select(list(self.RECS), "dev-val"))

    def test_sessions_not_split(self):
        side = {}
        for r in self.RECS * 3:
            side.setdefault(r["date"], set()).add(devsplit.subset_of(r))
        self.assertTrue(all(len(v) == 1 for v in side.values()))

    def test_sessions_not_split_real_data(self):
        try:
            ddir = run.data_dir()
        except SystemExit:
            self.skipTest("no data/processed")
        path = os.path.join(ddir, "dev.500.bal.jsonl")
        if not os.path.exists(path):
            self.skipTest("no dev.500.bal.jsonl")
        recs = run.load_records(path)
        tr, va = devsplit.select(recs, "dev-train"), devsplit.select(recs, "dev-val")
        self.assertEqual(len(tr) + len(va), len(recs))
        dtr, dva = {r["date"] for r in tr}, {r["date"] for r in va}
        self.assertFalse(dtr & dva)                    # no session date on both sides
        self.assertLess(max(dtr), devsplit.VAL_START)
        self.assertGreaterEqual(min(dva), devsplit.VAL_START)
        self.assertEqual([r["id"] for r in run.load_records(path, subset="dev-val")], [r["id"] for r in va])

    def test_harness_paths(self):
        self.assertEqual(run.file_split("dev-val"), "dev")
        self.assertEqual(run.file_split("test"), "test")
        self.assertTrue(run.data_path("/d", "dev-train", "500", True).endswith("/dev.500.bal.jsonl"))
        run.check_test_guard("dev-val", False)  # fine


if __name__ == "__main__":
    unittest.main()
