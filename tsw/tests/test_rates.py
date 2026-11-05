"""rates.py may only read dev-train (calibration never sees dev-val or test)."""
import contextlib
import io
import unittest

from tsw.tools import rates


class TestRatesSplitGuard(unittest.TestCase):
    def _refused(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                rates.main(argv)
        return str(cm.exception.code)

    def test_refuses_val_dev_test(self):
        for split in ("dev-val", "test", "dev"):
            self.assertIn("REFUSING", self._refused(["--words", "ju", "--split", split]), split)

    def test_load_refuses_before_reading(self):
        # check_split runs before any file is opened, so a bogus data dir does not matter
        for split in ("dev-val", "test"):
            with self.assertRaises(SystemExit):
                rates.load(split=split, data_dir="/nonexistent")

    def test_default_is_dev_train(self):
        self.assertEqual(rates.CALIBRATION_SPLIT, "dev-train")
        rates.check_split("dev-train")  # no exception


if __name__ == "__main__":
    unittest.main()
