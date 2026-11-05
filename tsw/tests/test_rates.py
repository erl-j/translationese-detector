"""rates.py may only read trn (calibration and KWIC never see val or test)."""
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

    def test_refuses_val_and_test(self):
        for split in ("val", "test", "dev-val"):
            msg = self._refused(["--words", "ju", "--kwic", "3", "--split", split])
            self.assertIn("REFUSING", msg, split)

    def test_load_refuses_before_reading(self):
        # check_split runs before any file is opened, so a bogus data dir does not matter
        for split in ("val", "test"):
            with self.assertRaises(SystemExit):
                rates.load(split=split, data_dir="/nonexistent")

    def test_default_is_trn(self):
        self.assertEqual(rates.CALIBRATION_SPLIT, "trn")
        rates.check_split("trn")  # no exception
        rates.check_split("dev")  # old name of trn


if __name__ == "__main__":
    unittest.main()
