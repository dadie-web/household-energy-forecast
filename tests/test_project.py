import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from main import clean, features, ROOT, run


class Checks(unittest.TestCase):
    """Check preprocessing, day-ahead timing and reproducible outputs."""

    def test_clean_units_dates(self):
        """Retain only 2009 and aggregate power as a mean rather than a sum."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date;Time;Global_active_power\n")
            for m in range(60):
                f.write(f"01/01/2009;00:{m:02}:00;2\n")
            f.write("01/01/2008;00:00:00;9\ninvalid;x;-1\n")
            name = f.name
        h, q = clean(name)
        self.assertEqual(len(h), 8760)
        self.assertEqual(h.load.iloc[0], 2)
        self.assertEqual(q["unparseable_timestamp"], 1)
        Path(name).unlink()

    def test_no_future(self):
        """Keep all 24 horizons independent of target-day observations."""
        h = pd.DataFrame(
            {"load": np.arange(8760, dtype=float)},
            index=pd.date_range("2009-01-01", periods=8760, freq="h"),
        )
        x = features(h)
        h.loc["2009-11-01":, "load"] = 99999
        y = features(h)
        self.assertEqual(len(x.loc["2009-11-01"]), 24)
        np.testing.assert_allclose(x.loc["2009-11-01"], y.loc["2009-11-01"])
        for hour in range(24):
            changed = h.copy()
            changed.loc["2009-11-01", "load"] = np.arange(24) * 12345
            np.testing.assert_allclose(
                x.loc["2009-11-01"].iloc[hour],
                features(changed).loc["2009-11-01"].iloc[hour],
            )

    def test_outputs_and_splits(self):
        """Check split boundaries, common evaluation support and finite outputs."""
        frames = [
            pd.read_csv(ROOT / f"data/processed/{s}.csv")
            for s in ["train", "validation", "test"]
        ]
        self.assertEqual(sum(map(len, frames)), 8760)
        expected_bounds = [
            ("2009-01-01 00:00:00", "2009-08-31 23:00:00"),
            ("2009-09-01 00:00:00", "2009-10-31 23:00:00"),
            ("2009-11-01 00:00:00", "2009-12-31 23:00:00"),
        ]
        for frame, (first, last) in zip(frames, expected_bounds):
            self.assertEqual(frame.timestamp.iloc[0], first)
            self.assertEqual(frame.timestamp.iloc[-1], last)
        self.assertEqual(len(set().union(*(set(f.timestamp) for f in frames))), 8760)
        r = pd.read_csv(ROOT / "results/forecast_results.csv")
        self.assertTrue(np.isfinite(r.select_dtypes("number")).all().all())
        m = pd.read_csv(ROOT / "results/metrics.csv")
        self.assertTrue((m.n == len(r)).all())
        self.assertTrue((ROOT / "figures/figure1_consumption_pattern.png").exists())
        self.assertTrue((ROOT / "figures/figure2_forecast_comparison.png").exists())

    def test_reproducibility(self):
        """Repeat the experiment and compare forecast values within tolerance."""
        before = pd.read_csv(ROOT / "results/forecast_results.csv")
        run(os.environ.get("FORECAST_TEST_RAW"))
        after = pd.read_csv(ROOT / "results/forecast_results.csv")
        np.testing.assert_allclose(
            before.select_dtypes("number"),
            after.select_dtypes("number"),
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
