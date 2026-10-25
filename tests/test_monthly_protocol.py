"""Protocol and information-set tests that do not require fitted new models."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from main import FEATURES, features
from monthly_backtest import (
    audit_counts,
    fit_experiment,
    load_inputs,
    verify_baseline,
    window_masks,
)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h, cls.config, cls.params = load_inputs()
        cls.x = features(cls.h)

    def test_exact_window_boundaries(self):
        expected = [
            ("2009-04-01", "2009-05-01", 2160, 720),
            ("2009-07-01", "2009-08-01", 4344, 744),
            ("2009-10-01", "2009-11-01", 6552, 744),
        ]
        for e, (start, end, train_n, forecast_n) in zip(
            self.config["experiments"], expected
        ):
            tr, fc, _, _ = window_masks(self.h, self.x, self.config, e)
            self.assertEqual(e["train_end_exclusive"], start)
            self.assertEqual(e["forecast_start"], start)
            self.assertEqual(e["forecast_end_exclusive"], end)
            self.assertEqual(int(tr.sum()), train_n)
            self.assertEqual(int(fc.sum()), forecast_n)
            self.assertFalse((tr & fc).any())

    def test_all_horizons_use_pre_midnight_loads(self):
        for e in self.config["experiments"]:
            day = e["forecast_start"]
            changed = self.h.copy()
            changed.loc[day:, "load"] = 98765.0
            altered = features(changed)
            original_day = self.x.loc[day]
            self.assertEqual(len(original_day), 24)
            np.testing.assert_allclose(original_day, altered.loc[day], equal_nan=True)
            for column in ["mean24", "mean168"]:
                self.assertLessEqual(original_day[column].nunique(), 1)

    def test_training_inputs_ignore_forecast_month_targets(self):
        class RecordingModel:
            fits = []

            def __init__(self, **kwargs):
                pass

            def fit(self, x, y):
                self.fits.append((x.copy(), y.copy()))

            def predict(self, x):
                return np.ones(len(x))

        for e in self.config["experiments"]:
            RecordingModel.fits = []
            with patch(
                "monthly_backtest.HistGradientBoostingRegressor", RecordingModel
            ):
                fit_experiment(self.h, self.x, self.config, self.params, e)
                changed = self.h.copy()
                changed.loc[e["forecast_start"] :, "load"] = 87654.0
                fit_experiment(changed, features(changed), self.config, self.params, e)
            (x1, y1), (x2, y2) = RecordingModel.fits
            self.assertLess(x1.index.max(), pd.Timestamp(e["forecast_start"]))
            pd.testing.assert_frame_equal(x1, x2)
            pd.testing.assert_series_equal(y1, y2)
            self.assertEqual(list(x1.columns), FEATURES)

    def test_exclusion_counts_and_baseline_integrity(self):
        verify_baseline()
        for e in self.config["experiments"]:
            row = audit_counts(self.h, self.x, self.config, e)
            for split in ["train", "forecast"]:
                self.assertEqual(
                    row[split + "_excluded_hours"],
                    row[split + "_invalid_target_hours"]
                    + row[split + "_history_only_excluded_hours"],
                )
                self.assertEqual(
                    row[split + "_planned_hours"],
                    row[split + "_valid_hours"] + row[split + "_excluded_hours"],
                )


if __name__ == "__main__":
    unittest.main()
