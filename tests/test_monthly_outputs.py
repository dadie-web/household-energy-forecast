"""Verify completed backtests against the unchanged input series and original scores."""
import json
import unittest
import numpy as np
import pandas as pd
from main import features, score
from monthly_backtest import OUTPUT, ROOT, load_inputs, window_masks, verify_baseline

class OutputTests(unittest.TestCase):

    def test_completed_months_common_support_and_metrics(self):
        h, config, parameters = load_inputs()
        x = features(h)
        completed = 0
        for e in config['experiments']:
            path = OUTPUT / f"{e['name']}_forecasts.csv"
            if not path.exists():
                continue
            completed += 1
            r = pd.read_csv(path, parse_dates=['timestamp', 'forecast_origin']).set_index('timestamp')
            _, forecast, history, target = window_masks(h, x, config, e)
            self.assertTrue(r.index.equals(h.index[forecast]))
            np.testing.assert_array_equal(r.evaluation_eligible, (history & target)[forecast])
            np.testing.assert_array_equal(r.forecast_origin, r.index.normalize())
            np.testing.assert_allclose(r.actual, h.load[forecast], equal_nan=True)
            for name in ['yesterday', 'weekly']:
                np.testing.assert_allclose(r[name], x.loc[forecast, name], equal_nan=True)
            eligible = r.loc[r.evaluation_eligible]
            self.assertTrue(np.isfinite(eligible[['actual', 'yesterday', 'weekly', 'AI']]).all().all())
            measured = pd.read_csv(OUTPUT / f"{e['name']}_metrics.csv")
            expected = score(eligible)
            pd.testing.assert_frame_equal(measured[expected.columns], expected, check_dtype=False, atol=1e-12, rtol=1e-12)
            self.assertTrue((measured.n == len(eligible)).all())
            audit = json.loads((OUTPUT / f"{e['name']}_audit.json").read_text())
            self.assertEqual(audit['parameters'], parameters)
            self.assertLess(pd.Timestamp(audit['fit_last_timestamp']), pd.Timestamp(e['forecast_start']))
        self.assertGreater(completed, 0, 'Run at least one monthly experiment first')

    def test_original_scores_and_artifact_hashes(self):
        verify_baseline()
        baseline = pd.read_csv(ROOT / 'results/metrics.csv')
        np.testing.assert_allclose(baseline.MAE_kW.round(6), [0.707789, 0.629595, 0.512628])
        np.testing.assert_allclose(baseline.RMSE_kW.round(6), [1.009972, 0.925265, 0.735804])
        self.assertTrue((baseline.n == 1464).all())
if __name__ == '__main__':
    unittest.main()
