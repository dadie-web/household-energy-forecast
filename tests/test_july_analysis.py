"""Independent checks of descriptive July findings and preserved experiments."""
import hashlib
import json
import unittest
import numpy as np
import pandas as pd
from monthly_backtest import ROOT

class JulyAnalysisTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = pd.read_csv(ROOT / 'results/monthly_backtest/july_forecasts.csv', parse_dates=['timestamp']).set_index('timestamp')
        cls.r = cls.r.loc[cls.r.evaluation_eligible]
        cls.summary = json.loads((ROOT / 'results/july_analysis/july_summary.json').read_text())

    def test_load_distribution_and_bias(self):
        r, s = (self.r, self.summary)
        self.assertEqual(len(r), 744)
        self.assertAlmostEqual(s['actual_mean_kW'], r.actual.mean())
        self.assertAlmostEqual(s['actual_median_kW'], r.actual.median())
        self.assertAlmostEqual(s['HGB_mean_kW'], r.AI.mean())
        self.assertAlmostEqual(s['HGB_mean_error_kW'], (r.AI - r.actual).mean())
        self.assertEqual(int((r.actual < 0.6).sum()), 503)
        self.assertEqual(s['low_hours'], 503)
        self.assertEqual(s['low_HGB_overestimated_hours'], int(((r.actual < 0.6) & (r.AI > r.actual)).sum()))
        high = r.actual >= r.actual.quantile(0.9)
        self.assertEqual(s['high_HGB_underestimated_hours'], int((high & (r.AI < r.actual)).sum()))
        self.assertAlmostEqual(s['high_HGB_bias_kW'], (r.AI - r.actual)[high].mean())

    def test_daily_wins_and_error_rankings(self):
        r, s = (self.r, self.summary)
        ae = r[['yesterday', 'weekly', 'AI']].sub(r.actual, axis=0).abs()
        daily = ae.resample('D').mean()
        self.assertEqual(int((daily.AI < daily.yesterday - 1e-12).sum()), 10)
        self.assertEqual(int((daily.AI > daily.yesterday + 1e-12).sum()), 21)
        self.assertEqual(s['days_HGB_better_MAE_than_yesterday'], 10)
        self.assertEqual(s['days_HGB_worse_MAE_than_yesterday'], 21)
        errors = pd.read_csv(ROOT / 'results/july_analysis/error_distribution.csv').set_index('method')
        for method in ae:
            self.assertAlmostEqual(errors.loc[method, 'sum_squared_error_kW2'], (ae[method] ** 2).sum())
            self.assertEqual(errors.loc[method, 'hours_absolute_error_gt_1kW'], int((ae[method] > 1).sum()))
        self.assertGreater(ae.AI.mean(), ae.yesterday.mean())
        self.assertLess((ae.AI ** 2).mean(), (ae.yesterday ** 2).mean())

    def test_plot_is_first_complete_week(self):
        s = self.summary
        self.assertEqual(s['figure_window_start'], '2009-07-01 00:00:00')
        self.assertEqual(s['figure_window_end'], '2009-07-07 23:00:00')
        self.assertEqual(len(self.r.loc[s['figure_window_start']:s['figure_window_end']]), 168)
if __name__ == '__main__':
    unittest.main()
