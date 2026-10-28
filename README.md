# Household Electricity Load Forecasting

This project studies midnight day-ahead forecasting for one household using 2009 UCI electricity data. HistGradientBoostingRegressor uses calendar and pre-midnight historical-load features, with Yesterday Naive and Weekly Naive as persistence baselines.

## Data and methods

Data: G. Hebrail and A. Berard, UCI Individual Household Electric Power Consumption, https://doi.org/10.24432/C58K54 (CC BY 4.0). The included hourly series requires at least 54 valid minute readings; missing targets are not interpolated. Training is January-August, validation September-October and final testing November-December.

The supplementary expanding windows train January-March for April, January-June for July and January-September for October. Model parameters and feature definitions are shared across months. All methods use identical eligible evaluation hours. Training length and month change together, so their separate causal effects cannot be identified. See [monthly_backtest_protocol.md](monthly_backtest_protocol.md).

## Installation and checks

Use Python 3.12:

```bash
python -m pip install -r requirements.txt
python monthly_backtest.py --validate-only
python -m unittest discover -s tests -p "test_monthly_protocol.py" -v
python -m unittest discover -s tests -p "test_monthly_outputs.py" -v
python -m unittest discover -s tests -p "test_july_analysis.py" -v
```

Protocol validation uses saved hourly data without fitting models or downloading raw measurements. `results/monthly_backtest/experiment_plan.csv` records sample counts and exclusions. To rerun a monthly experiment, use `python monthly_backtest.py --experiment april` (or `july`/`october`); this trains the selected monthly model and overwrites its monthly outputs. `python main.py` reruns the original experiment. The existing `python data_processing.py --raw /path/to/file` utility remains available for preprocessing. See [TESTING.md](TESTING.md).

## Results

Original final-test MAE/RMSE: Yesterday Naive 0.707789/1.009972; Weekly Naive 0.629595/0.925265; HGB 0.512628/0.735804 kW. All methods use the same 1464 November-December hours.

April HGB MAE/RMSE: 0.488954/0.687920 kW on 720 hours.

July HGB MAE/RMSE: 0.495760/0.610989 kW on 744 hours. MAE is higher than both persistence baselines, although RMSE is slightly lower. Mean actual/predicted loads are 0.618/0.935 kW; 491 of 503 low-load hours are overestimated. HGB wins daily MAE against Yesterday Naive on 10 days and loses on 21.

October HGB MAE/RMSE: 0.447173/0.651662 kW on 744 hours. Results for April, July, October and the earlier final test are compared in results/monthly_backtest/period_comparison_metrics.csv and figures/figure3_monthly_performance.png.

Dependencies are listed in requirements.txt; upstream copyright and license terms apply.
