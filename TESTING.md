# Testing

## Environment and commands

Use Python 3.12 and the dependencies in requirements.txt. From the project directory:

```bash
python -m unittest discover -s tests -v
```

Tests require the processed data and result files included in the package. The complete reproducibility test also needs the original data (downloaded by main.py if absent). The reproducibility test reruns the complete experiment and replaces those outputs. To use an external original file for that test, set FORECAST_TEST_RAW. PowerShell example:

```powershell
$env:FORECAST_TEST_RAW = 'C:\data\household_power_consumption.txt'
python -m unittest discover -s tests -v
```

## Coverage

- test_clean_units_dates: explicit date parsing, 2009 filtering, invalid timestamp counting and hourly mean units using a small synthetic fixture.
- test_no_future: changing target-day and subsequent loads leaves every feature for all 24 target hours unchanged; individual horizons are also checked after target-day perturbation.
- test_outputs_and_splits: chronological boundaries, disjoint timestamps, full-year coverage, finite forecasts, common metric sample counts and two figures.
- test_reproducibility: repeats the full experiment and compares forecasts with relative and absolute tolerances of 1e-10.

Synthetic fixtures are used only in tests. Production aggregation requires 54 valid minutes. Invalid targets and valid targets with incomplete history are counted separately in results/data_quality_summary.csv.

## Expected results

The command should exit with status zero and report four passing original tests when selected with `-p test_project.py`. A nonzero exit status indicates a test failure or error.

November-December evaluation uses 1464 common timestamps. Expected MAE/RMSE values rounded to six decimals are 0.707789/1.009972 kW for Yesterday Naive, 0.629595/0.925265 kW for Weekly Naive and 0.512628/0.735804 kW for HistGradientBoosting.

## Scope and limitations

The tests check preprocessing, feature timing, output support and repeatability on the pinned environment. They do not establish performance on other households, years or dependency versions. Download requires access to UCI; the local-file option supports offline execution. Missing targets are not interpolated and incomplete histories reduce training coverage. Report consistency and Word pagination are checked separately from the four automated tests.

## Checks without retraining

```bash
python -m unittest discover -s tests -v -k test_clean_units_dates -k test_no_future -k test_outputs_and_splits
```

This selection checks preprocessing, all 24 forecast horizons and the included outputs without downloading raw data or rerunning training.


## Published extension checks

```bash
python monthly_backtest.py --validate-only
python -m unittest discover -s tests -p "test_monthly_protocol.py" -v
python -m unittest discover -s tests -p "test_monthly_outputs.py" -v
python -m unittest discover -s tests -p "test_july_analysis.py" -v
```

Protocol tests check exact chronological windows, target-month training isolation, all 24 midnight horizons, constant rolling means and baseline hashes. Output tests compare available forecasts, common evaluation masks, audit boundaries and stored metrics with the original scoring functions. July tests check saved load/bias summaries, daily MAE rankings and the first complete seven-day plot window. The cross-month suite also requires all three monthly outputs and checks the combined tables. These extension commands do not fit a real model or download data. The original reproducibility test still refits and overwrites original outputs; run it only in a disposable copy.

Integrity checks normalize only text line endings (CRLF to LF), because Git checkouts can use either form. Binary model/image hashes remain byte-exact. A protocol test confirms that text value changes and binary byte changes still fail the hash comparison.
