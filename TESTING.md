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
- test_outputs_and_splits: chronological boundaries, disjoint timestamps, full-year coverage, finite forecasts, common metric sample counts and the two original figures.
- test_reproducibility: repeats the full experiment and compares forecasts with relative and absolute tolerances of 1e-10.

Synthetic fixtures are used only in tests. Production aggregation requires 54 valid minutes. Invalid targets and valid targets with incomplete history are counted separately in results/data_quality_summary.csv.

## Expected results

The full suite contains 19 tests: four original experiment tests, eleven monthly-extension tests and four July-analysis tests. A successful run exits with status zero and no failures or errors.

November-December evaluation uses 1464 common timestamps. Expected MAE/RMSE values rounded to six decimals are 0.707789/1.009972 kW for Yesterday Naive, 0.629595/0.925265 kW for Weekly Naive and 0.512628/0.735804 kW for HistGradientBoosting.

## Scope and limitations

The tests check preprocessing, feature timing, output support and repeatability on the pinned environment. They do not establish performance on other households, years or dependency versions. Download requires access to UCI; the local-file option supports offline execution. Missing targets are not interpolated and incomplete histories reduce training coverage. Report content is checked by the extension tests; actual Word/PDF pagination requires a separate rendering check.

## Checks without retraining

```bash
python -m unittest discover -s tests -v -k test_clean_units_dates -k test_no_future -k test_outputs_and_splits
```

This selection runs three original tests for preprocessing, all 24 forecast horizons and the included outputs without downloading raw data or rerunning training. The monthly-extension and July-analysis commands below also avoid training.

## Monthly extension

After running the three backtests and monthly_reporting.py:

```bash
python -m unittest discover -s tests -p "test_monthly_*.py" -v
```

The extension suite checks exact date boundaries, training-input isolation by changing forecast-month targets, all 24 horizons after changing target-day/future loads, constant pre-midnight rolling means, exclusion accounting, all three month outputs, recomputed metrics on common timestamps, summary agreement, baseline SHA256 integrity, report/CSV agreement and three report figures. Four figure files are included: the annual pattern, July forecast curve, monthly comparison and retained November curve. The baseline test checks the two named original figures without limiting the total file count.

Run the full original suite only in a disposable copy if baseline artifacts must remain byte-for-byte unchanged: its reproducibility test invokes main.run(), refits the baseline and replaces outputs. Use FORECAST_TEST_RAW for an existing local original text file to avoid downloading. PDF/Word pagination and final ZIP completeness are also checked outside the runtime suite.

## July descriptive analysis

`python july_analysis.py` reads the existing July forecasts and generates diagnostics without training. `python monthly_reporting.py` reads the saved diagnostics and rebuilds only Report.md, Report.pdf and Report.docx, without rewriting CSVs or figures. Run `python -m unittest discover -s tests -p test_july_analysis.py -v` to check distribution summaries, low/high-load counts, daily comparisons, error rankings, objective figure dates and protected experiment files. The new diagnostic thresholds describe the already evaluated sample and do not change benchmark scores.

Integrity checks normalize only text line endings (CRLF to LF), because Git checkouts can use either form. Binary model/image hashes remain byte-exact. A protocol test confirms that text value changes and binary byte changes still fail the hash comparison.
