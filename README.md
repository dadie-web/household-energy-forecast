# Household Electricity Load Forecasting

## Project Overview

This project evaluates day-ahead electricity load forecasting for one household using UCI data from 2009. Histogram gradient boosting combines calendar and historical-load features and is compared with daily and weekly persistence. The extension evaluates April, July and October with expanding training windows while retaining the original November-December test.

## Dataset

The UCI Individual Household Electric Power Consumption dataset measures minute-average active power in kW. The study retains January 1-December 31, 2009. The original preprocessing command downloads the source data when needed or reads a local original text file; monthly backtests reuse the included hourly series. Source: https://doi.org/10.24432/C58K54 (CC BY 4.0).

Hourly load is the mean of at least 54 valid minutes. Invalid measurements remain missing; targets are not interpolated. Duplicate timestamps are averaged and no timezone is assigned. The processed series retains all 8760 hourly timestamps, including 74 insufficiently observed hours.

## Methodology

At midnight, the model predicts the next 24 hourly loads. Features are target hour, weekday, weekend, month, previous-day and previous-week same-hour load, and means over the preceding 24 and 168 hours. Required histories must be complete and precede the forecast origin.

The initial model is trained on January-August and evaluated on September-October. The final model is refitted on January-October and tested on November-December. HistGradientBoostingRegressor uses the fixed settings in configs/config.json, including seed 42. Yesterday Naive and Weekly Naive copy the corresponding hour one day and seven days earlier. All methods share the same valid evaluation timestamps.

## Project Structure

```text
main.py                 Processing, training, evaluation and figures
data_processing.py      Retained standalone preprocessing utility
reporting.py            Markdown and PDF report generation
report_word.py          Editable Word report generation from Markdown
monthly_backtest.py     Expanding-window supplementary monthly backtests
monthly_reporting.py    Extended report from saved forecasts
july_analysis.py        July diagnostics and the earliest complete seven-day plot
monthly_backtest_protocol.md  Experiment definitions and eligibility rules
requirements.txt        Runtime dependencies
configs/config.json     Fixed model parameters
data/raw/               Original data
data/processed/         Hourly series and train/validation/test subsets
results/                Forecasts, metrics and data-quality statistics
models/                 Fitted model and feature specification
figures/                Three report figures and the retained November plot
report/                 Report.md, Report.pdf and Report.docx
tests/test_project.py   Automated tests
TESTING.md              Test instructions, coverage and limitations
```

## Installation

Use Python 3.12. From the project directory:

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Dependencies are listed in requirements.txt. Upstream copyright and license terms apply to each library.

## Usage

Run the original experiment (this refits the original model and rewrites its outputs):

```bash
python main.py
```

Use a local original file and export a complete test day's forecasts:

```bash
python main.py --raw /path/to/household_power_consumption.txt --date 2009-11-01
```

Quote paths containing spaces. Outputs are relative to the project directory and replaced on subsequent runs. results/forecast_results.csv contains actual loads and three forecasts; metrics.csv and validation_metrics.csv contain evaluation scores. The date option writes results/forecast_2009-11-01.csv. Four processed files are stored under data/processed/ and the fitted model in models/forecasting_model.joblib.

The extension reads the included hourly data without downloading raw measurements or refitting the original model:

```bash
python monthly_backtest.py --validate-only
python monthly_backtest.py --experiment april
python monthly_backtest.py --experiment july
python monthly_backtest.py --experiment october
python july_analysis.py
python monthly_reporting.py
```

Alternatively, run `python monthly_backtest.py --experiment all --report` for the complete extension. New files are stored in results/monthly_backtest/. Full forecast grids retain eligibility flags and blank unavailable predictions. Training counts, missing-target exclusions and history-only exclusions are recorded in experiment_plan.csv and per-month audit JSONs. The model parameters, features and original baseline artifact hashes are fixed.

Reports are saved as report/Report.md, report/Report.pdf and report/Report.docx. The three report figures show the annual pattern, July 1-7 forecasts and four-period error comparison. The earlier November plot remains in figures/. To rebuild reports and July diagnostics without training:

```bash
python monthly_reporting.py
```

July diagnostics are saved in results/july_analysis/: load distribution, daily MAE comparisons, low/high-load errors and error-size distributions. The July plot uses the earliest complete seven-day window, not the largest-error week. Report generation reads these saved diagnostics and does not modify any CSVs or figures. After rerunning the backtests, use `python july_analysis.py` to refresh the diagnostics before rebuilding the report.

## Results

The original test set comprises all 1464 hours from November 1 to December 31, 2009.

- Yesterday Naive: MAE **0.707789 kW**, RMSE **1.009972 kW**.
- Weekly Naive: MAE **0.629595 kW**, RMSE **0.925265 kW**.
- HistGradientBoosting: MAE **0.512628 kW**, RMSE **0.735804 kW**.

MAE/RMSE reductions are 27.57%/27.15% relative to Yesterday Naive and 18.58%/20.48% relative to Weekly Naive. Peak loads remain harder to predict; the report discusses this behavior and single-household limitations.

Supplementary HGB MAE/RMSE (kW) are 0.488954/0.687920 for April (720 hours), 0.495760/0.610989 for July (744), and 0.447173/0.651662 for October (744). July HGB MAE is higher than both persistence baselines. These retrospective months do not replace the original final test; training size and calendar context change together. Full three-method results are in results/monthly_backtest/monthly_backtest_metrics.csv.

## Testing

Run the full test suite (the reproducibility test reruns the experiment):

```bash
python -m unittest discover -s tests -v
```

Tests cover parsing, aggregation units, chronological subsets, all 24 horizons' feature isolation, output consistency and numerical reproducibility. See [TESTING.md](TESTING.md).

For extension checks without rerunning the original model:

```bash
python -m unittest discover -s tests -p "test_monthly_*.py" -v
python -m unittest discover -s tests -p "test_july_analysis.py" -v
```

The protected baseline hashes in baseline_manifest.json must match before and after extension runs. Existing monthly forecasts, metrics, model files and original figures are also checked against preserved_experiment_manifest.json.

## References

[1] G. Hebrail and A. Berard, "Individual Household Electric Power Consumption," UCI Machine Learning Repository, 2006, doi: 10.24432/C58K54.

[2] Scikit-learn developers, "HistGradientBoostingRegressor," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html.
