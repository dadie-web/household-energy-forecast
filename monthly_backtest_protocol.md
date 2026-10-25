# Expanding Window Backtest Protocol

## Scope and preserved baseline

This extension uses the existing `data/processed/hourly_load_2009.csv`; it never downloads or cleans new source data. The original `main.clean`, `main.features`, `main.score`, model parameters and seed 42 are reused. The four processed files, original November-December predictions and metrics, saved model, two figures and `main.py` are protected by `baseline_manifest.json`. The existing `data_processing.py` utility is also retained and protected.

## Fixed windows

- A / april: fit January 1-March 31; forecast April 1-30.
- B / july: fit January 1-June 30; forecast July 1-31.
- C / october: fit January 1-September 30; forecast October 1-31.

All start boundaries are inclusive; end boundaries in `configs/monthly_backtest.json` are exclusive at midnight. One model is fitted per window, with no tuning on forecast-month targets. The existing November-December evaluation remains the original independent final test. These additional months are supplementary retrospective backtests, including periods formerly used for training/validation; they do not replace that test or supply a new independent test set.

## Midnight information set

Each origin is 00:00. Target hours 00:00-23:00 use only earlier observed loads and known calendar variables. Same-hour lags are 24 and 168 hours earlier. Rolling means use the 24 and 168 hours ending strictly before midnight and are constant within a target day. Observed loads from earlier forecast-month days are available for later origins; target-day loads are never inputs for that day's predictions. Training targets and features are restricted to timestamps before the fit cutoff.

## Eligibility and output schema

The baseline hourly observation rule requires 54 valid minutes; insufficiently observed targets remain missing. Required history must be complete, without imputation. Model fitting requires both valid targets and complete features. Forecasts can be generated when features are complete even if a target is unavailable; evaluation additionally requires a valid actual load. Yesterday Naive, Weekly Naive and HGB are scored on exactly that common mask using the original MAE/RMSE functions (kW). Nonnegative clipping matches the baseline. No records are excluded according to prediction errors.

Each forecast CSV retains every scheduled hour with `timestamp`, `forecast_origin`, `actual`, `yesterday`, `weekly`, `AI`, `history_available`, `target_available`, and `evaluation_eligible`. `AI` is retained as an internal compatibility name; reports use HGB. Missing predictions remain blank. Each metrics CSV has three method rows and includes dates, MAE, RMSE, n, planned hours, valid hours and exclusion counts. Invalid targets and additional valid-target hours lacking history are mutually exclusive exclusion categories. Audit JSON files record actual fit boundaries, parameters, feature names, runtime versions and completion time. Composite metrics contain the same values as the individual files.

## Interpretation

Training size and calendar context change together. Differences cannot be attributed solely to season or to more training data. A month's lower aggregate error does not establish accurate prediction of every peak. All measured results, including any loss to a persistence baseline, must be retained.

## Protocol checks

Use Python 3.12 and install `requirements.txt`. These commands use the included hourly data and do not fit a model or download data:

```bash
python monthly_backtest.py --validate-only
python -m unittest discover -s tests -p "test_monthly_protocol.py" -v
```

The validation command writes `results/monthly_backtest/experiment_plan.csv`, containing date boundaries, sample counts and missing-data exclusions. Monthly forecasts and reports are not part of this protocol release.

## Integrity hashes and Git line endings

The public integrity manifests use SHA256 after converting CRLF to LF in Python, CSV, JSON and Markdown files. All other bytes remain significant; model and image files are hashed without conversion. This accounts for Git checkout line-ending settings without changing values or weakening checks for content changes. The accepted ZIP is kept unchanged locally.
