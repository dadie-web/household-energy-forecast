# Household Electricity Load Forecasting

## Stage 3: Results and figures

Prediction CSVs, validation/test metrics and the two research figures.

This is a staged publication of an existing completed project. Dates describe upload milestones, not newly performed experiments.

Data: Hebrail and Berard, UCI Individual Household Electric Power Consumption, https://doi.org/10.24432/C58K54 (CC BY 4.0).

The study uses 2009 only; training is January-August, validation September-October and testing November-December.

```bash
python -m pip install -r requirements.txt
python main.py
```

Test MAE/RMSE (kW): Yesterday Naive 0.707789/1.009972; Weekly Naive 0.629595/0.925265; HGB 0.512628/0.735804. All methods use the same 1464 November-December hours.
