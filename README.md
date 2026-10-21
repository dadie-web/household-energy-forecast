# Household Electricity Load Forecasting

## Stage 1: Data preparation

Existing cleaning rules, four processed datasets and source information.

This is a staged publication of an existing completed project. Dates describe upload milestones, not newly performed experiments.

Data: Hebrail and Berard, UCI Individual Household Electric Power Consumption, https://doi.org/10.24432/C58K54 (CC BY 4.0).

The study uses 2009 only; training is January-August, validation September-October and testing November-December.

```bash
python -m pip install -r requirements.txt
python data_processing.py --raw /path/to/household_power_consumption.txt
```
