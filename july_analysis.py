"""Describe saved July forecasts and plot the first complete seven-day window."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from main import score, features

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/july_analysis"
METHODS = ["yesterday", "weekly", "AI"]


def analyze_july():
    """Recompute descriptive statistics without fitting or changing forecasts."""
    forecasts = pd.read_csv(
        ROOT / "results/monthly_backtest/july_forecasts.csv",
        parse_dates=["timestamp", "forecast_origin"],
    ).set_index("timestamp")
    r = forecasts.loc[forecasts.evaluation_eligible].copy()
    if len(r) != 744 or not np.isfinite(r[["actual", *METHODS]]).all().all():
        raise ValueError("Expected the complete existing July evaluation")
    stored = pd.read_csv(ROOT / "results/monthly_backtest/july_metrics.csv")
    measured = score(r)
    np.testing.assert_allclose(
        stored[["MAE_kW", "RMSE_kW", "n"]],
        measured[["MAE_kW", "RMSE_kW", "n"]],
        rtol=1e-12,
        atol=1e-12,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    quantiles = r.actual.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    distribution = pd.DataFrame(
        {"quantile": quantiles.index, "actual_kW": quantiles.values}
    )
    distribution.to_csv(OUTPUT / "actual_load_quantiles.csv", index=False)
    daily = (
        r[METHODS]
        .sub(r.actual, axis=0)
        .abs()
        .resample("D")
        .mean()
        .rename(columns={m: f"{m}_MAE_kW" for m in METHODS})
    )
    daily["hours"] = r.actual.resample("D").count()
    daily["HGB_better_than_yesterday"] = (
        daily.AI_MAE_kW < daily.yesterday_MAE_kW - 1e-12
    )
    daily["HGB_worse_than_yesterday"] = daily.AI_MAE_kW > daily.yesterday_MAE_kW + 1e-12
    daily.to_csv(OUTPUT / "daily_mae_comparison.csv", index_label="date")
    low = r.actual < 0.6
    high_threshold = float(quantiles.loc[0.9])
    high = r.actual >= high_threshold
    group_rows = []
    error_rows = []
    for method in METHODS:
        error = r[method] - r.actual
        absolute = error.abs()
        error_rows.append(
            {
                "method": method,
                "MAE_kW": absolute.mean(),
                "RMSE_kW": np.sqrt((error**2).mean()),
                "median_absolute_error_kW": absolute.median(),
                "p90_absolute_error_kW": absolute.quantile(0.9),
                "max_absolute_error_kW": absolute.max(),
                "hours_absolute_error_gt_1kW": int((absolute > 1).sum()),
                "sum_squared_error_kW2": (error**2).sum(),
                "mean_error_kW": error.mean(),
            }
        )
        for name, mask in [
            ("actual_below_0.6kW", low),
            ("actual_at_or_above_p90", high),
        ]:
            group_rows.append(
                {
                    "group": name,
                    "method": method,
                    "n": int(mask.sum()),
                    "actual_mean_kW": r.actual[mask].mean(),
                    "predicted_mean_kW": r[method][mask].mean(),
                    "mean_error_kW": error[mask].mean(),
                    "MAE_kW": absolute[mask].mean(),
                    "overestimated_hours": int((error[mask] > 0).sum()),
                    "underestimated_hours": int((error[mask] < 0).sum()),
                }
            )
    pd.DataFrame(error_rows).to_csv(OUTPUT / "error_distribution.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUTPUT / "load_group_errors.csv", index=False)
    bands = pd.cut(
        r.actual,
        [-np.inf, 0.3, 0.6, 1, 2, np.inf],
        right=False,
        labels=[
            "below_0.3",
            "0.3_to_below_0.6",
            "0.6_to_below_1",
            "1_to_below_2",
            "2_and_above",
        ],
    )
    bands.value_counts(sort=False).rename_axis("actual_load_band_kW").rename(
        "hours"
    ).to_csv(OUTPUT / "actual_load_bands.csv")
    peak = r.actual.idxmax()
    low_error_daily = (r.AI - r.actual).where(low).resample("D").mean().dropna()
    summary = {
        "n": len(r),
        "actual_mean_kW": r.actual.mean(),
        "actual_median_kW": r.actual.median(),
        "actual_p25_kW": quantiles.loc[0.25],
        "actual_p75_kW": quantiles.loc[0.75],
        "HGB_mean_kW": r.AI.mean(),
        "HGB_mean_error_kW": (r.AI - r.actual).mean(),
        "low_threshold_kW": 0.6,
        "low_hours": int(low.sum()),
        "low_fraction": float(low.mean()),
        "low_HGB_bias_kW": (r.AI - r.actual)[low].mean(),
        "low_HGB_overestimated_hours": int(((r.AI > r.actual) & low).sum()),
        "days_with_low_load": len(low_error_daily),
        "days_with_positive_low_load_HGB_bias": int((low_error_daily > 0).sum()),
        "high_threshold_definition": "actual load at or above the full-July 90th percentile; descriptive only",
        "high_threshold_kW": high_threshold,
        "high_hours": int(high.sum()),
        "high_HGB_bias_kW": (r.AI - r.actual)[high].mean(),
        "high_HGB_underestimated_hours": int(((r.AI < r.actual) & high).sum()),
        "days_HGB_better_MAE_than_yesterday": int(
            daily.HGB_better_than_yesterday.sum()
        ),
        "days_HGB_worse_MAE_than_yesterday": int(daily.HGB_worse_than_yesterday.sum()),
        "days_MAE_tied": int(
            (~(daily.HGB_better_than_yesterday | daily.HGB_worse_than_yesterday)).sum()
        ),
        "monthly_peak_time": str(peak),
        "monthly_peak_actual_kW": r.loc[peak, "actual"],
        "monthly_peak_HGB_kW": r.loc[peak, "AI"],
    }
    hourly = pd.read_csv(
        ROOT / "data/processed/hourly_load_2009.csv", parse_dates=["timestamp"]
    ).set_index("timestamp")
    history = features(hourly).notna().all(axis=1)
    training = (hourly.index < "2009-07-01") & history & hourly.load.notna()
    summary["july_fit_samples"] = int(training.sum())
    summary["july_fit_target_mean_kW"] = hourly.load[training].mean()
    summary["july_fit_target_below_0.6_fraction"] = float(
        (hourly.load[training] < 0.6).mean()
    )
    start = next(
        day
        for day in pd.date_range("2009-07-01", "2009-07-25")
        if pd.date_range(day, periods=168, freq="h").isin(r.index).all()
    )
    window = r.loc[start : start + pd.Timedelta(hours=167)]
    summary.update(
        {
            "figure_window_start": str(window.index.min()),
            "figure_window_end": str(window.index.max()),
            "figure_selection_rule": "earliest continuous seven-day July window with 168 common valid hours",
        }
    )
    (OUTPUT / "july_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for column, label, color, linestyle in [
        ("actual", "Actual Load", "#202020", "-"),
        ("AI", "HistGradientBoosting", "#167b8b", "-"),
        ("yesterday", "Yesterday Naive", "#c68639", "--"),
        ("weekly", "Weekly Naive", "#a87891", ":"),
    ]:
        ax.plot(
            window.index,
            window[column],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
        )
    ax.set(
        title="Actual vs Predicted Household Load in July 2009",
        xlabel="Date",
        ylabel="Hourly mean load (kW)",
    )
    ax.legend(ncol=2, fontsize=10)
    ax.set_xlim(window.index[0], window.index[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/figure2_july_forecast.png", dpi=220)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze_july(), indent=2))
