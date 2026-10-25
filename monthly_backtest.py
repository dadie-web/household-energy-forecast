"""Fixed-parameter expanding-window backtests using the existing hourly series."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from main import FEATURES, HistGradientBoostingRegressor, features, np, pd, score

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/monthly_backtest"
METHODS = ["yesterday", "weekly", "AI"]


def artifact_sha256(path):
    """Hash text with LF line endings; preserve every byte of binary artifacts."""
    content = path.read_bytes()
    if path.suffix in {".py", ".csv", ".json", ".md"}:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def verify_baseline():
    """Reject changes to the original data, model, configuration and results."""
    manifest = json.loads((ROOT / "baseline_manifest.json").read_text())
    for relative, expected in manifest.items():
        if artifact_sha256(ROOT / relative) != expected:
            raise ValueError(f"Protected baseline file changed: {relative}")
    return manifest


def load_inputs():
    """Read the existing regular hourly index and the fixed experiment windows."""
    verify_baseline()
    h = pd.read_csv(
        ROOT / "data/processed/hourly_load_2009.csv", parse_dates=["timestamp"]
    ).set_index("timestamp")
    expected = pd.date_range("2009-01-01", "2009-12-31 23:00", freq="h")
    if not h.index.equals(expected) or h.index.has_duplicates:
        raise ValueError("Expected the original 8760 unique ordered hourly timestamps")
    config = json.loads((ROOT / "configs/monthly_backtest.json").read_text())
    parameters = json.loads((ROOT / config["model_config"]).read_text())
    for e in config["experiments"]:
        if not (
            pd.Timestamp(config["train_start"])
            < pd.Timestamp(e["train_end_exclusive"])
            == pd.Timestamp(e["forecast_start"])
            < pd.Timestamp(e["forecast_end_exclusive"])
        ):
            raise ValueError(f"Invalid chronological boundaries: {e['name']}")
    return h, config, parameters


def window_masks(h, x, config, experiment):
    """Separate target availability, pre-origin history and chronological windows."""
    history = x[FEATURES].notna().all(axis=1)
    target = h.load.notna()
    train = (h.index >= config["train_start"]) & (
        h.index < experiment["train_end_exclusive"]
    )
    forecast = (h.index >= experiment["forecast_start"]) & (
        h.index < experiment["forecast_end_exclusive"]
    )
    return train, forecast, history, target


def audit_counts(h, x, config, experiment):
    """Count mutually exclusive invalid-target and missing-history exclusions."""
    train, forecast, history, target = window_masks(h, x, config, experiment)
    row = dict(experiment)
    row["train_start"] = config["train_start"]
    for name, mask in [("train", train), ("forecast", forecast)]:
        row[name + "_planned_hours"] = int(mask.sum())
        row[name + "_invalid_target_hours"] = int((mask & ~target).sum())
        row[name + "_history_only_excluded_hours"] = int(
            (mask & target & ~history).sum()
        )
        row[name + "_valid_hours"] = int((mask & target & history).sum())
        row[name + "_excluded_hours"] = int((mask & ~(target & history)).sum())
    return row


def make_plan():
    """Export the protocol audit without fitting or downloading anything."""
    h, config, parameters = load_inputs()
    x = features(h)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = pd.DataFrame([audit_counts(h, x, config, e) for e in config["experiments"]])
    plan.to_csv(OUTPUT / "experiment_plan.csv", index=False)
    return h, x, config, parameters, plan


def fit_experiment(h, x, config, parameters, experiment):
    """Fit once before the forecast month and predict each midnight's 24 horizons."""
    train, forecast, history, target = window_masks(h, x, config, experiment)
    fit_mask = train & history & target
    if not fit_mask.any():
        raise ValueError("No valid training samples")
    model = HistGradientBoostingRegressor(**parameters)
    model.fit(x.loc[fit_mask, FEATURES], h.load[fit_mask])
    result = x.loc[forecast, ["yesterday", "weekly"]].copy()
    result["forecast_origin"] = result.index.normalize()
    result["actual"] = h.load[forecast]
    result["history_available"] = history[forecast]
    result["target_available"] = target[forecast]
    result["evaluation_eligible"] = (history & target)[forecast]
    result["AI"] = np.nan
    for origin in result.forecast_origin.unique():
        eligible_history = forecast & history & (h.index.normalize() == origin)
        if eligible_history.any():
            result.loc[eligible_history[forecast], "AI"] = np.maximum(
                0, model.predict(x.loc[eligible_history, FEATURES])
            )
    evaluated = result.loc[result.evaluation_eligible]
    if evaluated.empty or not np.isfinite(evaluated[["actual", *METHODS]]).all().all():
        raise ValueError("No finite common evaluation support")
    metrics = score(evaluated)
    counts = audit_counts(h, x, config, experiment)
    for key, value in counts.items():
        metrics[key] = value
    counts["fit_first_timestamp"] = str(h.index[fit_mask][0])
    counts["fit_last_timestamp"] = str(h.index[fit_mask][-1])
    counts["fit_sample_count"] = int(fit_mask.sum())
    return result, metrics, counts


def summarize(config):
    """Combine completed months and retain the original final-test scores verbatim."""
    completed = [
        pd.read_csv(OUTPUT / f"{e['name']}_metrics.csv")
        for e in config["experiments"]
        if (OUTPUT / f"{e['name']}_metrics.csv").exists()
    ]
    if not completed:
        return
    summary = pd.concat(completed, ignore_index=True)
    summary.to_csv(OUTPUT / "monthly_backtest_metrics.csv", index=False)
    comparison = summary[["name", "method", "MAE_kW", "RMSE_kW", "n"]].rename(
        columns={"name": "period"}
    )
    baseline = pd.read_csv(ROOT / "results/metrics.csv")
    baseline.insert(0, "period", "november_december")
    pd.concat([comparison, baseline], ignore_index=True).to_csv(
        OUTPUT / "period_comparison_metrics.csv", index=False
    )


def run(experiment="all", validate_only=False, report=False):
    """Run selected predefined months without altering baseline artifacts."""
    h, x, config, parameters, plan = make_plan()
    if validate_only:
        print(plan.to_string(index=False))
        return
    selected = [e for e in config["experiments"] if experiment in ("all", e["name"])]
    if not selected:
        raise ValueError(f"Unknown experiment: {experiment}")
    for e in selected:
        predictions, metrics, counts = fit_experiment(h, x, config, parameters, e)
        predictions.to_csv(
            OUTPUT / f"{e['name']}_forecasts.csv", index_label="timestamp"
        )
        metrics.to_csv(OUTPUT / f"{e['name']}_metrics.csv", index=False)
        audit = {
            **counts,
            "parameters": parameters,
            "features": FEATURES,
            "run_completed_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scikit-learn"]
            },
        }
        (OUTPUT / f"{e['name']}_audit.json").write_text(json.dumps(audit, indent=2))
        print(
            e["name"],
            metrics[["method", "MAE_kW", "RMSE_kW", "n"]].to_string(index=False),
        )
    summarize(config)
    verify_baseline()
    if report:
        from monthly_reporting import build_report

        build_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=["all", "april", "july", "october"], default="all"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    run(args.experiment, args.validate_only, args.report)
