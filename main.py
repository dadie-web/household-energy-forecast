import os

os.environ["OMP_NUM_THREADS"] = "2"
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
FEATURES = [
    "hour",
    "weekday",
    "weekend",
    "month",
    "yesterday",
    "weekly",
    "mean24",
    "mean168",
]


def clean(raw):
    """Parse minute records and return hourly means with quality statistics."""
    d = pd.read_csv(
        raw, sep=";", usecols=["Date", "Time", "Global_active_power"], dtype=str
    )
    t = pd.to_datetime(
        d.Date + " " + d.Time, format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    v = pd.to_numeric(d.Global_active_power, errors="coerce")
    q = {
        "raw_rows": len(d),
        "unparseable_timestamp": int(t.isna().sum()),
        "missing_or_invalid_power_all_years": int(v.isna().sum()),
        "negative_power_all_years": int((v < 0).sum()),
    }
    d = pd.DataFrame({"timestamp": t, "load": v.mask(v < 0)}).dropna(
        subset=["timestamp"]
    )
    d = d[(d.timestamp >= "2009-01-01") & (d.timestamp < "2010-01-01")]
    q["rows_2009"] = len(d)
    q["duplicates_2009"] = int(d.timestamp.duplicated().sum())
    q["invalid_power_2009"] = int(d.load.isna().sum())
    d = d.groupby("timestamp").load.mean().sort_index()
    h = (
        d.resample("h")
        .agg(["mean", "count"])
        .reindex(pd.date_range("2009-01-01", "2009-12-31 23:00", freq="h"))
    )
    h["load"] = h["mean"].where(h["count"] >= 54)
    h = h[["load", "count"]]
    h.index.name = "timestamp"
    q["invalid_hours"] = int(h.load.isna().sum())
    q["valid_hours"] = int(h.load.notna().sum())
    return h, q


def features(h):
    """Build calendar and history features available before each midnight origin."""
    s = h.load
    x = pd.DataFrame(index=h.index)
    x["hour"] = x.index.hour
    x["weekday"] = x.index.dayofweek
    x["weekend"] = (x.weekday >= 5).astype(int)
    x["month"] = x.index.month
    x["yesterday"] = s.shift(24)
    x["weekly"] = s.shift(168)
    for n in (24, 168):
        m = s.shift(1).rolling(n, min_periods=n).mean()
        x["mean" + str(n)] = m.reindex(x.index.normalize()).to_numpy()
    return x


def score(r):
    """Compute MAE and RMSE on shared actual and forecast timestamps."""
    return pd.DataFrame(
        [
            {
                "method": c,
                "MAE_kW": np.abs(r[c] - r.actual).mean(),
                "RMSE_kW": np.sqrt(((r[c] - r.actual) ** 2).mean()),
                "n": len(r),
            }
            for c in ["yesterday", "weekly", "AI"]
        ]
    )


def load_config(path):
    """Read model settings, saving the existing defaults only if absent."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    config = {
        "random_state": 42,
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "learning_rate": 0.05,
        "l2_regularization": 1.0,
        "early_stopping": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def run(raw=None, date=None):
    """Run the fixed chronological experiment and export its artifacts."""
    for p in [
        "data/raw",
        "data/processed",
        "results",
        "figures",
        "models",
        "report",
        "configs",
    ]:
        (ROOT / p).mkdir(parents=True, exist_ok=True)
    raw = Path(raw) if raw else ROOT / "data/raw/household_power_consumption.txt"
    if (
        raw is not None
        and not raw.exists()
        and str(raw) != str(ROOT / "data/raw/household_power_consumption.txt")
    ):
        raise FileNotFoundError(f"Local raw file does not exist: {raw}")
    if not raw.exists():
        z = ROOT / "data/raw/uci.zip"
        urllib.request.urlretrieve(URL, z)
        with zipfile.ZipFile(z) as f:
            raw.write_bytes(f.read("household_power_consumption.txt"))
        z.unlink()
    h, q = clean(raw)
    x = features(h)
    valid = x.notna().all(axis=1) & h.load.notna()
    h.to_csv(ROOT / "data/processed/hourly_load_2009.csv")
    masks = {
        "train": h.index < "2009-09-01",
        "validation": (h.index >= "2009-09-01") & (h.index < "2009-11-01"),
        "test": h.index >= "2009-11-01",
    }
    for name, mask in masks.items():
        h.loc[mask].to_csv(ROOT / f"data/processed/{name}.csv")
        q[name + "_excluded"] = int((mask & ~valid).sum())
        q[name + "_invalid_target"] = int((mask & h.load.isna()).sum())
        q[name + "_history_only_excluded"] = int(
            (mask & h.load.notna() & ~x.notna().all(axis=1)).sum()
        )
    config = load_config(ROOT / "configs/config.json")
    model = HistGradientBoostingRegressor(**config)
    tr = valid & masks["train"]
    va = valid & masks["validation"]
    te = valid & masks["test"]
    model.fit(x.loc[tr, FEATURES], h.load[tr])

    def predict(mask):
        """Predict the selected eligible timestamps using the current fitted model."""
        r = x.loc[mask, ["yesterday", "weekly"]].copy()
        r["actual"] = h.load[mask]
        r["AI"] = np.maximum(0, model.predict(x.loc[mask, FEATURES]))
        return r

    score(predict(va)).to_csv(ROOT / "results/validation_metrics.csv", index=False)
    model.fit(x.loc[tr | va, FEATURES], h.load[tr | va])
    joblib.dump(
        {"model": model, "features": FEATURES, "config": config},
        ROOT / "models/forecasting_model.joblib",
    )
    r = predict(te)
    r.to_csv(ROOT / "results/forecast_results.csv")
    metrics = score(r)
    metrics.to_csv(ROOT / "results/metrics.csv", index=False)
    q["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    pd.Series(q, name="value").rename_axis("statistic").to_csv(
        ROOT / "results/data_quality_summary.csv"
    )
    if date:
        day = x.index.normalize() == pd.Timestamp(date)
        if not (day.sum() == 24 and valid[day].all() and masks["test"][day].all()):
            raise ValueError("Date must be a complete valid test day")
        predict(day).to_csv(ROOT / f"results/forecast_{date}.csv")
    daily = h.load.resample("D").mean().where(h.load.resample("D").count() == 24)
    fig, ax = plt.subplots(figsize=(10, 3))
    daily.plot(ax=ax)
    ax.set(
        title="Household Electricity Consumption Pattern in 2009",
        ylabel="Daily mean power (kW)",
        xlabel="Date",
    )
    fig.tight_layout()
    fig.savefig(ROOT / "figures/figure1_consumption_pattern.png", dpi=200)
    plt.close(fig)
    start = None
    for d in pd.date_range("2009-11-01", "2009-11-24"):
        idx = pd.date_range(d, periods=168, freq="h")
        if idx.isin(r.index).all():
            start = d
            break
    if start is None:
        raise ValueError("No complete November seven-day window")
    fig, ax = plt.subplots(figsize=(10, 3))
    r.loc[
        start : start + pd.Timedelta(hours=167), ["actual", "AI", "yesterday", "weekly"]
    ].rename(
        columns={
            "actual": "Actual",
            "AI": "HGB",
            "yesterday": "Yesterday Naive",
            "weekly": "Weekly Naive",
        }
    ).plot(
        ax=ax, lw=1
    )
    ax.set(
        title="Actual vs Predicted Electricity Load",
        ylabel="Hourly mean power (kW)",
        xlabel="Date",
    )
    fig.tight_layout()
    fig.savefig(ROOT / "figures/figure2_forecast_comparison.png", dpi=200)
    plt.close(fig)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw")
    p.add_argument("--date")
    args = p.parse_args()
    run(args.raw, args.date)
