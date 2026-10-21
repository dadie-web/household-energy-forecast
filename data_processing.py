"""Minute-to-hour preprocessing for the UCI 2009 household study."""

import argparse
from pathlib import Path

import pandas as pd

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

def main():
    """Clean an original UCI text file and export the chronological subsets."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, type=Path)
    args = parser.parse_args()
    hourly, quality = clean(args.raw)
    output = Path(__file__).resolve().parent
    (output / 'data/processed').mkdir(parents=True, exist_ok=True)
    (output / 'results').mkdir(exist_ok=True)
    hourly.to_csv(output / 'data/processed/hourly_load_2009.csv')
    bounds = {
        'train': hourly.index < '2009-09-01',
        'validation': (hourly.index >= '2009-09-01') & (hourly.index < '2009-11-01'),
        'test': hourly.index >= '2009-11-01',
    }
    for name, mask in bounds.items():
        hourly.loc[mask].to_csv(output / f'data/processed/{name}.csv')
    pd.Series(quality, name='value').rename_axis('statistic').to_csv(output / 'results/data_quality_summary.csv')

if __name__ == '__main__':
    main()
