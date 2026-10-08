"""Load, clean and cache the Kaggle Hanoi AQI dataset (hourly, 2022-01-13 .. 2025-06-30)."""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

DATASET = "phungdinhdat/aqi-in-hanoi-2022-2025"
PROCESSED = os.path.join("data", "processed", "hanoi_hourly.parquet")

RENAME = {
    "Local Time": "time", "AQI": "aqi", "CO": "co", "NO2": "no2", "O3": "o3", "PM10": "pm10",
    "PM25": "pm25", "SO2": "so2", "Clouds": "clouds", "Precipitation": "precip", "Pressure": "pressure",
    "Relative Humidity": "rh", "Temperature": "temp", "UV Index": "uv", "Wind Speed": "wind",
}
POLLUTANTS = ["pm10", "o3", "no2", "so2", "co"]
METEO = ["temp", "rh", "wind", "pressure", "precip", "clouds", "uv"]
TARGET = "pm25"
MAX_GAP_INTERP = 3  # hours; longer gaps stay NaN and samples touching them are dropped


def raw_files() -> list[str]:
    import kagglehub
    path = kagglehub.dataset_download(DATASET)
    return sorted(glob.glob(os.path.join(path, "*.csv")))


def load_raw() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in raw_files()], ignore_index=True)
    df = df.rename(columns=RENAME)
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time"] + [c for c in RENAME.values() if c != "time"]]
    df = df.sort_values("time").drop_duplicates("time", keep="last").set_index("time")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete hourly grid, interpolate short gaps, fix CO level shift."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full)
    df.index.name = "time"
    df["is_imputed"] = df[TARGET].isna()
    # CO in this dataset changes scale during 2024 (source/unit change). Robust-scale per calendar
    # year so models do not learn a spurious year effect. Documented in the paper's Data section.
    med = df["co"].groupby(df.index.year).transform("median")
    iqr = df["co"].groupby(df.index.year).transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    df["co"] = (df["co"] - med) / iqr.replace(0, np.nan)
    num = df.columns.difference(["is_imputed"])
    df[num] = df[num].interpolate(limit=MAX_GAP_INTERP, limit_area="inside")
    return df


def load(force: bool = False) -> pd.DataFrame:
    if os.path.exists(PROCESSED) and not force:
        return pd.read_parquet(PROCESSED)
    df = clean(load_raw())
    os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
    df.to_parquet(PROCESSED)
    return df


if __name__ == "__main__":
    d = load(force=True)
    print(d.shape, d.index.min(), d.index.max())
    print(d.isna().sum())
