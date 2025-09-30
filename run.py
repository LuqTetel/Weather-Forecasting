# run.py — robust weather analytics script (auto-detects column names)
# ---------------------------------------------------------------
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from statsmodels.tsa.arima.model import ARIMA

CSV_PATH = "weather.csv"  # keep your file in the same folder

# ---------- helpers ----------
def first_match(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in s:
            return s[cand.lower()]
    return None

def coerce_numeric(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")

def parse_any_datetime(series: pd.Series) -> pd.Series:
    """Parse a wide range of date representations into pandas Timestamps."""
    raw = series.copy()

    # normalise numeric encodings such as YYYYMMDD integers
    if pd.api.types.is_numeric_dtype(raw):
        raw = raw.astype("Int64").astype(str)

    raw = raw.astype(str).str.strip()

    parsed = pd.to_datetime(raw, errors="coerce", utc=False)
    if parsed.notna().all():
        return parsed

    digits_mask = raw.str.fullmatch(r"\d{8}")
    if digits_mask.any():
        parsed_digits = pd.to_datetime(raw.where(digits_mask), format="%Y%m%d", errors="coerce")
        parsed = parsed.combine_first(parsed_digits)
        if parsed.notna().all():
            return parsed

    return pd.to_datetime(raw, dayfirst=True, errors="coerce")

# ---------- load ----------
df = pd.read_csv(CSV_PATH)
if df.empty:
    raise SystemExit("ERROR: CSV is empty.")

# ---------- detect columns ----------
date_candidates = [
    "date","Date","DATE","observation_date","obs_date","dt","datetime","DateTime",
    "date_time","timestamp","time","day","record_date","Date Time"
]
temp_candidates = [
    "temperature_C","temperature","Temperature","temp","Temp","tavg","tmean","avg_temp","AverageTemperature","dly-tavg-normal"
]
temp_min_candidates = [
    "dly-tmin-normal","tmin","daily_min_temp","daily_minimum_temperature","min_temperature","min_temp"
]
temp_max_candidates = [
    "dly-tmax-normal","tmax","daily_max_temp","daily_maximum_temperature","max_temperature","max_temp"
]
rain_candidates = [
    "rainfall","Rainfall","precipitation","Precipitation","precip","prcp","rain","rain_mm","mm","total_rain"
]
humid_candidates = [
    "humidity","Humidity","rel_humidity","RH","rh","relative_humidity","Relative Humidity"
]
wind_candidates = [
    "wind_speed","WindSpeed","wind","Wind","windspeed","ws","wind_kmh","wind_mps"
]

cols = list(df.columns)
date_col = first_match(cols, date_candidates)
temp_col = first_match(cols, temp_candidates)
rain_col = first_match(cols, rain_candidates)
humid_col = first_match(cols, humid_candidates)
wind_col = first_match(cols, wind_candidates)

if temp_col is None:
    temp_min_col = first_match(cols, temp_min_candidates)
    temp_max_col = first_match(cols, temp_max_candidates)

    def _to_fahrenheit(series: pd.Series, column_name: str) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        name = column_name.lower()
        if "dly-" in name or name.endswith("-normal"):
            values = values / 10.0
        return values

    temp_components: List[pd.Series] = []
    source_labels: List[str] = []
    if temp_min_col:
        temp_components.append(_to_fahrenheit(df[temp_min_col], temp_min_col))
        source_labels.append(temp_min_col)
    if temp_max_col:
        temp_components.append(_to_fahrenheit(df[temp_max_col], temp_max_col))
        source_labels.append(temp_max_col)
    if temp_components:
        temp_f = temp_components[0]
        if len(temp_components) > 1:
            temp_f = sum(temp_components) / len(temp_components)
        df["temperature_C"] = (temp_f - 32.0) * (5.0 / 9.0)
        temp_col = "temperature_C"
        if source_labels:
            print(f"Derived temperature from {', '.join(source_labels)} (Fahrenheit to Celsius).")

missing_core = []
if date_col is None: missing_core.append("date")
if temp_col is None: missing_core.append("temperature")
if missing_core:
    raise SystemExit(f"ERROR: Missing required column(s): {', '.join(missing_core)}. "
                     f"Found columns: {cols}")

# ---------- standardize names ----------
df.rename(columns={date_col: "date", temp_col: "temperature_C"}, inplace=True)
if rain_col and rain_col != "rainfall":
    df.rename(columns={rain_col: "rainfall"}, inplace=True)
if humid_col and humid_col != "humidity":
    df.rename(columns={humid_col: "humidity"}, inplace=True)
if wind_col and wind_col != "wind_speed":
    df.rename(columns={wind_col: "wind_speed"}, inplace=True)

# ---------- types / cleaning ----------
df["date"] = parse_any_datetime(df["date"])
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

coerce_numeric(df, "temperature_C")
if "rainfall" in df:
    coerce_numeric(df, "rainfall")
    df["rainfall"] = df["rainfall"].fillna(0.0)
if "humidity" in df:   coerce_numeric(df, "humidity")
if "wind_speed" in df: coerce_numeric(df, "wind_speed")

# optional: simple outlier clamps for obviously broken entries
if "temperature_C" in df:
    df["temperature_C"] = df["temperature_C"].clip(lower=-50, upper=60)
if "rainfall" in df:
    df["rainfall"] = df["rainfall"].clip(lower=0)

# quick sanity view
print("Columns standardized as:", list(df.columns))
print(df.head(3))
print(df.info())

# ---------- EDA ----------
# monthly average temperature
df["month"] = df["date"].dt.month
monthly_temp = df.groupby("month")["temperature_C"].mean().reindex(range(1,13))
plt.plot(monthly_temp.index, monthly_temp.values, marker="o")
plt.title("Average Monthly Temperature")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# rainfall distribution (only if available)
if "rainfall" in df:
    df["rainfall"].hist(bins=30)
    plt.title("Rainfall Distribution")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# ---------- extremes ----------
heatwave_days = int((df["temperature_C"] > 35).sum())
print("Number of heatwave days:", heatwave_days)

if "rainfall" in df:
    heavy_rain_days = int((df["rainfall"] > 50).sum())
    print("Number of heavy rain days:", heavy_rain_days)

# ---------- forecasting (monthly) ----------
# need enough data points for ARIMA; guard against tiny series
monthly_series = (
    df.set_index("date")
      .resample("M")["temperature_C"]
      .mean()
      .dropna()
)

if monthly_series.size >= 24 and monthly_series.notna().sum() >= 24:
    # simple ARIMA(1,1,1) baseline; adjust later if needed
    model = ARIMA(monthly_series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)
    print("Forecast next 12 months:")
    print(forecast)

    ax = monthly_series.plot(label="History")
    forecast.plot(ax=ax, label="Forecast", legend=True, title="Monthly Temperature Forecast")
    plt.tight_layout()
    plt.show()
else:
    print(f"Forecast skipped: insufficient monthly data ({monthly_series.size} points, need ≥ 24).")
# run.py — robust weather analytics script (auto-detects column names)
# ---------------------------------------------------------------
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from statsmodels.tsa.arima.model import ARIMA

CSV_PATH = "weather.csv"  # keep your file in the same folder

# ---------- helpers ----------
def first_match(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in s:
            return s[cand.lower()]
    return None

def coerce_numeric(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")

def parse_any_datetime(series: pd.Series) -> pd.Series:
    """Parse a wide range of date representations into pandas Timestamps."""
    raw = series.copy()

    # normalise numeric encodings such as YYYYMMDD integers
    if pd.api.types.is_numeric_dtype(raw):
        raw = raw.astype("Int64").astype(str)

    raw = raw.astype(str).str.strip()

    parsed = pd.to_datetime(raw, errors="coerce", utc=False)
    if parsed.notna().all():
        return parsed

    digits_mask = raw.str.fullmatch(r"\d{8}")
    if digits_mask.any():
        parsed_digits = pd.to_datetime(raw.where(digits_mask), format="%Y%m%d", errors="coerce")
        parsed = parsed.combine_first(parsed_digits)
        if parsed.notna().all():
            return parsed

    return pd.to_datetime(raw, dayfirst=True, errors="coerce")

# ---------- load ----------
df = pd.read_csv(CSV_PATH)
if df.empty:
    raise SystemExit("ERROR: CSV is empty.")

# ---------- detect columns ----------
date_candidates = [
    "date","Date","DATE","observation_date","obs_date","dt","datetime","DateTime",
    "date_time","timestamp","time","day","record_date","Date Time"
]
temp_candidates = [
    "temperature_C","temperature","Temperature","temp","Temp","tavg","tmean","avg_temp","AverageTemperature","dly-tavg-normal"
]
temp_min_candidates = [
    "dly-tmin-normal","tmin","daily_min_temp","daily_minimum_temperature","min_temperature","min_temp"
]
temp_max_candidates = [
    "dly-tmax-normal","tmax","daily_max_temp","daily_maximum_temperature","max_temperature","max_temp"
]
rain_candidates = [
    "rainfall","Rainfall","precipitation","Precipitation","precip","prcp","rain","rain_mm","mm","total_rain"
]
humid_candidates = [
    "humidity","Humidity","rel_humidity","RH","rh","relative_humidity","Relative Humidity"
]
wind_candidates = [
    "wind_speed","WindSpeed","wind","Wind","windspeed","ws","wind_kmh","wind_mps"
]

cols = list(df.columns)
date_col = first_match(cols, date_candidates)
temp_col = first_match(cols, temp_candidates)
rain_col = first_match(cols, rain_candidates)
humid_col = first_match(cols, humid_candidates)
wind_col = first_match(cols, wind_candidates)

if temp_col is None:
    temp_min_col = first_match(cols, temp_min_candidates)
    temp_max_col = first_match(cols, temp_max_candidates)

    def _to_fahrenheit(series: pd.Series, column_name: str) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        name = column_name.lower()
        if "dly-" in name or name.endswith("-normal"):
            values = values / 10.0
        return values

    temp_components: List[pd.Series] = []
    source_labels: List[str] = []
    if temp_min_col:
        temp_components.append(_to_fahrenheit(df[temp_min_col], temp_min_col))
        source_labels.append(temp_min_col)
    if temp_max_col:
        temp_components.append(_to_fahrenheit(df[temp_max_col], temp_max_col))
        source_labels.append(temp_max_col)
    if temp_components:
        temp_f = temp_components[0]
        if len(temp_components) > 1:
            temp_f = sum(temp_components) / len(temp_components)
        df["temperature_C"] = (temp_f - 32.0) * (5.0 / 9.0)
        temp_col = "temperature_C"
        if source_labels:
            print(f"Derived temperature from {', '.join(source_labels)} (Fahrenheit to Celsius).")

missing_core = []
if date_col is None: missing_core.append("date")
if temp_col is None: missing_core.append("temperature")
if missing_core:
    raise SystemExit(f"ERROR: Missing required column(s): {', '.join(missing_core)}. "
                     f"Found columns: {cols}")

# ---------- standardize names ----------
df.rename(columns={date_col: "date", temp_col: "temperature_C"}, inplace=True)
if rain_col and rain_col != "rainfall":
    df.rename(columns={rain_col: "rainfall"}, inplace=True)
if humid_col and humid_col != "humidity":
    df.rename(columns={humid_col: "humidity"}, inplace=True)
if wind_col and wind_col != "wind_speed":
    df.rename(columns={wind_col: "wind_speed"}, inplace=True)

# ---------- types / cleaning ----------
df["date"] = parse_any_datetime(df["date"])
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

coerce_numeric(df, "temperature_C")
if "rainfall" in df:
    coerce_numeric(df, "rainfall")
    df["rainfall"] = df["rainfall"].fillna(0.0)
if "humidity" in df:   coerce_numeric(df, "humidity")
if "wind_speed" in df: coerce_numeric(df, "wind_speed")

# optional: simple outlier clamps for obviously broken entries
if "temperature_C" in df:
    df["temperature_C"] = df["temperature_C"].clip(lower=-50, upper=60)
if "rainfall" in df:
    df["rainfall"] = df["rainfall"].clip(lower=0)

# quick sanity view
print("Columns standardized as:", list(df.columns))
print(df.head(3))
print(df.info())

# ---------- EDA ----------
# monthly average temperature
df["month"] = df["date"].dt.month
monthly_temp = df.groupby("month")["temperature_C"].mean().reindex(range(1,13))
plt.plot(monthly_temp.index, monthly_temp.values, marker="o")
plt.title("Average Monthly Temperature")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# rainfall distribution (only if available)
if "rainfall" in df:
    df["rainfall"].hist(bins=30)
    plt.title("Rainfall Distribution")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# ---------- extremes ----------
heatwave_days = int((df["temperature_C"] > 35).sum())
print("Number of heatwave days:", heatwave_days)

if "rainfall" in df:
    heavy_rain_days = int((df["rainfall"] > 50).sum())
    print("Number of heavy rain days:", heavy_rain_days)

# ---------- forecasting (monthly) ----------
# need enough data points for ARIMA; guard against tiny series
monthly_series = (
    df.set_index("date")
      .resample("M")["temperature_C"]
      .mean()
      .dropna()
)

if monthly_series.size >= 24 and monthly_series.notna().sum() >= 24:
    # simple ARIMA(1,1,1) baseline; adjust later if needed
    model = ARIMA(monthly_series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)
    print("Forecast next 12 months:")
    print(forecast)

    ax = monthly_series.plot(label="History")
    forecast.plot(ax=ax, label="Forecast", legend=True, title="Monthly Temperature Forecast")
    plt.tight_layout()
    plt.show()
else:
    print(f"Forecast skipped: insufficient monthly data ({monthly_series.size} points, need ≥ 24).")
