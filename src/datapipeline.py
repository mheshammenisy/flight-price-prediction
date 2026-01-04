# src/data_pipeline.py

from __future__ import annotations

import numpy as np
import pandas as pd


def load_raw_excel(file_path: str) -> pd.DataFrame:
    """Load raw training data from Excel."""
    df = pd.read_excel(file_path)
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function matches your notebook EXACTLY:

    - Additional_Info: strip + lower
    - drop_duplicates
    - drop rows with missing Route or Total_Stops
    - Date_of_Journey -> journey_day, journey_month, journey_weekday
    - Dep_Time -> dep_hour
    - Duration -> duration_mins (then cast to int)
    - Total_Stops: 'non-stop' -> 0, extract digit, cast int
    - Airline/Source/Destination: lower + strip
    - Arrival_Time -> arr_hour (hour only from first 5 chars)
    - Drop columns: Date_of_Journey, Dep_Time, Arrival_Time, Duration, Route, journey_day
    - Price_log = log1p(Price)
    """
    df = df.copy()

    # Additional_Info normalization
    df["Additional_Info"] = df["Additional_Info"].astype(str).str.strip().str.lower()

    # drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # drop rows with missing Route / Total_Stops
    df = df.dropna(subset=["Route", "Total_Stops"]).reset_index(drop=True)

    # Date_of_Journey -> datetime + features
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True, errors="coerce")
    df["journey_day"] = df["Date_of_Journey"].dt.day
    df["journey_month"] = df["Date_of_Journey"].dt.month
    df["journey_weekday"] = df["Date_of_Journey"].dt.dayofweek  # 0=Mon

    # Dep_Time -> dep_hour
    dep = pd.to_datetime(df["Dep_Time"], format="%H:%M", errors="coerce")
    df["dep_hour"] = dep.dt.hour

    # Duration -> minutes
    dur = df["Duration"].astype(str).str.extract(r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?")
    df["duration_mins"] = (
        pd.to_numeric(dur[0], errors="coerce").fillna(0) * 60
        + pd.to_numeric(dur[1], errors="coerce").fillna(0)
    )
    df["duration_mins"] = df["duration_mins"].astype(int)

    # Total_Stops -> int
    df["Total_Stops"] = (
        df["Total_Stops"]
        .replace("non-stop", "0")
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    # normalize Airline/Source/Destination
    df["Airline"] = df["Airline"].astype(str).str.lower().str.strip()
    df["Source"] = df["Source"].astype(str).str.lower().str.strip()
    df["Destination"] = df["Destination"].astype(str).str.lower().str.strip()

    # Arrival_Time -> arr_hour
    df["arr_hour"] = pd.to_datetime(
        df["Arrival_Time"].astype(str).str[:5],
        format="%H:%M",
        errors="coerce"
    ).dt.hour

    # drop columns (exactly your list)
    cols_to_drop = [
        "Date_of_Journey",
        "Dep_Time",
        "Arrival_Time",
        "Duration",
        "Route",
        "journey_day",
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # target transform
    df["Price_log"] = np.log1p(df["Price"])

    return df


def build_featured_dataset_from_excel(file_path: str) -> pd.DataFrame:
    """Load raw Excel then clean+engineer features."""
    df = load_raw_excel(file_path)
    df = clean_and_engineer(df)
    return df


def save_featured(df: pd.DataFrame, out_path: str) -> None:
    """
    Save engineered dataset (before encoding/scaling).
    Recommended: .csv or .parquet
    """
    if out_path.lower().endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
