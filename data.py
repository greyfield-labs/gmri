import os
import warnings

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

START_DATE = "1970-01-01"


def get_fred_client() -> Fred:
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not found. Create a .env file with FRED_API_KEY=<your_key>."
        )
    return Fred(api_key=api_key)


def fetch_series(fred: Fred, series_id: str, start: str = START_DATE) -> pd.Series:
    series = fred.get_series(series_id, observation_start=start)
    series.name = series_id
    return series.dropna()


def resample_to_month_end(series: pd.Series) -> pd.Series:
    series = series.copy()
    series.index = series.index + pd.offsets.MonthEnd(0)
    return series


def interpolate_quarterly_to_monthly(
    series: pd.Series, method: str = "cubic"
) -> pd.Series:
    # Expand to monthly frequency, creating NaN for the two intermediate months
    monthly = series.resample("ME").asfreq()

    # Check for multi-quarter gaps in source data and warn
    raw_gaps = series.isna().sum()
    if raw_gaps > 0:
        warnings.warn(f"{series.name}: {raw_gaps} NaN values in raw quarterly series")

    # Interpolate between quarterly anchors
    monthly = monthly.interpolate(method=method)

    # Forward-fill trailing NaN (most recent quarter may have only 1 observation)
    monthly = monthly.ffill()

    return monthly


def compute_cpi_yoy(cpi: pd.Series) -> pd.Series:
    yoy = cpi.pct_change(12) * 100
    yoy.name = "CPI_YOY"
    return yoy.dropna()


def fetch_recession_indicator(fred: Fred, start: str = START_DATE) -> pd.Series:
    series = fetch_series(fred, "USREC", start=start)
    series = resample_to_month_end(series)
    series.name = "RECESSION"
    return series


def build_master_dataframe(fred: Fred) -> pd.DataFrame:
    print("  Fetching CPIAUCSL...")
    cpi_raw = fetch_series(fred, "CPIAUCSL")
    cpi_raw = resample_to_month_end(cpi_raw)
    cpi_yoy = compute_cpi_yoy(cpi_raw)

    print("  Fetching UNRATE...")
    unrate = fetch_series(fred, "UNRATE")
    unrate = resample_to_month_end(unrate)

    print("  Fetching GDPC1...")
    gdpc1_q = fetch_series(fred, "GDPC1", start="1947-01-01")
    gdpc1_q = resample_to_month_end(gdpc1_q)
    gdpc1 = interpolate_quarterly_to_monthly(gdpc1_q, method="cubic")
    gdpc1.name = "GDPC1"

    print("  Fetching NROU...")
    nrou_q = fetch_series(fred, "NROU", start="1947-01-01")
    nrou_q = resample_to_month_end(nrou_q)
    nrou = interpolate_quarterly_to_monthly(nrou_q, method="linear")
    nrou.name = "NROU"

    print("  Fetching GDPPOT...")
    gdppot_q = fetch_series(fred, "GDPPOT", start="1947-01-01")
    gdppot_q = resample_to_month_end(gdppot_q)
    gdppot = interpolate_quarterly_to_monthly(gdppot_q, method="cubic")
    gdppot.name = "GDPPOT"

    print("  Fetching USREC...")
    recession = fetch_recession_indicator(fred)

    # Inner-join the five classification signals
    main_signals = pd.concat(
        [cpi_yoy, unrate, gdpc1, nrou, gdppot], axis=1, join="inner"
    )

    # Left-join recession indicator (may have different coverage; fill missing with 0)
    df = main_signals.join(recession, how="left")
    df["RECESSION"] = df["RECESSION"].fillna(0).astype(int)

    # Trim to project start date
    df = df.loc[START_DATE:]

    # Drop any rows missing a classification input
    df = df.dropna(subset=["CPI_YOY", "UNRATE", "GDPC1", "NROU", "GDPPOT"])

    return df
