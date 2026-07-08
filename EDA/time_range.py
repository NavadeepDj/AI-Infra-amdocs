import pandas as pd
from pathlib import Path

# ==========================================================
# STEP 6: Time Range
#
# Goal:
# - Find the earliest timestamp in each CSV
# - Find the latest timestamp in each CSV
# - Calculate how many days of data each file covers
#
# Why this matters:
# The assignment says we have one month of infrastructure data.
# Before doing time-series analysis, anomaly detection, or forecasting,
# we must verify the actual time coverage.
#
# Important:
# Some timestamps look like 2/6/2026.
# This can mean 2 June 2026 or 6 February 2026 depending on date format.
# So this script tests both day/month/year and month/day/year, then selects
# the parse result that best matches a one-month export.
# ==========================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "datasets"
csv_files = sorted(DATA_FOLDER.glob("*.csv"))


def get_time_column(df):
    if "timestamp" in df.columns:
        return "timestamp"
    if "recorded_at" in df.columns:
        return "recorded_at"
    return None


def parse_timestamp(values, dayfirst):
    return pd.to_datetime(values, dayfirst=dayfirst, errors="coerce")


def score_parse_result(parsed_time):
    invalid_count = parsed_time.isna().sum()
    if parsed_time.notna().sum() == 0:
        return float("inf")

    duration_days = (parsed_time.max() - parsed_time.min()).total_seconds() / 86400

    # A good parse should have few invalid timestamps and a duration near one month.
    return invalid_count * 1000 + abs(duration_days - 30)


def choose_best_timestamp_parse(values):
    day_month_year = parse_timestamp(values, dayfirst=True)
    month_day_year = parse_timestamp(values, dayfirst=False)

    dmy_score = score_parse_result(day_month_year)
    mdy_score = score_parse_result(month_day_year)

    if dmy_score <= mdy_score:
        return day_month_year, "day/month/year"

    return month_day_year, "month/day/year"


print("=" * 80)
print("STEP 6: TIME RANGE")
print("=" * 80)

for file in csv_files:
    df = pd.read_csv(file)
    time_col = get_time_column(df)

    print(f"\nDataset: {file.name}")
    print("-" * 70)

    if time_col is None:
        print("No timestamp column found.")
        continue

    parsed_time, selected_format = choose_best_timestamp_parse(df[time_col])
    invalid_count = parsed_time.isna().sum()

    start_time = parsed_time.min()
    end_time = parsed_time.max()
    duration = end_time - start_time
    unique_dates = parsed_time.dt.date.nunique()

    print(f"Time column       : {time_col}")
    print(f"Selected format   : {selected_format}")
    print(f"Invalid timestamps: {invalid_count}")
    print(f"Start time        : {start_time}")
    print(f"End time          : {end_time}")
    print(f"Duration          : {duration}")
    print(f"Unique dates      : {unique_dates}")

    print("\nFirst 3 timestamps")
    for value in df[time_col].head(3):
        print(f"  {value}")

    print("\nLast 3 timestamps")
    for value in df[time_col].tail(3):
        print(f"  {value}")
