import pandas as pd
from pathlib import Path

# ==========================================================
# STEP 7: Monitoring Frequency
#
# Goal:
# - Use the aligned 20260702 files
# - Check how often each machine is monitored
# - Verify whether the expected pattern is 6 observations per day
# - Identify the common monitoring hours
#
# Why only 20260702 files for now?
# These files cover the same one-month window:
# 2026-06-02 to 2026-07-02
#
# Once the logic is clear, we can apply the same process to other exports.
# ==========================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "datasets"

datasets = [
    {
        "name": "Ping",
        "file": DATA_FOLDER / "ping_status_export_20260702_mockup.csv",
        "machine_col": "vm_name",
        "time_col": "timestamp",
    },
    {
        "name": "HPE iLO",
        "file": DATA_FOLDER / "hpe_ilo_health_export_20260702_mockup.csv",
        "machine_col": "server_name",
        "time_col": "recorded_at",
    },
    {
        "name": "Dell iDRAC",
        "file": DATA_FOLDER / "dell_idrac_health_ext_export_20260702_mockup.csv",
        "machine_col": "server_name",
        "time_col": "timestamp",
    },
]


def parse_time(series):
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def create_monitoring_slot(timestamp_series):
    timestamps = parse_time(timestamp_series)

    # Observed monitoring schedule is near:
    # 02:00, 06:00, 10:00, 14:00, 18:00, 22:00
    return (timestamps - pd.Timedelta(hours=2)).dt.floor("4h") + pd.Timedelta(hours=2)


print("=" * 80)
print("STEP 7: MONITORING FREQUENCY")
print("=" * 80)

for config in datasets:
    df = pd.read_csv(config["file"])
    df["event_time"] = parse_time(df[config["time_col"]])
    df["event_date"] = df["event_time"].dt.date
    df["event_hour"] = df["event_time"].dt.hour
    df["monitoring_slot"] = create_monitoring_slot(df[config["time_col"]])
    df["slot_hour"] = df["monitoring_slot"].dt.hour

    observations_per_machine = df.groupby(config["machine_col"]).size()
    observations_per_machine_per_day = (
        df.groupby([config["machine_col"], "event_date"]).size()
    )

    print(f"\nDataset: {config['name']}")
    print("-" * 70)
    print(f"File                         : {config['file'].name}")
    print(f"Rows                         : {len(df)}")
    print(f"Unique machines              : {df[config['machine_col']].nunique()}")
    print(f"Unique dates                 : {df['event_date'].nunique()}")
    print(f"Unique monitoring slots      : {df['monitoring_slot'].nunique()}")

    print("\nObservations per machine")
    print(f"Minimum                      : {observations_per_machine.min()}")
    print(f"Maximum                      : {observations_per_machine.max()}")
    print(f"Median                       : {observations_per_machine.median()}")

    print("\nObservations per machine per day")
    print(f"Minimum                      : {observations_per_machine_per_day.min()}")
    print(f"Maximum                      : {observations_per_machine_per_day.max()}")
    print(f"Median                       : {observations_per_machine_per_day.median()}")

    print("\nRaw timestamp hours")
    print(sorted(df["event_hour"].dropna().unique()))

    print("\nDerived monitoring slot hours")
    print(sorted(df["slot_hour"].dropna().unique()))

    print("\nCounts by monitoring slot hour")
    print(df["slot_hour"].value_counts().sort_index().to_string())
