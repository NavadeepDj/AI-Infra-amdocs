import pandas as pd
from pathlib import Path

# ==========================================================
# STEP 8: Monitoring Timeline Validation
#
# Goal:
# - Create the monitoring_slot feature
# - Verify each machine has every expected slot
# - Verify intervals are consistently 4 hours
# - Check duplicate machine + timestamp records
# - Check duplicate machine + monitoring_slot records
#
# Why this matters:
# The final ML dataset should have one row per:
# machine_name + ip_address + monitoring_slot
# ==========================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "datasets"

EXPECTED_SLOT_HOURS = [2, 6, 10, 14, 18, 22]
EXPECTED_OBSERVATIONS_PER_DAY = 6

datasets = [
    {
        "name": "Ping",
        "file": DATA_FOLDER / "ping_status_export_20260702_mockup.csv",
        "machine_col": "vm_name",
        "ip_col": "vm_ip",
        "time_col": "timestamp",
    },
    {
        "name": "HPE iLO",
        "file": DATA_FOLDER / "hpe_ilo_health_export_20260702_mockup.csv",
        "machine_col": "server_name",
        "ip_col": "ip_address",
        "time_col": "recorded_at",
    },
    {
        "name": "Dell iDRAC",
        "file": DATA_FOLDER / "dell_idrac_health_ext_export_20260702_mockup.csv",
        "machine_col": "server_name",
        "ip_col": "ip_address",
        "time_col": "timestamp",
    },
]


def parse_time(series):
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def create_monitoring_slot(timestamp_series):
    timestamps = parse_time(timestamp_series)
    return (timestamps - pd.Timedelta(hours=2)).dt.floor("4h") + pd.Timedelta(hours=2)


def validate_dataset(config):
    df = pd.read_csv(config["file"])
    machine_col = config["machine_col"]
    ip_col = config["ip_col"]
    time_col = config["time_col"]

    df["event_time"] = parse_time(df[time_col])
    df["monitoring_slot"] = create_monitoring_slot(df[time_col])
    df["event_date"] = df["event_time"].dt.date
    df["slot_hour"] = df["monitoring_slot"].dt.hour

    min_date = df["event_date"].min()
    max_date = df["event_date"].max()
    unique_days = df["event_date"].nunique()
    expected_observations = unique_days * EXPECTED_OBSERVATIONS_PER_DAY

    observations_per_machine = df.groupby(machine_col).size()
    machines_with_wrong_count = observations_per_machine[
        observations_per_machine != expected_observations
    ]

    hours_per_machine = (
        df.groupby(machine_col)["slot_hour"]
        .apply(lambda values: sorted(values.dropna().unique()))
    )
    machines_with_wrong_hours = hours_per_machine[
        hours_per_machine.apply(lambda hours: hours != EXPECTED_SLOT_HOURS)
    ]

    duplicate_timestamp_rows = df.duplicated(
        subset=[machine_col, ip_col, "event_time"]
    ).sum()
    duplicate_slot_rows = df.duplicated(
        subset=[machine_col, ip_col, "monitoring_slot"]
    ).sum()

    interval_summary = []
    machines_with_bad_intervals = []

    for machine, machine_df in df.sort_values("event_time").groupby(machine_col):
        intervals = machine_df["monitoring_slot"].diff().dropna()
        unique_intervals = sorted(intervals.unique())
        interval_hours = [interval / pd.Timedelta(hours=1) for interval in unique_intervals]

        if interval_hours != [4.0]:
            machines_with_bad_intervals.append((machine, interval_hours[:10]))

        interval_summary.extend(interval_hours)

    print(f"\nDataset: {config['name']}")
    print("-" * 70)
    print(f"File                                      : {config['file'].name}")
    print(f"Date range                                : {min_date} to {max_date}")
    print(f"Unique days                               : {unique_days}")
    print(f"Expected observations per machine          : {expected_observations}")
    print(f"Actual min observations per machine        : {observations_per_machine.min()}")
    print(f"Actual max observations per machine        : {observations_per_machine.max()}")
    print(f"Machines with wrong observation count      : {len(machines_with_wrong_count)}")
    print(f"Machines with wrong monitoring hours       : {len(machines_with_wrong_hours)}")
    print(f"Machines with non-4-hour slot intervals    : {len(machines_with_bad_intervals)}")
    print(f"Duplicate machine + IP + timestamp rows    : {duplicate_timestamp_rows}")
    print(f"Duplicate machine + IP + monitoring_slot   : {duplicate_slot_rows}")

    print("\nExpected slot hours")
    print(EXPECTED_SLOT_HOURS)

    print("\nSample machine schedules")
    for machine, hours in hours_per_machine.head(5).items():
        print(f"{machine}: {hours}")

    if len(machines_with_wrong_count) > 0:
        print("\nExamples: wrong observation count")
        print(machines_with_wrong_count.head(10).to_string())

    if len(machines_with_wrong_hours) > 0:
        print("\nExamples: wrong monitoring hours")
        for machine, hours in machines_with_wrong_hours.head(10).items():
            print(f"{machine}: {hours}")

    if len(machines_with_bad_intervals) > 0:
        print("\nExamples: non-4-hour intervals")
        for machine, intervals in machines_with_bad_intervals[:10]:
            print(f"{machine}: {intervals}")


print("=" * 80)
print("STEP 8: MONITORING TIMELINE VALIDATION")
print("=" * 80)

for dataset_config in datasets:
    validate_dataset(dataset_config)
