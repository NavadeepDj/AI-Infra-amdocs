import pandas as pd
from pathlib import Path

# ==========================================================
# STEP 5: HPE vs Dell Redundancy Check
#
# Hypothesis:
# "The HPE and Dell datasets contain redundant hardware health
# information for the overlapping 15 machines."
#
# Goal:
# - Align HPE and Dell records by machine, IP, and monitoring cycle
# - Compare component health fields row by row
# - Decide whether both sources are duplicates or independent signals
# ==========================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "datasets"

hpe_file = DATA_FOLDER / "hpe_ilo_health_export_20260702_mockup.csv"
dell_file = DATA_FOLDER / "dell_idrac_health_ext_export_20260702_mockup.csv"

COMPONENT_COLUMNS = ["cpu", "memory", "temperature", "power", "fans", "storage"]


def parse_datetime(series):
    # The 20260702 files use day/month/year format, for example 2/6/2026.
    return pd.to_datetime(series, dayfirst=True)


def create_monitoring_cycle(timestamp_series):
    timestamps = parse_datetime(timestamp_series)
    # Monitoring cycles in this dataset start near 02:00, 06:00, 10:00,
    # 14:00, 18:00, and 22:00. Shift by 2 hours before flooring so the
    # cycle labels match the observed schedule.
    return (timestamps - pd.Timedelta(hours=2)).dt.floor("4h") + pd.Timedelta(hours=2)


def prepare_hpe(df):
    prepared = df.copy()
    prepared["machine_name"] = prepared["server_name"]
    prepared["monitoring_cycle"] = create_monitoring_cycle(prepared["recorded_at"])

    keep_columns = ["machine_name", "ip_address", "monitoring_cycle"] + COMPONENT_COLUMNS
    return prepared[keep_columns].add_prefix("hpe_")


def prepare_dell(df):
    prepared = df.copy()
    prepared["machine_name"] = prepared["server_name"]
    prepared["monitoring_cycle"] = create_monitoring_cycle(prepared["timestamp"])

    keep_columns = ["machine_name", "ip_address", "monitoring_cycle"] + COMPONENT_COLUMNS
    return prepared[keep_columns].add_prefix("dell_")


print("=" * 80)
print("STEP 5: HPE VS DELL REDUNDANCY CHECK")
print("=" * 80)

hpe = pd.read_csv(hpe_file)
dell = pd.read_csv(dell_file)

hpe_prepared = prepare_hpe(hpe)
dell_prepared = prepare_dell(dell)

merged = hpe_prepared.merge(
    dell_prepared,
    left_on=["hpe_machine_name", "hpe_ip_address", "hpe_monitoring_cycle"],
    right_on=["dell_machine_name", "dell_ip_address", "dell_monitoring_cycle"],
    how="inner",
)

print(f"HPE rows                         : {len(hpe_prepared)}")
print(f"Dell rows                        : {len(dell_prepared)}")
print(f"Aligned HPE/Dell rows             : {len(merged)}")

print("\nComponent match rate")
print("-" * 60)

for column in COMPONENT_COLUMNS:
    hpe_col = f"hpe_{column}"
    dell_col = f"dell_{column}"
    same_count = (merged[hpe_col] == merged[dell_col]).sum()
    match_rate = same_count / len(merged) * 100 if len(merged) else 0
    mismatch_count = len(merged) - same_count

    print(f"{column:12s}: {match_rate:6.2f}% match ({mismatch_count} mismatches)")

print("\nRows where at least one component differs")
print("-" * 60)

component_match_columns = []
for column in COMPONENT_COLUMNS:
    match_col = f"{column}_matches"
    merged[match_col] = merged[f"hpe_{column}"] == merged[f"dell_{column}"]
    component_match_columns.append(match_col)

different_rows = merged[~merged[component_match_columns].all(axis=1)]

print(f"Different rows: {len(different_rows)}")

if len(different_rows) > 0:
    display_columns = [
        "hpe_machine_name",
        "hpe_ip_address",
        "hpe_monitoring_cycle",
    ]

    for column in COMPONENT_COLUMNS:
        display_columns.extend([f"hpe_{column}", f"dell_{column}"])

    print("\nFirst 20 differences")
    print("-" * 60)
    print(different_rows[display_columns].head(20).to_string(index=False))

print("\nInterpretation guide")
print("-" * 60)
print("99-100% match  : likely redundant information")
print("High mismatch  : independent measurements, preserve both")
print("Low mismatch   : mostly redundant, but investigate conflicting rows")
