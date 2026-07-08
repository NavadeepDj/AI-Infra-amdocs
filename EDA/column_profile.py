import pandas as pd
from pathlib import Path

# ==========================================================
# STEP 9: Column Profiling and Data Dictionary
#
# Goal:
# - Profile every column in the aligned 20260702 files
# - Identify data type, missing values, unique values, and ML role
# - Prepare a data dictionary for later feature engineering
#
# This is the final EDA bridge before data engineering.
# ==========================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "datasets"
DOCS_FOLDER = PROJECT_FOLDER / "docs" / "EDA_docs"

datasets = [
    {
        "dataset": "Ping",
        "file": DATA_FOLDER / "ping_status_export_20260702_mockup.csv",
        "time_columns": ["timestamp"],
        "identifier_columns": ["id", "vm_name", "vm_ip"],
    },
    {
        "dataset": "HPE iLO",
        "file": DATA_FOLDER / "hpe_ilo_health_export_20260702_mockup.csv",
        "time_columns": ["recorded_at"],
        "identifier_columns": ["id", "server_name", "ip_address"],
    },
    {
        "dataset": "Dell iDRAC",
        "file": DATA_FOLDER / "dell_idrac_health_ext_export_20260702_mockup.csv",
        "time_columns": ["timestamp"],
        "identifier_columns": ["id", "server_name", "ip_address"],
    },
]

STATUS_FEATURE_COLUMNS = {
    "status",
    "overall_status",
    "fans",
    "cpu",
    "memory",
    "storage",
    "temperature",
    "power",
}

TEXT_FEATURE_COLUMNS = {
    "issues_detected",
    "comments",
    "current_problems",
}


def classify_column(column_name, config):
    if column_name in config["identifier_columns"]:
        return "Identifier", "Keep for joins/grouping, not direct ML feature"

    if column_name in config["time_columns"]:
        return "Time", "Derive monitoring_slot and time features"

    if column_name in STATUS_FEATURE_COLUMNS:
        return "Categorical feature", "Use after encoding/severity mapping"

    if column_name in TEXT_FEATURE_COLUMNS:
        return "Text feature", "Use for issue flags or text extraction"

    return "Review", "Inspect before using"


def sample_unique_values(series, limit=8):
    values = series.dropna().astype(str).unique()
    values = sorted(values)[:limit]
    return ", ".join(values)


print("=" * 80)
print("STEP 9: COLUMN PROFILING AND DATA DICTIONARY")
print("=" * 80)

all_profiles = []

for config in datasets:
    df = pd.read_csv(config["file"])

    print(f"\nDataset: {config['dataset']}")
    print("-" * 100)
    print(
        f"{'Column':20s} {'Pandas Type':15s} {'Role':22s} "
        f"{'Missing':>8s} {'Missing %':>10s} {'Unique':>8s}  Sample Values"
    )
    print("-" * 100)

    for column in df.columns:
        role, keep_decision = classify_column(column, config)
        missing_count = df[column].isna().sum()
        missing_percent = missing_count / len(df) * 100
        unique_count = df[column].nunique(dropna=True)
        sample_values = sample_unique_values(df[column])

        print(
            f"{column:20s} {str(df[column].dtype):15s} {role:22s} "
            f"{missing_count:8d} {missing_percent:9.2f}% {unique_count:8d}  "
            f"{sample_values}"
        )

        all_profiles.append(
            {
                "dataset": config["dataset"],
                "column": column,
                "pandas_type": str(df[column].dtype),
                "role": role,
                "missing_count": missing_count,
                "missing_percent": round(missing_percent, 2),
                "unique_count": unique_count,
                "sample_values": sample_values,
                "keep_decision": keep_decision,
            }
        )

profile_df = pd.DataFrame(all_profiles)
DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
output_file = DOCS_FOLDER / "column_profile_summary.csv"
profile_df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("Column profile CSV written to:")
print(output_file)
