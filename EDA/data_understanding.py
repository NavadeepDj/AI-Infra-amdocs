import pandas as pd
from pathlib import Path

# ==========================================================
# STEP 1: Basic Dataset Understanding
#
# Goal:
# - Load each CSV file
# - Print number of rows and columns
# - Print column names
#
# This is only the first simple step.
# We are not cleaning, merging, or building ML models yet.
# ==========================================================

# CSV files are currently stored in the main project folder.
# This makes the script work even when it is inside the EDA folder.
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "datasets"

# Automatically load all CSV files
csv_files = sorted(DATA_FOLDER.glob("*.csv"))

print("=" * 80)
print("STEP 1: INFRASTRUCTURE DATASET OVERVIEW")
print("=" * 80)

if not csv_files:
    print("No CSV files found.")
    print(f"Checked folder: {DATA_FOLDER}")

for file in csv_files:
    print(f"\nDataset : {file.name}")

    df = pd.read_csv(file)

    print("-" * 50)
    print(f"Rows              : {df.shape[0]:,}")
    print(f"Columns           : {df.shape[1]}")
    print(f"Memory Usage      : {df.memory_usage(deep=True).sum()/1024:.2f} KB")

    print("\nColumn Names:")

    for i, col in enumerate(df.columns, start=1):
        print(f"{i:2d}. {col}")

    print("-" * 50)
