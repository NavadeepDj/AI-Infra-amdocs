import pandas as pd
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "datasets"
csv_files = sorted(DATA_FOLDER.glob("*.csv"))

print("=" * 80)
print("STEP 2: UNIQUE MACHINES")
print("=" * 80)

for file in csv_files:

    df = pd.read_csv(file)

    print(f"\nDataset : {file.name}")
    print("-" * 60)

    if "vm_name" in df.columns:
        machine_col = "vm_name"
    elif "server_name" in df.columns:
        machine_col = "server_name"
    else:
        print("Machine column not found!")
        continue

    unique_machines = df[machine_col].nunique()

    print(f"Machine Column : {machine_col}")
    print(f"Unique Machines: {unique_machines}")

    print("\nFirst 10 Machines:")

    for machine in sorted(df[machine_col].unique())[:10]:
        print("  •", machine)

    print("-" * 60)
