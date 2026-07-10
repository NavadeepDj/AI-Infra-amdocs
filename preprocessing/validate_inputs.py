import pandas as pd
from pathlib import Path
import sys

# Define dataset paths
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
PING_PATH = DATASETS_DIR / "ping_status_export_20260702_mockup.csv"
HPE_PATH = DATASETS_DIR / "hpe_ilo_health_export_20260702_mockup.csv"
DELL_PATH = DATASETS_DIR / "dell_idrac_health_ext_export_20260702_mockup.csv"

# Expected row counts from 20260702 EDA
EXPECTED_COUNTS = {
    "ping": 45756,
    "hpe": 2790,
    "dell": 4836
}

# Expected raw columns
EXPECTED_COLUMNS = {
    "ping": ["id", "vm_name", "vm_ip", "status", "timestamp"],
    "hpe": ["id", "ip_address", "fans", "cpu", "memory", "storage", "temperature", "power", "recorded_at", "server_name", "current_problems"],
    "dell": ["id", "ip_address", "status", "issues_detected", "comments", "timestamp", "overall_status", "fans", "cpu", "memory", "storage", "temperature", "power", "server_name", "current_problems"]
}

def validate_files():
    print("=== Step 1: Input Validation ===")
    
    # 1. Check file existence
    for name, path in [("Ping Status", PING_PATH), ("HPE iLO Health", HPE_PATH), ("Dell iDRAC Extended", DELL_PATH)]:
        if not path.exists():
            print(f"[ERROR] File does not exist: {path}")
            sys.exit(1)
        print(f"[OK] Found file: {name} ({path.name})")
        
    # 2. Load files and verify row counts
    try:
        ping_df = pd.read_csv(PING_PATH)
        hpe_df = pd.read_csv(HPE_PATH)
        dell_df = pd.read_csv(DELL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load raw CSV files: {e}")
        sys.exit(1)
        
    for label, df, expected_count in [
        ("Ping Status", ping_df, EXPECTED_COUNTS["ping"]),
        ("HPE iLO Health", hpe_df, EXPECTED_COUNTS["hpe"]),
        ("Dell iDRAC Extended", dell_df, EXPECTED_COUNTS["dell"])
    ]:
        actual_count = len(df)
        if actual_count != expected_count:
            print(f"[ERROR] Row count mismatch for {label}. Expected: {expected_count}, Actual: {actual_count}")
            sys.exit(1)
        print(f"[OK] {label} row count matches expected: {actual_count}")
        
    # 3. Check for presence of expected columns
    for label, df, expected_cols in [
        ("Ping Status", ping_df, EXPECTED_COLUMNS["ping"]),
        ("HPE iLO Health", hpe_df, EXPECTED_COLUMNS["hpe"]),
        ("Dell iDRAC Extended", dell_df, EXPECTED_COLUMNS["dell"])
    ]:
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"[ERROR] {label} is missing expected columns: {missing_cols}")
            sys.exit(1)
        print(f"[OK] {label} schema contains all expected columns.")
        
    print("[SUCCESS] Step 1: All raw inputs validated successfully.\n")
    return ping_df, hpe_df, dell_df

if __name__ == "__main__":
    validate_files()
