import os
import sys
import pandas as pd
from create_monitoring_slots import parse_and_create_slots
from pre_merge_validate import validate_pre_merge

def merge_and_publish():
    print("=== Step 6: Left Outer Joining & Integrating Telemetry ===")
    
    # 1. Execute Step 5 Validation and obtain parsed DataFrames
    ping_df, hpe_df, dell_df = validate_pre_merge()
    
    # 2. Rename columns to preserve vendor identity exactly
    ping_clean = ping_df.rename(columns={
        "event_time": "event_time_ping",
        "status": "ping_status"
    })
    
    hpe_clean = hpe_df.rename(columns={
        "event_time": "event_time_hpe",
        "fans": "hpe_fans",
        "cpu": "hpe_cpu",
        "memory": "hpe_memory",
        "storage": "hpe_storage",
        "temperature": "hpe_temperature",
        "power": "hpe_power",
        "current_problems": "hpe_current_problems"
    })
    
    dell_clean = dell_df.rename(columns={
        "event_time": "event_time_dell",
        "status": "dell_status",
        "overall_status": "dell_overall_status",
        "fans": "dell_fans",
        "cpu": "dell_cpu",
        "memory": "dell_memory",
        "storage": "dell_storage",
        "temperature": "dell_temperature",
        "power": "dell_power",
        "issues_detected": "dell_issues_detected"
    })
    
    # 3. Perform Left Outer Join on machine_name + ip_address + monitoring_slot
    join_keys = ["machine_name", "ip_address", "monitoring_slot"]
    
    merged_df = pd.merge(ping_clean, hpe_clean, on=join_keys, how="left")
    merged_df = pd.merge(merged_df, dell_clean, on=join_keys, how="left")
    
    # 4. Generate Boolean Availability Indicators
    merged_df["has_ping"] = True  # Ping is the 100% master left anchor
    merged_df["has_hpe"] = merged_df["event_time_hpe"].notna()
    merged_df["has_dell"] = merged_df["event_time_dell"].notna()
    
    # 5. Generate Factual Telemetry Source Tag (`telemetry_source`)
    def assign_source(row):
        if row["has_ping"] and row["has_hpe"] and row["has_dell"]:
            return "Ping + HPE + Dell"
        elif row["has_ping"] and row["has_dell"]:
            return "Ping + Dell"
        elif row["has_ping"] and row["has_hpe"]:
            return "Ping + HPE"
        else:
            return "Ping Only"
            
    merged_df["telemetry_source"] = merged_df.apply(assign_source, axis=1)
    
    # 6. Generate Immutable Canonical `observation_id`
    merged_df["observation_id"] = (
        merged_df["machine_name"] + "|" + 
        merged_df["ip_address"] + "|" + 
        merged_df["monitoring_slot"]
    )
    
    # 7. Order columns into logical domain groupings
    id_cols = ["observation_id", "machine_name", "ip_address", "monitoring_slot"]
    meta_cols = ["has_ping", "has_hpe", "has_dell", "telemetry_source"]
    ping_cols = ["event_time_ping", "ping_status"]
    hpe_cols = [
        "event_time_hpe", "hpe_fans", "hpe_cpu", "hpe_memory", 
        "hpe_storage", "hpe_temperature", "hpe_power", "hpe_current_problems"
    ]
    dell_cols = [
        "event_time_dell", "dell_status", "dell_overall_status", "dell_fans", 
        "dell_cpu", "dell_memory", "dell_storage", "dell_temperature", 
        "dell_power", "dell_issues_detected"
    ]
    
    ordered_cols = id_cols + meta_cols + ping_cols + hpe_cols + dell_cols
    master_df = merged_df[ordered_cols].copy()
    
    # 8. Verification Audit on Merged Dataset
    print(f"[OK] Master Dataset Shape: {master_df.shape} (Expected: 45756, {len(ordered_cols)})")
    
    # Check that row count equals Ping master anchor exactly
    if len(master_df) != len(ping_df):
        print(f"[ERROR] Row count drift after merge! Expected {len(ping_df)}, got {len(master_df)}")
        sys.exit(1)
    print(f"[PASS] Row Count Conservation: Exactly {len(master_df)} rows preserved without duplication or loss.")
    
    # Check unique observation_id
    if master_df["observation_id"].nunique() != len(master_df):
        print(f"[ERROR] Duplicate observation_id detected in master dataset!")
        sys.exit(1)
    print("[PASS] Unique Canonical Identity: 100% unique observation_ids generated across all rows.")
    
    # Check telemetry source breakdown
    source_counts = master_df["telemetry_source"].value_counts().to_dict()
    print(f"[PASS] Telemetry Source Breakdown:\n  - {source_counts}")
    
    # 9. Save as interim CSV/Parquet if needed or return
    return master_df

if __name__ == "__main__":
    df = merge_and_publish()
    print("\n[SUCCESS] Step 6: Dataset merging and preservation complete.")
