import os
import sys
import pandas as pd
from merge_datasets import merge_and_publish

def run_post_merge_audit():
    print("=== Step 7: 10-Point Post-Merge Gatekeeper Validation Audit ===")
    
    # 1. Ingest merged master dataset (`45756 x 28`) from Step 6
    df = merge_and_publish()
    
    audit_results = []
    
    # Check 1: Row Count Audit
    if len(df) != 45756:
        print(f"[ERROR] Check 1 FAILED: Row count mismatch! Expected 45756, got {len(df)}")
        sys.exit(1)
    audit_results.append(("1. Row Count Audit", "45,756 rows", f"{len(df)} rows", "PASS"))
    print("[PASS] Check 1: Row Count Audit confirmed exactly 45,756 observations.")
    
    # Check 2: Machine Count Audit
    n_machines = df["machine_name"].nunique()
    if n_machines != 246:
        print(f"[ERROR] Check 2 FAILED: Machine count mismatch! Expected 246, got {n_machines}")
        sys.exit(1)
    audit_results.append(("2. Machine Count Audit", "246 machines", f"{n_machines} machines", "PASS"))
    print("[PASS] Check 2: Machine Count Audit confirmed exactly 246 unique machines.")
    
    # Check 3: Monitoring Slot Audit
    n_slots = df["monitoring_slot"].nunique()
    if n_slots != 186:
        print(f"[ERROR] Check 3 FAILED: Monitoring slot count mismatch! Expected 186, got {n_slots}")
        sys.exit(1)
    audit_results.append(("3. Monitoring Slot Audit", "186 slots", f"{n_slots} slots", "PASS"))
    print("[PASS] Check 3: Monitoring Slot Audit confirmed exactly 186 canonical time slots (`Slot 02..22`).")
    
    # Check 4: Dual-Key Unique Audit (`observation_id` AND `machine_name + ip_address + monitoring_slot`)
    n_obs_id = df["observation_id"].nunique()
    n_comp_key = len(df.drop_duplicates(subset=["machine_name", "ip_address", "monitoring_slot"]))
    if n_obs_id != len(df) or n_comp_key != len(df):
        print(f"[ERROR] Check 4 FAILED: Duplicate keys detected! unique IDs: {n_obs_id}, unique comp keys: {n_comp_key}")
        sys.exit(1)
    audit_results.append(("4. Dual-Key Unique Audit", "0 duplicates", "0 duplicates (100% unique)", "PASS"))
    print("[PASS] Check 4: Dual-Key Unique Audit confirmed 0 duplicates over observation_id and composite keys.")
    
    # Check 5: Per-Machine Timeline Audit (`every machine must have exactly 186 slots after join`)
    machine_counts = df.groupby("machine_name").size()
    bad_machines = machine_counts[machine_counts != 186]
    if not bad_machines.empty:
        print(f"[ERROR] Check 5 FAILED: Found machines with != 186 observations after merge:\n{bad_machines}")
        sys.exit(1)
    audit_results.append(("5. Per-Machine Timeline Audit", "186 obs/machine", "All 246 machines have exactly 186 obs", "PASS"))
    print("[PASS] Check 5: Per-Machine Timeline Audit confirmed exactly 186 observations per machine without loss.")
    
    # Check 6: Lost Ping Records Audit
    n_ping = df["ping_status"].notna().sum()
    if n_ping != len(df):
        print(f"[ERROR] Check 6 FAILED: Lost Ping status records! Expected {len(df)}, got {n_ping}")
        sys.exit(1)
    audit_results.append(("6. Lost Ping Records Audit", "0 lost records", "45,756 valid ping_status records", "PASS"))
    print("[PASS] Check 6: Lost Ping Records Audit confirmed 100% network reachability status retention.")
    
    # Check 7: Telemetry Distribution Check
    dist = df["telemetry_source"].value_counts().to_dict()
    exp_dist = {"Ping Only": 40920, "Ping + HPE + Dell": 2790, "Ping + Dell": 2046}
    if dist != exp_dist:
        print(f"[ERROR] Check 7 FAILED: Telemetry distribution drift! Expected {exp_dist}, got {dist}")
        sys.exit(1)
    audit_results.append(("7. Telemetry Distribution Check", f"{exp_dist}", f"{dist}", "PASS"))
    print(f"[PASS] Check 7: Telemetry Distribution Check matched exact counts: {dist}")
    
    # Check 8: Vendor Overlap Check (`15 machines -> 2790 obs`)
    overlap_df = df[df["has_hpe"] & df["has_dell"]]
    if len(overlap_df) != 2790 or overlap_df["machine_name"].nunique() != 15:
        print(f"[ERROR] Check 8 FAILED: Vendor overlap mismatch! Expected 2790 obs over 15 machines, got {len(overlap_df)} obs across {overlap_df['machine_name'].nunique()} machines.")
        sys.exit(1)
    audit_results.append(("8. Vendor Overlap Check", "15 servers / 2,790 obs", f"{overlap_df['machine_name'].nunique()} servers / {len(overlap_df)} obs", "PASS"))
    print("[PASS] Check 8: Vendor Overlap Check confirmed exactly 15 servers sharing both iLO and iDRAC telemetry across 2,790 observations.")
    
    # Check 9: Exact Column Name Conservation Audit
    expected_cols = [
        "observation_id", "machine_name", "ip_address", "monitoring_slot",
        "has_ping", "has_hpe", "has_dell", "telemetry_source",
        "event_time_ping", "ping_status",
        "event_time_hpe", "hpe_fans", "hpe_cpu", "hpe_memory", 
        "hpe_storage", "hpe_temperature", "hpe_power", "hpe_current_problems",
        "event_time_dell", "dell_status", "dell_overall_status", "dell_fans", 
        "dell_cpu", "dell_memory", "dell_storage", "dell_temperature", 
        "dell_power", "dell_issues_detected"
    ]
    if list(df.columns) != expected_cols:
        print(f"[ERROR] Check 9 FAILED: Column schema or order mismatch!\nExpected: {expected_cols}\nActual: {list(df.columns)}")
        sys.exit(1)
    audit_results.append(("9. Exact Column Name Audit", "28 canonical attributes", "Exact match across all 28 names/ordering", "PASS"))
    print("[PASS] Check 9: Exact Column Name Audit confirmed 100% preservation of uncoalesced vendor column names.")
    
    # Check 10: Null Propagation Audit
    # Ping Only rows -> all hpe_* and dell_* must be NULL
    ping_only = df[df["telemetry_source"] == "Ping Only"]
    hpe_cols = [c for c in expected_cols if c.startswith("hpe_") or c == "event_time_hpe"]
    dell_cols = [c for c in expected_cols if c.startswith("dell_") or c == "event_time_dell"]
    
    if not ping_only[hpe_cols].isna().all().all() or not ping_only[dell_cols].isna().all().all():
        print("[ERROR] Check 10 FAILED: Null Propagation error on 'Ping Only' rows! Non-null hardware telemetry discovered.")
        sys.exit(1)
        
    # Ping + Dell rows -> all hpe_* must be NULL, all dell_* must be NOT NULL
    ping_dell = df[df["telemetry_source"] == "Ping + Dell"]
    if not ping_dell[hpe_cols].isna().all().all() or not ping_dell[dell_cols].notna().all().all():
        print("[ERROR] Check 10 FAILED: Null Propagation error on 'Ping + Dell' rows!")
        sys.exit(1)
        
    # Ping + HPE + Dell rows -> all hpe_* and dell_* must be NOT NULL
    ping_hpe_dell = df[df["telemetry_source"] == "Ping + HPE + Dell"]
    if not ping_hpe_dell[hpe_cols].notna().all().all() or not ping_hpe_dell[dell_cols].notna().all().all():
        print("[ERROR] Check 10 FAILED: Null Propagation error on 'Ping + HPE + Dell' rows!")
        sys.exit(1)
        
    audit_results.append(("10. Null Propagation Audit", "Strict NULL preservation", "100% correct missingness propagation across all subsets", "PASS"))
    print("[PASS] Check 10: Null Propagation Audit confirmed strict preservation of missing telemetry as NULL.")
    
    print("\n[SUCCESS] Gatekeeper Verdict: All 10 Post-Merge Validation checks PASSED cleanly!")
    
    # Write Certification Summary Artifact
    cert_path = "docs/preprocessing/merge_validation_summary.md"
    os.makedirs("docs/preprocessing", exist_ok=True)
    
    cert_content = f"""# Data Engineering Validation Certification & Gatekeeper Summary

**Audit Script:** [`preprocessing/post_merge_validate.py`](file:///c:/Users/navad/ML_data/preprocessing/post_merge_validate.py)  
**Gatekeeper Status:** **`PASS` (10/10 Engineering Validation Checks Passed)**  
**Certification Verdict:** **`APPROVED FOR FEATURE ENGINEERING`**  

---

## 1. Executive Summary & Design Compliance
The Phase 2 Data Engineering pipeline has successfully performed a lossless Left Outer Join over the 31-day operational window (`45,756` observations). In strict adherence to our core philosophy (*"Data Engineering must Preserve; Feature Engineering must Transform"*), zero vendor attributes were coalesced (`hpe_cpu` and `dell_cpu` preserved intact), all raw temporal signatures were retained (`event_time_ping`, `event_time_hpe`, `event_time_dell`), and Null Propagation verified that missing hardware telemetry is faithfully represented by `NULL` (`pd.isna`).

> **Engineering Certification Statement:**  
> *The unified monitoring dataset (`master_infrastructure_health_v1.parquet`) has successfully passed all defined Data Engineering validation checks and is approved for downstream Feature Engineering and Machine Learning.*

---

## 2. 10-Point Gatekeeper Audit Results

| # | Audit Check Name | Expected Boundary | Verified Actual Boundary | Gatekeeper Verdict |
| :---: | :--- | :--- | :--- | :---: |
"""
    for check_name, exp, act, status in audit_results:
        cert_content += f"| **{check_name}** | `{exp}` | `{act}` | **`{status}`** |\n"
        
    cert_content += """
---

## 3. Verified Telemetry Distribution Table

| Telemetry Source Tag | Machine Count | Observations (`Slots`) | Hardware Vendor Attributes | Null Propagation Behavior |
| :--- | :---: | :---: | :--- | :--- |
| **`Ping Only`** | `220` | `40,920` (`89.43%`) | None (`Network Reachability Only`) | All `hpe_*` and `dell_*` columns are `100% NULL` |
| **`Ping + HPE + Dell`** | `15` | `2,790` (`6.10%`) | Both `HPE iLO` & `Dell iDRAC Ext` | Both `hpe_*` and `dell_*` columns are `100% NOT NULL` |
| **`Ping + Dell`** | `11` | `2,046` (`4.47%`) | `Dell iDRAC Ext Only` | `hpe_*` columns `100% NULL`, `dell_*` `100% NOT NULL` |
| **Total Master Inventory** | **`246`** | **`45,756` (`100%`)** | **`28 Canonical Attributes`** | **`100% Preservation of Available Source Telemetry`** |

---

## 4. Phase 2 Sign-Off & Downstream Readiness
Because all 10 checks passed cleanly without errors or warnings, the master dataset (`45,756 x 28`) is approved for permanent versioned export to `datasets/master_infrastructure_health_v1.parquet` and `datasets/master_infrastructure_health_metadata_v1.json` via **Step 8 (`preprocessing/export_gold_dataset.py`)**.
"""
    with open(cert_path, "w", encoding="utf-8") as f:
        f.write(cert_content)
    print(f"[CERTIFICATION] Saved Engineering Validation Certification to {cert_path}")
    
    return df

if __name__ == "__main__":
    run_post_merge_audit()
