#!/usr/bin/env python3
"""
feature_engineering/analyze_not_ok.py

Investigates all occurrences of 'NOT OK' across the unified gold dataset
(datasets/master_infrastructure_health_v1.parquet) to determine:
1. Which exact columns contain 'NOT OK' and frequency counts.
2. Machine distribution and co-occurrence patterns.
3. Co-occurrence with 'Critical' or 'Degraded' across other components/vendors.
4. Associated diagnostic text (dell_issues_detected, hpe_current_problems).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    gold_path = Path("datasets/master_infrastructure_health_v1.parquet")
    if not gold_path.exists():
        print(f"[ERROR] Gold dataset not found at {gold_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading gold dataset: {gold_path} ...")
    df = pd.read_parquet(gold_path)
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}\n")

    # 1. Identify which columns contain 'NOT OK'
    not_ok_cols = []
    print("=== 1. Column Frequency of 'NOT OK' ===")
    for col in df.columns:
        if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
            count = (df[col] == 'NOT OK').sum()
            if count > 0:
                not_ok_cols.append(col)
                pct = (count / total_rows) * 100
                print(f"  - {col}: {count:,} occurrences ({pct:.4f}% of total rows)")

    if not not_ok_cols:
        print("  No occurrences of 'NOT OK' found in any column.")
        return

    # Create boolean mask for rows where ANY column is 'NOT OK'
    mask = pd.Series(False, index=df.index)
    for col in not_ok_cols:
        mask = mask | (df[col] == 'NOT OK')

    not_ok_df = df[mask].copy()
    total_not_ok_rows = len(not_ok_df)
    print(f"\nTotal unique observations with at least one 'NOT OK': {total_not_ok_rows:,} ({total_not_ok_rows/total_rows*100:.4f}%)\n")

    # 2. Machine Distribution
    print("=== 2. Machine Distribution of 'NOT OK' ===")
    machine_counts = not_ok_df['machine_name'].value_counts()
    for mach, cnt in machine_counts.items():
        # Check if machine is dual monitored or Dell only or HPE only
        sources = not_ok_df[not_ok_df['machine_name'] == mach]['telemetry_source'].unique()
        print(f"  - {mach} ({', '.join(sources)}): {cnt} observations")

    # 3. Co-occurrence Analysis with 'Critical' and 'Degraded'
    print("\n=== 3. Co-occurrence with 'Critical' / 'Degraded' on the same row ===")
    hardware_cols = [c for c in df.columns if any(comp in c for comp in ['cpu', 'memory', 'storage', 'fans', 'temperature', 'power'])]
    
    co_critical = 0
    co_degraded = 0
    co_both = 0
    only_not_ok = 0

    for idx, row in not_ok_df.iterrows():
        has_crit = any(row[c] == 'Critical' for c in hardware_cols if c in df.columns and pd.notna(row[c]))
        has_deg = any(row[c] == 'Degraded' for c in hardware_cols if c in df.columns and pd.notna(row[c]))
        
        if has_crit and has_deg:
            co_both += 1
        elif has_crit:
            co_critical += 1
        elif has_deg:
            co_degraded += 1
        else:
            only_not_ok += 1

    print(f"  - Rows with 'NOT OK' AND 'Critical' (on another/same component): {co_critical + co_both}")
    print(f"  - Rows with 'NOT OK' AND 'Degraded' (on another/same component): {co_degraded + co_both}")
    print(f"  - Rows with 'NOT OK' alongside BOTH 'Critical' and 'Degraded': {co_both}")
    print(f"  - Rows where 'NOT OK' is the ONLY non-OK hardware status: {only_not_ok}")

    # 4. Detailed Breakdown of Every 'NOT OK' Row
    print("\n=== 4. Detailed Inspection of ALL 'NOT OK' Observations ===")
    for idx, row in not_ok_df.iterrows():
        print(f"\nObservation ID: {row['observation_id']} | Slot: {row['monitoring_slot']} | Machine: {row['machine_name']} | Source: {row['telemetry_source']}")
        
        # List what is NOT OK
        not_ok_comps = [col for col in not_ok_cols if row[col] == 'NOT OK']
        print(f"  [NOT OK Columns]: {', '.join(not_ok_comps)}")
        
        # List other non-OK statuses on this row
        other_non_ok = []
        for col in hardware_cols:
            if pd.notna(row[col]) and row[col] not in ['OK', 'NOT OK'] and 'rank' not in col:
                other_non_ok.append(f"{col}={row[col]}")
        if other_non_ok:
            print(f"  [Other Non-OK Hardware]: {', '.join(other_non_ok)}")
        else:
            print("  [Other Non-OK Hardware]: None (All other hardware components are OK)")

        # Diagnostic Text
        dell_status = row.get('dell_overall_status', 'N/A')
        dell_issues = row.get('dell_issues_detected', 'N/A')
        hpe_problems = row.get('hpe_current_problems', 'N/A')
        
        print(f"  [Dell Overall Status]: {dell_status}")
        if pd.notna(dell_issues) and str(dell_issues).strip() != "" and str(dell_issues) != "None":
            print(f"  [Dell Issues Detected]: {dell_issues}")
        else:
            print("  [Dell Issues Detected]: (None/Blank)")
            
        if pd.notna(hpe_problems) and str(hpe_problems).strip() != "" and str(hpe_problems) != "None":
            print(f"  [HPE Current Problems]: {hpe_problems}")
        else:
            print("  [HPE Current Problems]: (None/Blank)")

if __name__ == "__main__":
    main()
