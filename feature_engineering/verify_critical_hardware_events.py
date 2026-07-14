#!/usr/bin/env python3
"""
feature_engineering/verify_critical_hardware_events.py

Inspects diagnostic text logs (`hpe_current_problems`, `dell_issues_detected`) and overall status
for all observations where `critical_component_count > 0` to empirically verify whether every
Rank 3 hardware event corresponds to a genuine operational failure.
"""

import pandas as pd
from pathlib import Path

def main():
    in_path = Path("datasets/features_stage2_temporal_v1.parquet")
    df = pd.read_parquet(in_path)
    
    # Filter where critical_component_count > 0
    crit_df = df[df['critical_component_count'] > 0]
    print(f"=== Verification of Critical Hardware Events ===")
    print(f"Total rows with critical_component_count > 0: {len(crit_df)} out of {len(df):,}\n")
    
    # Group by machine_name to see which servers experience critical hardware events
    for machine, m_df in crit_df.groupby('machine_name'):
        print(f"--- Server: {machine} ({len(m_df)} critical slots) ---")
        for idx, row in m_df.iterrows():
            slot = row['monitoring_slot']
            crit_cnt = row['critical_component_count']
            
            # Find which exact columns are Critical (= 3)
            crit_cols = []
            for col in ['hardware_cpu_worst_status', 'hardware_memory_worst_status', 
                        'hardware_fans_worst_status', 'hardware_storage_worst_status', 
                        'hardware_temperature_worst_status', 'hardware_power_worst_status']:
                if row[col] == 3:
                    crit_cols.append(col.replace('hardware_', '').replace('_worst_status', ''))
            
            # Extract diagnostic logs and overall status
            hpe_prob = row.get('hpe_current_problems', 'N/A')
            dell_prob = row.get('dell_issues_detected', 'N/A')
            dell_status = row.get('dell_overall_status', 'N/A')
            
            print(f"  Slot: {slot} | Critical Subsystems: {crit_cols} | Dell Overall Status: {dell_status}")
            if pd.notna(hpe_prob) and str(hpe_prob).strip() != "" and str(hpe_prob) != "None":
                print(f"    [HPE Problems]: {hpe_prob}")
            if pd.notna(dell_prob) and str(dell_prob).strip() != "" and str(dell_prob) != "None":
                print(f"    [Dell Issues]: {dell_prob}")
        print()

if __name__ == "__main__":
    main()
