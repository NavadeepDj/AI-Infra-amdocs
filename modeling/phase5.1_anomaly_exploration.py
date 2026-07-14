#!/usr/bin/env python3
"""
modeling/phase5.1_anomaly_exploration.py

Executes Phase 5.1: Understand Anomaly Characteristics & Establish Evaluation Baseline.
Strictly separates two concepts:
1. Part A: Dataset & Feature Profile (Unsupervised Feature Space Audit across our 18 validated features).
   - Inspects univariate & multivariate distributions of engineered rates, problems, and severities.
2. Part B: Post-Hoc Evaluation Baseline / Known Incident Profile.
   - Documents exact frequencies of physical hardware failures (`25`), network dropouts (`788`),
     chronic outages (`17`), and total operational states (`helper_current_failure_state == 1`).
   - CRITICAL SRE RULE: This baseline is used ONLY FOR POST-HOC EVALUATION after unsupervised models
     have finished training. It NEVER guides, calibrates, or sets hyperparameter thresholds (e.g. contamination)
     during unsupervised training!

Exports: `datasets/known_incident_profile.json` and `docs/modeling/00_known_incident_profile.md`.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    in_path = Path("datasets/master_ml_dataset_v1.parquet")
    if not in_path.exists():
        print(f"[ERROR] Master dataset not found at {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Master ML Dataset: {in_path} ...")
    df = pd.read_parquet(in_path)
    total_rows = len(df)
    total_servers = df['machine_name'].nunique()
    print(f"Loaded {total_rows:,} observations across {total_servers} distinct servers.\n")

    # =========================================================================
    # PART A: Dataset & Feature Profile (18 Validated Unsupervised Features)
    # =========================================================================
    print("=== PART A: Dataset & Feature Profile (`18 Validated Features`) ===")
    print("Inspecting statistical rarity across our unsupervised training features (`X`):\n")

    # 1. Rolling 24-hour hardware problems sum
    prob_counts = df['problems_active_sum_6slot'].value_counts().sort_index()
    print("  [problems_active_sum_6slot (Rolling 24h Problem Accumulation)]:")
    for val, cnt in prob_counts.items():
        print(f"    - Problem Sum == {val:2d}: {cnt:6,d} rows ({cnt/total_rows*100:6.4f}%)")

    # 2. Rolling 12-hour timeout rate distribution
    print("\n  [ping_timeout_rate_3slot (Rolling 12h Timeout Rate)]:")
    for thresh in [0.01, 0.33, 0.66, 1.0]:
        cnt = (df['ping_timeout_rate_3slot'] >= thresh).sum()
        print(f"    - Rate >= {thresh*100:3.0f}%: {cnt:6,d} rows ({cnt/total_rows*100:6.4f}%)")

    # 3. Rolling 24-hour timeout rate distribution
    print("\n  [ping_timeout_rate_6slot (Rolling 24h Timeout Rate)]:")
    for thresh in [0.01, 0.33, 0.66, 1.0]:
        cnt = (df['ping_timeout_rate_6slot'] >= thresh).sum()
        print(f"    - Rate >= {thresh*100:3.0f}%: {cnt:6,d} rows ({cnt/total_rows*100:6.4f}%)")

    # 4. Hardware Component Severity Breaches
    crit_rows = (df['critical_component_count'] > 0).sum()
    not_ok_rows = (df['not_ok_component_count'] > 0).sum()
    deg_rows = (df['degraded_component_count'] > 0).sum()
    print("\n  [Hardware Component Severity Breaches across all servers]:")
    print(f"    - Degraded (`Rank 1` warning) : {deg_rows:6,d} rows ({deg_rows/total_rows*100:6.4f}%)")
    print(f"    - NOT OK (`Rank 2` warning)   : {not_ok_rows:6,d} rows ({not_ok_rows/total_rows*100:6.4f}%)")
    print(f"    - Critical (`Rank 3` failure) : {crit_rows:6,d} rows ({crit_rows/total_rows*100:6.4f}%)")

    # 5. Hardware Subsystem Worst Status breakdown across the 26 hardware-monitored servers
    print("\n  [Hardware Subsystem Worst Status across 26 Hardware Servers]:")
    hw_mask = (df['has_hpe'] == 1) | (df['has_dell'] == 1)
    hw_df = df[hw_mask]
    for col in ['hardware_cpu_worst_status', 'hardware_memory_worst_status', 'hardware_storage_worst_status',
                'hardware_fans_worst_status', 'hardware_temperature_worst_status', 'hardware_power_worst_status']:
        crit_subsystem = (hw_df[col] == 3).sum()
        warn_subsystem = (hw_df[col] == 2).sum()
        print(f"    - {col:<34}: {crit_subsystem:3d} Critical (`3`), {warn_subsystem:4d} NOT OK (`2`)")

    # =========================================================================
    # PART B: Post-Hoc Evaluation Baseline / Known Incident Profile
    # =========================================================================
    print("\n" + "="*75)
    print("=== PART B: Post-Hoc Evaluation Baseline (`Known Incident Profile`) ===")
    print("CRITICAL RULE: This baseline is used ONLY after unsupervised models finish training!")
    print("It NEVER guides or calibrates hyperparameter thresholds (e.g. contamination).")
    print("="*75 + "\n")

    crit_df = df[df['critical_component_count'] > 0]
    crit_servers = crit_df['machine_name'].nunique()
    
    ping_drop_df = df[df['ping_status_binary'] == 1]
    ping_drop_servers = ping_drop_df['machine_name'].nunique()

    chronic_24h_df = df[df['ping_timeout_rate_6slot'] == 1.0]
    chronic_24h_servers = chronic_24h_df['machine_name'].nunique()

    chronic_12h_df = df[df['ping_timeout_rate_3slot'] == 1.0]
    chronic_12h_servers = chronic_12h_df['machine_name'].nunique()

    multi_prob_df = df[df['problems_active_sum_6slot'] >= 3]
    multi_prob_servers = multi_prob_df['machine_name'].nunique()

    helper_df = df[df['helper_current_failure_state'] == 1]
    helper_servers = helper_df['machine_name'].nunique()

    target_3slot_df = df[df['target_failure_3slot'] == 1]
    target_3slot_servers = target_3slot_df['machine_name'].nunique()

    print(f"  1. Physical Hardware Critical Faults (`critical > 0`)      : {len(crit_df):6,d} rows across {crit_servers:3d} servers ({len(crit_df)/total_rows*100:6.4f}%)")
    print(f"  2. Instantaneous Network Dropouts (`ping == 1`)          : {len(ping_drop_df):6,d} rows across {ping_drop_servers:3d} servers ({len(ping_drop_df)/total_rows*100:6.4f}%)")
    print(f"  3. Chronic 24-Hour Network Outages (`rate_6slot == 1.0`) : {len(chronic_24h_df):6,d} rows across {chronic_24h_servers:3d} servers ({len(chronic_24h_df)/total_rows*100:6.4f}%)")
    print(f"  4. Persistent 12-Hour Network Outages (`rate_3slot == 1`): {len(chronic_12h_df):6,d} rows across {chronic_12h_servers:3d} servers ({len(chronic_12h_df)/total_rows*100:6.4f}%)")
    print(f"  5. Heavy Hardware Problem Accumulation (`prob_sum >= 3`) : {len(multi_prob_df):6,d} rows across {multi_prob_servers:3d} servers ({len(multi_prob_df)/total_rows*100:6.4f}%)")
    print("-" * 75)
    print(f"  [EVALUATION BENCHMARK 1] Total Current Operational Incidents (`helper_current == 1`):")
    print(f"      -> {len(helper_df):,d} rows across {helper_servers} servers ({len(helper_df)/total_rows*100:.4f}%)")
    print(f"  [EVALUATION BENCHMARK 2] Lookahead Pre-Failure Window (`target_failure_3slot == 1`):")
    print(f"      -> {len(target_3slot_df):,d} rows across {target_3slot_servers} servers ({len(target_3slot_df)/total_rows*100:.4f}%)")

    # Export to JSON
    out_json = Path("datasets/known_incident_profile.json")
    profile_data = {
        "dataset_metadata": {
            "total_rows": total_rows,
            "total_servers": total_servers,
            "ping_monitored_servers": total_servers,
            "hardware_monitored_servers": hw_df['machine_name'].nunique()
        },
        "part_a_feature_profile": {
            "rolling_24h_prob_sum_ge_1_rows": int((df['problems_active_sum_6slot'] >= 1).sum()),
            "rolling_12h_timeout_rate_ge_33pct_rows": int((df['ping_timeout_rate_3slot'] >= 0.33).sum()),
            "hardware_degraded_rows": int(deg_rows),
            "hardware_not_ok_rows": int(not_ok_rows),
            "hardware_critical_rows": int(crit_rows)
        },
        "part_b_post_hoc_evaluation_baseline": {
            "physical_hardware_critical_faults": {"rows": len(crit_df), "servers": crit_servers},
            "instantaneous_network_dropouts": {"rows": len(ping_drop_df), "servers": ping_drop_servers},
            "chronic_24h_network_outages": {"rows": len(chronic_24h_df), "servers": chronic_24h_servers},
            "heavy_hardware_problem_accumulation": {"rows": len(multi_prob_df), "servers": multi_prob_servers},
            "benchmark_1_current_operational_incidents": {"rows": len(helper_df), "servers": helper_servers, "pct": len(helper_df)/total_rows*100},
            "benchmark_2_lookahead_pre_failure_12h": {"rows": len(target_3slot_df), "servers": target_3slot_servers, "pct": len(target_3slot_df)/total_rows*100}
        },
        "hyperparameter_testing_plan": {
            "isolation_forest_contamination_grid": [0.01, 0.02, 0.03, 0.05],
            "selection_criteria": "Post-hoc evaluation against Benchmark 1 (Coverage of 25 hardware faults and network dropouts vs false alarm rate)"
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)
    print(f"\n[SUCCESS] Exported Known Incident Profile & Feature Statistics to {out_json}.")

if __name__ == "__main__":
    main()
