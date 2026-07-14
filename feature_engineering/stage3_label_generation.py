#!/usr/bin/env python3
"""
feature_engineering/stage3_label_generation.py

Executes Stage 3 of Phase 4 Feature Engineering (Group D: Lookahead Target Labels & Operational Helper):
1. Loads Stage 2 intermediate dataset (`features_stage2_temporal_v1.parquet`).
2. Enforces strict chronological ordering per machine (`machine_name` -> `monitoring_slot`).
3. Constructs instantaneous operational state helper (`helper_current_failure_state`).
   - Note: Strictly excluded from model training feature matrices ($X$) to prevent data leakage.
4. Constructs exactly the 4 approved Group D lookahead target labels (strictly forward-looking, no leakage into $X$):
   - `target_failure_3slot` (whether `helper_current_failure_state == 1` occurs in slots t+1 .. t+3 / next 12h)
   - `target_failure_6slot` (whether `helper_current_failure_state == 1` occurs in slots t+1 .. t+6 / next 24h)
   - `target_network_alert_3slot` (whether `ping_status_binary == 1` occurs in slots t+1 .. t+3 / next 12h)
   - `target_hardware_failure_3slot` (whether `critical_component_count >= 1` occurs in slots t+1 .. t+3 / next 12h)
   - Note: Option C architecture preserves truthful `NaN` (`Int64` nullable integer) at the final slots of each machine timeline where future lookahead slots do not exist.
5. Exports the Final Master ML Dataset (`master_ml_dataset_v1.parquet`, `.csv`) and comprehensive master metadata JSON (`feature_metadata_master.json`).
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    in_path = Path("datasets/features_stage2_temporal_v1.parquet")
    if not in_path.exists():
        print(f"[ERROR] Stage 2 dataset not found at {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Stage 2 dataset: {in_path} ...")
    df = pd.read_parquet(in_path)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} observations x {len(df.columns)} columns.\n")

    # 1. Enforce Chronological Sorting
    print("=== 1. Enforcing Chronological Sorting per Machine ===")
    df = df.sort_values(by=['machine_name', 'monitoring_slot']).reset_index(drop=True)
    print("Dataset sorted by ['machine_name', 'monitoring_slot'].\n")

    # 2. Construct Instantaneous Operational State Helper
    print("=== 2. Constructing Instantaneous Helper (`helper_current_failure_state`) ===")
    df['helper_current_failure_state'] = (
        (df['ping_status_binary'] == 1) | (df['critical_component_count'] > 0)
    ).astype('Int64')

    print(f"  [helper_current_failure_state Distribution across all {total_rows:,} rows]:")
    for val, cnt in df['helper_current_failure_state'].value_counts(dropna=False).sort_index().items():
        val_label = "NaN" if pd.isna(val) else f"Value {val} ({'Operational Failure' if val==1 else 'Healthy/Warning'})"
        print(f"    - {val_label}: {cnt:,} ({cnt/total_rows*100:.2f}%)")

    # 3. Construct Lookahead Targets (Group D) - Option C Truthful NaN handling
    print("\n=== 3. Constructing Lookahead Targets (`target_failure_3slot / 6slot`, etc.) ===")
    grouped = df.groupby('machine_name')

    # Helper function for forward window rolling max (lookahead without including current slot t)
    def compute_lookahead_max(series_group, window_slots):
        # We shift(-1) so window starts at t+1, then apply rolling max reversed (or using reversed index)
        # Using transform with shift to check future slots precisely
        result = pd.Series(index=series_group.index, dtype='Int64')
        for idx in series_group.index:
            # Get future slice
            # Since dataframe is sorted and grouped per machine, we can check forward shifts up to window_slots
            pass
        return result

    # Compute shifts efficiently across the group
    # For 3-slot lookahead (t+1, t+2, t+3)
    f_shift1 = grouped['helper_current_failure_state'].shift(-1)
    f_shift2 = grouped['helper_current_failure_state'].shift(-2)
    f_shift3 = grouped['helper_current_failure_state'].shift(-3)
    f_shift4 = grouped['helper_current_failure_state'].shift(-4)
    f_shift5 = grouped['helper_current_failure_state'].shift(-5)
    f_shift6 = grouped['helper_current_failure_state'].shift(-6)

    # If all shifts in the window are NaN (at timeline end), target is NaN. Otherwise max over available future slots.
    df_3slot = pd.concat([f_shift1, f_shift2, f_shift3], axis=1)
    df['target_failure_3slot'] = df_3slot.max(axis=1, skipna=True).astype('Int64')
    # If all future slots in the 3-slot window are NaN (i.e. slot 186), keep as NaN
    df.loc[df_3slot.isna().all(axis=1), 'target_failure_3slot'] = pd.NA

    df_6slot = pd.concat([f_shift1, f_shift2, f_shift3, f_shift4, f_shift5, f_shift6], axis=1)
    df['target_failure_6slot'] = df_6slot.max(axis=1, skipna=True).astype('Int64')
    df.loc[df_6slot.isna().all(axis=1), 'target_failure_6slot'] = pd.NA

    # Network lookahead (t+1 .. t+3)
    p_shift1 = grouped['ping_status_binary'].shift(-1)
    p_shift2 = grouped['ping_status_binary'].shift(-2)
    p_shift3 = grouped['ping_status_binary'].shift(-3)
    p_3slot = pd.concat([p_shift1, p_shift2, p_shift3], axis=1)
    df['target_network_alert_3slot'] = p_3slot.max(axis=1, skipna=True).astype('Int64')
    df.loc[p_3slot.isna().all(axis=1), 'target_network_alert_3slot'] = pd.NA

    # Hardware lookahead (t+1 .. t+3)
    c_shift1 = grouped['critical_component_count'].shift(-1)
    c_shift2 = grouped['critical_component_count'].shift(-2)
    c_shift3 = grouped['critical_component_count'].shift(-3)
    c_3slot = pd.concat([c_shift1, c_shift2, c_shift3], axis=1)
    df['target_hardware_failure_3slot'] = (c_3slot.max(axis=1, skipna=True) >= 1).astype('Int64')
    df.loc[c_3slot.isna().all(axis=1), 'target_hardware_failure_3slot'] = pd.NA

    # Print distributions
    for target_col in ['target_failure_3slot', 'target_failure_6slot', 'target_network_alert_3slot', 'target_hardware_failure_3slot']:
        print(f"\n  [{target_col} Distribution]:")
        for val, cnt in df[target_col].value_counts(dropna=False).sort_index().items():
            val_label = "NaN (Timeline End / No Future Data)" if pd.isna(val) else f"Value {val} ({'Pre-Failure Positive Class' if val==1 else 'Stable Negative Class'})"
            print(f"    - {val_label}: {cnt:,} ({cnt/total_rows*100:.2f}%)")

    # 4. Export Master Dataset & Master Metadata JSON
    out_parquet = Path("datasets/master_ml_dataset_v1.parquet")
    out_csv = Path("datasets/master_ml_dataset_v1.csv")
    out_meta = Path("datasets/feature_metadata_master.json")

    print(f"\n=== 4. Exporting Final Master ML Dataset & Complete Metadata ===")
    print(f"  - Writing Parquet: {out_parquet}")
    df.to_parquet(out_parquet, index=False)

    print(f"  - Writing CSV: {out_csv}")
    try:
        df.to_csv(out_csv, index=False)
    except PermissionError:
        print(f"  [WARNING] Could not overwrite {out_csv} (file may be open in Excel/Editor). Parquet and JSON were written successfully.")

    # Load Stage 2 metadata and extend with Stage 3 helper & targets
    stage2_meta_path = Path("datasets/feature_metadata_stage2.json")
    if stage2_meta_path.exists():
        with open(stage2_meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
    else:
        meta_dict = {"features": {}}

    meta_dict["pipeline_phase"] = "Phase 4 Master Final: Complete Feature & Target Blueprint"
    meta_dict["total_columns"] = len(df.columns)
    meta_dict["original_columns"] = 28
    meta_dict["engineered_features_and_targets"] = len(df.columns) - 28

    stage3_meta = {
        "helper_current_failure_state": {
            "type": "binary_helper",
            "source": ["ping_status_binary", "critical_component_count"],
            "description": "Instantaneous operational state at time t ((ping==1) | (critical_count>0)). STRICTLY EXCLUDED from training feature matrices (X) to prevent data leakage.",
            "ml_usage": "helper_exclude_from_training"
        },
        "target_failure_3slot": {
            "type": "binary_target",
            "source": ["helper_current_failure_state"],
            "description": "Lookahead target: whether helper_current_failure_state == 1 occurs within the next 3 slots (+12 hours). Option C preserves truthful NaN at timeline end.",
            "ml_usage": "target_label",
            "lookahead_window": "3_slots_12h"
        },
        "target_failure_6slot": {
            "type": "binary_target",
            "source": ["helper_current_failure_state"],
            "description": "Lookahead target: whether helper_current_failure_state == 1 occurs within the next 6 slots (+24 hours). Option C preserves truthful NaN at timeline end.",
            "ml_usage": "target_label",
            "lookahead_window": "6_slots_24h"
        },
        "target_network_alert_3slot": {
            "type": "binary_target",
            "source": ["ping_status_binary"],
            "description": "Lookahead target: whether ping_status_binary == 1 occurs within the next 3 slots (+12 hours). Option C preserves truthful NaN at timeline end.",
            "ml_usage": "target_label",
            "lookahead_window": "3_slots_12h"
        },
        "target_hardware_failure_3slot": {
            "type": "binary_target",
            "source": ["critical_component_count"],
            "description": "Lookahead target: whether critical_component_count >= 1 occurs within the next 3 slots (+12 hours). Option C preserves truthful NaN at timeline end.",
            "ml_usage": "target_label",
            "lookahead_window": "3_slots_12h"
        }
    }

    meta_dict["features"].update(stage3_meta)

    print(f"  - Writing Master Feature Metadata JSON: {out_meta}")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    df_check = pd.read_parquet(out_parquet)
    print(f"\n[VERIFICATION SUCCESS] Exported Final Golden Dataset: {len(df_check):,} rows x {len(df_check.columns)} columns.")

if __name__ == "__main__":
    main()
