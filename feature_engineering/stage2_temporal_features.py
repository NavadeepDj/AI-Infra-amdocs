#!/usr/bin/env python3
"""
feature_engineering/stage2_temporal_features.py

Executes Stage 2 of Phase 4 Feature Engineering (Group C: Empirically Justified Temporal Features):
1. Loads Stage 1 intermediate dataset (features_stage1_generic_v1.parquet).
2. Enforces strict chronological ordering per machine (machine_name -> monitoring_slot).
3. Constructs exactly the 5 approved Group C temporal features (strictly backward-looking, no leakage):
   - ping_status_binary_lag1 (Option C: preserved as truthful NaN at Slot 1)
   - ping_status_binary_lag2 (Option C: preserved as truthful NaN at Slot 1 & 2)
   - ping_timeout_rate_3slot (12-hour rolling mean of ping_status_binary)
   - ping_timeout_rate_6slot (24-hour rolling mean of ping_status_binary)
   - problems_active_sum_6slot (24-hour rolling sum of has_active_problem)
4. Exports intermediate Stage 2 dataset and updated metadata JSON:
   - datasets/features_stage2_temporal_v1.parquet
   - datasets/features_stage2_temporal_v1.csv (handled safely if open/locked)
   - datasets/feature_metadata_stage2.json
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    in_path = Path("datasets/features_stage1_generic_v1.parquet")
    if not in_path.exists():
        print(f"[ERROR] Stage 1 dataset not found at {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Stage 1 dataset: {in_path} ...")
    df = pd.read_parquet(in_path)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} observations x {len(df.columns)} columns.\n")

    # 1. Enforce Chronological Sorting
    print("=== 1. Enforcing Chronological Sorting per Machine ===")
    df = df.sort_values(by=['machine_name', 'monitoring_slot']).reset_index(drop=True)
    print("Dataset sorted by ['machine_name', 'monitoring_slot'].\n")

    # 2. Construct Lags (Ping Status Binary) - Option C Pure Truthful NaN
    print("=== 2. Constructing Backward Lags (ping_status_binary_lag1 / lag2) ===")
    grouped = df.groupby('machine_name')
    
    # Lag 1 (-4 hours) and Lag 2 (-8 hours), strictly preserving NaN where history does not exist (Slot 1 / Slot 2)
    df['ping_status_binary_lag1'] = grouped['ping_status_binary'].shift(1).astype('Int64')
    df['ping_status_binary_lag2'] = grouped['ping_status_binary'].shift(2).astype('Int64')

    print(f"  [ping_status_binary_lag1 Distribution (including truthful NaNs at Slot 1)]:")
    for val, cnt in df['ping_status_binary_lag1'].value_counts(dropna=False).sort_index().items():
        val_label = "NaN (Slot 1 / No History)" if pd.isna(val) else f"Value {val}"
        print(f"    - {val_label}: {cnt:,} ({cnt/total_rows*100:.2f}%)")

    print(f"\n  [ping_status_binary_lag2 Distribution (including truthful NaNs at Slot 1 & 2)]:")
    for val, cnt in df['ping_status_binary_lag2'].value_counts(dropna=False).sort_index().items():
        val_label = "NaN (Slot 1-2 / No History)" if pd.isna(val) else f"Value {val}"
        print(f"    - {val_label}: {cnt:,} ({cnt/total_rows*100:.2f}%)")

    # 3. Construct Rolling Means (Ping Timeout Rates)
    print("\n=== 3. Constructing Rolling Windows (ping_timeout_rate_3slot / 6slot) ===")
    df['ping_timeout_rate_3slot'] = grouped['ping_status_binary'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['ping_timeout_rate_6slot'] = grouped['ping_status_binary'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean()
    )

    print(f"  - ping_timeout_rate_3slot (12h): min={df['ping_timeout_rate_3slot'].min():.4f}, max={df['ping_timeout_rate_3slot'].max():.4f}, mean={df['ping_timeout_rate_3slot'].mean():.4f}")
    print(f"  - ping_timeout_rate_6slot (24h): min={df['ping_timeout_rate_6slot'].min():.4f}, max={df['ping_timeout_rate_6slot'].max():.4f}, mean={df['ping_timeout_rate_6slot'].mean():.4f}")

    # 4. Construct Rolling Sum (Active Problem Duration)
    print("\n=== 4. Constructing Rolling Sum (problems_active_sum_6slot) ===")
    df['problems_active_sum_6slot'] = grouped['has_active_problem'].transform(
        lambda x: x.rolling(window=6, min_periods=1).sum()
    ).astype('Int64')

    print(f"  [problems_active_sum_6slot Distribution (24h problem duration out of 6 slots)]:")
    for val, cnt in df['problems_active_sum_6slot'].value_counts().sort_index().items():
        pct = (cnt / total_rows) * 100
        pct_str = f"{pct:.2f}%" if pct >= 0.01 or pct == 0 else "<0.01%"
        print(f"    - {val} slots active: {cnt:,} ({pct_str})")

    # 5. Export Stage 2 Dataset & Updated Metadata JSON
    out_parquet = Path("datasets/features_stage2_temporal_v1.parquet")
    out_csv = Path("datasets/features_stage2_temporal_v1.csv")
    out_meta = Path("datasets/feature_metadata_stage2.json")

    print(f"\n=== 5. Exporting Stage 2 Temporal Features & Metadata ===")
    print(f"  - Writing Parquet: {out_parquet}")
    df.to_parquet(out_parquet, index=False)

    print(f"  - Writing CSV: {out_csv}")
    try:
        df.to_csv(out_csv, index=False)
    except PermissionError:
        print(f"  [WARNING] Could not overwrite {out_csv} (file may be open in Excel/Editor). Parquet and JSON were written successfully.")

    stage1_meta_path = Path("datasets/feature_metadata_stage1.json")
    if stage1_meta_path.exists():
        with open(stage1_meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
    else:
        meta_dict = {"features": {}}

    meta_dict["pipeline_phase"] = "Phase 4 Stage 2: Temporal Infrastructure Features (Option C Separation of Concerns)"
    meta_dict["total_columns"] = len(df.columns)
    meta_dict["original_columns"] = 28
    meta_dict["engineered_features"] = len(df.columns) - 28

    stage2_features = {
        "ping_status_binary_lag1": {
            "type": "binary",
            "source": ["ping_status_binary"],
            "description": "Network reachability in immediate preceding slot (-4h). Preserved as truthful NaN at Slot 1 (no prior history).",
            "ml_usage": "core_feature",
            "window": "lag1"
        },
        "ping_status_binary_lag2": {
            "type": "binary",
            "source": ["ping_status_binary"],
            "description": "Network reachability two slots ago (-8h). Preserved as truthful NaN at Slot 1 & 2. Identifies flapping with lag1.",
            "ml_usage": "core_feature",
            "window": "lag2"
        },
        "ping_timeout_rate_3slot": {
            "type": "continuous",
            "source": ["ping_status_binary"],
            "description": "Rolling mean of network timeouts over last 3 slots (12h). min_periods=1 evaluates available slots without generating NaNs.",
            "ml_usage": "core_feature",
            "window": "rolling_3slot_12h"
        },
        "ping_timeout_rate_6slot": {
            "type": "continuous",
            "source": ["ping_status_binary"],
            "description": "Rolling mean of network timeouts over last 6 slots (24h). min_periods=1 evaluates available slots without generating NaNs.",
            "ml_usage": "core_feature",
            "window": "rolling_6slot_24h"
        },
        "problems_active_sum_6slot": {
            "type": "count",
            "source": ["has_active_problem"],
            "description": "Rolling sum of active problem slots over last 6 slots (24h). Measures sustained daily instability.",
            "ml_usage": "core_feature",
            "window": "rolling_6slot_24h"
        }
    }

    meta_dict["features"].update(stage2_features)

    print(f"  - Writing Feature Metadata JSON: {out_meta}")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    df_check = pd.read_parquet(out_parquet)
    print(f"\n[VERIFICATION SUCCESS] Exported {len(df_check):,} rows x {len(df_check.columns)} columns (28 original + {len(df_check.columns)-28} engineered).")

if __name__ == "__main__":
    main()
