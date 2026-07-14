#!/usr/bin/env python3
"""
feature_engineering/stage1_generic_features.py

Executes Stage 1 of Phase 4 Feature Engineering:
1. Applies empirical ordinal encoding (SEVERITY_MAP: OK=0, Degraded=1, NOT OK=2, Critical=3).
2. Constructs canonical Generic Infrastructure Features:
   - hardware_{comp}_worst_status (for cpu, memory, fans, storage, temperature, power)
   - critical_component_count (count of Rank 3 components)
   - not_ok_component_count (count of Rank 2 components)
   - degraded_component_count (count of Rank 1 components)
   - has_active_problem (convenience dashboard indicator)
3. Constructs explainability & diagnostic flags (excluded from ML training):
   - hardware_{comp}_disagreement_flag
4. Exports intermediate dataset and machine-readable feature metadata:
   - datasets/features_stage1_generic_v1.parquet
   - datasets/features_stage1_generic_v1.csv
   - datasets/feature_metadata_stage1.json
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

SEVERITY_MAP = {
    "OK": 0,
    "Degraded": 1,
    "Warning": 2,      # Defensive mapping
    "NOT OK": 2,       # Option C empirical mapping (intermediate high-severity)
    "Critical": 3
}

PING_MAP = {
    "Reachable": 0,
    "Unreachable": 1
}

COMPONENTS = ['cpu', 'memory', 'fans', 'storage', 'temperature', 'power']

def main():
    gold_path = Path("datasets/master_infrastructure_health_v1.parquet")
    if not gold_path.exists():
        print(f"[ERROR] Gold dataset not found at {gold_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading gold dataset: {gold_path} ...")
    df = pd.read_parquet(gold_path)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} observations x {len(df.columns)} columns.\n")

    # 1. Apply Ordinal Encoding
    print("=== 1. Applying Ordinal Severity Mapping ===")
    df['ping_status_binary'] = df['ping_status'].map(PING_MAP).astype('Int64')

    for comp in COMPONENTS:
        hpe_col = f"hpe_{comp}"
        dell_col = f"dell_{comp}"
        hpe_rank_col = f"hpe_{comp}_rank"
        dell_rank_col = f"dell_{comp}_rank"
        
        if hpe_col in df.columns:
            df[hpe_rank_col] = df[hpe_col].map(SEVERITY_MAP).astype('Int64')
        else:
            df[hpe_rank_col] = pd.Series(dtype='Int64')
            
        if dell_col in df.columns:
            df[dell_rank_col] = df[dell_col].map(SEVERITY_MAP).astype('Int64')
        else:
            df[dell_rank_col] = pd.Series(dtype='Int64')

    # 2. Construct Canonical Worst Status (ML Training Feature)
    print("=== 2. Constructing `hardware_{comp}_worst_status` ===")
    for comp in COMPONENTS:
        hpe_rank_col = f"hpe_{comp}_rank"
        dell_rank_col = f"dell_{comp}_rank"
        worst_col = f"hardware_{comp}_worst_status"
        
        # Take max across hpe and dell ranks, preserving NaN for Ping-Only machines
        ranks_df = df[[hpe_rank_col, dell_rank_col]].copy()
        ranks_float = ranks_df.astype(float)
        worst_series = ranks_float.max(axis=1, skipna=True)
        
        df[worst_col] = pd.Series(worst_series, index=df.index, dtype='Int64')
        
        vc = df[worst_col].value_counts(dropna=False).sort_index()
        print(f"\n  [{worst_col} Distribution]:")
        for val, cnt in vc.items():
            pct = (cnt / total_rows) * 100
            pct_str = f"{pct:.2f}%" if pct >= 0.01 or pct == 0 else "<0.01%"
            val_label = "NaN (Ping-Only)" if pd.isna(val) else f"Rank {val}"
            print(f"    - {val_label}: {cnt:,} ({pct_str})")

    # 3. Construct Explainability Disagreement Flags (NOT for ML)
    print("\n=== 3. Constructing Explainability Disagreement Flags ===")
    for comp in COMPONENTS:
        hpe_col = f"hpe_{comp}"
        dell_col = f"dell_{comp}"
        flag_col = f"hardware_{comp}_disagreement_flag"
        
        disagree = (df['has_hpe'] & df['has_dell'] & (df[hpe_col] != df[dell_col]))
        df[flag_col] = disagree.astype(int)
        
        cnt = df[flag_col].sum()
        print(f"  - {flag_col}: {cnt:,} disagreements out of 2,790 dual-monitored rows ({cnt/2790*100:.2f}%)")

    # 4. Construct Aggregate Counts & Active Problem Flag
    print("\n=== 4. Constructing Granular Severity Counts ===")
    worst_cols = [f"hardware_{comp}_worst_status" for comp in COMPONENTS]
    
    # Granular severity counters
    df['critical_component_count'] = (df[worst_cols] == 3).sum(axis=1).astype('Int64')
    df['not_ok_component_count'] = (df[worst_cols] == 2).sum(axis=1).astype('Int64')
    df['degraded_component_count'] = (df[worst_cols] == 1).sum(axis=1).astype('Int64')
    
    # For Ping-Only machines (where all worst_cols are NaN), counts should be 0
    ping_only_mask = df[worst_cols].isna().all(axis=1)
    df.loc[ping_only_mask, 'critical_component_count'] = 0
    df.loc[ping_only_mask, 'not_ok_component_count'] = 0
    df.loc[ping_only_mask, 'degraded_component_count'] = 0

    df['has_active_problem'] = (
        (df['critical_component_count'] > 0) | 
        (df['not_ok_component_count'] > 0) | 
        (df['degraded_component_count'] > 0) | 
        (df['ping_status_binary'] == 1)
    ).astype(int)

    for count_col in ['critical_component_count', 'not_ok_component_count', 'degraded_component_count']:
        print(f"\n  [{count_col} Distribution]:")
        for val, cnt in df[count_col].value_counts().sort_index().items():
            pct = (cnt / total_rows) * 100
            pct_str = f"{pct:.2f}%" if pct >= 0.01 or pct == 0 else "<0.01%"
            print(f"    - {val} components: {cnt:,} ({pct_str})")

    print("\n  [has_active_problem Distribution]:")
    for val, cnt in df['has_active_problem'].value_counts().sort_index().items():
        label = "No Active Problem (All OK + Reachable)" if val == 0 else "Active Problem Detected"
        print(f"    - {label} ({val}): {cnt:,} ({cnt/total_rows*100:.2f}%)")

    # 5. Export Stage 1 Intermediate Dataset & Feature Metadata JSON
    out_parquet = Path("datasets/features_stage1_generic_v1.parquet")
    out_csv = Path("datasets/features_stage1_generic_v1.csv")
    out_meta = Path("datasets/feature_metadata_stage1.json")
    
    print(f"\n=== 5. Exporting Stage 1 Generic Features & Metadata ===")
    print(f"  - Writing Parquet: {out_parquet}")
    df.to_parquet(out_parquet, index=False)
    
    print(f"  - Writing CSV: {out_csv}")
    df.to_csv(out_csv, index=False)
    
    # Generate Feature Metadata Dictionary
    meta_dict = {
        "pipeline_phase": "Phase 4 Stage 1: Generic Infrastructure Features",
        "total_columns": len(df.columns),
        "original_columns": 28,
        "engineered_features": len(df.columns) - 28,
        "features": {
            "ping_status_binary": {
                "type": "binary",
                "source": ["ping_status"],
                "description": "Numeric encoding of network reachability (0=Reachable, 1=Unreachable).",
                "ml_usage": "core_feature"
            },
            "critical_component_count": {
                "type": "count",
                "source": [f"hardware_{comp}_worst_status" for comp in COMPONENTS],
                "description": "Count of hardware components simultaneously reporting Rank 3 (Critical).",
                "ml_usage": "core_feature"
            },
            "not_ok_component_count": {
                "type": "count",
                "source": [f"hardware_{comp}_worst_status" for comp in COMPONENTS],
                "description": "Count of hardware components simultaneously reporting Rank 2 (NOT OK / intermediate anomaly).",
                "ml_usage": "core_feature"
            },
            "degraded_component_count": {
                "type": "count",
                "source": [f"hardware_{comp}_worst_status" for comp in COMPONENTS],
                "description": "Count of hardware components simultaneously reporting Rank 1 (Degraded).",
                "ml_usage": "core_feature"
            },
            "has_active_problem": {
                "type": "binary",
                "source": ["critical_component_count", "not_ok_component_count", "degraded_component_count", "ping_status_binary"],
                "description": "Convenience dashboard alert flag (1 if any component > 0 or ping unreachable).",
                "ml_usage": "convenience_indicator"
            }
        }
    }
    
    for comp in COMPONENTS:
        meta_dict["features"][f"hardware_{comp}_worst_status"] = {
            "type": "ordinal",
            "source": [f"hpe_{comp}", f"dell_{comp}"],
            "description": f"Maximum available severity rank across HPE and Dell observations ({comp}), ignoring missing values (skipna=True). NaN for Ping-Only servers.",
            "ml_usage": "core_feature"
        }
        meta_dict["features"][f"hardware_{comp}_disagreement_flag"] = {
            "type": "flag",
            "source": [f"hpe_{comp}", f"dell_{comp}"],
            "description": f"Binary flag (1 if both HPE and Dell report {comp} and differ). Excluded from ML training to avoid learning mock-data generation noise.",
            "ml_usage": "explainability_only"
        }
        
    print(f"  - Writing Feature Metadata JSON: {out_meta}")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    df_check = pd.read_parquet(out_parquet)
    print(f"\n[VERIFICATION SUCCESS] Exported {len(df_check):,} rows x {len(df_check.columns)} columns (28 original + {len(df_check.columns)-28} engineered).")

if __name__ == "__main__":
    main()
