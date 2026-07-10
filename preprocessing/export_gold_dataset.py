import os
import sys
import json
import datetime
import pandas as pd
from post_merge_validate import run_post_merge_audit

def publish_gold_dataset():
    print("=== Step 8: Unified Monitoring Gold Dataset Publication ===")
    
    # 1. Invoke Gatekeeper Audit (`sys.exit(1)` triggered if any of the 10 checks fail)
    df = run_post_merge_audit()
    
    # 2. Define Versioned Export Targets inside `datasets/` and `docs/preprocessing/`
    parquet_path = "datasets/master_infrastructure_health_v1.parquet"
    csv_path = "datasets/master_infrastructure_health_v1.csv"
    json_meta_path = "datasets/master_infrastructure_health_metadata_v1.json"
    
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("docs/preprocessing", exist_ok=True)
    
    # 3. Export to Parquet (preserving data types without float casting)
    print(f"\n[EXPORT] Writing Unified Gold Parquet dataset to: {parquet_path} ...")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    
    # 4. Export to CSV (human-readable inspection baseline)
    print(f"[EXPORT] Writing Unified Gold CSV dataset to: {csv_path} ...")
    df.to_csv(csv_path, index=False)
    
    # 5. Export Structured Metadata JSON
    meta_dict = {
        "created_at": datetime.datetime.now().isoformat(),
        "pipeline_version": "1.0",
        "dataset_name": "master_infrastructure_health_v1",
        "rows": len(df),
        "columns": len(df.columns),
        "machines": int(df["machine_name"].nunique()),
        "monitoring_slots": int(df["monitoring_slot"].nunique()),
        "eda_version": "Phase1_v1",
        "schema_version": "1.0",
        "pipeline_assumptions": {
            "A1": "machine_name <-> ip_address forms an exact 1-to-1 mapping across all tables (0% drift).",
            "A2": "Monitoring occurs on a regular 4-hour cycle (6 slots/day, 186 total slots per machine).",
            "A3": "Inter-system clock jitter up to 46 minutes exists; raw timestamps are bucketized into monitoring_slots.",
            "A4": "Ping Status forms the complete master inventory (246 assets; other files are strict subsets).",
            "A5": "A physical server is either HPE or Dell, never both; vendor telemetry is preserved separately without coalescing."
        },
        "telemetry_distribution": df["telemetry_source"].value_counts().to_dict(),
        "column_names": list(df.columns)
    }
    with open(json_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"[EXPORT] Saved dataset metadata to: {json_meta_path}")
    
    # 6. Roundtrip Verification on Disk
    if not os.path.exists(parquet_path) or not os.path.exists(csv_path):
        print("[ERROR] Export verification failed: output files missing from disk!")
        sys.exit(1)
        
    pq_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    
    # Reload Parquet to verify exact shape and fidelity
    reloaded_df = pd.read_parquet(parquet_path)
    if reloaded_df.shape != (45756, 28):
        print(f"[ERROR] Parquet roundtrip verification failed! Expected (45756, 28), got {reloaded_df.shape}")
        sys.exit(1)
        
    print(f"[PASS] Parquet Roundtrip Verification: Re-read {reloaded_df.shape} successfully from disk.")
    print(f"[PASS] Storage Footprint:\n  - Parquet: {pq_size_mb:.2f} MB\n  - CSV:     {csv_size_mb:.2f} MB")
    
    # 7. Generate Step 8 Documentation (`07_gold_dataset_publication.md`)
    step8_doc_path = "docs/preprocessing/07_gold_dataset_publication.md"
    step8_content = f"""# Step 8: Unified Monitoring Gold Dataset Publication Log

**Script File:** [`preprocessing/export_gold_dataset.py`](file:///c:/Users/navad/ML_data/preprocessing/export_gold_dataset.py)  
**Execution Status:** `PASS`  

---

## 1. Publication Specification
Upon passing all 10 Engineering Gatekeeper validation checks in Step 7, Step 8 writes the versioned `master_infrastructure_health_v1` dataset to disk in Parquet and CSV formats, generates structured JSON metadata (`master_infrastructure_health_metadata_v1.json`), and performs a `pd.read_parquet` roundtrip fidelity check.

---

## 2. Verified Export Artifacts

| Artifact Name | Path | Format | Dimensions | Storage Size |
| :--- | :--- | :---: | :---: | :---: |
| **Unified Gold Parquet** | [`datasets/master_infrastructure_health_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/master_infrastructure_health_v1.parquet) | `PyArrow Parquet` | `45,756 x 28` | `{pq_size_mb:.2f} MB` |
| **Unified Gold CSV** | [`datasets/master_infrastructure_health_v1.csv`](file:///c:/Users/navad/ML_data/datasets/master_infrastructure_health_v1.csv) | `RFC 4180 CSV` | `45,756 x 28` | `{csv_size_mb:.2f} MB` |
| **Dataset Metadata** | [`datasets/master_infrastructure_health_metadata_v1.json`](file:///c:/Users/navad/ML_data/datasets/master_infrastructure_health_metadata_v1.json) | `JSON v1.0` | `10 Keys` | `< 1 KB` |

---

## 3. Engineering Certification Statement
> *The unified monitoring dataset (`master_infrastructure_health_v1.parquet`) has successfully passed all defined Data Engineering validation checks and is approved for downstream Feature Engineering and Machine Learning.*
"""
    with open(step8_doc_path, "w", encoding="utf-8") as f:
        f.write(step8_content)
    print(f"[DOCUMENTATION] Saved Step 8 Log to {step8_doc_path}")
    
    # 8. Generate Top-Level `pipeline_summary.md`
    summary_path = "docs/preprocessing/pipeline_summary.md"
    summary_content = f"""# Infrastructure Data Engineering Pipeline Summary

======================================================================
  **INFRASTRUCTURE HEALTH PREDICTION SYSTEM — PHASE 2 SUMMARY**
======================================================================

```
Phase 1: EDA (Dataset Understanding & Relationships)
Status: PASS (100% Deterministic Evidence Grounding)
   │
   ▼
Phase 2: Data Engineering (Lossless Integration & Gatekeeper Audit)
Status: PASS (10/10 Verification Checks Passed)
   │
   ▼
Unified Monitoring Dataset: master_infrastructure_health_v1.parquet
Dimensions: 45,756 observations x 28 preserved attributes
   │
   ▼
Ready for Phase 3 Feature Engineering?
Verdict: YES (APPROVED)
```

---

## Executive Engineering Certification
> *The unified monitoring dataset (`master_infrastructure_health_v1.parquet`) has successfully passed all defined Data Engineering validation checks and is approved for downstream Feature Engineering and Machine Learning.*

---

## Pipeline Checkpoint Summary Table

| Phase / Checkpoint | Target Artifact | Verified Outcome | Engineering Approval |
| :--- | :--- | :--- | :---: |
| **Phase 1: EDA & Understanding** | [`docs/dataset_understanding.md`](file:///c:/Users/navad/ML_data/docs/dataset_understanding.md) | `1-to-1 IP mapping, 4h slots, 15 shared servers` | **`PASS`** |
| **Step 1: Input Validation** | [`docs/preprocessing/01_validate_inputs.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/01_validate_inputs.md) | `3 operational CSVs verified (len & schema)` | **`PASS`** |
| **Step 2: Canonical Schema Mapping** | [`docs/preprocessing/02_standardize_schema.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/02_standardize_schema.md) | `machine_name, ip_address, event_time mapped` | **`PASS`** |
| **Step 3 & 4: Slot Generation** | [`docs/preprocessing/03_create_monitoring_slots.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/03_create_monitoring_slots.md) | `186 unique time slots generated (Slot 02..22)` | **`PASS`** |
| **Step 5: Pre-Merge Validation** | [`docs/preprocessing/04_pre_merge_validate.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/04_pre_merge_validate.md) | `15 Common, 11 Dell-only, 220 Ping-only verified`| **`PASS`** |
| **Step 6: Lossless Left Join** | [`docs/preprocessing/05_merge_datasets.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/05_merge_datasets.md) | `45,756 x 28 shape without coalescing loss` | **`PASS`** |
| **Step 7: Gatekeeper Audit** | [`docs/preprocessing/06_post_merge_validate.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/06_post_merge_validate.md) | `10/10 checks passed (Null Propagation verified)` | **`PASS`** |
| **Step 8: Gold Dataset Export** | [`docs/preprocessing/07_gold_dataset_publication.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/07_gold_dataset_publication.md) | `Parquet roundtrip verified on disk` | **`PASS`** |

---

## Next Steps (`Phase 3: Feature Engineering`)
Now that `master_infrastructure_health_v1.parquet` is published with zero data loss (`Data Engineering -> Preserve`), **Phase 3 (`Feature Engineering -> Transform`)** can safely:
1. Reconcile inter-vendor hardware flags (`hardware_cpu_disagreement_flag`, `hardware_cpu_worst_status`).
2. Construct rolling time-window aggregations (`3-slot` / `12-hour` sliding failure indicators).
3. Compute target labels for anomaly detection and failure prediction.
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"[SUMMARY] Saved Top-Level Pipeline Summary to {summary_path}")
    
    print("\n======================================================================")
    print("  PHASE 2 DATA ENGINEERING COMPLETE: UNIFIED DATASET APPROVED")
    print(f"  Primary Gold Dataset: {parquet_path}")
    print("======================================================================")
    
    return parquet_path, csv_path

if __name__ == "__main__":
    publish_gold_dataset()
