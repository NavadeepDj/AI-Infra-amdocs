# Data Engineering Pipeline: Building the Unified Monitoring Gold Dataset

This directory (`docs/preprocessing/`) contains the modular documentation, canonical schemas, pipeline assumptions, and verified execution results for every individual script in the Phase 2 Data Engineering pipeline.

---

## Preprocessing Pipeline Architecture

```
Raw CSVs (datasets/)
   │
   ▼
[Step 1: Input Validation] ──► 01_validate_inputs.md (Row counts, existence, missing/unexpected columns)
   │
   ▼
[Step 2: Canonical Schema Mapping] ──► 02_standardize_schema.md (Unified domain field mappings)
   │
   ▼
[Step 3: Normalize Datetime Formats] ──► 03_create_monitoring_slots.md (Parse mixed date formats)
   │
   ▼
[Step 4: Monitoring Slot Generation] ──► 03_create_monitoring_slots.md (4-hour cycle algorithm & uniqueness)
   │
   ▼
[Step 5: Pre-Merge Validation] ──► 04_pre_merge_validate.md (Subset checks, machine counts, discrete values)
   │
   ▼
[Step 6: Left Outer Joining] ──► 05_merge_datasets.md (Coalescing telemetry + immutable observation_id)
   │
   ▼
[Step 7: Post-Merge Validation] ──► 06_post_merge_validate.md (5-Point PASS/FAIL Audit)
   │
   ▼
[Step 8: Gold Dataset Publication] ──► 07_gold_dataset_publication.md (Gold Parquet Schema & Certification)
```

---

## Pipeline Assumptions & Evidence Grounding

Every relational decision in this pipeline is anchored in our deterministic EDA findings:

| # | Engineering Assumption | Evidence Basis | Confidence | Impact on Pipeline |
| :---: | :--- | :--- | :---: | :--- |
| **A1** | `machine_name` $\leftrightarrow$ `ip_address` forms an exact 1-to-1 mapping across all tables. | EDA Step 24 (`0%` multi-IP drift observed across 31 days). | **100%** | Enables dual-key validation and safe joining across network and hardware layers. |
| **A2** | Monitoring occurs on a regular 4-hour cycle (`6` slots/day, `186` total per machine). | EDA Step 38 (`45,756 / 246 = 186.0` exact observations for every Ping machine). | **100%** | Allows us to define a fixed time grid (`Slot 02..22`) instead of relying on irregular timestamps. |
| **A3** | Inter-system clock jitter up to 46 minutes exists (`02:00` vs `02:24` vs `02:46`). | EDA Step 32 (HPE average offset `+24m`, Dell `+46m`). | **100%** | Raw timestamps **cannot** be used as join keys. We must bucketize timestamps into canonical `monitoring_slot`s. |
| **A4** | Ping Status forms the complete master inventory (`246` assets). | EDA Step 15 (`15` HPE and `26` Dell servers are strict subsets of Ping's `246` IPs). | **100%** | Requires `ping_status` to act as the primary `LEFT JOIN` anchor so zero assets are dropped. |
| **A5** | A physical server is either HPE or Dell, never both. | Domain law & EDA (`0%` IP overlap between `hpe_ilo` and `dell_idrac`). | **100%** | Allows clean coalescing of common component columns (`cpu`, `memory`, etc.) without collision. |

---

## Modular Script Documentation Index

| Step | Script File | Documentation File | Status |
| :---: | :--- | :--- | :---: |
| **1** | `validate_inputs.py` | [`01_validate_inputs.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/01_validate_inputs.md) | `PASS` |
| **2** | `standardize_schema.py` | [`02_standardize_schema.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/02_standardize_schema.md) | `PASS` |
| **3 & 4** | `create_monitoring_slots.py` | [`03_create_monitoring_slots.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/03_create_monitoring_slots.md) | `PASS` |
| **5** | `pre_merge_validate.py` | [`04_pre_merge_validate.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/04_pre_merge_validate.md) | `PENDING` |
| **6** | `merge_datasets.py` | [`05_merge_datasets.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/05_merge_datasets.md) | `PENDING` |
| **7** | `post_merge_validate.py` | [`06_post_merge_validate.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/06_post_merge_validate.md) | `PENDING` |
| **8** | `export_gold_dataset.py` | [`07_gold_dataset_publication.md`](file:///c:/Users/navad/ML_data/docs/preprocessing/07_gold_dataset_publication.md) | `PENDING` |
