# Phase 4 Stage 1: Generic Infrastructure Feature Engineering Documentation

**Implementation Script:** [`feature_engineering/stage1_generic_features.py`](file:///c:/Users/navad/ML_data/feature_engineering/stage1_generic_features.py)  
**Input Dataset:** [`datasets/master_infrastructure_health_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/master_infrastructure_health_v1.parquet) (`45,756 rows x 28 original columns`)  
**Output Dataset:** [`datasets/features_stage1_generic_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/features_stage1_generic_v1.parquet) (`45,756 rows x 56 total columns: 28 original + 28 engineered`)  
**Feature Metadata:** [`datasets/feature_metadata_stage1.json`](file:///c:/Users/navad/ML_data/datasets/feature_metadata_stage1.json)  

---

## 1. Stage 1 Overview & Design Goal

Stage 1 transforms raw, vendor-specific string attributes (`hpe_cpu`, `dell_cpu`, `ping_status`, etc.) into **universally useful Generic Infrastructure Features** that serve as the cross-sectional foundation for downstream ML models (anomaly detection, failure prediction, and forecasting).

Per our architectural consensus, Stage 1 involves **no time-series shifts or lookahead windows**. It focuses exclusively on cross-sectional consolidation, ordinal quantification, and granular severity counting while strictly preserving all `28 original vendor columns` for complete auditability.

---

## 2. Feature Provenance & Transformation Table

| Engineered Feature | Derived From (Source Columns) | Transformation & Mathematical Rule | ML Usage / Role |
| :--- | :--- | :--- | :--- |
| **`ping_status_binary`** | `ping_status` | `Reachable -> 0, Unreachable -> 1` | **Core Feature:** Binary encoding of reachability |
| **`hardware_{comp}_worst_status`** | `hpe_{comp}`, `dell_{comp}` (`$COMP \in [cpu, memory, fans, storage, temp, power]$`) | Maximum available severity (`skipna=True`) across HPE and Dell observations using empirical `SEVERITY_MAP` (`OK=0, Degraded=1, NOT OK=2, Critical=3`). `NaN` if neither vendor exists. | **Core Feature:** Unified subsystem severity |
| **`critical_component_count`** | `hardware_{comp}_worst_status` (`all 6 components`) | Count of components where `worst_status == 3` (`Critical`) per observation (`0` for Ping-Only) | **Core Feature:** Measures concurrent critical failures |
| **`not_ok_component_count`** | `hardware_{comp}_worst_status` (`all 6 components`) | Count of components where `worst_status == 2` (`NOT OK` / intermediate anomaly) (`0` for Ping-Only) | **Core Feature:** Measures intermediate anomalous faults |
| **`degraded_component_count`** | `hardware_{comp}_worst_status` (`all 6 components`) | Count of components where `worst_status == 1` (`Degraded`) (`0` for Ping-Only) | **Core Feature:** Measures early subsystem wear |
| **`has_active_problem`** | `critical_count`, `not_ok_count`, `degraded_count`, `ping_status_binary` | `1 if (critical > 0 OR not_ok > 0 OR degraded > 0 OR ping == 1) else 0` | **Convenience Indicator:** High-level dashboard alert flag (downstream ML models rely on granular counts instead) |
| **`hardware_{comp}_disagreement_flag`** | `hpe_{comp}`, `dell_{comp}` (`all 6 components`) | `1 if (has_hpe & has_dell & hpe_{comp} != dell_{comp}) else 0` | **Explainability Only:** Excluded from ML training feature set to prevent learning synthetic mock-data noise |

---

## 3. Critical Engineering & Semantic Decisions

### A. Why `max(skipna=True)` Matters for Missing Values
When computing `hardware_{comp}_worst_status = max(hpe_rank, dell_rank)` across observations, our implementation explicitly enforces **`skipna=True`**:
1. **Dual-Monitored Servers (`HPE=1, Dell=3`):** Returns `3` (conservatively takes the worst reported severity).
2. **Single-Vendor Servers (`HPE=NaN, Dell=3`):** Returns `3` (conservatively takes the available vendor data rather than returning `NaN`).
3. **Ping-Only Servers (`HPE=NaN, Dell=NaN`):** Returns `NaN`.

> [!IMPORTANT]
> **Semantic Meaning of `NaN`:** Hardware-derived features (`worst_status`) are intentionally unavailable (`NaN`) for the `220 Ping-Only servers` that lack hardware telemetry. This reflects the available data (`Ping Only`) rather than an absence of hardware or data loss.

---

### B. Why Granular Severity Counters Are Kept Separate
Instead of squishing `Degraded` (`Rank 1`) and `NOT OK` (`Rank 2`) into one count, our pipeline maintains three exact, separated counters:
- **`critical_component_count` (Rank 3):** Tracks active out-of-bounds or failing subsystems.
- **`not_ok_component_count` (Rank 2):** Tracks intermediate Dell-specific anomalous degradation (`NOT OK`).
- **`degraded_component_count` (Rank 1):** Tracks standard early subsystem degradation (`Degraded`).

This ensures our ML models can distinguish between a minor degraded fan (`Rank 1`) and a serious `NOT OK` CPU (`Rank 2`).

---

### C. Communication of Rare Events (`<0.01%`)
In statistical and engineering reporting, mathematically rounding a single event (`1 / 45,756 = 0.00218%`) to `0.00%` is communication-wise poor because it visually implies "zero occurrences." Across our documentation and terminal outputs, rare events occurring under `0.01%` are explicitly formatted as **`<0.01%`** (e.g., `1 observation (<0.01%)`) so reviewers immediately see that the rare anomaly exists.

---

## 4. Output Verification Summary
- **Row Conservation:** Exactly `45,756 rows` maintained (`0` rows dropped or altered).
- **Column Provenance:** `28 original vendor columns + 28 engineered features = 56 total columns`.
- **Machine-Readable Metadata:** Generated [`datasets/feature_metadata_stage1.json`](file:///c:/Users/navad/ML_data/datasets/feature_metadata_stage1.json) detailing exact types (`ordinal`, `count`, `binary`, `flag`), source columns, definitions, and ML usage roles (`core_feature` vs `convenience_indicator` vs `explainability_only`) to power automated queries by our Explainable Data Understanding Agent.
