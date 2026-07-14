# Phase 4 Stage 3: Lookahead Target Label Generation Documentation

**Implementation Script:** [`feature_engineering/stage3_label_generation.py`](file:///c:/Users/navad/ML_data/feature_engineering/stage3_label_generation.py)  
**Input Dataset:** [`datasets/features_stage2_temporal_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/features_stage2_temporal_v1.parquet) (`45,756 x 62`)  
**Output Master Dataset:** [`datasets/master_ml_dataset_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/master_ml_dataset_v1.parquet) (`45,756 rows x 67 total columns`)  
**Master Metadata JSON:** [`datasets/feature_metadata_master.json`](file:///c:/Users/navad/ML_data/datasets/feature_metadata_master.json)  

---

## 1. Executive Summary & Architecture Overview

Stage 3 completes **Phase 4 (Feature Engineering)** by translating our frozen operational failure definitions (`08_server_failure_definition.md`) into machine-readable **Lookahead Target Labels (`Group D`)**. 

To strictly enforce scientific rigor and eliminate data leakage, Stage 3 introduces a two-layer label architecture:
1. **Instantaneous Helper (`helper_current_failure_state`):** Captures the operational state at time $t$. **Strictly excluded from training feature matrices ($X$).**
2. **Lookahead Prediction Targets (`target_*`):** Evaluates whether an operational failure ($F$) occurs strictly within future lookahead windows ($t+1 \dots t+W$). **This is what our ML models are trained to predict.**

---

## 2. Helper vs. Target Specification & Anti-Leakage Standard

| Column Name | Type | Derived From | Window / Horizon | Exact Formula | ML Usage Role & Leakage Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`helper_current_failure_state`** | `binary_helper` | `ping_status_binary`, `critical_component_count` | Instantaneous ($t$) | `((ping == 1) | (critical > 0)).astype(int)` | **Helper (`helper_exclude_from_training`)**.<br>Must NEVER be included in training matrix $X$. |
| **`target_failure_3slot`** | `binary_target` | `helper_current_failure_state` | Next 12 Hours (`+1 to +3 slots`) | `1 if max(helper[t+1..t+3]) == 1 else 0` | **Target Label (`target_label`)**.<br>Used as $y$ for 12-hour operational SLA prediction. |
| **`target_failure_6slot`** | `binary_target` | `helper_current_failure_state` | Next 24 Hours (`+1 to +6 slots`) | `1 if max(helper[t+1..t+6]) == 1 else 0` | **Target Label (`target_label`)**.<br>Used as $y$ for daily operational SLA prediction. |
| **`target_network_alert_3slot`** | `binary_target` | `ping_status_binary` | Next 12 Hours (`+1 to +3 slots`) | `1 if max(ping[t+1..t+3]) == 1 else 0` | **Target Label (`target_label`)**.<br>Used as $y$ for granular network dropout prediction. |
| **`target_hardware_failure_3slot`** | `binary_target` | `critical_component_count` | Next 12 Hours (`+1 to +3 slots`) | `1 if max(critical[t+1..t+3]) >= 1 else 0` | **Target Label (`target_label`)**.<br>Used as $y$ for physical subsystem failure prediction. |

> [!IMPORTANT]
> **Option C Separation of Concerns (Truthful `NaN` at Timeline End):**  
> For the final monitoring slot of each machine (`Slot 186`), future slots (`t+1..t+3`) literally do not exist because monitoring terminated. Therefore, Stage 3 **strictly preserves these boundary slots as `NaN` (`Int64` nullable integer)** inside `master_ml_dataset_v1.parquet`. Downstream model training scripts will filter out or drop `y.isna()` rows during train/test splitting (`Option C`).

---

## 3. Quantitative Verification & Exact Output Distributions

Running `stage3_label_generation.py` successfully computed all helpers and targets across the exact `45,756` observations without dropping a single row.

### A. Instantaneous Operational State (`helper_current_failure_state` across 45,756 rows):
- **`Healthy / Warning State (0)`:** `44,968 rows (98.28%)`
- **`Operational Failure State (1)`:** `788 rows (1.72%)`

### B. Lookahead Target Distributions (Class Balances across 45,756 rows):
| Target Label Column | Stable Negative Class (`0`) | Pre-Failure Positive Class (`1`) | Positive Class Percentage | Option C Boundary `NaN`s (Timeline End) | Operational Value |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`target_failure_3slot` (`12h ahead`)** | `43,587` | **`1,923`** | **`4.20%`** | `246 (0.54% — exactly 1 per server)` | Strong class balance for early-warning SLA prediction. Answers **Q6 & Q7**. |
| **`target_failure_6slot` (`24h ahead`)** | `42,084` | **`3,426`** | **`7.49%`** | `246 (0.54% — exactly 1 per server)` | Excellent daily lookahead target (`~7.5% positive`). Answers **Q6 & Q7**. |
| **`target_network_alert_3slot` (`12h ahead`)** | `43,643` | **`1,867`** | **`4.08%`** | `246 (0.54% — exactly 1 per server)` | Isolates near-term network routing dropouts. Answers **Q6 (`Ping instability`)**. |
| **`target_hardware_failure_3slot` (`12h ahead`)** | `45,452` | **`58`** | **`0.13%`** | `246 (0.54% — exactly 1 per server)` | Isolates physical hardware crashes across 11 servers (`~0.13%`). Ideal for rare anomaly classifiers. |

---

## 4. Final Master Dataset Provenance (`45,756 x 67 columns`)

Our golden master dataset (`datasets/master_ml_dataset_v1.parquet`) is frozen and fully documented in [`datasets/feature_metadata_master.json`](file:///c:/Users/navad/ML_data/datasets/feature_metadata_master.json):
1. **Group A (Original Preserved Audits & Identifiers):** `28 columns` (`observation_id`, `machine_name`, `monitoring_slot`, raw text logs, vendor states).
2. **Group B (Stage 1 Generic Cross-Sectional Features):** `29 columns` (`ping_status_binary`, `worst_status` severities, counts, disagreement flags).
3. **Group C (Stage 2 Empirically Justified Temporal Features):** `5 columns` (`ping_status_binary_lag1/lag2`, `timeout_rate_3slot/6slot`, `problems_active_sum_6slot`).
4. **Group D (Stage 3 Operational Helper & Lookahead Targets):** `5 columns` (`helper_current_failure_state`, `target_failure_3slot/6slot`, `target_network_alert_3slot`, `target_hardware_failure_3slot`).

---

## 5. Phase 4 Sign-Off & Ready for Phase 5 (Predictive & Anomaly Modeling)
With `master_ml_dataset_v1.parquet` created, verified, and cataloged, **Phase 4 (Feature & Target Engineering)** is officially complete. Every single engineered column is backed by empirical evidence, documented in detail, and perfectly aligned with **Assignment Questions Q2, Q11, Q7, and Q6**.
