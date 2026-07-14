# Final ML Feature Catalog & Blueprint

**Assignment Alignment:** Q2 (Feature Selection), Q11 (Create 10 Useful Features), Q7 (Target Label Preparation), Q6 (Failure Prediction), Q3/Q4 (Anomaly Detection)  
**Status:** Frozen & Approved Blueprint for Pipeline Completion  

---

## 1. Executive Summary & Design Rationale

This document serves as the **frozen blueprint** for all downstream machine learning and anomaly detection models. Rather than jumping straight into model building or creating endless unvalidated lag features, we explicitly define:
1. Every feature that will exist in the final ML dataset (`45,756 rows`).
2. Why each feature exists and which specific assignment question it answers.
3. Whether the feature is used as an **Active Training Feature**, an **Explainability/Diagnostic Column**, or a **Target Label** (strictly excluded from training matrices to prevent data leakage).

---

## 2. Comprehensive Feature & Label Blueprint

### Group A: Raw Preserved Features (Audit & Explainability Foundation)
*These columns are strictly preserved from the unified gold dataset (`master_v1`) to maintain full data lineage and auditability.*

| Feature Name | Type / Values | Why It Exists | Assignment Alignment | ML Usage Role |
| :--- | :--- | :--- | :--- | :--- |
| **`observation_id`** | String | Unique primary key for every machine-slot snapshot (`{machine_name}|{ip}|{slot}`). | All Questions | **Identifier** |
| **`machine_name`**, **`ip_address`** | String | Identifies the specific physical or virtual machine. | Q3, Q17 | **Identifier / Grouping Key** |
| **`monitoring_slot`** | String / Time | Chronological ordering key (`YYYY-MM-DD_Slot-XX`, 4-hour intervals). | Q6, Q7 | **Time Index / Sorting Key** |
| **`ping_status`** | Categorical (`Reachable`, `Unreachable`) | Raw network reachability status from Ping tool. | Q2, Q3, Q6 | **Raw Audit** (encoded via binary feature) |
| **`hpe_{comp}`** (`$COMP \in [cpu, memory, fans, storage, temp, power]$`) | Categorical (`OK`, `Degraded`, `Critical`) | Raw telemetry reported by HPE OneView sensor. | Q2, Q17 | **Raw Audit** (transformed via `worst_status`) |
| **`dell_{comp}`** (`$COMP \in [cpu, memory, fans, storage, temp, power]$`) | Categorical (`OK`, `Degraded`, `NOT OK`, `Critical`) | Raw telemetry reported by Dell iDRAC sensor. | Q2, Q17 | **Raw Audit** (transformed via `worst_status`) |
| **`hpe_current_problems`** | Free-text string | Unstructured diagnostic logs from HPE controllers. | Q17 (Root Cause Analysis) | **Explainability Only** (Text audit) |
| **`dell_issues_detected`** | Free-text JSON string | Unstructured diagnostic alerts from Dell hardware. | Q17 (Root Cause Analysis) | **Explainability Only** (Text audit) |
| **`dell_overall_status`** | Categorical | Raw overall health reported by Dell enclosure. | Q17 | **Raw Audit** |
| **`has_ping`**, **`has_hpe`**, **`has_dell`** | Boolean | Tracks which vendor tools observed the server in that slot. | Q2, Q11 | **Metadata / Filtering** |

---

### Group B: Stage 1 Generic Features (Cross-Sectional Health)
*Direct answers to **Question 11 ("Create 10 useful features")** and **Question 2 ("Which columns would you use as features?")**.*

| Feature Name | Formula / Mapping | Why It Exists | Assignment Alignment | ML Usage Role |
| :--- | :--- | :--- | :--- | :--- |
| **`ping_status_binary`** | `Reachable -> 0, Unreachable -> 1` | Converts text network reachability into a mathematical binary indicator. | Q2, Q11, Q3, Q6 | **Core Training Feature** |
| **`hardware_{comp}_worst_status`** (`6 columns`) | `max(skipna=True)` across HPE & Dell using empirical `SEVERITY_MAP` (`OK=0, Degraded=1, NOT OK=2, Critical=3`). `NaN` for Ping-Only servers. | Unifies dual-monitored hardware into a single, conservative severity rank while preserving `NaN` for servers without hardware sensors. | Q2, Q11, Q3, Q6, Q17 | **Core Training Feature** |
| **`critical_component_count`** | Count of the 6 `worst_status` columns where rank $= 3$. | Measures concurrent catastrophic hardware failures on a physical server (`0` for Ping-Only). | Q2, Q11, Q3, Q6, Q17 | **Core Training Feature** |
| **`not_ok_component_count`** | Count of the 6 `worst_status` columns where rank $= 2$. | Measures intermediate/anomalous Dell-specific faults (`NOT OK`). | Q2, Q11, Q3 | **Core Training Feature** |
| **`degraded_component_count`** | Count of the 6 `worst_status` columns where rank $= 1$. | Measures early subsystem degradation/wear (`0` for Ping-Only). | Q2, Q11, Q3, Q6 | **Core Training Feature** |
| **`has_active_problem`** | `1 if (critical > 0 OR not_ok > 0 OR degraded > 0 OR ping == 1) else 0` | Universal binary alert indicator across network and hardware. | Q11, Dashboarding | **Convenience Indicator** |
| **`hardware_{comp}_disagreement_flag`** (`6 columns`) | `1 if (has_hpe & has_dell & hpe != dell) else 0` | Flags sensor divergence between HPE and Dell across dual-monitored servers. | Q11, Q17 | **Explainability Only** (Excluded from ML training to prevent learning mock noise) |

---

### Group C: Stage 2 Temporal Features (Empirically Justified Time-Series)
*Carefully selected based on Stage 1.5 empirical transition evidence showing that Ping/network is highly volatile while hardware fails abruptly.*

| Feature Name | Derived From | Window / Operator | Why It Exists | Assignment Alignment | ML Usage Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ping_status_binary_lag1`** | `ping_status_binary` | Lag 1 (`-4 hours`) | Provides immediate preceding network state. | Q3 (Anomaly), Q6 (Failure) | **Core Training Feature** |
| **`ping_status_binary_lag2`** | `ping_status_binary` | Lag 2 (`-8 hours`) | Identifies short-term network flapping ($1 \rightarrow 0 \rightarrow 1$) when combined with `lag1`. | Q3 (Anomaly), Q6 (Failure) | **Core Training Feature** |
| **`ping_timeout_rate_3slot`** | `ping_status_binary` | Rolling Mean (`12 hours`) | Quantifies recent frequency of network dropouts over half a day. | Q6 (Failure Prediction) | **Core Training Feature** |
| **`ping_timeout_rate_6slot`** | `ping_status_binary` | Rolling Mean (`24 hours`) | Quantifies sustained daily network availability vs. isolated blips. | Q6 (Failure Prediction) | **Core Training Feature** |
| **`problems_active_sum_6slot`** | `has_active_problem` | Rolling Sum (`24 hours`) | Measures the duration and persistence of overall instability over the last day. | Q6 (Failure Prediction) | **Core Training Feature** |

> [!NOTE]
> **Why Hardware Lags Are Excluded from Group C:** As proven in Stage 1.5 (`analyze_state_transitions.py`), 100% of Critical CPU events occurred instantaneously ($0 \rightarrow 3$) without gradual warning transitions. Lagging hardware components provides zero early-warning predictive signal in this dataset.

---

### Group D: Stage 3 Target Labels (Lookahead Training Targets)
*Direct answers to **Question 7 ("How would you prepare training labels?")** and **Question 6 ("Can you predict failures?")**.*

| Target Name | Derived From | Lookahead Window | Definition & Formula | Assignment Alignment | ML Usage Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`target_ping_failure_3slot`** | `ping_status_binary` | Next 3 slots (`+12 hours`) | `1 if max(ping_status_binary in [t+1, t+2, t+3]) == 1 else 0` | Q7, Q6 (Near-term network failure) | **Target Label** (Strictly dropped during training) |
| **`target_ping_failure_6slot`** | `ping_status_binary` | Next 6 slots (`+24 hours`) | `1 if max(ping_status_binary in [t+1..t+6]) == 1 else 0` | Q7, Q6 (Daily network failure) | **Target Label** (Strictly dropped during training) |
| **`target_hardware_failure_3slot`** | `critical_component_count` | Next 3 slots (`+12 hours`) | `1 if max(critical_component_count in [t+1, t+2, t+3]) > 0 else 0` | Q7, Q6 (Near-term hardware crash) | **Target Label** (Strictly dropped during training) |
| **`target_hardware_failure_6slot`** | `critical_component_count` | Next 6 slots (`+24 hours`) | `1 if max(critical_component_count in [t+1..t+6]) > 0 else 0` | Q7, Q6 (Daily hardware crash) | **Target Label** (Strictly dropped during training) |

> [!CAUTION]
> **Anti-Leakage Standard:** Target labels involve forward-looking window operations (`shift(-1)`, `shift(-2)`, etc.). Whenever a model is trained to predict `target_X`, **ALL Group D target columns must be explicitly dropped from the feature matrix $X$** so the model only learns from Group B & C historical data.

---

## 3. Assignment Completion Scorecard (Powered by this Blueprint)

By finalizing this catalog, our completion status across the primary assignment tasks becomes clear:

| Assignment Question | Supported By Catalog Groups | Completion Status |
| :--- | :--- | :--- |
| **Q2: Which columns as features?** | Group B (`worst_status`, counts, binary ping) & Group C (empirically justified lags) | **100% Finalized Blueprint** |
| **Q11: Create 10 useful features** | Group B (exactly 12 engineered features) + Group C (5 temporal features) = **17 Features Created & Justified** | **100% Finalized Blueprint** |
| **Q7: Prepare training labels** | Group D (3-slot/12h and 6-slot/24h lookahead binary target formulations) | **100% Finalized Blueprint** (Ready for coding in Stage 3) |
| **Q6: Predict failures** | Models predicting Group D targets using Group B + C features | **Framework & Targets Ready** |
| **Q3/Q4: Anomaly Detection** | Unsupervised models evaluating Group B cross-sectional health + Group C network volatility | **Feature Foundation Ready** |

---

## 4. Next Implementation Steps (Mechanical Execution)

With the blueprint frozen, our remaining code generation becomes straightforward and 100% justified:
1. **`stage2_temporal_features.py`**: Reads `features_stage1_generic_v1.parquet`, groups by `machine_name`, sorts chronologically by `monitoring_slot`, and creates exactly the 5 approved features in **Group C**. Exports `features_stage2_temporal_v1.parquet`.
2. **`stage3_label_generation.py`**: Reads `features_stage2_temporal_v1.parquet`, groups by `machine_name`, applies lookahead windows (`shift(-1..-6)`) to generate the 4 approved target labels in **Group D**. Exports the **Final Master ML Dataset** (`master_ml_dataset_v1.parquet`).
