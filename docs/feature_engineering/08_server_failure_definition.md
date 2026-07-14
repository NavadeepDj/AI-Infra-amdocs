# Phase 4 Stage 3 Foundation: Operational Failure Definition (`Option C`)

**Assignment Alignment:** Question 6 ("Predict failures"), Question 7 ("Prepare training labels")  
**Empirical Verification Script:** [`feature_engineering/verify_critical_hardware_events.py`](file:///c:/Users/navad/ML_data/feature_engineering/verify_critical_hardware_events.py)  
**Status:** Frozen Operational Specification for Stage 3 Lookahead Target Generation  

---

## 1. What is an Operational Failure? (`Option C` Specification)

From the perspective of an SRE or infrastructure engineer managing a production telecommunications core (e.g., Amdocs 5G network functions), we distinguish between *underlying hardware warnings* and an **Operational Failure**. 

An **Operational Failure** occurs at timestamp $t$ when the server is operationally unhealthy from a client or infrastructure perspective—either because it is unresponsive on the network (`ping_status_binary == 1`) or because a physical hardware subsystem has breached safe operating bounds (`critical_component_count > 0`), requiring immediate operator intervention.

Formally, for any server observation at slot $t$, instantaneous operational state ($F_t$) is captured via a dedicated derived helper:

$$\text{helper\_current\_failure\_state}_t = (\text{ping\_status\_binary}_t == 1) \lor (\text{critical\_component\_count}_t > 0)$$

> [!CAUTION]
> **Why `helper_current_failure_state` is a Helper, NOT a Prediction Target or Training Feature:**  
> `helper_current_failure_state` represents the *current* state at time $t$. **Our machine learning models do NOT predict what is currently happening—they predict whether a server will enter an operational failure state within the next 12 hours (`+1..+3 slots`) or 24 hours (`+1..+6 slots`).** Furthermore, `helper_current_failure_state` must be strictly excluded from any training feature matrix ($X$). If included, models would achieve artificial ~99% accuracy due to data leakage (`current failure state -> future failure state`).

---

## 2. Why is this Definition Appropriate? (Empirical Verification)

Before adopting `Option C`, we executed [`verify_critical_hardware_events.py`](file:///c:/Users/navad/ML_data/feature_engineering/verify_critical_hardware_events.py) to inspect the unstructured diagnostic logs across all 25 observations where `critical_component_count > 0`.

**In this dataset, every observed Rank 3 (`Critical`) hardware event corresponded to a severe physical hardware issue** requiring physical replacement or emergency thermal remediation, with zero false-positive alarms:
- **Storage Crashes:** `"Drive 0 failed ; System status is in critical state"`, `"Disk 1 in drive bay failed."`
- **Power Losses:** `"Power Supply 2 failed"`, `"Power Supply 1 failed"`
- **Cooling & Thermal Breaches:** `"Fan 2 has failed."`, `"Fan 2 has failed. ; Power supply redundancy is lost."`, `"System temperature exceeded safe operating limit."`
- **CPU/Memory Faults:** `"CPU 2 critical hardware fault"`, `"Memory Module 3 failed"`

Simultaneously, `ping_status == 'Unreachable'` (`ping_status_binary == 1`) represents up to 4 hours where the server interface stopped responding to network health checks. By combining both into `helper_current_failure_state`, we ensure comprehensive coverage across both network reachability and hardware integrity.

---

## 3. Warning Precursors vs. Operational Failure Boundaries

To prevent downstream ML models from confusing non-actionable warnings with operational failures, we enforce a strict 3-tier production hierarchy:

```text
Symptom / Precursor (Rank 1 / Rank 2) ──> Does NOT trigger failure (Server remains functional)
True Operational Failure (Rank 3 / Ping=1) ──> Triggers helper_current_failure_state = 1
Lookahead Target (t+1 .. t+W) ──> The actual objective our ML models predict
```

| Signal Category | Exact Dataset Indicators | Operational Meaning & SRE Action | Labeled as `helper_current_failure_state = 1`? |
| :--- | :--- | :--- | :--- |
| **Operational Precursors (Warnings)** | • `degraded_component_count > 0` (`Rank 1`)<br>• `not_ok_component_count > 0` (`Rank 2`)<br>• Text: `"CPU 1 is throttling due to high load"`<br>• Text: `"Disk array controller is in warning state"` | Early thermal wear, heavy load throttling, or single-fan redundancy loss. The server is still processing traffic. | **NO (Warning Precursor Only)** |
| **Network Operational Failure** | • `ping_status_binary == 1` (`Unreachable`) | Server network stack, switch port, or OS has stopped responding to pings. Client requests fail. | **YES (Operational Failure)** |
| **Hardware Operational Failure** | • `critical_component_count > 0` (`Rank 3`) | Physical component breakdown (`Drive 0 failed`, `Fan 2 failed`, `PSU failed`). Subsystem safe limits breached. | **YES (Operational Failure)** |

---

## 4. Converting State into ML Lookahead Targets (`target_*`)

To answer **Question 6 ("Predict failures")** and **Question 7 ("Prepare training labels")**, we generate forward-looking binary target labels evaluating whether `helper_current_failure_state == 1` occurs within future lookahead windows ($t+1 \dots t+W$).

### A. Overall Operational Lookahead Targets (Core SLAs)
- **Near-Term Operational Failure (`12 Hours / 3 Slots ahead`):**
  $$\text{target\_failure\_3slot}_t = \begin{cases} 1 & \text{if } \max(\text{helper\_current\_failure\_state}_{t+1..t+3}) == 1 \\ 0 & \text{otherwise} \end{cases}$$
- **Extended Operational Failure (`24 Hours / 6 Slots ahead`):**
  $$\text{target\_failure\_6slot}_t = \begin{cases} 1 & \text{if } \max(\text{helper\_current\_failure\_state}_{t+1..t+6}) == 1 \\ 0 & \text{otherwise} \end{cases}$$

### B. Root-Cause Specific Lookahead Targets (Diagnostic Granularity)
To support granular root-cause prediction, we also isolate the lookahead across network and hardware specifically:
- **`target_network_alert_3slot` (`12 Hours ahead`):** `1 if max(ping_status_binary in [t+1..t+3]) == 1 else 0`
- **`target_hardware_failure_3slot` (`12 Hours ahead`):** `1 if max(critical_component_count in [t+1..t+3]) >= 1 else 0`

---

## 5. Strict Anti-Leakage Standard for Downstream Modeling

> [!IMPORTANT]
> **The Golden Rule of Training Matrix Construction:**  
> During model training, the feature matrix $X$ **must exclude all columns prefixed with `target_` and `helper_`** derived from the current or future failure state to prevent data leakage.  
> 
> Specifically, when predicting `target_failure_3slot`, the algorithm must learn purely from historical precursors at time $t$ (`ping_status_binary`, `hardware_{comp}_worst_status`, `degraded_component_count`, `not_ok_component_count`, `ping_timeout_rate_3slot`, lags, and rolling durations).

---

## 6. Mechanical Stage 3 Execution Plan

`stage3_label_generation.py` will:
1. Load `features_stage2_temporal_v1.parquet` (`45,756 x 62`).
2. Construct `helper_current_failure_state = ((df['ping_status_binary'] == 1) | (df['critical_component_count'] > 0)).astype(int)`.
3. Group strictly by `machine_name` sorted chronologically, and apply `.shift(-i)` across lookahead windows ($1..3$ and $1..6$) to compute `target_failure_3slot`, `target_failure_6slot`, `target_network_alert_3slot`, and `target_hardware_failure_3slot`.
4. Export the **Final Master ML Dataset (`master_ml_dataset_v1.parquet`)** containing exactly `45,756 rows x 67 columns` (`28 original` + `34 Stage 1/2 features` + `1 derived helper` + `4 lookahead targets`).
