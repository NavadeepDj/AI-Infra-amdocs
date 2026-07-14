# Phase 4 Stage 3 Foundation: Infrastructure Failure Definitions & Target Formulation

**Assignment Alignment:** Question 6 ("Predict failures"), Question 7 ("Prepare training labels")  
**Purpose:** Establish unambiguous, scientifically defensible business definitions of "Failure" before writing any lookahead label generation code (`stage3_label_generation.py`).  

---

## 1. Executive Summary & The Core Problem

A common error in machine learning pipelines is jumping directly to coding `target_failure = 1` without first answering the foundational business question:

> **"What exact operational event constitutes a FAILURE versus a temporary warning or transient anomaly?"**

If a target label treats a single 4-minute ping packet loss identical to a catastrophic CPU hardware crash, the downstream classification model learns conflicting boundaries and produces meaningless predictions. To ensure our target labels (`target_*`) accurately reflect infrastructure reality and satisfy **Question 6** and **Question 7**, we formally define three distinct failure regimes: **Network Outages**, **Hardware Criticality**, and **Sustained System Instability**.

---

## 2. Formal Business & Technical Definitions of Failure

### A. Network Failure Regime (`Network Outage` vs `Transient Blip`)
In our dataset, `monitoring_slot` operates at **4-hour intervals**. Therefore, a single `Unreachable` slot (`ping_status_binary = 1`) represents up to 4 hours of network timeout.
- **Level 1 — Transient Network Alert (`Network Blip`):**  
  A single isolated slot where `ping_status_binary == 1` followed by immediate recovery (`0`).  
  *Operational Meaning:* Connection drop, router reboot, or temporary packet loss.  
- **Level 2 — True Network Failure (`Sustained Outage`):**  
  Where `ping_status_binary == 1` occurs across **$\ge 2$ monitoring slots within a 24-hour window (`6 slots`)** or **2 consecutive slots (`8 hours straight`)**.  
  *Operational Meaning:* Chronic network routing breakdown or unreachable server interface requiring SRE intervention.

---

### B. Hardware Failure Regime (`Component Degradation` vs `True Hardware Failure`)
Our ordinal severity scale (`0=OK, 1=Degraded, 2=NOT OK, 3=Critical`) differentiates early wear from catastrophic physical failure:
- **Non-Failure State (`Degraded` / `NOT OK`):**  
  When `degraded_component_count > 0` (`Rank 1`) or `not_ok_component_count > 0` (`Rank 2`).  
  *Operational Meaning:* Early thermal elevation, single fan redundancy loss, or warning state. The server is still operational. **This is NOT labeled as a hardware failure.**
- **True Hardware Failure (`Critical Component Crash`):**  
  When `critical_component_count >= 1` (`Rank 3` on any subsystem: CPU, memory, fans, storage, temperature, or power).  
  *Operational Meaning:* Physical hardware fault (e.g., `"Fan 2 has failed"`, `"System temperature exceeded safe operating limit"`, `"Disk array controller in warning/critical state"`). The server subsystem has breached safe operating bounds.
- **Catastrophic System Crash (`Multi-Component Failure`):**  
  When `critical_component_count >= 2`. Exactly `4 observations` in our dataset exhibit concurrent critical hardware faults (`v5G-AMF-Backup-02`).

---

### C. Chronic Instability Regime (`Sustained Unhealthy Operation`)
- **Chronic System Failure:**  
  When `problems_active_sum_6slot == 6`. Exactly `17 observations` (`0.04%`) across the dataset spend **24 consecutive hours (`6 slots`)** in a continuous alert state without recovery.

---

## 3. Approved Target Label Formulations (Stage 3 Blueprint)

Based on these formal definitions and our empirical dataset support (`45,756 rows` spanning 1 month), we approve exactly **4 binary lookahead target labels** for `stage3_label_generation.py`:

| Target Label Name | Failure Regime | Lookahead Horizon | Exact Mathematical & Logical Formula | Why It Exists & Operational Value |
| :--- | :--- | :--- | :--- | :--- |
| **`target_network_alert_3slot`** | Network Alert (Level 1) | Next 12 Hours (`+1 to +3 slots`) | `1 if max(ping_status_binary in [t+1, t+2, t+3]) == 1 else 0` | Predicts near-term network connection dropouts. Answers Q6 (`Ping instability`). |
| **`target_network_outage_6slot`** | Network Failure (Level 2) | Next 24 Hours (`+1 to +6 slots`) | `1 if sum(ping_status_binary in [t+1..t+6]) >= 2 else 0` | Predicts chronic daily network outages ($\ge 8\text{h}$ total loss). Answers Q6 (`Predict failures`). |
| **`target_hardware_failure_3slot`** | True Hardware Failure | Next 12 Hours (`+1 to +3 slots`) | `1 if max(critical_component_count in [t+1, t+2, t+3]) >= 1 else 0` | Predicts near-term physical component failure (`Rank 3`). Answers Q6 (`Hardware alerts`). |
| **`target_hardware_failure_6slot`** | True Hardware Failure | Next 24 Hours (`+1 to +6 slots`) | `1 if max(critical_component_count in [t+1..t+6]) >= 1 else 0` | Predicts daily physical hardware faults. Answers Q6 (`Predict failures over extended horizons`). |

---

## 4. Empirical Dataset Feasibility & Class Balance

Before writing the Stage 3 script, let's verify that the dataset (`45,756 rows x 186 slots across 246 servers`) actually supports these labels with sufficient positive examples:
- **Hardware Critical Events (`critical_component_count >= 1`):** `25 total observations` (`21` single critical + `4` dual critical) across `11` distinct hardware servers. When expanded across a `3-slot (+12h)` or `6-slot (+24h)` lookahead window, the number of positive pre-failure training examples ($t$) expands dynamically to approximately **`75 to 150 rows`**, providing an ideal, highly-targeted anomaly prediction class balance (`~0.2% to 0.4%`).
- **Network Outage Events (`ping_status_binary == 1`):** `766 total observations` across `135 servers`. When formulated as a `6-slot (+24h)` lookahead (`sum >= 2`), positive training examples span several hundred observations (`~1.5% to 2.5%`).

> [!IMPORTANT]
> **Why Not 7-Day Prediction (`42 Slots`)?**  
> While "7 days" is a common business phrase, in a 31-day dataset sampled every 4 hours (`186 slots total`), a 7-day lookahead window equals **42 monitoring slots**. If we predicted 42 slots ahead, the last 42 observations (`22% of every machine's entire history`) would have `NaN` targets due to timeline exhaustion. Furthermore, predicting a fan failure 7 days in advance when failures occur instantaneously (`Stage 1.5`) is scientifically unsound. The approved **12-hour (`3-slot`) and 24-hour (`6-slot`) horizons** represent the exact, actionable SRE response windows supported by our 31-day telemetry history.

---

## 5. Next Step: Stage 3 Code Generation

With this failure definition frozen and justified, `stage3_label_generation.py` will:
1. Load `features_stage2_temporal_v1.parquet` (`45,756 x 62`).
2. Group strictly by `machine_name` sorted by `monitoring_slot`.
3. Compute the 4 lookahead targets (`target_network_alert_3slot`, `target_network_outage_6slot`, `target_hardware_failure_3slot`, `target_hardware_failure_6slot`) using clean `.shift(-i)` lookahead window transformations.
4. Export the **Final Golden Master ML Dataset** (`master_ml_dataset_v1.parquet`).
