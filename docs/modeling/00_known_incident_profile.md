# Phase 5.1: Known Incident Profile & Unsupervised Feature Characteristics

**Input Dataset:** [`datasets/master_ml_dataset_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/master_ml_dataset_v1.parquet) (`45,756 observations across 246 servers`)  
**Profile Export JSON:** [`datasets/known_incident_profile.json`](file:///c:/Users/navad/ML_data/datasets/known_incident_profile.json)  
**Implementation Script:** [`modeling/phase5.1_anomaly_exploration.py`](file:///c:/Users/navad/ML_data/modeling/phase5.1_anomaly_exploration.py)  

---

## Executive Summary & Separation of Responsibilities

To ensure mathematical rigor across **Track A (Unsupervised Anomaly Detection: Q3 & Q4)** versus **Track B (Supervised Prediction: Q6 & Q8)**, we strictly separate two fundamental concepts before fitting any models:

```text
Unsupervised Training Features (Part A)      vs.      Evaluation Baseline (Part B)
• What our anomaly detectors actually see             • What SREs know post-hoc about physical faults
• Completely unlabeled & unsupervised                 • Used strictly for post-hoc recall benchmarking
• Measures statistical rarity & density               • Never used during unsupervised model fitting
```

> [!IMPORTANT]
> **Why Anomaly $\neq$ Failure:**  
> An unsupervised anomaly detector (`Isolation Forest`) does not know what a "failure" is; its sole job is to isolate statistically unusual multivariate observations. A transient single-slot ping timeout might happen hundreds of times (`not rare`), whereas a server experiencing simultaneous thermal warning, fan degradation, and rolling network instability (`rare`) is exactly what an anomaly detector should catch. Therefore, **we never compute or fix model hyperparameters (like `contamination`) from our evaluation labels.** Instead, we treat `contamination` as an experimental hyperparameter (`1%, 2%, 3%, 5%`) and evaluate post-hoc against this profile.

---

## Part A: Dataset & Feature Profile (`18 Validated Unsupervised Features`)

We audited the exact empirical distributions across our `45,756 observations` to see what abnormal behavior looks like in our candidate feature space ($X$):

| Unsupervised Feature / Behavior | Exact Row Count | % of Dataset (`45,756`) | SRE & Statistical Interpretation |
| :--- | :---: | :---: | :--- |
| **`problems_active_sum_6slot == 0` (Zero Problems)** | `42,097` | `92.00%` | Baseline normal infrastructure state (`92% of all server hours`). |
| **`problems_active_sum_6slot >= 3` (Heavy Accumulation)** | `330` | `0.72%` | Extreme tail: servers accumulating $\ge 3$ distinct hardware problem flags over 24h. |
| **`ping_timeout_rate_3slot >= 66%` (Severe 12h Instability)** | `335` | `0.73%` | Extreme tail: servers unreachable for at least 8 of the last 12 hours. |
| **`ping_timeout_rate_3slot == 100%` (Persistent 12h Outage)** | `80` | `0.17%` | Rare complete reachability blackout lasting $\ge 12$ consecutive hours. |
| **`ping_timeout_rate_6slot == 100%` (Chronic 24h Outage)** | `14` | `0.03%` | Extremely rare chronic network dead-state lasting $\ge 24$ consecutive hours. |
| **`critical_component_count > 0` (`Rank 3` Hardware Fault)** | `25` | `0.05%` | Physical component failures (`CPU, Memory, Storage, Fans, Temp, Power`). |
| **`degraded_component_count > 0` (`Rank 1` Warning)** | `89` | `0.19%` | Physical hardware early-warning degradation. |

---

## Part B: Post-Hoc Evaluation Baseline (`Known Incident Profile`)

This exact 1-page reference details what physical and operational incidents actually exist in the dataset. When an anomaly detector flags $N$ observations, we cross-reference against this baseline to benchmark coverage and false-alarm rates:

```text
======================================================================================
                         SRE KNOWN INCIDENT BASELINE PROFILE
======================================================================================
1. Physical Hardware Critical Faults (`critical > 0`)      :    25 rows across  11 servers
2. Instantaneous Network Dropouts (`ping == 1`)            :   763 rows across 135 servers
3. Chronic 24-Hour Network Outages (`rate_6slot == 1.0`)   :    14 rows across   5 servers
4. Persistent 12-Hour Network Outages (`rate_3slot == 1`)  :    80 rows across  21 servers
5. Heavy Hardware Problem Accumulation (`prob_sum >= 3`)   :   330 rows across  37 servers
--------------------------------------------------------------------------------------
[BENCHMARK 1] Total Current Operational Incidents (`helper_current == 1`):
    -> exactly 788 rows across 139 distinct servers (1.7222% of total dataset)
[BENCHMARK 2] Lookahead Pre-Failure Window (`target_failure_3slot == 1`):
    -> exactly 1,923 rows across 139 distinct servers (4.2027% of total dataset)
======================================================================================
```

---

## Hyperparameter Testing Grid for Phase 5.2 (`Isolation Forest`)

Because `contamination` is an unsupervised model hyperparameter rather than a derived label statistic, Phase 5.2 will execute an experimental grid search across four candidate thresholds:

| Tested Contamination (`c`) | Expected Anomalies Flagged out of `45,756 rows` | Post-Hoc Evaluation Goals (`Benchmark 1 vs False Alarms`) |
| :---: | :---: | :--- |
| **`c = 0.01` (1.0%)** | `~458 anomalies` | Ultra-strict precision: does it capture all `25` hardware faults and top chronic timeouts? |
| **`c = 0.02` (2.0%)** | `~915 anomalies` | Balanced envelope: close to our ~1.72% observed operational incident volume. |
| **`c = 0.03` (3.0%)** | `~1,373 anomalies` | Broader recall: how many additional early-warning degradations (`prob_sum_6slot`) are caught? |
| **`c = 0.05` (5.0%)** | `~2,288 anomalies` | High-recall frontier: does it capture early pre-failure lookahead observations at the cost of P1 noise? |

---

## Sign-Off & Next Step
With our Known Incident Profile frozen and strictly segregated from training features, we proceed directly to **Phase 5.2: Isolation Forest Baseline (`Question Q3`)**, testing `c = [0.01, 0.02, 0.03, 0.05]` and evaluating exact post-hoc overlap against this document.
