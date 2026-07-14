# Phase 4 Stage 1.5: Temporal Behavior Analysis

**Implementation Script:** [`feature_engineering/analyze_state_transitions.py`](file:///c:/Users/navad/ML_data/feature_engineering/analyze_state_transitions.py)  
**Input Dataset:** [`datasets/features_stage1_generic_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/features_stage1_generic_v1.parquet)  

---

## 1. Executive Summary

Before building Stage 2 (Rolling / Historical Features), we ran an empirical analysis on the state transitions within our dataset to determine whether temporal/lag features are actually justified.

> **Key Finding:** Ping/network statuses exhibit frequent transitions and chronic instability patterns, heavily justifying temporal network features. However, hardware component failures (like CPU) occur **instantaneously** with zero gradual degradation, rendering hardware lag features useless for early warning prediction.

---

## 2. Empirical Transition Evidence

### A. Ping Status Transitions (Highly Volatile)
Network reachability changes frequently, providing rich temporal signals.
- **Total servers with state changes:** `135 / 246` (54.8%)
- **Transitions into failure ($0 \rightarrow 1$):** `568` occurrences
- **Recovery transitions ($1 \rightarrow 0$):** `566` occurrences
- **Sustained outages ($1 \rightarrow 1$):** `194` occurrences

**Verdict:** Ping lags and rolling averages (e.g., `ping_timeout_rate_3slot`) are highly justified because network instability occurs gradually and repeatedly.

---

### B. CPU Status Transitions (Instant Failure)
Hardware state changes are sparse and abrupt.
- **Total servers with state changes:** `11 / 26` (42.3%)
- **Gradual degradation ($0 \rightarrow 1$):** `10` occurrences
- **Warning signs before Critical CPU ($1 \rightarrow 3$ or $2 \rightarrow 3$):** **0 occurrences**
- **Instant failures ($0 \rightarrow 3$):** **1 occurrence (100% of critical events)**

**Verdict:** In this mock dataset, hardware failures are injected abruptly. There is no historical "warning sign" trajectory (like going from Degraded to Critical). Therefore, creating `hardware_cpu_worst_status_lag1` or `hardware_cpu_worst_trend_3slot` is **not justified** because the history provides zero early warning power for predicting failures.

---

## 3. Revised Strategy for Stage 2

Based on this evidence, our previously proposed 10 temporal features must be drastically reduced. We will **drop all hardware component lags and rolling averages**, as they mathematically provide zero predictive leverage.

Our Stage 2 temporal features will focus exclusively on the proven volatile metrics:
1. `ping_status_binary_lag1`
2. `ping_status_binary_lag2`
3. `ping_timeout_rate_3slot` (12h rolling mean)
4. `ping_timeout_rate_6slot` (24h rolling mean)
5. `problems_active_sum_6slot` (24h duration of instability)
