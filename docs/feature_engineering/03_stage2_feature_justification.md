# Phase 4 Stage 2: Temporal Feature Selection & Justification Matrix

**Task Alignment:** Anomaly Detection (Q3), Failure Prediction (Q6), Root Cause Analysis (Q17)  
**Engineering Philosophy:** **10 Highly-Justified Temporal Features > 40 Generic Lag Features**  

---

## 1. The Core Philosophy of Temporal Feature Selection

In time-series machine learning, it is tempting to run lag and rolling window operations across every available metric. However, doing so blindly creates a combinatorial explosion of features that:
1. **Dilutes predictive signals** with sparse, noisy columns (e.g., rolling memory averages where memory failure is extremely rare).
2. **Increases training time** and risks overfitting (the curse of dimensionality).
3. **Reduces model explainability** for root-cause analysis (Q17).

To ensure every feature directly answers a business question, we define a **Feature Justification Matrix**. If a feature cannot answer:
- *"If I removed this feature, what information would the model lose?"*
- *"Which specific assignment question does this answer?"*

...then it is excluded from our pipeline.

---

## 2. Temporal Feature Justification Matrix

We select exactly **10 key temporal features** that capture state transitions, degradation trajectories, and recent peak distress:

| Feature Name | Derived From | Window / Operator | If Removed, What Does the Model Lose? | Business Task & Assignment Q |
| :--- | :--- | :--- | :--- | :--- |
| **`ping_status_binary_lag1`** | `ping_status_binary` | Lag 1 (4h ago) | The network state in the immediate preceding slot. | **Anomaly Detection (Q3) & Failure Prediction (Q6):** Differentiates a new connection loss ($0 \rightarrow 1$) from a persistent outage ($1 \rightarrow 1$). |
| **`ping_status_binary_lag2`** | `ping_status_binary` | Lag 2 (8h ago) | The network state 8 hours ago. | **Anomaly Detection (Q3) & Failure Prediction (Q6):** In tandem with `lag1`, identifies connection flapping ($1 \rightarrow 0 \rightarrow 1$) vs. stable networks. |
| **`ping_timeout_rate_3slot`** | `ping_status_binary` | Rolling Mean (12h) | The recent frequency of network dropouts. | **Failure Prediction (Q6):** Measures short-term network instability. |
| **`ping_timeout_rate_6slot`** | `ping_status_binary` | Rolling Mean (24h) | The daily network availability trend. | **Failure Prediction (Q6):** Differentiates chronic daily network issues from isolated transient blips. |
| **`hardware_cpu_worst_status_lag1`** | `hardware_cpu_worst_status` | Lag 1 (4h ago) | The immediate preceding CPU health state. | **Anomaly Detection (Q3) & Root Cause (Q17):** Detects sudden spikes in CPU degradation (e.g. $0 \rightarrow 2$ or $0 \rightarrow 3$ transition). |
| **`hardware_cpu_worst_trend_3slot`** | `hardware_cpu_worst_status` | Rolling Mean (12h) | The trajectory of CPU health. | **Failure Prediction (Q6) & Root Cause (Q17):** Distinguishes a temporary compute spike from a steady, worsening CPU overload. |
| **`hardware_temp_worst_status_lag1`** | `hardware_temperature_worst_status` | Lag 1 (4h ago) | The immediate preceding thermal state. | **Anomaly Detection (Q3) & Failure Prediction (Q6):** Identifies sudden thermal spikes. |
| **`hardware_temp_worst_peak_3slot`** | `hardware_temperature_worst_status` | Rolling Max (12h) | The maximum thermal distress in the last 12 hours. | **Failure Prediction (Q6) & Root Cause (Q17):** Retains memory of thermal stress even if the server temporarily cooled down in the current slot. |
| **`critical_component_count_lag1`** | `critical_component_count` | Lag 1 (4h ago) | The count of failed subsystems in the preceding slot. | **Anomaly Detection (Q3) & Failure Prediction (Q6):** Identifies rapid cascade failures (e.g., jumping from 0 to 2 critical components in 4 hours). |
| **`problems_active_sum_6slot`** | `has_active_problem` | Rolling Sum (24h) | The duration of overall instability over the last day. | **Failure Prediction (Q6):** Quantifies whether a server has been unstable all day or just encountered its first warning. |

---

## 3. Excluded Temporal Features & Rationale

We intentionally **exclude** rolling windows and lags for the remaining components (`memory`, `storage`, `fans`, `power`):

- **Sparse Value Domains:** Hardware memory errors (`memory_worst_status`) and storage disk failures (`storage_worst_status`) are extremely sparse in our dataset (e.g., only 2 critical memory events across 45,756 rows). 
- **Noise Prevention:** Creating rolling averages or lags for these columns would result in long sequences of zeros, introducing useless features that degrade model training efficiency.
- **Aggregation Coverage:** Any critical or degraded states in these excluded components are still captured by our aggregate counters (`critical_component_count`, `degraded_component_count`, and `has_active_problem`), which *are* lagged and summed over rolling windows. Therefore, no failure signal is lost.
