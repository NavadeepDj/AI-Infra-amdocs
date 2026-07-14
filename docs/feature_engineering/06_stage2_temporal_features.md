# Phase 4 Stage 2: Empirically Justified Temporal Features Documentation

**Implementation Script:** [`feature_engineering/stage2_temporal_features.py`](file:///c:/Users/navad/ML_data/feature_engineering/stage2_temporal_features.py)  
**Input Dataset:** [`datasets/features_stage1_generic_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/features_stage1_generic_v1.parquet) (`45,756 x 57`)  
**Output Dataset:** [`datasets/features_stage2_temporal_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/features_stage2_temporal_v1.parquet) (`45,756 rows x 62 total columns: 28 original + 34 engineered`)  
**Feature Metadata:** [`datasets/feature_metadata_stage2.json`](file:///c:/Users/navad/ML_data/datasets/feature_metadata_stage2.json)  

---

## 1. Stage 2 Overview & Anti-Leakage Standard

Stage 2 adds **temporal memory** (past states and rolling averages) to our cross-sectional Stage 1 features. To ensure zero data leakage and exact chronological alignment:
1. **Strict Chronological Sorting:** The dataset is grouped by `machine_name` and sorted ascending by `monitoring_slot`.
2. **Backward-Looking Only:** Every operation uses strictly backward-looking windows ($t, t-1, t-2, \dots$). No forward-looking ($t+1$) data is accessed (that belongs in Stage 3: Lookahead Target Labels).
3. **Empirically Pruned Scope:** As proven in our Stage 1.5 Temporal Analysis (`04_stage1.5_temporal_analysis.md`), hardware components fail instantaneously ($0 \rightarrow 3$) without gradual degradation warning signs. Therefore, we **dropped all hardware lags/rolling means** and focused Stage 2 exclusively on the proven volatile metrics: Ping connectivity and active problem duration.

---

## 2. Group C Feature Provenance & Specification Table (Option C Architecture)

| Feature Name | Derived From | Window / Operator | Why It Helps ML | Assignment Alignment | ML Usage Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ping_status_binary_lag1`** | `ping_status_binary` | Lag 1 (`-4 hours`), **Option C (Truthful `NaN`)** | Immediate preceding network reachability state (`0=Reachable, 1=Unreachable`). Preserved as `NaN` at `Slot 1` where no history exists. | Q3 (Anomaly), Q6 (Failure) | **Core Training Feature** |
| **`ping_status_binary_lag2`** | `ping_status_binary` | Lag 2 (`-8 hours`), **Option C (Truthful `NaN`)** | Network reachability 8 hours ago. Combined with `lag1`, identifies connection flapping ($1 \rightarrow 0 \rightarrow 1$). Preserved as `NaN` at `Slot 1` & `Slot 2`. | Q3 (Anomaly), Q6 (Failure) | **Core Training Feature** |
| **`ping_timeout_rate_3slot`** | `ping_status_binary` | Rolling Mean (`12 hours`) | Measures short-term network instability (`0.33` = 1 timeout in 12h, `1.0` = down 12h straight). | Q6 (Failure Prediction) | **Core Training Feature** |
| **`ping_timeout_rate_6slot`** | `ping_status_binary` | Rolling Mean (`24 hours`) | Measures daily network availability trend to distinguish chronic issues from isolated blips. | Q6 (Failure Prediction) | **Core Training Feature** |
| **`problems_active_sum_6slot`** | `has_active_problem` | Rolling Sum (`24 hours`) | Quantifies sustained daily instability (`6` = unstable every single slot for 24 hours straight). | Q6 (Failure Prediction) | **Core Training Feature** |

> [!IMPORTANT]
> **Option C Separation of Concerns (Truthful `NaN` vs. Training Preprocessing):**  
> We enforce a clean architectural separation between Feature Engineering and Model Training:
> - **Feature Engineering Responsibility (Truthful Representation):** For `Slot 1` (`lag1`) and `Slot 1 & 2` (`lag2`), historical observations literally do not exist. Therefore, we **strictly preserve them as `NaN` (`Int64` nullable integer)** inside `features_stage2_temporal_v1.parquet`.
> - **Model Training Responsibility (Algorithm-Specific Preprocessing):** Downstream training pipelines decide how to handle these `NaN` boundary conditions based on the algorithm:
>   - *XGBoost / LightGBM:* Keep `NaN` (handled natively via sparsity-aware split finding).
>   - *Random Forest / Logistic Regression:* Impute (`0` or median) or drop `Slot 1` during pipeline preprocessing.

---

## 3. Quantitative Verification & Output Distributions

Running `stage2_temporal_features.py` successfully constructed all 5 Group C temporal features across all `45,756` observations (`0` dropped rows).

### Network Lags (Option C Truthful Distributions):
- **`ping_status_binary_lag1`:**
  - `Reachable (0)`: `44,750 (97.80%)`
  - `Unreachable (1)`: `760 (1.66%)`
  - **`NaN (Slot 1 / No History)`: `246 (0.54% — exactly 1 per server)`**
- **`ping_status_binary_lag2`:**
  - `Reachable (0)`: `44,511 (97.28%)`
  - `Unreachable (1)`: `753 (1.65%)`
  - **`NaN (Slot 1-2 / No History)`: `492 (1.08% — exactly 2 per server)`**

### Network Rolling Rates (`min_periods=1` prevents `NaN` generation at start):
- **`ping_timeout_rate_3slot` (`12h`):** `min = 0.0000`, `max = 1.0000`, `mean = 0.0166`
- **`ping_timeout_rate_6slot` (`24h`):** `min = 0.0000`, `max = 1.0000`, `mean = 0.0165`

### Active Problem Duration (`problems_active_sum_6slot` across 45,756 rows):
| Slots Active in Last 24 Hours | Observation Count | Percentage | Interpretation |
| :---: | :---: | :---: | :--- |
| **`0 slots`** | `42,097` | `92.00%` | Stable 24-hour operation |
| **`1 slot`** | `2,715` | `5.93%` | Single transient warning/timeout |
| **`2 slots`** | `614` | `1.34%` | Intermittent recurring issues |
| **`3 slots`** | `206` | `0.45%` | Unstable half-day (`12h`) |
| **`4 slots`** | `78` | `0.17%` | Sustained degradation (`16h`) |
| **`5 slots`** | `29` | `0.06%` | Persistent fault (`20h`) |
| **`6 slots`** | **`17`** | **`0.04%`** | **Chronic 24-hour continuous failure** |

---

## 4. Audit & Verification Check
- **Row Conservation:** Exact `45,756 rows` maintained (`0` rows dropped).
- **Column Expansion:** Expanded from `57` columns (`Stage 1`) to `62` columns (`Stage 2: 28 original + 34 engineered`).
- **Machine-Readable Metadata:** Updated [`datasets/feature_metadata_stage2.json`](file:///c:/Users/navad/ML_data/datasets/feature_metadata_stage2.json) with exact Group C Option C `NaN` specifications.
