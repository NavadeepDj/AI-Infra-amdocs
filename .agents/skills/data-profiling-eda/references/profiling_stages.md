# The 8-Stage Data Profiling Methodology

This reference document explains the exact mindset, execution order, and analytical checks required for every stage of the 8-Stage Data Profiling methodology for infrastructure health and time-series monitoring datasets.

---

## Stage 1: Dataset Overview
**Goal:** Establish high-level structural metrics for every ingested dataset.

### Metrics to Record:
- **Number of Rows & Columns**
- **Unique Machines / Assets (`machine_name`, `server_name`)**
- **Unique IP Addresses (`vm_ip`, `ip_address`)**
- **Start Date & End Date**
- **Expected Monitoring Interval (e.g., 4 Hours)**
- **Duplicate Row Count**

### Why It Matters:
If one monitoring dataset has `45,756` rows (`Ping`) while another has only `2,790` (`HPE iLO`), this immediate contrast highlights differing asset coverage (`246` VMs vs `15` servers) and guides our left-join merge strategy.

---

## Stage 2: Schema Analysis
**Goal:** Classify every column into its conceptual role before preprocessing.

### Standard Roles:
- **Identifier**: `id`, `vm_name`, `server_name`, `vm_ip`, `ip_address`. Essential for joins, grouping, and per-VM historical baselines. Excluded from direct ML training to avoid memorization.
- **Time Column**: `timestamp`, `recorded_at`. Used to derive standardized monitoring slots (`monitoring_slot`), time-of-day features (`hour`, `day_of_week`), and rolling trends.
- **Categorical Feature**: `status`, `overall_status`, `fans`, `cpu`, `memory`, `storage`, `temperature`, `power`. Must be checked for distinct categories and mapped to ordinal severity scores (`OK=0, Warning=1, Critical=2`).
- **Numerical Feature**: `cpu_usage`, `memory_usage`, `disk_usage`, `datastore_latency`. Used for lag features and rolling slopes.
- **Text Feature**: `current_problems`, `issues_detected`, `comments`. Used to extract boolean flags (`has_active_problem`, `problem_type_fan`).

---

## Stage 3: Data Quality Assessment
**Goal:** Inspect data cleanliness and detect anomalies that could corrupt ML models.

### Key Checks:
1. **Missing Values**: Calculate total missing rows and percentage per column. Never assume a missing hardware metric means the server is healthy. Create explicit missing data flags (`data_missing_flag = 1`).
2. **Duplicate Rows**: Check for duplicate entries occurring for the same `machine_name` + `timestamp`.
3. **Invalid Categories / Out-of-Range Values**: Verify that categorical status columns only contain legitimate domain values (e.g., checking that `temperature` contains only `OK`, `Warning`, or `Critical`, and detecting erroneous entries like `"Very Bad"`).

---

## Stage 4: Value Distribution & Class Imbalance
**Goal:** Understand target class distributions and numerical spread across all health metrics.

### Key Checks:
- Calculate frequency distribution for categorical health columns.
- Notice class imbalances: Infrastructure health datasets are almost always heavily imbalanced (`~98.5%` Normal / Reachable vs `~1.5%` Abnormal / Unreachable).
- Document this imbalance explicitly, as it dictates the choice of anomaly detection algorithms (**Isolation Forest**) and evaluation metrics (**PR-AUC**, **F1-Score** instead of raw accuracy).

---

## Stage 5: Time-Series & Monitoring Frequency Analysis
**Goal:** Understand temporal cadence, asynchronous scheduling offsets, and observation gaps.

### Key Checks:
1. **Discover Monitoring Intervals**: Calculate time differences (`diff()`) between consecutive observations per asset to confirm the regular schedule (e.g., 4-hour monitoring cycles).
2. **Account for Asynchronous Execution**: Recognize that different tools execute at slightly different minutes within the same window (`Ping` at `02:24`, `Dell iDRAC` at `02:45`, `HPE iLO` at `02:46`).
3. **Derive Standardized Monitoring Slots**: Standardize all timestamps into discrete 4-hour observation slots:
   ```python
   # Formula to align timestamps around 02:00, 06:00, 10:00, 14:00, 18:00, 22:00
   monitoring_slot = (pd.to_datetime(timestamp) - pd.Timedelta(hours=2)).dt.floor('4h') + pd.Timedelta(hours=2)
   ```
4. **Identify Missing Intervals**: Check if an asset misses an expected monitoring slot (e.g., slot `14:00` is missing between `10:00` and `18:00`). Flag missing slots to prevent artificial gaps in rolling calculations.

---

## Stage 6: Relationship & Asset Identity Analysis
**Goal:** Establish primary keys and understand how datasets relate across the infrastructure.

### Key Checks:
1. **Verify 1-to-1 Mapping**: Confirm that every `machine_name` maps to exactly one `ip_address`, and vice versa.
2. **Verify Cross-Source Asset Overlap**: Check which servers exist in multiple datasets (e.g., comparing `v5G-AMF-Backup-02` across `Ping`, `iLO`, and `iDRAC`).
3. **Establish Unified Primary Key**: Because each row represents one machine at one monitoring instant, the composite primary key for merging across all monitoring sources is:
   `machine_name + ip_address + monitoring_slot`

---

## Stage 7: Data Consistency & Cross-Tool Validation
**Goal:** Detect conflicting signals across different monitoring tools within the exact same monitoring slot.

### Key Checks:
- Check cases where `Ping Status` = `Reachable` but `HPE iLO` / `Dell iDRAC` = `Critical` (or vice versa).
- Do not treat these conflicts as data errors; they represent vital diagnostic reality:
  - **Ping OK + Hardware Bad**: Early physical degradation before network failure occurs.
  - **Ping Down + Hardware OK**: Network, OS, or application crash without physical hardware failure.
- Document these cross-source states as potential engineered features (`ping_down_but_hardware_ok`).

---

## Stage 8: ML Readiness Assessment
**Goal:** Finalize the preprocessing, feature engineering, and imputation blueprint.

### Preprocessing Blueprint Checklist:
- [ ] Standardize column names across tools (`machine_name`, `ip_address`, `event_time`).
- [ ] Convert raw `timestamp` columns into `monitoring_slot`.
- [ ] Left-join hardware and ESXi telemetry onto the widest base dataset (`Ping`).
- [ ] Create missingness indicators (`data_missing_flag = 1`) and fill missing numerical gaps using within-VM forward-fill (`ffill()`).
- [ ] Encode ordinal categorical health statuses into numeric severity scores (`fans_severity`, `temperature_severity`).
- [ ] Extract boolean flags from text fields (`has_active_problem`, `problem_type_fan`).
- [ ] Construct rolling and lag features per VM (`cpu_rolling_mean_24h`, `ping_unreachable_count_24h`, `cpu_lag_1`) grouped by `machine_name`.
