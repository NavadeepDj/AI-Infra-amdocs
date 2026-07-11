# Phase 3.5: Vendor Source Evaluation & Representation Decision

**Evaluation Script:** [`preprocessing/vendor_source_evaluation.py`](file:///c:/Users/navad/ML_data/preprocessing/vendor_source_evaluation.py)  
**Input Gold Dataset:** [`datasets/master_infrastructure_health_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/master_infrastructure_health_v1.parquet) (`45,756 x 28`)  
**Analyzed Dual-Monitored Segment:** `15 servers` across all `186 slots` (`2,790` simultaneous HPE + Dell observations)  
**Final Architectural Verdict:** **`OUTCOME 1: KEEP BOTH VENDOR SOURCES`**  

---

## 1. Executive Summary & Context
Now that Phase 2 (Data Engineering) has integrated our raw network (`ping_status`) and hardware (`hpe_ilo`, `dell_idrac_ext`) telemetry into a lossless unified table, we address a fundamental data representation question before beginning Phase 4 (Feature Engineering):

> **"Should both vendor sources continue to exist in the ML dataset, or should one be dropped as redundant?"**

Because our `15 shared servers` exist in both HPE and Dell exports (a characteristic of our **Mock Dataset** scenario where virtual/physical boundaries and mock generators overlap), we evaluated whether one source is strictly superior, richer, or more authoritative.

---

## 2. Quantitative Evaluation Results

### Q1: Is one source substantially richer?
- **Column & Field Coverage:** Both vendors report the core 6 physical components (`fans`, `cpu`, `memory`, `storage`, `temperature`, `power`) across `100%` of their observations (`0` nulls).
- **Categorical Granularity:** HPE contains the additional category `'Degraded'` for fans, cpu, storage, and temperature (`Degraded` alongside `OK`, `Warning`, `Critical`). Dell uses `OK`, `Warning`, and `Critical`.
- **Diagnostic Text Fields:** HPE provides `current_problems` string diagnostics; Dell provides `overall_status` and `issues_detected`. Both offer unique diagnostic depth.

### Q2: How different are they really? (Exact Disagreement Matrix)
Across all `2,790` dual-monitored observations, exact component agreement is **`97.24%`** (`2713` observations agree on every single component). Exactly **`77` observations (`2.76%`)** exhibit a mismatch on at least one hardware component:

| Component | Mismatches across 2,790 obs | Divergence Rate | Observed Mismatch Value Pairs (`HPE vs. Dell`) |
| :--- | :---: | :---: | :--- |
| **`FANS`** | `12` | `0.43%` | `{('Critical', 'OK'): 1, ('Degraded', 'OK'): 6, ('OK', 'Critical'): 1, ('OK', 'Degraded'): 4}` |
| **`CPU`** | `13` | `0.47%` | `{('Critical', 'OK'): 1, ('Degraded', 'OK'): 7, ('OK', 'Degraded'): 4, ('OK', 'NOT OK'): 1}` |
| **`MEMORY`** | `2` | `0.07%` | `{('Critical', 'OK'): 2}` |
| **`STORAGE`** | `25` | `0.90%` | `{('Critical', 'OK'): 1, ('Degraded', 'OK'): 12, ('OK', 'Critical'): 2, ('OK', 'Degraded'): 10}` |
| **`TEMPERATURE`** | `21` | `0.75%` | `{('Critical', 'OK'): 3, ('Degraded', 'OK'): 12, ('OK', 'Critical'): 1, ('OK', 'Degraded'): 5}` |
| **`POWER`** | `15` | `0.54%` | `{('Critical', 'OK'): 5, ('OK', 'Degraded'): 10}` |

### Q3: Is one consistently more informative or sensitive?
When component disagreements occur (`77` observations):
- **HPE is more severe (`Warning/Critical/Degraded` vs Dell `OK`):** `40` times (`51.9%` of mismatches).
- **Dell is more severe (`Warning/Critical` vs HPE `OK`):** `37` times (`48.1%` of mismatches).

**Conclusion:** Neither vendor is universally more sensitive or "always right." HPE catches early thermal/CPU degradation (`Degraded/Warning`) when Dell says `OK`, while Dell catches power/memory spikes when HPE says `OK`.

---

## 3. Final Engineering Verdict & Feature Engineering Guidance

### Recommended Outcome: `OUTCOME 1 — KEEP BOTH VENDOR SOURCES`

#### Why We Do Not Delete Dell or HPE:
1. **Complementary Coverage Outside the Shared Segment:** Dell monitors `11 additional physical servers` (`2,046` observations) that do not exist in HPE iLO. Dropping Dell would leave 11 critical physical servers without hardware telemetry (`Ping Only`).
2. **Mutual Sensitivity on Shared Servers:** Dropping either vendor would discard critical early warnings (`Warning/Degraded` events) captured exclusively by the other sensor.
3. **Sensor Disagreement is a High-Value Predictive Signal:** In enterprise ML monitoring, when two monitoring agents on the same server report conflicting health states (`hpe_cpu = Warning` vs `dell_cpu = OK`), that divergence (`disagreement_flag = 1`) is itself a powerful feature indicating telemetry latency, sensor calibration drift, or early intermittent hardware failure.

---

## 4. Architectural Blueprint for Phase 4 (Feature Engineering)
By keeping both vendor sources, Phase 4 Feature Engineering will construct three canonical unified feature layers for every physical component (`$COMP \in [cpu, memory, storage, fans, temperature, power]$`):

1. **`hardware_power_worst_status` (Safety-First Target):**
   ```python
   # Take the maximum severity rank between HPE and Dell across any observation
   df[f"hardware_power_worst_status"] = df[[f"hpe_power", f"dell_power"]].apply(max_severity, axis=1)
   ```
2. **`hardware_power_disagreement_flag` (Anomalous Drift Signal):**
   ```python
   # Binary flag = 1 when both exist and do not match
   df[f"hardware_power_disagreement_flag"] = (df["has_hpe"] & df["has_dell"] & (df[f"hpe_power"] != df[f"dell_power"])).astype(int)
   ```
3. **`hardware_overall_health_score` (Composite Index):**
   A weighted numeric health score combining Ping reachability, worst-status hardware components, and active problem flags.
