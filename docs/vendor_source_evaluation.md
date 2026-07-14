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
3. **Sensor Disagreement Context:** In real-world deployments, vendor disagreements could indicate telemetry latency, calibration differences, or emerging hardware issues. In this mock dataset, disagreements (`hardware_*_disagreement_flag`) are retained for explainability and vendor consistency audits but are not assumed to represent real infrastructure behavior.

---

## 4. Architectural Blueprint for Phase 4 (Feature Engineering)
By keeping both vendor sources, Phase 4 Feature Engineering will construct two distinct layers for every physical component (`$COMP \in [cpu, memory, storage, fans, temperature, power]$`):

1. **`hardware_{comp}_worst_status` (Canonical ML Training Feature):**
   ```python
   # Take the maximum severity rank between HPE and Dell across any observation
   df[f"hardware_{comp}_worst_status"] = df[[f"hpe_{comp}", f"dell_{comp}"]].apply(max_severity, axis=1)
   ```
2. **`hardware_{comp}_disagreement_flag` (Explainability & Diagnostics Only — NOT for ML):**
   ```python
   # Binary flag = 1 when both exist and do not match
   df[f"hardware_{comp}_disagreement_flag"] = (df["has_hpe"] & df["has_dell"] & (df[f"hpe_{comp}"] != df[f"dell_{comp}"])).astype(int)
   ```

---

## 5. Design Limitation & ML Exclusion Policy
> [!IMPORTANT]
> **Design Limitation:** The vendor disagreement features (`hardware_*_disagreement_flag`) are intentionally **excluded** from the ML training feature set.
> 
> **Rationale:** Based on the available evidence from this mock dataset, disagreement features are excluded from the initial ML feature set because their semantic meaning cannot be validated (i.e., we cannot prove whether they reflect real hardware divergence or synthetic generation randomness). They are retained for explainability and can be re-evaluated if authoritative production metadata or ground-truth labels become available. This ensures our ML models learn from grounded, verifiable component severities (`hardware_{comp}_worst_status`, `critical_component_count`, etc.) rather than synthetic noise.
