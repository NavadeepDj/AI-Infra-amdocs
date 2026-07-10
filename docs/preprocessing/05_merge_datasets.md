# Step 6: Lossless Left Outer Join & Telemetry Integration Log

**Script File:** [`preprocessing/merge_datasets.py`](file:///c:/Users/navad/ML_data/preprocessing/merge_datasets.py)  
**Execution Status:** `PASS`  

---

## 1. Core Engineering Philosophy: Preserve vs. Transform
> **"The purpose of Data Engineering is to faithfully integrate data from multiple sources without losing information. It should not resolve disagreements or derive business features. Those responsibilities belong to Feature Engineering."**  
> *Rule: Data Engineering must `Preserve`. Feature Engineering must `Transform`.*

Under this architecture:
1. **Zero Component Coalescing:** Vendor-specific columns (`hpe_cpu`, `dell_cpu`, `hpe_memory`, `dell_memory`, etc.) are preserved intact without calling `combine_first()`. Any inter-vendor disagreement (`99.53%` CPU match or `99.25%` Temperature match observed during EDA) is preserved for Phase 3 Feature Engineering (`hardware_cpu_disagreement_flag`, `hardware_cpu_worst_status`).
2. **Raw Timestamps Retained:** `event_time_ping`, `event_time_hpe`, and `event_time_dell` are kept alongside `monitoring_slot` so inter-system jitter (`02:24` vs `02:46` vs `02:47`) remains accessible.
3. **Factual Telemetry Labeling (`telemetry_source`):** Instead of making unproven assumptions (`Virtual/Unmonitored`), we strictly tag factual telemetry availability (`Ping Only`, `Ping + Dell`, `Ping + HPE + Dell`).
4. **Canonical Identity:** Every row receives a permanent, deterministic observation key: `machine_name | ip_address | monitoring_slot`.

---

## 2. Integrated Master Schema (`45,756 x 28`)

The merged dataset contains 28 distinct, non-overlapping attributes grouped by domain role:

| Column Group | Attributes | Description |
| :--- | :--- | :--- |
| **Primary Identifiers (`4`)** | `observation_id`, `machine_name`, `ip_address`, `monitoring_slot` | Deterministic composite keys guaranteeing 100% row uniqueness (`45,756` unique keys). |
| **Telemetry Metadata (`4`)** | `has_ping`, `has_hpe`, `has_dell`, `telemetry_source` | Boolean availability flags (`True/False`) and factual source tags for fast downstream filtering. |
| **Ping Telemetry (`2`)** | `event_time_ping`, `ping_status` | Network reachability timestamp and status (`Reachable` / `Unreachable`). |
| **HPE Hardware Telemetry (`8`)** | `event_time_hpe`, `hpe_fans`, `hpe_cpu`, `hpe_memory`, `hpe_storage`, `hpe_temperature`, `hpe_power`, `hpe_current_problems` | Physical health metrics recorded by HPE iLO (`2,790` non-null observations across `15` servers). |
| **Dell Hardware Telemetry (`10`)** | `event_time_dell`, `dell_status`, `dell_overall_status`, `dell_fans`, `dell_cpu`, `dell_memory`, `dell_storage`, `dell_temperature`, `dell_power`, `dell_issues_detected` | Physical health metrics recorded by Dell iDRAC Extended (`4,836` non-null observations across `26` servers). |

---

## 3. Verified Execution Results

```text
=== Step 6: Left Outer Joining & Integrating Telemetry ===
[OK] Master Dataset Shape: (45756, 28) (Expected: 45756, 28)
[PASS] Row Count Conservation: Exactly 45756 rows preserved without duplication or loss.
[PASS] Unique Canonical Identity: 100% unique observation_ids generated across all rows.
[PASS] Telemetry Source Breakdown:
  - {'Ping Only': 40920, 'Ping + HPE + Dell': 2790, 'Ping + Dell': 2046}
[SUCCESS] Step 6: Dataset merging and preservation complete.
```

---

## 4. Verification Verdict
**`PASS`** — Left Outer Join successfully executed over `machine_name + ip_address + monitoring_slot`. Exactly `45,756` rows (`246 machines * 186 slots`) and `28` preserved columns generated without data loss or unintended coalescence.
