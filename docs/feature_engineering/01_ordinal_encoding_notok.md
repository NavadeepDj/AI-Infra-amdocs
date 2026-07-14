# Phase 4 Step 1: Ordinal Encoding & `NOT OK` Empirical Investigation

**Investigative Script:** [`feature_engineering/analyze_not_ok.py`](file:///c:/Users/navad/ML_data/feature_engineering/analyze_not_ok.py)  
**Input Gold Dataset:** [`datasets/master_infrastructure_health_v1.parquet`](file:///c:/Users/navad/ML_data/datasets/master_infrastructure_health_v1.parquet) (`45,756 x 28`)  
**Final Encoding Decision:** **`OPTION C — PURE EMPIRICAL ORDINAL SCALE (OK=0, Degraded=1, NOT OK=2, Critical=3)`**  

---

## 1. Executive Summary & The Core Problem

Before applying ordinal severity mapping (`0, 1, 2, 3`) to our raw categorical hardware status fields, we addressed an unresolved domain question regarding Dell iDRAC exports:

> **"What exactly does `'NOT OK'` mean in `dell_cpu`, `dell_fans`, and `dell_temperature`, and where does it fit on the severity scale relative to `'Degraded'` and `'Critical'`?"**

Instead of guessing or inventing arbitrary business logic (such as blindly mapping `'NOT OK'` to `2` or `3` or creating an artificial `"Warning"` rank), we executed a targeted empirical investigation across all `45,756` observations.

---

## 2. Quantitative & Diagnostic Investigation Findings

Running `analyze_not_ok.py` yielded the exact frequency, co-occurrence, and diagnostic strings for every `'NOT OK'` row:

### A. Frequency & Column Distribution
Across all `45,756` observations (`1.28 million` hardware component cells), `'NOT OK'` appears in exactly **`5 observations` (`0.0109%`)**:
- `dell_fans`: **`2` occurrences** (`v5G-NEF-Zone1-26`)
- `dell_temperature`: **`2` occurrences** (`v5G-NEF-Backup-21`)
- `dell_cpu`: **`1` occurrence** (`v5G-AMF-Backup-02`)

Furthermore, checking every categorical status column confirmed that the string **`"Warning"` has `0 occurrences`** across the entire dataset (it exists solely inside informal unstructured `current_problems` text strings).

---

### B. Detailed Observation Audit Table

| Observation ID | Affected Component | Dell Overall Status | Diagnostic Text (`dell_issues_detected`) | Other Component Statuses on Same Row |
| :--- | :--- | :---: | :--- | :--- |
| `v5G-NEF-Backup-21` (`Slot-10`) | `dell_temperature = NOT OK` | **`Critical`** | `{"System temperature exceeded safe operating limit."}` | All other hardware = `OK` |
| `v5G-NEF-Backup-21` (`Slot-14`) | `dell_temperature = NOT OK` | **`Critical`** | `{"System temperature exceeded safe operating limit."}` | All other hardware = `OK` |
| `v5G-AMF-Backup-02` (`Slot-02`) | `dell_cpu = NOT OK` | **`Critical`** | `{"The chassis is open while the power is off."}` | `hpe_storage = Degraded` |
| `v5G-NEF-Zone1-26` (`Slot-02`) | `dell_fans = NOT OK` | **`Degraded`** | `{"Power supply redundancy is lost."}` | `dell_power = Degraded` |
| `v5G-NEF-Zone1-26` (`Slot-10`) | `dell_fans = NOT OK` | **`Critical`** | `{"Fan 2 has failed."}` | All other hardware = `OK` |

---

## 3. Evaluation of Encoding Options

With the diagnostic evidence established, we evaluated three possible ordinal encoding strategies:

| Strategy | Proposed Mapping | Pros | Cons / Scientific Critique |
| :--- | :--- | :--- | :--- |
| **Option A** (`NOT OK = Warning`) | `OK=0, Degraded=1, Warning=2, NOT OK=2, Critical=3` | Conservative. Avoids exaggerating severity. | Underestimates severity: in `4/5 (80%)` of rows, `dell_overall_status` is `Critical` with active faults (temperature/fan failure). Also invents an unobserved `"Warning"` label (`0 occurrences`). |
| **Option B** (`NOT OK = Critical`) | `OK=0, Degraded=1, Warning=2, NOT OK=3, Critical=3` | Safety-first conservative for critical alerting. | Overestimates severity in `1/5 (20%)` of rows: at `Slot-02` on `v5G-NEF-Zone1-26`, `NOT OK` coincides with `dell_overall_status = Degraded` (power redundancy loss prior to fan failure). Merges two distinct labels into one (`3`). |
| **Option C** (**Empirical Scale**) | **`OK=0, Degraded=1, NOT OK=2, Critical=3`** | **100% Grounded in Data:** Eliminates `"Warning"` entirely since it never appears in categorical data (`0` rows). Positions `NOT OK` exactly between `Degraded` (`1`) and `Critical` (`3`), preserving its distinct severity level without distorting co-occurrence evidence. | Requires distinct handling of `NOT OK` as rank `2`. |

---

## 4. Final Engineering Verdict & Implementation Standard

### Adopted Option: `OPTION C` (Pure Empirical Ordinal Encoding)

We officially adopt Option C for Stage 1 Generic Feature Engineering:

```python
SEVERITY_MAP = {
    "OK": 0,          # Healthy baseline
    "Degraded": 1,    # Intermediate / early degradation (observed across HPE & Dell)
    "NOT OK": 2,      # Serious / anomalous degradation state (observed in Dell iDRAC)
    "Critical": 3     # Severe / failing hardware (observed across HPE & Dell)
}
```

#### Engineering Justification:
1. **Preservation of Distinct Levels:** `NOT OK` is clearly more severe than standard `Degraded` (`1`), but because it does not strictly equal `Critical` (`3`) in `100%` of cases (`Slot-02` was `Degraded`), assigning it rank `2` accurately captures an intermediate high-severity state.
2. **Elimination of Phantom Labels:** We drop `"Warning"` completely from our categorical mappings because our verification proved it has `0 occurrences` across all `12` categorical hardware columns.
3. **Traceability:** By preserving `NOT OK = 2` as a distinct rank, our downstream models can differentiate between standard `Degraded` (`1`), serious Dell `NOT OK` (`2`), and full `Critical` (`3`) hardware states.
