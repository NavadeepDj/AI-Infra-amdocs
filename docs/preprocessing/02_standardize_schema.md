# Step 2: Canonical Schema Mapping Specification & Verification Log

**Script File:** [`preprocessing/standardize_schema.py`](file:///c:/Users/navad/ML_data/preprocessing/standardize_schema.py)  
**Execution Status:** `PASS`  

---

## 1. Objective & Engineering Rationale
Disparate vendor schemas (`vm_name` vs `server_name`, `vm_ip` vs `ip_address`, `timestamp` vs `recorded_at`) must be unified into a single canonical domain model before relational joining. We map all identity and temporal fields to canonical names and select only required features, dropping raw auto-incrementing `id` sequence numbers.

---

## 2. Canonical Field Mapping Table

| Canonical Name | Ping Status Source | HPE iLO Source | Dell iDRAC Ext Source | Domain Role |
| :--- | :--- | :--- | :--- | :--- |
| **`machine_name`** | `vm_name` | `server_name` | `server_name` | Primary entity identity (hostname / VM name) |
| **`ip_address`** | `vm_ip` | `ip_address` | `ip_address` | Network IPv4 address (relational join anchor) |
| **`event_time`** | `timestamp` | `recorded_at` | `timestamp` | Raw observation time (to be bucketized to `monitoring_slot`) |

---

## 3. Verified Execution Results

```text
=== Step 2: Schema Standardization ===
[OK] Ping Status schema standardized.
     Columns: ['machine_name', 'ip_address', 'event_time', 'status']
     Shape: (45756, 4)
[OK] HPE iLO Health schema standardized.
     Columns: ['machine_name', 'ip_address', 'event_time', 'fans', 'cpu', 'memory', 'storage', 'temperature', 'power', 'current_problems']
     Shape: (2790, 10)
[OK] Dell iDRAC Health Extended schema standardized.
     Columns: ['machine_name', 'ip_address', 'event_time', 'status', 'overall_status', 'fans', 'cpu', 'memory', 'storage', 'temperature', 'power', 'issues_detected']
     Shape: (4836, 12)
[SUCCESS] Step 2: Schema standardization complete.
```

---

## 4. Verification Verdict
**`PASS`** — Canonical schemas across all three tables successfully mapped to uniform `machine_name`, `ip_address`, and `event_time` boundaries without row loss.
