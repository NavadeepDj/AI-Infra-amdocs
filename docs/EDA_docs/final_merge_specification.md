# Final Merge Specification

## Purpose

This document defines how the final unified monitoring dataset should be created.

This is not the merge code yet.

It is the design decision produced by EDA.

---

# Data Engineering Flow

The merge should be treated as a small data engineering pipeline:

```text
Raw CSV Files
     |
     v
Schema Standardization
     |
     v
Canonical Source Tables
     |
     v
Create monitoring_slot
     |
     v
LEFT JOIN on machine_name + ip_address + monitoring_slot
     |
     v
Unified Gold Monitoring Dataset
     |
     v
Feature Engineering
```

This separation is important:

```text
Raw data       = original CSV columns
Canonical data = standardized source-level columns
Gold data      = final merged monitoring table
```

---

# Primary Dataset

Use Ping as the base dataset.

Reason:

```text
Ping contains the full monitoring inventory.
```

Observed:

```text
Ping machines      : 246
Dell iDRAC machines: 26
HPE iLO machines   : 15
```

So the merge should start from Ping and enrich it with hardware data where available.

---

# Schema Standardization

Before merging, standardize column names.

The goal is to convert each raw CSV into a canonical source table.

## Ping

| Original Column | Standard Column |
|---|---|
| `vm_name` | `machine_name` |
| `vm_ip` | `ip_address` |
| `timestamp` | `event_time` |
| `status` | `ping_status` |

## HPE iLO

| Original Column | Standard Column |
|---|---|
| `server_name` | `machine_name` |
| `ip_address` | `ip_address` |
| `recorded_at` | `event_time` |
| `fans` | `hpe_fans` |
| `cpu` | `hpe_cpu` |
| `memory` | `hpe_memory` |
| `storage` | `hpe_storage` |
| `temperature` | `hpe_temperature` |
| `power` | `hpe_power` |
| `current_problems` | `hpe_current_problems` |

## Dell iDRAC

| Original Column | Standard Column |
|---|---|
| `server_name` | `machine_name` |
| `ip_address` | `ip_address` |
| `timestamp` | `event_time` |
| `status` | `dell_status` |
| `overall_status` | `dell_overall_status` |
| `fans` | `dell_fans` |
| `cpu` | `dell_cpu` |
| `memory` | `dell_memory` |
| `storage` | `dell_storage` |
| `temperature` | `dell_temperature` |
| `power` | `dell_power` |
| `issues_detected` | `dell_issues_detected` |
| `comments` | `dell_comments` |
| `current_problems` | `dell_current_problems` |

---

# Unified Dataset Schema

The expected unified table should look like this at a high level:

| Column | Source | Type | Role |
|---|---|---|---|
| `machine_name` | Ping | String | Key |
| `ip_address` | Ping | String | Key |
| `monitoring_slot` | Derived | Datetime | Key |
| `ping_event_time` | Ping | Datetime | Source timestamp |
| `ping_status` | Ping | Categorical | ML feature |
| `hpe_event_time` | HPE iLO | Datetime | Source timestamp |
| `hpe_fans` | HPE iLO | Categorical | ML feature |
| `hpe_cpu` | HPE iLO | Categorical | ML feature |
| `hpe_memory` | HPE iLO | Categorical | ML feature |
| `hpe_storage` | HPE iLO | Categorical | ML feature |
| `hpe_temperature` | HPE iLO | Categorical | ML feature |
| `hpe_power` | HPE iLO | Categorical | ML feature |
| `hpe_current_problems` | HPE iLO | Text | RCA / feature extraction |
| `dell_event_time` | Dell iDRAC | Datetime | Source timestamp |
| `dell_status` | Dell iDRAC | Categorical | Low-value feature in this export |
| `dell_overall_status` | Dell iDRAC | Categorical | ML feature |
| `dell_fans` | Dell iDRAC | Categorical | ML feature |
| `dell_cpu` | Dell iDRAC | Categorical | ML feature |
| `dell_memory` | Dell iDRAC | Categorical | Low-value feature in this export |
| `dell_storage` | Dell iDRAC | Categorical | ML feature |
| `dell_temperature` | Dell iDRAC | Categorical | ML feature |
| `dell_power` | Dell iDRAC | Categorical | ML feature |
| `dell_issues_detected` | Dell iDRAC | Text | RCA / feature extraction |
| `dell_comments` | Dell iDRAC | Text | 100% missing in this export |
| `dell_current_problems` | Dell iDRAC | Text | RCA / feature extraction |

Note:

```text
Source event times should be preserved separately.
Do not collapse Ping, HPE, and Dell timestamps into one raw timestamp column.
```

The shared time key is:

```text
monitoring_slot
```

---

# Merge Key

Use:

```text
machine_name + ip_address + monitoring_slot
```

EDA evidence:

```text
Each machine maps to one IP.
Each IP maps to one machine.
Every machine has exactly 186 monitoring slots.
No duplicate machine + IP + monitoring_slot records were found.
```

---

# Join Type

Use left joins.

```text
Ping
LEFT JOIN HPE iLO
LEFT JOIN Dell iDRAC
```

Reason:

```text
Ping is the complete inventory.
HPE and Dell are hardware-monitoring subsets.
```

---

# Merge Logic

```text
1. Load Ping, HPE, and Dell aligned 20260702 files.
2. Parse timestamps.
3. Create monitoring_slot.
4. Rename columns using the standard names above.
5. Start with Ping.
6. Left join HPE on machine_name + ip_address + monitoring_slot.
7. Left join Dell on machine_name + ip_address + monitoring_slot.
8. Preserve source-specific columns.
```

---

# Missing vs Unavailable

After the left join, many Ping rows will not have HPE or Dell values.

This does not always mean the data is missing.

It can mean:

```text
The machine is not monitored by that hardware system.
```

For example:

```text
Ping machine exists
HPE columns are null
```

Possible meaning:

```text
Not applicable: this machine has no HPE iLO record.
```

This is different from:

```text
The HPE collector failed to report a machine that should have HPE data.
```

Recommended approach:

```text
Create source availability flags:

has_hpe_record
has_dell_record
```

Then missing hardware fields can be interpreted correctly during feature engineering.

---

# Conflict Strategy

HPE and Dell overlap on 15 machines.

They have the same machine names and IPs, but their health values are not perfectly identical.

Observed component match rates:

| Component | Match Rate |
|---|---:|
| CPU | 99.53% |
| Memory | 99.93% |
| Temperature | 99.25% |
| Power | 99.46% |
| Fans | 99.57% |
| Storage | 99.10% |

Therefore:

```text
Do not drop either HPE or Dell during the merge.
Preserve both vendor observations first.
Create unified hardware features later.
```

---

# Future Unified Features

After merging, create derived features such as:

```text
hardware_cpu_worst_status
hardware_memory_worst_status
hardware_temperature_worst_status
hardware_power_worst_status
hardware_fans_worst_status
hardware_storage_worst_status
hardware_source_disagreement_flag
num_degraded_components
num_critical_components
has_active_hardware_problem
```

---

# Expected Output

The unified monitoring dataset should contain one row per:

```text
machine_name + ip_address + monitoring_slot
```

Meaning:

```text
One machine
+
One monitoring cycle
=
One infrastructure health observation
```

Because Ping has:

```text
246 machines x 186 monitoring slots = 45,756 rows
```

the unified dataset should keep approximately:

```text
45,756 rows
```

after left joining HPE and Dell, assuming all Ping observations are preserved and no corrupt rows are filtered.

---

# Merge Validation Checks

After implementing the merge, validate the output before using it for feature engineering.

Expected checks:

| Validation | Expected Result |
|---|---|
| Output rows | Approximately `45,756` |
| Lost Ping rows | `0` |
| Duplicate `machine_name + ip_address + monitoring_slot` keys | `0` |
| Matched HPE rows | `2,790` |
| Matched Dell rows | `4,836` |
| Rows with HPE available | `2,790` |
| Rows with Dell available | `4,836` |

Recommended validation fields:

```text
has_hpe_record
has_dell_record
```

These flags help separate:

```text
not monitored by source
```

from:

```text
unexpected missing data
```

---

# EDA Handoff

At this point, EDA has answered:

```text
Who is monitored?        machine_name + ip_address
When is it monitored?    monitoring_slot
How often?               every 4 hours
How should we merge?     Ping left join HPE and Dell
What should we preserve? source-specific hardware fields
```

The next phase is:

```text
Data Engineering: Build the Gold Monitoring Dataset
```
