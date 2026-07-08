# Step 9: Column Profiling and Data Dictionary

## Question

Before building the unified ML dataset, we need to know:

```text
What does each column represent?
Is it an identifier, time column, feature, or text field?
Does it have missing values?
Should it be used directly in ML?
```

## Script Used

[column_profile.py](C:/Users/navad/ML_data/EDA/column_profile.py)

The script also generated:

[column_profile_summary.csv](C:/Users/navad/ML_data/EDA_docs/column_profile_summary.csv)

## Files Used

```text
ping_status_export_20260702_mockup.csv
hpe_ilo_health_export_20260702_mockup.csv
dell_idrac_health_ext_export_20260702_mockup.csv
```

---

# Ping Status Data Dictionary

| Column | Role | Missing | Unique | Keep? | Notes |
|---|---|---:|---:|---|---|
| `id` | Identifier | 0 | 246 | No direct ML use | Useful only as source record ID |
| `vm_name` | Identifier | 0 | 246 | Join/group key | Rename to `machine_name` |
| `vm_ip` | Identifier | 0 | 246 | Join/group key | Rename to `ip_address` |
| `status` | Categorical feature | 0 | 2 | Yes | Values: `Reachable`, `Unreachable` |
| `timestamp` | Time | 0 | 10,788 | Derived | Parse into `event_time` and `monitoring_slot` |

## Ping Feature Use

Useful raw feature:

```text
status
```

Useful derived features later:

```text
ping_reachable_flag
ping_unreachable_count_24h
ping_flap_count_24h
```

---

# HPE iLO Data Dictionary

| Column | Role | Missing | Unique | Keep? | Notes |
|---|---|---:|---:|---|---|
| `id` | Identifier | 0 | 15 | No direct ML use | Source record/server ID |
| `ip_address` | Identifier | 0 | 15 | Join/group key | Stable machine identity |
| `fans` | Categorical feature | 0 | 3 | Yes | Values include `OK`, `Degraded`, `Critical` |
| `cpu` | Categorical feature | 0 | 3 | Yes | Hardware CPU health |
| `memory` | Categorical feature | 0 | 2 | Yes | Hardware memory health |
| `storage` | Categorical feature | 0 | 3 | Yes | Storage health |
| `temperature` | Categorical feature | 0 | 3 | Yes | Temperature health |
| `power` | Categorical feature | 0 | 2 | Yes | Power health |
| `recorded_at` | Time | 0 | 2,790 | Derived | Parse into `event_time` and `monitoring_slot` |
| `server_name` | Identifier | 0 | 15 | Join/group key | Rename to `machine_name` |
| `current_problems` | Text feature | 0 | 20 | Yes, after extraction | Use for issue keywords and problem flags |

## HPE Feature Use

Useful raw features:

```text
fans
cpu
memory
storage
temperature
power
current_problems
```

Useful derived features later:

```text
hpe_hardware_degraded_count
hpe_hardware_critical_count
hpe_has_active_problem
hpe_temperature_severity
```

---

# Dell iDRAC Data Dictionary

| Column | Role | Missing | Unique | Keep? | Notes |
|---|---|---:|---:|---|---|
| `id` | Identifier | 0 | 26 | No direct ML use | Source record/server ID |
| `ip_address` | Identifier | 0 | 26 | Join/group key | Stable machine identity |
| `status` | Categorical feature | 0 | 1 | Low value | Always `OK` in this export |
| `issues_detected` | Text feature | 0 | 12 | Yes, after extraction | Useful for problem flags |
| `comments` | Text feature | 4,836 | 0 | No for this export | 100% missing |
| `timestamp` | Time | 0 | 744 | Derived | Parse into `event_time` and `monitoring_slot` |
| `overall_status` | Categorical feature | 0 | 3 | Yes | Values include `OK`, `Degraded`, `Critical` |
| `fans` | Categorical feature | 0 | 4 | Yes | Includes `NOT OK` in addition to normal statuses |
| `cpu` | Categorical feature | 0 | 3 | Yes | Includes `NOT OK` |
| `memory` | Categorical feature | 0 | 1 | Low value | Always `OK` in this export |
| `storage` | Categorical feature | 0 | 3 | Yes | Storage health |
| `temperature` | Categorical feature | 0 | 4 | Yes | Includes `NOT OK` |
| `power` | Categorical feature | 0 | 2 | Yes | Power health |
| `server_name` | Identifier | 0 | 26 | Join/group key | Rename to `machine_name` |
| `current_problems` | Text feature | 0 | 12 | Yes, after extraction | Useful for RCA and issue flags |

## Dell Feature Use

Useful raw features:

```text
overall_status
fans
cpu
storage
temperature
power
issues_detected
current_problems
```

Columns with limited value in this export:

```text
status
memory
comments
```

`status` and `memory` are constant in this export, so they add little model signal here.
`comments` is 100% missing.

---

# Feature Readiness Summary

## Identifier Columns

Use for joining and grouping, not direct ML modeling:

```text
id
vm_name / server_name
vm_ip / ip_address
```

## Time Columns

Use to create time features:

```text
timestamp
recorded_at
```

Derived features:

```text
event_time
monitoring_slot
hour
day_of_week
```

## Categorical ML Features

Use after encoding or severity mapping:

```text
ping_status
hpe_fans
hpe_cpu
hpe_memory
hpe_storage
hpe_temperature
hpe_power
dell_overall_status
dell_fans
dell_cpu
dell_storage
dell_temperature
dell_power
```

## Text Features

Use after extracting flags or keywords:

```text
issues_detected
current_problems
```

Example derived flags:

```text
has_fan_issue
has_power_issue
has_temperature_issue
has_storage_issue
has_cpu_issue
has_active_problem
```

## Current Conclusion

The raw columns are now understood well enough to answer:

```text
Which columns are identifiers?
Which columns are features?
Which columns are time fields?
Which columns need transformation?
Which columns have missing values?
```

This completes the final EDA bridge before building the unified monitoring dataset.
