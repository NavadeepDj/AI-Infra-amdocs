---
name: data-profiling-eda
description: Systematic 8-Stage Exploratory Data Analysis (EDA) and Data Profiling for infrastructure health monitoring and time-series datasets. Triggers when the user asks to perform EDA, data profiling, data understanding, inspect new datasets, or run diagnostic scripts on monitoring data.
---

# Data Profiling & Exploratory Data Analysis (EDA) Skill

This skill provides a systematic, production-grade 8-Stage Data Profiling framework specifically tailored for time-series infrastructure health monitoring data (`Ping Status`, `HPE iLO`, `Dell iDRAC`, `ESXi SNMP`, etc.).

Whenever you are tasked with understanding a dataset, performing EDA, checking data quality, or preparing data for ML pipelines, **you MUST follow this exact 8-Stage workflow and leverage the pre-existing diagnostic scripts inside `c:\Users\navad\ML_data\EDA\` before drawing conclusions.**

---

## 1. The 8-Stage Data Profiling Workflow

Do not skip stages. Execute them sequentially to build up evidence from structural overview to ML feature readiness:

1. **Dataset Overview**: Check row count, column count, unique assets (`machine_name`, `server_name`), unique IP addresses (`vm_ip`, `ip_address`), and exact date boundaries (`start_date`, `end_date`).
2. **Schema Analysis**: Classify columns into `Identifiers` (not used as direct ML features), `Time Columns` (used to derive monitoring slots), `Numerical/Categorical Features`, and `Text Features`.
3. **Data Quality Assessment**: Identify missing values (count and percentage), duplicate rows (same machine + same timestamp), and invalid categories/out-of-range values.
4. **Value Distribution**: Check categorical class distributions (especially monitoring health imbalances like `Reachable` vs `Unreachable` or `OK` vs `Critical`) and numerical outlier ranges.
5. **Time-Series Analysis**: Verify exact monitoring frequencies (`4-hour intervals`), discover timestamp offsets across tools (`02:24` vs `02:45`/`02:46`), derive `monitoring_slot` buckets (`02:00, 06:00, 10:00, 14:00, 18:00, 22:00`), and check for missing observation intervals.
6. **Relationship Analysis**: Validate mapping between identifiers (e.g. verifying 1-to-1 relationship between `machine_name` and `ip_address`), check asset overlap across different datasets (`Ping` vs `iLO` vs `iDRAC`), and establish the composite primary key: `machine_name + ip_address + monitoring_slot`.
7. **Data Consistency**: Cross-reference conflicting health states between tools across identical monitoring slots (e.g. `Ping = Reachable` while `iLO = Power Failure`).
8. **ML Readiness Assessment**: Categorize all columns (`Identifier`, `Time Feature`, `Categorical ML Feature`, `Numerical ML Feature`, `Target/Label Candidate`) and specify exact preprocessing and encoding rules.

*(For detailed explanations of each stage, read `references/profiling_stages.md` using `view_file`.)*

---

## 2. Leveraging the `EDA/` Diagnostic Scripts

Instead of writing one-off scripts from scratch every time, always check and run the specialized scripts available in `c:\Users\navad\ML_data\EDA\`:

| Script Name | Purpose & When to Run |
|---|---|
| `data_understanding.py` | Quick summary of datasets and basic loading sanity check. Run in Stage 1. |
| `time_range.py` | Calculates minimum and maximum timestamps and overall time span. Run in Stage 1/5. |
| `unique_machines.py` | Counts unique server names, VM names, and IP addresses. Run in Stage 1/6. |
| `machine_ip_relationship.py` | Validates whether machine names map 1-to-1 or 1-to-many to IP addresses. Run in Stage 6. |
| `hpe_dell_redundancy_check.py` | Checks if HPE and Dell monitor distinct servers or redundant sets. Run in Stage 6. |
| `machine_set_comparison.py` | Compares machine overlaps across `Ping`, `HPE iLO`, and `Dell iDRAC`. Run in Stage 6. |
| `monitoring_frequency.py` | Evaluates timestamp intervals and derives `monitoring_slot` (`(timestamp - 2h).floor('4h') + 2h`). Run in Stage 5. |
| `timeline_validation.py` | Checks exact cycle alignments and validates asynchronous offsets. Run in Stage 5/7. |
| `column_profile.py` | Full profiling of missing counts, unique counts, sample values, and ML roles. Writes `column_profile_summary.csv`. Run in Stage 2/3/8. |

### Running & Adapting Scripts
- If analyzing new CSV files or updated exports, check the `DATA_FOLDER` and `datasets` configuration dictionary inside the script first. If needed, create an adapted script or pass parameters to run these exact diagnostic checks on the target files.
- Always execute scripts via `run_command` (e.g. `python c:\Users\navad\ML_data\EDA\column_profile.py`).

---

## 3. Core Architectural Rules for Infrastructure Data

When profiling infrastructure telemetry, adhere to these mandatory principles:
- **Never join on exact `timestamp`**: Monitoring tools trigger asynchronously within the same scheduled window (`02:24` vs `02:45`). Always derive and join on `monitoring_slot` (`machine_name + ip_address + monitoring_slot`).
- **Treat rows as Time-Series Observations**: A row represents one asset at one timestamp (`Asset + Time`), not a unique server inventory item.
- **Do not drop missing data blindly**: In infrastructure monitoring, a missing hardware record or unmonitored slot is a strong signal (`data_missing_flag = 1`). Categorical unknowns should be filled as `Unknown`, never `OK`.
- **Always preserve identifiers for baselines**: `machine_name` and `ip_address` must be kept during EDA and grouping to calculate per-VM rolling statistics and baselines, even if excluded from generalized ML training.

---

## 4. Expected Deliverables

When completing an EDA task using this skill, output clear, structured findings formatted in GitHub Flavored Markdown:
1. **Executive Summary Table**: Summarizing rows, columns, unique machines, monitoring cycles, and date ranges across all datasets.
2. **Data Quality & Anomaly Report**: Documenting missing percentages, duplicate checks, and timestamp anomalies.
3. **Merge & Alignment Strategy**: Explicitly defining primary keys and how datasets should be combined.
4. **ML Readiness & Data Dictionary Table**: Mapping every column to its recommended data type and ML role (`Feature`, `Identifier`, `Target`, `Time Slot`).
