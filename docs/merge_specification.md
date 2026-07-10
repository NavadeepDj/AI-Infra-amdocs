# Dataset Merge Specification & Preprocessing Blueprint

## 1. Selected Master Files
- **Network Reachability:** `datasets/ping_status_export_20260703_mockup.csv` (`45,756` rows, `246` unique VMs).
- **HPE Hardware Health:** `datasets/hpe_ilo_health_export_20260703_mockup.csv` (`2,610` rows, `15` unique IPs, `0` timestamp errors across `29` dates).
- **Dell Hardware Health:** `datasets/dell_idrac_health_ext_export_20260703_mockup.csv` (`4,524` rows, `26` unique IPs, `0` timestamp errors across `29` dates).
- **CRITICAL EXCLUSION:** DO NOT USE `dell_idrac_health_export_20260703_mockup.csv` (regular file has `2,808` corrupted date strings and irregular intervals).

## 2. Alignment Strategy
- **Join Entity Keys:** `ping_status.vm_ip` == `hpe_ilo_health.ip_address` == `dell_idrac_health_ext.ip_address`.
- **Time Alignment Grid:** All 3 files operate on a regular 4-hour monitoring interval (`02:00, 06:00, 10:00, 14:00, 18:00, 22:00 UTC`). Round timestamps to the nearest 4-hour interval (`pd.Series.dt.round('4h')`) and perform an outer join across `['ip_address', 'timestamp_grid']`.

## 3. Handling Missing & Unmatched Records
- VMs present only in `ping_status` (no physical iDRAC/iLO hardware metrics) represent virtualized instances or ESXi guests. For these records, impute hardware component flags as `'Virtual_Instance'` or `0`.
- For missing consecutive hardware time slots, apply Forward Fill (`ffill`) up to `3` slots (`12 hours`).
