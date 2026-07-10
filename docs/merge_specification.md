# Canonical Dataset Merge Specification & Preprocessing Blueprint

## 1. Selected Base Exports (31-Day Aligned Window)
- `[EVIDENCE]` **Network Reachability Base:** `datasets/ping_status_export_20260702_mockup.csv` (`45,756` rows, `246` unique machines, `186` slots exactly).
- `[EVIDENCE]` **HPE iLO Hardware Base:** `datasets/hpe_ilo_health_export_20260702_mockup.csv` (`2,610` rows, `15` machines, `0` timestamp errors).
- `[EVIDENCE]` **Dell iDRAC Hardware Base:** `datasets/dell_idrac_health_ext_export_20260702_mockup.csv` (`4,524` rows, `26` machines, `0` timestamp errors).
- `[CONCLUSION]` **EXCLUDED DATASET:** `datasets/dell_idrac_health_export_20260702_mockup.csv` (`2,808` invalid date parsing rows (`MM/DD/YYYY` vs `YYYY-MM-DD`). NEVER use this regular file.

## 2. Canonical Composite Observation Identity
`[CONCLUSION]` All joins and grouping must strictly operate over the 3-part canonical observation key:
```text
machine_name  (e.g., 'esx-host-01')
  +  
ip_address    (e.g., '100.100.58.45')
  +  
monitoring_slot (e.g., '2026-06-03_Slot-06')
```
- **Why `monitoring_slot`?** `[EVIDENCE]` Raw timestamps across Ping (`02:00:00`), HPE (`02:24:11`), and Dell (`02:46:05`) exhibit up to 46 minutes of inter-system jitter. `[RECOMMENDATION]` Map any timestamp occurring in `[00:00 - 03:59]` -> `Slot 02`, `[04:00 - 07:59]` -> `Slot 06`, etc.

## 3. Handling Unmatched & Missing Records (Epistemic Separation)
- **Ping-Only Machines (`246 - 41 = 205` machines without iLO/iDRAC records):**
  - `[EVIDENCE]` These 205 IPs appear reliably in `ping_status` (`186` slots each) but have zero rows in `hpe_ilo` or `dell_idrac_ext`.
  - `[ASSUMPTION: Low Confidence]` They likely represent virtual instances (`VMs`) or unmonitored ESXi guest OS machines. The exact underlying reason is unknown from available telemetry.
  - `[RECOMMENDATION]` Retain all 205 machines in the master dataset; impute hardware component columns as `'Virtual_Instance'` or `-1` rather than dropping them.
- **Missing Consecutive Hardware Time Slots:**
  - `[RECOMMENDATION: Engineering Design Choice]` For occasional missing hardware monitoring slots within a physical server's timeline, apply Forward Fill (`ffill`) up to `3` consecutive slots (`12 hours`), since physical hardware health persists until reboot or repair. Clearly flag imputed cells with an `is_imputed = 1` boolean indicator.
