# Master Data Dictionary & Canonical Column Specification

Every field below is strictly categorized by its epistemic origin (`[EVIDENCE]`, `[CONCLUSION]`, or `[RECOMMENDATION]`).

## Canonical Composite Observation Identity
| Column / Component | Origin | Type | ML Role | Meaning | Action / Engineering Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `machine_name` (`vm_name` / `server_name`) | `[EVIDENCE]` | string | Canonical Identity (Part 1) | Stable physical or virtual asset hostname | `[RECOMMENDATION]` Never drop; keep for debugging, validation, and explainable RCA alongside IP |
| `ip_address` (`vm_ip`) | `[EVIDENCE]` | string | Canonical Identity (Part 2) | IPv4 Network Address (1-to-1 match with `machine_name`) | `[RECOMMENDATION]` Primary relational join condition across tables |
| `monitoring_slot` | `[CONCLUSION]` | string | Canonical Identity (Part 3) | Derived 4-hour business interval (`Slot 02, 06, 10, 14, 18, 22`) | `[RECOMMENDATION]` Derive by bucketizing `timestamp` into 6 daily cycles to overcome inter-system clock jitter (`02:24` -> `Slot 02`) |

## Ping Status (`ping_status_export_20260702_mockup.csv`)
| Column | Origin | Type | Null % | Meaning | Action / Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `[EVIDENCE]` | int | 0% | Raw auto-increment sequence | `[RECOMMENDATION]` Drop during preprocessing |
| `status` | `[EVIDENCE]` | categorical | 0% | Reachability status (`Reachable` / `Unreachable`) | `[RECOMMENDATION]` Encode as binary feature `is_unreachable = (status == 'Unreachable')` |
| `timestamp` | `[EVIDENCE]` | datetime | 0% | Raw observation time (`YYYY-MM-DD HH:MM:SS`) | `[RECOMMENDATION]` Convert to `monitoring_slot` (`Slot 02..22`) |

## HPE iLO Health (`hpe_ilo_health_export_20260702_mockup.csv`)
| Column | Origin | Type | Null % | Meaning | Action / Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fans`, `cpu`, `memory`, `storage`, `temperature`, `power` | `[EVIDENCE]` | categorical | 0% across active rows | Component health (`OK`, `Warning`, `Critical`) | `[RECOMMENDATION]` Ordinal encode (`OK=0`, `Warning=1`, `Critical=2`) |
| `recorded_at` | `[EVIDENCE]` | datetime | 0% | Raw observation time | `[RECOMMENDATION]` Rename to `timestamp`, derive `monitoring_slot` |
| `current_problems` | `[EVIDENCE]` | string | 0% | Text diagnostic strings | `[RECOMMENDATION]` Extract specific boolean regex indicators |

## Dell iDRAC Health Extended (`dell_idrac_health_ext_export_20260702_mockup.csv`)
| Column | Origin | Type | Null % | Meaning | Action / Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `status`, `overall_status` | `[EVIDENCE]` | categorical | 0% | Overall hardware status (`OK`, `Warning`, `Critical`) | `[RECOMMENDATION]` Use as trigger anchor for lead-time target generation |
| `issues_detected` | `[EVIDENCE]` | string | 0% | Diagnostic telemetry | `[RECOMMENDATION]` Parse regex failure indicators |
| `timestamp` | `[EVIDENCE]` | datetime | 0% | Clean timestamp (0% parsing errors) | `[RECOMMENDATION]` Derive `monitoring_slot` |
