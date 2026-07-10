# Master Data Dictionary & Column Specification

## Ping Status (`ping_status`)
| Column | Type | ML Role | Null % | Meaning | Action / Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | int | Identifier | 0% | Record ID | Drop during preprocessing |
| `vm_name` | string | Secondary Key | 0% | Virtual Machine Name | Use as validation key alongside IP |
| `vm_ip` | string | Primary Key | 0% | Virtual Machine IP Address | Primary join key across all datasets |
| `status` | categorical | Feature / Target Indicator | 0% | Reachability status (`Reachable` / `Unreachable`) | Encode as binary `is_unreachable = 1` |
| `timestamp` | datetime | Time Key | 0% | Observation Timestamp | Align to 4-hour UTC grid |

## HPE iLO Health (`hpe_ilo_health`)
| Column | Type | ML Role | Null % | Meaning | Action / Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ip_address` | string | Primary Key | 0% | Server IP Address | Primary join key with `ping_status.vm_ip` |
| `fans`, `cpu`, `memory`, `storage`, `temperature`, `power` | categorical | Core Features | 0% | Component health flags (`OK`, `Warning`, `Critical`) | Ordinal encode (`OK=0`, `Warning=1`, `Critical=2`) |
| `recorded_at` | datetime | Time Key | 0% | Observation Timestamp | Rename to `timestamp`, align to 4-hour grid |
| `server_name` | string | Secondary Key | 0% | Server Name | Secondary validation key |
| `current_problems` | string | Feature | 0% | Diagnostic error strings | Extract boolean warning indicators |

## Dell iDRAC Health Extended (`dell_idrac_health_ext`)
| Column | Type | ML Role | Null % | Meaning | Action / Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ip_address` | string | Primary Key | 0% | Server IP Address | Primary join key |
| `status`, `overall_status` | categorical | Target / Risk Flag | 0% | Overall server status (`OK`, `Warning`, `Critical`) | Use for lead-time failure label backpropagation |
| `fans`, `cpu`, `memory`, `storage`, `temperature`, `power` | categorical | Core Features | 0% | Component health flags | Ordinal encode (`0/1/2`) |
| `issues_detected` | string | Text Diagnostic | 0% | JSON/text diagnostic descriptions | Extract specific failure/warning regex flags |
| `timestamp` | datetime | Time Key | 0% | Clean observation timestamp (0% errors) | Align to 4-hour grid |
