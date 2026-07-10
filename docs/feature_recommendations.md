# Feature Engineering Recommendations

## Tier 1: Raw Status & Ordinal Encodings
- Ordinal encode component health: `fans_score`, `cpu_score`, `memory_score`, `storage_score`, `temperature_score`, `power_score` (`OK=0`, `Warning=1`, `Critical=2`).

## Tier 2: Rolling & Temporal Derived Features
1. `rolling_24h_ping_drops`: Sum of `is_unreachable` over rolling 6 time slots (24 hours).
2. `component_warning_sum`: Total sum of active warning flags across all 6 hardware components.
3. `temp_delta_from_baseline`: Difference between current temperature score and server's 7-day historical mode.
4. `hours_since_last_warning`: Elapsed hours (`slots * 4`) since the server last reported a non-OK state.
5. `ping_state_flip_rate`: Number of reachability transitions (`OK <-> Unreachable`) over past 48h (`flapping` indicator).

## Tier 3: Diagnostic Boolean Regex Flags
- `has_power_redundancy_loss`: `1` if `issues_detected` contains `'Power supply redundancy is lost'`, else `0`.
- `has_thermal_throttling`: `1` if `issues_detected` contains `'CPU 1 throttling due to thermal threshold'`, else `0`.
- `has_disk_array_warning`: `1` if `issues_detected` contains `'Disk array controller'`, else `0`.
