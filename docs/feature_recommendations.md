# Feature Engineering Recommendations (Evidence vs. Suggested)

To prevent hallucination, features are strictly segregated into those proven directly by our 50-step deterministic evidence versus suggested ML engineering enhancements.

## Category A: Evidence-Derived Features (`[EVIDENCE / CONCLUSION]`)
These features directly reflect verified telemetry patterns established during our data profiling:
1. `is_unreachable`: Binary (`1` if `ping_status == 'Unreachable'`, `0` otherwise) (`[EVIDENCE]` verified ping reachability status).
2. `component_warning_count`: Sum of ordinal warning/critical flags (`0, 1, 2`) across `fans`, `cpu`, `memory`, `storage`, `temperature`, and `power` (`[EVIDENCE]` verified `0%` null rate when active).
3. `has_power_redundancy_loss`: Binary (`1` if `issues_detected` exactly matches or contains `'Power supply redundancy is lost'`) (`[EVIDENCE]` proven diagnostic string in Step 48/50).
4. `has_thermal_throttling`: Binary (`1` if `issues_detected` contains `'CPU 1 throttling due to thermal threshold'`) (`[EVIDENCE]` proven diagnostic string).
5. `has_disk_array_warning`: Binary (`1` if `issues_detected` contains `'Disk array controller'`) (`[EVIDENCE]` proven diagnostic string).

## Category B: Suggested / Hypothetical ML Features (`[RECOMMENDATION]`)
These are domain-informed feature engineering suggestions proposed by the Senior Data Scientist to improve model accuracy (`NOT proven as existing in historical evidence, but recommended to compute`):
1. `rolling_24h_ping_drops`: `[RECOMMENDATION]` Compute rolling sum of `is_unreachable` over the past `6` slots (`24 hours`) per `machine_name + ip_address`.
2. `temp_delta_from_baseline`: `[RECOMMENDATION]` Compute `(current_temperature_score - historical_7d_mode)` to detect abnormal thermal drift before critical thresholds trigger.
3. `hours_since_last_warning`: `[RECOMMENDATION]` Track consecutive slots (`slots * 4 hours`) since `component_warning_count > 0`.
4. `ping_state_flip_rate`: `[RECOMMENDATION]` Count transitions (`OK <-> Unreachable`) over rolling 12 slots (`48 hours`) to capture network flapping/instability.
