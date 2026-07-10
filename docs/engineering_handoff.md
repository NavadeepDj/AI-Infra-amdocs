# Official Engineering Handoff & ML Roadmap

Every item below is strictly governed by our epistemic labeling (`[EVIDENCE]`, `[CONCLUSION]`, `[RECOMMENDATION]`).

## Project Phase Status
| Phase | Status | Owner | Next Milestone |
| :--- | :--- | :--- | :--- |
| **1. Data Understanding & Profiling** | **COMPLETE (`[EVIDENCE]`)** | ExplainableDataAgent | Canonical 8-artifact handoff package verified across `docs/*` |
| **2. Data Preprocessing & Merging** | **READY (`[CONCLUSION]`)** | ML Engineering Team | Execute `preprocess_master_dataset.py` using canonical observation key (`machine + ip + slot`) |
| **3. Feature Engineering** | **NOT STARTED** | ML Engineering Team | Build evidence-derived warning counts and suggested rolling lag features (`[RECOMMENDATION]`) |
| **4. Anomaly Detection Engine** | **NOT STARTED** | ML Engineering Team | Train Isolation Forest on 4-hour tabular feature matrix (`[RECOMMENDATION]`) |
| **5. Failure Prediction & Forecasting**| **NOT STARTED** | ML Engineering Team | Train XGBoost Time Series models for 7-day failure risk and CPU usage forecasting (`[RECOMMENDATION]`) |
| **6. AI Operations Assistant** | **NOT STARTED** | ML Engineering Team | Integrate RAG + SQL agent with exact deterministic tool grounding (`[RECOMMENDATION]`) |

--- 

## Immediate Next Steps (Preprocessing Workflow)
1. `[RECOMMENDATION]` Write and execute `preprocess_master_dataset.py` merging `ping_status_20260702`, `hpe_ilo_health_20260702`, and `dell_idrac_health_ext_20260702` on **`machine_name + ip_address + monitoring_slot`**.
2. `[RECOMMENDATION]` Validate that the resulting master dataset has exactly `45,756` rows (`246 machines * 186 slots`) with zero duplicate composite keys.
3. `[RECOMMENDATION]` Verify that `is_imputed = 1` boolean indicators accurately track all `ffill` hardware slots.
