# Phase 6 AIOps Agent Walkthrough & Explainability Guide

This walkthrough explains how to interactively test the Explainable AIOps Agent, how to introspect its internal reasoning steps, and how the state and processing pipeline work.

---

## 1. How to Run the Interactive Agent CLI

Navigate to your workspace directory `c:\Users\navad\ML_data` and execute:

```powershell
python -m aiops_agent.interactive_agent
```

---

## 2. Interactive Explainability Shortcuts

While in the interactive CLI, you can type special commands at the `SRE> ` prompt to query the agent's internal state:

| Command | Action | Description |
| :--- | :--- | :--- |
| `evidence` | Display evidence chain | Shows the exact sequence of thoughts, tools executed, inputs/outputs, and latencies from the last query. |
| `metrics` | Display session metrics | Shows average latency, total steps taken, error rates, and tool counts for the current session. |
| `history` | Display conversation history | Shows the raw chat transcripts between the user, coordinator, and tools. |

---

## 3. Grounding & Verification Example

### Query: *"Is 192.168.148.171 healthy?"*
1. **IP Resolution**: The agent recognizes the IP and calls `find_server_by_ip("192.168.148.171")`, returning `v5G-NRF-Edge-08`.
2. **Diagnostic Sweep**: The agent automatically checks:
   * `get_server_telemetry("v5G-NRF-Edge-08")`
   * `detect_anomaly("v5G-NRF-Edge-08")`
   * `predict_failure_12h("v5G-NRF-Edge-08")`
3. **Structured Synthesis**:
   * It presents the server hostname and IP.
   * It formats reachability and component counts as clear bullet points.
   * It explicitly reports the **Confidence Score** and **Confidence Reason**. When `SHOW_DATA_FRESHNESS = True`, staleness warnings degrade confidence appropriately; when `False` (Demo Mode), output remains cleanly focused on AI diagnostics.
   * It lists actionable recommendations for the SRE while preserving internal model provenance (`model_version`, `latency`, `decision_threshold`) securely inside `EvidenceStore` (`audit_metadata`).

### Verification Results

All core tools have been verified using:
```powershell
python -m aiops_agent.test_tools_quick
```
* **Status**: `[PASSED]`
* **Models Loaded**: Isolation Forest, XGBoost 12h, XGBoost 24h.
* **Average Inference Latency**: `< 20ms`.
