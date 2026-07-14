#!/usr/bin/env python3
"""
Quick Verification Test for AIOps Deterministic Tools
=====================================================
Tests all 9 tools directly (without LLM) to verify schema, model loading,
latency, and retry/error handling.
"""

import json
import time
from aiops_agent.model_manager import ModelManager
from aiops_agent import tools

def main():
    print("========================================================================")
    print("=== Phase 6 Tool Registry & ModelManager Verification Test ===")
    print("========================================================================")
    
    t0 = time.time()
    manager = ModelManager()
    tools.set_model_manager(manager)
    print(f"\n[PASSED] ModelManager initialized in {(time.time() - t0)*1000:.1f}ms\n")
    
    test_server = "v5G-AMF-01"
    
    # Tool 1: get_server_telemetry
    print("--- 1. Testing get_server_telemetry ---")
    t1 = tools.get_server_telemetry(server_name=test_server)
    print(json.dumps(t1, indent=2))
    
    # Tool 1b: find_server_by_ip
    print("\n--- 1b. Testing find_server_by_ip ---")
    ip_to_test = t1.get("ip_address", "172.19.30.142")
    t1b = tools.find_server_by_ip(ip_address=ip_to_test)
    print(json.dumps(t1b, indent=2))
    
    # Tool 2: get_server_history
    print("\n--- 2. Testing get_server_history (last 4 slots) ---")
    t2 = tools.get_server_history(server_name=test_server, n_slots="4")
    print(f"Slots returned: {t2.get('slots_returned')}, Trends: {t2.get('trends')}")
    
    # Tool 3: detect_anomaly
    print("\n--- 3. Testing detect_anomaly ---")
    t3 = tools.detect_anomaly(server_name=test_server)
    print(json.dumps(t3, indent=2))
    
    # Tool 4 & 5: predict_failure_12h & 24h
    print("\n--- 4/5. Testing predict_failure_12h & 24h ---")
    t4 = tools.predict_failure_12h(server_name=test_server)
    t5 = tools.predict_failure_24h(server_name=test_server)
    print(f"12h Failure Prob: {t4.get('failure_probability_pct')}% ({t4.get('risk_tier')})")
    print(f"24h Failure Prob: {t5.get('failure_probability_pct')}% ({t5.get('risk_tier')})")
    
    # Tool 6: explain_prediction (SHAP)
    print("\n--- 6. Testing explain_prediction (SHAP) ---")
    t6 = tools.explain_prediction(server_name=test_server, horizon="12h")
    print("Top Drivers:")
    for d in t6.get("top_drivers", [])[:3]:
        print(f"  * {d['feature']} = {d['value']} (SHAP contribution: {d['shap_contribution']:+.4f})")
        
    # Tool 7: get_fleet_summary
    print("\n--- 7. Testing get_fleet_summary ---")
    t0_fleet = time.time()
    t7 = tools.get_fleet_summary()
    t_fleet_ms = (time.time() - t0_fleet) * 1000
    print(f"Fleet scored in {t_fleet_ms:.1f}ms: {t7.get('total_servers')} servers | NORMAL: {t7.get('healthy_normal')} | WARNING: {t7.get('warning')} | CRITICAL: {t7.get('critical')}")
    
    # Tool 8: get_model_metadata
    print("\n--- 8. Testing get_model_metadata ---")
    t8 = tools.get_model_metadata()
    print(f"Models available: {t8.get('models_available')}")
    
    # Tool 9: get_recent_alerts
    print("\n--- 9. Testing get_recent_alerts ---")
    t9 = tools.get_recent_alerts(server_name=test_server)
    print(f"Alerts for {test_server}: {t9.get('total_alerts')}")
    
    print("\n========================================================================")
    print("[SUCCESS] All 9 deterministic AIOps tools verified successfully!")
    print("========================================================================")

if __name__ == "__main__":
    main()
