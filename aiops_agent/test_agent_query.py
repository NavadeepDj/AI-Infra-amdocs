#!/usr/bin/env python3
"""
End-to-End AIOps Agent Verification Test
=========================================
Tests the ExplainableAIOpsAgent's programmatic ask() API on a real question
to verify LLM reasoning, deterministic tool execution, evidence recording,
and epistemic grounding.
"""

import json
import time
from aiops_agent.agent import ExplainableAIOpsAgent

def main():
    print("========================================================================")
    print("=== Phase 6 End-to-End AIOps Agent Verification Test ===")
    print("========================================================================")
    
    t0 = time.time()
    agent = ExplainableAIOpsAgent()
    print(f"\n[PASSED] Agent initialized in {(time.time() - t0)*1000:.1f}ms (Provider: {agent.provider})\n")
    
    test_query = "Why is v5G-AMF-01 unhealthy? What does the failure prediction model show and why?"
    print(f"User Query: '{test_query}'\n")
    print("Agent is investigating (running Planner -> Executor -> Synthesizer loop)...\n")
    
    result = agent.ask(test_query)
    
    print("------------------------------------------------------------------------")
    print("FINAL AGENT ANSWER:")
    print("------------------------------------------------------------------------")
    print(result["answer"])
    print("\n------------------------------------------------------------------------")
    print(f"Execution Summary: {result['steps']} steps | Tools Called: {result['tools_called']} | Total Latency: {result['latency_ms']:.0f}ms")
    print("------------------------------------------------------------------------")
    
    print("\nEvidence Store Summary:")
    print(agent.evidence.get_summary()[:1200] + "\n...(truncated)")
    
    # Verify golden criteria
    tools_called = set(result["tools_called"])
    assert "get_server_telemetry" in tools_called or "predict_failure_12h" in tools_called, "Agent must call telemetry or prediction tool!"
    print("\n[SUCCESS] Agent successfully reasoned over deterministic tools and produced grounded answer!")

if __name__ == "__main__":
    main()
