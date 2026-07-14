#!/usr/bin/env python3
"""
interactive_agent.py — Interactive CLI for the Explainable AIOps Intelligence Agent
=================================================================================
Allows SREs to ask natural language questions about infrastructure health,
see deterministic tool traces, and review audit/traceability metadata.
"""

import sys
import time
from aiops_agent.agent import ExplainableAIOpsAgent

def main():
    print("\n" + "=" * 80)
    print("      🌟 WELCOME TO THE EXPLAINABLE AIOPS INTELLIGENCE PLATFORM CLI 🌟      ")
    print("=" * 80)
    print("Initializing AIOps models and telemetry registry...")
    
    t0 = time.time()
    try:
        agent = ExplainableAIOpsAgent()
    except Exception as e:
        print(f"\n❌ Initialization Failed: {e}")
        return
        
    init_ms = (time.time() - t0) * 1000
    print(f"✔ Agent initialized successfully in {init_ms:.1f}ms!")
    print(f"✔ Active LLM Provider: {agent.provider.upper()}")
    print("-" * 80)
    print("Type a natural language question about infrastructure health.")
    print("Examples:")
    print("  - 'Which servers are in a critical state?'")
    print("  - 'Why is v5G-UPF-01 unhealthy?'")
    print("  - 'Is server v5G-AMF-02 reachable?'")
    print("  - 'Provide a summary of our fleet health.'")
    print("Type 'exit' or 'quit' to end session.")
    print("CLI Session Shortcuts:")
    print("  * 'evidence' : Display step-by-step evidence trace for the last query.")
    print("  * 'metrics'  : Display session performance metrics & latencies.")
    print("  * 'history'  : Display the active conversation history.")
    print("-" * 80 + "\n")

    while True:
        try:
            query = input("SRE> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if query.lower() in ("evidence", "show-evidence"):
            print("\n" + "=" * 80)
            print("📜 LAST QUERY EVIDENCE CHAIN")
            print("=" * 80)
            print(agent.evidence.get_summary())
            print("=" * 80 + "\n")
            continue

        if query.lower() in ("metrics", "show-metrics"):
            print("\n" + "=" * 80)
            print("📊 SESSION PERFORMANCE METRICS")
            print("=" * 80)
            metrics = agent.evidence.get_metrics_summary()
            for key, val in metrics.items():
                print(f"  * {key.replace('_', ' ').title()}: {val}")
            print("=" * 80 + "\n")
            continue

        if query.lower() in ("history", "show-history"):
            print("\n" + "=" * 80)
            print("💬 CONVERSATION HISTORY")
            print("=" * 80)
            for idx, msg in enumerate(agent.conversation_history):
                print(f"[{idx+1}] {msg['role'].upper()}: {msg['content'][:300]}")
                if len(msg['content']) > 300:
                    print("    ...[truncated]")
            print("=" * 80 + "\n")
            continue

        print("\n🔍 Investigating (reasoning over deterministic tools)...")
        t_start = time.time()
        
        try:
            result = agent.ask(query)
            elapsed_sec = time.time() - t_start
            
            print("\n" + "=" * 80)
            print("🤖 AGENT RESPONSE")
            print("=" * 80)
            print(result["answer"])
            print("-" * 80)
            print(f"⏱ Response Latency: {elapsed_sec:.2f}s | Steps: {result['steps']}")
            print(f"🛠 Tools Executed: {', '.join(result['tools_called']) if result['tools_called'] else 'None'}")
            print("=" * 80 + "\n")
            
            # Auto-save live monitoring audit trail to docs/live_session_evidence.json
            try:
                from pathlib import Path
                evidence_path = Path("docs/live_session_evidence.json")
                agent.evidence.save_json(evidence_path)
                print(f"📊 [MONITORING] Live audit log updated: docs/live_session_evidence.json\n")
            except Exception as save_err:
                pass
            
        except Exception as e:
            print(f"\n❌ Error executing agent query: {e}\n")

if __name__ == "__main__":
    main()
