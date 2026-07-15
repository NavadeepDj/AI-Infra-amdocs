#!/usr/bin/env python3
"""
agent.py — Explainable AIOps Intelligence Agent
=================================================
Master orchestrator implementing the Planner → Executor → Synthesizer
architecture within a single autonomous agent class.

The LLM reasons. Python tools compute. Every claim traces to tool evidence.

Usage:
    # Interactive CLI
    python -m aiops_agent.agent

    # Programmatic single-query
    from aiops_agent.agent import ExplainableAIOpsAgent
    agent = ExplainableAIOpsAgent()
    result = agent.ask("Why is v5G-AMF-01 unhealthy?")
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

from . import config
from .model_manager import ModelManager
from .evidence_store import EvidenceStore
from .tools import TOOL_REGISTRY, set_model_manager

# Optional Google GenAI SDK
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

logger = logging.getLogger("aiops_agent.agent")

# ─── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Explainable AIOps Intelligence Agent — an autonomous, evidence-driven SRE diagnostic assistant for infrastructure health monitoring.

## Your Role
You diagnose infrastructure health by reasoning over DETERMINISTIC tool outputs. You NEVER compute statistics, predict probabilities, or inspect raw data directly. Every numerical fact must come from a tool.

## Cognitive Loop
At each step:
1. PLAN: What information is needed to answer the user's question?
2. EXECUTE: Which deterministic tool provides that exact information?
3. SYNTHESIZE: What does the combined evidence tell us?

## Tool Selectivity & Efficiency (Planner Rules)
Be highly selective and efficient with tool calls to minimize latency and avoid redundant computation:
1. **List / Overview queries**: If the user asks for a fleet overview ("List all servers", "Summary of fleet health", "How many servers are in critical?"), call `get_fleet_summary` ONLY. Do NOT call prediction or SHAP tools on individual servers unless specifically asked.
2. **Telemetry / Ping queries**: If the user asks about basic reachability or hardware status ("Is v5G-AMF-01 reachable?", "Check telemetry for esx-01"), call `get_server_telemetry` ONLY. Do NOT call `predict_failure_12h/24h` or `explain_prediction` unless specifically asked or if the server is in a degraded/unreachable state and further diagnosis is requested.
3. **Deep Diagnostic / Failure queries**: If the user asks "Why is [server] unhealthy?" or "Predict failure for [server]", check `get_server_telemetry` and `predict_failure_12h`. ONLY call `explain_prediction` if the failure prediction is WARNING or CRITICAL (or exceeds optimal F1 threshold). Do NOT call both 12h and 24h prediction unless both horizons are relevant.
4. **Data Freshness Disclosure**: Whenever `data_staleness_warning` or `data_staleness` is returned by any tool, you MUST include the exact natural phrasing: *"The latest telemetry is approximately [X] hours old, so predictions should be interpreted cautiously."* in your final answer.
5. **Missing Sensor Disclosure**: When discussing SHAP top drivers where `value = -1`, explicitly state that `-1` indicates "Sensor Missing/Unmonitored" (not necessarily a physical hardware failure) as explained in `operational_sensor_notes`.
6. **Server Health / Diagnosis Queries**: If the user asks general health questions ("Is X healthy?", "Check health of X", "Is X good or failing?"), you must automatically run a complete diagnostic sweep: call `get_server_telemetry`, call `detect_anomaly`, and call `predict_failure_12h`. If the failure prediction risk is elevated (WARNING/CRITICAL) or exceeds the optimal F1 threshold, automatically call `explain_prediction` before concluding. Present the integrated assessment in a single response.

## Epistemic Labels (MANDATORY)
Every statement must be categorized:
- [EVIDENCE]: Verified fact directly from a tool (e.g., "Failure probability = 82.1%")
- [CONCLUSION]: Logical deduction from evidence (e.g., "Server is in CRITICAL risk tier")
- [RECOMMENDATION]: Suggested action (e.g., "Inspect NIC connectivity")
- [ASSUMPTION]: Unverified hypothesis — must include "Confidence: Low"

## Available Tools
{tools_description}

## SHAP Auto-Trigger Rule
When ANY failure prediction probability exceeds the optimal F1 threshold or is in WARNING/CRITICAL tier, you MUST call `explain_prediction` BEFORE forming your answer. Never skip SHAP when risk is elevated.

## Anomaly Explain Auto-Trigger Rule
When a user asks "Why is this server anomalous?", "Prove this anomaly", "Explain the anomaly", "Why is this NOT anomalous?", "Why is this server normal?", or ANY question about anomaly justification or evidence, you MUST call `explain_anomaly` BEFORE forming your answer. The `explain_anomaly` tool provides complete traceability: score, threshold, percentile, rank, key observed signals, and evidence sources. It works for BOTH anomalous AND normal servers.

## Hallucination Prevention (STRICT POLICY)
- NEVER invent sensor values, probabilities, predictions, or root causes
- NEVER say "the temperature is high" unless a tool returned that exact evidence
- If no tool provides evidence for a claim, say: "I don't have enough evidence to answer that."
- Every number in your answer must trace to a specific tool output
- **Vendor Neutrality:** Never reference specific hardware brands (Dell, HPE) in conclusions unless quoting diagnostic logs exactly. Use: *"The available hardware management telemetry reports..."*

## Confidence Score & Grounding Rules
Every final answer must explicitly output:
- **Confidence Rating**: HIGH, MEDIUM, or LOW
- **Confidence Reason**: The justification (e.g., *"LOW because telemetry is 278 hours old"*, or *"HIGH because telemetry is fresh (0.2h old) and all sensors are reporting normally"*).

## Conversation Context
- If the user says "it", "this server", or "that one", refer to the most recently discussed server
- Maintain context across conversation turns within a session

## Response Formats

### Format 1: Call a tool
```json
{{
  "action": "call_tool",
  "thought": "PLAN: What I need → TOOL: Why this tool → EXPECTED: What I'll learn",
  "tool": "tool_name",
  "args": {{"param": "value"}}
}}
```

### Format 2: Record conclusion
```json
{{
  "action": "conclude",
  "thought": "Evidence gathered → Reasoning → Conclusion",
  "hypothesis": "Specific claim being evaluated",
  "verdict": "ACCEPTED or REJECTED",
  "confidence_score": "HIGH / MEDIUM / LOW",
  "confidence_reason": "Why this rating",
  "reasoning": "Detailed justification citing [EVIDENCE] from tools"
}}
```

### Format 3: Final answer
```json
{{
  "action": "finish",
  "answer": "Complete, evidence-grounded SRE answer using the following exact layout:\n\n**Overall Assessment**\nServer: [hostname]\nIP: [IP Address]\n\n**Current Health**\n• Reachability: [status]\n• Subsystems: [CPU/Memory/Storage/Fans/Temp/Power statuses]\n• Degraded/Critical component counts: [counts]\n\n**Assessment**\n[Detailed grounding with [EVIDENCE] and [CONCLUSION] labels]\n\n**Confidence**\n• Rating: [HIGH/MEDIUM/LOW]\n• Reason: [Why this rating]\n\n**Recommendation**\n[Actionable SRE recommendations using [RECOMMENDATION] labels]"
}}
```

Always respond with valid JSON. Do NOT include text outside the JSON block.
"""


def build_tools_description() -> str:
    """Build human-readable description of all available tools."""
    lines = []
    for name, info in TOOL_REGISTRY.items():
        params = info["parameters"]
        param_str = ", ".join(f'"{k}": "{v}"' for k, v in params.items()) if params else "none"
        lines.append(f"- **{name}**: {info['description']}")
        lines.append(f"  Parameters: {{{param_str}}}")
    return "\n".join(lines)


def extract_json_from_response(text: str) -> dict | None:
    """Extract a JSON object from the LLM response, handling markdown code blocks."""
    text = text.strip()

    # Try ```json ... ``` blocks
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { ... } block
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : i + 1])
                except json.JSONDecodeError:
                    break
    return None


# ─── The Agent ──────────────────────────────────────────────────────────────

class ExplainableAIOpsAgent:
    """
    Autonomous Explainable AIOps Intelligence Agent.
    Orchestrates ML models and deterministic tools to diagnose infrastructure health.
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("  INITIALIZING EXPLAINABLE AIOPS INTELLIGENCE AGENT")
        logger.info("=" * 60)

        # ── Load Models & Data (once) ────────────────────────────────────
        self.manager = ModelManager()
        set_model_manager(self.manager)

        # ── Evidence & Memory ────────────────────────────────────────────
        self.evidence = EvidenceStore()
        self.conversation_history = []
        self.last_server_name = None  # Coreference tracking
        self.start_time = datetime.now()

        # ── LLM Provider Setup (OpenCode → Gemini fallback) ─────────────
        opencode_key = os.environ.get("OPENCODE_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        self.fallback_provider = None
        self.fallback_key = None
        self.gemini_models = [
            "gemini-3.5-flash",
            "gemini-2.0-flash",
            "gemini-3.1-flash-lite"
        ]

        if opencode_key:
            self.provider = "opencode"
            self.api_key = opencode_key
            self.prioritized_models = [
                "deepseek-v4-flash-free",
                "mimo-v2.5-free",
                "north-mini-code-free",
                "nemotron-3-ultra-free",
                "big-pickle"
            ]
            self.current_model = self.prioritized_models[0]
            logger.info(f"[LLM BRAIN] OpenCode Zen API. Model: {self.current_model}")
            if gemini_key:
                self.fallback_provider = "gemini"
                self.fallback_key = gemini_key
                logger.info("[LLM BRAIN] Gemini fallback configured")
        elif gemini_key:
            self.provider = "gemini"
            self.api_key = gemini_key
            self.prioritized_models = self.gemini_models
            self.current_model = self.prioritized_models[0]
            self.has_sdk = False
            if HAS_GEMINI_SDK:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    self.has_sdk = True
                except Exception:
                    pass
            logger.info(f"[LLM BRAIN] Gemini API. Model: {self.current_model}")
        else:
            raise ValueError("No API key found. Set OPENCODE_API_KEY or GEMINI_API_KEY in .env")

        # ── Build System Prompt ──────────────────────────────────────────
        tools_desc = build_tools_description()
        self.system_prompt = SYSTEM_PROMPT.replace("{tools_description}", tools_desc)

        logger.info("Agent ready. Type your infrastructure question.")

    # ─── LLM Communication ──────────────────────────────────────────────

    def _call_llm(self, user_message: str) -> str:
        """Send message to LLM with provider cascade and retry logic."""
        self.conversation_history.append({"role": "user", "content": user_message})
        time.sleep(config.API_DELAY_SECONDS)

        for attempt in range(1, config.MAX_LLM_RETRIES + 1):
            try:
                if self.provider == "opencode":
                    payload = {
                        "model": self.current_model,
                        "messages": [{"role": "system", "content": self.system_prompt}]
                                    + self.conversation_history,
                        "temperature": 0.2
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    res = requests.post(
                        "https://opencode.ai/zen/v1/chat/completions",
                        json=payload, headers=headers, timeout=35
                    )
                    if res.status_code == 200:
                        reply = res.json()["choices"][0]["message"]["content"].strip()
                        self.conversation_history.append({"role": "assistant", "content": reply})
                        return reply
                    else:
                        raise Exception(f"status {res.status_code}: {res.text[:200]}")

                elif self.provider == "gemini":
                    contents = []
                    for msg in self.conversation_history:
                        role = "user" if msg["role"] == "user" else "model"
                        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

                    if getattr(self, 'has_sdk', False):
                        response = self.client.models.generate_content(
                            model=self.current_model,
                            contents=contents,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=self.system_prompt,
                                temperature=0.2,
                                max_output_tokens=4096,
                            ),
                        )
                        reply = response.text.strip()
                    else:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.current_model}:generateContent?key={self.api_key}"
                        payload = {
                            "contents": contents,
                            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
                            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}
                        }
                        res = requests.post(url, json=payload,
                                            headers={"Content-Type": "application/json"}, timeout=35)
                        if res.status_code == 200:
                            reply = (res.json().get("candidates", [{}])[0]
                                     .get("content", {}).get("parts", [{}])[0]
                                     .get("text", "").strip())
                        else:
                            raise Exception(f"status {res.status_code}: {res.text[:200]}")

                    self.conversation_history.append({"role": "assistant", "content": reply})
                    return reply

            except Exception as e:
                error_str = str(e).lower()

                # Hot-swap to Gemini fallback
                if self.provider == "opencode" and self.fallback_provider == "gemini":
                    logger.warning(f"[PROVIDER SWAP] OpenCode error: {error_str[:80]}. Switching to Gemini.")
                    self.provider = "gemini"
                    self.api_key = self.fallback_key
                    self.prioritized_models = self.gemini_models
                    self.current_model = self.prioritized_models[0]
                    self.fallback_provider = None
                    self.has_sdk = False
                    if HAS_GEMINI_SDK:
                        try:
                            self.client = genai.Client(api_key=self.api_key)
                            self.has_sdk = True
                        except Exception:
                            pass
                    if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                        self.conversation_history.pop()
                    return self._call_llm(user_message)

                # Model cascade within provider
                try:
                    idx = self.prioritized_models.index(self.current_model)
                    if idx + 1 < len(self.prioritized_models):
                        self.current_model = self.prioritized_models[idx + 1]
                        logger.warning(f"[MODEL SWAP] Switching to: {self.current_model}")
                        time.sleep(2)
                        continue
                except ValueError:
                    pass

                if attempt < config.MAX_LLM_RETRIES:
                    wait = 15 * attempt
                    logger.warning(f"[RETRY {attempt}/{config.MAX_LLM_RETRIES}] Waiting {wait}s...")
                    time.sleep(wait)

        # All retries exhausted
        fallback = ('{"action": "finish", "answer": "I apologize, but I am temporarily unable '
                     'to process your question due to API connectivity issues. Please try again '
                     'in a moment."}')
        self.conversation_history.append({"role": "assistant", "content": fallback})
        return fallback

    # ─── Tool Execution ─────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute a tool from the registry and return structured output."""
        if tool_name not in TOOL_REGISTRY:
            return {"error": f"Unknown tool: {tool_name}. Available: {list(TOOL_REGISTRY.keys())}"}

        t0 = time.time()
        try:
            func = TOOL_REGISTRY[tool_name]["function"]
            result = func(**args)
            latency = (time.time() - t0) * 1000

            # Track last server for coreference
            if "server_name" in args and args["server_name"]:
                self.last_server_name = args["server_name"]

            logger.info(f"  Tool '{tool_name}' executed in {latency:.1f}ms")
            return result, latency, "SUCCESS"
        except Exception as e:
            latency = (time.time() - t0) * 1000
            logger.error(f"  Tool '{tool_name}' failed: {e}")
            return {"error": str(e)}, latency, "ERROR"

    # ─── Single Query API ───────────────────────────────────────────────

    def ask(self, query: str) -> dict:
        """
        Process a single natural language query.
        Returns: {"answer": str, "tools_called": list, "latency_ms": float, "evidence": list}
        """
        self.evidence.reset_for_new_query()
        self.evidence.start_query(query)
        query_start = time.time()

        # Inject coreference context
        context_query = query
        if self.last_server_name:
            context_query = (f"[Context: The most recently discussed server is "
                             f"'{self.last_server_name}'. If the user refers to 'it', "
                             f"'this server', or 'that one', use '{self.last_server_name}'.]\n\n"
                             f"User Question: {query}")

        reply = self._call_llm(context_query)

        for step in range(1, config.MAX_AGENT_STEPS + 1):
            parsed = extract_json_from_response(reply)

            if parsed is None:
                reply = self._call_llm(
                    "Your previous response was not valid JSON. Please respond with exactly "
                    "one JSON block using call_tool, conclude, or finish."
                )
                continue

            action = parsed.get("action", "unknown")

            # ── call_tool ────────────────────────────────────────────────
            if action == "call_tool":
                thought = parsed.get("thought", "")
                tool_name = parsed.get("tool", "")
                tool_args = parsed.get("args", {})

                logger.info(f"  Step {step}: call_tool -> {tool_name}({json.dumps(tool_args)})")

                result, latency, status = self._execute_tool(tool_name, tool_args)
                result_copy = dict(result) if isinstance(result, dict) else result
                audit_meta = result_copy.pop("_audit_metadata", None) if isinstance(result_copy, dict) else None
                if isinstance(result_copy, dict) and not config.SHOW_ADVANCED_METADATA:
                    audit_meta = audit_meta or result_copy.pop("metadata", None)

                self.evidence.record(
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    evidence=result_copy if isinstance(result_copy, dict) else {},
                    latency_ms=latency,
                    tool_status=status,
                    data_staleness_hours=self.manager.data_age_hours(),
                    audit_metadata=audit_meta
                )

                evidence_str = json.dumps(result_copy, indent=2, default=str)
                # Truncate for LLM context window
                prompt_evidence = evidence_str
                if len(evidence_str) > 2000:
                    prompt_evidence = evidence_str[:2000] + "\n... [truncated, full data in Evidence Store]"

                reply = self._call_llm(
                    f"Tool `{tool_name}` returned:\n```json\n{prompt_evidence}\n```\n"
                    "Based on this evidence, call another tool, record a conclusion, or "
                    "give your final answer using 'finish'."
                )

            # ── conclude ─────────────────────────────────────────────────
            elif action == "conclude":
                thought = parsed.get("thought", "")
                hypothesis = parsed.get("hypothesis", "")
                verdict = parsed.get("verdict", "")
                reasoning = parsed.get("reasoning", "")

                logger.info(f"  Step {step}: conclude -> {hypothesis[:60]}... [{verdict}]")

                self.evidence.record(
                    thought=thought,
                    conclusion=f"Hypothesis: {hypothesis} → {verdict}. {reasoning}"
                )

                reply = self._call_llm(
                    "Good. Call another tool, record another conclusion, or give your final "
                    "answer using 'finish'."
                )

            # ── finish ───────────────────────────────────────────────────
            elif action == "finish":
                answer = (parsed.get("answer") or parsed.get("final_answer") or 
                          parsed.get("response") or parsed.get("summary") or 
                          parsed.get("conclusion") or parsed.get("message") or "")
                if not answer and len(parsed) > 1:
                    for k, v in parsed.items():
                        if k != "action" and isinstance(v, str) and len(v) > 10:
                            answer = v
                            break
                if not answer:
                    answer = reply

                total_latency = (time.time() - query_start) * 1000

                self.evidence.end_query(answer)

                logger.info(f"  Answer generated in {total_latency:.0f}ms "
                            f"({len(self.evidence.get_tools_called())} tool calls)")

                return {
                    "answer": answer,
                    "tools_called": self.evidence.get_tools_called(),
                    "latency_ms": round(total_latency, 1),
                    "steps": len(self.evidence.steps),
                    "session_id": self.evidence.session_id
                }

            else:
                reply = self._call_llm(
                    f"Unknown action '{action}'. Use 'call_tool', 'conclude', or 'finish'."
                )

        # Max steps reached — force finish
        logger.warning("Max agent steps reached. Forcing final answer.")
        reply = self._call_llm(
            f"You have reached the maximum of {config.MAX_AGENT_STEPS} steps. "
            "Give your final answer NOW using 'finish'. Summarize all evidence gathered."
        )
        parsed = extract_json_from_response(reply)
        answer = ""
        if parsed and parsed.get("action") == "finish":
            answer = (parsed.get("answer") or parsed.get("final_answer") or 
                      parsed.get("response") or parsed.get("summary") or 
                      parsed.get("conclusion") or parsed.get("message") or "")
            if not answer and len(parsed) > 1:
                for k, v in parsed.items():
                    if k != "action" and isinstance(v, str) and len(v) > 10:
                        answer = v
                        break
        if not answer:
            answer = reply if reply else "Investigation reached maximum steps. Please narrow your question."

        total_latency = (time.time() - query_start) * 1000
        self.evidence.end_query(answer)
        return {
            "answer": answer,
            "tools_called": self.evidence.get_tools_called(),
            "latency_ms": round(total_latency, 1),
            "steps": len(self.evidence.steps),
            "session_id": self.evidence.session_id
        }

    # ─── Interactive CLI ────────────────────────────────────────────────

    def chat(self):
        """Interactive REPL for SRE engineers."""
        print("\n" + "=" * 60)
        print("  EXPLAINABLE AIOPS INTELLIGENCE AGENT")
        print(f"  Model: {self.current_model}")
        print(f"  Servers: {len(self.manager.list_all_servers())}")
        print(f"  Data Freshness: {self.manager.data_age_hours():.0f}h old")
        print("=" * 60)
        print("\nAsk me anything about your infrastructure.")
        print("Type 'exit' to quit, 'metrics' for session stats.\n")

        while True:
            try:
                query = input("SRE> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if query.lower() == "metrics":
                metrics = self.evidence.get_metrics_summary()
                print(json.dumps(metrics, indent=2))
                continue
            if query.lower() == "evidence":
                print(self.evidence.to_markdown())
                continue

            result = self.ask(query)
            print(f"\n{result['answer']}")
            print(f"\n  [Tools: {', '.join(result['tools_called'])} | "
                  f"Steps: {result['steps']} | "
                  f"Latency: {result['latency_ms']:.0f}ms]\n")

    # ─── Export ──────────────────────────────────────────────────────────

    def save_session(self, path: Path | None = None):
        """Save the complete session evidence to JSON."""
        if path is None:
            path = config.EVIDENCE_EXPORT_DIR / "session_evidence.json"
        self.evidence.save_json(path)
        logger.info(f"Session evidence saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    agent = ExplainableAIOpsAgent()
    agent.chat()


if __name__ == "__main__":
    main()
