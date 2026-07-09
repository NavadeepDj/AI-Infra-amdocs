#!/usr/bin/env python3
"""
=============================================================================
Explainable Data Understanding Agent (eda_agent.py)
=============================================================================

A TRUE AI Agent that autonomously investigates datasets using an LLM-driven
reasoning loop. The LLM (Gemini) acts as a senior data scientist who:

  1. Understands the business objective
  2. Decides which analysis tool to call next
  3. Reads the structured evidence returned by deterministic Python tools
  4. Forms hypotheses and accepts/rejects them based on evidence
  5. Decides what to investigate next (dynamic, not predetermined)
  6. Produces an engineering-ready report backed entirely by tool evidence

The Python tools (tools.py) NEVER hallucinate — they return reproducible
structured facts. The LLM NEVER computes statistics — it only reasons.

Usage:
    python eda_agent.py

Configuration:
    API Key: Set GEMINI_API_KEY in .env or environment variable
    Max Steps: 25 (configurable below)
    Model: gemini-2.5-flash (configurable below)
    Report: docs/agent_investigation_report.md
=============================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from tools import TOOL_REGISTRY
from evidence_store import EvidenceStore

# ─── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_ROOT / "docs" / "agent_investigation_report.md"
EVIDENCE_JSON_PATH = PROJECT_ROOT / "docs" / "investigation_evidence.json"

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please add it to your .env file.")

PRIORITIZED_MODELS = [
    "gemini-3.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro"
]
MAX_STEPS = 25
API_DELAY_SECONDS = 4  # Delay between API calls to stay under free-tier rate limits
MAX_RETRIES = 3        # Number of retries on rate-limit (429) errors

# ─── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Explainable Data Understanding Agent — an autonomous, evidence-driven data scientist investigator.

## Your Role
You investigate unfamiliar datasets by calling deterministic analysis tools, reading their structured evidence outputs, forming hypotheses, and accepting or rejecting them based strictly on evidence. You behave like a senior data scientist directing a team of specialized tools.

## Critical Rules
1. You NEVER compute statistics yourself. You ALWAYS call a tool to get facts.
2. Every conclusion must cite specific evidence from a tool output.
3. You decide dynamically which tool to call next based on what you've learned so far.
4. If you discover something unexpected, you PIVOT your investigation — you do NOT follow a fixed checklist.
5. You form explicit hypotheses and state whether you ACCEPT or REJECT them based on evidence.

## Business Context
Infrastructure health data is collected daily from multiple monitoring systems (Ping Status, HPE iLO, Dell iDRAC). The goal is to build an AI solution for anomaly detection, failure prediction, forecasting, and explainable infrastructure health analysis.

## Available Tools
You have access to the following deterministic analysis tools. Call them by responding with a JSON block:

{tools_description}

## How to Respond

At each step, respond with EXACTLY ONE of these formats:

### Format 1: Call a tool
```json
{{
  "action": "call_tool",
  "thought": "Your reasoning about why you're calling this tool",
  "tool": "tool_name",
  "args": {{"param": "value"}}
}}
```

### Format 2: Record a conclusion (after seeing tool evidence)
```json
{{
  "action": "conclude",
  "thought": "What this evidence tells you",
  "hypothesis": "The specific claim you tested",
  "verdict": "ACCEPTED or REJECTED",
  "reasoning": "Why, citing specific evidence"
}}
```

### Format 3: End investigation and write final report
```json
{{
  "action": "finish",
  "summary": "Your complete investigation report in markdown format. Include all findings, evidence citations, and engineering recommendations. Structure it with clear sections."
}}
```

## Investigation Strategy
Start by discovering what datasets are available, then systematically understand each one. Look for:
- How many datasets exist and what they contain
- How many machines/assets are monitored
- Whether datasets can be merged (shared identifiers)
- Time coverage and monitoring frequency
- Data quality issues (missing values, duplicates)
- Class distributions and imbalances
- Engineering recommendations for downstream ML

Always respond with valid JSON. Do NOT include any text outside the JSON block.
"""


def build_tools_description() -> str:
    """Build a human-readable description of all available tools for the system prompt."""
    lines = []
    for name, info in TOOL_REGISTRY.items():
        params = info["parameters"]
        param_str = ", ".join(f'"{k}": "{v}"' for k, v in params.items()) if params else "none"
        lines.append(f"- **{name}**: {info['description']}")
        lines.append(f"  Parameters: {{{param_str}}}")
    return "\n".join(lines)


def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a registered tool and return its structured output."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}. Available: {list(TOOL_REGISTRY.keys())}"}
    try:
        func = TOOL_REGISTRY[tool_name]["function"]
        return func(**args)
    except Exception as e:
        return {"error": f"Tool '{tool_name}' failed: {str(e)}"}


def extract_json_from_response(text: str) -> dict | None:
    """Extract a JSON object from the LLM response, handling markdown code blocks."""
    text = text.strip()

    # Try to find JSON inside ```json ... ``` blocks
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
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


# ─── The Agent Loop ─────────────────────────────────────────────────────────

class ExplainableDataAgent:
    def __init__(self):
        self.client = genai.Client(api_key=API_KEY)
        self.evidence = EvidenceStore()
        self.start_time = datetime.now()
        self.conversation_history = []
        self.current_model = PRIORITIZED_MODELS[0]

        # Build system prompt with tool descriptions
        tools_desc = build_tools_description()
        self.system_prompt = SYSTEM_PROMPT.replace("{tools_description}", tools_desc)

    def _print_step_header(self, step: int):
        print(f"\n{'-' * 70}")
        print(f"  STEP {step} / {MAX_STEPS}")
        print(f"{'-' * 70}")

    def _call_llm(self, user_message: str) -> str:
        """Send a message to Gemini and get the response, with dynamic model fallback and rate-limit handling."""
        self.conversation_history.append({"role": "user", "parts": [{"text": user_message}]})

        # Rate-limit: wait between calls to avoid 429 errors
        time.sleep(API_DELAY_SECONDS)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.current_model,
                    contents=self.conversation_history,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=0.2,
                        max_output_tokens=4096,
                    ),
                )
                reply = response.text.strip()
                self.conversation_history.append({"role": "model", "parts": [{"text": reply}]})
                return reply
            except Exception as e:
                error_str = str(e).lower()
                is_quota_error = any(kw in error_str for kw in ["429", "resource_exhausted", "quota", "rate"])

                # Dynamic Model Fallback: if we hit a quota limit, try to switch to the next model in the list
                if is_quota_error:
                    try:
                        current_idx = PRIORITIZED_MODELS.index(self.current_model)
                        if current_idx + 1 < len(PRIORITIZED_MODELS):
                            next_model = PRIORITIZED_MODELS[current_idx + 1]
                            print(f"  [MODEL ROUTER] Quota exceeded for '{self.current_model}'. Switching to fallback model: '{next_model}'...")
                            self.current_model = next_model
                            # Retry immediately with the new model
                            continue
                    except ValueError:
                        pass

                is_retryable = is_quota_error or any(kw in error_str for kw in [
                    "connect", "getaddrinfo", "timeout", "network",
                ])
                if is_retryable and attempt < MAX_RETRIES:
                    wait = 20 * attempt  # 20s, 40s backoff
                    error_type = "RATE LIMIT" if is_quota_error else "NETWORK"
                    print(f"  [{error_type}] {str(e)[:80]}... Waiting {wait}s (retry {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                elif is_retryable:
                    # Last retry — wait longer
                    wait = 60
                    print(f"  [RETRY {attempt}/{MAX_RETRIES}] Final retry. Waiting {wait}s...")
                    time.sleep(wait)
                    try:
                        response = self.client.models.generate_content(
                            model=self.current_model,
                            contents=self.conversation_history,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=self.system_prompt,
                                temperature=0.2,
                                max_output_tokens=4096,
                            ),
                        )
                        reply = response.text.strip()
                        self.conversation_history.append({"role": "model", "parts": [{"text": reply}]})
                        return reply
                    except Exception:
                        pass  # Fall through to fallback
                else:
                    raise

        # Final fallback after all retries exhausted
        fallback = '{"action": "conclude", "thought": "API temporarily unavailable after retries.", "hypothesis": "N/A", "verdict": "SKIPPED", "reasoning": "Could not reach LLM. Will continue with next step."}'
        self.conversation_history.append({"role": "model", "parts": [{"text": fallback}]})
        return fallback

    def run(self):
        """Execute the autonomous investigation loop."""
        print("\n" + "=" * 70)
        print("  EXPLAINABLE DATA UNDERSTANDING AGENT")
        print(f"  Model: {self.current_model} | Max Steps: {MAX_STEPS}")
        print(f"  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Kick off the investigation
        reply = self._call_llm(
            "Begin your investigation. Start by discovering what datasets are available, "
            "then systematically investigate their structure, quality, relationships, and "
            "ML readiness. Remember: call tools to get facts, never guess."
        )

        for step in range(1, MAX_STEPS + 1):
            self._print_step_header(step)

            parsed = extract_json_from_response(reply)

            if parsed is None:
                print(f"  [WARNING] Could not parse LLM response as JSON. Asking to retry...")
                print(f"  Raw response (first 200 chars): {reply[:200]}")
                reply = self._call_llm(
                    "Your previous response was not valid JSON. Please respond with exactly one JSON block "
                    "using the format specified in your instructions (call_tool, conclude, or finish)."
                )
                continue

            action = parsed.get("action", "unknown")

            # ── Action: Call a tool ──
            if action == "call_tool":
                thought = parsed.get("thought", "")
                tool_name = parsed.get("tool", "")
                tool_args = parsed.get("args", {})

                print(f"  [THOUGHT]    {thought}")
                print(f"  [TOOL CALL]  {tool_name}({json.dumps(tool_args, default=str)})")

                evidence = execute_tool(tool_name, tool_args)
                evidence_str = json.dumps(evidence, indent=2, default=str)

                # Print evidence (truncated for readability)
                preview = evidence_str[:500]
                if len(evidence_str) > 500:
                    preview += "\n  ...(truncated)"
                print(f"  [EVIDENCE]   {preview}")

                self.evidence.record(
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    evidence=evidence,
                )

                # Feed evidence back to LLM
                reply = self._call_llm(
                    f"Tool `{tool_name}` returned this evidence:\n```json\n{evidence_str}\n```\n"
                    "Based on this evidence, what do you want to do next? You can call another tool, "
                    "record a conclusion, or finish the investigation."
                )

            # ── Action: Record a conclusion ──
            elif action == "conclude":
                thought = parsed.get("thought", "")
                hypothesis = parsed.get("hypothesis", "")
                verdict = parsed.get("verdict", "")
                reasoning = parsed.get("reasoning", "")

                print(f"  [THOUGHT]      {thought}")
                print(f"  [HYPOTHESIS]   {hypothesis}")
                print(f"  [VERDICT]      {verdict}")
                print(f"  [REASONING]    {reasoning}")

                self.evidence.record(
                    thought=thought,
                    conclusion=f"Hypothesis: {hypothesis} → {verdict}. {reasoning}",
                )

                # Ask LLM to continue
                reply = self._call_llm(
                    "Good. What do you want to investigate next? Call a tool or finish if done."
                )

            # ── Action: Finish ──
            elif action == "finish":
                summary = parsed.get("summary", "")
                print(f"\n  [INVESTIGATION COMPLETE]")
                print(f"  {summary[:200]}...")
                self._save_report(summary)
                break

            else:
                print(f"  [WARNING] Unknown action: {action}. Asking LLM to correct...")
                reply = self._call_llm(
                    f"Unknown action '{action}'. Use 'call_tool', 'conclude', or 'finish'."
                )

        else:
            # Hit max steps — ask LLM to produce final report
            print(f"\n  [MAX STEPS REACHED] Asking agent to synthesize final report...")
            reply = self._call_llm(
                f"You have reached the maximum of {MAX_STEPS} investigation steps. "
                "Please produce your final report now using the 'finish' action. "
                "Summarize all findings with evidence citations."
            )
            parsed = extract_json_from_response(reply)
            if parsed and parsed.get("action") == "finish":
                self._save_report(parsed.get("summary", ""))
            else:
                # Force report from evidence store
                self._save_report(
                    "# Investigation Report\n\n"
                    "The agent reached the maximum step limit. Below is the evidence chain.\n\n"
                    + self.evidence.to_markdown()
                )

        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{'=' * 70}")
        print(f"  Agent completed {len(self.evidence.steps)} investigation steps in {duration:.1f}s")
        print(f"  Report: {REPORT_PATH}")
        print(f"  Evidence: {EVIDENCE_JSON_PATH}")
        print(f"{'=' * 70}\n")

    def _save_report(self, summary: str):
        """Save the final investigation report and evidence chain."""
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Build the full report
        header = (
            "# Explainable Data Understanding Agent — Investigation Report\n\n"
            f"*Generated autonomously on {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            f"*Model: `{self.current_model}` | Steps: {len(self.evidence.steps)} | "
            f"Max Allowed: {MAX_STEPS}*\n\n"
            "Every conclusion in this report is backed by deterministic tool evidence. "
            "The LLM never computed statistics — it only reasoned over verified tool outputs.\n\n"
            "---\n\n"
        )

        report = header + summary + "\n\n---\n\n" + self.evidence.to_markdown()
        REPORT_PATH.write_text(report, encoding="utf-8")
        self.evidence.save_json(EVIDENCE_JSON_PATH)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = ExplainableDataAgent()
    agent.run()
