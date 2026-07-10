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

import requests
from tools import TOOL_REGISTRY
from evidence_store import EvidenceStore

# ─── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_ROOT / "docs" / "agent_investigation_report.md"
EVIDENCE_JSON_PATH = PROJECT_ROOT / "docs" / "investigation_evidence.json"

MAX_STEPS = 150
API_DELAY_SECONDS = 1  # Delay between API calls to stay under limits
MAX_RETRIES = 3        # Number of retries on API errors

# Optional Google GenAI SDK import
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

def load_business_context() -> str:
    """Load business goals and assignment tasks from my_task.md."""
    try:
        task_path = PROJECT_ROOT / "my_task.md"
        if task_path.exists():
            return task_path.read_text(encoding="utf-8")
    except Exception:
        pass
    return "No my_task.md found."

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

## Business Context & Target Tasks
You must investigate the data in order to answer the following specific tasks and questions:
{my_tasks}

## Available Tools
You have access to the following deterministic analysis tools. Call them by responding with a JSON block:

{tools_description}

## How to Respond

At each step, respond with EXACTLY ONE of these formats:

### Format 1: Call a tool
```json
{
  "action": "call_tool",
  "thought": "Your reasoning about why you're calling this tool",
  "tool": "tool_name",
  "args": {"param": "value"}
}
```

### Format 2: Record a conclusion (after seeing tool evidence)
```json
{
  "action": "conclude",
  "thought": "What this evidence tells you",
  "hypothesis": "The specific claim you tested",
  "verdict": "ACCEPTED or REJECTED",
  "reasoning": "Why, citing specific evidence"
}
```

### Format 3: End investigation and write final report
```json
{
  "action": "finish",
  "summary": "Your complete investigation report in markdown format. You MUST address each of the target tasks and questions from the business context in detail, citing specific tool evidence for each conclusion (especially on merging datasets, feature selection, anomalies, and labels)."
}
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
        self.evidence = EvidenceStore()
        self.start_time = datetime.now()
        self.conversation_history = []

        # Determine API Provider dynamically based on environment keys
        opencode_key = os.environ.get("OPENCODE_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        # Initialize fallback options
        self.fallback_provider = None
        self.fallback_key = None
        self.gemini_models = [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-pro",
            "gemini-3.5-flash"
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
            print(f"[AGENT BRAIN] Initialized OpenCode Zen API gateway. Prioritizing: {self.current_model}")
            if gemini_key:
                self.fallback_provider = "gemini"
                self.fallback_key = gemini_key
                print(f"[AGENT BRAIN] Gemini fallback configured (will hot-swap if OpenCode times out).")
        elif gemini_key:
            self.provider = "gemini"
            self.api_key = gemini_key
            self.prioritized_models = self.gemini_models
            self.current_model = self.prioritized_models[0]
            if HAS_GEMINI_SDK:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    self.has_sdk = True
                    print(f"[AGENT BRAIN] Initialized Gemini API via SDK. Prioritizing: {self.current_model}")
                except Exception:
                    self.has_sdk = False
                    print(f"[AGENT BRAIN] Initialized Gemini API via REST. Prioritizing: {self.current_model}")
            else:
                self.has_sdk = False
                print(f"[AGENT BRAIN] Initialized Gemini API via REST. Prioritizing: {self.current_model}")
        else:
            raise ValueError("No API key found. Please set OPENCODE_API_KEY or GEMINI_API_KEY in your .env file.")

        # Build system prompt with tool descriptions and business tasks
        tools_desc = build_tools_description()
        tasks_content = load_business_context()
        self.system_prompt = SYSTEM_PROMPT.replace("{tools_description}", tools_desc).replace("{my_tasks}", tasks_content)

    def _print_step_header(self, step: int):
        print(f"\n{'-' * 70}")
        print(f"  STEP {step} / {MAX_STEPS}")
        print(f"{'-' * 70}")

    def _call_llm(self, user_message: str) -> str:
        """Send a message to the active LLM provider (OpenCode Zen or Gemini) and get the response."""
        self.conversation_history.append({"role": "user", "content": user_message})

        # Rate-limit: wait between calls to avoid errors
        time.sleep(API_DELAY_SECONDS)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self.provider == "opencode":
                    payload = {
                        "model": self.current_model,
                        "messages": [{"role": "system", "content": self.system_prompt}] + self.conversation_history,
                        "temperature": 0.2
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    url = "https://opencode.ai/zen/v1/chat/completions"
                    res = requests.post(url, json=payload, headers=headers, timeout=35)

                    if res.status_code == 200:
                        data = res.json()
                        reply = data["choices"][0]["message"]["content"].strip()
                        self.conversation_history.append({"role": "assistant", "content": reply})
                        return reply
                    else:
                        raise Exception(f"status code {res.status_code}: {res.text}")

                elif self.provider == "gemini":
                    # Convert to Gemini parts format
                    contents = []
                    for msg in self.conversation_history:
                        role = "user" if msg["role"] == "user" else "model"
                        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

                    if self.has_sdk:
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
                        self.conversation_history.append({"role": "assistant", "content": reply})
                        return reply
                    else:
                        # REST API fallback
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.current_model}:generateContent?key={self.api_key}"
                        payload = {
                            "contents": contents,
                            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
                            "generationConfig": {
                                "temperature": 0.2,
                                "maxOutputTokens": 4096
                            }
                        }
                        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=35)
                        if res.status_code == 200:
                            data = res.json()
                            reply = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                            self.conversation_history.append({"role": "assistant", "content": reply})
                            return reply
                        else:
                            raise Exception(f"status code {res.status_code}: {res.text}")

            except Exception as e:
                error_str = str(e).lower()
                is_quota_error = any(kw in error_str for kw in ["429", "resource_exhausted", "quota", "rate"])
                is_timeout_or_net = any(kw in error_str for kw in ["timeout", "read timeout", "connect", "getaddrinfo", "connection"])

                # Dynamic Provider Fallback (Hot-Swap to Gemini if OpenCode has issues)
                if (is_quota_error or is_timeout_or_net) and self.fallback_provider == "gemini":
                    print(f"\n  [PROVIDER ROUTER] Primary provider '{self.provider}' encountered error: {str(e)[:100]}. Hot-swapping to fallback provider 'gemini'...")
                    self.provider = "gemini"
                    self.api_key = self.fallback_key
                    self.prioritized_models = self.gemini_models
                    self.current_model = self.prioritized_models[0]
                    self.fallback_provider = None  # Clear to prevent loops

                    if HAS_GEMINI_SDK:
                        try:
                            self.client = genai.Client(api_key=self.api_key)
                            self.has_sdk = True
                        except Exception:
                            self.has_sdk = False
                    else:
                        self.has_sdk = False

                    # Pop user message so retry doesn't duplicate it in history
                    if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                        self.conversation_history.pop()

                    return self._call_llm(user_message)

                # Dynamic Model Fallback: switch to next model in the provider's list
                if is_quota_error:
                    try:
                        current_idx = self.prioritized_models.index(self.current_model)
                        if current_idx + 1 < len(self.prioritized_models):
                            next_model = self.prioritized_models[current_idx + 1]
                            print(f"  [MODEL ROUTER] Quota exceeded for '{self.current_model}'. Switching to fallback model: '{next_model}'...")
                            self.current_model = next_model
                            # Retry immediately
                            continue
                    except ValueError:
                        pass

                is_retryable = is_quota_error or is_timeout_or_net
                if is_retryable and attempt < MAX_RETRIES:
                    wait = 15 * attempt  # 15s, 30s backoff
                    error_type = "RATE LIMIT" if is_quota_error else "NETWORK"
                    print(f"  [{error_type}] {str(e)[:80]}... Waiting {wait}s (retry {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                elif is_retryable:
                    # Final retry
                    wait = 30
                    print(f"  [RETRY {attempt}/{MAX_RETRIES}] Final retry. Waiting {wait}s...")
                    time.sleep(wait)
                    try:
                        if self.provider == "opencode":
                            payload = {
                                "model": self.current_model,
                                "messages": [{"role": "system", "content": self.system_prompt}] + self.conversation_history,
                                "temperature": 0.2
                            }
                            res = requests.post("https://opencode.ai/zen/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, timeout=35)
                            if res.status_code == 200:
                                reply = res.json()["choices"][0]["message"]["content"].strip()
                                self.conversation_history.append({"role": "assistant", "content": reply})
                                return reply
                        elif self.provider == "gemini":
                            contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in self.conversation_history]
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.current_model}:generateContent?key={self.api_key}"
                            payload = {
                                "contents": contents,
                                "systemInstruction": {"parts": [{"text": self.system_prompt}]},
                                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}
                            }
                            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=35)
                            if res.status_code == 200:
                                reply = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                                self.conversation_history.append({"role": "assistant", "content": reply})
                                return reply
                    except Exception:
                        pass
                else:
                    raise

        # Final fallback after all retries exhausted
        fallback = '{"action": "conclude", "thought": "API temporarily unavailable after retries.", "hypothesis": "N/A", "verdict": "SKIPPED", "reasoning": "Could not reach LLM. Will continue with next step."}'
        self.conversation_history.append({"role": "assistant", "content": fallback})
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

                # Truncate evidence for LLM prompt to prevent token bloat and timeouts
                evidence_for_prompt = evidence_str
                if len(evidence_str) > 1500:
                    evidence_for_prompt = evidence_str[:1500] + "\n... [Evidence truncated in LLM history to prevent timeout. Full data is saved in Evidence Store] ..."

                # Feed evidence back to LLM
                reply = self._call_llm(
                    f"Tool `{tool_name}` returned this evidence:\n```json\n{evidence_for_prompt}\n```\n"
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
