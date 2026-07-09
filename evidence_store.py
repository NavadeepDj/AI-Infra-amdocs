"""
=============================================================================
evidence_store.py — Structured Evidence Store for the Explainable Data Understanding Agent
=============================================================================

Tracks every step of the investigation: hypotheses, tool calls, evidence,
verdicts, and reasoning. Every conclusion in the final report can be traced
back to a specific tool output stored here.
=============================================================================
"""

import json
from datetime import datetime
from pathlib import Path


class EvidenceStore:
    """
    Immutable log of every investigative step.
    Each entry records: what was hypothesized, which tool was called,
    what evidence was returned, and what conclusion was drawn.
    """

    def __init__(self):
        self.steps: list[dict] = []

    def record(
        self,
        thought: str,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        evidence: dict | None = None,
        conclusion: str | None = None,
    ) -> int:
        """Record a single investigation step. Returns the step number."""
        step_num = len(self.steps) + 1
        entry = {
            "step": step_num,
            "timestamp": datetime.now().isoformat(),
            "thought": thought,
            "tool_name": tool_name,
            "tool_args": tool_args or {},
            "evidence": evidence or {},
            "conclusion": conclusion,
        }
        self.steps.append(entry)
        return step_num

    def get_summary(self) -> str:
        """Return a compact text summary of all steps so far for the LLM context window."""
        lines = []
        for s in self.steps:
            lines.append(f"Step {s['step']}:")
            lines.append(f"  Thought: {s['thought']}")
            if s["tool_name"]:
                args_str = json.dumps(s["tool_args"], default=str)
                lines.append(f"  Tool: {s['tool_name']}({args_str})")
                # Truncate large evidence to keep context manageable
                ev_str = json.dumps(s["evidence"], default=str)
                if len(ev_str) > 600:
                    ev_str = ev_str[:600] + "...(truncated)"
                lines.append(f"  Evidence: {ev_str}")
            if s["conclusion"]:
                lines.append(f"  Conclusion: {s['conclusion']}")
            lines.append("")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Export the full evidence chain as a markdown document."""
        lines = ["## Investigation Evidence Chain\n"]
        for s in self.steps:
            lines.append(f"### Step {s['step']} — {s['timestamp']}")
            lines.append(f"**Thought:** {s['thought']}\n")
            if s["tool_name"]:
                lines.append(f"**Tool Called:** `{s['tool_name']}`\n")
                lines.append(f"**Arguments:** `{json.dumps(s['tool_args'], default=str)}`\n")
                lines.append(f"**Evidence Returned:**")
                lines.append(f"```json\n{json.dumps(s['evidence'], indent=2, default=str)}\n```\n")
            if s["conclusion"]:
                lines.append(f"**Conclusion:** {s['conclusion']}\n")
            lines.append("---\n")
        return "\n".join(lines)

    def save_json(self, path: Path):
        """Save the full evidence chain as JSON for programmatic access."""
        path.write_text(json.dumps(self.steps, indent=2, default=str), encoding="utf-8")
