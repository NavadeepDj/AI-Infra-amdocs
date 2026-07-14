"""
evidence_store.py — Enhanced Immutable Evidence Store for the AIOps Agent
=========================================================================
Every reasoning step records: tool called, arguments, evidence returned,
latency, status, and conclusion. Every claim in the final answer is traceable.

Enhanced over the original EDA agent's evidence_store.py with:
- Latency tracking per tool call
- Tool status (SUCCESS / ERROR / SKIPPED)
- Data staleness tracking
- Session-level query logging & metrics
"""

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("aiops_agent.evidence_store")


class EvidenceStore:
    """
    Immutable audit trail for the AIOps Agent.
    Every tool call, hypothesis, and verdict is recorded with timestamps
    and execution metrics.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.now()
        self.steps: list[dict] = []
        self.query_log: list[dict] = []
        self._current_query_tools: list[str] = []
        self._current_query_start: float | None = None

    def start_query(self, query: str):
        """Mark the beginning of processing a user query."""
        import time
        self._current_query_start = time.time()
        self._current_query_tools = []
        self._current_query_text = query

    def end_query(self, answer: str):
        """Mark the end of processing a user query and log metrics."""
        import time
        latency = 0.0
        if self._current_query_start is not None:
            latency = (time.time() - self._current_query_start) * 1000  # ms

        self.query_log.append({
            "query": getattr(self, "_current_query_text", ""),
            "tools_called": list(self._current_query_tools),
            "total_latency_ms": round(latency, 1),
            "answer_length": len(answer),
            "timestamp": datetime.now().isoformat()
        })
        self._current_query_start = None

    def record(
        self,
        thought: str,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        evidence: dict | None = None,
        conclusion: str | None = None,
        latency_ms: float = 0.0,
        tool_status: str = "SUCCESS",
        data_staleness_hours: float | None = None,
        audit_metadata: dict | None = None,
    ) -> int:
        """Record a single investigation step. Returns the step number."""
        step_num = len(self.steps) + 1
        entry = {
            "step": step_num,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "thought": thought,
            "tool_name": tool_name,
            "tool_args": tool_args or {},
            "evidence": evidence or {},
            "conclusion": conclusion,
            "latency_ms": round(latency_ms, 2),
            "tool_status": tool_status,
        }
        if data_staleness_hours is not None:
            entry["data_staleness_hours"] = data_staleness_hours
        if audit_metadata:
            entry["audit_metadata"] = audit_metadata

        self.steps.append(entry)

        if tool_name:
            self._current_query_tools.append(tool_name)

        return step_num

    def get_tools_called(self) -> list[str]:
        """Return list of all tool names called in the current session."""
        return [s["tool_name"] for s in self.steps if s.get("tool_name")]

    def get_evidence_for_tool(self, tool_name: str) -> list[dict]:
        """Retrieve all evidence returned by a specific tool."""
        return [s["evidence"] for s in self.steps
                if s.get("tool_name") == tool_name and s.get("evidence")]

    def get_metrics_summary(self) -> dict:
        """Compute aggregate performance metrics for the session."""
        total_steps = len(self.steps)
        tool_calls = [s for s in self.steps if s.get("tool_name")]
        error_calls = [s for s in tool_calls if s.get("tool_status") == "ERROR"]
        latencies = [s["latency_ms"] for s in tool_calls if s.get("latency_ms", 0) > 0]

        # Tool usage counts
        tool_counts = {}
        for s in tool_calls:
            name = s["tool_name"]
            tool_counts[name] = tool_counts.get(name, 0) + 1

        return {
            "session_id": self.session_id,
            "total_steps": total_steps,
            "total_tool_calls": len(tool_calls),
            "total_errors": len(error_calls),
            "error_rate_pct": round(len(error_calls) / max(len(tool_calls), 1) * 100, 1),
            "tool_usage_counts": tool_counts,
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1) if latencies else 0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
            "total_queries": len(self.query_log),
        }

    def get_summary(self) -> str:
        """Return a compact text summary of all steps for context window."""
        lines = []
        for s in self.steps:
            lines.append(f"Step {s['step']}:")
            lines.append(f"  Thought: {s['thought']}")
            if s["tool_name"]:
                args_str = json.dumps(s["tool_args"], default=str)
                lines.append(f"  Tool: {s['tool_name']}({args_str}) [{s['latency_ms']:.0f}ms, {s['tool_status']}]")
                ev_str = json.dumps(s["evidence"], default=str)
                if len(ev_str) > 600:
                    ev_str = ev_str[:600] + "...(truncated)"
                lines.append(f"  Evidence: {ev_str}")
            if s["conclusion"]:
                lines.append(f"  Conclusion: {s['conclusion']}")
            lines.append("")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Export the full evidence chain as markdown."""
        lines = ["## Investigation Evidence Chain\n"]
        for s in self.steps:
            lines.append(f"### Step {s['step']} — {s['timestamp']}")
            lines.append(f"**Thought:** {s['thought']}\n")
            if s["tool_name"]:
                lines.append(f"**Tool Called:** `{s['tool_name']}` "
                             f"(Latency: `{s['latency_ms']:.1f}ms`, Status: `{s['tool_status']}`)\n")
                lines.append(f"**Arguments:** `{json.dumps(s['tool_args'], default=str)}`\n")
                lines.append(f"**Evidence Returned:**")
                lines.append(f"```json\n{json.dumps(s['evidence'], indent=2, default=str)}\n```\n")
            if s["conclusion"]:
                lines.append(f"**Conclusion:** {s['conclusion']}\n")
            lines.append("---\n")
        return "\n".join(lines)

    def to_evaluation_record(self) -> dict:
        """Export structured format for the evaluation framework."""
        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "total_steps": len(self.steps),
            "tools_called": self.get_tools_called(),
            "query_log": self.query_log,
            "metrics": self.get_metrics_summary(),
            "steps": self.steps
        }

    def save_json(self, path: Path):
        """Save the full evidence chain as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_evaluation_record(), f, indent=2, default=str)

    def reset_for_new_query(self):
        """Clear steps for a new query while preserving session-level metadata."""
        self.steps = []
        self._current_query_tools = []
