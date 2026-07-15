"""
decision_policy.py — Deterministic Risk Classification for the AIOps Agent
===========================================================================
Pure-function module that converts raw model outputs into standardized
risk tiers (NORMAL / WARNING / CRITICAL) using explicit rules from
thresholds.json.

No ML. No LLM. Just rules.
"""

import json
import logging
from pathlib import Path

from . import config

logger = logging.getLogger("aiops_agent.decision_policy")

# Load thresholds once at module import
_thresholds = {}
try:
    with open(config.THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        _thresholds = json.load(f)
    logger.info("Decision policy loaded from thresholds.json")
except Exception as e:
    logger.error(f"Failed to load thresholds.json: {e}")


def classify_anomaly(
    anomaly_score: float,
    prediction: int,
    threshold: float = 0.0,
    percentile: float = 0.0,
    fleet_rank: int = 0,
    total_servers: int = 0,
    score_delta: float = 0.0,
) -> dict:
    """
    Classify an Isolation Forest result into a risk tier with a human-readable
    reason string explaining WHY the decision was reached.

    Works for both anomalous AND normal servers — explains normals too.

    Args:
        anomaly_score: Negated decision function (higher = more anomalous)
        prediction: Raw predict() output (-1 = anomaly, 1 = normal)
        threshold: Fleet anomaly threshold from contamination percentile
        percentile: Server's percentile rank across the fleet
        fleet_rank: Ordinal rank (1 = most anomalous)
        total_servers: Total servers in the fleet
        score_delta: How far above/below threshold (anomaly_score - threshold)

    Returns:
        {
            "is_anomaly": bool, "tier": str, "anomaly_score": float,
            "threshold": float, "percentile": float, "fleet_rank": int,
            "score_delta": float, "reason": str
        }
    """
    is_anomaly = prediction == -1
    tier = "ANOMALOUS" if is_anomaly else "NORMAL"
    contamination = _thresholds.get("isolation_forest", {}).get("contamination", 0.02)

    # Build human-readable reason string
    if is_anomaly:
        reason = (
            f"Anomaly score ({anomaly_score:.2f}) exceeded fleet threshold ({threshold:.2f}). "
            f"Server is in the {percentile:.1f}th percentile"
        )
        if fleet_rank > 0 and total_servers > 0:
            reason += f", ranked #{fleet_rank} of {total_servers} most anomalous."
        else:
            reason += "."
    else:
        if score_delta < -0.3:
            distance_desc = "well below"
        elif score_delta < -0.1:
            distance_desc = "below"
        else:
            distance_desc = "near but below"
        reason = (
            f"Anomaly score ({anomaly_score:.2f}) is {distance_desc} the fleet threshold ({threshold:.2f}). "
            f"Server operates within normal fleet behavior."
        )

    return {
        "is_anomaly": is_anomaly,
        "tier": tier,
        "anomaly_score": round(anomaly_score, 4),
        "threshold": round(threshold, 4),
        "score_delta": round(score_delta, 4),
        "percentile": round(percentile, 1),
        "fleet_rank": fleet_rank,
        "contamination": contamination,
        "reason": reason,
    }


def classify_failure_risk(probability: float, horizon: str = "3slot") -> dict:
    """
    Classify a failure prediction probability into NORMAL / WARNING / CRITICAL
    using the risk tier boundaries from thresholds.json.

    Args:
        probability: Raw probability from XGBoost (0.0 to 1.0)
        horizon: "3slot" (12h) or "6slot" (24h)

    Returns:
        {"probability_pct": float, "risk_tier": str, "action": str,
         "exceeds_optimal_f1": bool, "optimal_f1_threshold": float}
    """
    target_key = f"target_failure_{horizon}"
    target_config = _thresholds.get(target_key, {})
    risk_tiers = target_config.get("risk_tiers", {})
    optimal_f1 = target_config.get("optimal_f1_threshold", 0.5)

    prob_pct = round(probability * 100, 1)

    # Determine tier
    risk_tier = "NORMAL"
    action = "Server stable; no action required."

    for tier_name, tier_bounds in risk_tiers.items():
        min_p = tier_bounds.get("min_prob", 0.0)
        max_p = tier_bounds.get("max_prob", 1.0)
        if min_p <= probability < max_p:
            risk_tier = tier_name
            action = tier_bounds.get("action", "")
            break
    # Handle edge case: probability == 1.0 -> CRITICAL
    if probability >= 1.0:
        risk_tier = "CRITICAL"
        action = risk_tiers.get("CRITICAL", {}).get("action", "Immediate attention required.")

    return {
        "probability_pct": prob_pct,
        "risk_tier": risk_tier,
        "action": action,
        "exceeds_optimal_f1": probability >= optimal_f1,
        "optimal_f1_threshold": optimal_f1,
        "lookahead_window": target_config.get("lookahead_window", "Unknown")
    }


def compute_composite_risk(anomaly_result: dict, failure_12h: dict, failure_24h: dict) -> dict:
    """
    Compute an overall composite risk assessment combining all three engines.

    Rules:
    - If Isolation Forest flags anomalous AND XGBoost 12h >= CRITICAL -> "CRITICAL"
    - If any engine alone flags WARNING or higher -> "WARNING"
    - If all engines show NORMAL -> "NORMAL"

    Returns:
        {"composite_risk": str, "contributing_signals": list, "summary": str}
    """
    signals = []

    is_anomaly = anomaly_result.get("is_anomaly", False)
    risk_12h = failure_12h.get("risk_tier", "NORMAL")
    risk_24h = failure_24h.get("risk_tier", "NORMAL")

    if is_anomaly:
        signals.append(f"Isolation Forest: ANOMALOUS (score={anomaly_result.get('anomaly_score', 'N/A')})")
    if risk_12h in ("WARNING", "CRITICAL"):
        signals.append(f"XGBoost 12h: {risk_12h} ({failure_12h.get('probability_pct', 0)}%)")
    if risk_24h in ("WARNING", "CRITICAL"):
        signals.append(f"XGBoost 24h: {risk_24h} ({failure_24h.get('probability_pct', 0)}%)")

    # Composite logic
    if is_anomaly and risk_12h == "CRITICAL":
        composite = "CRITICAL"
        summary = "Server is both anomalous AND predicted to fail within 12 hours. Immediate intervention required."
    elif risk_12h == "CRITICAL" or risk_24h == "CRITICAL":
        composite = "CRITICAL"
        summary = "High failure probability detected. Dispatch SRE remediation ticket."
    elif is_anomaly or risk_12h == "WARNING" or risk_24h == "WARNING":
        composite = "WARNING"
        summary = "Elevated risk signals detected. Schedule preventative inspection within 24 hours."
    else:
        composite = "NORMAL"
        summary = "All engines report normal operating conditions."

    return {
        "composite_risk": composite,
        "contributing_signals": signals,
        "summary": summary,
        "anomaly_tier": anomaly_result.get("tier", "NORMAL"),
        "failure_12h_tier": risk_12h,
        "failure_24h_tier": risk_24h
    }
