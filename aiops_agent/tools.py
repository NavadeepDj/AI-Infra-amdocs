"""
tools.py — Deterministic Infrastructure Tools for the AIOps Agent
==================================================================
Nine tools that wrap the processed dataset and serialized ML models.
Every tool:
- Accepts simple arguments (typically just server_name)
- Returns a structured dict (never prints, never uses LLM)
- Catches exceptions gracefully: returns {"error": "..."} instead of crashing
- Includes retry logic for transient failures
- Never hallucinates — returns only reproducible facts

The LLM agent calls these tools and reasons over their structured outputs.
"""

import time
import logging
import functools
from typing import Callable
import pandas as pd

def safe_int(val, default=0) -> int:
    if val is None or pd.isna(val):
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def safe_float(val, default=0.0) -> float:
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

from . import config
from .decision_policy import classify_anomaly, classify_failure_risk, compute_composite_risk

logger = logging.getLogger("aiops_agent.tools")

# ─── Shared ModelManager Instance ────────────────────────────────────────────
# Initialized once by the agent, then injected here
_manager = None


def set_model_manager(manager):
    """Inject the ModelManager instance (called once by agent.py at startup)."""
    global _manager
    _manager = manager
    logger.info("Tool registry linked to ModelManager")


# ─── Retry Decorator ────────────────────────────────────────────────────────
def with_retry(func: Callable) -> Callable:
    """
    Retry a tool function up to TOOL_MAX_RETRIES times on failure.
    On final failure, return a graceful error dict instead of crashing.
    
    Flow: Tool failed → Retry → Still fails? → Graceful response
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(1, config.TOOL_MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < config.TOOL_MAX_RETRIES:
                    logger.warning(f"[RETRY {attempt}/{config.TOOL_MAX_RETRIES}] "
                                   f"Tool '{func.__name__}' failed: {e}. Retrying...")
                    time.sleep(config.TOOL_RETRY_DELAY_SECONDS)
        # All retries exhausted — graceful response
        logger.error(f"[TOOL FAILED] '{func.__name__}' after {config.TOOL_MAX_RETRIES} retries: {last_error}")
        return {
            "error": f"Tool '{func.__name__}' failed after {config.TOOL_MAX_RETRIES} attempts: {str(last_error)}",
            "tool": func.__name__,
            "retries_exhausted": True
        }
    return wrapper


# ─── Tool 1: get_server_telemetry ───────────────────────────────────────────

@with_retry
def get_server_telemetry(server_name: str) -> dict:
    """
    Get the latest instantaneous telemetry snapshot for a specific server.
    Returns ping status, all 6 hardware subsystem statuses, component counts,
    and raw diagnostic text alerts.
    """
    row = _manager.get_server_row(server_name)
    if row is None:
        return {"error": f"Server '{server_name}' not found",
                "available_sample": _manager.list_all_servers()[:5]}

    # Build hardware subsystem status breakdown
    hw_status = {}
    for col, label in config.HARDWARE_SUBSYSTEMS:
        val = row.get(col)
        if val is not None and not pd.isna(val):
            int_val = safe_int(val, default=-999)
            hw_status[label] = config.HARDWARE_STATUS_MAP.get(int_val, f"Unknown ({val})")
        else:
            hw_status[label] = "No Sensor (Virtual/Unmonitored)"

    result = {
        "server_name": server_name,
        "timestamp": str(row.get("event_time_ping", "Unknown")),
        "monitoring_slot": str(row.get("monitoring_slot", "Unknown")),
        "ip_address": str(row.get("ip_address", "Unknown")),
        "telemetry_source": str(row.get("telemetry_source", "Unknown")),
        "ping_status": "Unreachable" if safe_int(row.get("ping_status_binary", 0)) == 1 else "Reachable",
        "hardware_subsystems": hw_status,
        "critical_component_count": safe_int(row.get("critical_component_count", 0)),
        "degraded_component_count": safe_int(row.get("degraded_component_count", 0)),
        "not_ok_component_count": safe_int(row.get("not_ok_component_count", 0)),
    }

    # Add raw diagnostic text if available
    for diag_col in ["hpe_current_problems", "dell_issues_detected"]:
        val = row.get(diag_col)
        if val is not None and str(val) not in ("nan", "", "None"):
            result[diag_col] = str(val)

    # Data staleness
    if config.SHOW_DATA_FRESHNESS:
        staleness = _manager.get_staleness_warning()
        if staleness:
            result["data_staleness_warning"] = staleness

    return result


# ─── Tool 2: get_server_history ─────────────────────────────────────────────

@with_retry
def get_server_history(server_name: str, n_slots: str = "6") -> dict:
    """
    Get the rolling history for a server over the last N monitoring slots.
    Shows the trajectory of ping timeout rates, problem accumulation,
    lag values, and hardware status changes over time.
    """
    n = int(n_slots)
    history_df = _manager.get_server_history(server_name, n_slots=n)
    if history_df.empty:
        return {"error": f"Server '{server_name}' not found or has no history",
                "available_sample": _manager.list_all_servers()[:5]}

    timeline = []
    for _, row in history_df.iterrows():
        slot_data = {
            "timestamp": str(row.get("event_time_ping", "")),
            "slot": str(row.get("monitoring_slot", "")),
            "ping_status": "Unreachable" if safe_int(row.get("ping_status_binary", 0)) == 1 else "Reachable",
            "ping_timeout_rate_3slot": round(safe_float(row.get("ping_timeout_rate_3slot", 0)), 3),
            "ping_timeout_rate_6slot": round(safe_float(row.get("ping_timeout_rate_6slot", 0)), 3),
            "problems_active_sum_6slot": safe_int(row.get("problems_active_sum_6slot", 0)),
            "critical_components": safe_int(row.get("critical_component_count", 0)),
            "degraded_components": safe_int(row.get("degraded_component_count", 0)),
        }
        timeline.append(slot_data)

    # Compute trend indicators
    if len(timeline) >= 2:
        first_timeout = timeline[0].get("ping_timeout_rate_6slot", 0)
        last_timeout = timeline[-1].get("ping_timeout_rate_6slot", 0)
        timeout_trend = "INCREASING" if last_timeout > first_timeout else "DECREASING" if last_timeout < first_timeout else "STABLE"

        first_problems = timeline[0].get("problems_active_sum_6slot", 0)
        last_problems = timeline[-1].get("problems_active_sum_6slot", 0)
        problem_trend = "INCREASING" if last_problems > first_problems else "DECREASING" if last_problems < first_problems else "STABLE"
    else:
        timeout_trend = "INSUFFICIENT_DATA"
        problem_trend = "INSUFFICIENT_DATA"

    return {
        "server_name": server_name,
        "slots_returned": len(timeline),
        "timeline": timeline,
        "trends": {
            "ping_timeout_trend": timeout_trend,
            "problem_accumulation_trend": problem_trend
        }
    }


# ─── Tool 3: detect_anomaly ────────────────────────────────────────────────

@with_retry
def detect_anomaly(server_name: str) -> dict:
    """
    Run the Isolation Forest anomaly detector on a specific server.
    Returns the anomaly score, classification (NORMAL/ANOMALOUS), and
    contamination cutoff used.
    """
    raw = _manager.score_iforest(server_name)
    if "error" in raw:
        return raw

    classified = classify_anomaly(
        anomaly_score=raw["anomaly_score"],
        prediction=raw["isolation_forest_prediction"]
    )

    res = {
        "server_name": server_name,
        "anomaly_score": raw["anomaly_score"],
        "is_anomaly": classified["is_anomaly"],
        "tier": classified["tier"],
        "contamination": classified["contamination"],
        "interpretation": "This server exhibits unusual behavior compared to the fleet."
            if classified["is_anomaly"]
            else "This server is operating within normal fleet behavior."
    }
    if config.SHOW_ADVANCED_METADATA:
        res["metadata"] = raw.get("metadata", {})
    else:
        res["_audit_metadata"] = raw.get("metadata", {})
    return res


# ─── Tool 4: predict_failure_12h ───────────────────────────────────────────

@with_retry
def predict_failure_12h(server_name: str) -> dict:
    """
    Run the XGBoost 12-hour failure prediction on a specific server.
    Returns failure probability, risk tier (NORMAL/WARNING/CRITICAL),
    and the recommended SRE action.
    """
    raw = _manager.score_xgb(server_name, horizon="3slot")
    if "error" in raw:
        return raw

    classified = classify_failure_risk(
        probability=raw["failure_probability"],
        horizon="3slot"
    )

    res = {
        "server_name": server_name,
        "failure_probability_pct": classified["probability_pct"],
        "risk_tier": classified["risk_tier"],
        "action": classified["action"],
        "exceeds_optimal_f1": classified["exceeds_optimal_f1"],
        "optimal_f1_threshold": classified["optimal_f1_threshold"],
        "lookahead_window": classified["lookahead_window"],
    }
    if config.SHOW_ADVANCED_METADATA:
        res["metadata"] = raw.get("metadata", {})
    else:
        res["_audit_metadata"] = raw.get("metadata", {})
    return res


# ─── Tool 5: predict_failure_24h ───────────────────────────────────────────

@with_retry
def predict_failure_24h(server_name: str) -> dict:
    """
    Run the XGBoost 24-hour failure prediction on a specific server.
    Returns failure probability, risk tier, and recommended SRE action.
    """
    raw = _manager.score_xgb(server_name, horizon="6slot")
    if "error" in raw:
        return raw

    classified = classify_failure_risk(
        probability=raw["failure_probability"],
        horizon="6slot"
    )

    res = {
        "server_name": server_name,
        "failure_probability_pct": classified["probability_pct"],
        "risk_tier": classified["risk_tier"],
        "action": classified["action"],
        "exceeds_optimal_f1": classified["exceeds_optimal_f1"],
        "optimal_f1_threshold": classified["optimal_f1_threshold"],
        "lookahead_window": classified["lookahead_window"],
    }
    if config.SHOW_ADVANCED_METADATA:
        res["metadata"] = raw.get("metadata", {})
    else:
        res["_audit_metadata"] = raw.get("metadata", {})
    return res


# ─── Tool 6: explain_prediction ────────────────────────────────────────────

@with_retry
def explain_prediction(server_name: str, horizon: str = "12h") -> dict:
    """
    Compute SHAP explainability for a server's failure prediction.
    Returns the top contributing features with their SHAP values,
    separating positive drivers (pushing toward failure) from
    protective factors (pushing toward normal).
    """
    model_horizon = "3slot" if horizon in ("12h", "3slot") else "6slot"
    shap_result = _manager.get_shap_explanation(server_name, horizon=model_horizon)
    if "error" in shap_result:
        return shap_result

    # Get the prediction to contextualize
    pred = _manager.score_xgb(server_name, horizon=model_horizon)
    classified = classify_failure_risk(
        probability=pred.get("failure_probability", 0.0),
        horizon=model_horizon
    )

    # Annotate positive drivers with operational context for missing sensors (-1)
    missing_sensor_notes = []
    for driver in shap_result.get("positive_drivers", []):
        feat = driver.get("feature", "")
        val = driver.get("value")
        if val in (-1, -1.0) and ("hardware_" in feat or "status" in feat):
            missing_sensor_notes.append(
                f"Note on '{feat} = -1': -1 indicates Sensor Missing/Unmonitored (not a confirmed hardware failure). "
                f"XGBoost treats missing sensors as statistically informative because unmonitored/virtual servers historical baseline risk is elevated."
            )

    if config.SHOW_DATA_FRESHNESS:
        stale_info = _manager.get_staleness_warning()
        staleness_note = f" {stale_info.get('natural_phrasing', '')}" if stale_info else ""
    else:
        stale_info = None
        staleness_note = ""

    res = {
        "server_name": server_name,
        "horizon": horizon,
        "failure_probability_pct": classified["probability_pct"],
        "risk_tier": classified["risk_tier"],
        "shap": {
            "top_drivers": shap_result.get("top_drivers", []),
            "positive_drivers_toward_failure": shap_result.get("positive_drivers", []),
            "protective_factors_toward_normal": shap_result.get("protective_factors", []),
            "base_value": shap_result.get("base_value")
        },
        "top_drivers": shap_result.get("top_drivers", []),
        "positive_drivers_toward_failure": shap_result.get("positive_drivers", []),
        "protective_factors_toward_normal": shap_result.get("protective_factors", []),
        "base_value": shap_result.get("base_value"),
        "operational_sensor_notes": missing_sensor_notes,
        "interpretation": (
            f"The model predicts {classified['probability_pct']}% failure probability ({classified['risk_tier']}). "
            f"Strongest drivers pushing toward failure: {[d['feature'] for d in shap_result.get('positive_drivers', [])[:3]]}. "
            f"{' '.join(missing_sensor_notes)}{staleness_note}"
        )
    }
    if config.SHOW_ADVANCED_METADATA:
        res["metadata"] = pred.get("metadata", {})
    else:
        res["_audit_metadata"] = pred.get("metadata", {})
    if config.SHOW_DATA_FRESHNESS and stale_info:
        res["data_staleness"] = stale_info
    return res


# ─── Tool 7: get_fleet_summary ─────────────────────────────────────────────

@with_retry
def get_fleet_summary() -> dict:
    """
    Get a fleet-wide health summary across all monitored servers.
    Returns total servers, counts by risk tier, anomalies detected,
    and the top 5 highest-risk servers.
    """
    servers = _manager.list_all_servers()
    total = len(servers)

    # Score ALL servers using the latest observation
    anomaly_count = 0
    tier_counts = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
    risk_list = []

    for srv in servers:
        # XGBoost 12h prediction
        pred = _manager.score_xgb(srv, horizon="3slot")
        if "error" in pred:
            continue
        classified = classify_failure_risk(pred["failure_probability"], "3slot")
        tier = classified["risk_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Isolation Forest
        iforest = _manager.score_iforest(srv)
        if not iforest.get("error") and iforest.get("is_anomaly", False):
            anomaly_count += 1

        if tier in ("WARNING", "CRITICAL"):
            risk_list.append({
                "server_name": srv,
                "failure_probability_pct": classified["probability_pct"],
                "risk_tier": tier
            })

    # Sort highest risk first
    risk_list.sort(key=lambda x: x["failure_probability_pct"], reverse=True)

    result = {
        "total_servers": total,
        "healthy_normal": tier_counts.get("NORMAL", 0),
        "warning": tier_counts.get("WARNING", 0),
        "critical": tier_counts.get("CRITICAL", 0),
        "anomalies_detected": anomaly_count,
        "top_5_at_risk": risk_list[:5],
    }

    if config.SHOW_DATA_FRESHNESS:
        staleness = _manager.get_staleness_warning()
        if staleness:
            result["data_staleness_warning"] = staleness

    return result


# ─── Tool 8: get_model_metadata ────────────────────────────────────────────

@with_retry
def get_model_metadata() -> dict:
    """
    Return the model metadata: feature lists, imputation rules,
    risk tier boundaries, and threshold configuration.
    """
    return {
        "feature_order": _manager.get_feature_meta(),
        "thresholds": _manager.get_thresholds(),
        "models_available": {
            "isolation_forest": _manager._iforest is not None,
            "xgboost_12h": _manager._xgb_3slot is not None,
            "xgboost_24h": _manager._xgb_6slot is not None,
        },
        "data_freshness_hours": _manager.data_age_hours(),
        "total_servers": len(_manager.list_all_servers())
    }


# ─── Tool 9: get_recent_alerts ─────────────────────────────────────────────

@with_retry
def get_recent_alerts(server_name: str = "") -> dict:
    """
    Get recent monitoring slots where servers crossed WARNING or CRITICAL.
    If server_name is provided, filters to that specific server.
    If empty, returns the latest fleet-wide alerts.
    """
    servers = [server_name] if server_name else _manager.list_all_servers()
    alerts = []

    for srv in servers:
        history = _manager.get_server_history(srv, n_slots=6)
        if history.empty:
            continue

        for _, row in history.iterrows():
            # Check if this slot has any warning signs
            timeout_rate = safe_float(row.get("ping_timeout_rate_6slot", 0))
            critical_count = safe_int(row.get("critical_component_count", 0))
            problems_sum = safe_int(row.get("problems_active_sum_6slot", 0))
            ping_down = safe_int(row.get("ping_status_binary", 0))

            if critical_count > 0 or timeout_rate > 0.5 or problems_sum >= 3 or ping_down == 1:
                alerts.append({
                    "server_name": srv,
                    "timestamp": str(row.get("event_time_ping", "")),
                    "slot": str(row.get("monitoring_slot", "")),
                    "alert_reasons": [],
                    "ping_status": "Unreachable" if ping_down else "Reachable",
                    "ping_timeout_rate_6slot": round(timeout_rate, 3),
                    "critical_components": critical_count,
                    "problems_active_sum_6slot": problems_sum
                })
                # Add specific reasons
                if critical_count > 0:
                    alerts[-1]["alert_reasons"].append(f"Critical hardware: {critical_count} component(s)")
                if timeout_rate > 0.5:
                    alerts[-1]["alert_reasons"].append(f"High ping timeout: {timeout_rate*100:.0f}%")
                if problems_sum >= 3:
                    alerts[-1]["alert_reasons"].append(f"Sustained problems: {problems_sum} active over 24h")
                if ping_down:
                    alerts[-1]["alert_reasons"].append("Server unreachable")

    # Sort by severity (most recent / highest timeout first)
    alerts.sort(key=lambda x: x.get("ping_timeout_rate_6slot", 0), reverse=True)

    result = {
        "total_alerts": len(alerts),
        "alerts": alerts[:20]  # Cap at 20 for readability
    }

    if server_name:
        result["server_name"] = server_name

    return result


@with_retry
def find_server_by_ip(ip_address: str) -> dict:
    """
    Find the canonical server hostname matching a given IP address.
    Returns the hostname, IP, and lookup status.
    """
    hostname = _manager.find_server_by_ip(ip_address)
    if hostname:
        return {
            "hostname": hostname,
            "ip": ip_address,
            "found": True
        }
    return {
        "hostname": None,
        "ip": ip_address,
        "found": False,
        "error": f"No server found with IP address '{ip_address}'"
    }


# ─── Tool Registry ──────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "find_server_by_ip": {
        "function": find_server_by_ip,
        "description": "Find the canonical server hostname for a given IP address. SREs can use this to resolve an IP address first, and then continue querying other tools using the resolved hostname.",
        "parameters": {"ip_address": "Exact IP address of the server (e.g., '172.19.30.142')"},
    },
    "get_server_telemetry": {
        "function": get_server_telemetry,
        "description": "Get the latest telemetry snapshot for a specific server: ping status, "
                       "hardware subsystem health (CPU, Memory, Fans, Storage, Temperature, Power), "
                       "component counts, and diagnostic alerts.",
        "parameters": {"server_name": "Exact server hostname (e.g., 'v5G-AMF-01')"},
    },
    "get_server_history": {
        "function": get_server_history,
        "description": "Get the rolling history trajectory for a server over the last N monitoring "
                       "slots (default 6 = 24 hours). Shows ping timeout rates, problem accumulation, "
                       "and trend direction (INCREASING/DECREASING/STABLE).",
        "parameters": {
            "server_name": "Exact server hostname",
            "n_slots": "Number of historical slots to retrieve (default: '6' = 24 hours)"
        },
    },
    "detect_anomaly": {
        "function": detect_anomaly,
        "description": "Run the Isolation Forest anomaly detector on a server. Returns anomaly score, "
                       "classification (NORMAL/ANOMALOUS), and fleet comparison. Use this to check if a "
                       "server's current behavior is unusual compared to the fleet.",
        "parameters": {"server_name": "Exact server hostname"},
    },
    "predict_failure_12h": {
        "function": predict_failure_12h,
        "description": "Predict the probability of server failure within the next 12 hours using "
                       "XGBoost. Returns probability percentage, risk tier (NORMAL/WARNING/CRITICAL), "
                       "and recommended SRE action.",
        "parameters": {"server_name": "Exact server hostname"},
    },
    "predict_failure_24h": {
        "function": predict_failure_24h,
        "description": "Predict the probability of server failure within the next 24 hours using "
                       "XGBoost. Returns probability percentage, risk tier, and recommended action.",
        "parameters": {"server_name": "Exact server hostname"},
    },
    "explain_prediction": {
        "function": explain_prediction,
        "description": "Explain WHY a failure prediction was made using SHAP explainability. Returns "
                       "the top contributing features pushing toward failure and protective factors "
                       "keeping risk lower. MUST be called when failure probability exceeds the optimal "
                       "F1 threshold.",
        "parameters": {
            "server_name": "Exact server hostname",
            "horizon": "Prediction horizon: '12h' or '24h' (default: '12h')"
        },
    },
    "get_fleet_summary": {
        "function": get_fleet_summary,
        "description": "Get a fleet-wide infrastructure health summary. Returns total server count, "
                       "breakdown by risk tier (NORMAL/WARNING/CRITICAL), anomaly count, and the "
                       "top 5 highest-risk servers. Use for daily health reports or fleet overviews.",
        "parameters": {},
    },
    "get_model_metadata": {
        "function": get_model_metadata,
        "description": "Return model metadata including the exact feature list used by each model, "
                       "imputation rules, risk tier boundaries, threshold values, and which models "
                       "are currently available.",
        "parameters": {},
    },
    "get_recent_alerts": {
        "function": get_recent_alerts,
        "description": "Get recent monitoring slots where servers showed WARNING or CRITICAL signals: "
                       "critical hardware components, high ping timeout rates, sustained problems, or "
                       "unreachable status. Filter by server_name or leave empty for fleet-wide alerts.",
        "parameters": {"server_name": "Server hostname to filter (optional, leave empty for fleet-wide)"},
    },
}
