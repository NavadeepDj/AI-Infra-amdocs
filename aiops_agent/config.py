"""
config.py — Centralized Configuration for the Explainable AIOps Agent
=====================================================================
All paths, constants, and logging configuration in one place.
Nothing is hardcoded across modules.
"""

import logging
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_DIR = MODELS_DIR / "metadata"
DATASETS_DIR = PROJECT_ROOT / "datasets"

DATASET_PATH = DATASETS_DIR / "master_ml_dataset_v1.parquet"
DATASET_CSV_PATH = DATASETS_DIR / "master_ml_dataset_v1.csv"

IFOREST_PATH = MODELS_DIR / "isolation_forest.joblib"
XGB_3SLOT_PATH = MODELS_DIR / "xgboost_failure_3slot.joblib"
XGB_6SLOT_PATH = MODELS_DIR / "xgboost_failure_6slot.joblib"

FEATURE_ORDER_PATH = METADATA_DIR / "feature_order.json"
THRESHOLDS_PATH = METADATA_DIR / "thresholds.json"

DOCS_DIR = PROJECT_ROOT / "docs" / "aiops_agent"
EVIDENCE_EXPORT_DIR = DOCS_DIR

# ─── Agent Constants ─────────────────────────────────────────────────────────
MAX_AGENT_STEPS = 25          # Safety limit for reasoning loop per query
API_DELAY_SECONDS = 1         # Delay between LLM API calls
MAX_LLM_RETRIES = 3           # Max retries on LLM API errors
TOOL_MAX_RETRIES = 2          # Max retries for tool execution failures
TOOL_RETRY_DELAY_SECONDS = 0.5

STALE_DATA_THRESHOLD_HOURS = 8.0  # Warn if dataset is older than this
SHAP_SAMPLE_SIZE = 100            # Background sample for SHAP TreeExplainer

# ─── Display & Demo Configuration ────────────────────────────────────────────
SHOW_DATA_FRESHNESS = False       # Set True in production to warn operators when telemetry is stale
SHOW_ADVANCED_METADATA = False    # Set True to expose model_version, decision_threshold, inference_latency_ms to end-users


# ─── Hardware Subsystem Labels ───────────────────────────────────────────────
HARDWARE_STATUS_MAP = {
    -1: "No Sensor (Virtual/Unmonitored)",
    0: "OK",
    1: "Degraded / Warning",
    2: "Critical"
}

HARDWARE_SUBSYSTEMS = [
    ("hardware_cpu_worst_status", "CPU"),
    ("hardware_memory_worst_status", "Memory / RAM"),
    ("hardware_fans_worst_status", "Cooling Fans"),
    ("hardware_storage_worst_status", "Storage / Disk Array"),
    ("hardware_temperature_worst_status", "Temperature Sensors"),
    ("hardware_power_worst_status", "Power Supply"),
]

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_LEVEL = logging.INFO

def setup_logging():
    """Configure structured logging for the AIOps agent."""
    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
    # Suppress noisy libraries
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)
    return logging.getLogger("aiops_agent")

logger = setup_logging()
