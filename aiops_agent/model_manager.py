"""
model_manager.py — Singleton ModelManager for the Explainable AIOps Agent
=========================================================================
Loads all serialized .joblib models and the parquet dataset ONCE at startup.
Provides sub-20ms inference methods for downstream tools.

Key Design:
- Load-once caching: models stay in memory for the agent's lifetime
- Schema validation: verifies feature columns match feature_order.json
- Stale data detection: warns if dataset is older than threshold
- Graceful degradation: missing models marked unavailable, not crashing
"""

import json
import time
from datetime import datetime, timezone
import logging
import numpy as np
import pandas as pd
try:
    pd.set_option("future.no_silent_downcasting", True)
except (pd.errors.OptionError, AttributeError):
    pass
from pathlib import Path

import joblib
import shap

from . import config

logger = logging.getLogger("aiops_agent.model_manager")


class SchemaValidationError(Exception):
    """Raised when dataset columns don't match the expected feature schema."""
    pass


class ModelManager:
    """
    Singleton-style model and data manager.
    Call __init__ once at agent startup. All tools reference the same instance.
    """

    def __init__(self):
        logger.info("Initializing ModelManager — loading models and dataset...")
        t0 = time.time()

        # ── Load Metadata ────────────────────────────────────────────────
        self._feature_meta = self._load_json(config.FEATURE_ORDER_PATH)
        self._thresholds = self._load_json(config.THRESHOLDS_PATH)

        self._iforest_features = self._feature_meta.get("isolation_forest_features", [])
        self._xgb_features = self._feature_meta.get("xgboost_failure_features", [])
        self._imputation_rules = self._feature_meta.get("imputation_rules", {})

        # ── Load Dataset ─────────────────────────────────────────────────
        self._df = self._load_dataset()
        self._server_list = sorted(self._df["machine_name"].unique().tolist())
        self._data_freshness = self._compute_data_freshness()

        # ── Load Models (Graceful Degradation) ───────────────────────────
        self._iforest = self._load_model(config.IFOREST_PATH, "Isolation Forest")
        self._xgb_3slot = self._load_model(config.XGB_3SLOT_PATH, "XGBoost 12h")
        self._xgb_6slot = self._load_model(config.XGB_6SLOT_PATH, "XGBoost 24h")

        # ── Pre-compute SHAP background (small sample for speed) ─────────
        self._shap_background = None
        if self._xgb_3slot is not None:
            try:
                bg_idx = np.random.RandomState(42).choice(
                    len(self._df), size=min(config.SHAP_SAMPLE_SIZE, len(self._df)), replace=False
                )
                bg_data = self._df.iloc[bg_idx][self._xgb_features].copy()
                bg_data = self._apply_imputation(bg_data, mode="xgb")
                self._shap_background = bg_data
                logger.info(f"  SHAP background sample prepared: {len(bg_data)} rows")
            except Exception as e:
                logger.warning(f"  SHAP background preparation failed: {e}")

        # ── Schema Validation ────────────────────────────────────────────
        self._validate_schema()

        load_time_ms = (time.time() - t0) * 1000
        logger.info(f"ModelManager ready in {load_time_ms:.0f}ms | "
                     f"{len(self._server_list)} servers | "
                     f"Data freshness: {self._data_freshness:.1f}h old")

    # ─── Private Helpers ─────────────────────────────────────────────────

    def _load_json(self, path: Path) -> dict:
        """Load a JSON file. Returns empty dict on failure."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return {}

    def _load_dataset(self) -> pd.DataFrame:
        """Load master dataset from parquet (preferred) or CSV."""
        if config.DATASET_PATH.exists():
            df = pd.read_parquet(config.DATASET_PATH)
            logger.info(f"  Loaded dataset: {config.DATASET_PATH.name} ({len(df):,} rows)")
        elif config.DATASET_CSV_PATH.exists():
            df = pd.read_csv(config.DATASET_CSV_PATH)
            logger.info(f"  Loaded dataset: {config.DATASET_CSV_PATH.name} ({len(df):,} rows)")
        else:
            raise FileNotFoundError(f"No dataset found at {config.DATASET_PATH} or {config.DATASET_CSV_PATH}")

        # Ensure event_time_ping is datetime
        if "event_time_ping" in df.columns:
            df["event_time_ping"] = pd.to_datetime(df["event_time_ping"], errors="coerce")
        return df

    def _load_model(self, path: Path, name: str):
        """Load a joblib model. Returns None on failure (graceful degradation)."""
        if not path.exists():
            logger.warning(f"  [UNAVAILABLE] {name}: {path} not found")
            return None
        try:
            model = joblib.load(path)
            size_kb = path.stat().st_size / 1024
            logger.info(f"  Loaded {name}: {path.name} ({size_kb:.0f} KB)")
            return model
        except Exception as e:
            logger.error(f"  [CORRUPT] {name}: {e}")
            return None

    def _compute_data_freshness(self) -> float:
        """How many hours old is the most recent observation in the dataset?"""
        if "event_time_ping" in self._df.columns:
            max_time = self._df["event_time_ping"].max()
            if pd.notna(max_time):
                age = (pd.Timestamp.now() - max_time).total_seconds() / 3600
                return round(age, 1)
        return -1.0  # Unknown

    def _validate_schema(self):
        """Verify expected feature columns exist in the dataset."""
        missing_xgb = [c for c in self._xgb_features if c not in self._df.columns]
        missing_if = [c for c in self._iforest_features if c not in self._df.columns]

        if missing_xgb:
            raise SchemaValidationError(
                f"XGBoost features missing from dataset: {missing_xgb}"
            )
        if missing_if:
            raise SchemaValidationError(
                f"Isolation Forest features missing from dataset: {missing_if}"
            )
        logger.info("  Schema validation: PASSED")

    def _apply_imputation(self, df: pd.DataFrame, mode: str = "xgb") -> pd.DataFrame:
        """Apply domain-aware imputation rules from feature_order.json."""
        result = df.copy()
        hw_cols = self._imputation_rules.get("hardware_cols", [])
        hw_val = self._imputation_rules.get("hardware_impute_value", -1)
        lag_cols = self._imputation_rules.get("lag_cols", [])
        lag_val = self._imputation_rules.get("lag_impute_value", 0)

        for col in hw_cols:
            if col in result.columns:
                result[col] = result[col].fillna(hw_val).infer_objects(copy=False)
        for col in lag_cols:
            if col in result.columns:
                result[col] = result[col].fillna(lag_val).infer_objects(copy=False)
        # Fill any remaining NaN with 0
        result = result.fillna(0).infer_objects(copy=False)
        return result

    # ─── Public API ──────────────────────────────────────────────────────

    def list_all_servers(self) -> list[str]:
        """Return sorted list of all monitored server names."""
        return self._server_list

    def data_age_hours(self) -> float:
        """How many hours old is the latest data point?"""
        return self._data_freshness

    def get_staleness_warning(self) -> dict | None:
        """Return a warning dict if data is stale, else None."""
        age = self.data_age_hours()
        if age > config.STALE_DATA_THRESHOLD_HOURS:
            return {
                "warning": "DATA_STALE",
                "data_age_hours": age,
                "message": f"Dataset is {age:.0f} hours old (threshold: {config.STALE_DATA_THRESHOLD_HOURS}h). Results may not reflect current state.",
                "natural_phrasing": f"The latest telemetry is approximately {age:.0f} hours old, so predictions should be interpreted cautiously."
            }
        return None

    def get_server_row(self, server_name: str, slot: str = "latest") -> pd.Series | None:
        """
        Get a single observation row for a server.
        slot="latest" returns the most recent time slot.
        """
        mask = self._df["machine_name"] == server_name
        if not mask.any():
            return None

        server_df = self._df[mask].sort_values("event_time_ping")
        if slot == "latest":
            return server_df.iloc[-1]
        else:
            slot_mask = server_df["monitoring_slot"] == slot
            if slot_mask.any():
                return server_df[slot_mask].iloc[-1]
            return server_df.iloc[-1]

    def get_server_history(self, server_name: str, n_slots: int = 6) -> pd.DataFrame:
        """Get the last N monitoring slots for a server, sorted chronologically."""
        mask = self._df["machine_name"] == server_name
        if not mask.any():
            return pd.DataFrame()
        return self._df[mask].sort_values("event_time_ping").tail(n_slots)

    def score_iforest(self, server_name: str) -> dict:
        """Score a single server with the Isolation Forest anomaly detector."""
        if self._iforest is None:
            return {"error": "Isolation Forest model not available", "model": "isolation_forest.joblib"}

        row = self.get_server_row(server_name)
        if row is None:
            return {"error": f"Server '{server_name}' not found",
                    "available_sample": self._server_list[:5]}

        t0 = time.perf_counter()
        X = pd.DataFrame([row[self._iforest_features]])
        X = self._apply_imputation(X, mode="iforest")

        score = -float(self._iforest.decision_function(X)[0])
        prediction = int(self._iforest.predict(X)[0])
        is_anomaly = prediction == -1
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        contamination = self._thresholds.get("isolation_forest", {}).get("contamination", 0.02)

        return {
            "server_name": server_name,
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly,
            "isolation_forest_prediction": prediction,
            "contamination": contamination,
            "metadata": {
                "model_name": "isolation_forest.joblib",
                "model_version": "1.0.0-phase5.2",
                "decision_threshold": 0.0,
                "prediction_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "inference_latency_ms": latency_ms
            }
        }

    def score_xgb(self, server_name: str, horizon: str = "3slot") -> dict:
        """Score a single server with XGBoost failure prediction."""
        model = self._xgb_3slot if horizon == "3slot" else self._xgb_6slot
        model_name = "xgboost_failure_3slot.joblib" if horizon == "3slot" else "xgboost_failure_6slot.joblib"

        if model is None:
            return {"error": f"XGBoost model not available", "model": model_name}

        row = self.get_server_row(server_name)
        if row is None:
            return {"error": f"Server '{server_name}' not found",
                    "available_sample": self._server_list[:5]}

        t0 = time.perf_counter()
        X = pd.DataFrame([row[self._xgb_features]])
        X = self._apply_imputation(X, mode="xgb")

        prob = float(model.predict_proba(X)[:, 1][0])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        target_key = f"target_failure_{horizon}"
        optimal_f1 = self._thresholds.get(target_key, {}).get("optimal_f1_threshold", 0.5)

        return {
            "server_name": server_name,
            "failure_probability": round(prob, 4),
            "failure_probability_pct": round(prob * 100, 1),
            "horizon": "12h" if horizon == "3slot" else "24h",
            "metadata": {
                "model_name": model_name,
                "model_version": "1.0.0-phase5.6",
                "decision_threshold": optimal_f1,
                "prediction_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "inference_latency_ms": latency_ms
            }
        }

    def get_shap_explanation(self, server_name: str, horizon: str = "3slot") -> dict:
        """Compute local SHAP attribution for a specific server's prediction."""
        model = self._xgb_3slot if horizon == "3slot" else self._xgb_6slot
        if model is None:
            return {"error": "XGBoost model not available for SHAP"}

        row = self.get_server_row(server_name)
        if row is None:
            return {"error": f"Server '{server_name}' not found"}

        X = pd.DataFrame([row[self._xgb_features]])
        X = self._apply_imputation(X, mode="xgb")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_row = shap_values[0]

            contributions = []
            for i, feat in enumerate(self._xgb_features):
                contributions.append({
                    "feature": feat,
                    "value": round(float(X.iloc[0][feat]), 4),
                    "shap_contribution": round(float(shap_row[i]), 4)
                })
            # Sort by absolute SHAP contribution
            contributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)

            return {
                "server_name": server_name,
                "horizon": "12h" if horizon == "3slot" else "24h",
                "top_drivers": contributions[:6],
                "positive_drivers": [c for c in contributions if c["shap_contribution"] > 0][:4],
                "protective_factors": [c for c in contributions if c["shap_contribution"] < 0][:3],
                "base_value": round(float(explainer.expected_value), 4)
            }
        except Exception as e:
            return {"error": f"SHAP computation failed: {str(e)}"}

    def get_thresholds(self) -> dict:
        """Return the full thresholds configuration."""
        return self._thresholds

    def get_feature_meta(self) -> dict:
        """Return the full feature metadata configuration."""
        return self._feature_meta

    def find_server_by_ip(self, ip_address: str) -> str | None:
        """Find hostname for a given IP address."""
        mask = self._df["ip_address"] == ip_address
        if not mask.any():
            return None
        return str(self._df[mask].iloc[-1]["machine_name"])
